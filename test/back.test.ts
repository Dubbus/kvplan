import { test } from "node:test";
import assert from "node:assert/strict";

import { computeMemory } from "../src/memory.ts";
import { ACTIVATION_RESERVE_BYTES, computeCapacity } from "../src/capacity.ts";
import { emitVllm } from "../src/codegen/vllm.ts";
import { GPU_PRESETS, MODEL_PRESETS, MODEL_SERVING } from "../src/presets.ts";
import type { DType, Plan } from "../src/types.ts";

function plan(opts: {
  model?: string;
  modelDtype?: DType;
  gpu?: string;
  blockSize?: number;
  maxContext?: number;
  kvDtype?: DType;
  utilization?: number;
} = {}): Plan {
  const model = { ...MODEL_PRESETS[opts.model ?? "llama3_8b"] };
  if (opts.modelDtype) model.dtype = opts.modelDtype;
  const gpu = opts.gpu ?? "A10G";
  return {
    model,
    cache: {
      blockSize: opts.blockSize ?? 16,
      maxContext: opts.maxContext ?? 8192,
      dtype: opts.kvDtype ?? model.dtype,
    },
    target: { name: gpu, memoryBytes: GPU_PRESETS[gpu].memoryBytes, utilization: opts.utilization ?? 0.9 },
  };
}

test("Llama 3 8B fp16 is exactly 131072 bytes/token", () => {
  const mem = computeMemory(plan({ modelDtype: "fp16", kvDtype: "fp16" }));
  assert.equal(mem.bytesPerToken, 131072);
  assert.equal(mem.bytesPerBlock, 131072 * 16);
  assert.equal(mem.weightBytes, 8_030_261_248 * 2);
});

test("bytesPerSequence rounds up to whole blocks", () => {
  const exact = computeMemory(plan({ maxContext: 8192, blockSize: 16 }));
  assert.equal(exact.bytesPerSequence, 512 * exact.bytesPerBlock);
  const ragged = computeMemory(plan({ maxContext: 8193, blockSize: 16 }));
  assert.equal(ragged.bytesPerSequence, 513 * ragged.bytesPerBlock);
  const tiny = computeMemory(plan({ maxContext: 1, blockSize: 32 }));
  assert.equal(tiny.bytesPerSequence, tiny.bytesPerBlock);
});

test("ragged max_context is noted in capacity", () => {
  const p = plan({ maxContext: 8000, blockSize: 16 * 3 });
  const cap = computeCapacity(p, computeMemory(p));
  assert.ok(cap.notes.some((n) => n.includes("not a multiple of block_size")));
});

test("int8 and fp8 KV caches halve bytes/token vs fp16", () => {
  const fp16 = computeMemory(plan({ kvDtype: "fp16" })).bytesPerToken;
  assert.equal(computeMemory(plan({ kvDtype: "int8" })).bytesPerToken, fp16 / 2);
  assert.equal(computeMemory(plan({ kvDtype: "fp8" })).bytesPerToken, fp16 / 2);
  // ...and roughly double the blocks that fit.
  const p16 = plan({ kvDtype: "fp16" });
  const p8 = plan({ kvDtype: "fp8" });
  const b16 = computeCapacity(p16, computeMemory(p16)).totalBlocks;
  const b8 = computeCapacity(p8, computeMemory(p8)).totalBlocks;
  assert.ok(b8 === 2 * b16 || b8 === 2 * b16 + 1);
});

test("Llama 3 8B fits on A10G; capacity arithmetic is consistent", () => {
  const p = plan();
  const mem = computeMemory(p);
  const cap = computeCapacity(p, mem);
  assert.equal(cap.usableBytes, Math.floor(p.target.memoryBytes * 0.9));
  assert.equal(cap.kvBudgetBytes, cap.usableBytes - mem.weightBytes - ACTIVATION_RESERVE_BYTES);
  assert.equal(cap.totalBlocks, Math.floor(cap.kvBudgetBytes / mem.bytesPerBlock));
  assert.equal(cap.maxTokens, cap.totalBlocks * 16);
  assert.equal(cap.maxConcurrentSeqsAtMaxContext, Math.floor(cap.totalBlocks / 512));
  assert.ok(cap.fits);
  assert.ok(cap.notes.some((n) => n.includes("ACTIVATION_RESERVE_BYTES")));
});

test("Llama 3 70B bf16 does not fit on a single A100-80GB", () => {
  const p = plan({ model: "llama3_70b", gpu: "A100-80GB" });
  const cap = computeCapacity(p, computeMemory(p));
  assert.equal(cap.fits, false);
  assert.equal(cap.kvBudgetBytes, 0);
  assert.equal(cap.totalBlocks, 0);
  assert.ok(cap.notes.some((n) => n.includes("will not load")));
  assert.match(emitVllm(p, cap), /# WARNING: This plan does not fit/);
});

test("weights fit but a full-context sequence does not", () => {
  // Llama 3 8B on a T4: 16 GB of bf16 weights > 15 GiB * 0.9. Use fp8 weights instead.
  const p = plan({ gpu: "T4", modelDtype: "fp8", maxContext: 8192, kvDtype: "bf16", utilization: 0.6 });
  const cap = computeCapacity(p, computeMemory(p));
  assert.equal(cap.fits, false);
  assert.ok(cap.totalBlocks > 0);
  assert.equal(cap.maxConcurrentSeqsAtMaxContext, 0);
  assert.ok(cap.notes.some((n) => n.includes("less than one")));
});

test("reserve is overridable", () => {
  const p = plan();
  const mem = computeMemory(p);
  const a = computeCapacity(p, mem);
  const b = computeCapacity(p, mem, { reserveBytes: 0 });
  assert.equal(b.kvBudgetBytes - a.kvBudgetBytes, ACTIVATION_RESERVE_BYTES);
});

test("presets: every model has serving metadata and sane GQA", () => {
  for (const [key, m] of Object.entries(MODEL_PRESETS)) {
    assert.equal(m.name, key);
    assert.ok(MODEL_SERVING[key], key);
    assert.equal(m.numHeads % m.kvHeads, 0, key);
  }
  // A10G "24GB" reports ~22.5 GiB, not 24 GiB.
  assert.equal((GPU_PRESETS.A10G.memoryBytes / 2 ** 30).toFixed(1), "22.5");
});

test("vllm: clean plan emits the five flags and no warnings", () => {
  const p = plan();
  const out = emitVllm(p, computeCapacity(p, computeMemory(p)));
  assert.doesNotMatch(out, /WARNING/);
  assert.match(out, /^vllm serve meta-llama\/Meta-Llama-3-8B-Instruct/);
  for (const flag of ["--max-model-len 8192", "--block-size 16", "--kv-cache-dtype auto",
    "--gpu-memory-utilization 0.9", "--max-num-seqs "]) {
    assert.ok(out.includes(flag), flag);
  }
});

test("vllm: int8 KV, bad block size and over-long context all warn", () => {
  const p = plan({ kvDtype: "int8", blockSize: 24, maxContext: 16384 });
  const out = emitVllm(p, computeCapacity(p, computeMemory(p)));
  assert.match(out, /# WARNING: vLLM has no int8 KV cache/);
  assert.match(out, /--kv-cache-dtype fp8/);
  assert.match(out, /# WARNING: block_size 24/);
  assert.match(out, /# WARNING: max_context 16384 exceeds/);
  assert.match(out, /--max-model-len 8192/);
});

test("vllm: renamed model resolves HF id via its preset key", () => {
  const p = plan();
  p.model = { ...p.model, name: "my_llama", preset: "llama3_8b" };
  const out = emitVllm(p, computeCapacity(p, computeMemory(p)));
  assert.match(out, /^vllm serve meta-llama\/Meta-Llama-3-8B-Instruct/);
  assert.doesNotMatch(out, /WARNING/);
});

test("vllm: unknown model without preset gets a placeholder warning", () => {
  const p = plan();
  p.model = { ...p.model, name: "my_llama" };
  const out = emitVllm(p, computeCapacity(p, computeMemory(p)));
  assert.match(out, /# WARNING: No Hugging Face id known for "my_llama"/);
  assert.match(out, /^vllm serve my_llama/m);
});

test("vllm: fp8 KV maps to fp8 without warning", () => {
  const p = plan({ kvDtype: "fp8" });
  const out = emitVllm(p, computeCapacity(p, computeMemory(p)));
  assert.match(out, /--kv-cache-dtype fp8/);
  assert.doesNotMatch(out, /WARNING/);
});
