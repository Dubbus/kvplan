import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { buildReport, compareRun, loadResults, parseVllmLog, planFor } from "../validation/compare.ts";
import type { RunResult } from "../validation/compare.ts";
import { ACTIVATION_RESERVE_BYTES } from "../src/capacity.ts";

const GiB = 2 ** 30;

// Hand-written fixtures shaped like the notebook's output. Numbers are made up
// but self-consistent: vLLM's token count = floor(available / page) * 16.

// T4, Qwen2.5-0.5B, fp16. KV bytes/token = 2*24*2*64*2 = 12288; block = 196608.
// 10.53 GiB / 196608 = 57507.3 blocks -> 920112 tokens.
const t4: RunResult = {
  schema: 1,
  status: "ok",
  gpu: { name: "Tesla T4", slug: "T4", totalBytes: 15_835_398_144, computeCapability: "7.5" },
  vllmVersion: "0.19.0",
  model: { preset: "qwen2.5_0.5b", hfId: "Qwen/Qwen2.5-0.5B-Instruct" },
  flags: { maxModelLen: 4096, blockSize: 16, gpuMemoryUtilization: 0.9, kvCacheDtype: "auto", dtype: "half" },
  rawLogLines: [
    "\u001b[1;36m(EngineCore_DP0 pid=812)\u001b[0;0m INFO 04-03 10:11:12 [gpu_model_runner.py:2602] Model loading took 0.9276 GiB and 3.2 seconds",
    "(EngineCore_DP0 pid=812) INFO 04-03 10:11:20 [gpu_worker.py:298] Available KV cache memory: 10.53 GiB",
    "(EngineCore_DP0 pid=812) INFO 04-03 10:11:20 [kv_cache_utils.py:1087] GPU KV cache size: 920,112 tokens",
    "(EngineCore_DP0 pid=812) INFO 04-03 10:11:20 [kv_cache_utils.py:1091] Maximum concurrency for 4,096 tokens per request: 224.64x",
  ],
};

// L4, Qwen2.5-7B, bf16 weights, fp8 KV. KV bytes/token = 2*28*4*128*1 = 28672; block = 458752.
// 3.21 GiB / 458752 = 7513.3 blocks -> 120208 tokens.
const l4fp8: RunResult = {
  schema: 1,
  status: "ok",
  gpu: { name: "NVIDIA L4", slug: "L4", totalBytes: 23_580_639_232, computeCapability: "8.9" },
  vllmVersion: "0.19.0",
  model: { preset: "qwen2.5_7b", hfId: "Qwen/Qwen2.5-7B-Instruct" },
  flags: { maxModelLen: 4096, blockSize: 16, gpuMemoryUtilization: 0.9, kvCacheDtype: "fp8", dtype: "bfloat16" },
  rawLogLines: ["INFO Available KV cache memory: 3.21 GiB", "INFO GPU KV cache size: 120,208 tokens"],
};

// V0-style log: no token line, only "# GPU blocks".
const v0: RunResult = {
  ...t4,
  vllmVersion: "0.6.6",
  rawLogLines: [
    "INFO worker.py:241] Memory profiling takes 4.1 seconds ... the rest of the memory reserved for KV Cache is 10.53GiB.",
    "INFO executor_base.py:110] # GPU blocks: 57507, # CPU blocks: 21845",
  ],
};

const failed: RunResult = {
  ...t4,
  status: "failed",
  error: "vLLM exited with code 1",
  model: { preset: "llama3_8b", hfId: "meta-llama/Meta-Llama-3-8B-Instruct" },
  rawLogLines: [],
};

test("parseVllmLog: V1 lines with ANSI prefixes and thousands separators", () => {
  const f = parseVllmLog(t4.rawLogLines);
  assert.equal(f.availableKvBytes, 10.53 * GiB);
  assert.equal(f.kvCacheTokens, 920112);
  assert.equal(f.modelLoadingBytes, 0.9276 * GiB);
  assert.equal(f.numGpuBlocks, undefined);
});

test("parseVllmLog: V0 lines", () => {
  const f = parseVllmLog(v0.rawLogLines);
  assert.equal(f.availableKvBytes, 10.53 * GiB);
  assert.equal(f.numGpuBlocks, 57507);
  assert.equal(f.kvCacheTokens, undefined);
});

test("planFor maps vLLM dtypes onto kvplan dtypes", () => {
  const p = planFor(t4);
  assert.equal(p.model.dtype, "fp16");
  assert.equal(p.cache.dtype, "fp16");
  assert.equal(p.target.memoryBytes, 15_835_398_144);
  const q = planFor(l4fp8);
  assert.equal(q.model.dtype, "bf16");
  assert.equal(q.cache.dtype, "fp8");
});

test("(a) formula error is exactly 0 on self-consistent fixtures", () => {
  assert.equal(compareRun(t4).formulaTokens, 920112);
  assert.equal(compareRun(t4).formulaErr, 0);
  assert.equal(compareRun(l4fp8).formulaTokens, 120208);
  assert.equal(compareRun(l4fp8).formulaErr, 0);
  // V0: tokens derived from blocks x block size.
  assert.equal(compareRun(v0).measuredTokens, 57507 * 16);
  assert.equal(compareRun(v0).formulaErr, 0);
});

test("(b) end-to-end matches a hand calculation", () => {
  // usable = floor(15835398144 * 0.9); weights = 494032768 * 2; KV budget = usable - weights - reserve.
  const usable = Math.floor(15_835_398_144 * 0.9);
  const budget = usable - 494_032_768 * 2 - ACTIVATION_RESERVE_BYTES;
  const predicted = Math.floor(budget / 196608) * 16;
  const r = compareRun(t4);
  assert.equal(r.predictedTokens, predicted);
  assert.equal(r.endToEndErr, (predicted - 920112) / 920112);
  assert.equal(r.measuredOverheadBytes, usable - 494_032_768 * 2 - 10.53 * GiB);
});

test("buildReport: table, summary and failed runs from a results directory", () => {
  const dir = mkdtempSync(join(tmpdir(), "kvplan-results-"));
  writeFileSync(join(dir, "T4_qwen2.5_0.5b_auto.json"), JSON.stringify(t4));
  writeFileSync(join(dir, "L4_qwen2.5_7b_fp8.json"), JSON.stringify(l4fp8));
  writeFileSync(join(dir, "T4_llama3_8b_auto.json"), JSON.stringify(failed));
  writeFileSync(join(dir, "notes.txt"), "ignored");
  const { rows, problems, markdown } = buildReport(loadResults(dir));
  assert.equal(rows.length, 2);
  assert.deepEqual(problems, ["T4/llama3_8b/auto: failed (vLLM exited with code 1)"]);
  assert.match(markdown, /^\| GPU \| Model \|/);
  assert.match(markdown, /\| T4 \| qwen2.5_0.5b \| auto \| 4096 \| 920,112 \| 920,112 \| \+0\.00% \|/);
  assert.match(markdown, /2 runs, 2 models, 2 GPU type\(s\)/);
  assert.match(markdown, /\(a\) Per-token formula.*mean \|error\| 0\.00%/);
});

test("compareRun rejects a log with no KV facts", () => {
  assert.throws(() => compareRun({ ...t4, rawLogLines: ["nothing useful"] }), /Available KV cache memory/);
});
