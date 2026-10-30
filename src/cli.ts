#!/usr/bin/env node
// kvplan check <file.kv> [--gpu NAME]          validate only
// kvplan fit   <file.kv> [--gpu NAME] [--json] memory, capacity, vLLM flags

import { readFileSync } from "node:fs";
import { computeCapacity } from "./capacity.ts";
import { emitVllm } from "./codegen/vllm.ts";
import { formatDiagnostic, KvError } from "./errors.ts";
import { blocksFor, computeMemory } from "./memory.ts";
import { parse } from "./parser.ts";
import { GPU_PRESETS, MODEL_PRESETS } from "./presets.ts";
import type { Plan } from "./types.ts";
import { validate } from "./validate.ts";

const USAGE = `usage: kvplan check <file.kv> [--gpu NAME]
       kvplan fit   <file.kv> [--gpu NAME] [--json]

models: ${Object.keys(MODEL_PRESETS).join(", ")}
gpus:   ${Object.keys(GPU_PRESETS).join(", ")}`;

function main(argv: string[]): number {
  const [cmd, file, ...rest] = argv;
  if ((cmd !== "check" && cmd !== "fit") || !file) {
    console.error(USAGE);
    return 2;
  }
  let gpu: string | undefined;
  let json = false;
  for (let i = 0; i < rest.length; i++) {
    if (rest[i] === "--gpu" && rest[i + 1]) gpu = rest[++i];
    else if (rest[i] === "--json") json = true;
    else {
      console.error(`unknown option '${rest[i]}'\n\n${USAGE}`);
      return 2;
    }
  }

  let source: string;
  try {
    source = readFileSync(file, "utf8");
  } catch {
    console.error(`error: can't read ${file}`);
    return 2;
  }

  let plan: Plan | undefined;
  try {
    const result = validate(parse(source), { presets: { models: MODEL_PRESETS, gpus: GPU_PRESETS }, gpu });
    for (const d of result.diagnostics) console.error(formatDiagnostic(d, source, file) + "\n");
    plan = result.plan;
  } catch (e) {
    if (!(e instanceof KvError)) throw e;
    console.error(formatDiagnostic(e.diagnostic, source, file));
    return 1;
  }
  if (!plan) return 1;
  if (cmd === "check") {
    console.log(`ok: ${plan.model.name} on ${plan.target.name}`);
    return 0;
  }

  const mem = computeMemory(plan);
  const cap = computeCapacity(plan, mem);
  const vllm = emitVllm(plan, cap);
  if (json) {
    console.log(JSON.stringify({ plan, memory: mem, capacity: cap, vllm }, null, 2));
    return 0;
  }

  const { model, cache, target } = plan;
  const row = (label: string, value: string, note = "") => console.log(`  ${label.padEnd(22)}${value.padStart(12)}   ${note}`);
  const int = (n: number) => n.toLocaleString("en-US");

  console.log(`\nkvplan · ${model.name} on ${target.name} · KV ${cache.dtype}, block ${cache.blockSize}\n`);
  console.log("  Memory");
  row("KV per token", bytes(mem.bytesPerToken), `2 × ${model.layers} layers × ${model.kvHeads} kv_heads × ${model.headDim} × ${cache.dtype}`);
  row("KV per block", bytes(mem.bytesPerBlock), `${cache.blockSize} tokens`);
  row("KV per sequence", bytes(mem.bytesPerSequence), `${int(cache.maxContext)} tokens`);
  row("Weights", bytes(mem.weightBytes), `${(model.params / 1e9).toFixed(2)}B params × ${model.dtype}`);
  if (model.numHeads !== model.kvHeads) {
    console.log(`  GQA: ${model.numHeads} query heads share ${model.kvHeads} KV heads (${model.numHeads / model.kvHeads}× smaller cache than MHA)`);
  }

  console.log(`\n  Capacity`);
  row("GPU memory", bytes(target.memoryBytes));
  row("Usable", bytes(cap.usableBytes), `× ${target.utilization} utilization`);
  row("KV budget", bytes(cap.kvBudgetBytes), "after weights and reserve");
  row("Blocks", int(cap.totalBlocks));
  row("Tokens in cache", int(cap.maxTokens));
  row("Fits", cap.fits ? "yes" : "NO");

  console.log(`\n  Max batch by context length`);
  const contexts = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072].filter((n) => n < cache.maxContext);
  for (const ctx of [...contexts, cache.maxContext]) {
    row(`${int(ctx)} tokens`, int(Math.floor(cap.totalBlocks / blocksFor(ctx, cache.blockSize))), "sequences");
  }

  if (cap.notes.length) {
    console.log(`\n  Notes`);
    for (const n of cap.notes) console.log(`  · ${n}`);
  }
  console.log(`\n  vLLM\n`);
  console.log(vllm.trimEnd().replace(/^/gm, "  ") + "\n");
  return cap.fits ? 0 : 3;
}

function bytes(n: number): string {
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i++;
  }
  return `${i === 0 ? n : n.toFixed(2)} ${units[i]}`;
}

process.exitCode = main(process.argv.slice(2));
