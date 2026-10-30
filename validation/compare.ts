// Compare kvplan's predictions with KV-cache capacity measured by vLLM
// (results written by notebooks/validate_vllm.ipynb).
//
//   node validation/compare.ts [results-dir]      (default: validation/results)
//
// Two error numbers per run, both (kvplan - vLLM) / vLLM:
//   (a) formula:    floor(vLLM's available KV bytes / kvplan bytesPerBlock) x blockSize
//                   vs vLLM's KV cache tokens. Isolates the per-token math.
//   (b) end-to-end: kvplan's predicted maxTokens (with ACTIVATION_RESERVE_BYTES)
//                   vs vLLM's KV cache tokens.
// No calibration: ACTIVATION_RESERVE_BYTES is used as committed, never fitted
// to these runs. The "measured overhead" column shows what the reserve would
// have needed to be, for information only.

import { readdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { ACTIVATION_RESERVE_BYTES, computeCapacity } from "../src/capacity.ts";
import { computeMemory } from "../src/memory.ts";
import { MODEL_PRESETS } from "../src/presets.ts";
import type { DType, Plan } from "../src/types.ts";

const GiB = 2 ** 30;

/** One run, as written by the notebook (schema 1). */
export interface RunResult {
  schema: number;
  status: "ok" | "failed" | "skipped";
  error?: string;
  gpu: { name: string; slug: string; totalBytes: number; computeCapability?: string };
  vllmVersion?: string;
  model: { preset: string; hfId: string };
  flags: { maxModelLen: number; blockSize: number; gpuMemoryUtilization: number; kvCacheDtype: string; dtype: string };
  measured?: Partial<VllmFacts>;
  rawLogLines: string[];
}

export interface VllmFacts {
  availableKvBytes: number;
  kvCacheTokens: number;
  numGpuBlocks: number;
  modelLoadingBytes: number;
}

const num = (s: string) => Number(s.replace(/,/g, ""));

/** Pull the capacity facts out of vLLM startup log lines. Tolerant of V0/V1
 *  wording, thousands separators, and ANSI/log prefixes; the last match wins. */
export function parseVllmLog(lines: string[]): Partial<VllmFacts> {
  const out: Partial<VllmFacts> = {};
  for (const line of lines) {
    let m: RegExpMatchArray | null;
    // V1: "Available KV cache memory: 9.73 GiB"
    if ((m = line.match(/Available KV cache memory:\s*([\d.,]+)\s*GiB/i))) out.availableKvBytes = num(m[1]) * GiB;
    // V0: "... the rest of the memory reserved for KV Cache is 9.73GiB."
    else if ((m = line.match(/reserved for KV Cache is\s*([\d.,]+)\s*GiB/i))) out.availableKvBytes = num(m[1]) * GiB;
    // V1: "GPU KV cache size: 1,234,560 tokens"
    if ((m = line.match(/GPU KV cache size:\s*([\d,]+)\s*tokens/i))) out.kvCacheTokens = num(m[1]);
    // V0: "# GPU blocks: 5678, # CPU blocks: 2048" (also "num_gpu_blocks=5678")
    if ((m = line.match(/#\s*GPU blocks:\s*([\d,]+)/i) ?? line.match(/num_gpu_blocks\s*[=:]\s*([\d,]+)/i))) {
      out.numGpuBlocks = num(m[1]);
    }
    // "Model loading took 0.93 GiB and 2.1 seconds" / "... took 0.9276 GiB memory ..."
    if ((m = line.match(/Model loading took\s*([\d.,]+)\s*GiB/i))) out.modelLoadingBytes = num(m[1]) * GiB;
  }
  return out;
}

/** Facts from the raw log lines, falling back to the notebook's own parse. */
export function factsFor(run: RunResult): Partial<VllmFacts> {
  const parsed = parseVllmLog(run.rawLogLines ?? []);
  const facts = { ...run.measured, ...parsed };
  if (facts.kvCacheTokens === undefined && facts.numGpuBlocks !== undefined) {
    facts.kvCacheTokens = facts.numGpuBlocks * run.flags.blockSize;
  }
  return facts;
}

function weightDtype(vllmDtype: string, presetDtype: DType): DType {
  switch (vllmDtype) {
    case "half":
    case "float16":
      return "fp16";
    case "bfloat16":
      return "bf16";
    case "float32":
    case "float":
      return "fp32";
    default:
      return presetDtype; // "auto"
  }
}

/** kvplan Plan for a measured run: preset architecture + the run's flags + the
 *  GPU total that vLLM itself saw. */
export function planFor(run: RunResult): Plan {
  const preset = MODEL_PRESETS[run.model.preset];
  if (!preset) throw new Error(`unknown preset "${run.model.preset}"`);
  const dtype = weightDtype(run.flags.dtype, preset.dtype);
  return {
    model: { ...preset, preset: run.model.preset, dtype },
    cache: {
      blockSize: run.flags.blockSize,
      maxContext: run.flags.maxModelLen,
      dtype: run.flags.kvCacheDtype.startsWith("fp8") ? "fp8" : dtype,
    },
    target: { name: run.gpu.slug, memoryBytes: run.gpu.totalBytes, utilization: run.flags.gpuMemoryUtilization },
  };
}

export interface Row {
  gpu: string;
  model: string;
  kvDtype: string;
  maxModelLen: number;
  measuredTokens: number;
  formulaTokens: number;
  formulaErr: number; // (a)
  formulaBound: number; // |error| the log's 0.01 GiB rounding alone can cause in (a)
  predictedTokens: number;
  endToEndErr: number; // (b)
  measuredOverheadBytes: number; // usable - kvplan weights - vLLM available KV
}

export function compareRun(run: RunResult): Row {
  const facts = factsFor(run);
  if (facts.availableKvBytes === undefined || facts.kvCacheTokens === undefined) {
    throw new Error("log has no 'Available KV cache memory' / 'GPU KV cache size' (or V0 equivalents)");
  }
  const plan = planFor(run);
  const mem = computeMemory(plan);
  const cap = computeCapacity(plan, mem);
  const formulaTokens = Math.floor(facts.availableKvBytes / mem.bytesPerBlock) * plan.cache.blockSize;
  const measured = facts.kvCacheTokens;
  return {
    gpu: run.gpu.slug,
    model: run.model.preset,
    kvDtype: run.flags.kvCacheDtype,
    maxModelLen: run.flags.maxModelLen,
    measuredTokens: measured,
    formulaTokens,
    formulaErr: (formulaTokens - measured) / measured,
    formulaBound: (0.005 * GiB) / facts.availableKvBytes,
    predictedTokens: cap.maxTokens,
    endToEndErr: (cap.maxTokens - measured) / measured,
    measuredOverheadBytes: cap.usableBytes - mem.weightBytes - facts.availableKvBytes,
  };
}

export function loadResults(dir: string): RunResult[] {
  return readdirSync(dir)
    .filter((f) => f.endsWith(".json"))
    .sort()
    .map((f) => JSON.parse(readFileSync(join(dir, f), "utf8")) as RunResult);
}

const pct = (x: number) => `${x >= 0 ? "+" : ""}${(x * 100).toFixed(2)}%`;
const fmtInt = (n: number) => n.toLocaleString("en-US");
const gib = (b: number) => (b / GiB).toFixed(2);

export interface Report {
  rows: Row[];
  problems: string[]; // runs that failed in vLLM or couldn't be compared
  markdown: string;
}

export function buildReport(runs: RunResult[]): Report {
  const rows: Row[] = [];
  const problems: string[] = [];
  for (const run of runs) {
    const id = `${run.gpu?.slug}/${run.model?.preset}/${run.flags?.kvCacheDtype}`;
    if (run.status !== "ok") {
      problems.push(`${id}: ${run.status}${run.error ? ` (${run.error})` : ""}`);
      continue;
    }
    try {
      rows.push(compareRun(run));
    } catch (e) {
      problems.push(`${id}: ${(e as Error).message}`);
    }
  }

  const md: string[] = [];
  md.push(
    "| GPU | Model | KV dtype | max len | vLLM KV tokens | (a) formula tokens | (a) error | (b) kvplan predicted | (b) error | measured overhead (GiB) |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
  );
  for (const r of rows) {
    md.push(
      `| ${r.gpu} | ${r.model} | ${r.kvDtype} | ${r.maxModelLen} | ${fmtInt(r.measuredTokens)} | ` +
        `${fmtInt(r.formulaTokens)} | ${pct(r.formulaErr)} | ${fmtInt(r.predictedTokens)} | ${pct(r.endToEndErr)} | ` +
        `${gib(r.measuredOverheadBytes)} |`,
    );
  }
  if (rows.length) {
    const abs = (xs: number[]) => xs.map(Math.abs);
    const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
    const a = abs(rows.map((r) => r.formulaErr));
    const b = abs(rows.map((r) => r.endToEndErr));
    const models = new Set(rows.map((r) => r.model)).size;
    const gpus = new Set(rows.map((r) => r.gpu)).size;
    md.push(
      "",
      `${rows.length} runs, ${models} models, ${gpus} GPU type(s). Errors are (kvplan - vLLM) / vLLM.`,
      "",
      `- **(a) Per-token formula** (vLLM's own available KV memory / kvplan bytes per block): ` +
        `mean |error| ${(mean(a) * 100).toFixed(2)}%, max ${(Math.max(...a) * 100).toFixed(2)}%. ` +
        `vLLM logs available memory rounded to 0.01 GiB, which alone allows up to ` +
        `${(Math.max(...rows.map((r) => r.formulaBound)) * 100).toFixed(2)}% here.`,
      `- **(b) End to end** (kvplan's predicted maxTokens, including the uncalibrated ` +
        `${gib(ACTIVATION_RESERVE_BYTES)} GiB activation/CUDA-graph reserve): ` +
        `mean |error| ${(mean(b) * 100).toFixed(2)}%, max ${(Math.max(...b) * 100).toFixed(2)}%.`,
      `- **Measured overhead** = usable memory - kvplan weight bytes - vLLM available KV memory: ` +
        `what the reserve would have had to be. Informational only; the reserve was not tuned on these runs.`,
    );
  } else {
    md.push("", "No comparable runs.");
  }
  if (problems.length) md.push("", "Not compared:", ...problems.map((p) => `- ${p}`));
  return { rows, problems, markdown: md.join("\n") + "\n" };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const dir = process.argv[2] ?? join(fileURLToPath(new URL(".", import.meta.url)), "results");
  process.stdout.write(buildReport(loadResults(dir)).markdown);
}
