// Validate pass: checks the AST, merges in presets, converts units, and
// produces the fully resolved Plan the back end consumes. Collects every
// diagnostic it can instead of stopping at the first.

import type { Block, KvFile, Span } from "./ast.ts";
import { suggest } from "./errors.ts";
import type { Diagnostic } from "./errors.ts";
import { DTYPE_BYTES } from "./types.ts";
import type { DType, Plan } from "./types.ts";

export interface Presets {
  models: Record<string, Plan["model"]>;
  gpus: Record<string, { memoryBytes: number }>;
}

export interface ValidateOptions {
  presets: Presets;
  gpu?: string; // --gpu on the CLI; overrides the file's target
}

export interface ValidateResult {
  plan?: Plan; // present only when there are no errors
  diagnostics: Diagnostic[];
}

type Kind = "int" | "count" | "context" | "bytes" | "fraction" | "dtype";

const MODEL_FIELDS: Record<string, Kind> = {
  layers: "int",
  heads: "int",
  kv_heads: "int",
  head_dim: "int",
  hidden_size: "int",
  params: "count",
  dtype: "dtype",
};
const CACHE_FIELDS: Record<string, Kind> = { block_size: "int", max_context: "context", quantize: "dtype" };
const TARGET_FIELDS: Record<string, Kind> = { memory: "bytes", utilization: "fraction" };

const UNITS: Record<"count" | "context" | "bytes", Record<string, number>> = {
  count: { "": 1, k: 1e3, K: 1e3, M: 1e6, B: 1e9, T: 1e12 },
  context: { "": 1, k: 1024, K: 1024 }, // "8k context" conventionally means 8192
  bytes: { B: 1, KB: 1e3, MB: 1e6, GB: 1e9, TB: 1e12, KiB: 2 ** 10, MiB: 2 ** 20, GiB: 2 ** 30, TiB: 2 ** 40 },
};

const DEFAULT_BLOCK_SIZE = 16;
const DEFAULT_UTILIZATION = 0.9; // vLLM's --gpu-memory-utilization default
const NO_SPAN: Span = { line: 1, col: 1, length: 0 };

interface Checked {
  value: number | DType;
  span: Span;
}

class Checker {
  diagnostics: Diagnostic[] = [];

  error(span: Span, message: string, hint?: string) {
    this.diagnostics.push({ severity: "error", message, span, hint });
  }

  warn(span: Span, message: string, hint?: string) {
    this.diagnostics.push({ severity: "warning", message, span, hint });
  }

  /** Checks a block's fields against its spec and converts their values. */
  fields(block: Block, spec: Record<string, Kind>): Map<string, Checked> {
    const out = new Map<string, Checked>();
    for (const f of block.fields) {
      const kind = spec[f.key];
      if (!kind) {
        const guess = suggest(f.key, Object.keys(spec));
        this.error(
          f.keySpan,
          `unknown field '${f.key}' in ${block.kind} block`,
          guess ? `did you mean '${guess}'?` : `expected one of: ${Object.keys(spec).join(", ")}`,
        );
        continue;
      }
      const prev = out.get(f.key);
      if (prev) {
        this.error(f.keySpan, `'${f.key}' is set twice`, `first set on line ${prev.span.line}`);
        continue;
      }
      const value = this.convert(f.key, f.value, kind);
      if (value !== undefined) out.set(f.key, { value, span: f.value.span });
    }
    return out;
  }

  private convert(key: string, v: Block["fields"][number]["value"], kind: Kind): number | DType | undefined {
    if (kind === "dtype") {
      if (v.kind === "ident" && v.value in DTYPE_BYTES) return v.value as DType;
      const guess = v.kind === "ident" ? suggest(v.value, Object.keys(DTYPE_BYTES)) : undefined;
      this.error(v.span, `'${key}' must be a dtype`, guess ? `did you mean '${guess}'?` : `one of: ${Object.keys(DTYPE_BYTES).join(", ")}`);
      return;
    }
    if (v.kind !== "number") {
      this.error(v.span, `'${key}' must be a number`);
      return;
    }
    if (kind === "int" || kind === "fraction") {
      if (v.unit) {
        this.error(v.span, `'${key}' doesn't take a unit`, `write ${key} ${v.raw.slice(0, -v.unit.length)}`);
      } else if (kind === "int" && (!Number.isInteger(v.value) || v.value <= 0)) {
        this.error(v.span, `'${key}' must be a positive whole number`);
      } else if (kind === "fraction" && !(v.value > 0 && v.value <= 1)) {
        this.error(v.span, `'${key}' must be between 0 and 1`, `e.g. ${key} 0.9`);
      } else {
        return v.value;
      }
      return;
    }
    const units = UNITS[kind];
    const mult = units[v.unit];
    if (mult === undefined) {
      const known = Object.keys(units).filter(Boolean).join(", ");
      if (v.unit) this.error(v.span, `unknown unit '${v.unit}' for '${key}'`, `use one of: ${known}`);
      else this.error(v.span, `'${key}' needs a unit`, `e.g. ${key} ${v.raw}GB`);
      return;
    }
    const n = kind === "bytes" ? v.value * mult : Math.round(v.value * mult);
    if (n <= 0) {
      this.error(v.span, `'${key}' must be greater than zero`);
      return;
    }
    return n;
  }
}

export function validate(file: KvFile, opts: ValidateOptions): ValidateResult {
  const c = new Checker();
  const blocks: Partial<Record<Block["kind"], Block>> = {};
  for (const b of file.blocks) {
    const first = blocks[b.kind];
    if (first) c.error(b.span, `more than one ${b.kind} block`, `the first is on line ${first.span.line}`);
    else blocks[b.kind] = b;
  }

  let model: Plan["model"] | undefined;
  if (blocks.model) model = resolveModel(c, blocks.model, opts.presets.models);
  else c.error(NO_SPAN, "no model block", "add one, e.g. model llama3_8b { }");

  let cache: Plan["cache"] | undefined;
  if (blocks.cache) cache = resolveCache(c, blocks.cache, model?.dtype ?? "fp16");
  else c.error(NO_SPAN, "no cache block", "add one, e.g. cache { max_context 8192 }");

  const target = resolveTarget(c, blocks.target, opts);

  const ok = model && cache && target && !c.diagnostics.some((d) => d.severity === "error");
  return { plan: ok ? { model: model!, cache: cache!, target: target! } : undefined, diagnostics: c.diagnostics };
}

function resolveModel(c: Checker, b: Block, presets: Record<string, Plan["model"]>): Plan["model"] | undefined {
  const f = c.fields(b, MODEL_FIELDS);
  const name = b.name!;
  let base: Plan["model"] | undefined = presets[name];
  if (b.from) {
    base = presets[b.from.name];
    if (!base) {
      const guess = suggest(b.from.name, Object.keys(presets));
      c.error(b.from.span, `unknown preset '${b.from.name}'`, guess ? `did you mean '${guess}'?` : `presets: ${Object.keys(presets).join(", ")}`);
      return;
    }
  }
  const num = (key: string) => f.get(key)?.value as number | undefined;

  const layers = num("layers") ?? base?.layers;
  const heads = num("heads") ?? base?.numHeads;
  const kvHeads = num("kv_heads") ?? base?.kvHeads;
  const params = num("params") ?? base?.params;
  const dtype = (f.get("dtype")?.value as DType | undefined) ?? base?.dtype ?? "fp16";

  let headDim = num("head_dim");
  const hidden = f.get("hidden_size");
  if (headDim === undefined && hidden && heads !== undefined) {
    const h = hidden.value as number;
    if (h % heads !== 0) {
      c.error(hidden.span, `hidden_size ${h} doesn't divide evenly across ${heads} heads`, `that's ${(h / heads).toFixed(2)} per head; set head_dim explicitly`);
      return;
    }
    headDim = h / heads;
  }
  headDim ??= base?.headDim;

  const missing = Object.entries({ layers, heads, kv_heads: kvHeads, head_dim: headDim, params })
    .filter(([, v]) => v === undefined)
    .map(([k]) => k);
  if (missing.length) {
    const presetNames = Object.keys(presets);
    c.error(
      b.nameSpan!,
      `model '${name}' is missing ${missing.join(", ")}`,
      presetNames.length ? `set them, or start from a preset: model ${name} from ${presetNames[0]} { ... }` : undefined,
    );
    return;
  }

  if (heads! % kvHeads! !== 0) {
    const at = f.get("kv_heads")?.span ?? f.get("heads")?.span ?? b.nameSpan!;
    c.error(at, `heads (${heads}) must be a multiple of kv_heads (${kvHeads})`, "with GQA, each KV head is shared by heads / kv_heads query heads");
    return;
  }

  const preset = b.from?.name ?? (base ? name : undefined);
  return { name, preset, layers: layers!, numHeads: heads!, kvHeads: kvHeads!, headDim: headDim!, params: params!, dtype };
}

function resolveCache(c: Checker, b: Block, modelDtype: DType): Plan["cache"] | undefined {
  const f = c.fields(b, CACHE_FIELDS);
  const maxContext = f.get("max_context")?.value as number | undefined;
  if (maxContext === undefined) {
    c.error(b.span, "cache block needs max_context", "e.g. max_context 8192 (or 8k)");
    return;
  }
  return {
    blockSize: (f.get("block_size")?.value as number | undefined) ?? DEFAULT_BLOCK_SIZE,
    maxContext,
    dtype: (f.get("quantize")?.value as DType | undefined) ?? modelDtype,
  };
}

function resolveTarget(c: Checker, b: Block | undefined, opts: ValidateOptions): Plan["target"] | undefined {
  const f = b ? c.fields(b, TARGET_FIELDS) : new Map<string, Checked>();
  const overridden = opts.gpu !== undefined && b !== undefined && opts.gpu !== b.name;
  if (overridden) c.warn(b!.nameSpan!, `--gpu ${opts.gpu} overrides target "${b!.name}"`, "the file's memory setting is ignored");

  const name = opts.gpu ?? b?.name;
  if (name === undefined) {
    c.error(NO_SPAN, "no target GPU", 'add target gpu "A10G" { } or pass --gpu A10G');
    return;
  }
  const at = (!overridden && b?.nameSpan) || NO_SPAN;
  const gpus = opts.presets.gpus;
  const presetKey = Object.keys(gpus).find((k) => k.toLowerCase() === name.toLowerCase());
  const preset = presetKey ? gpus[presetKey] : undefined;
  const explicit = overridden ? undefined : f.get("memory");

  let memoryBytes: number;
  if (explicit) {
    memoryBytes = explicit.value as number;
    if (preset && Math.abs(memoryBytes - preset.memoryBytes) / preset.memoryBytes > 0.02) {
      c.warn(
        explicit.span,
        `${presetKey} actually reports ${(preset.memoryBytes / 2 ** 30).toFixed(2)} GiB, not ${(memoryBytes / 2 ** 30).toFixed(2)} GiB`,
        "drop 'memory' to use the measured value",
      );
    }
  } else if (preset) {
    memoryBytes = preset.memoryBytes;
  } else {
    // "A100" should offer A100-40GB / A100-80GB before an edit-distance neighbour like A10G.
    const variants = Object.keys(gpus).filter((k) => k.toLowerCase().startsWith(name.toLowerCase()));
    const guesses = variants.length ? variants : [suggest(name, Object.keys(gpus))].filter((g) => g !== undefined);
    c.error(
      at,
      `unknown GPU '${name}'`,
      guesses.length ? `did you mean ${guesses.map((g) => `'${g}'`).join(" or ")}?` : `known GPUs: ${Object.keys(gpus).join(", ")}; or set memory 24GB`,
    );
    return;
  }

  return {
    name: presetKey ?? name,
    memoryBytes,
    utilization: (f.get("utilization")?.value as number | undefined) ?? DEFAULT_UTILIZATION,
  };
}
