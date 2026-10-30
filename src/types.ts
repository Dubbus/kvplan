// Shared contract between the front end (lexer/parser/validate) and the
// back end (memory/capacity/codegen). Owned by the lead session — change only
// by agreement, since both halves compile against it.

export interface Pos {
  line: number; // 1-based
  col: number; // 1-based
}

export type DType = "fp32" | "fp16" | "bf16" | "fp8" | "int8" | "int4";

export const DTYPE_BYTES: Record<DType, number> = {
  fp32: 4,
  fp16: 2,
  bf16: 2,
  fp8: 1,
  int8: 1,
  int4: 0.5,
};

/** Fully resolved, validated input. The parser + validate pass produce this;
 *  every back-end pass consumes it. All fields are required by this point
 *  (presets have already been merged in). */
export interface Plan {
  model: {
    name: string;
    preset?: string; // preset key this model came from (== name, or the `from` target)
    layers: number;
    numHeads: number; // query heads
    kvHeads: number; // numHeads / kvHeads = GQA group size
    headDim: number;
    params: number; // total parameter count, e.g. 8.03e9
    dtype: DType; // weight dtype
  };
  cache: {
    blockSize: number; // tokens per paged-attention block
    maxContext: number; // tokens per sequence
    dtype: DType; // KV cache dtype (from `quantize`, defaults to model dtype)
  };
  target: {
    name: string; // e.g. "A10G"
    memoryBytes: number; // total device memory in bytes
    utilization: number; // fraction usable, vLLM's --gpu-memory-utilization (default 0.9)
  };
}

/** Back-end API (implemented in src/memory.ts, src/capacity.ts, src/codegen/vllm.ts).
 *  Exact report shapes are up to the back-end owner; these are the entry points
 *  the CLI will call. */
export interface MemoryReport {
  bytesPerToken: number;
  bytesPerBlock: number;
  bytesPerSequence: number; // at maxContext, rounded up to whole blocks
  weightBytes: number;
}

export interface CapacityReport {
  usableBytes: number; // memoryBytes * utilization
  kvBudgetBytes: number; // usable - weights - activation/overhead reserve
  totalBlocks: number;
  maxTokens: number; // totalBlocks * blockSize
  maxConcurrentSeqsAtMaxContext: number;
  fits: boolean; // weights fit and at least one full-context sequence fits
  notes: string[]; // human-readable warnings/assumptions
}
