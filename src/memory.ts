// Memory pass: per-token / per-block / per-sequence KV-cache size and weight size.

import { DTYPE_BYTES } from "./types.ts";
import type { MemoryReport, Plan } from "./types.ts";

/** Number of paged-attention blocks one sequence of `tokens` occupies. */
export function blocksFor(tokens: number, blockSize: number): number {
  return Math.ceil(tokens / blockSize);
}

export function computeMemory(plan: Plan): MemoryReport {
  const { model, cache } = plan;
  // K and V, for every layer and KV head.
  const bytesPerToken = 2 * model.layers * model.kvHeads * model.headDim * DTYPE_BYTES[cache.dtype];
  const bytesPerBlock = bytesPerToken * cache.blockSize;
  const bytesPerSequence = blocksFor(cache.maxContext, cache.blockSize) * bytesPerBlock;
  const weightBytes = Math.ceil(model.params * DTYPE_BYTES[model.dtype]);
  return { bytesPerToken, bytesPerBlock, bytesPerSequence, weightBytes };
}
