// Capacity pass: how many KV blocks / tokens / full-context sequences fit
// in what's left of the GPU after weights and the runtime reserve.

import { blocksFor } from "./memory.ts";
import type { CapacityReport, MemoryReport, Plan } from "./types.ts";

const GiB = 2 ** 30;

/** ASSUMPTION: memory inside the --gpu-memory-utilization budget that vLLM
 *  can't give to the KV cache. It covers the peak activations of vLLM's
 *  profiling forward pass (default max_num_batched_tokens), CUDA-graph capture
 *  buffers, and the CUDA context / non-torch allocations. Real usage depends
 *  on model, batch settings and vLLM version; 1.5 GiB is a reasonable figure
 *  for 7B-70B models on one GPU with CUDA graphs on. Override it via
 *  `computeCapacity(plan, mem, { reserveBytes })`. */
export const ACTIVATION_RESERVE_BYTES = 1.5 * GiB;

export interface CapacityOptions {
  reserveBytes?: number;
}

const gib = (b: number) => `${(b / GiB).toFixed(2)} GiB`;

export function computeCapacity(plan: Plan, mem: MemoryReport, opts: CapacityOptions = {}): CapacityReport {
  const { cache, target } = plan;
  const reserve = opts.reserveBytes ?? ACTIVATION_RESERVE_BYTES;
  const notes: string[] = [];

  const usableBytes = Math.floor(target.memoryBytes * target.utilization);
  const kvBudgetBytes = Math.max(0, usableBytes - mem.weightBytes - reserve);
  const totalBlocks = Math.floor(kvBudgetBytes / mem.bytesPerBlock);
  const maxTokens = totalBlocks * cache.blockSize;
  const blocksPerSeq = blocksFor(cache.maxContext, cache.blockSize);
  const maxConcurrentSeqsAtMaxContext = Math.floor(totalBlocks / blocksPerSeq);

  const weightsFit = mem.weightBytes + reserve <= usableBytes;
  const fits = weightsFit && maxConcurrentSeqsAtMaxContext >= 1;

  notes.push(
    `Assumes ${gib(reserve)} reserved for activations / CUDA graphs / CUDA context ` +
      `(ACTIVATION_RESERVE_BYTES; an estimate, not measured).`,
  );
  notes.push(
    `Usable = ${gib(target.memoryBytes)} x ${target.utilization} = ${gib(usableBytes)}; ` +
      `weights take ${gib(mem.weightBytes)}.`,
  );

  if (!weightsFit) {
    notes.push(
      `Weights (${gib(mem.weightBytes)}) plus reserve (${gib(reserve)}) exceed usable memory ` +
        `(${gib(usableBytes)}); the model will not load on ${target.name}.`,
    );
  } else if (maxConcurrentSeqsAtMaxContext < 1) {
    notes.push(
      `KV budget (${gib(kvBudgetBytes)}) holds ${maxTokens} tokens, less than one ` +
        `${cache.maxContext}-token sequence (${gib(mem.bytesPerSequence)}). ` +
        `Lower max_context to <= ${maxTokens} or use a smaller KV dtype.`,
    );
  }

  const tail = cache.maxContext % cache.blockSize;
  if (tail !== 0) {
    notes.push(
      `max_context ${cache.maxContext} is not a multiple of block_size ${cache.blockSize}; ` +
        `each full sequence wastes ${cache.blockSize - tail} token slots in its last block.`,
    );
  }
  if (cache.dtype !== plan.model.dtype) {
    notes.push(`KV cache stored as ${cache.dtype} (weights are ${plan.model.dtype}).`);
  }

  return { usableBytes, kvBudgetBytes, totalBlocks, maxTokens, maxConcurrentSeqsAtMaxContext, fits, notes };
}
