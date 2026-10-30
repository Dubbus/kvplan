// Codegen: a paste-ready `vllm serve` command, with `# WARNING:` comment lines
// for anything in the plan that vLLM can't do as written.

import { MODEL_SERVING } from "../presets.ts";
import type { CapacityReport, DType, Plan } from "../types.ts";

/** Block sizes vLLM's CUDA attention backends accept. */
export const VLLM_BLOCK_SIZES = [8, 16, 32];

/** Map a plan KV dtype to a --kv-cache-dtype value, plus a warning if lossy. */
function kvCacheDtype(cacheDtype: DType, modelDtype: DType): { flag: string; warning?: string } {
  switch (cacheDtype) {
    case "fp8":
      return { flag: "fp8" };
    case "int8":
      return {
        flag: "fp8",
        warning:
          "vLLM has no int8 KV cache (--kv-cache-dtype accepts auto|fp8|fp8_e4m3|fp8_e5m2); " +
          "using fp8, which has the same 1 byte/elem footprint, so capacity numbers still hold.",
      };
    case "int4":
      return {
        flag: "fp8",
        warning:
          "vLLM has no int4 KV cache; using fp8 (1 byte/elem). Real capacity is about HALF " +
          "what kvplan computed at 0.5 bytes/elem.",
      };
    case "fp32":
      return {
        flag: "auto",
        warning: `vLLM can't store the KV cache in fp32; auto uses the model dtype (${modelDtype}), ` +
          "so the cache takes half the memory kvplan computed.",
      };
    default: // fp16 / bf16
      if (cacheDtype !== modelDtype && (modelDtype === "fp16" || modelDtype === "bf16")) {
        return {
          flag: "auto",
          warning: `KV dtype ${cacheDtype} differs from model dtype ${modelDtype}; vLLM's auto ` +
            `stores the cache in the model dtype (same 2-byte size).`,
        };
      }
      return { flag: "auto" };
  }
}

export function emitVllm(plan: Plan, cap: CapacityReport): string {
  const warnings: string[] = [];
  // `model my_llama from llama3_8b` keeps name "my_llama" but preset "llama3_8b".
  const presetKey = plan.model.preset ?? plan.model.name;
  const serving = MODEL_SERVING[presetKey];

  let modelId = presetKey;
  if (!serving) {
    warnings.push(`No Hugging Face id known for "${presetKey}"; replace it with the real repo id.`);
  } else {
    modelId = serving.hfId;
  }

  let maxModelLen = plan.cache.maxContext;
  if (serving && maxModelLen > serving.maxContext) {
    warnings.push(
      `max_context ${maxModelLen} exceeds ${modelId}'s native ${serving.maxContext} ` +
        `(max_position_embeddings); vLLM rejects it without rope scaling. Clamped to ${serving.maxContext}.`,
    );
    maxModelLen = serving.maxContext;
  }

  let blockSize = plan.cache.blockSize;
  if (!VLLM_BLOCK_SIZES.includes(blockSize)) {
    const nearest = VLLM_BLOCK_SIZES.reduce((a, b) => (Math.abs(b - blockSize) < Math.abs(a - blockSize) ? b : a));
    warnings.push(
      `block_size ${blockSize} is not supported by vLLM's CUDA backends (${VLLM_BLOCK_SIZES.join(", ")}); ` +
        `using ${nearest}. Block counts above assume ${blockSize}.`,
    );
    blockSize = nearest;
  }

  const kv = kvCacheDtype(plan.cache.dtype, plan.model.dtype);
  if (kv.warning) warnings.push(kv.warning);

  if (!cap.fits) {
    warnings.push(
      `This plan does not fit on ${plan.target.name} (see notes); vLLM will fail at startup ` +
        `with an out-of-memory or "insufficient KV cache" error.`,
    );
  }
  // Size max-num-seqs so every running sequence can reach max_context without preemption.
  const maxNumSeqs = Math.max(1, cap.maxConcurrentSeqsAtMaxContext);

  const args = [
    `vllm serve ${modelId}`,
    `--max-model-len ${maxModelLen}`,
    `--block-size ${blockSize}`,
    `--kv-cache-dtype ${kv.flag}`,
    `--gpu-memory-utilization ${plan.target.utilization}`,
    `--max-num-seqs ${maxNumSeqs}`,
  ];
  const lines = warnings.map((w) => `# WARNING: ${w}`);
  lines.push(args.join(" \\\n  "));
  return lines.join("\n") + "\n";
}
