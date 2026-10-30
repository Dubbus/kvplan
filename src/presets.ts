// Model and GPU presets. Validate merges these into a Plan when a `.kv` file
// names a known model/GPU without spelling out every field.

import type { Plan } from "./types.ts";

const MiB = 2 ** 20;

/** Architecture presets, keyed by the name used in `model <name> { ... }`.
 *  Values come from each model's Hugging Face config.json:
 *    layers   = num_hidden_layers
 *    numHeads = num_attention_heads
 *    kvHeads  = num_key_value_heads
 *    headDim  = hidden_size / num_attention_heads (none of these set head_dim explicitly
 *               except Mistral v0.3, which sets head_dim: 128)
 *    dtype    = torch_dtype
 *  params is the total parameter count listed on the model card / safetensors index. */
export const MODEL_PRESETS: Record<string, Plan["model"]> = {
  // https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/config.json
  // hidden_size 4096, 32 layers, 32 heads, 8 KV heads, bf16; 8.03B params.
  llama3_8b: {
    name: "llama3_8b",
    layers: 32,
    numHeads: 32,
    kvHeads: 8,
    headDim: 128,
    params: 8.03e9,
    dtype: "bf16",
  },
  // https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct/blob/main/config.json
  // hidden_size 8192, 80 layers, 64 heads, 8 KV heads, bf16; 70.6B params.
  llama3_70b: {
    name: "llama3_70b",
    layers: 80,
    numHeads: 64,
    kvHeads: 8,
    headDim: 128,
    params: 70.6e9,
    dtype: "bf16",
  },
  // https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/main/config.json
  // hidden_size 4096, 32 layers, 32 heads, 8 KV heads, head_dim 128, bf16; 7.25B params.
  mistral_7b: {
    name: "mistral_7b",
    layers: 32,
    numHeads: 32,
    kvHeads: 8,
    headDim: 128,
    params: 7.25e9,
    dtype: "bf16",
  },
  // https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json
  // hidden_size 3584, 28 layers, 28 heads, 4 KV heads, bf16; 7.62B params.
  "qwen2.5_7b": {
    name: "qwen2.5_7b",
    layers: 28,
    numHeads: 28,
    kvHeads: 4,
    headDim: 128,
    params: 7.62e9,
    dtype: "bf16",
  },
};

/** Serving metadata that isn't part of the Plan contract: the Hugging Face repo
 *  id for `vllm serve`, and the model's native context limit
 *  (config.json max_position_embeddings). Keyed like MODEL_PRESETS. */
export const MODEL_SERVING: Record<string, { hfId: string; maxContext: number }> = {
  llama3_8b: { hfId: "meta-llama/Meta-Llama-3-8B-Instruct", maxContext: 8192 },
  llama3_70b: { hfId: "meta-llama/Meta-Llama-3-70B-Instruct", maxContext: 8192 },
  mistral_7b: { hfId: "mistralai/Mistral-7B-Instruct-v0.3", maxContext: 32768 },
  // 32768 natively; 131072 only with YaRN rope_scaling added to config.json.
  "qwen2.5_7b": { hfId: "Qwen/Qwen2.5-7B-Instruct", maxContext: 32768 },
};

/** Device memory as the driver reports it (`nvidia-smi --query-gpu=memory.total`,
 *  in MiB), not the marketing "GB" figure. This is the total that
 *  --gpu-memory-utilization is a fraction of. Sources: NVIDIA product datasheets
 *  for the nominal size, and nvidia-smi output on the named instance types for
 *  the reported size. ECC on/off and driver version can shift these by a few
 *  hundred MiB. */
export const GPU_PRESETS: Record<string, { memoryBytes: number }> = {
  // AWS g5 (A10G, "24GB" GDDR6): nvidia-smi reports 23028MiB (~22.5 GiB).
  A10G: { memoryBytes: 23028 * MiB },
  // GCP g2 (L4, "24GB" GDDR6): nvidia-smi reports 23034MiB.
  L4: { memoryBytes: 23034 * MiB },
  // GCP/Colab T4 ("16GB" GDDR6): nvidia-smi reports 15360MiB.
  T4: { memoryBytes: 15360 * MiB },
  // A100-SXM4-40GB (HBM2): nvidia-smi reports 40960MiB.
  "A100-40GB": { memoryBytes: 40960 * MiB },
  // A100-SXM4-80GB (HBM2e): nvidia-smi reports 81920MiB.
  "A100-80GB": { memoryBytes: 81920 * MiB },
  // H100 80GB HBM3 (SXM5): nvidia-smi reports 81559MiB.
  "H100-80GB": { memoryBytes: 81559 * MiB },
  // GeForce RTX 4090 ("24GB" GDDR6X): nvidia-smi reports 24564MiB.
  "RTX 4090": { memoryBytes: 24564 * MiB },
};
