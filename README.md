# kvplan

> 🚧 **In progress.** The CLI and the [web playground](https://dubbus.github.io/kvplan/) work. Validation against real vLLM runs is underway. See [Status](#status).

**[Try it in the browser →](https://dubbus.github.io/kvplan/)** Edit the plan on the left; the memory breakdown, capacity chart, cache layout and vLLM command update as you type.

**Will this model fit on my GPU, and how should I configure its KV cache?**

kvplan is a small DSL and compiler. You describe a model, its cache setup and a target GPU. It computes the KV cache memory, works out how many sequences of each length fit after the model weights, and generates the matching `vllm serve` command.

```
model llama3_8b {
  layers 32; kv_heads 8; head_dim 128; dtype fp16
}
cache {
  block_size 16; max_context 8192; quantize int8
}
target gpu "A10G" { memory 24GB }
```

```
$ node src/cli.ts fit examples/llama3_8b.kv

kvplan · llama3_8b on A10G · KV int8, block 16

  Memory
  KV per token             64.00 KiB   2 × 32 layers × 8 kv_heads × 128 × int8
  KV per block              1.00 MiB   16 tokens
  KV per sequence         512.00 MiB   8,192 tokens
  Weights                  14.96 GiB   8.03B params × fp16
  GQA: 32 query heads share 8 KV heads (4× smaller cache than MHA)

  Capacity
  GPU memory               22.35 GiB
  Usable                   20.12 GiB   × 0.9 utilization
  KV budget                 3.66 GiB   after weights and reserve
  Blocks                       3,747
  Tokens in cache             59,952
  Fits                           yes

  Max batch by context length
  1,024 tokens                    58   sequences
  2,048 tokens                    29   sequences
  4,096 tokens                    14   sequences
  8,192 tokens                     7   sequences

  vLLM

  # WARNING: vLLM has no int8 KV cache (--kv-cache-dtype accepts auto|fp8|fp8_e4m3|fp8_e5m2); using fp8, which has the same 1 byte/elem footprint, so capacity numbers still hold.
  vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --max-model-len 8192 \
    --block-size 16 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 7
```

Errors point at the exact token and suggest a fix:

```
error: unknown field 'kv_head' in model block
  --> examples/broken.kv:2:14
  |
2 |   layers 32; kv_head 8
  |              ^^^^^^^ did you mean 'kv_heads'?

error: unknown GPU 'A100'
  --> examples/broken.kv:5:12
  |
5 | target gpu "A100"
  |            ^^^^^^ did you mean 'A100-40GB' or 'A100-80GB'?
```

## How it works

It's a classic compiler pipeline, and each stage has one job:

```
source ─▶ lexer ─▶ recursive-descent parser ─▶ AST
       ─▶ validate (merge presets, convert units, check GQA / head_dim divisibility) ─▶ Plan
       ─▶ memory pass ─▶ capacity pass ─▶ codegen (vLLM flags)
```

- **Memory:** KV bytes per token = 2 (K and V) × layers × kv_heads × head_dim × dtype bytes. vLLM pages the cache in blocks, so each sequence is rounded up to whole blocks.
- **Capacity:** usable memory = GPU memory × `gpu_memory_utilization`. The KV budget is whatever's left after the weights and an activation/CUDA-graph reserve. The reserve is a stated assumption, and the output shows it.
- **Codegen:** the generated command flags anything vLLM can't do, such as an int8 KV cache, unsupported block sizes, or a context longer than the model supports.

Presets cover Llama 3 8B and 70B, Mistral 7B and Qwen2.5 7B, plus A10G, L4, T4, A100 40/80GB, H100 and RTX 4090. You can start a custom model from a preset: `model mine from llama3_8b { layers 40 }`.

## Usage

Requires Node ≥ 23.6. It has **zero dependencies**: Node runs the TypeScript directly.

```
node src/cli.ts fit   <file.kv> [--gpu NAME] [--json]
node src/cli.ts check <file.kv> [--gpu NAME]
node --test 'test/*.test.ts'
yarn serve        # build the playground and open http://localhost:8917
```

The playground runs the same compiler as the CLI. `scripts/build-web.ts` strips the TypeScript types with Node's built-in `stripTypeScriptTypes`, so there's no bundler.

## Status

- [x] Lexer and recursive-descent parser with line/column error messages
- [x] Validation pass: presets, units (`24GB`, `8k`, `8.03B`), "did you mean" suggestions
- [x] Memory and capacity passes, max batch for each context length
- [x] vLLM flag generation
- [ ] **Validation against real vLLM runs:** measure actual KV cache capacity on GPUs and report kvplan's prediction error
- [x] Web playground: live editor, memory breakdown, capacity chart, GQA and paged-block layout diagrams, shareable links
- [x] Side-by-side comparisons: 16 / 8 / 4-bit KV cache × block sizes 8 / 16 / 32
