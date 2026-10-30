# kvplan

A DSL + compiler that answers: "Will this model fit on my GPU, and how should I
configure its KV cache?" Input is a `.kv` file; output is memory math, capacity,
cache layout, and paste-ready vLLM flags.

```
model llama3_8b {
  layers 32; kv_heads 8; head_dim 128; dtype fp16
}
cache {
  block_size 16; max_context 8192; quantize int8
}
target gpu "A10G" { memory 24GB }
```

## Pipeline

source → lexer → parser (AST) → validate (merge presets, check, produce `Plan`)
→ memory pass → capacity pass → codegen (vLLM flags) → CLI table

The contract between stages is `src/types.ts` (`Plan`, `MemoryReport`,
`CapacityReport`). Don't change it without coordinating with the other session.

## Two-session ownership (parallel work)

Two Claude Code sessions work in this directory at once. Edit only your files.

| Owner | Files |
|---|---|
| Lead (front end) | `src/types.ts`, `src/lexer.ts`, `src/parser.ts`, `src/ast.ts`, `src/errors.ts`, `src/validate.ts`, `src/cli.ts`, `test/front*.test.ts`, `examples/`, `web/`, `scripts/`, `README.md`, `CLAUDE.md`, `package.json` |
| Worker (back end) | `src/memory.ts`, `src/capacity.ts`, `src/presets.ts`, `src/codegen/vllm.ts`, `test/back*.test.ts`, `notebooks/`, `validation/` |

Only the lead runs `git commit`. Workers leave changes uncommitted and report back.

## Conventions

- TypeScript run directly by Node 23 type stripping. **Zero dependencies** — no
  npm/yarn packages. Tests use `node:test` + `node:assert/strict`.
- Type stripping rules: use `import type` for type-only imports, include the
  `.ts` extension in relative imports, no `enum`/`namespace`/parameter properties.
- Run tests: `node --test 'test/*.test.ts'` (a bare directory arg fails on Node 23) · Run CLI: `node src/cli.ts fit examples/llama3_8b.kv --gpu A10G`
- Sizes: `GB` = 1e9 bytes, `GiB` = 2^30 bytes. Be explicit in output which you show.

## Domain notes

- KV bytes/token = 2 (K and V) × layers × kv_heads × head_dim × dtype_bytes.
  Llama 3 8B fp16 → 131,072 B (128 KiB) per token.
- vLLM pages the cache in blocks of `block_size` tokens; a sequence occupies
  ceil(tokens / block_size) blocks.
- vLLM `--kv-cache-dtype` supports `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2` — not
  int8. Codegen must warn when the plan asks for an unsupported dtype.
- Real GPUs expose less than the marketing number (A10G "24GB" ≈ 22.5 GiB usable).
  vLLM also reserves memory for activations/CUDA graphs; model that as a
  documented assumption, not a hidden constant.
