import assert from "node:assert/strict";
import { test } from "node:test";
import { formatDiagnostic, KvError } from "../src/errors.ts";
import { lex } from "../src/lexer.ts";
import { parse } from "../src/parser.ts";
import type { Plan } from "../src/types.ts";
import { validate } from "../src/validate.ts";
import type { Diagnostic } from "../src/errors.ts";

// Fixture presets, so these tests don't depend on the real preset table.
const llama: Plan["model"] = { name: "llama3_8b", layers: 32, numHeads: 32, kvHeads: 8, headDim: 128, params: 8.03e9, dtype: "bf16" };
const presets = { models: { llama3_8b: llama }, gpus: { A10G: { memoryBytes: 23028 * 2 ** 20 } } };

function syntaxError(src: string): Diagnostic {
  try {
    parse(src);
  } catch (e) {
    if (e instanceof KvError) return e.diagnostic;
    throw e;
  }
  assert.fail("expected a syntax error");
}

const check = (src: string, gpu?: string) => validate(parse(src), { presets, gpu });
const errors = (src: string, gpu?: string) => check(src, gpu).diagnostics.filter((d) => d.severity === "error");

test("lexer: numbers carry units; idents allow dots", () => {
  const toks = lex('memory 24GB; params 8.03B # comment\nqwen2.5_7b "A10G"');
  assert.deepEqual(
    toks.map((t) => [t.kind, t.value ?? t.text, t.unit ?? ""]),
    [["ident", "memory", ""], ["number", 24, "GB"], [";", ";", ""], ["ident", "params", ""], ["number", 8.03, "B"], ["ident", "qwen2.5_7b", ""], ["string", "A10G", ""], ["eof", "", ""]],
  );
  assert.deepEqual(toks[5].span, { line: 2, col: 1, length: 10 });
});

test("lexer: malformed number and stray character", () => {
  assert.throws(() => lex("layers 3x2"), /malformed number '3x2'/);
  assert.throws(() => lex("layers @"), /unexpected character '@'/);
  assert.throws(() => lex('target gpu "A10G'), /unterminated string/);
});

test("parser: the README example", () => {
  const file = parse(`model llama3_8b {
  layers 32; kv_heads 8; head_dim 128; dtype fp16
}
cache {
  block_size 16; max_context 8192; quantize int8
}
target gpu "A10G" { memory 24GB }`);
  assert.deepEqual(file.blocks.map((b) => [b.kind, b.name, b.fields.length]), [["model", "llama3_8b", 4], ["cache", undefined, 3], ["target", "A10G", 1]]);
});

test("parser: missing '}' points at the unclosed brace", () => {
  const d = syntaxError("model x {\n  layers 32\ncache { max_context 8k }");
  assert.match(d.message, /unclosed '\{' in model block/);
  assert.deepEqual(d.span, { line: 1, col: 9, length: 1 });
  assert.match(d.hint!, /before 'cache' starts \(line 3\)/);
});

test("parser: missing value and misspelled block keyword", () => {
  assert.match(syntaxError("cache { max_context; }").message, /expected a value for 'max_context', found ';'/);
  const d = syntaxError("modle x { }");
  assert.equal(d.hint, "did you mean 'model'?");
});

test("validate: preset fills in missing fields; explicit fields win", () => {
  const { plan, diagnostics } = check('model llama3_8b { dtype fp16 }\ncache { max_context 8k; quantize int8 }\ntarget gpu "a10g"');
  assert.deepEqual(diagnostics, []);
  assert.equal(plan!.model.dtype, "fp16");
  assert.equal(plan!.model.preset, "llama3_8b");
  assert.equal(plan!.cache.maxContext, 8192);
  assert.equal(plan!.cache.blockSize, 16);
  assert.equal(plan!.target.name, "A10G");
  assert.equal(plan!.target.utilization, 0.9);
});

test("validate: `from` keeps the user's name and records the preset", () => {
  const { plan } = check("model mine from llama3_8b { layers 40 }\ncache { max_context 4096 }", "A10G");
  assert.equal(plan!.model.name, "mine");
  assert.equal(plan!.model.preset, "llama3_8b");
  assert.equal(plan!.model.layers, 40);
});

test("validate: unknown field gets a suggestion at the right position", () => {
  const [d] = errors('model llama3_8b {\n  kv_head 8\n}\ncache { max_context 8k }\ntarget gpu "A10G"');
  assert.equal(d.message, "unknown field 'kv_head' in model block");
  assert.equal(d.hint, "did you mean 'kv_heads'?");
  assert.deepEqual(d.span, { line: 2, col: 3, length: 7 });
});

test("validate: GQA and head_dim divisibility", () => {
  const base = '\ncache { max_context 8k }\ntarget gpu "A10G"';
  assert.match(errors("model x { layers 2; heads 32; kv_heads 6; head_dim 64; params 1B }" + base)[0].message, /multiple of kv_heads/);
  const [d] = errors("model x { layers 2; heads 32; kv_heads 8; hidden_size 4100; params 1B }" + base);
  assert.match(d.message, /hidden_size 4100 doesn.t divide evenly across 32 heads/);
});

test("validate: missing fields, units and blocks", () => {
  assert.match(errors('model custom { layers 32 }\ncache { max_context 8k }\ntarget gpu "A10G"')[0].message, /missing heads, kv_heads, head_dim, params/);
  assert.match(errors('model llama3_8b { }\ncache { max_context 8k }\ntarget gpu "X" { memory 24 }')[0].message, /'memory' needs a unit/);
  assert.match(errors("model llama3_8b { }\ncache { max_context 8k }")[0].message, /no target GPU/);
  assert.match(errors('model llama3_8b { }\ncache { max_context 8k }\ntarget gpu "A10" ')[0].hint!, /did you mean 'A10G'/);
});

test("validate: marketing memory size warns; --gpu overrides the file", () => {
  const { plan, diagnostics } = check('model llama3_8b { }\ncache { max_context 8k }\ntarget gpu "A10G" { memory 24GiB }');
  assert.equal(plan!.target.memoryBytes, 24 * 2 ** 30);
  assert.match(diagnostics[0].message, /A10G actually reports 22.49 GiB, not 24.00 GiB/);
  const over = check('model llama3_8b { }\ncache { max_context 8k }\ntarget gpu "H100" { memory 80GB }', "A10G");
  assert.equal(over.plan!.target.memoryBytes, 23028 * 2 ** 20);
  assert.match(over.diagnostics[0].message, /--gpu A10G overrides target "H100"/);
});

test("formatDiagnostic renders a caret under the token", () => {
  const src = "model x {\n  kv_head 8\n}";
  const out = formatDiagnostic({ severity: "error", message: "unknown field", span: { line: 2, col: 3, length: 7 }, hint: "did you mean 'kv_heads'?" }, src, "x.kv");
  assert.equal(out, ["error: unknown field", "  --> x.kv:2:3", "  |", "2 |   kv_head 8", "  |   ^^^^^^^ did you mean 'kv_heads'?"].join("\n"));
});
