// Recursive descent parser. Grammar:
//
//   file   := block*
//   block  := 'model' IDENT ('from' IDENT)? body
//           | 'cache' body
//           | 'target' 'gpu' (STRING | IDENT) body?
//   body   := '{' (field ';'?)* '}'
//   field  := IDENT value
//   value  := NUMBER | IDENT | STRING
//
// Each field is exactly one key and one value, so ';' separators are optional.

import type { Block, Field, KvFile, Value } from "./ast.ts";
import { KvError, suggest } from "./errors.ts";
import { lex } from "./lexer.ts";
import type { Token } from "./lexer.ts";

const BLOCK_KEYWORDS = ["model", "cache", "target"];

export function parse(src: string): KvFile {
  return new Parser(lex(src)).file();
}

function describe(t: Token): string {
  if (t.kind === "eof") return "end of file";
  if (t.kind === "string") return `"${t.text}"`;
  return `'${t.text}'`;
}

class Parser {
  private tokens: Token[];
  private i = 0;

  constructor(tokens: Token[]) {
    this.tokens = tokens;
  }

  file(): KvFile {
    const blocks: Block[] = [];
    while (this.peek().kind !== "eof") blocks.push(this.block());
    return { blocks };
  }

  private peek(): Token {
    return this.tokens[this.i];
  }

  private next(): Token {
    return this.tokens[this.i++];
  }

  private fail(t: Token, message: string, hint?: string): never {
    throw new KvError({ severity: "error", message, span: { ...t.span, length: Math.max(t.span.length, 1) }, hint });
  }

  private expectIdent(what: string, hint?: string): Token {
    const t = this.next();
    if (t.kind !== "ident") this.fail(t, `expected ${what}, found ${describe(t)}`, hint);
    return t;
  }

  private block(): Block {
    const kw = this.next();
    if (kw.kind === "ident") {
      if (kw.text === "model") return this.model(kw);
      if (kw.text === "cache") return { kind: "cache", span: kw.span, fields: this.body(kw) };
      if (kw.text === "target") return this.target(kw);
    }
    const guess = kw.kind === "ident" ? suggest(kw.text, BLOCK_KEYWORDS) : undefined;
    this.fail(kw, `expected 'model', 'cache' or 'target', found ${describe(kw)}`, guess && `did you mean '${guess}'?`);
  }

  private model(kw: Token): Block {
    const name = this.expectIdent("a model name", "e.g. model llama3_8b { ... }");
    let from: Block["from"];
    if (this.peek().kind === "ident" && this.peek().text === "from") {
      this.next();
      const preset = this.expectIdent("a preset name after 'from'", "e.g. model mine from llama3_8b { ... }");
      from = { name: preset.text, span: preset.span };
    }
    return { kind: "model", span: kw.span, name: name.text, nameSpan: name.span, from, fields: this.body(kw) };
  }

  private target(kw: Token): Block {
    const k = this.expectIdent("'gpu'", 'e.g. target gpu "A10G"');
    if (k.text !== "gpu") this.fail(k, `unknown target kind '${k.text}'`, "only 'gpu' is supported");
    const name = this.next();
    if (name.kind !== "string" && name.kind !== "ident") this.fail(name, `expected a GPU name, found ${describe(name)}`, 'e.g. target gpu "A10G"');
    const fields = this.peek().kind === "{" ? this.body(kw) : [];
    return { kind: "target", span: kw.span, name: name.text, nameSpan: name.span, fields };
  }

  private body(kw: Token): Field[] {
    const open = this.next();
    if (open.kind !== "{") this.fail(open, `expected '{' to start the ${kw.text} block, found ${describe(open)}`);
    const fields: Field[] = [];
    for (;;) {
      const t = this.peek();
      if (t.kind === "}") {
        this.next();
        return fields;
      }
      if (t.kind === ";") {
        this.next();
        continue;
      }
      if (t.kind === "eof" || (t.kind === "ident" && BLOCK_KEYWORDS.includes(t.text))) {
        // Most likely a forgotten '}' — point at the brace that was never closed.
        const where = t.kind === "eof" ? "the file ends" : `'${t.text}' starts`;
        this.fail(open, `unclosed '{' in ${kw.text} block`, `add '}' before ${where} (line ${t.span.line})`);
      }
      if (t.kind !== "ident") this.fail(t, `expected a field name, found ${describe(t)}`);
      this.next();
      fields.push({ key: t.text, keySpan: t.span, value: this.value(t) });
    }
  }

  private value(key: Token): Value {
    const t = this.peek();
    if (t.kind === "number") {
      this.next();
      return { kind: "number", value: t.value!, unit: t.unit!, raw: t.text, span: t.span };
    }
    if (t.kind === "ident" || t.kind === "string") {
      this.next();
      return { kind: t.kind, value: t.text, span: t.span };
    }
    this.fail(t, `expected a value for '${key.text}', found ${describe(t)}`);
  }
}
