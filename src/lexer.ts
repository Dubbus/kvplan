import type { Span } from "./ast.ts";
import { KvError } from "./errors.ts";

export type TokenKind = "ident" | "number" | "string" | "{" | "}" | ";" | "eof";

export interface Token {
  kind: TokenKind;
  text: string; // source text (string tokens: contents without quotes)
  span: Span;
  value?: number; // number tokens
  unit?: string; // number tokens: suffix like "GB", "k", "B"; "" if none
}

const isDigit = (c: string | undefined) => c !== undefined && c >= "0" && c <= "9";
const isAlpha = (c: string | undefined) => c !== undefined && /[A-Za-z]/.test(c);
const isIdentChar = (c: string | undefined) => c !== undefined && /[A-Za-z0-9_.\-]/.test(c);

/** Splits source into tokens. Comments run from `#` or `//` to end of line.
 *  Numbers carry an attached unit suffix (`24GB`, `8.03B`, `8k`); the validate
 *  pass decides what each unit means for each field. */
export function lex(src: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  let line = 1;
  let col = 1;
  const advance = () => {
    if (src[i] === "\n") {
      line++;
      col = 1;
    } else {
      col++;
    }
    i++;
  };

  while (i < src.length) {
    const c = src[i];
    if (c === " " || c === "\t" || c === "\r" || c === "\n") {
      advance();
      continue;
    }
    if (c === "#" || (c === "/" && src[i + 1] === "/")) {
      while (i < src.length && src[i] !== "\n") advance();
      continue;
    }

    const start = i;
    const pos = { line, col };
    const span = () => ({ ...pos, length: i - start });

    if (c === "{" || c === "}" || c === ";") {
      advance();
      tokens.push({ kind: c, text: c, span: span() });
    } else if (c === '"') {
      advance();
      while (i < src.length && src[i] !== '"' && src[i] !== "\n") advance();
      if (src[i] !== '"') {
        throw new KvError({ severity: "error", message: "unterminated string", span: span(), hint: 'add a closing "' });
      }
      advance();
      tokens.push({ kind: "string", text: src.slice(start + 1, i - 1), span: span() });
    } else if (isDigit(c)) {
      while (isDigit(src[i]) || src[i] === "_") advance();
      if (src[i] === "." && isDigit(src[i + 1])) {
        advance();
        while (isDigit(src[i]) || src[i] === "_") advance();
      }
      const numEnd = i;
      while (isAlpha(src[i])) advance();
      if (isIdentChar(src[i])) {
        while (isIdentChar(src[i])) advance();
        throw new KvError({
          severity: "error",
          message: `malformed number '${src.slice(start, i)}'`,
          span: span(),
          hint: "numbers look like 32, 0.9, 24GB or 8.03B",
        });
      }
      tokens.push({
        kind: "number",
        text: src.slice(start, i),
        span: span(),
        value: Number(src.slice(start, numEnd).replaceAll("_", "")),
        unit: src.slice(numEnd, i),
      });
    } else if (isAlpha(c) || c === "_") {
      while (isIdentChar(src[i])) advance();
      tokens.push({ kind: "ident", text: src.slice(start, i), span: span() });
    } else {
      advance();
      throw new KvError({ severity: "error", message: `unexpected character '${c}'`, span: span() });
    }
  }

  tokens.push({ kind: "eof", text: "", span: { line, col, length: 0 } });
  return tokens;
}
