import type { Span } from "./ast.ts";

export interface Diagnostic {
  severity: "error" | "warning";
  message: string;
  span: Span;
  hint?: string;
}

/** Thrown by the lexer and parser, which stop at the first syntax error.
 *  The validate pass collects diagnostics instead of throwing. */
export class KvError extends Error {
  diagnostic: Diagnostic;
  constructor(diagnostic: Diagnostic) {
    super(diagnostic.message);
    this.diagnostic = diagnostic;
  }
}

/** Renders a diagnostic rustc-style:
 *
 *   error: unknown field 'kv_head' in model block
 *     --> llama.kv:2:14
 *     |
 *   2 |   layers 32; kv_head 8
 *     |              ^^^^^^^ did you mean 'kv_heads'?
 */
export function formatDiagnostic(d: Diagnostic, source: string, file: string): string {
  const { line, col, length } = d.span;
  const out = [`${d.severity}: ${d.message}`, `  --> ${file}:${line}:${col}`];
  const text = source.split("\n")[line - 1];
  if (length > 0 && text !== undefined) {
    const pad = " ".repeat(String(line).length);
    const caret = " ".repeat(col - 1) + "^".repeat(length);
    out.push(`${pad} |`, `${line} | ${text}`, `${pad} | ${caret}${d.hint ? " " + d.hint : ""}`);
  } else if (d.hint) {
    out.push(`   = ${d.hint}`);
  }
  return out.join("\n");
}

/** Closest candidate within a small edit distance, for "did you mean" hints. */
export function suggest(word: string, candidates: string[]): string | undefined {
  let best: string | undefined;
  let bestDist = Math.min(2, Math.max(1, Math.ceil(word.length / 3)));
  for (const c of candidates) {
    const d = editDistance(word.toLowerCase(), c.toLowerCase());
    if (d <= bestDist) {
      best = c;
      bestDist = d - 1; // strictly better matches only
    }
  }
  return best;
}

function editDistance(a: string, b: string): number {
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    prev = cur;
  }
  return prev[b.length];
}
