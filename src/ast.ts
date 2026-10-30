// Syntax tree produced by the parser. Every node keeps its source span so later
// passes can point errors at the exact token.

export interface Span {
  line: number; // 1-based
  col: number; // 1-based
  length: number; // characters to underline; 0 = no specific token
}

export type Value =
  | { kind: "number"; value: number; unit: string; raw: string; span: Span }
  | { kind: "ident"; value: string; span: Span }
  | { kind: "string"; value: string; span: Span };

export interface Field {
  key: string;
  keySpan: Span;
  value: Value;
}

export interface Block {
  kind: "model" | "cache" | "target";
  span: Span; // the keyword
  name?: string; // model name or GPU name
  nameSpan?: Span;
  from?: { name: string; span: Span }; // model x from <preset>
  fields: Field[];
}

export interface KvFile {
  blocks: Block[];
}
