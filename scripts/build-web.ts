// Builds the web playground: strips types from the compiler sources so the
// browser runs the exact code the CLI runs. No bundler, no dependencies.
//
//   node scripts/build-web.ts   →   web/dist/*.js

import { mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";
import { dirname, join, relative } from "node:path";

const SRC = "src";
const OUT = "web/dist";
const SKIP = new Set(["cli.ts"]); // Node-only entry point

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    return statSync(path).isDirectory() ? walk(path) : path.endsWith(".ts") ? [path] : [];
  });
}

rmSync(OUT, { recursive: true, force: true });
for (const file of walk(SRC)) {
  if (SKIP.has(relative(SRC, file))) continue;
  const js = stripTypeScriptTypes(readFileSync(file, "utf8"), { mode: "strip" })
    // Browsers resolve imports literally, so point them at the emitted .js files.
    .replace(/(from\s+["'])(\.{1,2}\/[^"']+)\.ts(["'])/g, "$1$2.js$3");
  const out = join(OUT, relative(SRC, file)).replace(/\.ts$/, ".js");
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, js);
  console.log(`${file} → ${out}`);
}
