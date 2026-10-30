// kvplan playground. Everything runs client-side on the compiler in ./dist,
// which is src/*.ts with the types stripped (see scripts/build-web.ts).

import { ACTIVATION_RESERVE_BYTES, computeCapacity } from "./dist/capacity.js";
import { emitVllm } from "./dist/codegen/vllm.js";
import { formatDiagnostic, KvError } from "./dist/errors.js";
import { blocksFor, computeMemory } from "./dist/memory.js";
import { parse } from "./dist/parser.js";
import { GPU_PRESETS, MODEL_PRESETS } from "./dist/presets.js";
import { DTYPE_BYTES } from "./dist/types.js";
import { validate } from "./dist/validate.js";

const DEFAULT_SOURCE = `# Describe a model, its KV cache and a GPU.
# Fields left out of a preset model (here: heads, params) come from the preset.
model llama3_8b {
  layers 32; kv_heads 8; head_dim 128; dtype fp16
}
cache {
  block_size 16; max_context 8192; quantize fp8
}
target gpu "A10G"
`;

// Three KV precisions compared side by side; colors are categorical slots 1-3.
const VARIANTS = [
  { label: "16-bit", dtype: "fp16", color: "var(--s1)", note: "fp16 / bf16" },
  { label: "8-bit", dtype: "fp8", color: "var(--s2)", note: "fp8 (int8 has the same footprint)" },
  { label: "4-bit", dtype: "int4", color: "var(--s3)", note: "int4, not a vLLM option today" },
];

const $ = (id) => document.getElementById(id);
const srcEl = $("src");
const gpuEl = $("gpu");
const presetEl = $("preset");
const tip = $("tooltip");
let last; // last successful compile, for re-rendering on resize

// ---------- formatting ----------

const int = (n) => Math.round(n).toLocaleString("en-US");
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
function bytes(n) {
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let i = 0;
  while (Math.abs(n) >= 1024 && i < units.length - 1) {
    n /= 1024;
    i++;
  }
  return `${i === 0 ? n : n.toFixed(n < 10 ? 2 : 1)} ${units[i]}`;
}
const ctxLabel = (n) => (n >= 1024 && n % 1024 === 0 ? `${n / 1024}k` : int(n));
const svgEl = (tag, attrs = {}) => {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
};

// ---------- tooltip ----------

function showTip(e, html) {
  tip.innerHTML = html;
  tip.style.display = "block";
  moveTip(e);
}
function moveTip(e) {
  const pad = 14;
  const r = tip.getBoundingClientRect();
  let x = e.clientX + pad;
  let y = e.clientY + pad;
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = e.clientY - r.height - pad;
  tip.style.left = `${Math.max(8, x)}px`;
  tip.style.top = `${Math.max(8, y)}px`;
}
const hideTip = () => (tip.style.display = "none");
function hover(el, html) {
  el.addEventListener("mouseenter", (e) => showTip(e, html()));
  el.addEventListener("mousemove", moveTip);
  el.addEventListener("mouseleave", hideTip);
}
const tipRow = (color, text) => `<div class="row"><i style="background:${color}"></i>${text}</div>`;

// ---------- compile ----------

function compile(source, gpu) {
  try {
    const r = validate(parse(source), { presets: { models: MODEL_PRESETS, gpus: GPU_PRESETS }, gpu: gpu || undefined });
    return { diagnostics: r.diagnostics, plan: r.plan };
  } catch (e) {
    if (e instanceof KvError) return { diagnostics: [e.diagnostic] };
    throw e;
  }
}

function analyze(plan, cachePatch = {}) {
  const p = { ...plan, cache: { ...plan.cache, ...cachePatch } };
  const mem = computeMemory(p);
  return { plan: p, mem, cap: computeCapacity(p, mem) };
}

function update() {
  const source = srcEl.value;
  const { diagnostics, plan } = compile(source, gpuEl.value);
  renderDiagnostics(diagnostics, source, plan);
  $("out").classList.toggle("stale", !plan);
  if (plan) {
    last = analyze(plan);
    render(last);
  }
  saveToHash();
}

// ---------- render ----------

function renderDiagnostics(diagnostics, source, plan) {
  const el = $("diagnostics");
  el.replaceChildren();
  if (plan && diagnostics.length === 0) {
    el.innerHTML = `<pre class="ok">✓ compiled: ${esc(plan.model.name)} on ${esc(plan.target.name)}</pre>`;
    return;
  }
  for (const d of diagnostics) {
    const pre = document.createElement("pre");
    pre.className = d.severity;
    pre.textContent = formatDiagnostic(d, source, "playground.kv") + "\n";
    el.append(pre);
  }
}

function render(a) {
  renderTiles(a);
  renderMemory(a);
  renderCapacityChart(a);
  renderCompare(a);
  renderGqa(a);
  renderPool(a);
  $("vllm").textContent = emitVllm(a.plan, a.cap).trimEnd();
}

function renderTiles({ plan, mem, cap }) {
  const tile = (label, value, note, cls = "") =>
    `<div class="tile"><div class="label">${label}</div><div class="value ${cls}">${value}</div><div class="note">${note}</div></div>`;
  $("tiles").innerHTML = [
    tile("KV per token", bytes(mem.bytesPerToken), `${plan.cache.dtype} cache`),
    tile("KV per sequence", bytes(mem.bytesPerSequence), `${int(plan.cache.maxContext)} tokens`),
    tile("Full-length sequences", int(cap.maxConcurrentSeqsAtMaxContext), `${int(cap.maxTokens)} tokens cached`),
    cap.fits
      ? tile("Fits?", "✓ Yes", esc(plan.target.name), "fit-yes")
      : tile("Fits?", "✗ No", esc(plan.target.name), "fit-no"),
  ].join("");
}

function renderMemory({ plan, mem, cap }) {
  const total = plan.target.memoryBytes;
  const held = total - cap.usableBytes;
  const kv = cap.totalBlocks * mem.bytesPerBlock;
  const segments = [
    { name: "Weights", value: mem.weightBytes, color: "var(--s1)", detail: `${(plan.model.params / 1e9).toFixed(2)}B params × ${plan.model.dtype}` },
    { name: "Activation reserve", value: ACTIVATION_RESERVE_BYTES, color: "var(--s2)", detail: "estimate: activations, CUDA graphs, context" },
    { name: "KV cache", value: kv, color: "var(--s3)", detail: `${int(cap.totalBlocks)} blocks × ${bytes(mem.bytesPerBlock)}` },
    { name: "Held back", value: held, color: "var(--held)", detail: `1 − gpu_memory_utilization (${plan.target.utilization})` },
  ].filter((s) => s.value > 0);
  const scale = Math.max(total, segments.reduce((sum, s) => sum + s.value, 0));

  $("mem-sub").textContent = `${plan.target.name}: ${bytes(total)} total, ${bytes(cap.usableBytes)} usable by vLLM`;
  const bar = $("membar");
  bar.replaceChildren();
  for (const s of segments) {
    const div = document.createElement("div");
    div.style.flex = `0 0 calc(${(s.value / scale) * 100}% - 2px)`;
    div.style.background = s.color;
    hover(div, () => `<strong>${s.name}</strong>${tipRow(s.color, `${bytes(s.value)} · ${((s.value / total) * 100).toFixed(1)}% of GPU`)}<div style="color:var(--ink-2)">${s.detail}</div>`);
    bar.append(div);
  }
  $("mem-legend").innerHTML = segments.map((s) => `<span><i style="background:${s.color}"></i>${s.name} <b>${bytes(s.value)}</b></span>`).join("");
  const over = mem.weightBytes + ACTIVATION_RESERVE_BYTES - cap.usableBytes;
  $("mem-over").textContent = over > 0 ? `✗ Weights + reserve exceed usable memory by ${bytes(over)}: the model won't load.` : "";
}

function renderCapacityChart(a) {
  const { plan } = a;
  const bs = plan.cache.blockSize;
  const maxCtx = plan.cache.maxContext;
  const xs = [];
  for (let c = 512; c < maxCtx; c *= 2) xs.push(c);
  xs.push(maxCtx);

  const series = VARIANTS.map((v) => {
    const { cap } = analyze(plan, { dtype: v.dtype });
    return { ...v, points: xs.map((x) => ({ x, y: Math.floor(cap.totalBlocks / blocksFor(x, bs)) })) };
  });
  const yMaxRaw = Math.max(1, ...series.flatMap((s) => s.points.map((p) => p.y)));
  const yMax = 10 ** Math.max(1, Math.ceil(Math.log10(yMaxRaw + 1)));

  const host = $("cap-chart");
  const W = Math.max(280, host.clientWidth);
  const H = 260;
  const m = { l: 44, r: 56, t: 12, b: 30 };
  const lx = Math.log2(xs[0]);
  const span = Math.log2(xs[xs.length - 1]) - lx;
  const X = (x) => (span === 0 ? (m.l + W - m.r) / 2 : m.l + ((Math.log2(x) - lx) / span) * (W - m.l - m.r));
  const Y = (y) => m.t + (1 - Math.log10(y) / Math.log10(yMax)) * (H - m.t - m.b);

  const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Max concurrent sequences by context length" });
  for (let p = 1; p <= yMax; p *= 10) {
    svg.append(svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(p), y2: Y(p), style: `stroke:var(--grid)` }));
    const t = svgEl("text", { x: m.l - 8, y: Y(p) + 4, "text-anchor": "end", "font-size": 11, style: "fill:var(--muted)" });
    t.textContent = p >= 1000 ? `${p / 1000}k` : p;
    svg.append(t);
  }
  svg.append(svgEl("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, style: "stroke:var(--axis)" }));
  const every = Math.ceil(xs.length / Math.max(2, Math.floor((W - m.l - m.r) / 48)));
  xs.forEach((x, i) => {
    if (i % every !== 0 && i !== xs.length - 1) return;
    const t = svgEl("text", { x: X(x), y: H - m.b + 18, "text-anchor": "middle", "font-size": 11, style: "fill:var(--muted)" });
    t.textContent = ctxLabel(x);
    svg.append(t);
  });

  const labels = [];
  for (const s of series) {
    const pts = s.points.filter((p) => p.y >= 1);
    if (pts.length === 0) continue;
    svg.append(svgEl("path", { d: pts.map((p, i) => `${i ? "L" : "M"}${X(p.x)},${Y(p.y)}`).join(""), style: `fill:none;stroke:${s.color};stroke-width:2;stroke-linejoin:round` }));
    for (const p of pts) svg.append(svgEl("circle", { cx: X(p.x), cy: Y(p.y), r: 4, style: `fill:${s.color};stroke:var(--surface);stroke-width:2` }));
    const end = pts[pts.length - 1];
    labels.push({ text: s.label, x: X(end.x) + 8, y: Y(end.y) + 4 });
  }
  // Direct labels at line ends, nudged apart so they never overlap.
  labels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 14);
  for (const l of labels) {
    const t = svgEl("text", { x: l.x, y: l.y, "font-size": 12, style: "fill:var(--ink-2)" });
    t.textContent = l.text;
    svg.append(t);
  }

  // Hover: crosshair snaps to the nearest context length.
  const cross = svgEl("line", { y1: m.t, y2: H - m.b, style: "stroke:var(--axis);visibility:hidden" });
  svg.append(cross);
  const hit = svgEl("rect", { x: m.l - 10, y: 0, width: W - m.l - m.r + 20, height: H, style: "fill:transparent" });
  svg.append(hit);
  hit.addEventListener("mousemove", (e) => {
    const px = e.clientX - svg.getBoundingClientRect().left;
    let i = 0;
    xs.forEach((x, j) => Math.abs(X(x) - px) < Math.abs(X(xs[i]) - px) && (i = j));
    cross.setAttribute("x1", X(xs[i]));
    cross.setAttribute("x2", X(xs[i]));
    cross.style.visibility = "visible";
    showTip(e, `<strong>${int(xs[i])} tokens</strong>` + series.map((s) => tipRow(s.color, `${s.label}: ${s.points[i].y ? int(s.points[i].y) + " sequences" : "doesn't fit"}`)).join(""));
  });
  hit.addEventListener("mouseleave", () => {
    cross.style.visibility = "hidden";
    hideTip();
  });
  host.replaceChildren(svg);

  const current = currentVariant(plan);
  $("cap-legend").innerHTML = series
    .map((s) => `<span><i style="background:${s.color}"></i>${s.label} <span style="color:var(--muted)">${s.note}${s === current ? " · your plan" : ""}</span></span>`)
    .join("");
}

function currentVariant(plan) {
  const b = DTYPE_BYTES[plan.cache.dtype];
  return VARIANTS.find((v) => DTYPE_BYTES[v.dtype] === b);
}

function renderCompare({ plan }) {
  const sizes = [...new Set([8, 16, 32, plan.cache.blockSize])].sort((a, b) => a - b);
  const current = currentVariant(plan);
  $("cmp-sub").textContent = `${plan.model.name} on ${plan.target.name}. Full-length (${int(plan.cache.maxContext)}-token) sequences that fit, by KV precision and block size. Larger blocks waste more of each sequence's last block; vLLM's CUDA kernels accept 8, 16 or 32.`;
  const head = `<tr><th>KV cache</th><th>Per token</th><th>Tokens cached</th>${sizes.map((s) => `<th>block ${s}${s === plan.cache.blockSize ? " (yours)" : ""}</th>`).join("")}</tr>`;
  const rows = VARIANTS.map((v) => {
    const base = analyze(plan, { dtype: v.dtype });
    const cells = sizes.map((s) => `<td>${int(analyze(plan, { dtype: v.dtype, blockSize: s }).cap.maxConcurrentSeqsAtMaxContext)}</td>`).join("");
    return `<tr class="${v === current ? "current" : ""}"><td>${v.label} <span style="color:var(--muted)">${v.dtype}</span></td><td>${bytes(base.mem.bytesPerToken)}</td><td>${int(base.cap.maxTokens)}</td>${cells}</tr>`;
  });
  $("cmp").innerHTML = `<thead>${head}</thead><tbody>${rows.join("")}</tbody>`;
}

function renderGqa({ plan }) {
  const { numHeads: q, kvHeads: kv } = plan.model;
  const host = $("gqa");
  const W = Math.max(280, host.clientWidth);
  const cell = Math.max(3, Math.min(14, Math.floor((W - 4) / q) - 2));
  const pitch = cell + 2;
  const rowW = q * pitch - 2;
  const x0 = (W - rowW) / 2;
  const H = 128;
  const yQ = 22;
  const yK = 82;
  const group = q / kv;
  const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${q} query heads share ${kv} KV heads` });

  const kvCell = Math.max(cell, 8);
  for (let k = 0; k < kv; k++) {
    const groupMid = x0 + (k * group + group / 2) * pitch - 1;
    for (let j = 0; j < group; j++) {
      const qx = x0 + (k * group + j) * pitch + cell / 2;
      svg.append(svgEl("line", { x1: qx, y1: yQ + cell, x2: groupMid, y2: yK, style: "stroke:var(--axis)" }));
    }
  }
  for (let i = 0; i < q; i++) {
    svg.append(svgEl("rect", { x: x0 + i * pitch, y: yQ, width: cell, height: cell, rx: 2, style: "fill:var(--held);stroke:var(--axis)" }));
  }
  for (let k = 0; k < kv; k++) {
    const mid = x0 + (k * group + group / 2) * pitch - 1;
    const r = svgEl("rect", { x: mid - kvCell / 2, y: yK, width: kvCell, height: kvCell, rx: 2, style: "fill:var(--seq)" });
    hover(r, () => `<strong>KV head ${k + 1}</strong><div>serves query heads ${k * group + 1}–${(k + 1) * group}</div>`);
    svg.append(r);
  }
  const label = (text, y) => {
    const t = svgEl("text", { x: W / 2, y, "text-anchor": "middle", "font-size": 12, style: "fill:var(--ink-2)" });
    t.textContent = text;
    svg.append(t);
  };
  label(`${q} query heads (not cached)`, yQ - 8);
  label(group === 1 ? `${kv} KV heads, one per query head (MHA)` : `${kv} KV heads, cached · ${group}× smaller than one per query head`, yK + kvCell + 18);
  host.replaceChildren(svg);
}

// Deterministic shuffle, so the pool picture is stable between keystrokes.
function shuffled(n, seed = 7) {
  const a = Array.from({ length: n }, (_, i) => i);
  let s = seed;
  const rand = () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function renderPool({ plan, mem, cap }) {
  const pool = $("pool");
  const bs = plan.cache.blockSize;
  const total = cap.totalBlocks;
  pool.replaceChildren();
  if (total === 0) {
    $("pool-sub").textContent = "Paged block pool: no memory left for KV blocks.";
    $("pool-legend").innerHTML = "";
    $("anatomy").innerHTML = "";
    return;
  }
  const MAX_CELLS = 768;
  const k = Math.ceil(total / MAX_CELLS); // blocks per square
  const cells = Math.ceil(total / k);
  const seqBlocks = blocksFor(plan.cache.maxContext, bs);
  const perSeq = Math.max(1, Math.round(seqBlocks / k));
  const n = Math.min(cap.maxConcurrentSeqsAtMaxContext, Math.floor(cells / perSeq));
  const hasTail = k === 1 && plan.cache.maxContext % bs !== 0;

  const owner = new Array(cells).fill(-1);
  const tail = new Set();
  const order = shuffled(cells);
  for (let s = 0; s < n; s++) {
    for (let j = 0; j < perSeq; j++) {
      const c = order[s * perSeq + j];
      owner[c] = s;
      if (hasTail && j === perSeq - 1) tail.add(c);
    }
  }
  const frag = document.createDocumentFragment();
  owner.forEach((s, c) => {
    const d = document.createElement("div");
    d.dataset.s = s;
    if (s < 0) d.className = "free";
    else if (tail.has(c)) d.className = "tail";
    frag.append(d);
  });
  pool.append(frag);

  $("pool-sub").textContent =
    `Paged block pool: ${int(total)} blocks of ${bs} tokens${k > 1 ? ` (each square = ${k} blocks)` : ""}, ` +
    `holding ${int(n)} full-length sequences of ${int(seqBlocks)} blocks each. vLLM hands out whichever blocks are free, ` +
    `so a sequence's blocks are scattered across the pool. Hover a block to highlight its sequence.`;
  $("pool-legend").innerHTML =
    `<span><i style="background:var(--seq)"></i>in use</span><span><i style="box-shadow:inset 0 0 0 1px var(--axis)"></i>free</span>` +
    (hasTail ? `<span><i style="background:var(--tail)"></i>last block, partly empty</span>` : "");
  $("anatomy").innerHTML =
    `1 block = <b>${bs}</b> tokens × 2 (K and V) × <b>${plan.model.layers}</b> layers × <b>${plan.model.kvHeads}</b> KV heads × ` +
    `<b>${plan.model.headDim}</b> dims × <b>${DTYPE_BYTES[plan.cache.dtype]}</b> bytes (${plan.cache.dtype}) = <b>${bytes(mem.bytesPerBlock)}</b>`;
}

$("pool").addEventListener("mouseover", (e) => {
  const s = Number(e.target.dataset?.s ?? -1);
  const pool = $("pool");
  pool.querySelectorAll(".hl").forEach((d) => d.classList.remove("hl"));
  if (s < 0 || Number.isNaN(s)) {
    pool.classList.remove("focus");
    hideTip();
    return;
  }
  const mine = pool.querySelectorAll(`[data-s="${s}"]`);
  mine.forEach((d) => d.classList.add("hl"));
  pool.classList.add("focus");
  showTip(e, `<strong>Sequence ${s + 1}</strong><div>${mine.length} squares, scattered across the pool</div>`);
});
$("pool").addEventListener("mousemove", moveTip);
$("pool").addEventListener("mouseleave", () => {
  $("pool").classList.remove("focus");
  hideTip();
});

// ---------- state: URL hash so a plan can be shared ----------

function saveToHash() {
  const json = JSON.stringify({ s: srcEl.value, g: gpuEl.value });
  const b64 = btoa(String.fromCharCode(...new TextEncoder().encode(json))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  history.replaceState(null, "", `#${b64}`);
}

function loadFromHash() {
  try {
    const b64 = location.hash.slice(1).replace(/-/g, "+").replace(/_/g, "/");
    if (!b64) return false;
    const { s, g } = JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))));
    srcEl.value = s;
    gpuEl.value = g in GPU_PRESETS ? g : "";
    return true;
  } catch {
    return false;
  }
}

// ---------- wiring ----------

presetEl.innerHTML = `<option value="">Load preset…</option>` + Object.keys(MODEL_PRESETS).map((k) => `<option>${esc(k)}</option>`).join("");
gpuEl.innerHTML = `<option value="">from file</option>` + Object.keys(GPU_PRESETS).map((k) => `<option>${esc(k)}</option>`).join("");

presetEl.addEventListener("change", () => {
  const name = presetEl.value;
  if (!name) return;
  const block = `model ${name} { }`;
  srcEl.value = /model\s+[^{]*\{[^}]*\}/.test(srcEl.value) ? srcEl.value.replace(/model\s+[^{]*\{[^}]*\}/, block) : `${block}\n${srcEl.value}`;
  presetEl.value = "";
  update();
});
gpuEl.addEventListener("change", update);

let timer;
srcEl.addEventListener("input", () => {
  clearTimeout(timer);
  timer = setTimeout(update, 120);
});
srcEl.addEventListener("keydown", (e) => {
  if (e.key !== "Tab") return;
  e.preventDefault();
  srcEl.setRangeText("  ", srcEl.selectionStart, srcEl.selectionEnd, "end");
});

async function copy(text, button) {
  try {
    await navigator.clipboard.writeText(text);
    const old = button.textContent;
    button.textContent = "Copied";
    setTimeout(() => (button.textContent = old), 1200);
  } catch {
    button.textContent = "Copy failed";
  }
}
$("copy").addEventListener("click", (e) => copy($("vllm").textContent, e.currentTarget));
$("share").addEventListener("click", (e) => copy(location.href, e.currentTarget));

let resizeTimer;
addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => last && (renderCapacityChart(last), renderGqa(last)), 100);
});

if (!loadFromHash()) srcEl.value = DEFAULT_SOURCE;
update();
