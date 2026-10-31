// Вкладка «Диаграмма (μ, T)»: фазовые диаграммы минимума Ω_T(M, b) (формула (A)) в плоскости (μ, T) при
// фиксированном L (посчитаны заранее, notebooks/92_muT_maps.py), расчёт точки в фоновом потоке (worker_inhT.js).
const $$ = (id) => document.getElementById(id);
const $ = (id) => $$('z-' + id);
const root = document.querySelector('.viz-root');
const LN10 = Math.log(10);
const LS = [0.1, 0.3, 0.5, 1, 3, 5];
const PHASE_COLORS = { V: '#2166ac', I: '#67a9cf', B: '#ef8a62', T: '#bdbcb6' };
const ORANGE_LIGHT = ['#fde6da', '#fbccb3', '#f8ad87', '#f38d5c', '#eb6834', '#cc5422', '#a84417', '#83340f'];

function tokens() {
  const cs = getComputedStyle(root);
  const v = (n) => cs.getPropertyValue(n).trim();
  const dark = cs.colorScheme === 'dark' || v('--surface-1').toLowerCase() === '#1a1a19';
  const seq = v('--seq').split(',').map((s) => s.trim());
  const orange = dark ? [...ORANGE_LIGHT].reverse() : ORANGE_LIGHT;
  return { dark, surface: v('--surface-1'), ink: v('--text-primary'), ink2: v('--text-secondary'), muted: v('--text-muted'),
           grid: v('--grid'), axis: v('--axis'), seq: seq.map((c, i) => [i / (seq.length - 1), c]),
           seq2: orange.map((c, i) => [i / (orange.length - 1), c]) };
}
function baseLayout(t, extra = {}) {
  const ax = { gridcolor: t.grid, linecolor: t.axis, zerolinecolor: t.axis, tickfont: { color: t.muted } };
  return {
    paper_bgcolor: t.surface, plot_bgcolor: t.surface,
    font: { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', size: 12, color: t.ink2 },
    margin: { l: 56, r: 12, t: 8, b: 44 }, hoverlabel: { bgcolor: t.surface, bordercolor: t.axis, font: { color: t.ink } },
    xaxis: { ...ax, ...(extra.xaxis || {}) }, yaxis: { ...ax, ...(extra.yaxis || {}) },
    ...Object.fromEntries(Object.entries(extra).filter(([k]) => k !== 'xaxis' && k !== 'yaxis')),
  };
}
const CFG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'],
              toImageButtonOptions: { format: 'png', scale: 2, filename: 'njl3_muT' } };

const state = { L: 1, logT: true, bnd: true, mu: 4.0, T: 0.01 };
function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  for (const k of Object.keys(state)) if (p.has('z_' + k)) {
    const d = state[k], v = p.get('z_' + k);
    state[k] = typeof d === 'boolean' ? v === '1' : Number(v);
  }
}
function writeHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  for (const [k, v] of Object.entries(state)) p.set('z_' + k, typeof v === 'boolean' ? (v ? '1' : '0') : String(v));
  history.replaceState(null, '', '#' + p.toString());
}

const cache = {};
let D = null;
async function load() {
  const u = `data/muT/muT_L${state.L}_phi0_g-1.json`;
  if (!cache[u]) { const r = await fetch(u); if (!r.ok) throw new Error('нет карты'); cache[u] = await r.json(); }
  return cache[u];
}
const cls = (p) => (p === 'V' ? 'V' : p[0] === 'I' ? 'I' : p[0] === 'B' ? 'B' : 'T');
const CODE = { V: 0, I: 1, B: 2, T: 3 };
const fmtM = (lnM) => (lnM === null ? '0' : lnM > -700 ? Math.exp(lnM).toPrecision(5) : `e^${lnM.toFixed(1)}`);

function boundaries() {
  // отрезки между соседними клетками с разной фазой (x = mu, y = T); D.phase[i = T][j = mu]
  const { mu, T, phase } = D, x = [], y = [];
  const mid = (a, i) => 0.5 * (a[i] + a[i + 1]);
  const ex = (j) => (j <= 0 ? mu[0] : j >= mu.length ? mu.at(-1) : mid(mu, j - 1));
  const ey = (i) => (i <= 0 ? T[0] : i >= T.length ? T.at(-1) : Math.sqrt(T[i - 1] * T[i]));
  for (let i = 0; i < T.length; i++) for (let j = 0; j < mu.length; j++) {
    const p = cls(phase[i][j]);
    if (i + 1 < T.length && p !== cls(phase[i + 1][j])) { const yy = Math.sqrt(T[i] * T[i + 1]); x.push(ex(j), ex(j + 1), null); y.push(yy, yy, null); }
    if (j + 1 < mu.length && p !== cls(phase[i][j + 1])) { const xx = mid(mu, j); x.push(xx, xx, null); y.push(ey(i), ey(i + 1), null); }
  }
  return { x, y };
}

function draw() {
  if (!D) return;
  const t = tokens();
  const { mu, T, phase, lnM, b } = D;
  const cd = phase.map((r, i) => r.map((p, j) =>
    `${p}<br>M* = ${fmtM(lnM[i][j])}<br>b* = ${b[i][j] === null ? '—' : b[i][j].toFixed(4)}${D.interp[i][j] ? '<br>(интерполяция)' : ''}`));
  const hov = 'μ = %{x:.3f}<br>T = %{y:.4g}<br>фаза %{customdata}<extra></extra>';
  const bnd = state.bnd ? boundaries() : { x: [], y: [] };
  const over = [
    { type: 'scatter', mode: 'lines', x: bnd.x, y: bnd.y, line: { color: t.ink, width: 1 }, hoverinfo: 'skip', showlegend: false },
    { type: 'scatter', mode: 'markers', x: [state.mu], y: [state.T], hoverinfo: 'skip', showlegend: false,
      marker: { symbol: 'circle-open', size: 13, color: t.ink, line: { width: 2.5, color: t.ink } } },
  ];
  const yax = { title: { text: 'T' }, type: state.logT ? 'log' : 'linear',
                range: state.logT ? [Math.log10(T[0]), Math.log10(T.at(-1))] : [0, T.at(-1)], exponentformat: 'power' };
  const lay = () => baseLayout(t, { xaxis: { title: { text: 'μ' }, range: [mu[0], mu.at(-1)] }, yaxis: yax,
                                    margin: { l: 56, r: 8, t: 6, b: 44 } });
  const z = phase.map((r) => r.map((p) => CODE[cls(p)]));
  const P = PHASE_COLORS;
  Plotly.react($('map-phase'), [
    { type: 'heatmap', x: mu, y: T, z, customdata: cd, hovertemplate: hov, zmin: -0.5, zmax: 3.5, zsmooth: false,
      colorscale: [[0, P.V], [0.25, P.V], [0.25, P.I], [0.5, P.I], [0.5, P.B], [0.75, P.B], [0.75, P.T], [1, P.T]],
      colorbar: { thickness: 12, outlinewidth: 0, tickvals: [0, 1, 2, 3], ticktext: ['V', 'I', 'B', 'T'], tickfont: { color: t.muted } } },
    ...over], lay(), CFG);
  const logM = lnM.map((r) => r.map((v) => (v === null ? null : Math.max(v / LN10, -4))));
  const zero = phase.map((r) => r.map((p) => (p === 'T' ? 1 : null)));
  const zeroLayer = { type: 'heatmap', x: mu, y: T, z: zero, colorscale: [[0, t.muted], [1, t.muted]], opacity: 0.35, showscale: false,
                      zsmooth: false, hovertemplate: 'μ = %{x:.3f}<br>T = %{y:.4g}<br>M = 0<extra></extra>' };
  Plotly.react($('map-M'), [zeroLayer,
    { type: 'heatmap', x: mu, y: T, z: logM, customdata: cd, hovertemplate: hov, colorscale: t.seq, zmin: -4, zmax: 1, zsmooth: false,
      colorbar: { title: { text: 'M*' }, thickness: 12, outlinewidth: 0, tickvals: [-4, -3, -2, -1, 0, 1],
                  ticktext: ['≤10⁻⁴', '10⁻³', '10⁻²', '10⁻¹', '1', '10'], tickfont: { color: t.muted } } },
    ...over], lay(), CFG);
  Plotly.react($('map-b'), [zeroLayer,
    { type: 'heatmap', x: mu, y: T, z: b.map((r, i) => r.map((v, j) => (phase[i][j][0] === 'B' ? v : null))), customdata: cd,
      hovertemplate: hov, colorscale: t.seq2, zmin: 0, zsmooth: false,
      colorbar: { title: { text: 'b*' }, thickness: 12, outlinewidth: 0, tickfont: { color: t.muted } } },
    ...over], lay(), CFG);
}

// ---------------------------------------------------------------------------
// Срезы через точку (из данных карты)
// ---------------------------------------------------------------------------
const nearest = (arr, v, log) => arr.reduce((k, x, i) => (Math.abs(log ? Math.log(x / v) : x - v) < Math.abs(log ? Math.log(arr[k] / v) : arr[k] - v) ? i : k), 0);

function phaseSegments(xs, ys, phases, name) {
  // по одной линии на каждую фазу: разрывы там, где фаза другая (стык соединяется с соседней точкой)
  const out = [];
  for (const c of ['V', 'I', 'B', 'T']) {
    const y = ys.map((v, k) => (cls(phases[k]) === c || (k > 0 && cls(phases[k - 1]) === c) ? v : null));
    if (y.some((v, k) => v !== null && cls(phases[k]) === c))
      out.push({ type: 'scatter', mode: 'lines', x: xs, y, name: c, line: { color: PHASE_COLORS[c], width: 2.2 }, connectgaps: false,
                 hovertemplate: `${name} = %{x:.4g}<br>%{y:.5g}<extra>${c}</extra>` });
  }
  return out;
}

function drawSlices() {
  if (!D) return;
  const t = tokens();
  const { mu, T, phase, lnM, b } = D;
  const j = nearest(mu, state.mu, false), i = nearest(T, state.T, true);
  const Mval = (v) => (v === null ? 0 : Math.exp(v));
  // по T при mu = mu[j]
  const phT = T.map((_, k) => phase[k][j]), MT = T.map((_, k) => Mval(lnM[k][j]));
  const bT = T.map((_, k) => (phase[k][j][0] === 'B' ? b[k][j] : 0));
  // по mu при T = T[i]
  const phM = phase[i], Mmu = mu.map((_, k) => Mval(lnM[i][k]));
  const bmu = mu.map((_, k) => (phase[i][k][0] === 'B' ? b[i][k] : 0));
  const xT = { title: { text: 'T' }, type: state.logT ? 'log' : 'linear', exponentformat: 'power',
               range: state.logT ? [Math.log10(T[0]), Math.log10(T.at(-1))] : [0, T.at(-1)] };
  const xmu = { title: { text: 'μ' }, range: [mu[0], mu.at(-1)] };
  const mark = (x) => [{ type: 'line', yref: 'paper', y0: 0, y1: 1, x0: x, x1: x, line: { color: t.muted, width: 1, dash: 'dot' } }];
  const lay = (xa, yt, x0, ylog) => baseLayout(t, { xaxis: xa, showlegend: true, shapes: mark(x0),
    yaxis: { title: { text: yt }, rangemode: 'tozero', ...(ylog ? { type: 'log', exponentformat: 'power' } : {}) },
    legend: { orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', font: { size: 10, color: t.ink2 } }, margin: { l: 56, r: 10, t: 8, b: 40 } });
  $('sT-title').innerHTML = `M*(T) при μ = ${mu[j].toFixed(3)}`;
  $('smu-title').innerHTML = `M*(μ) при T = ${T[i].toPrecision(3)}`;
  $('sT-b-title').innerHTML = `b*(T) при μ = ${mu[j].toFixed(3)}`;
  $('smu-b-title').innerHTML = `b*(μ) при T = ${T[i].toPrecision(3)}`;
  Plotly.react($('sT-M'), phaseSegments(T, MT, phT, 'T'), lay(xT, 'M*', T[i]), CFG);
  Plotly.react($('smu-M'), phaseSegments(mu, Mmu, phM, 'μ'), lay(xmu, 'M*', mu[j]), CFG);
  Plotly.react($('sT-b'), phaseSegments(T, bT, phT, 'T'), lay(xT, 'b*', T[i]), CFG);
  Plotly.react($('smu-b'), phaseSegments(mu, bmu, phM, 'μ'), lay(xmu, 'b*', mu[j]), CFG);
}

// ---------------------------------------------------------------------------
// Точка: расчёт в фоновом потоке вкладки «Неоднородный, T > 0»
// ---------------------------------------------------------------------------
const worker = new Worker(new URL('./worker_inhT.js', import.meta.url), { type: 'module' });
let job = 0;
const pending = new Map();
worker.onmessage = ({ data }) => {
  if (data.type === 'error') { $('status').textContent = 'Ошибка расчёта: ' + data.message; return; }
  const cb = pending.get(data.job); if (cb) cb(data);
};
function runPoint(mu, T) {
  state.mu = mu; state.T = T; writeHash(); draw(); drawSlices();
  $('pt-mu').value = mu; $('pt-T').value = T;
  $('status').textContent = `Расчёт точки μ = ${mu}, T = ${T}, L = ${state.L}…`;
  const t0 = performance.now(), id = ++job;
  pending.set(id, ({ result: r }) => {
    $('status').textContent = `Точка μ = ${mu}, T = ${T}, L = ${state.L}: ${((performance.now() - t0) / 1000).toFixed(1)} с.`;
    $('point-table').innerHTML = `
      <tr><th>Фаза</th><td><span class="phase-badge">${r.phase}</span>${r.phase === 'T' ? ' — M = 0' : ''}</td></tr>
      <tr><th>L, φ, g</th><td>${r.L}, ${r.phi}, ${r.g}</td></tr>
      <tr><th>M*</th><td>${r.M_star > 0 ? r.M_star.toPrecision(8) : '0'}</td></tr>
      <tr><th>b*</th><td>${Number.isFinite(r.b_star) ? r.b_star.toFixed(8) : '—'}</td></tr>
      <tr><th>Ω*<sub>T</sub> − Ω<sub>T</sub>(0)</th><td>${r.omega_star.toExponential(6)}</td></tr>
      <tr><th>Кандидат b = 0</th><td>${r.b0.phase}: M = ${r.b0.M.toPrecision(8)}, ΔΩ = ${r.b0.omega.toExponential(4)}</td></tr>
      <tr><th>При T = 0</th><td>${r.b0.T0_phase}, M* = ${r.b0.T0_M > 0 ? r.b0.T0_M.toPrecision(6) : '0'}</td></tr>`;
  });
  worker.postMessage({ type: 'point', job: id, mu, L: state.L, T, g: -1, phi: 0 });
}

function syncSeg() {
  document.querySelectorAll('[data-zl]').forEach((b) => b.classList.toggle('on', Number(b.dataset.zl) === state.L));
  document.querySelectorAll('[data-zscale]').forEach((b) => b.classList.toggle('on', (b.dataset.zscale === 'log') === state.logT));
}
async function reload() {
  syncSeg(); writeHash();
  try {
    D = await load();
    $('status').textContent = `L = ${state.L}, φ = 0, g = −1. Карта ${D.mu.length} × ${D.T.length} (μ ∈ [${D.mu[0]}, ${D.mu.at(-1)}], ` +
      `T ∈ [${D.T[0]}, ${D.T.at(-1)}] логарифмически); у границ фаз — уточнение, внутри фаз — интерполяция.`;
  } catch {
    D = null;
    $('status').textContent = `Карта для L = ${state.L} не найдена.`;
  }
  draw();
}

let started = false;
export async function initMuT() {
  if (started) { window.dispatchEvent(new Event('resize')); return; }
  started = true;
  readHash();
  $('show-bnd').checked = state.bnd;
  await reload();
  runPoint(state.mu, state.T);
  document.querySelectorAll('[data-zl]').forEach((b) => b.addEventListener('click', async () => {
    state.L = Number(b.dataset.zl); await reload(); runPoint(state.mu, state.T);
  }));
  document.querySelectorAll('[data-zscale]').forEach((b) => b.addEventListener('click', () => {
    state.logT = b.dataset.zscale === 'log'; syncSeg(); writeHash(); draw(); drawSlices();
  }));
  $('show-bnd').addEventListener('change', () => { state.bnd = $('show-bnd').checked; writeHash(); draw(); });
  for (const id of ['map-phase', 'map-M', 'map-b']) $(id).on('plotly_click', (ev) => {
    const p = ev.points[0];
    if (p && p.x != null && p.y != null) runPoint(+p.x.toFixed(4), +Number(p.y).toPrecision(4));
  });
  $('point-form').addEventListener('submit', (e) => {
    e.preventDefault();
    runPoint(Number($('pt-mu').value), Math.max(1e-4, Number($('pt-T').value)));
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { draw(); drawSlices(); });
}
export function redrawMuT() { if (started) { draw(); drawSlices(); } }
export { LS };
