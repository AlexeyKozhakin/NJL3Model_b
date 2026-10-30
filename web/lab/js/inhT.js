// Вкладка «Неоднородный, T > 0»: карты M*, b* минимума Ω_T(M, b) при температуре T (формула (A); посчитаны заранее,
// notebooks/90_inh_maps.py --thermal), расчёт точки в фоновом потоке (worker_inhT.js). Копия inh.js.
import { thresholds } from './njl3inh.js';

const $$ = (id) => document.getElementById(id);
const $ = (id) => $$('y-' + id);
const root = document.querySelector('.viz-root');
const LN10 = Math.log(10);

// Оранжевая последовательная шкала (вторая величина другого смысла — следующий оттенок палитры)
const ORANGE_LIGHT = ['#fde6da', '#fbccb3', '#f8ad87', '#f38d5c', '#eb6834', '#cc5422', '#a84417', '#83340f'];
const ORANGE_DARK = [...ORANGE_LIGHT].reverse();

function tokens() {
  const cs = getComputedStyle(root);
  const v = (n) => cs.getPropertyValue(n).trim();
  const dark = cs.colorScheme === 'dark' || v('--surface-1').toLowerCase() === '#1a1a19';
  const seq = v('--seq').split(',').map((s) => s.trim());
  const orange = dark ? ORANGE_DARK : ORANGE_LIGHT;
  return {
    dark, surface: v('--surface-1'), ink: v('--text-primary'), ink2: v('--text-secondary'), muted: v('--text-muted'),
    grid: v('--grid'), axis: v('--axis'), s1: v('--series-1'), s2: v('--series-2'),
    seq: seq.map((c, i) => [i / (seq.length - 1), c]),
    seq2: orange.map((c, i) => [i / (orange.length - 1), c]),
    div: [[0, dark ? '#3987e5' : '#2a78d6'], [0.5, dark ? '#383835' : '#f0efec'], [1, dark ? '#e66767' : '#e34948']],
  };
}

function baseLayout(t, extra = {}) {
  const ax = { gridcolor: t.grid, linecolor: t.axis, zerolinecolor: t.axis, tickfont: { color: t.muted } };
  return {
    paper_bgcolor: t.surface, plot_bgcolor: t.surface,
    font: { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', size: 12, color: t.ink2 },
    margin: { l: 56, r: 12, t: 8, b: 44 },
    hoverlabel: { bgcolor: t.surface, bordercolor: t.axis, font: { color: t.ink } },
    xaxis: { ...ax, ...(extra.xaxis || {}) }, yaxis: { ...ax, ...(extra.yaxis || {}) },
    ...Object.fromEntries(Object.entries(extra).filter(([k]) => k !== 'xaxis' && k !== 'yaxis')),
  };
}
const CFG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'],
              toImageButtonOptions: { format: 'png', scale: 2, filename: 'njl3_inhT' } };

const state = { xphi: 0, xg: -1, bnd: true, xmu: 4.0, xL: 2.0, T: 0.001, zoom: 'near', region: 'full',
                vmu0: 0.05, vmu1: 7, vL0: 0.1, vL1: 10, cmp: '', cmpNorm: false, cmpLogM: false };
const mapsCache = {};
const mapUrl = () => `data/inhT/maps_T${state.T}_phi${state.xphi}_g${state.xg}_${state.region}.json`;
async function loadRegion() {
  const u = mapUrl();
  if (!mapsCache[u]) {
    const r = await fetch(u);
    if (!r.ok) throw new Error('нет карты');
    mapsCache[u] = await r.json();
  }
  return mapsCache[u];
}
const fullView = () => { [state.vmu0, state.vmu1, state.vL0, state.vL1] = [MAPS.mu[0], MAPS.mu.at(-1), MAPS.L[0], MAPS.L.at(-1)]; };
const mapsStatus = () => `φ = ${state.xphi}, g = ${state.xg}. Карты: ${MAPS.mu.length} × ${MAPS.L.length} точек (μ ∈ [${MAPS.mu[0]}, ${MAPS.mu.at(-1)}], L ∈ [${MAPS.L[0]}, ${MAPS.L.at(-1)}]), T = ${fmtT(state.T)} (минимум Ω_T по формуле (A))${MAPS.refine ? `; у границ фаз — уточнение в ${2 ** MAPS.refine} раза, внутри фаз — интерполяция` : ''}.`;
function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  for (const k of Object.keys(state)) if (p.has('y_' + k)) {
    const d = state[k];
    const v = p.get('y_' + k);
    state[k] = typeof d === 'boolean' ? v === '1' : typeof d === 'string' ? v : Number(v);
  }
}
function writeHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  for (const [k, v] of Object.entries(state)) p.set('y_' + k, typeof v === 'boolean' ? (v ? '1' : '0') : String(v));
  history.replaceState(null, '', '#' + p.toString());
}

// ---------------------------------------------------------------------------
// Карты
// ---------------------------------------------------------------------------
let MAPS = null;
const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
const pow10 = (k) => (k === 0 ? '1' : '10' + String(k).split('').map((c) => SUP[c]).join(''));
const fmtT = (T) => (T > 0 ? T.toExponential(0).replace(/e([+-]\d+)/, (_, e) => ' · 10' + String(+e).split('').map((c) => SUP[c]).join('')).replace(/^1 · /, '') : '0');
const fmtM = (lnM) => (!Number.isFinite(lnM) ? '0' : lnM > -700 ? Math.exp(lnM).toPrecision(6) : `e^${lnM.toFixed(1)}`);

function boundaries() {
  // отрезки между соседними клетками с разной фазой (x = L, y = mu)
  const { L, mu, phase } = MAPS, x = [], y = [];
  const hx = (i) => 0.5 * (L[i] + L[i + 1]), hy = (j) => 0.5 * (mu[j] + mu[j + 1]);
  const ex = (i) => (i <= 0 ? L[0] : i >= L.length ? L[L.length - 1] : hx(i - 1));
  const ey = (j) => (j <= 0 ? mu[0] : j >= mu.length ? mu[mu.length - 1] : hy(j - 1));
  for (let i = 0; i < L.length; i++) for (let j = 0; j < mu.length; j++) {
    if (i + 1 < L.length && phase[i][j] !== phase[i + 1][j]) { x.push(hx(i), hx(i), null); y.push(ey(j), ey(j + 1), null); }
    if (j + 1 < mu.length && phase[i][j] !== phase[i][j + 1]) { x.push(ex(i), ex(i + 1), null); y.push(hy(j), hy(j), null); }
  }
  return { x, y };
}

function drawMaps() {
  if (!MAPS) return;
  const t = tokens();
  const { L, mu, lnM, b, phase, omega } = MAPS;
  const interp = MAPS.interp || null;
  const T = (A) => mu.map((_, j) => L.map((_, i) => A[i][j]));       // [i=L][j=mu] -> z[mu][L]
  const logM = T(lnM.map((r) => r.map((v) => (v === null ? null : Math.max(v / LN10, -12)))));
  // шкала M*: от наименьшей ненулевой щели (при пороге T она ~2T) до 10
  const lmin = Math.min(...logM.flat().filter((v) => v !== null));
  const zmin = Math.max(-8, Math.min(-1, Math.floor(lmin)));
  const step = -zmin > 4 ? 2 : 1;
  const ticks = Array.from({ length: Math.floor(-zmin / step) + 1 }, (_, k) => zmin + k * step);
  // M = 0 (щель не переживает T): отдельный нейтральный слой под картой
  const zero = T(phase.map((r) => r.map((p) => (p === 'T' ? 1 : null))));
  const zeroLayer = { type: 'heatmap', x: L, y: mu, z: zero, customdata: null, colorscale: [[0, t.muted], [1, t.muted]], opacity: 0.35,
                      showscale: false, zsmooth: false, hovertemplate: 'L = %{x:.3f}<br>μ = %{y:.3f}<br>M = 0 (щель не переживает T)<extra></extra>' };
  const cd = T(phase.map((r, i) => r.map((p, j) =>
    `${p}<br>M* = ${fmtM(lnM[i][j])}<br>b* = ${b[i][j] === null ? '—' : b[i][j].toFixed(5)}<br>Ω* = ${omega[i][j] === null ? '—' : omega[i][j].toExponential(3)}${interp && interp[i][j] ? '<br>(интерполяция)' : ''}`)));
  const hov = 'L = %{x:.3f}<br>μ = %{y:.3f}<br>фаза %{customdata}<extra></extra>';
  const bnd = state.bnd ? boundaries() : { x: [], y: [] };
  const over = [
    { type: 'scatter', mode: 'lines', x: bnd.x, y: bnd.y, line: { color: t.ink, width: 1 }, hoverinfo: 'skip', showlegend: false },
  ];
  over.push(...cmpTrace());
  over.push({ type: 'scatter', mode: 'markers', x: [state.xL], y: [state.xmu], hoverinfo: 'skip', showlegend: false,
              marker: { symbol: 'circle-open', size: 13, color: t.ink, line: { width: 2.5, color: t.ink } } });
  // область просмотра (кнопки-пресеты и выделение мышью) — общая для обеих карт
  const lay = (title) => baseLayout(t, { xaxis: { title: { text: 'L' }, range: [state.vL0, state.vL1] },
    yaxis: { title: { text: 'μ' }, range: [state.vmu0, state.vmu1] }, margin: { l: 52, r: 8, t: 6, b: 44 } });
  Plotly.react('y-map-M', [
    zeroLayer,
    { type: 'heatmap', x: L, y: mu, z: logM, customdata: cd, hovertemplate: hov, colorscale: t.seq, zmin, zmax: 1, zsmooth: false,
      colorbar: { title: { text: 'M*' }, thickness: 12, outlinewidth: 0, tickvals: ticks,
                  ticktext: ticks.map((k, i) => (i === 0 && zmin === -8 ? '≤' : '') + pow10(k)), tickfont: { color: t.muted } } },
    ...over], lay(), CFG);
  Plotly.react('y-map-b', [
    zeroLayer,
    { type: 'heatmap', x: L, y: mu, z: T(b), customdata: cd, hovertemplate: hov, colorscale: t.seq2, zmin: 0, zsmooth: false,
      colorbar: { title: { text: 'b*' }, thickness: 12, outlinewidth: 0, tickfont: { color: t.muted } } },
    ...over], lay(), CFG);
}

// ---------------------------------------------------------------------------
// Точка
// ---------------------------------------------------------------------------
const worker = new Worker(new URL('./worker_inhT.js', import.meta.url), { type: 'module' });
let job = 0, lastPoint = null;
const pending = new Map();
worker.onmessage = ({ data }) => {
  if (data.type === 'error') { $('status').textContent = 'Ошибка расчёта: ' + data.message; return; }
  const cb = pending.get(data.job); if (cb) cb(data);
};
worker.onerror = (e) => { $('status').textContent = 'Ошибка фонового потока: ' + (e.message || e.filename || 'неизвестная'); };
function ask(msg, cb) { const id = ++job; pending.set(id, cb); worker.postMessage({ g: state.xg, phi: state.xphi, T: state.T, ...msg, job: id }); return id; }

function linspace(a, b, n) { return Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1)); }

function runPoint(mu, L) {
  state.xmu = mu; state.xL = L; writeHash();
  $('pt-mu').value = mu; $('pt-L').value = L;
  drawMaps();
  $("status").textContent = `Расчёт точки μ = ${mu}, L = ${L}: глобальный минимум по всем каналам, несколько секунд…`;
  const t0 = performance.now();
  ask({ type: 'point', mu, L, T: state.T }, ({ result: r }) => {
    lastPoint = r;
    $('status').textContent = `Точка μ = ${mu}, L = ${L}: ${((performance.now() - t0) / 1000).toFixed(1)} с.`;
    $('point-table').innerHTML = `
      <tr><th>Фаза</th><td><span class="phase-badge">${r.phase}</span>${r.phase === 'T' ? ' — M = 0: ни одна щель не переживает T' : ''}</td></tr>
      <tr><th>φ, g, T</th><td>${r.phi}, ${r.g}, ${fmtT(r.T || 0)}</td></tr>
      <tr><th>M*</th><td>${fmtM(r.ln_M_star)} (ln M* = ${Number.isFinite(r.ln_M_star) ? r.ln_M_star.toFixed(5) : '−∞'})</td></tr>
      <tr><th>b*</th><td>${Number.isFinite(r.b_star) ? r.b_star.toFixed(8) : '—'}</td></tr>
      <tr><th>Ω*<sub>T</sub> − Ω<sub>T</sub>(0)</th><td>${r.omega_star.toExponential(6)}</td></tr>
      <tr><th>Кандидат b = 0</th><td>${r.b0.phase}: M = ${r.b0.M.toPrecision(8)}, ΔΩ = ${r.b0.omega.toExponential(4)}</td></tr>
      <tr><th>При T = 0</th><td>${r.b0.T0_phase}, M* = ${r.b0.T0_M > 0 ? r.b0.T0_M.toPrecision(6) : '0'}</td></tr>`;
    $('channels').innerHTML = '<tr><th>n</th><th>S<sub>n</sub></th><th>M* при T = 0</th><th>M* при T</th><th>b* − S<sub>n</sub></th><th>Ω*<sub>T</sub></th></tr>' +
      r.channels.map((c) => `<tr class="${r.phase === 'B' + c.n ? 'win' : ''}"><td>${c.n}</td><td>${c.S.toFixed(5)}</td>
        <td>${c.M0 > 0 ? c.M0.toPrecision(4) : '0'}</td><td>${c.M > 0 ? c.M.toPrecision(5) : (c.done ? '0 (не пережила T)' : '— (не уточнялся)')}</td>
        <td>${c.M > 0 ? (c.b - c.S).toExponential(2) : '—'}</td><td>${c.M > 0 ? c.omega.toExponential(3) : '0'}</td></tr>`).join('');
    drawOmegaMap(r);
    drawSlices(r);
  });
}

// ---------------------------------------------------------------------------
// Срезы через минимум: Ω(M*, b) и Ω(M, b*)
// ---------------------------------------------------------------------------
function drawSlices(r) {
  const t = tokens();
  if (!(r.M_star > 0) || !Number.isFinite(r.b_star)) {
    // M = 0: b не определено — срезов нет
    for (const id of ['chart-slice-b', 'chart-slice-M']) Plotly.purge($(id));
    $('slice-b-title').textContent = 'M* = 0: b* не определено, срезов нет';
    $('slice-M-title').textContent = '';
    return;
  }
  const Ms = r.M_star, bs = r.b_star, near = state.zoom === 'near';
  const wB = Math.min(8 * Ms, r.mu + 1);
  const [b0, b1] = near ? [Math.max(0, bs - wB), bs + wB] : [0, Math.max(r.mu + 1, bs + 0.5)];
  const [M0, M1] = near ? [0, 3 * Ms] : [0, Math.max(2, 1.5 * Ms)];
  const N = 241;
  const bArr = linspace(b0, b1, N), MArr = linspace(M0, M1, N);
  $('slice-b-title').innerHTML = `Ω(M*, b) − Ω(0, 0), M* = ${fmtM(r.ln_M_star)}`;
  $('slice-M-title').innerHTML = `Ω(M, b*) − Ω(0, 0), b* = ${bs.toFixed(6)}`;
  const thr = thresholds(r.mu, r.L, r.phi).filter((q) => q.S >= b0 && q.S <= b1);
  const star = (x) => ({ type: 'scatter', mode: 'markers', x: [x], y: [r.omega_star], name: 'минимум', hoverinfo: 'skip',
                         marker: { symbol: 'star', size: 13, color: t.ink, line: { color: t.surface, width: 1.5 } } });
  const lay = (xt, shapes) => baseLayout(t, { xaxis: { title: { text: xt } }, showlegend: false,
    yaxis: { title: { text: 'ΔΩ' }, zeroline: true, zerolinecolor: t.axis, zerolinewidth: 1.5, exponentformat: 'power' },
    shapes, margin: { l: 64, r: 8, t: 8, b: 40 } });
  ask({ type: 'slice', mu: r.mu, L: r.L, M: bArr.map(() => Ms), b: bArr }, ({ om }) => {
    Plotly.react('y-chart-slice-b', [
      { type: 'scatter', mode: 'lines', x: bArr, y: Array.from(om), line: { color: t.s1, width: 2 },
        hovertemplate: 'b = %{x:.6f}<br>ΔΩ = %{y:.5e}<extra></extra>' }, star(bs)],
      lay('b', thr.map((q) => ({ type: 'line', yref: 'paper', y0: 0, y1: 1, x0: q.S, x1: q.S,
                                 line: { color: t.muted, width: 1, dash: 'dot' } }))), CFG);
  });
  ask({ type: 'slice', mu: r.mu, L: r.L, M: MArr, b: MArr.map(() => bs) }, ({ om }) => {
    Plotly.react('y-chart-slice-M', [
      { type: 'scatter', mode: 'lines', x: MArr, y: Array.from(om), line: { color: t.s1, width: 2 },
        hovertemplate: 'M = %{x:.6f}<br>ΔΩ = %{y:.5e}<extra></extra>' }, star(Ms)],
      lay('M', []), CFG);
  });
}

function drawOmegaMap(r) {
  const t = tokens();
  const Mmax = Math.max(1.2, 1.4 * (r.M_star || 0));
  const bmax = Math.max(r.mu + 0.6, (r.b_star || 0) + 0.5);
  const M = linspace(Mmax / 60, Mmax, 60), b = linspace(0, bmax, 60);
  const z = b.map(() => new Array(M.length).fill(null));
  const draw = () => {
    const flat = z.flat().filter((v) => v !== null).map(Math.abs).sort((a, c) => a - c);
    const lim = flat.length ? flat[Math.floor(0.6 * flat.length)] || 1e-12 : 1;
    const thr = thresholds(r.mu, r.L, r.phi).filter((s) => s.S < bmax);
    Plotly.react('y-chart-omega', [
      { type: 'heatmap', x: M, y: b, z, colorscale: t.div, zmin: -lim, zmax: lim, zsmooth: false,
        colorbar: { thickness: 10, outlinewidth: 0, tickfont: { color: t.muted } },
        hovertemplate: 'M = %{x:.4f}<br>b = %{y:.4f}<br>ΔΩ = %{z:.4e}<extra></extra>' },
      { type: 'scatter', mode: 'markers', x: [r.M_star], y: [Number.isFinite(r.b_star) ? r.b_star : 0], hoverinfo: 'skip',
        marker: { symbol: 'star', size: 13, color: t.ink, line: { color: t.surface, width: 1.5 } } },
    ], baseLayout(t, { xaxis: { title: { text: 'M' } }, yaxis: { title: { text: 'b' } },
        shapes: thr.map((s) => ({ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: s.S, y1: s.S, line: { color: t.ink, width: 1, dash: 'dot' } })),
        margin: { l: 48, r: 6, t: 6, b: 40 } }), CFG);
  };
  let rows = 0;
  ask({ type: 'omega-map', mu: r.mu, L: r.L, M, b }, (d) => {
    if (d.done) { draw(); return; }
    z[d.i] = Array.from(d.row); rows++;
    if (rows % 10 === 0) draw();
  });
}

// ---------------------------------------------------------------------------
function syncSeg() {
  document.querySelectorAll('[data-yphi]').forEach((b) => b.classList.toggle('on', Number(b.dataset.yphi) === state.xphi));
  document.querySelectorAll('[data-yregion]').forEach((b) => b.classList.toggle('on', b.dataset.yregion === state.region));
  document.querySelectorAll('[data-yt]').forEach((b) => b.classList.toggle('on', Number(b.dataset.yt) === state.T));
  $('g').value = String(state.xg);
}

async function reloadMaps(resetView) {
  syncSeg(); writeHash();
  try {
    MAPS = await loadRegion();
    if (resetView) fullView();
    $('status').textContent = mapsStatus();
  } catch {
    MAPS = null;
    ['map-M', 'map-b'].forEach((id) => Plotly.purge($(id)));
    $('status').textContent = `Карта для T = ${fmtT(state.T)}, φ = ${state.xphi}, g = ${state.xg} (${state.region === 'full' ? 'вся область' : 'пики'}) ещё не посчитана. Расчёт точки работает.`;
  }
  drawMaps();
}

// ---------------------------------------------------------------------------
// Сравнение точек: глобальный минимум и срезы Ω(M, b*), Ω(M*, b); каждая точка помнит φ, g, T
// ---------------------------------------------------------------------------
const CMP_COLORS = ['#2a78d6', '#eb6834', '#1baa6e', '#8f5bd6', '#c9a227', '#d6457a', '#3aa6b9', '#7a6a58'];
const CMP_MAX = CMP_COLORS.length;
const CMP_RES = new Map();                  // key -> { r, M: {x, y}, b: {x, y} }
const cmpKey = (q) => [q.mu, q.L, q.phi, q.g, q.T].join(':');
function cmpList() {
  return state.cmp ? state.cmp.split(';').filter(Boolean).map((s) => {
    const [mu, L, phi, g, T] = s.split(':').map(Number);
    return { mu, L, phi, g, T };
  }) : [];
}
function cmpSave(list) { state.cmp = list.map(cmpKey).join(';'); writeHash(); }
const cmpLabel = (q) => `μ = ${q.mu}, L = ${q.L}, φ = ${q.phi}, g = ${q.g > 0 ? '+1' : '−1'}, T = ${fmtT(q.T)}`;

function cmpFeedback(msg) {
  const n = cmpList().length;
  $('cmp-msg').textContent = msg ? `${msg}; в сравнении ${n} ${n === 1 ? 'точка' : n < 5 ? 'точки' : 'точек'}.` : '';
  $('cmp-goto').hidden = n === 0;
  $('cmp-add').textContent = n ? `+ в сравнение (${n})` : '+ в сравнение';
}
function cmpAdd(mu, L) {
  const list = cmpList();
  const q = { mu: +(+mu).toFixed(4), L: +(+L).toFixed(4), phi: state.xphi, g: state.xg, T: state.T };
  if (list.some((x) => cmpKey(x) === cmpKey(q))) { cmpFeedback('эта точка уже в сравнении'); return; }
  if (list.length >= CMP_MAX) { cmpFeedback(`в сравнении уже ${CMP_MAX} точек — удалите лишние`); return; }
  list.push(q); cmpSave(list); drawMaps(); drawCompare();
  cmpFeedback(`✓ добавлена ${cmpLabel(q)}`);
}
function cmpRemove(k) { const list = cmpList(); list.splice(k, 1); cmpSave(list); drawMaps(); drawCompare(); cmpFeedback(''); }

function cmpTrace() {
  const list = cmpList();
  if (!list.length) return [];
  return [{ type: 'scatter', mode: 'markers', x: list.map((q) => q.L), y: list.map((q) => q.mu), text: list.map(cmpLabel),
            hovertemplate: '%{text}<extra>сравнение</extra>', showlegend: false,
            marker: { size: 11, color: list.map((_, k) => CMP_COLORS[k]), line: { color: '#ffffff', width: 1.5 } } }];
}

// компьютинг одной точки: минимум, затем два среза (асинхронно, в фоновом потоке)
function cmpCompute(q) {
  const key = cmpKey(q);
  if (CMP_RES.has(key)) return;
  const rec = { r: null, M: null, b: null };
  CMP_RES.set(key, rec);
  const base = { mu: q.mu, L: q.L, g: q.g, phi: q.phi };
  ask({ type: 'point', ...base, T: q.T }, ({ result: r }) => {
    rec.r = r;
    const Ms = r.M_star > 0 ? r.M_star : 0, bs = Number.isFinite(r.b_star) ? r.b_star : 0;
    // M: геометрическая сетка (узкие ямы) + равномерная
    const Mmax = Math.max(2.5, 1.5 * Ms);
    const Mx = [...new Set([...linspace(Math.log(1e-4), Math.log(Mmax), 160).map(Math.exp), ...linspace(Mmax / 160, Mmax, 160)])].sort((a, c) => a - c);
    ask({ type: 'slice', ...base, M: Mx, b: Mx.map(() => bs) }, ({ om }) => { rec.M = { x: Mx, y: Array.from(om) }; drawCompare(); });
    if (Ms > 0) {
      const bmax = q.mu + 1.0;
      const near = Ms > 0 ? linspace(Math.max(0, bs - 10 * Ms), Math.min(bmax, bs + 10 * Ms), 121) : [];
      const bx = [...new Set([...linspace(0, bmax, 241), ...near])].sort((a, c) => a - c);
      ask({ type: 'slice', ...base, M: bx.map(() => Ms), b: bx }, ({ om }) => { rec.b = { x: bx, y: Array.from(om) }; drawCompare(); });
    }
    drawCompare();
  });
}

function drawCompare() {
  const t = tokens(), list = cmpList();
  $('cmp-norm').checked = state.cmpNorm; $('cmp-logM').checked = state.cmpLogM;
  $('cmp-list').innerHTML = list.length ? list.map((q, k) => {
    const rec = CMP_RES.get(cmpKey(q)), r = rec && rec.r;
    const res = r ? ` → ${r.phase}, M* = ${r.M_star > 0 ? r.M_star.toPrecision(4) : '0'}${Number.isFinite(r.b_star) ? ', b* = ' + r.b_star.toPrecision(4) : ''}` : ' → считаю…';
    return `<span class="cmp-chip"><i style="background:${CMP_COLORS[k]}"></i>${cmpLabel(q)}${res}<button data-ycmp-del="${k}" title="убрать">×</button></span>`;
  }).join('') : '<span class="muted">точек нет</span>';
  document.querySelectorAll('[data-ycmp-del]').forEach((b) => b.addEventListener('click', () => cmpRemove(Number(b.dataset.ycmpDel))));
  if (!list.length) { Plotly.purge($('cmp-M')); Plotly.purge($('cmp-b')); return; }
  list.forEach(cmpCompute);
  const trM = [], trB = [];
  let Mright = 0.5;
  list.forEach((q, k) => {
    const rec = CMP_RES.get(cmpKey(q));
    if (!rec || !rec.r) return;
    const r = rec.r, col = CMP_COLORS[k], name = cmpLabel(q);
    const sc = (y) => (!state.cmpNorm ? 1 : r.omega_star < 0 ? Math.abs(r.omega_star) : Math.max(...y.map(Math.abs), 1e-300));
    Mright = Math.max(Mright, 1.5 * (r.M_star || 0), 2.0);
    if (rec.M) {
      const s = sc(rec.M.y);
      trM.push({ type: 'scatter', mode: 'lines', x: rec.M.x, y: rec.M.y.map((v) => v / s), name, line: { color: col, width: 2 },
                 hovertemplate: `${name}<br>M = %{x:.5g}<br>ΔΩ = %{y:.5e}<extra></extra>` });
      trM.push({ type: 'scatter', mode: 'markers', x: [r.M_star > 0 ? r.M_star : (state.cmpLogM ? 1e-4 : 0)], y: [r.omega_star / s],
                 showlegend: false, hoverinfo: 'skip', marker: { size: 9, color: col, line: { color: t.surface, width: 1.5 } } });
    }
    if (rec.b) {
      const s = sc(rec.b.y);
      trB.push({ type: 'scatter', mode: 'lines', x: rec.b.x, y: rec.b.y.map((v) => v / s), name, line: { color: col, width: 2 },
                 hovertemplate: `${name}<br>b = %{x:.5g}<br>ΔΩ = %{y:.5e}<extra></extra>` });
      trB.push({ type: 'scatter', mode: 'markers', x: [r.b_star], y: [r.omega_star / s], showlegend: false, hoverinfo: 'skip',
                 marker: { size: 9, color: col, line: { color: t.surface, width: 1.5 } } });
    }
  });
  const yr = state.cmpNorm ? { range: [-1.2, 1.5] } : {};
  const lay = (xt, xa) => baseLayout(t, { showlegend: true, xaxis: { title: { text: xt }, ...xa },
    yaxis: { title: { text: state.cmpNorm ? 'ΔΩ / |Ω*|' : 'Ω − Ω(0, 0)' }, zeroline: true, zerolinecolor: t.axis, zerolinewidth: 1.5,
             exponentformat: 'power', ...yr },
    legend: { orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', font: { size: 10, color: t.ink2 } }, margin: { l: 64, r: 10, t: 8, b: 40 } });
  Plotly.react($('cmp-M'), trM, lay('M', state.cmpLogM ? { type: 'log', range: [-4, Math.log10(Mright)], exponentformat: 'power' } : { range: [0, Mright] }), CFG);
  Plotly.react($('cmp-b'), trB, lay('b', {}), CFG);
}

let started = false;
export async function initInhT() {
  if (started) { window.dispatchEvent(new Event('resize')); return; }
  started = true;
  readHash();
  $('show-bnd').checked = state.bnd; $('slice-zoom').value = state.zoom;
  $('slice-zoom').addEventListener('change', () => { state.zoom = $('slice-zoom').value; writeHash(); if (lastPoint) drawSlices(lastPoint); });
  await reloadMaps(!location.hash.includes('y_vmu0'));
  runPoint(state.xmu, state.xL);
  for (const [id, key] of [['show-bnd', 'bnd']]) $(id).addEventListener('change', () => { state[key] = $(id).checked; writeHash(); drawMaps(); });
  const rerun = async (resetView) => { await reloadMaps(resetView); runPoint(state.xmu, state.xL); };
  document.querySelectorAll('[data-yphi]').forEach((b) => b.addEventListener('click', () => { state.xphi = Number(b.dataset.yphi); rerun(false); }));
  document.querySelectorAll('[data-yt]').forEach((b) => b.addEventListener('click', () => { state.T = Number(b.dataset.yt); rerun(false); }));
  $('g').addEventListener('change', () => { state.xg = Number($('g').value); rerun(false); });
  document.querySelectorAll('[data-yregion]').forEach((b) => b.addEventListener('click', () => { state.region = b.dataset.yregion; reloadMaps(true); }));
  let syncing = false;
  for (const id of ['map-M', 'map-b']) $(id).on('plotly_relayout', (ev) => {
    if (syncing) return;
    if (ev['xaxis.autorange'] || ev['yaxis.autorange']) fullView();
    else if ('xaxis.range[0]' in ev || 'yaxis.range[0]' in ev) {
      if ('xaxis.range[0]' in ev) { state.vL0 = +ev['xaxis.range[0]']; state.vL1 = +ev['xaxis.range[1]']; }
      if ('yaxis.range[0]' in ev) { state.vmu0 = +ev['yaxis.range[0]']; state.vmu1 = +ev['yaxis.range[1]']; }
    } else return;
    writeHash();
    syncing = true;
    Promise.all(['map-M', 'map-b'].map((m) => Plotly.relayout($(m), { 'xaxis.range': [state.vL0, state.vL1], 'yaxis.range': [state.vmu0, state.vmu1] })))
      .finally(() => { syncing = false; });
  });
  for (const id of ['map-M', 'map-b']) $(id).on('plotly_click', (ev) => {
    const p = ev.points[0];
    if (!(p && p.x != null && p.y != null)) return;
    if (ev.event && ev.event.shiftKey) cmpAdd(+p.y.toFixed(4), +p.x.toFixed(4)); else runPoint(+p.y.toFixed(4), +p.x.toFixed(4));
  });
  $('cmp-add').addEventListener('click', () => {
    cmpAdd(Number($('pt-mu').value), Math.max(0.05, Number($('pt-L').value)));
  });
  $('cmp-clear').addEventListener('click', () => { cmpSave([]); drawMaps(); drawCompare(); cmpFeedback(''); });
  $('cmp-goto').addEventListener('click', (e) => { e.preventDefault(); $('cmp-section').scrollIntoView({ behavior: 'smooth', block: 'start' }); });
  $('cmp-norm').addEventListener('change', () => { state.cmpNorm = $('cmp-norm').checked; writeHash(); drawCompare(); });
  $('cmp-logM').addEventListener('change', () => { state.cmpLogM = $('cmp-logM').checked; writeHash(); drawCompare(); });
  cmpFeedback(''); drawCompare();
  $('point-form').addEventListener('submit', (e) => {
    e.preventDefault();
    runPoint(Number($('pt-mu').value), Math.max(0.05, Number($('pt-L').value)));
  });
  $('dl-point').addEventListener('click', () => {
    if (!lastPoint) return;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(lastPoint, null, 1)], { type: 'application/json' }));
    a.download = `njl3_inhT_T${state.T}_phi${state.xphi}_g${state.xg}_mu${state.xmu}_L${state.xL}.json`; a.click();
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redrawInhT);
}

export function redrawInhT() {
  if (!started) return;
  drawMaps(); drawCompare();
  if (lastPoint) { drawOmegaMap(lastPoint); drawSlices(lastPoint); }
}
