// Интерфейс полной задачи: карты M*, b* (посчитаны заранее), расчёт точки в фоновом потоке.
import { thresholds, channelMMin } from './njl3full.js';

const G = -1.0;
const $ = (id) => document.getElementById(id);
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
              toImageButtonOptions: { format: 'png', scale: 2, filename: 'njl3_full' } };

const state = { theme: 'auto', bnd: true, asym: true, mu: 4.0, L: 2.0 };
function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  for (const k of Object.keys(state)) if (p.has(k)) {
    const d = state[k];
    state[k] = typeof d === 'boolean' ? p.get(k) === '1' : typeof d === 'string' ? p.get(k) : Number(p.get(k));
  }
}
function writeHash() {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(state)) p.set(k, typeof v === 'boolean' ? (v ? '1' : '0') : String(v));
  history.replaceState(null, '', '#' + p.toString());
}
function applyTheme() {
  if (state.theme === 'auto') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = state.theme;
}

// ---------------------------------------------------------------------------
// Карты
// ---------------------------------------------------------------------------
let MAPS = null;
const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
const pow10 = (k) => (k === 0 ? '1' : '10' + String(k).split('').map((c) => SUP[c]).join(''));
const tempT = () => (MAPS && MAPS.T) || 0;
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
  const { L, mu, lnM, b, phase, omega, asym } = MAPS;
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
    `${p}<br>M* = ${fmtM(lnM[i][j])}${asym[i][j] ? ' (асимпт.)' : ''}<br>b* = ${b[i][j] === null ? '—' : b[i][j].toFixed(5)}<br>Ω* = ${omega[i][j] === null ? '—' : omega[i][j].toExponential(3)}`)));
  const hov = 'L = %{x:.3f}<br>μ = %{y:.3f}<br>фаза %{customdata}<extra></extra>';
  const bnd = state.bnd ? boundaries() : { x: [], y: [] };
  const over = [
    { type: 'scatter', mode: 'lines', x: bnd.x, y: bnd.y, line: { color: t.ink, width: 1 }, hoverinfo: 'skip', showlegend: false },
  ];
  if (state.asym) {
    const ax = [], ay = [];
    L.forEach((Li, i) => mu.forEach((mj, j) => { if (asym[i][j]) { ax.push(Li); ay.push(mj); } }));
    over.push({ type: 'scatter', mode: 'markers', x: ax, y: ay, marker: { size: 2.5, color: t.muted }, hoverinfo: 'skip', showlegend: false });
  }
  over.push({ type: 'scatter', mode: 'markers', x: [state.L], y: [state.mu], hoverinfo: 'skip', showlegend: false,
              marker: { symbol: 'circle-open', size: 13, color: t.ink, line: { width: 2.5, color: t.ink } } });
  const lay = (title) => baseLayout(t, { xaxis: { title: { text: 'L' } }, yaxis: { title: { text: 'μ' } }, margin: { l: 52, r: 8, t: 6, b: 44 } });
  Plotly.react('map-M', [
    zeroLayer,
    { type: 'heatmap', x: L, y: mu, z: logM, customdata: cd, hovertemplate: hov, colorscale: t.seq, zmin, zmax: 1, zsmooth: false,
      colorbar: { title: { text: 'M*' }, thickness: 12, outlinewidth: 0, tickvals: ticks,
                  ticktext: ticks.map((k, i) => (i === 0 && zmin === -8 ? '≤' : '') + pow10(k)), tickfont: { color: t.muted } } },
    ...over], lay(), CFG);
  Plotly.react('map-b', [
    zeroLayer,
    { type: 'heatmap', x: L, y: mu, z: T(b), customdata: cd, hovertemplate: hov, colorscale: t.seq2, zmin: 0, zsmooth: false,
      colorbar: { title: { text: 'b*' }, thickness: 12, outlinewidth: 0, tickfont: { color: t.muted } } },
    ...over], lay(), CFG);
}

// ---------------------------------------------------------------------------
// Точка
// ---------------------------------------------------------------------------
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
let job = 0, lastPoint = null;
const pending = new Map();
worker.onmessage = ({ data }) => {
  if (data.type === 'error') { $('status').textContent = 'Ошибка расчёта: ' + data.message; return; }
  const cb = pending.get(data.job); if (cb) cb(data);
};
worker.onerror = (e) => { $('status').textContent = 'Ошибка фонового потока: ' + (e.message || e.filename || 'неизвестная'); };
function ask(msg, cb) { const id = ++job; pending.set(id, cb); worker.postMessage({ ...msg, job: id }); return id; }

function linspace(a, b, n) { return Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1)); }

function runPoint(mu, L) {
  state.mu = mu; state.L = L; writeHash();
  $('pt-mu').value = mu; $('pt-L').value = L;
  drawMaps();
  $("status").textContent = `Расчёт точки μ = ${mu}, L = ${L}: глобальный минимум по всем каналам, несколько секунд…`;
  const t0 = performance.now();
  ask({ type: 'point', mu, L, g: G, T: tempT() }, ({ result: r }) => {
    lastPoint = r;
    $('status').textContent = `Точка μ = ${mu}, L = ${L}: ${((performance.now() - t0) / 1000).toFixed(1)} с.`;
    $('point-table').innerHTML = `
      <tr><th>Фаза</th><td><span class="phase-badge">${r.phase}</span>${r.asymptotic ? ' (щель из асимптотики)' : ''}${r.phase === 'T' ? ' — M = 0: ни одна щель не переживает T' : ''}</td></tr>
      <tr><th>Порог T</th><td>${fmtT(r.T || 0)}</td></tr>
      <tr><th>M*</th><td>${fmtM(r.ln_M_star)} (ln M* = ${Number.isFinite(r.ln_M_star) ? r.ln_M_star.toFixed(5) : '−∞'})</td></tr>
      <tr><th>b*</th><td>${Number.isFinite(r.b_star) ? r.b_star.toFixed(8) : '—'}</td></tr>
      <tr><th>Ω* − Ω(0)</th><td>${r.omega_star.toExponential(6)}</td></tr>
      <tr><th>Кандидат b = 0</th><td>${r.b0.phase}: M = ${r.b0.M.toPrecision(8)}, ΔΩ = ${r.b0.omega.toExponential(4)}</td></tr>`;
    $('channels').innerHTML = '<tr><th>n</th><th>S<sub>n</sub></th><th>c<sub>n</sub></th><th>A<sub>n</sub></th><th>ln M*</th><th>b* − S<sub>n</sub></th><th>Ω*</th></tr>' +
      r.channels.map((c) => `<tr class="${r.phase === 'B' + c.n ? 'win' : ''}"><td>${c.n}</td><td>${c.S.toFixed(5)}</td><td>${c.c.toFixed(5)}</td>
        <td>${c.A.toFixed(5)}</td><td>${c.M > 0 ? c.lnM.toFixed(3) + (c.asym ? '*' : '') : '— (< порога)'}</td><td>${c.M > 0 ? (c.b - c.S).toExponential(2) : '—'}</td><td>${c.M > 0 ? c.omega.toExponential(3) : '0'}</td></tr>`).join('');
    drawOmegaMap(r);
    drawChannel(r);
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
    const thr = thresholds(r.mu, r.L).filter((s) => s.S < bmax);
    Plotly.react('chart-omega', [
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
  ask({ type: 'omega-map', mu: r.mu, L: r.L, g: G, M, b }, (d) => {
    if (d.done) { draw(); return; }
    z[d.i] = Array.from(d.row); rows++;
    if (rows % 10 === 0) draw();
  });
}

function drawChannel(r) {
  const t = tokens();
  let ch = r.channels.find((c) => r.phase === 'B' + c.n);
  // нет выигравшего канала (фаза T и др.): канал с наибольшей асимптотической щелью — ближайший к выживанию
  const lnMasy = (c) => -c.A / c.c - 0.5;
  if (!ch) ch = r.channels.reduce((a, c) => (lnMasy(c) > lnMasy(a) ? c : a), r.channels[0]);
  if (!ch) return;
  $('chan-title').innerHTML = `[Ω(M, S<sub>${ch.n}</sub>) − Ω(0)]/M² у порога n = ${ch.n}`;
  const lnM = linspace(Math.log(1e-8), Math.log(2), 120);
  const C = ch.n === 0 ? 1 : 2;
  ask({ type: 'channel', mu: r.mu, L: r.L, g: G, S: ch.S, C, lnM }, ({ num, as, asy }) => {
    const M = lnM.map(Math.exp);
    // ниже M ~ 1e-6 численная разность Ω тонет в округлении (~1e-14): показываем только надёжную часть
    const numSafe = num.map((v, i) => (M[i] >= 1e-6 ? v : null));
    Plotly.react('chart-chan', [
      { type: 'scatter', mode: 'lines', x: M, y: numSafe, name: 'численно', line: { color: t.s1, width: 2 },
        hovertemplate: 'M = %{x:.3e}<br>%{y:.5f}<extra>численно</extra>' },
      { type: 'scatter', mode: 'lines', x: M, y: as, name: 'A − c ln(1/M)', line: { color: t.s2, width: 2, dash: 'dash' },
        hovertemplate: 'M = %{x:.3e}<br>%{y:.5f}<extra>асимптотика</extra>' },
    ], baseLayout(t, { xaxis: { title: { text: 'M' }, type: 'log', exponentformat: 'power' },
        yaxis: { zeroline: true, zerolinecolor: t.axis, zerolinewidth: 1.5 },
        legend: { orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', font: { color: t.ink2 } },
        shapes: [
          ...(Number.isFinite(asy.lnM) && asy.lnM > Math.log(1e-8) ? [{ type: 'line', yref: 'paper', y0: 0, y1: 1,
            x0: Math.exp(asy.lnM), x1: Math.exp(asy.lnM), line: { color: t.muted, width: 1, dash: 'dot' } }] : []),
          // порог по температуре: щели левее не переживают T
          ...(tempT() > 0 ? [{ type: 'rect', yref: 'paper', y0: 0, y1: 1, x0: 1e-8, x1: channelMMin(ch.S, r.mu, tempT()),
            fillcolor: t.muted, opacity: 0.15, line: { width: 0 } }] : [])],
        annotations: tempT() > 0 ? [{ x: Math.log10(channelMMin(ch.S, r.mu, tempT())), y: 1, yref: 'paper', xanchor: 'right', yanchor: 'top',
          text: 'M < M<sub>min</sub>(T) ', showarrow: false, font: { color: t.muted, size: 11 } }] : [],
        margin: { l: 52, r: 6, t: 26, b: 40 } }), CFG);
  });
}

// ---------------------------------------------------------------------------
async function main() {
  readHash(); applyTheme();
  $('theme').value = state.theme; $('show-bnd').checked = state.bnd; $('show-asym').checked = state.asym;
  try {
    MAPS = await (await fetch('data/maps.json')).json();
    $('status').textContent = `Карты: ${MAPS.mu.length} × ${MAPS.L.length} точек (μ ∈ [${MAPS.mu[0]}, ${MAPS.mu.at(-1)}], L ∈ [${MAPS.L[0]}, ${MAPS.L.at(-1)}]), порог T = ${fmtT(tempT())}.`;
    $('temp-T').textContent = fmtT(tempT());
  } catch {
    $('status').textContent = 'Не удалось загрузить карты (data/maps.json).';
  }
  drawMaps();
  runPoint(state.mu, state.L);
  for (const [id, key] of [['show-bnd', 'bnd'], ['show-asym', 'asym']]) $(id).addEventListener('change', () => { state[key] = $(id).checked; writeHash(); drawMaps(); });
  $('theme').addEventListener('change', () => { state.theme = $('theme').value; writeHash(); applyTheme(); drawMaps(); if (lastPoint) { drawOmegaMap(lastPoint); drawChannel(lastPoint); } });
  for (const id of ['map-M', 'map-b']) $(id).on('plotly_click', (ev) => { const p = ev.points[0]; if (p && p.x != null && p.y != null) runPoint(+p.y.toFixed(4), +p.x.toFixed(4)); });
  $('point-form').addEventListener('submit', (e) => { e.preventDefault(); runPoint(Number($('pt-mu').value), Math.max(0.05, Number($('pt-L').value))); });
  $('dl-point').addEventListener('click', () => {
    if (!lastPoint) return;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(lastPoint, null, 1)], { type: 'application/json' }));
    a.download = `njl3_full_mu${state.mu}_L${state.L}.json`; a.click();
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', drawMaps);
}
main();
