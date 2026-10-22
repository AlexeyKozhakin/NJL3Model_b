// web/fast: карта глобального минимума Ω(M, b) и разбор точки, всё в браузере (fastmin.js в фоновых потоках).
import { thresholds } from '../../lab/js/njl3inh.js';

const $ = (id) => document.getElementById(id);
const root = document.querySelector('.viz-root');
const LN10 = Math.log(10);
const ORANGE = ['#fde6da', '#fbccb3', '#f8ad87', '#f38d5c', '#eb6834', '#cc5422', '#a84417', '#83340f'];

function tokens() {
  const cs = getComputedStyle(root);
  const v = (n) => cs.getPropertyValue(n).trim();
  const dark = v('--surface-1').toLowerCase() === '#1a1a19';
  const seq = v('--seq').split(',').map((s) => s.trim());
  const or = dark ? [...ORANGE].reverse() : ORANGE;
  return {
    surface: v('--surface-1'), ink: v('--text-primary'), ink2: v('--text-secondary'), muted: v('--text-muted'),
    grid: v('--grid'), axis: v('--axis'), s1: v('--series-1'), s2: v('--series-2'),
    seq: seq.map((c, i) => [i / (seq.length - 1), c]), seq2: or.map((c, i) => [i / (or.length - 1), c]),
    land: [[0, '#0d366b'], [0.35, '#2a78d6'], [0.7, '#9ec5f4'], [1, '#f4f8fd']],      // deep → shallow
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
              toImageButtonOptions: { format: 'png', scale: 2, filename: 'njl3_fast' } };

// ---------------------------------------------------------------------------
const state = { phi: 0, g: -1, Texp: -3, T0: false, mu0: 3.05, mu1: 3.3, L0: 2, L1: 10, res: 40,
                mu: 3.1409, L: 10, scale: 'deep', lres: 100 };
const T = () => (state.T0 ? 0 : 10 ** state.Texp);
const SUP = { '-': '⁻', '.': '.', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
const fmtT = () => {
  if (state.T0) return '0';
  const m = 10 ** (state.Texp - Math.floor(state.Texp)), e = Math.floor(state.Texp);
  return (Math.abs(m - 1) < 1e-9 ? '' : m.toFixed(2) + '·') + '10' + String(e).split('').map((c) => SUP[c]).join('');
};
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
function syncInputs() {
  document.querySelectorAll('[data-phi]').forEach((b) => b.classList.toggle('on', Number(b.dataset.phi) === state.phi));
  $('g').value = String(state.g); $('T-exp').value = state.Texp; $('T-zero').checked = state.T0; $('T-val').textContent = fmtT();
  $('mu-lo').value = state.mu0; $('mu-hi').value = state.mu1; $('L-lo').value = state.L0; $('L-hi').value = state.L1;
  $('res').value = String(state.res); $('pt-mu').value = state.mu; $('pt-L').value = state.L;
  $('land-scale').value = state.scale; $('land-res').value = String(state.lres);
}
function readInputs() {
  state.g = Number($('g').value); state.Texp = Number($('T-exp').value); state.T0 = $('T-zero').checked;
  state.mu0 = Number($('mu-lo').value); state.mu1 = Number($('mu-hi').value);
  state.L0 = Math.max(0.05, Number($('L-lo').value)); state.L1 = Number($('L-hi').value); state.res = Number($('res').value);
}
const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));

// ---------------------------------------------------------------------------
// Пул фоновых потоков
// ---------------------------------------------------------------------------
const NW = Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 1));
let pool = [];
function newPool() {
  pool.forEach((w) => w.terminate());
  pool = Array.from({ length: NW }, () => new Worker(new URL('./worker.js', import.meta.url), { type: 'module' }));
}
// отдельный поток для точки (не ждёт карту)
let pw = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
let pjob = 0;
const pending = new Map();
pw.onmessage = ({ data }) => {
  if (data.type === 'error') { $('status').textContent = 'Ошибка: ' + data.message; return; }
  const cb = pending.get(data.job); if (cb) cb(data);
};
function ask(msg, cb) { const id = ++pjob; pending.set(id, cb); pw.postMessage({ ...msg, phi: state.phi, g: state.g, T: T(), job: id }); }

// ---------------------------------------------------------------------------
// Карта
// ---------------------------------------------------------------------------
let MAP = null, mapJob = 0;
function buildMap() {
  readInputs(); writeHash(); syncInputs();
  const n = state.res, job = ++mapJob;
  MAP = { mu: linspace(state.mu0, state.mu1, n), L: linspace(state.L0, state.L1, n) };
  MAP.lnM = MAP.L.map(() => new Array(n).fill(null)); MAP.b = MAP.L.map(() => new Array(n).fill(null));
  MAP.phase = MAP.L.map(() => new Array(n).fill('')); MAP.om = MAP.L.map(() => new Array(n).fill(null));
  newPool();
  let done = 0, wdone = 0, pend = false, cpu = 0;
  const t0 = performance.now();
  $('build').disabled = true;
  const flush = () => { pend = false; drawMaps(); };
  pool.forEach((w, k) => {
    w.onmessage = ({ data }) => {
      if (data.job !== job) return;
      if (data.type === 'error') { $('status').textContent = 'Ошибка: ' + data.message; return; }
      if (data.done) {
        if (++wdone === NW) {
          drawMaps(); $('build').disabled = false; $('progress-bar').style.width = '0';
          $('status').textContent = `Карта ${n}×${n} (φ = ${state.phi}, g = ${state.g}, T = ${fmtT()}): ${((performance.now() - t0) / 1000).toFixed(1)} с ` +
            `на ${NW} потоках, ${(cpu / (n * n)).toFixed(0)} мс на точку.`;
        }
        return;
      }
      const i = data.i;
      MAP.lnM[i] = Array.from(data.lnM); MAP.b[i] = Array.from(data.b); MAP.om[i] = Array.from(data.om); MAP.phase[i] = data.phase;
      cpu += data.ms; done++;
      $('progress-bar').style.width = `${(100 * done / n).toFixed(1)}%`;
      const el = (performance.now() - t0) / 1000;
      $('status').textContent = `Карта: ${done} из ${n} столбцов по L, ${el.toFixed(0)} с, осталось ~${(el * (n - done) / done).toFixed(0)} с…`;
      if (!pend) { pend = true; setTimeout(flush, 300); }
    };
    const rows = [];
    for (let i = k; i < n; i += NW) rows.push({ i, L: MAP.L[i] });
    // дорогие (большие L) — вперемешку: строки распределены по потокам через одну
    w.postMessage({ type: 'map', job, rows, mu: MAP.mu, phi: state.phi, g: state.g, T: T() });
  });
}

function boundaries() {
  const { L, mu, phase } = MAP, x = [], y = [];
  for (let i = 0; i < L.length; i++) for (let j = 0; j < mu.length; j++) {
    if (!phase[i][j]) continue;
    if (i + 1 < L.length && phase[i + 1][j] && phase[i][j] !== phase[i + 1][j]) {
      const xm = 0.5 * (L[i] + L[i + 1]);
      x.push(xm, xm, null); y.push(j ? 0.5 * (mu[j - 1] + mu[j]) : mu[j], j + 1 < mu.length ? 0.5 * (mu[j] + mu[j + 1]) : mu[j], null);
    }
    if (j + 1 < mu.length && phase[i][j + 1] && phase[i][j] !== phase[i][j + 1]) {
      const ym = 0.5 * (mu[j] + mu[j + 1]);
      x.push(i ? 0.5 * (L[i - 1] + L[i]) : L[i], i + 1 < L.length ? 0.5 * (L[i] + L[i + 1]) : L[i], null); y.push(ym, ym, null);
    }
  }
  return { x, y };
}

function drawMaps() {
  if (!MAP) return;
  const t = tokens(), { L, mu } = MAP;
  const Tz = (A, f) => mu.map((_, j) => L.map((_, i) => f(A[i][j], i, j)));
  const logM = Tz(MAP.lnM, (v) => (v === null || !Number.isFinite(v) ? null : Math.max(v / LN10, -8)));
  const flat = logM.flat().filter((v) => v !== null);
  const zmin = Math.max(-8, Math.min(-1, Math.floor(flat.length ? Math.min(...flat) : -3)));
  const zero = Tz(MAP.phase, (p) => (p === 'T' ? 1 : null));
  const cd = Tz(MAP.phase, (p, i, j) => `${p || '…'}<br>M* = ${MAP.lnM[i][j] !== null && Number.isFinite(MAP.lnM[i][j]) ? Math.exp(MAP.lnM[i][j]).toPrecision(5) : '0'}` +
    `<br>b* = ${MAP.b[i][j] !== null && Number.isFinite(MAP.b[i][j]) ? MAP.b[i][j].toFixed(5) : '—'}<br>Ω* = ${MAP.om[i][j] !== null ? MAP.om[i][j].toExponential(3) : '—'}`);
  const hov = 'L = %{x:.3f}<br>μ = %{y:.4f}<br>фаза %{customdata}<extra></extra>';
  const bnd = boundaries();
  const over = [{ type: 'scatter', mode: 'lines', x: bnd.x, y: bnd.y, line: { color: t.ink, width: 1 }, hoverinfo: 'skip' },
                { type: 'scatter', mode: 'markers', x: [state.L], y: [state.mu], hoverinfo: 'skip',
                  marker: { symbol: 'circle-open', size: 13, color: t.ink, line: { width: 2.5, color: t.ink } } }];
  const zl = { type: 'heatmap', x: L, y: mu, z: zero, colorscale: [[0, t.muted], [1, t.muted]], opacity: 0.35, showscale: false, hoverinfo: 'skip' };
  const lay = () => baseLayout(t, { showlegend: false, xaxis: { title: { text: 'L' }, range: [L[0], L[L.length - 1]] },
    yaxis: { title: { text: 'μ' }, range: [mu[0], mu[mu.length - 1]] }, margin: { l: 56, r: 8, t: 6, b: 44 } });
  Plotly.react('map-M', [zl, { type: 'heatmap', x: L, y: mu, z: logM, customdata: cd, hovertemplate: hov, colorscale: t.seq, zmin, zmax: 0.6,
    colorbar: { title: { text: 'lg M*' }, thickness: 12, outlinewidth: 0, tickfont: { color: t.muted } } }, ...over], lay(), CFG);
  Plotly.react('map-b', [zl, { type: 'heatmap', x: L, y: mu, z: Tz(MAP.b, (v) => (v === null || !Number.isFinite(v) ? null : v)), customdata: cd,
    hovertemplate: hov, colorscale: t.seq2, zmin: 0, colorbar: { title: { text: 'b*' }, thickness: 12, outlinewidth: 0, tickfont: { color: t.muted } } },
    ...over], lay(), CFG);
}

// ---------------------------------------------------------------------------
// Точка: результат, все минимумы, рельеф
// ---------------------------------------------------------------------------
let LAND = null, MINIMA = [], POINT = null;
function runPoint(mu, L) {
  state.mu = mu; state.L = L; writeHash(); syncInputs(); drawMaps();
  $('point-table').innerHTML = '<tr><td>считаю…</td></tr>'; $('minima').innerHTML = '';
  const t0 = performance.now();
  ask({ type: 'point', mu, L }, ({ result: r }) => {
    POINT = r;
    $('point-table').innerHTML = `
      <tr><th>Глобальный минимум</th><td><span class="phase-badge">${r.phase}</span> ${r.asymptotic ? '(щель из асимптотики)' : ''}</td></tr>
      <tr><th>M*</th><td>${Number.isFinite(r.ln_M_star) ? Math.exp(r.ln_M_star).toPrecision(8) : '0'}</td></tr>
      <tr><th>b*</th><td>${Number.isFinite(r.b_star) ? r.b_star.toFixed(8) : '— (M = 0)'}</td></tr>
      <tr><th>Ω* − Ω(0, 0)</th><td>${r.omega_star.toExponential(6)}</td></tr>
      <tr><th>φ, g, T</th><td>${r.phi}, ${r.g}, ${fmtT()}</td></tr>
      <tr><th>Время</th><td>${(performance.now() - t0).toFixed(0)} мс</td></tr>`;
    drawLandscape();
  });
  ask({ type: 'minima', mu, L }, ({ minima }) => { MINIMA = minima; drawMinimaTable(); drawLandscape(); });
  const n = state.lres, M = linspace(0.02, 3.5, n), b = linspace(0, mu + 0.5, n);
  LAND = { mu, L, M, b, z: b.map(() => new Array(n).fill(null)) };
  let rows = 0;
  ask({ type: 'landscape', mu, L, M, b }, (d) => {
    if (d.done) { drawLandscape(); return; }
    LAND.z[d.i] = Array.from(d.row); rows++;
    if (rows % 15 === 0) drawLandscape();
  });
}

function drawMinimaTable() {
  if (!MINIMA.length) { $('minima').innerHTML = '<tr><td>локальных минимумов с Ω &lt; 0 нет</td></tr>'; return; }
  const best = MINIMA[0].omega;
  const thr = thresholds(state.mu, state.L, state.phi);
  const kind = (m) => {
    if (m.b === 0) return `b = 0, ${m.phase}`;
    const q = thr.reduce((a, x) => (Math.abs(x.S - m.b) < Math.abs(a.S - m.b) ? x : a), thr[0]);
    return q && Math.abs(q.S - m.b) < 0.15 * Math.max(m.M, 0.05) + 1e-3 ? `у порога S<sub>${q.n}</sub>` : 'между порогами';
  };
  $('minima').innerHTML = '<tr><th>№</th><th>M</th><th>b</th><th>Ω</th><th>ΔΩ</th><th>ΔΩ/|Ω|</th><th></th></tr>' +
    MINIMA.map((m, k) => `<tr class="${k === 0 ? 'best' : ''}"><td>${k + 1}</td><td>${m.M.toPrecision(5)}</td><td>${m.b.toPrecision(5)}</td>` +
      `<td>${m.omega.toExponential(4)}</td><td>${k ? (m.omega - best).toExponential(2) : '—'}</td><td>${k ? (100 * (m.omega - best) / Math.abs(best)).toFixed(2) + '%' : ''}</td>` +
      `<td>${kind(m)}</td></tr>`).join('');
}

function drawLandscape() {
  if (!LAND) return;
  const t = tokens(), { M, b, z, mu, L } = LAND;
  const vals = z.flat().filter((v) => v !== null && Number.isFinite(v));
  if (!vals.length) return;
  const zmin = Math.min(...vals, ...(MINIMA.map((m) => m.omega)));
  let lo = zmin, hi = 0;
  if (state.scale === 'near') hi = zmin / 2;
  if (state.scale === 'full') hi = Math.max(...vals);
  const thr = thresholds(mu, L, state.phi).filter((q) => q.S < mu + 0.5);
  const top = MINIMA.slice(0, 30);
  Plotly.react('landscape', [
    { type: 'heatmap', x: M, y: b, z, colorscale: t.land, zmin: lo, zmax: hi, zsmooth: 'best',
      colorbar: { title: { text: 'ΔΩ' }, thickness: 12, outlinewidth: 0, tickfont: { color: t.muted }, exponentformat: 'power' },
      hovertemplate: 'M = %{x:.4f}<br>b = %{y:.4f}<br>ΔΩ = %{z:.5e}<extra></extra>' },
    { type: 'scatter', mode: 'markers+text', x: top.map((m) => m.M), y: top.map((m) => m.b), text: top.map((_, k) => String(k + 1)),
      textposition: 'top right', textfont: { color: t.ink, size: 11 }, hoverinfo: 'text',
      hovertext: top.map((m, k) => `№${k + 1}: M = ${m.M.toPrecision(5)}, b = ${m.b.toPrecision(5)}<br>Ω = ${m.omega.toExponential(5)}`),
      marker: { size: 9, color: 'rgba(0,0,0,0)', line: { color: t.ink, width: 2 } } },
    ...(POINT && POINT.M_star > 0 ? [{ type: 'scatter', mode: 'markers', x: [POINT.M_star], y: [Number.isFinite(POINT.b_star) ? POINT.b_star : 0],
      hoverinfo: 'skip', marker: { symbol: 'star', size: 16, color: '#f2c94c', line: { color: t.ink, width: 1.5 } } }] : []),
  ], baseLayout(t, { showlegend: false, xaxis: { title: { text: 'M' }, range: [0, 3.5] }, yaxis: { title: { text: 'b' }, range: [0, mu + 0.5] },
      title: { text: `μ = ${mu}, L = ${L}, φ = ${state.phi}, g = ${state.g}`, font: { size: 12, color: t.ink2 }, x: 0.01, xanchor: 'left', y: 0.995 },
      shapes: thr.map((q) => ({ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: q.S, y1: q.S, line: { color: t.muted, width: 1, dash: 'dot' } })),
      margin: { l: 52, r: 8, t: 26, b: 44 } }), CFG);
}

// ---------------------------------------------------------------------------
function main() {
  readHash(); syncInputs();
  $('build').addEventListener('click', buildMap);
  document.querySelectorAll('[data-phi]').forEach((b) => b.addEventListener('click', () => {
    state.phi = Number(b.dataset.phi); syncInputs(); writeHash(); runPoint(state.mu, state.L);
    $('status').textContent = 'φ изменено: нажмите «Построить карту», чтобы пересчитать карту.';
  }));
  $('g').addEventListener('change', () => { readInputs(); writeHash(); runPoint(state.mu, state.L); });
  $('T-exp').addEventListener('input', () => { state.Texp = Number($('T-exp').value); $('T-val').textContent = fmtT(); });
  $('T-exp').addEventListener('change', () => { readInputs(); writeHash(); runPoint(state.mu, state.L);
    $('status').textContent = `T = ${fmtT()}: точка пересчитана; для карты нажмите «Построить карту».`; });
  $('T-zero').addEventListener('change', () => { readInputs(); syncInputs(); writeHash(); runPoint(state.mu, state.L); });
  document.querySelectorAll('[data-preset]').forEach((b) => b.addEventListener('click', () => {
    const [a, c, d, e] = b.dataset.preset.split(',').map(Number);
    $('mu-lo').value = a; $('mu-hi').value = c; $('L-lo').value = d; $('L-hi').value = e;
  }));
  document.querySelectorAll('[data-pt]').forEach((b) => b.addEventListener('click', () => {
    const [mu, L, phi] = b.dataset.pt.split(',').map(Number);
    state.phi = phi; syncInputs(); runPoint(mu, L);
  }));
  $('point-form').addEventListener('submit', (e) => { e.preventDefault(); runPoint(Number($('pt-mu').value), Math.max(0.05, Number($('pt-L').value))); });
  $('land-scale').addEventListener('change', () => { state.scale = $('land-scale').value; writeHash(); drawLandscape(); });
  $('land-res').addEventListener('change', () => { state.lres = Number($('land-res').value); writeHash(); runPoint(state.mu, state.L); });
  for (const id of ['map-M', 'map-b']) $(id).on('plotly_click', (ev) => { const p = ev.points[0]; if (p && p.x != null) runPoint(+p.y.toFixed(5), +p.x.toFixed(4)); });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { drawMaps(); drawLandscape(); });
  runPoint(state.mu, state.L);
}
main();
