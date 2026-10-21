// Интерфейс: карта устойчивости (готовые данные) и кривые Φ(k, q₁) в точке (считаются в браузере).
import { findMinimum, phaseLabel } from '../../b0/js/njl3b0.js';
import { channels, phi, phiMat, phiVac, singularQ1 } from './njl3stab.js';

const G = -1.0, T_MAP = 1e-3, Q_MAX = 10;
const $ = (id) => document.getElementById(id);
const root = document.querySelector('.viz-root');

function tokens() {
  const cs = getComputedStyle(root);
  const v = (n) => cs.getPropertyValue(n).trim();
  return {
    surface: v('--surface-1'), ink: v('--text-primary'), ink2: v('--text-secondary'), muted: v('--text-muted'),
    grid: v('--grid'), axis: v('--axis'), negS: v('--neg-strong'), neg: v('--neg'), mid: v('--mid'),
    pos: v('--pos'), posS: v('--pos-strong'), blank: v('--blank'),
    seq: v('--seq').split(',').map((s) => s.trim()),
  };
}

function baseLayout(t, extra = {}) {
  return {
    paper_bgcolor: t.surface, plot_bgcolor: t.blank,
    font: { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', size: 12, color: t.ink2 },
    margin: { l: 56, r: 16, t: 8, b: 44 },
    hoverlabel: { bgcolor: t.surface, bordercolor: t.axis, font: { color: t.ink } },
    xaxis: { gridcolor: t.grid, linecolor: t.axis, zerolinecolor: t.axis, tickfont: { color: t.muted } },
    yaxis: { gridcolor: t.grid, linecolor: t.axis, zerolinecolor: t.axis, tickfont: { color: t.muted } },
    ...extra,
  };
}
const PLOT_CONFIG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                      toImageButtonOptions: { format: 'png', scale: 2, filename: 'njl3_stab' } };

// ---------------------------------------------------------------------------
// Состояние и ссылка
// ---------------------------------------------------------------------------
const state = { region: 'full', quantity: 'phi_min', lines: true, theme: 'auto', mu: 3.3, L: 2, kmax: 3, T: true };

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  for (const k of Object.keys(state)) {
    if (!p.has(k)) continue;
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
function syncInputs() {
  $('quantity').value = state.quantity; $('show-lines').checked = state.lines; $('theme').value = state.theme;
  $('pt-mu').value = state.mu; $('pt-L').value = state.L; $('pt-kmax').value = state.kmax; $('pt-T').checked = state.T;
  for (const b of document.querySelectorAll('[data-region]')) b.classList.toggle('primary', b.dataset.region === state.region);
}

// ---------------------------------------------------------------------------
// Карта
// ---------------------------------------------------------------------------
let DATA = null;

function phaseCode(p) { return p === 'V' ? 0 : p === 'T' ? 2 : 1; }

function drawMap() {
  if (!DATA) return;
  const t = tokens(), m = DATA.maps[state.region], q = state.quantity;
  const traces = [];
  const hoverZ = m.phase.map((row, i) => row.map((p, j) =>
    `μ = ${m.mu[i].toFixed(4)}, L = ${m.L[j].toFixed(4)}<br>фаза ${p}` +
    (p === 'T' ? '' : `<br>min Φ (T=10⁻³) = ${m.phi_min[i][j]}<br>только k=0: ${m.phi_min_k0[i][j]}<br>k* = ${m.k[i][j]}` +
      (m.T0_inf[i][j] ? '<br>T = 0: −∞' : ''))));
  if (q === 'k') {
    const z = m.k.map((row, i) => row.map((v, j) => (m.phase[i][j].startsWith('I') ? v : null)));
    const zb = m.phase.map((row) => row.map((p) => (p === 'V' ? 1 : null)));
    traces.push({ type: 'heatmap', x: m.L, y: m.mu, z: zb, colorscale: [[0, t.pos], [1, t.pos]], showscale: false,
                  hoverinfo: 'skip' });
    traces.push({ type: 'heatmap', x: m.L, y: m.mu, z, text: hoverZ, hoverinfo: 'text',
                  colorscale: t.seq.map((c, i, a) => [i / (a.length - 1), c]), zmin: 0,
                  colorbar: { title: { text: 'k*', side: 'right' }, tickfont: { color: t.muted } } });
  } else {
    const src = q === 'phi_min_k0' ? m.phi_min_k0 : m.phi_min;
    let z = src;
    if (q === 'T0') z = src.map((row, i) => row.map((v, j) => (m.T0_inf[i][j] ? null : v)));
    let lo = 0, hi = 0;
    for (const row of z) for (const v of row) if (v !== null) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    hi = Math.min(hi, 1.0); lo = Math.min(lo, -0.05);
    const a = -lo / (hi - lo);                        // position of zero on [0, 1]
    traces.push({ type: 'heatmap', x: m.L, y: m.mu, z, text: hoverZ, hoverinfo: 'text', zmin: lo, zmax: hi,
                  colorscale: [[0, t.negS], [a * 0.5, t.neg], [a, t.mid], [a + (1 - a) * 0.5, t.pos], [1, t.posS]],
                  colorbar: { title: { text: 'min Φ', side: 'right' }, tickfont: { color: t.muted } } });
    if (q === 'T0') {
      const zi = m.T0_inf.map((row) => row.map((v) => (v ? 1 : null)));
      traces.push({ type: 'heatmap', x: m.L, y: m.mu, z: zi, text: hoverZ, hoverinfo: 'text',
                    colorscale: [[0, t.negS], [1, t.negS]], showscale: false, name: '−∞' });
    }
  }
  if (state.lines) {
    traces.push({ type: 'contour', x: m.L, y: m.mu, z: m.phase.map((row) => row.map(phaseCode)), hoverinfo: 'skip',
                  showscale: false, contours: { coloring: 'none', start: 0.5, end: 1.5, size: 1 },
                  line: { color: t.ink, width: 1 } });
  }
  traces.push({ type: 'scatter', mode: 'markers', x: [state.L], y: [state.mu], hoverinfo: 'skip', showlegend: false,
                marker: { size: 11, color: 'rgba(0,0,0,0)', line: { color: t.ink, width: 2 } } });
  const layout = baseLayout(t, {
    xaxis: { ...baseLayout(t).xaxis, title: { text: 'L' }, range: [m.L[0], m.L[m.L.length - 1]] },
    yaxis: { ...baseLayout(t).yaxis, title: { text: 'μ' }, range: [m.mu[0], m.mu[m.mu.length - 1]] },
    showlegend: false,
  });
  Plotly.react('diagram', traces, layout, PLOT_CONFIG);
}

// ---------------------------------------------------------------------------
// Точка: Φ(k, q₁)
// ---------------------------------------------------------------------------
const COLORS = ['#2a78d6', '#eb6834', '#1baa6e', '#8f5bd6', '#c9a227', '#d6457a', '#3aa6b9', '#7a6a58'];
let LAST = null;

function qGrid(sing) {
  const q = [];
  for (let i = 1; i <= 1200; i++) q.push(Q_MAX * i / 1200);
  for (const s of sing) for (let e = 2; e <= 9; e++) for (const sg of [-1, 1]) {
    const x = s + sg * 10 ** -e;
    if (x > 0 && x <= Q_MAX) q.push(x);
  }
  return [...new Set(q)].sort((a, b) => a - b);
}

function computePoint() {
  const mu = state.mu, L = state.L, kmax = Math.max(0, Math.min(20, Math.round(state.kmax)));
  const r = findMinimum(mu, L, G), M = r.M, ph = phaseLabel(M, mu, L);
  const out = { mu, L, M, ph, kmax, curves: [], ch: M > 0 ? channels(M, mu, L) : [] };
  if (M === 0) return out;
  for (let k = 0; k <= kmax; k++) {
    const sing = singularQ1(M, mu, L, k).filter((x) => x > 0 && x <= Q_MAX);
    const q = qGrid(sing);
    const v = q.map((x) => phiVac(k, x, M, L) + phiMat(k, x, M, mu, L));
    let vT = null;
    if (state.T) {
      // T = 10⁻³ differs from T = 0 only near the singularities (|Δ| < 1e-5 farther than ~0.3)
      vT = v.slice();
      for (let i = 0; i < q.length; i++)
        if (sing.some((s) => Math.abs(q[i] - s) < 0.3)) vT[i] = phi(k, q[i], M, mu, L, T_MAP);
    }
    let minT0 = Infinity, qmin = null;
    for (let i = 0; i < q.length; i++) if (!sing.some((s) => Math.abs(q[i] - s) < 1e-3) && v[i] < minT0) { minT0 = v[i]; qmin = q[i]; }
    let minT3 = null;
    if (state.T) {
      minT3 = Infinity;
      for (const x of vT) minT3 = Math.min(minT3, x);
      for (const s of sing) minT3 = Math.min(minT3, phi(k, s, M, mu, L, T_MAP));
    }
    out.curves.push({ k, q, v, vT, sing, minT0, qmin, minT3 });
  }
  return out;
}

function fmt(x, d = 4) { return x === null || x === undefined ? '—' : Number.isFinite(x) ? x.toFixed(d) : (x < 0 ? '−∞' : '∞'); }

function renderPoint() {
  const t = tokens();
  const P = LAST = computePoint();
  const rows = [
    ['Фаза', `<span class="phase-badge">${P.ph}</span> &nbsp; M* = ${fmt(P.M, 6)}`],
  ];
  if (P.M === 0) {
    rows.push(['', 'M* = 0: формула Φ (через уравнение щели) неприменима']);
  } else {
    rows.push(['Заполненные каналы', P.ch.length ? P.ch.filter((c) => c.n >= 0).map((c) =>
      `n = ${c.n === 0 ? '0' : '±' + c.n}: k<sub>F</sub> = ${fmt(c.P)}`).join('<br>') : 'нет (μ ≤ M*): Φ = Φ<sub>vac</sub> &gt; 0']);
    for (const c of P.curves) {
      const sing = c.sing.length ? ` · особенности q₁ = ${c.sing.slice(0, 4).map((x) => fmt(x, 3)).join(', ')}${c.sing.length > 4 ? '…' : ''}` : '';
      rows.push([`k = ${c.k}`, `min Φ: T=0 ${c.sing.length ? '−∞' : fmt(c.minT0)}` +
        (c.minT3 !== null ? `, T=10⁻³ ${fmt(c.minT3)}` : '') + sing]);
    }
    const unstable = P.curves.some((c) => c.sing.length || c.minT0 < 0);
    rows.push(['Итог (T = 0)', unstable ? '<b>неустойчива</b>' : 'устойчива (Φ &gt; 0 при всех показанных k)']);
  }
  $('point-table').innerHTML = rows.map(([a, b]) => `<tr><th>${a}</th><td>${b}</td></tr>`).join('');

  const traces = [];
  let lo = 0, hi = 0.2;
  for (const c of P.curves) {
    const col = COLORS[c.k % COLORS.length];
    traces.push({ type: 'scatter', mode: 'lines', x: c.q, y: c.v, name: `k = ${c.k}`, line: { color: col, width: 1.6 },
                  hovertemplate: `k = ${c.k}<br>q₁ = %{x:.4f}<br>Φ = %{y:.5f}<extra>T = 0</extra>` });
    if (c.vT) traces.push({ type: 'scatter', mode: 'lines', x: c.q, y: c.vT, name: `k = ${c.k}, T = 10⁻³`, showlegend: false,
                            line: { color: col, width: 1.2, dash: 'dash' },
                            hovertemplate: `k = ${c.k}<br>q₁ = %{x:.4f}<br>Φ = %{y:.5f}<extra>T = 10⁻³</extra>` });
    for (const x of (c.vT || [])) lo = Math.min(lo, x);
    for (const x of c.v) if (Number.isFinite(x)) hi = Math.max(hi, Math.min(x, 3));
  }
  lo = P.curves.some((c) => c.sing.length) ? Math.min(-0.5, 1.3 * lo) : -0.15;
  if (!state.T && P.curves.some((c) => c.sing.length)) lo = -1.5;
  const shapes = [];
  for (const c of P.curves) for (const s of c.sing)
    shapes.push({ type: 'line', x0: s, x1: s, y0: 0, y1: 1, yref: 'paper', line: { color: COLORS[c.k % COLORS.length], width: 1, dash: 'dot' } });
  shapes.push({ type: 'line', x0: 0, x1: Q_MAX, y0: 0, y1: 0, line: { color: t.ink, width: 1 } });
  const layout = baseLayout(t, {
    plot_bgcolor: t.surface, shapes,
    xaxis: { ...baseLayout(t).xaxis, title: { text: 'q₁' }, range: [0, Q_MAX] },
    yaxis: { ...baseLayout(t).yaxis, title: { text: 'Φ(k, q₁)' }, range: [lo, 1.05 * hi] },
    legend: { orientation: 'h', y: 1.12, font: { size: 11 } }, margin: { l: 56, r: 12, t: 30, b: 44 },
  });
  if (P.M === 0) {
    Plotly.react('chart-phi', [], { ...layout, annotations: [{ text: 'M* = 0', showarrow: false, xref: 'paper', yref: 'paper', x: 0.5, y: 0.5 }] }, PLOT_CONFIG);
  } else {
    Plotly.react('chart-phi', traces, layout, PLOT_CONFIG);
  }
}

function downloadPoint() {
  if (!LAST || !LAST.curves.length) return;
  const lines = ['k,q1,phi_T0,phi_T1e-3'];
  for (const c of LAST.curves) c.q.forEach((x, i) => lines.push(`${c.k},${x},${c.v[i]},${c.vT ? c.vT[i] : ''}`));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/csv' }));
  a.download = `phi_mu${LAST.mu}_L${LAST.L}.csv`;
  a.click();
}

// ---------------------------------------------------------------------------
// События
// ---------------------------------------------------------------------------
function setPoint(mu, L) {
  state.mu = +mu.toFixed(4); state.L = +L.toFixed(4);
  syncInputs(); writeHash();
  $('status').textContent = 'Считаю Φ(q₁)…';
  setTimeout(() => { const t0 = performance.now(); renderPoint(); drawMap();
    $('status').textContent = `Точка μ = ${state.mu}, L = ${state.L}: ${(performance.now() - t0).toFixed(0)} мс`; }, 10);
}

async function main() {
  readHash(); applyTheme(); syncInputs();
  $('status').textContent = 'Загружаю карты…';
  try {
    DATA = await (await fetch('data/maps.json')).json();
    $('status').textContent = '';
  } catch (e) {
    $('status').textContent = 'Не удалось загрузить data/maps.json: ' + e;
  }
  drawMap();
  setPoint(state.mu, state.L);
  $('diagram').on('plotly_click', (ev) => {
    const p = ev.points && ev.points[0];
    if (p) setPoint(Number(p.y), Number(p.x));
  });
  for (const b of document.querySelectorAll('[data-region]'))
    b.addEventListener('click', () => { state.region = b.dataset.region; syncInputs(); writeHash(); drawMap(); });
  $('quantity').addEventListener('change', () => { state.quantity = $('quantity').value; writeHash(); drawMap(); });
  $('show-lines').addEventListener('change', () => { state.lines = $('show-lines').checked; writeHash(); drawMap(); });
  $('theme').addEventListener('change', () => { state.theme = $('theme').value; applyTheme(); writeHash(); drawMap(); renderPoint(); });
  $('pt-T').addEventListener('change', () => { state.T = $('pt-T').checked; setPoint(state.mu, state.L); });
  $('point-form').addEventListener('submit', (e) => {
    e.preventDefault();
    state.kmax = Number($('pt-kmax').value);
    setPoint(Number($('pt-mu').value), Math.max(0.05, Number($('pt-L').value)));
  });
  $('dl-point').addEventListener('click', downloadPoint);
  $('copy-link').addEventListener('click', async () => {
    writeHash();
    try { await navigator.clipboard.writeText(location.href); $('status').textContent = 'Ссылка скопирована'; }
    catch { $('status').textContent = location.href; }
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { drawMap(); renderPoint(); });
}

main();
