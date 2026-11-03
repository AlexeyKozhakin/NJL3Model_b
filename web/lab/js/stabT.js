// Вкладка «Устойчивость (μ, T)»: устойчивость однородного решения при T > 0 (посчитано заранее,
// notebooks/93_stabT_muT.py; записка note_stab_T). В каждой точке — однородный минимум M(μ, L, T) формулы (A) и
// min по (k, q₁) функции Φ_full = ½ G_T(M) + Φ_T(k, q₁; M) (M ≠ 0: G_T = 0; M = 0: ½ G_T(0⁺) + петля).
const $$ = (id) => document.getElementById(id);
const $ = (id) => $$('w-' + id);
const root = document.querySelector('.viz-root');
const LS = [0.1, 0.3, 0.5, 1, 3];
const C0 = 0.01;                       // симметричный логарифм: z = sign(Φ) log10(1 + |Φ|/C0)
// устойчивость × фаза: V, I, T устойчиво; V, I, T неустойчиво
const STAB_COLORS = ['#2166ac', '#67a9cf', '#bdbcb6', '#67001f', '#d6604d', '#f4a582'];
const STAB_LABELS = ['V, устойчиво', 'I, устойчиво', 'T, устойчиво', 'V, неустойчиво', 'I, неустойчиво', 'T, неустойчиво'];

function tokens() {
  const cs = getComputedStyle(root);
  const v = (n) => cs.getPropertyValue(n).trim();
  const dark = cs.colorScheme === 'dark' || v('--surface-1').toLowerCase() === '#1a1a19';
  return { dark, surface: v('--surface-1'), ink: v('--text-primary'), ink2: v('--text-secondary'), muted: v('--text-muted'),
           grid: v('--grid'), axis: v('--axis') };
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
              toImageButtonOptions: { format: 'png', scale: 2, filename: 'njl3_stabT' } };

const state = { L: 1, logT: false, view: 'phi', bnd: true, zero: true, mu: 4.0, T: 0.05 };
function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  for (const k of Object.keys(state)) if (p.has('w_' + k)) {
    const d = state[k], v = p.get('w_' + k);
    state[k] = typeof d === 'boolean' ? v === '1' : typeof d === 'number' ? Number(v) : v;
  }
  if (!LS.includes(state.L)) state.L = 1;
  if (!['phi', 'stab'].includes(state.view)) state.view = 'phi';
}
function writeHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  for (const [k, v] of Object.entries(state)) p.set('w_' + k, typeof v === 'boolean' ? (v ? '1' : '0') : String(v));
  history.replaceState(null, '', '#' + p.toString());
}

const cache = {};
let D = null;
async function load() {
  const u = `data/stabT/stabT_L${state.L}_phi0_g-1.json`;
  if (!cache[u]) { const r = await fetch(u); if (!r.ok) throw new Error('нет карты'); cache[u] = await r.json(); }
  return cache[u];
}
const cls = (p) => (p === 'V' ? 'V' : p[0] === 'I' ? 'I' : 'T');
const CODE = { V: 0, I: 1, T: 2 };
const slog = (v) => (v === null ? null : Math.sign(v) * Math.log10(1 + Math.abs(v) / C0));
const fmt = (v, d = 5) => (v === null || v === undefined ? '—' : Number(v).toPrecision(d));

// отрезки между соседними клетками, у которых различается key(i, j) (x = μ, y = T); D.*[i = T][j = μ]
function edges(key) {
  const { mu, T } = D, x = [], y = [];
  const mid = (a, i) => 0.5 * (a[i] + a[i + 1]);
  const midT = (i) => (state.logT ? Math.sqrt(T[i] * T[i + 1]) : mid(T, i));
  const ex = (j) => (j <= 0 ? mu[0] : j >= mu.length ? mu.at(-1) : mid(mu, j - 1));
  const ey = (i) => (i <= 0 ? T[0] : i >= T.length ? T.at(-1) : midT(i - 1));
  for (let i = 0; i < T.length; i++) for (let j = 0; j < mu.length; j++) {
    const p = key(i, j);
    if (i + 1 < T.length && p !== key(i + 1, j)) { const yy = midT(i); x.push(ex(j), ex(j + 1), null); y.push(yy, yy, null); }
    if (j + 1 < mu.length && p !== key(i, j + 1)) { const xx = mid(mu, j); x.push(xx, xx, null); y.push(ey(i), ey(i + 1), null); }
  }
  return { x, y };
}

function divergingScale(t) {
  const mid = t.dark ? '#3a3a38' : '#f2f1ec';
  return [[0, '#67001f'], [0.2, '#b2182b'], [0.4, '#ef8a62'], [0.5, mid], [0.6, '#67a9cf'], [0.8, '#2166ac'], [1, '#053061']];
}
function yaxis() {
  const T = D.T;
  return { title: { text: 'T' }, type: state.logT ? 'log' : 'linear',
           range: state.logT ? [Math.log10(T[0]), Math.log10(T.at(-1))] : [0, T.at(-1)], exponentformat: 'power' };
}

function draw() {
  if (!D) return;
  const t = tokens();
  const { mu, T, phase, M, phi_min: P, k, q1, interp } = D;
  const cd = phase.map((r, i) => r.map((p, j) =>
    `фаза ${p}${P[i][j] === null ? '' : P[i][j] < 0 ? ' — <b>неустойчиво</b>' : ' — устойчиво'}<br>M = ${fmt(M[i][j])}` +
    `<br>min Φ = ${fmt(P[i][j])}<br>при k = ${k[i][j] ?? '—'}, q₁ = ${fmt(q1[i][j], 4)}` +
    `${interp[i][j] ? '<br>(интерполяция)' : ''}`));
  const hov = 'μ = %{x:.3f}<br>T = %{y:.4g}<br>%{customdata}<extra></extra>';
  const over = [];
  if (state.bnd) {
    const b = edges((i, j) => cls(phase[i][j]));
    over.push({ type: 'scatter', mode: 'lines', x: b.x, y: b.y, line: { color: t.muted, width: 1, dash: 'dot' },
                hoverinfo: 'skip', showlegend: false });
  }
  if (state.zero) {
    const z = edges((i, j) => (P[i][j] === null ? null : P[i][j] < 0));
    over.push({ type: 'scatter', mode: 'lines', x: z.x, y: z.y, line: { color: t.ink, width: 1.6 }, hoverinfo: 'skip',
                showlegend: false });
  }
  over.push({ type: 'scatter', mode: 'markers', x: [state.mu], y: [state.T], hoverinfo: 'skip', showlegend: false,
              marker: { symbol: 'circle-open', size: 13, color: t.ink, line: { width: 2.5, color: t.ink } } });
  let main;
  if (state.view === 'phi') {
    const z = P.map((r) => r.map(slog));
    const zm = Math.max(...z.flat().filter((v) => v !== null).map(Math.abs));
    const ticks = [-10, -1, -0.1, -0.01, 0, 0.01, 0.1, 1, 10].filter((v) => Math.abs(slog(v)) <= zm + 1e-9);
    main = { type: 'heatmap', x: mu, y: T, z, customdata: cd, hovertemplate: hov, zmin: -zm, zmax: zm, zsmooth: false,
             colorscale: divergingScale(t),
             colorbar: { title: { text: 'min Φ' }, thickness: 12, outlinewidth: 0, tickvals: ticks.map(slog),
                         ticktext: ticks.map(String), tickfont: { color: t.muted } } };
  } else {
    const z = phase.map((r, i) => r.map((p, j) => CODE[cls(p)] + (P[i][j] !== null && P[i][j] < 0 ? 3 : 0)));
    const cs = [];
    STAB_COLORS.forEach((c, n) => { cs.push([n / 6, c], [(n + 1) / 6, c]); });
    main = { type: 'heatmap', x: mu, y: T, z, customdata: cd, hovertemplate: hov, zmin: -0.5, zmax: 5.5, zsmooth: false,
             colorscale: cs,
             colorbar: { thickness: 12, outlinewidth: 0, tickvals: [0, 1, 2, 3, 4, 5], ticktext: STAB_LABELS,
                         tickfont: { color: t.muted, size: 10 } } };
  }
  Plotly.react($('map'), [main, ...over],
    baseLayout(t, { xaxis: { title: { text: 'μ' }, range: [mu[0], mu.at(-1)] }, yaxis: yaxis(), margin: { l: 56, r: 8, t: 6, b: 44 } }),
    CFG);
  drawSlices();
}

// ---------------------------------------------------------------------------
// Срезы через точку (из данных карты) и таблица
// ---------------------------------------------------------------------------
const nearest = (arr, v) => arr.reduce((kk, x, i) => (Math.abs(x - v) < Math.abs(arr[kk] - v) ? i : kk), 0);

function sliceTraces(xs, ys, phases, name, t) {
  // по одной линии на фазу; точки неустойчивости — красные маркеры
  const out = [];
  const PC = { V: '#2166ac', I: '#67a9cf', T: t.muted };
  for (const c of ['V', 'I', 'T']) {
    const y = ys.map((v, n) => (cls(phases[n]) === c || (n > 0 && cls(phases[n - 1]) === c) ? v : null));
    if (y.some((v, n) => v !== null && cls(phases[n]) === c))
      out.push({ type: 'scatter', mode: 'lines', x: xs, y, name: c, line: { color: PC[c], width: 2.2 }, connectgaps: false,
                 hovertemplate: `${name} = %{x:.4g}<br>min Φ = %{y:.5g}<extra>${c}</extra>` });
  }
  return out;
}
function drawSlices() {
  if (!D) return;
  const t = tokens();
  const { mu, T, phase, phi_min: P } = D;
  const i = nearest(T, state.T), j = nearest(mu, state.mu);
  const lay = (xt, extra = {}) => baseLayout(t, { xaxis: { title: { text: xt }, ...extra }, margin: { l: 56, r: 8, t: 6, b: 44 },
    yaxis: { title: { text: 'min Φ' }, type: 'linear' }, legend: { orientation: 'h', y: 1.12, font: { color: t.muted } },
    shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0, y1: 0, line: { color: t.axis, width: 1 } }] });
  $('sT-title').textContent = `min Φ(T) при μ = ${mu[j].toFixed(3)}`;
  $('smu-title').textContent = `min Φ(μ) при T = ${T[i].toPrecision(3)}`;
  Plotly.react($('sT'), sliceTraces(T, T.map((_, ii) => P[ii][j]), T.map((_, ii) => phase[ii][j]), 'T', t),
    lay('T', { type: state.logT ? 'log' : 'linear', exponentformat: 'power' }), CFG);
  Plotly.react($('smu'), sliceTraces(mu, P[i], phase[i], 'μ', t), lay('μ'), CFG);
  const row = (a, b) => `<tr><th>${a}</th><td>${b}</td></tr>`;
  const p = P[i][j];
  $('table').innerHTML = row('ближайший узел', `μ = ${mu[j].toFixed(4)}, T = ${T[i].toPrecision(4)}`) +
    row('фаза однородного минимума', phase[i][j]) + row('M(μ, L, T)', fmt(D.M[i][j], 6)) +
    row('min<sub>k,q₁</sub> Φ', `${fmt(p, 6)} — ${p < 0 ? '<b>неустойчиво</b>' : 'устойчиво'}`) +
    row('минимум при', `k = ${D.k[i][j] ?? '—'}, q₁ = ${fmt(D.q1[i][j], 5)}`) +
    row('значение', D.interp[i][j] ? 'интерполяция (внутри однородной ячейки)' : 'посчитано');
}

function syncSeg() {
  document.querySelectorAll('[data-wl]').forEach((b) => b.classList.toggle('on', Number(b.dataset.wl) === state.L));
  document.querySelectorAll('[data-wscale]').forEach((b) => b.classList.toggle('on', (b.dataset.wscale === 'log') === state.logT));
  document.querySelectorAll('[data-wview]').forEach((b) => b.classList.toggle('on', b.dataset.wview === state.view));
  $('map-title').textContent = state.view === 'phi'
    ? 'min по (k, q₁) функции Φ (синее > 0 — устойчиво, красное < 0 — неустойчиво; шкала — симметричный логарифм)'
    : 'Устойчивость однородного минимума × его фаза';
}
async function reload() {
  syncSeg(); writeHash();
  try {
    D = await load();
    $('status').textContent = `L = ${state.L}, φ = 0, g = −1. Карта ${D.mu.length} × ${D.T.length} (μ ∈ [${D.mu[0]}, ${D.mu.at(-1)}], ` +
      `T ∈ [${D.T[0]}, ${D.T.at(-1)}], сетка по T логарифмическая); у границ фаз, линии Φ = 0 и резких перепадов — ` +
      'уточнение, внутри — интерполяция.';
  } catch {
    D = null;
    $('status').textContent = `Карта для L = ${state.L} не найдена.`;
  }
  draw();
}

let started = false;
export async function initStabT() {
  if (started) { window.dispatchEvent(new Event('resize')); return; }
  started = true;
  readHash();
  $('show-bnd').checked = state.bnd;
  $('show-zero').checked = state.zero;
  await reload();
  document.querySelectorAll('[data-wl]').forEach((b) => b.addEventListener('click', async () => {
    state.L = Number(b.dataset.wl); await reload();
  }));
  document.querySelectorAll('[data-wscale]').forEach((b) => b.addEventListener('click', () => {
    state.logT = b.dataset.wscale === 'log'; syncSeg(); writeHash(); draw();
  }));
  document.querySelectorAll('[data-wview]').forEach((b) => b.addEventListener('click', () => {
    state.view = b.dataset.wview; syncSeg(); writeHash(); draw();
  }));
  $('show-bnd').addEventListener('change', () => { state.bnd = $('show-bnd').checked; writeHash(); draw(); });
  $('show-zero').addEventListener('change', () => { state.zero = $('show-zero').checked; writeHash(); draw(); });
  $('map').on('plotly_click', (ev) => {
    const p = ev.points[0];
    if (p && p.x != null && p.y != null) { state.mu = +p.x; state.T = +p.y; writeHash(); draw(); }
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
}
export function redrawStabT() { if (started) draw(); }
