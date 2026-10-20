// Интерфейс: карта M*(L, mu) в фоновых потоках, точные линии переходов, панель точки.
import { analyzePoint, findMinimum, gapG, omegaRel, phaseLabel, thresholds, vacuumMass, peak } from './njl3b0.js';

const G = -1.0;
const $ = (id) => document.getElementById(id);
const root = document.querySelector('.viz-root');

// ---------------------------------------------------------------------------
// Цвета из CSS-токенов (светлая/тёмная тема)
// ---------------------------------------------------------------------------
function tokens() {
  const cs = getComputedStyle(root);
  const v = (n) => cs.getPropertyValue(n).trim();
  const seq = v('--seq').split(',').map((s) => s.trim());
  return {
    surface: v('--surface-1'), page: v('--page'), ink: v('--text-primary'), ink2: v('--text-secondary'),
    muted: v('--text-muted'), grid: v('--grid'), axis: v('--axis'),
    s1: v('--series-1'), s2: v('--series-2'),
    seq: seq.map((c, i) => [i / (seq.length - 1), c]),
  };
}

function baseLayout(t, extra = {}) {
  return {
    paper_bgcolor: t.surface, plot_bgcolor: t.surface,
    font: { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', size: 12, color: t.ink2 },
    margin: { l: 56, r: 16, t: 8, b: 44 },
    hoverlabel: { bgcolor: t.surface, bordercolor: t.axis, font: { color: t.ink } },
    xaxis: { gridcolor: t.grid, linecolor: t.axis, zerolinecolor: t.axis, tickfont: { color: t.muted } },
    yaxis: { gridcolor: t.grid, linecolor: t.axis, zerolinecolor: t.axis, tickfont: { color: t.muted } },
    ...extra,
  };
}

const PLOT_CONFIG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                      toImageButtonOptions: { format: 'png', scale: 2, filename: 'njl3_b0' } };

// ---------------------------------------------------------------------------
// Состояние и ссылка
// ---------------------------------------------------------------------------
const state = { muLo: 0.05, muHi: 7, LLo: 0.1, LHi: 10, res: 160, vmax: 4, lines: true, thr: false, ptMu: 3.15, ptL: 3.5, theme: 'auto' };

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
function syncInputs() {
  $('mu-lo').value = state.muLo; $('mu-hi').value = state.muHi; $('L-lo').value = state.LLo; $('L-hi').value = state.LHi;
  $('res').value = String(state.res); $('vmax').value = state.vmax; $('show-lines').checked = state.lines;
  $('show-thr').checked = state.thr; $('pt-mu').value = state.ptMu; $('pt-L').value = state.ptL;
  $('theme').value = state.theme;
}

function applyTheme() {
  if (state.theme === 'auto') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = state.theme;
}
function readInputs() {
  state.muLo = Number($('mu-lo').value); state.muHi = Number($('mu-hi').value);
  state.LLo = Math.max(0.01, Number($('L-lo').value)); state.LHi = Number($('L-hi').value);
  state.res = Number($('res').value); state.vmax = Number($('vmax').value);
  state.lines = $('show-lines').checked; state.thr = $('show-thr').checked;
}

// ---------------------------------------------------------------------------
// Данные: точные линии (Python, njl3-b0 lines)
// ---------------------------------------------------------------------------
let LINES = null;
async function loadLines() {
  try {
    const r = await fetch('data/phase_lines.json');
    LINES = await r.json();
  } catch (e) {
    LINES = null;
    $('status').textContent = 'Не удалось загрузить линии переходов (data/phase_lines.json).';
  }
}

const PHASE_NAME = { V: 'вакуумная', T: 'тривиальная' };
const phaseText = (p) => (p[0] === 'I' ? `промежуточная, заполнены моды n = 0…${p.slice(1)}` : PHASE_NAME[p]);
const arrow = (key) => key.replace('-', ' → ');

function lineTraces(t) {
  if (!LINES || !state.lines) return [];
  const groups = { first: { x: [], y: [], text: [] }, second: { x: [], y: [], text: [] } };
  const dL = (LINES.L_range[1] - LINES.L_range[0]) / (LINES.nL - 1);
  for (const [key, pts] of Object.entries(LINES.lines)) {
    for (const order of [1, 2]) {
      const sel = pts.filter((p) => p[2] === order).sort((a, b) => a[0] - b[0]);
      const grp = order === 1 ? groups.first : groups.second;
      sel.forEach((p, i) => {
        if (i > 0 && p[0] - sel[i - 1][0] > 2.5 * dL) { grp.x.push(null); grp.y.push(null); grp.text.push(null); }
        grp.x.push(p[0]); grp.y.push(p[1]); grp.text.push(`${arrow(key)}, ${order}-й род`);
      });
      grp.x.push(null); grp.y.push(null); grp.text.push(null);
    }
  }
  const hov = 'L = %{x:.6f}<br>μ = %{y:.10f}<br>%{text}<extra></extra>';
  const out = [
    { type: 'scatter', mode: 'lines', name: 'переход 1-го рода', x: groups.first.x, y: groups.first.y, text: groups.first.text,
      line: { color: t.ink, width: 2 }, hovertemplate: hov, connectgaps: false },
    { type: 'scatter', mode: 'lines', name: 'переход 2-го рода', x: groups.second.x, y: groups.second.y, text: groups.second.text,
      line: { color: t.s2, width: 3, dash: 'dash' }, hovertemplate: hov, connectgaps: false },
  ];
  const sp = LINES.special;
  const pk = sp.filter((s) => s.kind === 'peak');
  const ce = sp.filter((s) => s.kind === 'critical_end' && s.mu != null);
  out.push({ type: 'scatter', mode: 'markers', name: 'пик μ = 2πn/L', x: pk.map((s) => s.L), y: pk.map((s) => s.mu),
             text: pk.map((s) => `пик n = ${s.n}`),
             marker: { symbol: 'triangle-up', size: 11, color: t.s2, line: { color: t.surface, width: 2 } },
             hovertemplate: '%{text}<br>L = %{x:.10f}<br>μ = %{y:.10f}<extra></extra>' });
  out.push({ type: 'scatter', mode: 'markers', name: 'критическая концевая точка', x: ce.map((s) => s.L), y: ce.map((s) => s.mu),
             marker: { symbol: 'circle', size: 10, color: t.ink, line: { color: t.surface, width: 2 } },
             hovertemplate: 'критическая концевая точка<br>L = %{x:.7f}<br>μ = %{y:.7f}<extra></extra>' });
  return out;
}

function thresholdTrace(t) {
  if (!state.thr) return [];
  const x = [], y = [];
  const Ls = Array.from({ length: 300 }, (_, i) => state.LLo + (state.LHi - state.LLo) * i / 299);
  for (let n = 1; n <= 60; n++) {
    let any = false;
    for (const L of Ls) {
      const mu = 2 * Math.PI * n / L;
      if (mu >= state.muLo && mu <= state.muHi) { any = true; x.push(L); y.push(mu); } else { x.push(null); y.push(null); }
    }
    x.push(null); y.push(null);
    if (!any && 2 * Math.PI * n / state.LHi > state.muHi) break;
  }
  return [{ type: 'scatter', mode: 'lines', name: 'порог моды μ = 2πn/L', x, y, hoverinfo: 'skip',
            line: { color: t.muted, width: 1, dash: 'dot' }, connectgaps: false }];
}

// ---------------------------------------------------------------------------
// Карта M*(L, mu)
// ---------------------------------------------------------------------------
let grid = null;           // { L, mu, M[mu][L], phase[mu][L], omega }
let jobId = 0;
let workers = [];

function linspace(a, b, n) { return Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1)); }

function heatmapTrace(t) {
  return {
    type: 'heatmap', name: 'M*', x: grid.L, y: grid.mu, z: grid.M, customdata: grid.phase,
    colorscale: t.seq, zmin: 0, zmax: state.vmax, zsmooth: false,
    colorbar: { title: { text: 'M*', side: 'right' }, thickness: 12, outlinewidth: 0, tickfont: { color: t.muted } },
    hovertemplate: 'L = %{x:.4f}<br>μ = %{y:.4f}<br>M* = %{z:.6f}<br>фаза %{customdata}<extra></extra>',
  };
}

function selectedTrace(t) {
  return { type: 'scatter', mode: 'markers', name: 'выбранная точка', x: [state.ptL], y: [state.ptMu], showlegend: false,
           marker: { symbol: 'circle-open', size: 14, color: t.ink, line: { width: 2.5, color: t.ink } }, hoverinfo: 'skip' };
}

function drawDiagram() {
  const t = tokens();
  const traces = [heatmapTrace(t), ...thresholdTrace(t), ...lineTraces(t), selectedTrace(t)];
  const layout = baseLayout(t, {
    xaxis: { ...baseLayout(t).xaxis, title: { text: 'L' }, range: [state.LLo, state.LHi] },
    yaxis: { ...baseLayout(t).yaxis, title: { text: 'μ' }, range: [state.muLo, state.muHi] },
    legend: { orientation: 'h', x: 0, y: 1.0, yanchor: 'bottom', bgcolor: 'rgba(0,0,0,0)', font: { color: t.ink2 } },
    margin: { l: 56, r: 16, t: 34, b: 44 },
  });
  Plotly.react('diagram', traces, layout, PLOT_CONFIG);
}

function buildGrid() {
  readInputs(); writeHash();
  if (!(state.muHi > state.muLo && state.LHi > state.LLo && state.LLo > 0)) { $('status').textContent = 'Проверьте диапазоны.'; return; }
  const n = state.res;
  if (lastPoint) drawSlices();                     // срезы — по диапазонам новой карты
  grid = { L: linspace(state.LLo, state.LHi, n), mu: linspace(state.muLo, state.muHi, n) };
  grid.M = grid.mu.map(() => new Array(n).fill(null));
  grid.phase = grid.mu.map(() => new Array(n).fill(''));
  grid.omega = grid.mu.map(() => new Array(n).fill(null));
  drawDiagram();

  workers.forEach((w) => w.terminate());
  const nw = Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 1));
  workers = Array.from({ length: nw }, () => new Worker(new URL('./worker.js', import.meta.url), { type: 'module' }));
  const job = ++jobId;
  let doneRows = 0, doneWorkers = 0, pending = false;
  const t0 = performance.now();
  $('build').disabled = true;
  const flush = () => {
    pending = false;
    Plotly.restyle('diagram', { z: [grid.M], customdata: [grid.phase] }, [0]);
  };
  workers.forEach((w, k) => {
    w.onmessage = ({ data }) => {
      if (data.job !== job) return;
      if (data.done) {
        if (++doneWorkers === nw) {
          flush();
          $('build').disabled = false;
          $('progress-bar').style.width = '0';
          $('status').textContent = `Карта ${n}×${n}: ${((performance.now() - t0) / 1000).toFixed(1)} с, ${nw} потоков.`;
        }
        return;
      }
      for (let j = 0; j < n; j++) { grid.M[j][data.i] = data.M[j]; grid.phase[j][data.i] = data.phase[j]; grid.omega[j][data.i] = data.omega[j]; }
      doneRows++;
      $('progress-bar').style.width = `${(100 * doneRows / n).toFixed(1)}%`;
      $('status').textContent = `Расчёт карты: ${doneRows} из ${n} столбцов по L…`;
      if (!pending) { pending = true; requestAnimationFrame(flush); }
    };
    const rows = [];
    for (let i = k; i < n; i += nw) rows.push({ i, L: grid.L[i] });
    w.postMessage({ job, rows, mu: grid.mu, g: G });
  });
}

// ---------------------------------------------------------------------------
// Панель точки
// ---------------------------------------------------------------------------
let lastPoint = null;
const fmt = (x, d = 10) => (x === 0 ? '0' : Number(x).toPrecision(d));

function showPoint(mu, L) {
  state.ptMu = mu; state.ptL = L; writeHash();
  $('pt-mu').value = mu; $('pt-L').value = L;
  const r = analyzePoint(mu, L, G);
  const Mv = vacuumMass(L, G);
  lastPoint = { ...r, M_vac: Mv };
  const rootsTxt = r.roots.length ? r.roots.map((q) => `${fmt(q.M, 10)} (${q.kind === 'min' ? 'мин' : q.kind === 'max' ? 'макс' : 'касание'})`).join('<br>') : '—';
  const minTxt = r.local_minima.length ? r.local_minima.map((m) => `${fmt(m.M, 8)} <span class="muted">(ΔΩ&nbsp;=&nbsp;${m.omega.toExponential(3)})</span>`).join('<br>') : '—';
  $('point-table').innerHTML = `
    <tr><th>Фаза</th><td><span class="phase-badge">${r.phase}</span> ${phaseText(r.phase)}</td></tr>
    <tr><th>M*</th><td>${fmt(r.M_star, 12)}</td></tr>
    <tr><th>Ω(M*) − Ω(0)</th><td>${r.omega_star === 0 ? '0' : r.omega_star.toExponential(6)}</td></tr>
    <tr><th>G(0)</th><td>${r.G0.toFixed(10)} — ${r.G0 > 0 ? 'M = 0 локально устойчив' : 'M = 0 неустойчив'}</td></tr>
    <tr><th>G₂</th><td>${Number.isFinite(r.G2) ? r.G2.toFixed(10) : '−∞ (порог моды)'}</td></tr>
    <tr><th>M<sub>vac</sub>(L)</th><td>${fmt(Mv, 12)}</td></tr>
    <tr><th>Корни G</th><td>${rootsTxt}</td></tr>
    <tr><th>Локальные минимумы</th><td>${minTxt}</td></tr>`;
  drawPointCharts(r, Mv);
  drawSlices();
  Plotly.restyle('diagram', { x: [[L]], y: [[mu]] }, [document.getElementById('diagram').data.length - 1]);
}

function drawPointCharts(r, Mv) {
  const t = tokens();
  const { mu, L } = r;
  const Ms = r.roots.map((q) => q.M);
  const Mmax = Math.max(0.5, 1.15 * Math.max(...Ms, mu, Math.min(Mv, 3 * mu + 3)));
  const xs = linspace(Mmax * 1e-4, Mmax, 400);
  const om = xs.map((M) => omegaRel(M, mu, L, G));
  const gg = xs.map((M) => gapG(M, mu, L, G));
  const thr = thresholds(mu, L).filter((s) => s < Mmax);
  const vlines = thr.map((s) => ({ type: 'line', x0: s, x1: s, yref: 'paper', y0: 0, y1: 1, line: { color: t.muted, width: 1, dash: 'dot' } }));
  const ax = (title) => ({ ...baseLayout(t).xaxis, title: { text: title } });

  const minima = r.local_minima;
  const glob = r.M_star === 0 ? { M: 0, omega: 0 } : minima.find((m) => m.M === r.M_star);
  Plotly.react('chart-omega', [
    { type: 'scatter', mode: 'lines', x: xs, y: om, line: { color: t.s1, width: 2 }, name: 'Ω(M) − Ω(0)',
      hovertemplate: 'M = %{x:.6f}<br>ΔΩ = %{y:.6e}<extra></extra>' },
    { type: 'scatter', mode: 'markers', x: minima.map((m) => m.M), y: minima.map((m) => m.omega), name: 'локальные минимумы',
      marker: { size: 8, color: t.surface, line: { color: t.ink, width: 2 } }, hovertemplate: 'минимум<br>M = %{x:.10f}<br>ΔΩ = %{y:.6e}<extra></extra>' },
    { type: 'scatter', mode: 'markers+text', x: [glob.M], y: [glob.omega], text: ['M*'], textposition: 'top center',
      textfont: { color: t.ink }, marker: { size: 10, color: t.ink, line: { color: t.surface, width: 2 } },
      hovertemplate: 'глобальный минимум<br>M* = %{x:.10f}<extra></extra>' },
  ], baseLayout(t, { showlegend: false, xaxis: ax('M'), shapes: vlines, margin: { l: 64, r: 10, t: 8, b: 40 } }), PLOT_CONFIG);

  const finite = gg.filter(Number.isFinite).sort((a, b) => a - b);
  const lo = finite[Math.floor(0.02 * finite.length)], hi = finite[Math.floor(0.98 * finite.length)];
  const pad = 0.15 * (hi - lo || 1);
  Plotly.react('chart-gap', [
    { type: 'scatter', mode: 'lines', x: xs, y: gg, line: { color: t.s1, width: 2 }, name: 'G(M)',
      hovertemplate: 'M = %{x:.6f}<br>G = %{y:.6f}<extra></extra>' },
    { type: 'scatter', mode: 'markers', x: r.roots.map((q) => q.M), y: r.roots.map(() => 0),
      text: r.roots.map((q) => (q.kind === 'min' ? 'минимум Ω' : q.kind === 'max' ? 'максимум Ω' : 'касание')),
      marker: { size: 8, color: r.roots.map((q) => (q.kind === 'max' ? t.surface : t.ink)), line: { color: t.ink, width: 2 } },
      hovertemplate: 'корень G: %{text}<br>M = %{x:.10f}<extra></extra>' },
  ], baseLayout(t, { showlegend: false, xaxis: ax('M'),
      yaxis: { ...baseLayout(t).yaxis, range: [Math.min(lo, 0) - pad, Math.max(hi, 0) + pad], zeroline: true, zerolinecolor: t.axis, zerolinewidth: 1.5 },
      shapes: vlines, margin: { l: 64, r: 10, t: 8, b: 40 } }), PLOT_CONFIG);
}

// ---------------------------------------------------------------------------
// Срезы через точку: M*(μ) при фиксированном L и M*(L) при фиксированном μ
// ---------------------------------------------------------------------------
const SLICE_N = 600;

// M* и фаза вдоль среза; at(x) -> [mu, L]
function sliceAt(at, x) {
  const [mu, L] = at(x);
  const M = findMinimum(mu, L, G).M;
  return { M, phase: phaseLabel(M, mu, L) };
}

// все смены фазы вдоль среза: бисекция до 1e-12 (как trace_L в Python), род — по скачку M*
function sliceTransitions(at, xs, pts) {
  const out = [];
  for (let k = 0; k + 1 < xs.length; k++) {
    let a = xs[k], pa = pts[k].phase;
    const b0 = xs[k + 1], pb = pts[k + 1].phase;
    let guard = 0;
    while (pa !== pb && guard++ < 8) {
      let lo = a, hi = b0;
      while (hi - lo > 1e-12 * Math.max(1, Math.abs(hi))) {
        const c = 0.5 * (lo + hi);
        if (sliceAt(at, c).phase === pa) lo = c; else hi = c;
      }
      const eps = 1e-9 * Math.max(1, Math.abs(hi));
      const left = sliceAt(at, lo - eps), right = sliceAt(at, hi + eps);
      const order = Math.abs(left.M - right.M) < 1e-2 ? 2 : 1;
      out.push({ x: 0.5 * (lo + hi), frm: pa, to: right.phase, order, Ml: left.M, Mr: right.M });
      a = hi + eps; pa = right.phase;
    }
  }
  return out;
}

function drawSlice(div, tableId, at, lo, hi, xname, xpoint, title) {
  const t = tokens();
  const xs = linspace(lo, hi, SLICE_N);
  const pts = xs.map((x) => sliceAt(at, x));
  const tr = sliceTransitions(at, xs, pts);
  // фаза на участках между переходами: подпись в середине участка
  const edges = [lo, ...tr.map((q) => q.x), hi];
  const labels = edges.slice(0, -1).map((e, i) => {
    const m = 0.5 * (e + edges[i + 1]);
    return { x: m, phase: sliceAt(at, m).phase };
  }).filter((q, i) => edges[i + 1] - edges[i] > 0.03 * (hi - lo));
  const Mmax = Math.max(...pts.map((p) => p.M), 0.1);
  const own = sliceAt(at, xpoint);
  Plotly.react(div, [
    { type: 'scatter', mode: 'lines', x: xs, y: pts.map((p) => p.M), customdata: pts.map((p) => p.phase),
      line: { color: t.s1, width: 2, shape: 'linear' }, connectgaps: false,
      hovertemplate: `${xname} = %{x:.6f}<br>M* = %{y:.8f}<br>фаза %{customdata}<extra></extra>` },
    { type: 'scatter', mode: 'markers', x: [xpoint], y: [own.M], hoverinfo: 'skip',
      marker: { symbol: 'circle-open', size: 12, color: t.ink, line: { width: 2.5, color: t.ink } } },
  ], baseLayout(t, {
    showlegend: false,
    xaxis: { ...baseLayout(t).xaxis, title: { text: xname }, range: [lo, hi] },
    yaxis: { ...baseLayout(t).yaxis, title: { text: 'M*' }, range: [0, 1.12 * Mmax] },
    shapes: tr.map((q) => ({ type: 'line', x0: q.x, x1: q.x, yref: 'paper', y0: 0, y1: 1,
      line: { color: q.order === 2 ? t.s2 : t.muted, width: 1.2, dash: q.order === 2 ? 'dash' : 'dot' } })),
    annotations: labels.map((q) => ({ x: q.x, y: 1.02, yref: 'paper', text: q.phase, showarrow: false,
      font: { color: t.ink2, size: 12 } })),
    margin: { l: 56, r: 10, t: 22, b: 44 },
  }), PLOT_CONFIG);
  $(div === 'chart-slice-mu' ? 'slice-mu-title' : 'slice-L-title').textContent = title;
  $(tableId).innerHTML = tr.length
    ? `<tr><th>${xname}</th><th>переход</th><th>род</th><th>M* слева → справа</th></tr>` +
      tr.map((q) => `<tr><td>${q.x.toFixed(10)}</td><td>${q.frm} → ${q.to}</td><td>${q.order === 2 ? '2-й' : '1-й'}</td>` +
        `<td>${fmt(q.Ml, 6)} → ${fmt(q.Mr, 6)}</td></tr>`).join('')
    : '<tr><td>переходов на срезе нет</td></tr>';
}

function drawSlices() {
  const { ptMu: mu, ptL: L } = state;
  drawSlice('chart-slice-mu', 'slice-mu-table', (x) => [x, L], state.muLo, state.muHi, 'μ', mu,
            `M*(μ) при L = ${L}`);
  drawSlice('chart-slice-L', 'slice-L-table', (x) => [mu, x], Math.max(state.LLo, 0.01), state.LHi, 'L', L,
            `M*(L) при μ = ${mu}`);
}

// ---------------------------------------------------------------------------
// Экспорт
// ---------------------------------------------------------------------------
function download(name, text, type) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function exportCSV() {
  if (!grid) return;
  const rows = ['L,mu,M_star,omega_star,phase'];
  grid.L.forEach((L, i) => grid.mu.forEach((mu, j) => {
    if (grid.M[j][i] !== null) rows.push(`${L},${mu},${grid.M[j][i]},${grid.omega[j][i]},${grid.phase[j][i]}`);
  }));
  download(`njl3_b0_map_mu${state.muLo}-${state.muHi}_L${state.LLo}-${state.LHi}.csv`, rows.join('\n'), 'text/csv');
}

// ---------------------------------------------------------------------------
// Запуск
// ---------------------------------------------------------------------------
async function main() {
  readHash(); syncInputs(); applyTheme();
  await loadLines();
  buildGrid();
  showPoint(state.ptMu, state.ptL);

  $('build').addEventListener('click', buildGrid);
  document.querySelectorAll('[data-preset]').forEach((b) => b.addEventListener('click', () => {
    const [a, c, d, e] = b.dataset.preset.split(',').map(Number);
    $('mu-lo').value = a; $('mu-hi').value = c; $('L-lo').value = d; $('L-hi').value = e;
    buildGrid();
  }));
  for (const id of ['show-lines', 'show-thr']) $(id).addEventListener('change', () => { readInputs(); writeHash(); drawDiagram(); });
  $('vmax').addEventListener('change', () => { readInputs(); writeHash(); Plotly.restyle('diagram', { zmax: [state.vmax] }, [0]); });
  $('point-form').addEventListener('submit', (e) => { e.preventDefault(); showPoint(Number($('pt-mu').value), Math.max(0.01, Number($('pt-L').value))); });
  $('diagram').on('plotly_click', (ev) => { const p = ev.points[0]; if (p && p.x != null && p.y != null) showPoint(p.y, p.x); });
  $('dl-csv').addEventListener('click', exportCSV);
  $('dl-lines').addEventListener('click', () => LINES && download('njl3_b0_phase_lines.json', JSON.stringify(LINES), 'application/json'));
  $('dl-point').addEventListener('click', () => lastPoint && download(`njl3_b0_point_mu${state.ptMu}_L${state.ptL}.json`, JSON.stringify(lastPoint, null, 1), 'application/json'));
  $('copy-link').addEventListener('click', async () => {
    writeHash();
    try { await navigator.clipboard.writeText(location.href); $('status').textContent = 'Ссылка скопирована.'; }
    catch { $('status').textContent = location.href; }
  });
  const redraw = () => { drawDiagram(); if (lastPoint) { drawPointCharts(lastPoint, lastPoint.M_vac); drawSlices(); } };
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redraw);
  $('theme').addEventListener('change', () => { state.theme = $('theme').value; writeHash(); applyTheme(); redraw(); });
}

main();
