// Лаборатория: вкладка «Однородный случай» (любые φ, g) и вкладка «Устойчивость» (web/stab во фрейме).
import { analyzePoint, findMinimum, gapG, levels, omegaRel, phaseLabel, reducePhi, thresholds, vacuumMass } from './njl3hom.js';

const $ = (id) => document.getElementById(id);
const root = document.querySelector('.viz-root');

function tokens() {
  const cs = getComputedStyle(root);
  const v = (n) => cs.getPropertyValue(n).trim();
  const seq = v('--seq').split(',').map((s) => s.trim());
  return {
    surface: v('--surface-1'), ink: v('--text-primary'), ink2: v('--text-secondary'), muted: v('--text-muted'),
    grid: v('--grid'), axis: v('--axis'), s1: v('--series-1'), s2: v('--series-2'),
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
                      toImageButtonOptions: { format: 'png', scale: 2, filename: 'njl3_lab' } };

// ---------------------------------------------------------------------------
// Состояние, ссылка, вкладки
// ---------------------------------------------------------------------------
const state = { tab: 'hom', phi: 0, g: -1, muLo: 0.05, muHi: 7, LLo: 0.1, LHi: 10, res: 160, vmax: 4,
                lines: true, thr: false, ptMu: 3.15, ptL: 3.5, theme: 'auto' };

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  for (const k of Object.keys(state)) {
    if (!p.has(k)) continue;
    const d = state[k];
    state[k] = typeof d === 'boolean' ? p.get(k) === '1' : typeof d === 'string' ? p.get(k) : Number(p.get(k));
  }
}
function writeHash() {
  const p = new URLSearchParams(location.hash.slice(1));          // параметры другой вкладки (x_…) сохраняются
  for (const [k, v] of Object.entries(state)) p.set(k, typeof v === 'boolean' ? (v ? '1' : '0') : String(v));
  history.replaceState(null, '', '#' + p.toString());
}
function syncInputs() {
  $('phi').value = state.phi; $('g').value = String(state.g);
  $('mu-lo').value = state.muLo; $('mu-hi').value = state.muHi; $('L-lo').value = state.LLo; $('L-hi').value = state.LHi;
  $('res').value = String(state.res); $('vmax').value = state.vmax; $('show-lines').checked = state.lines;
  $('show-thr').checked = state.thr; $('pt-mu').value = state.ptMu; $('pt-L').value = state.ptL; $('theme').value = state.theme;
}
function readInputs() {
  state.phi = Number($('phi').value) || 0; state.g = Number($('g').value);
  state.muLo = Number($('mu-lo').value); state.muHi = Number($('mu-hi').value);
  state.LLo = Math.max(0.01, Number($('L-lo').value)); state.LHi = Number($('L-hi').value);
  state.res = Number($('res').value); state.vmax = Number($('vmax').value);
  state.lines = $('show-lines').checked; state.thr = $('show-thr').checked;
}
function applyTheme() {
  if (state.theme === 'auto') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = state.theme;
}
function showTab(name) {
  state.tab = name; writeHash();
  for (const b of document.querySelectorAll('[data-tab]')) b.classList.toggle('active', b.dataset.tab === name);
  $('tab-hom').hidden = name !== 'hom';
  $('tab-inh').hidden = name !== 'inh';
  $('tab-stab').hidden = name !== 'stab';
  if (name === 'inh') import('./inh.js').then((m) => m.initInh());
  if (name === 'stab' && !$('stab-frame').src) $('stab-frame').src = '../stab/';
  if (name === 'hom') window.dispatchEvent(new Event('resize'));
}

const P = () => ({ phi: reducePhi(state.phi), g: state.g });

// ---------------------------------------------------------------------------
// Карта
// ---------------------------------------------------------------------------
let grid = null, jobId = 0, workers = [];
function linspace(a, b, n) { return Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1)); }

// границы фаз по сетке: отрезки между соседними клетками с разными метками
function boundaryTrace(t) {
  if (!grid || !state.lines) return [];
  const x = [], y = [], { L, mu, phase } = grid, nL = L.length, nm = mu.length;
  for (let j = 0; j < nm; j++) for (let i = 0; i < nL; i++) {
    const p = phase[j][i];
    if (!p) continue;
    if (i + 1 < nL && phase[j][i + 1] && phase[j][i + 1] !== p) {
      const xm = 0.5 * (L[i] + L[i + 1]);
      const y0 = j > 0 ? 0.5 * (mu[j - 1] + mu[j]) : mu[j], y1 = j + 1 < nm ? 0.5 * (mu[j] + mu[j + 1]) : mu[j];
      x.push(xm, xm, null); y.push(y0, y1, null);
    }
    if (j + 1 < nm && phase[j + 1][i] && phase[j + 1][i] !== p) {
      const ym = 0.5 * (mu[j] + mu[j + 1]);
      const x0 = i > 0 ? 0.5 * (L[i - 1] + L[i]) : L[i], x1 = i + 1 < nL ? 0.5 * (L[i] + L[i + 1]) : L[i];
      x.push(x0, x1, null); y.push(ym, ym, null);
    }
  }
  return [{ type: 'scatter', mode: 'lines', name: 'граница фаз', x, y, hoverinfo: 'skip', connectgaps: false,
            line: { color: t.ink, width: 1.2 } }];
}

function thresholdTrace(t) {
  if (!state.thr) return [];
  const { phi } = P(), x = [], y = [];
  const Ls = linspace(state.LLo, state.LHi, 300);
  for (let n = -60; n <= 60; n++) {
    const c = 2 * Math.PI * Math.abs(n + phi);
    if (c === 0) continue;
    if (c / state.LHi > state.muHi) continue;
    for (const L of Ls) { const m = c / L; if (m >= state.muLo && m <= state.muHi) { x.push(L); y.push(m); } else { x.push(null); y.push(null); } }
    x.push(null); y.push(null);
  }
  return [{ type: 'scatter', mode: 'lines', name: 'порог моды μ = |k_n|', x, y, hoverinfo: 'skip',
            line: { color: t.muted, width: 1, dash: 'dot' }, connectgaps: false }];
}

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
  const traces = [heatmapTrace(t), ...thresholdTrace(t), ...boundaryTrace(t), selectedTrace(t)];
  const { phi, g } = P();
  Plotly.react('diagram', traces, baseLayout(t, {
    xaxis: { ...baseLayout(t).xaxis, title: { text: 'L' }, range: [state.LLo, state.LHi] },
    yaxis: { ...baseLayout(t).yaxis, title: { text: 'μ' }, range: [state.muLo, state.muHi] },
    legend: { orientation: 'h', x: 0, y: 1.0, yanchor: 'bottom', bgcolor: 'rgba(0,0,0,0)', font: { color: t.ink2 } },
    title: { text: `φ = ${+phi.toFixed(6)}, g = ${g}`, font: { size: 13, color: t.ink2 }, x: 0.99, xanchor: 'right', y: 0.995 },
    margin: { l: 56, r: 16, t: 34, b: 44 },
  }), PLOT_CONFIG);
}

function buildGrid() {
  readInputs(); writeHash();
  if (!(state.muHi > state.muLo && state.LHi > state.LLo && state.LLo > 0)) { $('status').textContent = 'Проверьте диапазоны.'; return; }
  const n = state.res, { phi, g } = P();
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
  const flush = () => { pending = false; Plotly.restyle('diagram', { z: [grid.M], customdata: [grid.phase] }, [0]); };
  workers.forEach((w, k) => {
    w.onmessage = ({ data }) => {
      if (data.job !== job) return;
      if (data.done) {
        if (++doneWorkers === nw) {
          drawDiagram();
          $('build').disabled = false;
          $('progress-bar').style.width = '0';
          $('status').textContent = `Карта ${n}×${n} (φ = ${+phi.toFixed(6)}, g = ${g}): ${((performance.now() - t0) / 1000).toFixed(1)} с, ${nw} потоков.`;
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
    w.postMessage({ job, rows, mu: grid.mu, g, phi });
  });
  if (lastPoint) showPoint(state.ptMu, state.ptL);
}

// ---------------------------------------------------------------------------
// Точка
// ---------------------------------------------------------------------------
let lastPoint = null;
const fmt = (x, d = 10) => (x === 0 ? '0' : Number(x).toPrecision(d));
function phaseText(p) {
  if (p === 'T') return 'тривиальная (M* = 0)';
  if (p === 'V') return 'вакуумная (ферми-море пусто)';
  return `промежуточная, заполнены уровни |k| 0…${p.slice(1)}`;
}

function showPoint(mu, L) {
  state.ptMu = mu; state.ptL = L; writeHash();
  $('pt-mu').value = mu; $('pt-L').value = L;
  const { phi, g } = P();
  const r = analyzePoint(mu, L, g, phi);
  const Mv = vacuumMass(L, g, phi);
  lastPoint = { ...r, M_vac: Mv };
  const lv = levels(mu, L, phi);
  const rootsTxt = r.roots.length ? r.roots.map((q) => `${fmt(q.M, 10)} (${q.kind === 'min' ? 'мин' : q.kind === 'max' ? 'макс' : 'касание'})`).join('<br>') : '—';
  const minTxt = r.local_minima.length ? r.local_minima.map((m) => `${fmt(m.M, 8)} <span class="muted">(ΔΩ&nbsp;=&nbsp;${m.omega.toExponential(3)})</span>`).join('<br>') : '—';
  $('point-table').innerHTML = `
    <tr><th>φ, g</th><td>${+phi.toFixed(6)}, ${g}</td></tr>
    <tr><th>Фаза</th><td><span class="phase-badge">${r.phase}</span> ${phaseText(r.phase)}</td></tr>
    <tr><th>M*</th><td>${fmt(r.M_star, 12)}</td></tr>
    <tr><th>Ω(M*) − Ω(0)</th><td>${r.omega_star === 0 ? '0' : r.omega_star.toExponential(6)}</td></tr>
    <tr><th>G(0)</th><td>${Number.isFinite(r.G0) ? r.G0.toFixed(10) : r.G0} — ${r.G0 > 0 ? 'M = 0 локально устойчив' : 'M = 0 неустойчив'}</td></tr>
    <tr><th>M<sub>vac</sub>(L)</th><td>${fmt(Mv, 12)}</td></tr>
    <tr><th>Уровни |k<sub>n</sub>| &lt; μ</th><td>${lv.length ? lv.slice(0, 6).map((k) => k.toFixed(4)).join(', ') + (lv.length > 6 ? '…' : '') : 'нет'}</td></tr>
    <tr><th>Корни G</th><td>${rootsTxt}</td></tr>
    <tr><th>Локальные минимумы</th><td>${minTxt}</td></tr>`;
  drawPointCharts(r, Mv);
  drawSlices();
  const d = $('diagram');
  if (d.data) Plotly.restyle('diagram', { x: [[L]], y: [[mu]] }, [d.data.length - 1]);
}

function drawPointCharts(r, Mv) {
  const t = tokens(), { mu, L } = r, { phi, g } = P();
  const Ms = r.roots.map((q) => q.M);
  const Mmax = Math.max(0.5, 1.15 * Math.max(...Ms, mu, Math.min(Mv, 3 * mu + 3)));
  const xs = linspace(Mmax * 1e-4, Mmax, 400);
  const om = xs.map((M) => omegaRel(M, mu, L, g, phi));
  const gg = xs.map((M) => gapG(M, mu, L, g, phi));
  const vlines = thresholds(mu, L, phi).filter((s) => s < Mmax).map((s) => ({ type: 'line', x0: s, x1: s, yref: 'paper', y0: 0, y1: 1, line: { color: t.muted, width: 1, dash: 'dot' } }));
  const ax = (title) => ({ ...baseLayout(t).xaxis, title: { text: title } });
  const minima = r.local_minima;
  const glob = r.M_star === 0 ? { M: 0, omega: 0 } : minima.find((m) => m.M === r.M_star);
  Plotly.react('chart-omega', [
    { type: 'scatter', mode: 'lines', x: xs, y: om, line: { color: t.s1, width: 2 }, hovertemplate: 'M = %{x:.6f}<br>ΔΩ = %{y:.6e}<extra></extra>' },
    { type: 'scatter', mode: 'markers', x: minima.map((m) => m.M), y: minima.map((m) => m.omega),
      marker: { size: 8, color: t.surface, line: { color: t.ink, width: 2 } }, hovertemplate: 'минимум<br>M = %{x:.10f}<br>ΔΩ = %{y:.6e}<extra></extra>' },
    { type: 'scatter', mode: 'markers+text', x: [glob.M], y: [glob.omega], text: ['M*'], textposition: 'top center',
      textfont: { color: t.ink }, marker: { size: 10, color: t.ink, line: { color: t.surface, width: 2 } }, hovertemplate: 'M* = %{x:.10f}<extra></extra>' },
  ], baseLayout(t, { showlegend: false, xaxis: ax('M'), shapes: vlines, margin: { l: 64, r: 10, t: 8, b: 40 } }), PLOT_CONFIG);
  const finite = gg.filter(Number.isFinite).sort((a, b) => a - b);
  const lo = finite[Math.floor(0.02 * finite.length)], hi = finite[Math.floor(0.98 * finite.length)];
  const pad = 0.15 * (hi - lo || 1);
  Plotly.react('chart-gap', [
    { type: 'scatter', mode: 'lines', x: xs, y: gg, line: { color: t.s1, width: 2 }, hovertemplate: 'M = %{x:.6f}<br>G = %{y:.6f}<extra></extra>' },
    { type: 'scatter', mode: 'markers', x: r.roots.map((q) => q.M), y: r.roots.map(() => 0),
      marker: { size: 8, color: r.roots.map((q) => (q.kind === 'max' ? t.surface : t.ink)), line: { color: t.ink, width: 2 } },
      hovertemplate: 'корень G<br>M = %{x:.10f}<extra></extra>' },
  ], baseLayout(t, { showlegend: false, xaxis: ax('M'),
      yaxis: { ...baseLayout(t).yaxis, range: [Math.min(lo, 0) - pad, Math.max(hi, 0) + pad], zeroline: true, zerolinewidth: 1.5 },
      shapes: vlines, margin: { l: 64, r: 10, t: 8, b: 40 } }), PLOT_CONFIG);
}

// ---------------------------------------------------------------------------
// Срезы
// ---------------------------------------------------------------------------
const SLICE_N = 500;
function sliceAt(at, x) {
  const [mu, L] = at(x), { phi, g } = P();
  const M = findMinimum(mu, L, g, phi).M;
  return { M, phase: phaseLabel(M, mu, L, phi) };
}
function sliceTransitions(at, xs, pts) {
  const out = [];
  for (let k = 0; k + 1 < xs.length; k++) {
    let a = xs[k], pa = pts[k].phase;
    const b = xs[k + 1], pb = pts[k + 1].phase;
    let guard = 0;
    while (pa !== pb && guard++ < 8) {
      let lo = a, hi = b;
      while (hi - lo > 1e-12 * Math.max(1, Math.abs(hi))) {
        const c = 0.5 * (lo + hi);
        if (sliceAt(at, c).phase === pa) lo = c; else hi = c;
      }
      const eps = 1e-9 * Math.max(1, Math.abs(hi));
      const left = sliceAt(at, lo - eps), right = sliceAt(at, hi + eps);
      out.push({ x: 0.5 * (lo + hi), frm: pa, to: right.phase, order: Math.abs(left.M - right.M) < 1e-2 ? 2 : 1, Ml: left.M, Mr: right.M });
      a = hi + eps; pa = right.phase;
    }
  }
  return out;
}
function drawSlice(div, tableId, titleId, at, lo, hi, xname, xpoint, title) {
  const t = tokens();
  const xs = linspace(lo, hi, SLICE_N);
  const pts = xs.map((x) => sliceAt(at, x));
  const tr = sliceTransitions(at, xs, pts);
  const edges = [lo, ...tr.map((q) => q.x), hi];
  const labels = edges.slice(0, -1).map((e, i) => ({ x: 0.5 * (e + edges[i + 1]), phase: sliceAt(at, 0.5 * (e + edges[i + 1])).phase }))
    .filter((q, i) => edges[i + 1] - edges[i] > 0.03 * (hi - lo));
  const Mmax = Math.max(...pts.map((p) => p.M), 0.1);
  const own = sliceAt(at, xpoint);
  Plotly.react(div, [
    { type: 'scatter', mode: 'lines', x: xs, y: pts.map((p) => p.M), customdata: pts.map((p) => p.phase),
      line: { color: t.s1, width: 2 }, hovertemplate: `${xname} = %{x:.6f}<br>M* = %{y:.8f}<br>фаза %{customdata}<extra></extra>` },
    { type: 'scatter', mode: 'markers', x: [xpoint], y: [own.M], hoverinfo: 'skip',
      marker: { symbol: 'circle-open', size: 12, color: t.ink, line: { width: 2.5, color: t.ink } } },
  ], baseLayout(t, {
    showlegend: false,
    xaxis: { ...baseLayout(t).xaxis, title: { text: xname }, range: [lo, hi] },
    yaxis: { ...baseLayout(t).yaxis, title: { text: 'M*' }, range: [0, 1.12 * Mmax] },
    shapes: tr.map((q) => ({ type: 'line', x0: q.x, x1: q.x, yref: 'paper', y0: 0, y1: 1,
      line: { color: q.order === 2 ? t.s2 : t.muted, width: 1.2, dash: q.order === 2 ? 'dash' : 'dot' } })),
    annotations: labels.map((q) => ({ x: q.x, y: 1.02, yref: 'paper', text: q.phase, showarrow: false, font: { color: t.ink2, size: 12 } })),
    margin: { l: 56, r: 10, t: 22, b: 44 },
  }), PLOT_CONFIG);
  $(titleId).textContent = title;
  $(tableId).innerHTML = tr.length
    ? `<tr><th>${xname}</th><th>переход</th><th>род</th><th>M* слева → справа</th></tr>` +
      tr.map((q) => `<tr><td>${q.x.toFixed(10)}</td><td>${q.frm} → ${q.to}</td><td>${q.order === 2 ? '2-й' : '1-й'}</td><td>${fmt(q.Ml, 6)} → ${fmt(q.Mr, 6)}</td></tr>`).join('')
    : '<tr><td>переходов на срезе нет</td></tr>';
}
function drawSlices() {
  const { ptMu: mu, ptL: L } = state;
  drawSlice('chart-slice-mu', 'slice-mu-table', 'slice-mu-title', (x) => [x, L], state.muLo, state.muHi, 'μ', mu, `M*(μ) при L = ${L}`);
  drawSlice('chart-slice-L', 'slice-L-table', 'slice-L-title', (x) => [mu, x], Math.max(state.LLo, 0.01), state.LHi, 'L', L, `M*(L) при μ = ${mu}`);
}

// ---------------------------------------------------------------------------
// Экспорт и запуск
// ---------------------------------------------------------------------------
function download(name, text, type) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
function exportCSV() {
  if (!grid) return;
  const { phi, g } = P();
  const rows = ['phi,g,L,mu,M_star,omega_star,phase'];
  grid.L.forEach((L, i) => grid.mu.forEach((mu, j) => {
    if (grid.M[j][i] !== null) rows.push(`${phi},${g},${L},${mu},${grid.M[j][i]},${grid.omega[j][i]},${grid.phase[j][i]}`);
  }));
  download(`njl3_hom_phi${phi}_g${g}.csv`, rows.join('\n'), 'text/csv');
}

function main() {
  readHash(); syncInputs(); applyTheme();
  for (const b of document.querySelectorAll('[data-tab]')) b.addEventListener('click', () => showTab(b.dataset.tab));
  showTab(state.tab);
  buildGrid();
  showPoint(state.ptMu, state.ptL);
  $('build').addEventListener('click', buildGrid);
  $('g').addEventListener('change', buildGrid);
  $('phi').addEventListener('change', buildGrid);
  document.querySelectorAll('[data-phi]').forEach((b) => b.addEventListener('click', () => { $('phi').value = b.dataset.phi; buildGrid(); }));
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
  $('dl-point').addEventListener('click', () => lastPoint && download(`njl3_hom_point_mu${state.ptMu}_L${state.ptL}_phi${P().phi}_g${P().g}.json`, JSON.stringify(lastPoint, null, 1), 'application/json'));
  $('copy-link').addEventListener('click', async () => {
    writeHash();
    try { await navigator.clipboard.writeText(location.href); $('status').textContent = 'Ссылка скопирована.'; }
    catch { $('status').textContent = location.href; }
  });
  const redraw = () => { drawDiagram(); if (lastPoint) { drawPointCharts(lastPoint, lastPoint.M_vac); drawSlices(); } };
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redraw);
  $('theme').addEventListener('change', () => { state.theme = $('theme').value; writeHash(); applyTheme(); redraw();
    import('./inh.js').then((m) => m.redrawInh()); });
}

main();
