// Ω(M, b) на R² × S¹ при температуре T > 0 (формула (A)) и глобальный минимум — порт njl3/fast/thermal.py и
// njl3/fast/minimize_T.py.
//
//   Ω_T = M²/(2g) + ΔV + Ω_L|_{T=0} + W_T,   W_T(μ) = ∫_0^∞ [K_T(ν − μ) + K_T(ν + μ)] W_0(ν) dν,
//   K_T(x) = 1/(4T ch²(x/2T)) — часть с ферми-морем при T = 0 (omegaMuL), усреднённая по химическому потенциалу.
// Квадратура — двойная экспонента x = T sh(π/2 sh t) (узлы с весом ≲ 1e-19 отброшены), как в Python.
import { dU, omegaL, omegaMuL, thresholds, checkPhi, findMinimum as findMinimum0, nelderMead, OMEGA_NOISE, TC_RATIO }
  from './njl3inh.js';
import { findMinimum as homFindMinimum, phaseLabel as homPhaseLabel } from './njl3hom.js';

const N_DE = 81, T_MAX = 3.2;
const UX = [], UW = [];
{
  const h = 2 * T_MAX / (N_DE - 1);
  for (let i = 0; i < N_DE; i++) {
    const t = -T_MAX + i * h, s = Math.sinh(0.5 * Math.PI * Math.sinh(t));
    if (Math.abs(s) >= 45) continue;
    const ds = 0.5 * Math.PI * Math.cosh(t) * Math.cosh(0.5 * Math.PI * Math.sinh(t));
    UX.push(s);
    UW.push(ds * h / (4 * Math.cosh(Math.min(Math.abs(s) / 2, 350)) ** 2));
  }
}

const W0 = (M, b, nu, L, phi) => (nu > 0 ? omegaMuL(M, b, nu, L, phi) : 0);

/** Часть с ферми-морем при температуре T. */
export function WT(M, b, mu, L, phi, T) {
  let s = 0;
  for (let j = 0; j < UX.length; j++) {
    const x = T * UX[j];
    s += UW[j] * (W0(M, b, mu + x, L, phi) + W0(M, b, -mu + x, L, phi));
  }
  return s;
}

const ZERO = new Map();
function zeroTerms(mu, L, phi, T) {
  const key = [mu, L, phi, T].join(',');
  let v = ZERO.get(key);
  if (v === undefined) {
    if (ZERO.size > 4096) ZERO.clear();
    v = omegaL(0, 0, L, phi) + WT(0, 0, mu, L, phi, T);
    ZERO.set(key, v);
  }
  return v;
}

/** Ω_T(M, b) − Ω_T(0, 0). */
export function omegaRelT(M, b, mu, L, g, phi, T) {
  if (M === 0) return 0;
  if (b < 0) return 1e3;
  return M * M / (2 * g) + dU(M, b) + omegaL(M, b, L, phi) + WT(M, b, mu, L, phi, T) - zeroTerms(mu, L, phi, T);
}

// ---------------------------------------------------------------------------
// Глобальный минимум (как njl3/fast/minimize_T.py): кандидаты — минимумы при T = 0
// ---------------------------------------------------------------------------
// порог шума растёт с масштабом слагаемых Ω_T ~ (μ + T)³/L (округление), как njl3/fast/minimize_T.noise
const noise = (mu, L, T) => Math.max(OMEGA_NOISE, 1e-14 * (mu + T) ** 3 / L);
const DEAD = 3.0, LNM_LO = Math.log(1e-6), B_MIN = 1e-5;
const GM_B0 = Array.from({ length: 90 }, (_, i) => Math.exp(Math.log(1e-4) + (Math.log(8) - Math.log(1e-4)) * i / 89));

function boundedMin(f, a, b, tol = 1e-11) {
  // золотое сечение (функция в окне считается унимодальной)
  const r = (Math.sqrt(5) - 1) / 2;
  let x1 = b - r * (b - a), x2 = a + r * (b - a), f1 = f(x1), f2 = f(x2);
  while (b - a > tol) {
    if (f1 < f2) { b = x2; x2 = x1; f2 = f1; x1 = b - r * (b - a); f1 = f(x1); }
    else { a = x1; x1 = x2; f1 = f2; x2 = a + r * (b - a); f2 = f(x2); }
  }
  return f1 < f2 ? { x: x1, f: f1 } : { x: x2, f: f2 };
}

function homT(mu, L, g, phi, T, starts) {
  const f = (x) => omegaRelT(Math.exp(x), 0, mu, L, g, phi, T);
  const xg = GM_B0.map(Math.log), om = xg.map(f);
  const h = xg[1] - xg[0];
  const seeds = [];
  for (let i = 0; i < om.length; i++)
    if (om[i] < 0 && (i === 0 || om[i] <= om[i - 1]) && (i === om.length - 1 || om[i] <= om[i + 1])) seeds.push([xg[i], 1.5 * h]);
  for (const M of starts) if (M > 0) seeds.push([Math.log(M), 0.03]);
  let best = { M: 0, omega: 0 };
  for (const [x0, w] of seeds) {
    const r = boundedMin(f, x0 - w, x0 + w);
    if (r.f < best.omega) best = { M: Math.exp(r.x), omega: r.f };
  }
  return best.omega < -noise(mu, L, T) ? best : { M: 0, omega: 0 };
}

function channelT(mu, L, g, phi, T, S, M0, b0) {
  const f = ([x, t]) => {
    const M = Math.exp(Math.max(x, LNM_LO)), b = S + t * M;
    return b >= 0 ? omegaRelT(M, b, mu, L, g, phi, T) : 1e3;
  };
  const r = nelderMead(f, [Math.log(M0), (b0 - S) / M0], 0.3, { xatol: 1e-7, fatol: 1e-15, maxfev: 1600 });
  const lm = Math.max(r.x[0], LNM_LO), M = Math.exp(lm), b = S + r.x[1] * M;
  return r.f < -noise(mu, L, T) && lm > LNM_LO + 1e-6 && b > 1e-8 ? { M, b, omega: r.f } : { M: 0, b, omega: 0 };
}

/** Глобальный минимум Ω_T по (M, b) в точке (μ, L). extra — дополнительные старты [[M, b]] (продолжение вдоль
 *  линии: минимум в соседней точке); b = 0 — однородный старт. pre — {r0, hom} при T = 0 (зависят только от μ, L). */
export function zeroTCandidates(mu, L, g, phi) {
  return { r0: findMinimum0(mu, L, g, phi, 0), hom: homFindMinimum(mu, L, g, phi) };
}
export function findMinimumT(mu, L, g, phi = 0, T = 1e-3, extra = [], pre = null) {
  phi = checkPhi(phi);
  const { r0, hom } = pre || zeroTCandidates(mu, L, g, phi);
  const h = homT(mu, L, g, phi, T, [...hom.minima.map((m) => m.M), ...extra.filter(([M, b]) => M > 0 && !(b > 0)).map(([M]) => M)]);
  const cand = [{ phase: homPhaseLabel(h.M, mu, L, phi), M: h.M, b: 0, omega: h.omega }];
  const starts = [];
  for (const ch of r0.channels)
    if (ch.M > 0 && ch.b > 1e-8 && T <= DEAD * TC_RATIO * (ch.S / mu) * ch.M) starts.push([ch.n, ch.S, ch.M, ch.b]);
  if (r0.phase[0] === 'B' && r0.M_star > 0) {
    const ch = r0.channels.reduce((a, c) => (Math.abs(c.S - r0.b_star) < Math.abs(a.S - r0.b_star) ? c : a));
    if (!starts.some(([, S, M]) => Math.abs(S - ch.S) < 1e-12 && Math.abs(M - r0.M_star) < 1e-9))
      starts.push([ch.n, ch.S, r0.M_star, r0.b_star]);
  }
  for (const [M, b] of extra)
    if (M > 0 && b > 1e-8 && r0.channels.length) {
      const ch = r0.channels.reduce((a, c) => (Math.abs(c.S - b) < Math.abs(a.S - b) ? c : a));
      starts.push([ch.n, ch.S, M, b]);
    }
  const channels = r0.channels.map((c) => ({ n: c.n, S: c.S, M0: c.M, b0: c.b, M: 0, b: c.S, omega: 0, done: false }));
  for (const [n, S, M0, b0] of starts) {
    const c = channelT(mu, L, g, phi, T, S, M0, b0);
    const rec = channels.find((q) => q.n === n);
    if (rec && (!rec.done || c.omega < rec.omega)) Object.assign(rec, c, { done: true });
    if (c.M > 0) cand.push({ phase: 'B', M: c.M, b: c.b, omega: c.omega });
  }
  let best = cand[0];
  for (const c of cand.slice(1).sort((x, y) => x.omega - y.omega))
    if (c.b > B_MIN && c.omega < best.omega - Math.max(noise(mu, L, T), 1e-10 * Math.abs(best.omega))) { best = c; break; }
  if (best.omega >= -noise(mu, L, T)) best = { phase: 'T', M: 0, b: NaN, omega: 0 };
  if (best.phase === 'B') {
    const thr = thresholds(mu, L, phi);
    const near = thr.reduce((a, q) => (Math.abs(q.S - best.b) < Math.abs(a.S - best.b) ? q : a));
    best = { ...best, phase: 'B' + near.n };
  }
  return { mu, L, g, phi, T, phase: best.phase, M_star: best.M, ln_M_star: best.M > 0 ? Math.log(best.M) : -Infinity,
           b_star: best.phase === 'T' ? NaN : best.b, omega_star: best.omega, channels,
           b0: { M: h.M, omega: h.omega, phase: homPhaseLabel(h.M, mu, L, phi), T0_phase: r0.phase, T0_M: r0.M_star } };
}
