// njl3inh.js — Ω(M, b) модели NJL3 на R²×S¹ (b ≥ 0) при граничном условии φ ∈ {0, 0.1, …, 0.5} и любом g,
// глобальный минимум по (M, b). Порт njl3.inh (= njl3full.js с φ). Перед использованием: setTable(phi, table).
// Моды k_n = 2π(n + φ)/L, уровни |k| с кратностью C; s_φ(x) — таблица Чебышёва для каждого φ.
//
//   Ω(M, b) − Ω(0, 0) = M²/(2g) + Σ_X [X(M, b) − X(0, b)],  X ∈ {ΔU, Ω_L, Ω_μL}
//   ΔU   = M² b / 4 (b ≥ M);  (1/3π)[(2M² + b²)√(M² − b²)/2 + (3/2)M² b arcsin(b/M)] (b ≤ M)
//   Ω_L  = (2/(π² L²)) ∫_0^∞ dp Σ± s(L|E ± b|),  s(x) — таблица Чебышёва
//   Ω_μL = −(1/(πL)) Σ_n C_n Σ± ∫_{F>0} (μ − √((E ± b)² + k_n²)) dp
// У порогов b = S_n при малых M: Ω ≈ M²[A_n − c_n ln(1/M) + c_n h(t)], b = S_n + tM.
import { findMinimum as homFindMinimum, phaseLabel as homPhaseLabel } from './njl3hom.js';

const PI = Math.PI;

// ---------------------------------------------------------------------------
// s(x)
// ---------------------------------------------------------------------------
const TABLES = new Map();
export function setTable(phi, table) {
  TABLES.set(phi, { pieces: table.pieces.map((p) => ({ a: p.a, b: p.b, c: Float64Array.from(p.coef) })),
                    s0: table.s0, xMin: table.x_min || 0 });
}
export const hasTable = (phi) => TABLES.has(phi);
export const PHIS = [0, 0.1, 0.2, 0.3, 0.4, 0.5];
export function checkPhi(phi) {
  let p = phi % 1; if (p < 0) p += 1; p = Math.min(p, 1 - p);
  const q = PHIS.reduce((a, x) => (Math.abs(x - p) < Math.abs(a - p) ? x : a));
  if (Math.abs(q - p) > 1e-9) throw new Error(`φ = ${phi}: таблицы есть только для ${PHIS.join(', ')}`);
  return q;
}

function chebval(t, c) {
  let b1 = 0, b2 = 0;
  for (let k = c.length - 1; k >= 1; k--) { const b0 = 2 * t * b1 - b2 + c[k]; b2 = b1; b1 = b0; }
  return t * b1 - b2 + c[0];
}

export function sFunc(x, phi) {
  if (x >= 64) return 0;
  const T = TABLES.get(phi);
  if (phi !== 0 && x < T.xMin) return T.s0;
  const P = T.pieces;
  let lo = 0, hi = P.length - 1;                     // бинарный поиск куска
  while (lo < hi) { const m = (lo + hi) >> 1; if (x < P[m].b) hi = m; else lo = m + 1; }
  const p = P[lo];
  let v = chebval((2 * x - (p.a + p.b)) / (p.b - p.a), p.c);
  if (phi === 0 && lo === 0 && x > 0) v -= x * x / 4 * Math.log(x);
  return v;
}

// ---------------------------------------------------------------------------
// Квадратуры
// ---------------------------------------------------------------------------
function gaussLegendre(n) {
  const x = new Float64Array(n), w = new Float64Array(n);
  for (let i = 0; i < Math.ceil(n / 2); i++) {
    let z = Math.cos(PI * (i + 0.75) / (n + 0.5)), dp = 0;
    for (let it = 0; it < 100; it++) {
      let p0 = 1, p1 = z;
      for (let k = 2; k <= n; k++) { const p2 = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
      dp = n * (z * p1 - p0) / (z * z - 1);
      const dz = p1 / dp; z -= dz;
      if (Math.abs(dz) < 1e-16) break;
    }
    x[i] = -z; x[n - 1 - i] = z; w[i] = w[n - 1 - i] = 2 / ((1 - z * z) * dp * dp);
  }
  return [x, w];
}
function nodes(segs, [x0, w0]) {
  const xs = [], ws = [];
  for (const [a, b] of segs) {
    if (b <= a) continue;
    const h = 0.5 * (b - a), c = 0.5 * (a + b);
    for (let i = 0; i < x0.length; i++) { xs.push(h * x0[i] + c); ws.push(h * w0[i]); }
  }
  return [Float64Array.from(xs), Float64Array.from(ws)];
}
const GL16 = gaussLegendre(16);
function gradedTemplate(levels) {
  const segs = [];
  for (let j = 0; j < levels; j++) segs.push([2 ** -(j + 1), 2 ** -j]);
  segs.push([0, 2 ** -levels]);
  return nodes(segs, GL16);
}
const [GT_X, GT_W] = gradedTemplate(16);
const [TS_X, TS_W] = (() => {           // к обоим концам [0, 1]
  const [gx, gw] = gradedTemplate(20);
  const x = new Float64Array(2 * gx.length), w = new Float64Array(2 * gx.length);
  for (let i = 0; i < gx.length; i++) { x[i] = 0.5 * gx[i]; w[i] = 0.5 * gw[i]; x[gx.length + i] = 1 - 0.5 * gx[i]; w[gx.length + i] = 0.5 * gw[i]; }
  return [x, w];
})();
const [CH_X, CH_W] = nodes(Array.from({ length: 16 }, (_, i) => [i / 16, (i + 1) / 16]), GL16);

// ---------------------------------------------------------------------------
// Вклады
// ---------------------------------------------------------------------------
export function dU(M, b) {
  if (M === 0) return 0;
  if (b >= M) return M * M * b / 4;
  return ((2 * M * M + b * b) * Math.sqrt(M * M - b * b) / 2 + 1.5 * M * M * b * Math.asin(b / M)) / (3 * PI);
}

export function omegaL(M, b, L, phi) {
  const pb = b > M ? Math.sqrt(b * b - M * M) : 0;
  const Ehi = b + 64 / L, phi_ = Math.sqrt(Math.max(Ehi * Ehi - M * M, 0));
  const c = Math.min(pb + 4 / L, phi_);
  let tot = 0;
  const add = (p, w) => { const E = Math.sqrt(M * M + p * p); tot += w * (sFunc(L * (E + b), phi) + sFunc(L * Math.abs(E - b), phi)); };
  for (let i = 0; i < TS_X.length; i++) add(pb * TS_X[i], pb * TS_W[i]);
  for (let i = 0; i < GT_X.length; i++) add(pb + (c - pb) * GT_X[i], (c - pb) * GT_W[i]);
  for (let i = 0; i < CH_X.length; i++) add(c + (phi_ - c) * CH_X[i], (phi_ - c) * CH_W[i]);
  return 2 / (PI * PI * L * L) * tot;
}

const pE = (E, M) => Math.sqrt(Math.max(E * E - M * M, 0));
function J(p, M) { if (M === 0) return 0.5 * p * p; const E = Math.sqrt(M * M + p * p); return 0.5 * (p * E + M * M * Math.asinh(p / M)); }

function mode0(mu, b, M) {
  let tot = 0;
  if (mu - b > M) { const pr = pE(mu - b, M); tot += (mu - b) * pr - J(pr, M); }
  const lo = Math.max(M, b - mu), hi = b + mu;
  const e2 = Math.min(b, hi);
  if (e2 > lo) { const p1 = pE(lo, M), p2 = pE(e2, M); tot += (mu - b) * (p2 - p1) + J(p2, M) - J(p1, M); }
  const e3 = Math.max(b, lo);
  if (hi > e3) { const p3 = pE(e3, M), p4 = pE(hi, M); tot += (mu + b) * (p4 - p3) - (J(p4, M) - J(p3, M)); }
  return tot;
}

function segInt(a, c, M, bs, k, mu) {
  const h = c - a;
  if (h <= 0) return 0;
  let s = 0;
  for (let i = 0; i < TS_X.length; i++) {
    const p = a + h * TS_X[i], A = Math.sqrt(M * M + p * p) + bs;
    s += TS_W[i] * (mu - Math.sqrt(A * A + k * k));
  }
  return h * s;
}

// уровни |k_n| < mu с кратностью: [{k, C}], по возрастанию |k|
const LEVELS = new Map();
export function levels(mu, L, phi) {
  if (mu <= 0) return [];
  const key = mu + ',' + L + ',' + phi;
  let v = LEVELS.get(key);
  if (!v) {
    if (LEVELS.size > 4096) LEVELS.clear();
    v = levelsRaw(mu, L, phi);
    LEVELS.set(key, v);
  }
  return v;
}
function levelsRaw(mu, L, phi) {
  const m = new Map(), nmax = Math.floor(L * mu / (2 * PI)) + 1;
  for (let n = -nmax - 1; n <= nmax; n++) {
    const k = Math.abs(2 * PI * (n + phi) / L);
    if (k < mu) { const key = Math.round(k * 1e12) / 1e12; m.set(key, (m.get(key) || 0) + 1); }
  }
  return [...m.entries()].sort((a, b) => a[0] - b[0]).map(([k, C]) => ({ k, C }));
}

function modeK(M, b, mu, k) {
  const S = Math.sqrt(mu * mu - k * k);
  let I = S - b > M ? segInt(0, pE(S - b, M), M, b, k, mu) : 0;
  const lo = Math.max(M, b - S), hi = b + S;
  const pl = pE(lo, M), pr = pE(Math.max(hi, lo), M);
  const mid = Math.min(Math.max(pE(b, M), pl), pr);
  return I + segInt(pl, mid, M, -b, k, mu) + segInt(mid, pr, M, -b, k, mu);
}

export function omegaMuL(M, b, mu, L, phi) {
  let tot = 0;
  for (const { k, C } of levels(mu, L, phi)) tot += C * (k === 0 ? mode0(mu, b, M) : modeK(M, b, mu, k));
  return -tot / (PI * L);
}

// Ω_L(0, b) and Ω_μL(0, b) do not depend on b (at M = 0, b only shifts the momentum): cached at b = 0, as in Python
const ZERO = new Map();
function zeroTerms(mu, L, phi) {
  const key = mu + ',' + L + ',' + phi;
  let v = ZERO.get(key);
  if (v === undefined) {
    if (ZERO.size > 4096) ZERO.clear();
    v = omegaL(0, 0, L, phi) + omegaMuL(0, 0, mu, L, phi);
    ZERO.set(key, v);
  }
  return v;
}
export function omegaRel(M, b, mu, L, g, phi) {
  if (M === 0) return 0;
  return M * M / (2 * g) + dU(M, b) + omegaL(M, b, L, phi) + omegaMuL(M, b, mu, L, phi) - zeroTerms(mu, L, phi);
}

// ---------------------------------------------------------------------------
// Каналы у порогов и глобальный минимум
// ---------------------------------------------------------------------------
export const M_ASYMP = 1e-6, M_PROBE = 1e-4, M_ASYMP_VALID = 1e-2, OMEGA_NOISE = 1e-12;

export function thresholds(mu, L, phi) {
  return levels(mu, L, phi).map(({ k, C }, n) => ({ n, S: Math.sqrt(mu * mu - k * k), C }));
}

export function hUniversal(t) {
  t = Math.abs(t);
  return t <= 1 ? t * t : t * t - t * Math.sqrt(t * t - 1) + Math.acosh(t);
}

export function channelAsymptotics(S, C, mu, L, g, phi) {
  const c = C * S / (2 * PI * L * mu);
  const A = omegaRel(M_PROBE, S, mu, L, g, phi) / M_PROBE ** 2 + c * Math.log(1 / M_PROBE);
  const lnM = -A / c - 0.5;
  const M = lnM > -700 ? Math.exp(lnM) : 0;
  return { c, A, lnM, M, omega: -0.5 * c * M * M };
}

function nelderMead(f, x0, step, { xatol, fatol, maxfev }) {
  let simplex = [x0, [x0[0] + step, x0[1]], [x0[0], x0[1] + step]].map((x) => ({ x, f: f(x) }));
  let fev = 3;
  while (fev < maxfev) {
    simplex.sort((a, b) => a.f - b.f);
    const [B, G2, W] = simplex;
    const dx = Math.max(Math.abs(G2.x[0] - B.x[0]), Math.abs(W.x[0] - B.x[0]), Math.abs(G2.x[1] - B.x[1]), Math.abs(W.x[1] - B.x[1]));
    if (dx <= xatol && Math.abs(W.f - B.f) <= fatol) break;
    const c = [(B.x[0] + G2.x[0]) / 2, (B.x[1] + G2.x[1]) / 2];
    const pt = (a) => [c[0] + a * (W.x[0] - c[0]), c[1] + a * (W.x[1] - c[1])];
    const R = { x: pt(-1) }; R.f = f(R.x); fev++;
    if (R.f < B.f) {
      const E = { x: pt(-2) }; E.f = f(E.x); fev++;
      simplex[2] = E.f < R.f ? E : R;
    } else if (R.f < G2.f) {
      simplex[2] = R;
    } else {
      const K = { x: pt(R.f < W.f ? -0.5 : 0.5) }; K.f = f(K.x); fev++;
      if (K.f < Math.min(R.f, W.f)) simplex[2] = K;
      else {
        simplex = [B, ...[G2, W].map((p) => { const x = [B.x[0] + 0.5 * (p.x[0] - B.x[0]), B.x[1] + 0.5 * (p.x[1] - B.x[1])]; fev++; return { x, f: f(x) }; })];
      }
    }
  }
  simplex.sort((a, b) => a.f - b.f);
  return simplex[0];
}

const LNM = Array.from({ length: 40 }, (_, i) => Math.log(M_ASYMP) + (Math.log(4.5) - Math.log(M_ASYMP)) * i / 39);
const TG = [...Array.from({ length: 13 }, (_, i) => -4 + 3 * i / 12), ...Array.from({ length: 19 }, (_, i) => -0.9 + 1.8 * i / 18),
            ...Array.from({ length: 5 }, (_, i) => 1 + 2 * i / 4)];

// Порог по температуре (как njl3.full.minimize): щель канала n выживает при T < TC_RATIO·(S_n/mu)·M*_n
export const TC_RATIO = Math.exp(0.5772156649015329) / PI, PRUNE_FACTOR = 0.8;
export const channelMMin = (S, mu, T) => (T > 0 ? T / (TC_RATIO * S / mu) : 0);
const LNM_LO = Math.log(M_ASYMP), LNM_HI = Math.log(6), LN45 = Math.log(4.5);

export function channelMinimum(S, C, mu, L, g, phi, T = 0, asy = null) {
  asy = asy || channelAsymptotics(S, C, mu, L, g, phi);
  const mMin = channelMMin(S, mu, T);
  const lnmLo = mMin > 0 ? Math.max(LNM_LO, Math.log(mMin)) : LNM_LO;
  if (lnmLo >= LN45) return { M: 0, lnM: -Infinity, b: S, omega: 0, asym: false, c: asy.c, A: asy.A };
  let grid;
  if (asy.M < M_ASYMP_VALID && asy.lnM > lnmLo) grid = [[asy.lnM, 0]];   // яма у асимптотики — без сетки
  else {
    const nl = lnmLo === LNM_LO ? 40 : Math.max(8, Math.round(40 * (LN45 - lnmLo) / (LN45 - LNM_LO)));
    const lnm = lnmLo === LNM_LO ? LNM : Array.from({ length: nl }, (_, i) => lnmLo + (LN45 - lnmLo) * i / (nl - 1));
    grid = TG.flatMap((t) => lnm.map((lm) => [lm, t]));
  }
  let best = { M: 0, lnM: -Infinity, b: S, omega: Infinity, asym: false }, bi = null;
  for (const [lm, t] of grid) {
    const M = Math.exp(lm), b = S + t * M;
    if (b < 0) continue;
    const o = omegaRel(M, b, mu, L, g, phi);
    if (o < best.omega) { best = { M, lnM: lm, b, omega: o, asym: false }; bi = [lm, t]; }
  }
  const f = ([lm, t]) => {
    const lmc = Math.min(Math.max(lm, lnmLo), LNM_HI);
    const M = Math.exp(lmc), b = S + t * M;
    return b >= 0 ? omegaRel(M, b, mu, L, g, phi) : 1e3;
  };
  const r = nelderMead(f, bi, 0.3, { xatol: T > 0 ? 1e-6 : 1e-9,
                                     fatol: (T > 0 ? 1e-10 : 1e-13) * Math.abs(best.omega) + 1e-300, maxfev: 1200 });
  if (r.f < best.omega) {
    const lm = Math.min(Math.max(r.x[0], lnmLo), LNM_HI), M = Math.exp(lm);
    best = { M, lnM: lm, b: S + r.x[1] * M, omega: r.f, asym: false };
  }
  // мельче шума или ниже порога по T
  if (best.omega > -OMEGA_NOISE || best.M < mMin * (1 + 1e-9)) best = { M: 0, lnM: -Infinity, b: S, omega: 0, asym: false };
  if (asy.M < M_ASYMP_VALID && asy.M > mMin && asy.omega < best.omega)
    best = { M: asy.M, lnM: asy.lnM, b: S, omega: asy.omega, asym: true };
  return { ...best, c: asy.c, A: asy.A };
}

export function findMinimum(mu, L, g, phi = 0, T = 0) {
  phi = checkPhi(phi);
  const r0 = homFindMinimum(mu, L, g, phi);
  let Mb = r0.M, Ob = r0.omega;
  if (T > 0 && Mb > 0 && Mb <= T / TC_RATIO) { Mb = 0; Ob = 0; }     // b = 0: порог приближённый (скорость 1)
  const cand = [{ phase: homPhaseLabel(Mb, mu, L, phi), M: Mb, lnM: Mb > 0 ? Math.log(Mb) : -Infinity, b: 0, omega: Ob, asym: false }];
  const thr = thresholds(mu, L, phi);
  const asys = thr.map(({ S, C }) => channelAsymptotics(S, C, mu, L, g, phi));
  // каналы в порядке асимптотической глубины; при T > 0 явно мелкие асимптотические не уточняются
  const key = (k) => (asys[k].M < M_ASYMP_VALID ? asys[k].omega : -Infinity);
  const order = thr.map((_, k) => k).sort((a, b) => key(a) - key(b));
  const channels = new Array(thr.length);
  let bestOm = Math.min(Ob, 0);
  for (const k of order) {
    const { n, S, C } = thr[k], asy = asys[k], small = asy.M < M_ASYMP_VALID;
    let ch;
    if (T > 0 && small && asy.M <= channelMMin(S, mu, T))
      ch = { M: 0, lnM: -Infinity, b: S, omega: 0, asym: true, c: asy.c, A: asy.A, pruned: true };
    else if (T > 0 && small && asy.omega > PRUNE_FACTOR * bestOm)
      ch = { M: asy.M, lnM: asy.lnM, b: S, omega: asy.omega, asym: true, c: asy.c, A: asy.A, pruned: true };
    else ch = channelMinimum(S, C, mu, L, g, phi, T, asy);
    ch = { ...ch, n, S };
    channels[k] = ch;
    if (ch.M > 0) {
      bestOm = Math.min(bestOm, ch.omega);
      if (ch.b > 1e-8) cand.push({ phase: 'B' + n, M: ch.M, lnM: ch.lnM, b: ch.b, omega: ch.omega, asym: ch.asym });
    }
  }
  let best = cand[0];
  // каналы сравниваются между собой без запаса; b = 0 — точное решение: численный канал
  // побеждает его только с запасом больше шума, асимптотический — без запаса
  for (const c of cand.slice(1).sort((x, y) => x.omega - y.omega))
    if (c.omega < best.omega - (c.asym ? 0 : OMEGA_NOISE)) { best = c; break; }
  if (best.phase[0] === 'B') {
    // номер фазы — ближайший порог к найденному b* (поиск в канале n может прийти в яму соседнего порога)
    const near = thr.reduce((a, q) => (Math.abs(q.S - best.b) < Math.abs(a.S - best.b) ? q : a));
    best = { ...best, phase: 'B' + near.n };
  }
  return { mu, L, g, phi, T, phase: best.phase, M_star: best.M, ln_M_star: best.lnM, b_star: best.phase === 'T' ? NaN : best.b,
           omega_star: best.omega, asymptotic: best.asym, channels,
           b0: { M: r0.M, omega: r0.omega, phase: homPhaseLabel(r0.M, mu, L, phi) } };
}
