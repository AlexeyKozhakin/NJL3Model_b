// fastmin.js — fast global minimum of Ω(M, b) (JavaScript port of njl3/fast/minimize.py).
//
// Candidates: the homogeneous solution (njl3hom.js, exact); one well per channel j at the threshold
// S_j = sqrt(mu² − k_j²): leading asymptotics M_j = exp(−A_j/c_j − ½), b = S_j (A_j from Ω(M_PROBE, S_j));
// for M_j > M_REFINE — scouting grid + damped Newton in (ln M, t), b = S_j + t M; near-degenerate asymptotic
// rivals are refined too; in the strip near mu ≈ π (b = 0 solution intermediate or a large channel well) —
// a dense (M, b) scan 40×60 and Newton from its 5 best grid-local minima. Temperature cut-off
// T_c,j = (e^γ/π)(S_j/mu) M_j and candidate selection as in njl3.inh.
import { omegaRel, levels, checkPhi, M_PROBE, OMEGA_NOISE, TC_RATIO, channelMMin } from '../../lab/js/njl3inh.js';
import { findMinimum as homFindMinimum, phaseLabel as homPhaseLabel } from '../../lab/js/njl3hom.js';

const PI = Math.PI;
export const M_REFINE = 1e-2, M_RIVAL = 1e-3, RIVAL = 0.9, GLOBAL_TRIGGER = 0.3;
const LNM_MAX = Math.log(6), H = 1e-3;
const SX = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5];
const ST = [-4, -3, -2, -1, -0.6, -0.3, 0, 0.3, 0.6, 1, 2, 3];
const GM = Array.from({ length: 40 }, (_, i) => 0.02 * Math.pow(4.0 / 0.02, i / 39));

export function channels(mu, L, g, phi) {
  const c0 = Math.log(1 / M_PROBE);
  return levels(mu, L, phi).map(({ k, C }, j) => {
    const S = Math.sqrt(mu * mu - k * k), c = C * S / (2 * PI * L * mu);
    const A = omegaRel(M_PROBE, S, mu, L, g, phi) / (M_PROBE * M_PROBE) + c * c0;
    return { j, S, C, A, c };
  });
}

function omegaXT(x, t, S, mu, L, g, phi) {
  const M = Math.exp(Math.min(x, LNM_MAX)), b = S + t * M;
  return b >= 0 ? omegaRel(M, b, mu, L, g, phi) : 1e3;
}

// Local minimum of a channel near (ln M, t) = (x0, t0); t0 === null — start from a scouting grid.
export function refine(x0, S, mu, L, g, phi, lnmLo, t0 = null, steps = 12) {
  const clip = (x) => Math.min(Math.max(x, lnmLo), LNM_MAX);
  let x, t, fx;
  if (t0 === null) {
    fx = Infinity;
    for (const dx of SX) for (const tt of ST) {
      const xx = clip(x0 + dx), f = omegaXT(xx, tt, S, mu, L, g, phi);
      if (f < fx) { fx = f; x = xx; t = tt; }
    }
  } else { x = x0; t = t0; fx = omegaXT(x, t, S, mu, L, g, phi); }
  for (let it = 0; it < steps; it++) {
    const f = (i, j) => omegaXT(x + i * H, t + j * H, S, mu, L, g, phi);
    const f00 = fx, fp0 = f(1, 0), fm0 = f(-1, 0), f0p = f(0, 1), f0m = f(0, -1);
    const fpp = f(1, 1), fpm = f(1, -1), fmp = f(-1, 1), fmm = f(-1, -1);
    const gx = (fp0 - fm0) / (2 * H), gt = (f0p - f0m) / (2 * H);
    const hxx = (fp0 - 2 * f00 + fm0) / (H * H), htt = (f0p - 2 * f00 + f0m) / (H * H);
    const hxt = (fpp - fpm - fmp + fmm) / (4 * H * H), det = hxx * htt - hxt * hxt;
    let dx, dt;
    if (hxx > 0 && det > 0) { dx = -(htt * gx - hxt * gt) / det; dt = -(hxx * gt - hxt * gx) / det; }
    else { const sc = 0.1 / Math.max(Math.abs(gx), Math.abs(gt), 1e-300); dx = -gx * sc; dt = -gt * sc; }
    let lam = 1, xn, tn, fn, ok = false;
    for (let k = 0; k < 12; k++) {
      xn = clip(x + lam * dx); tn = t + lam * dt; fn = omegaXT(xn, tn, S, mu, L, g, phi);
      if (fn <= fx) { ok = true; break; }
      lam *= 0.5;
    }
    if (!ok) break;
    const step = Math.max(Math.abs(xn - x), Math.abs(tn - t));
    x = xn; t = tn; fx = fn;
    if (step < 1e-7) break;
  }
  return { x, t, om: fx };
}

// Dense (M, b) scan: grid-local minima, refined relative to the nearest threshold.
export function scanWells(mu, L, g, phi, thr, mMin, nBest = 5, GMs = GM, nb = 60) {
  const bg = Array.from({ length: nb }, (_, i) => (mu + 0.5) * i / (nb - 1));
  const nM = GMs.length, Om = new Float64Array(nM * nb);
  for (let i = 0; i < nM; i++) for (let k = 0; k < nb; k++) Om[i * nb + k] = omegaRel(GMs[i], bg[k], mu, L, g, phi);
  const loc = [];
  for (let i = 0; i < nM; i++) for (let k = 1; k < nb; k++) {
    const v = Om[i * nb + k];
    if (!(v < -OMEGA_NOISE)) continue;
    let ok = true;
    for (let di = -1; di <= 1 && ok; di++) for (let dk = -1; dk <= 1; dk++) {
      if (!di && !dk) continue;
      const ii = i + di, kk = k + dk;
      if (ii < 0 || ii >= nM || kk < 0 || kk >= nb) continue;
      if (Om[ii * nb + kk] < v) { ok = false; break; }
    }
    if (ok) loc.push([v, i, k]);
  }
  loc.sort((a, b) => a[0] - b[0]);
  const lnmLo = Math.log(Math.max(mMin, 1e-6)), out = [];
  for (const [, i, k] of loc.slice(0, nBest)) {
    const M = GMs[i], b = bg[k];
    const near = thr.reduce((a, q) => (Math.abs(q.S - b) < Math.abs(a.S - b) ? q : a));
    const r = refine(Math.max(Math.log(M), lnmLo), near.S, mu, L, g, phi, lnmLo, (b - near.S) / M);
    const Mr = Math.exp(r.x), br = near.S + r.t * Mr;
    if (r.om < -OMEGA_NOISE && Mr > mMin && br > 1e-8) {
      const jn = thr.reduce((a, q) => (Math.abs(q.S - br) < Math.abs(a.S - br) ? q : a)).j;
      out.push({ phase: 'B' + jn, M: Mr, lnM: r.x, b: br, omega: r.om, asym: false });
    }
  }
  return out;
}

export function findMinimum(mu, L, g, phi = 0, T = 0) {
  phi = checkPhi(phi);
  const r0 = homFindMinimum(mu, L, g, phi);
  let Mb = r0.M, Ob = r0.omega;
  const b0 = { M: Mb, omega: Ob, phase: homPhaseLabel(Mb, mu, L, phi) };
  if (T > 0 && Mb > 0 && Mb <= T / TC_RATIO) { Mb = 0; Ob = 0; }
  const cand = [{ phase: homPhaseLabel(Mb, mu, L, phi), M: Mb, lnM: Mb > 0 ? Math.log(Mb) : -Infinity, b: 0, omega: Ob, asym: false }];
  const chans = [];
  for (const { j, S, C, A, c } of channels(mu, L, g, phi)) {
    const lnM = -A / c - 0.5, mMin = channelMMin(S, mu, T);
    const ch = { n: j, S, C, c, A, M: 0, lnM: -Infinity, b: S, omega: 0, asym: true };
    if (lnM < Math.log(Math.max(mMin, 1e-300)) || lnM < -700) { chans.push(ch); continue; }
    if (lnM < Math.log(M_REFINE)) {
      const M = Math.exp(lnM);
      Object.assign(ch, { M, lnM, omega: -0.5 * c * M * M });
    } else {
      const lnmLo = mMin > 0 ? Math.log(mMin) : Math.log(1e-6);
      const r = refine(Math.min(Math.max(lnM, lnmLo), LNM_MAX), S, mu, L, g, phi, lnmLo);
      const M = Math.exp(r.x);
      if (r.om < -OMEGA_NOISE && M > mMin) Object.assign(ch, { M, lnM: r.x, b: S + r.t * M, omega: r.om, asym: false });
    }
    chans.push(ch);
    if (ch.M > 0 && ch.b > 1e-8) cand.push({ phase: 'B' + j, M: ch.M, lnM: ch.lnM, b: ch.b, omega: ch.omega, asym: ch.asym });
  }
  // near-degenerate asymptotic rivals of the best
  const bestOm = Math.min(0, ...cand.map((q) => q.omega));
  for (const ch of chans) {
    if (!(ch.asym && ch.M > M_RIVAL && ch.omega <= RIVAL * bestOm)) continue;
    const mMin = channelMMin(ch.S, mu, T), lnmLo = mMin > 0 ? Math.log(mMin) : Math.log(1e-6);
    const r = refine(Math.max(ch.lnM, lnmLo), ch.S, mu, L, g, phi, lnmLo);
    const M = Math.exp(r.x);
    if (r.om < ch.omega && M > mMin) {
      Object.assign(ch, { M, lnM: r.x, b: ch.S + r.t * M, omega: r.om, asym: false });
      for (const q of cand) if (q.phase === 'B' + ch.n) Object.assign(q, { M, lnM: r.x, b: ch.b, omega: r.om, asym: false });
    }
  }
  const thr = chans.map((ch) => ({ j: ch.n, S: ch.S }));
  const big = chans.some((ch) => ch.lnM > Math.log(GLOBAL_TRIGGER)) || (b0.M > 0 && b0.M < mu);
  if (thr.length && big) {
    const mMinAll = Math.min(...thr.map((q) => channelMMin(q.S, mu, T)));
    cand.push(...scanWells(mu, L, g, phi, thr, mMinAll));
  }
  let best = cand[0];
  for (const q of cand.slice(1).sort((a, b) => a.omega - b.omega))
    if (q.omega < best.omega - (q.asym ? 0 : OMEGA_NOISE)) { best = q; break; }
  if (best.phase[0] === 'B') {
    const near = chans.reduce((a, q) => (Math.abs(q.S - best.b) < Math.abs(a.S - best.b) ? q : a));
    best = { ...best, phase: 'B' + near.n };
  }
  return { mu, L, g, phi, T, phase: best.phase, M_star: best.M, ln_M_star: best.lnM,
           b_star: best.phase === 'T' ? NaN : best.b, omega_star: best.omega, asymptotic: best.asym,
           channels: chans, candidates: cand, b0 };
}

// All local minima of Ω(M, b) on a dense grid (for the point panel), refined; plus the b = 0 minima.
export function allMinima(mu, L, g, phi, nM = 70, nb = 70, Mmax = 3.5) {
  phi = checkPhi(phi);
  const thr = channels(mu, L, g, phi).map(({ j, S }) => ({ j, S }));
  const GMs = Array.from({ length: nM }, (_, i) => 0.03 + (Mmax - 0.03) * i / (nM - 1));
  const wells = thr.length ? scanWells(mu, L, g, phi, thr, 0, 40, GMs, nb) : [];
  const h = homFindMinimum(mu, L, g, phi);
  for (const m of h.minima) wells.push({ phase: homPhaseLabel(m.M, mu, L, phi), M: m.M, lnM: Math.log(m.M), b: 0, omega: m.omega, asym: false });
  // merge duplicates (Newton from neighbouring grid minima lands in the same well)
  const out = [];
  for (const w of wells.sort((a, b) => a.omega - b.omega))
    if (!out.some((q) => Math.abs(q.M - w.M) < 1e-4 * (1 + q.M) && Math.abs(q.b - w.b) < 1e-4 * (1 + q.b))) out.push(w);
  return out;
}
