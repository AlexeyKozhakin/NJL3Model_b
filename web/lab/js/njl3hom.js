// njl3hom.js — homogeneous NJL3/GN potential on R²×S¹ (b = 0) for an arbitrary twist φ and coupling g
// (JavaScript port of njl3/hom). At φ = 0 everything is delegated to the paper-1 module njl3b0.js.
//
// Modes k_n = 2π(n + φ)/L, n ∈ Z; m_n = sqrt(M² + k_n²); φ is reduced to [0, 1/2].
//   G(M) = 1/g + M/π + (1/(πL)) ln D(LM) + (2/(πL)) Σ_{m_n<mu} arccosh(mu/m_n),
//   D(x) = (1 − e^{−x})² + 4 e^{−x} sin² πφ.
// Certified roots: G = 1/g + A + B, A = M/π + ln D/(πL) increasing, B = (2/(πL)) Σ arccosh decreasing.
import * as B0 from '../../b0/js/njl3b0.js';

const PI = Math.PI;

export function reducePhi(phi) {
  let p = phi % 1;
  if (p < 0) p += 1;
  return Math.min(p, 1 - p);
}

export function modes(mu, L, phi) {
  if (mu <= 0) return [];
  const nmax = Math.floor(L * mu / (2 * PI)) + 1, ks = [];
  for (let n = -nmax - 1; n <= nmax; n++) {
    const k = 2 * PI * (n + phi) / L;
    if (Math.abs(k) < mu) ks.push(k);
  }
  return ks.sort((a, b) => Math.abs(a) - Math.abs(b));
}

export function levels(mu, L, phi) {
  return [...new Set(modes(mu, L, phi).map((k) => Math.round(Math.abs(k) * 1e13) / 1e13))].sort((a, b) => a - b);
}

export function thresholds(mu, L, phi) {
  phi = reducePhi(phi);
  if (phi === 0) return B0.thresholds(mu, L);
  return levels(mu, L, phi).map((k) => Math.sqrt(mu * mu - k * k)).sort((a, b) => a - b);
}

export function filledLevel(M, mu, L, phi) {
  let j = -1;
  levels(mu, L, phi).forEach((k, i) => { if (M * M + k * k < mu * mu) j = i; });
  return j;
}

export function Smax(mu, L, phi) {
  const ks = modes(mu, L, phi);
  return ks.length ? Math.sqrt(mu * mu - ks[0] * ks[0]) : 0;
}

function lnD(x, phi) {
  const e = Math.expm1(-x), s = Math.sin(PI * phi);
  return Math.log(e * e + 4 * Math.exp(-x) * s * s);
}

export function gapG(M, mu, L, g, phi) {
  phi = reducePhi(phi);
  if (phi === 0) return B0.gapG(M, mu, L, g);
  const ks = modes(mu, L, phi), x = L * M;
  let s, rest = ks;
  if (ks.length && Math.abs(ks[0]) * L < 1e-2) {
    const k0 = ks[0], m0sq = M * M + k0 * k0;
    s = lnD(x, phi);
    if (m0sq < mu * mu) s += -Math.log(x * x + (L * k0) ** 2) + 2 * Math.log(L * (mu + Math.sqrt(mu * mu - m0sq)));
    rest = ks.slice(1);
  } else s = lnD(x, phi);
  for (const k of rest) {
    const m = Math.sqrt(M * M + k * k);
    if (m < mu) s += 2 * Math.log((mu + Math.sqrt(mu * mu - m * m)) / m);
  }
  return 1 / g + M / PI + s / (PI * L);
}

export function gapG0(mu, L, g, phi) {
  phi = reducePhi(phi);
  return phi === 0 ? B0.gapG0(mu, L, g) : gapG(0, mu, L, g, phi);
}

function partA(M, L, phi) { return M / PI + lnD(L * M, phi) / (PI * L); }
function partB(M, mu, L, phi) {
  let s = 0;
  for (const k of modes(mu, L, phi)) { const m = Math.sqrt(M * M + k * k); if (m < mu) s += Math.acosh(mu / m); }
  return 2 * s / (PI * L);
}

export function isolateRoots(mu, L, g, phi, tol = 1e-10) {
  phi = reducePhi(phi);
  if (phi === 0) return B0.isolateRoots(mu, L, g, tol);
  const G = (x) => gapG(x, mu, L, g, phi), ig = 1 / g, roots = [];
  const S = Smax(mu, L, phi);
  if (S > 0) {
    const A = (x) => partA(x, L, phi), Bf = (x) => partB(x, mu, L, phi);
    const windows = [], stack = [[0, S, A(0), A(S), Bf(0), Bf(S)]];
    while (stack.length) {
      const [p, q, Ap, Aq, Bp, Bq] = stack.pop();
      if (ig + Ap + Bq > 0 || ig + Aq + Bp < 0) continue;
      if (q - p <= tol) { windows.push([p, q]); continue; }
      const c = 0.5 * (p + q), Ac = A(c), Bc = Bf(c);
      stack.push([c, q, Ac, Aq, Bc, Bq]);
      stack.push([p, c, Ap, Ac, Bp, Bc]);
    }
    windows.sort((u, v) => u[0] - v[0]);
    const merged = [];
    for (const [p, q] of windows) {
      if (merged.length && p <= merged[merged.length - 1][1] + 1e-15) merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], q);
      else merged.push([p, q]);
    }
    for (const [p, q] of merged) {
      if (p === 0 && q <= tol) continue;
      const Gp = G(p), Gq = G(q);
      if (Gp < 0 && Gq > 0) roots.push({ M: B0.brent(G, p, q), kind: 'min' });
      else if (Gp > 0 && Gq < 0) roots.push({ M: B0.brent(G, p, q), kind: 'max' });
      else roots.push({ M: 0.5 * (p + q), kind: 'touch' });
    }
  }
  const a = Math.max(S, 1e-300);
  if (G(a) < 0) {
    let b = 2 * a + 1;
    while (G(b) < 0) b *= 2;
    roots.push({ M: B0.brent(G, a, b), kind: 'min' });
  }
  return roots.sort((u, v) => u.M - v.M);
}

// Ω(M) − Ω(0) = ∫_0^M m G(m) dm, Gauss–Legendre; m = S − u² near a threshold S.
function gaussLegendre(n) {
  const x = new Float64Array(n), w = new Float64Array(n);
  for (let i = 0; i < Math.ceil(n / 2); i++) {
    let z = Math.cos(PI * (i + 0.75) / (n + 0.5)), dp = 0;
    for (let it = 0; it < 100; it++) {
      let p0 = 1, p1 = z;
      for (let k = 2; k <= n; k++) { const p2 = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
      dp = n * (z * p1 - p0) / (z * z - 1);
      const dz = p1 / dp;
      z -= dz;
      if (Math.abs(dz) < 1e-16) break;
    }
    x[i] = -z; x[n - 1 - i] = z;
    w[i] = w[n - 1 - i] = 2 / ((1 - z * z) * dp * dp);
  }
  return [x, w];
}
const [GX, GW] = gaussLegendre(48);

export function omegaRel(M, mu, L, g, phi) {
  phi = reducePhi(phi);
  if (phi === 0) return B0.omegaRel(M, mu, L, g);
  if (M === 0) return 0;
  const S = thresholds(mu, L, phi);
  const edges = [0, ...S.filter((s) => s > 0 && s < M)];
  let last = edges[edges.length - 1];
  const top = S.length ? S[S.length - 1] : 0;
  while (M - last > 1 && last >= top) { last += 1; edges.push(last); }
  edges.push(M);
  let total = 0;
  for (let i = 0; i + 1 < edges.length; i++) {
    const a = edges[i], b = edges[i + 1];
    if (!(b > a)) continue;
    const Sn = S.find((s) => s >= b);
    let part = 0;
    if (Sn === undefined || Sn - b >= b - a) {
      const h = 0.5 * (b - a), c = 0.5 * (a + b);
      for (let j = 0; j < 48; j++) { const m = h * GX[j] + c; part += GW[j] * m * gapG(m, mu, L, g, phi); }
      total += h * part;
    } else {
      const ua = Math.sqrt(Sn - b), ub = Math.sqrt(Sn - a), h = 0.5 * (ub - ua), c = 0.5 * (ua + ub);
      for (let j = 0; j < 48; j++) { const u = h * GX[j] + c, m = Math.max(Sn - u * u, 1e-300); part += GW[j] * 2 * u * m * gapG(m, mu, L, g, phi); }
      total += h * part;
    }
  }
  return total;
}

export function phaseLabel(M, mu, L, phi) {
  phi = reducePhi(phi);
  if (phi === 0) return B0.phaseLabel(M, mu, L);
  if (M === 0) return 'T';
  const j = filledLevel(M, mu, L, phi);
  return j < 0 ? 'V' : 'I' + j;
}

export function findMinimum(mu, L, g, phi) {
  phi = reducePhi(phi);
  if (phi === 0) return B0.findMinimum(mu, L, g);
  const roots = isolateRoots(mu, L, g, phi);
  const minima = roots.filter((r) => r.kind !== 'max').map((r) => ({ M: r.M, omega: omegaRel(r.M, mu, L, g, phi) }));
  let best = { M: 0, omega: 0 };
  for (const m of minima) if (m.omega < best.omega) best = m;
  return { M: best.M, omega: best.omega, minima, roots };
}

export function analyzePoint(mu, L, g, phi) {
  phi = reducePhi(phi);
  const r = findMinimum(mu, L, g, phi);
  return { mu, L, g, phi, phase: phaseLabel(r.M, mu, L, phi), M_star: r.M, omega_star: r.omega,
           G0: gapG0(mu, L, g, phi), roots: r.roots, local_minima: r.minima };
}

export function vacuumMass(L, g, phi) {
  phi = reducePhi(phi);
  if (phi === 0) return B0.vacuumMass(L, g);
  const f = (M) => 1 / g + partA(M, L, phi);
  if (f(0) >= 0) return 0;
  let hi = 1;
  while (f(hi) < 0) hi *= 2;
  return B0.brent(f, 0, hi);
}
