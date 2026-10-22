// njl3stab.js — second variation Phi(k, q1) of the GN action on R²×S¹ (twist φ) around the homogeneous
// solution M = M*(mu, L) (JavaScript port of njl3/stab/formulas.py).
//
//   Phi(k, q1) = ∫ dp1/(2πL) Σ_n N · (θ(E_n − mu)/E_n − θ(E_nk − mu)/E_nk) / (E_nk² − E_n²),
//   N = 2M² + q²/2,  q² = q1² + (2πk/L)²  (symmetric form of the numerator), Phi(−k) = Phi(k).
//
// Vacuum part (exact sum over n, Feynman parameter x):
//   Phi_vac = N/(4π) ∫_0^1 dx sinh(La)/(a (cosh(La) − cos 2πkx)),  a² = M² + x(1−x) q².
// Twist φ: modes 2π(n + φ)/L; Φ_vac kernel with cos 2π(kx − φ).
// Matter part, T = 0 (filled channels m_n < mu, P_n = sqrt(mu² − m_n²), c± = q1² + (2π/L)²((n+φ±k)² − (n+φ)²)):
//   Phi_mat = −(N/(2πL)) Σ_n Σ_± PV ∫_{−P_n}^{P_n} dp / (sqrt(m_n² + p²)(2 q1 p + c±))  — closed form in logs.
// T > 0: θ(E − mu) → 1 − n_F(E − mu) − n_F(E + mu); the matter integral is done numerically.
// Phi < 0 — the homogeneous phase is unstable against the fluctuation (k, q1).

const PI = Math.PI;

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
const [X32, W32] = gaussLegendre(32);
const [X16, W16] = gaussLegendre(16);

export function numerator(M, q1, k, L) {
  const q2 = 2 * PI * k / L;
  return 2 * M * M + 0.5 * (q1 * q1 + q2 * q2);
}

// Filled channels at T = 0: [{n, m, P}] with m_n = sqrt(M² + (2π(n + φ)/L)²) < mu.
export function channels(M, mu, L, phi = 0) {
  const out = [];
  if (mu <= M) return out;
  const nMax = Math.floor(L * Math.sqrt(mu * mu - M * M) / (2 * PI)) + 1;
  for (let n = -nMax - 1; n <= nMax; n++) {
    const m = Math.hypot(M, 2 * PI * (n + phi) / L);
    if (m < mu) out.push({ n, m, P: Math.sqrt(mu * mu - m * m) });
  }
  return out;
}

// Positions q1 > 0 of the logarithmic singularities (levels n and n + k both on the Fermi surface).
export function singularQ1(M, mu, L, k = 0, phi = 0) {
  const w = (2 * PI / L) ** 2, out = new Set();
  for (const { n, P } of channels(M, mu, L, phi)) {
    for (const kk of new Set([k, -k])) {                 // levels n and n ± k
      const mk2 = M * M + w * (n + phi + kk) ** 2;
      if (mk2 < mu * mu) {
        const Pk = Math.sqrt(mu * mu - mk2);
        for (const q of [P + Pk, Math.abs(P - Pk)]) if (q > 0) out.add(Math.round(q * 1e12) / 1e12);
      }
    }
  }
  return [...out].sort((a, b) => a - b);
}

export function phiVac(k, q1, M, L, phi = 0) {
  k = Math.abs(k);
  const nSub = Math.max(4, 4 * k);
  const q = Math.hypot(q1, 2 * PI * k / L);
  let edges = [];
  for (let s = 0; s <= nSub; s++) edges.push(s / nSub);
  if (M < 1e-2) {
    // small mass: peaks of width ~M²/q² at x = 0, 1 (massless modes) — geometric grading to the ends
    const g = [];
    for (let i = 0; i < 48; i++) g.push(1e-16 * (0.5 / nSub / 1e-16) ** (i / 47));
    edges = [...new Set([...edges, ...g, ...g.map((x) => 1 - x)])].sort((a, b) => a - b);
  }
  let I = 0;
  for (let s = 0; s + 1 < edges.length; s++) {
    const h = 0.5 * (edges[s + 1] - edges[s]), c = 0.5 * (edges[s + 1] + edges[s]);
    for (let i = 0; i < 32; i++) {
      const x = c + h * X32[i];
      const a = Math.sqrt(M * M + x * (1 - x) * q * q);
      const ex = Math.exp(-Math.min(L * a, 700));
      I += h * W32[i] * (1 - ex * ex) / (a * (1 + ex * ex - 2 * Math.cos(2 * PI * (k * x - phi)) * ex));
    }
  }
  return numerator(M, q1, k, L) / (4 * PI) * I;
}

// Σ over the couplings n → n ± k (for φ ∉ {0, ½} the two are different; for k = 0 one, weight 2)
export function phiMat(k, q1, M, mu, L, phi = 0) {
  const w2 = (2 * PI / L) ** 2;
  const a = 2 * q1, ks = k ? [k, -k] : [0], wt = k ? 1 : 2;
  let s = 0;
  for (const { n, m, P } of channels(M, mu, L, phi)) {
    const nu = n + phi;
    const uP = (P + mu) / m, uM = (mu - P) / m;              // u(±P), sqrt(m² + P²) = mu
    for (const kk of ks) {
      const c = q1 * q1 + w2 * ((nu + kk) ** 2 - nu * nu);
      if (q1 === 0) { s += wt * 2 * Math.asinh(P / m) / c; continue; }
      const D = Math.sqrt(c * c + a * a * m * m);
      const up = (-c + D) / (a * m), um = (-c - D) / (a * m);
      s += wt * (Math.log(Math.abs((uP - up) / (uP - um))) - Math.log(Math.abs((uM - up) / (uM - um)))) / D;
    }
  }
  return -numerator(M, q1, k, L) / (2 * PI * L) * s;
}

function nF(x, T) {
  if (x > 40 * T) return Math.exp(-x / T);
  if (x < -40 * T) return 1 - Math.exp(x / T);
  return 1 / (Math.exp(x / T) + 1);
}

// ∫_a^b g with geometric grading towards both ends (resolves Fermi steps of width ~T at the break points
// and, with T → min(T, m), the peak of 1/E of width m at p = 0 for a light mode).
function gradedIntegral(g, a, b, T) {
  if (!(b > a)) return 0;
  const mid = 0.5 * (a + b), pts = [a];
  for (let d = 0.25 * T; a + d < mid; d *= 3) pts.push(a + d);
  const right = [];
  for (let d = 0.25 * T; b - d > mid; d *= 3) right.push(b - d);
  pts.push(mid, ...right.reverse(), b);
  let tot = 0;
  for (let j = 0; j + 1 < pts.length; j++) {
    const lo = pts[j], hi = pts[j + 1], h = 0.5 * (hi - lo), c = 0.5 * (hi + lo);
    if (!(hi > lo)) continue;
    let s = 0;
    for (let i = 0; i < 16; i++) s += W16[i] * g(c + h * X16[i]);
    tot += h * s;
  }
  return tot;
}

export function phiMatT(k, q1, M, mu, L, T, phi = 0) {
  if (T === 0) return phiMat(k, q1, M, mu, L, phi);
  const w2 = (2 * PI / L) ** 2, Emax = mu + 40 * T;
  if (Emax <= M) return 0;
  const nMax = Math.floor(Math.sqrt(Emax * Emax - M * M) * L / (2 * PI)) + 1;
  const ks = k ? [k, -k] : [0], wt = k ? 1 : 2;
  let s = 0;
  for (let n = -nMax - 1; n <= nMax; n++) {
    const nu = n + phi, m = Math.sqrt(M * M + w2 * nu * nu);
    if (m >= Emax) continue;
    const pm = Math.sqrt(Emax * Emax - m * m);
    const f = (p) => { const E = Math.sqrt(m * m + p * p); return (nF(E - mu, T) + nF(E + mu, T)) / E; };
    const Tg = Math.min(T, m);                             // grading scale
    for (const kk of ks) {
      const c = q1 * q1 + w2 * ((nu + kk) ** 2 - nu * nu);
      const br = [-pm, pm];
      if (m < 1e-2) br.push(0);                            // light mode: peak of 1/E at p = 0
      if (m < mu) { const P = Math.sqrt(mu * mu - m * m); br.push(-P, P); }
      if (q1 === 0) {
        br.sort((u, v) => u - v);
        for (let j = 0; j + 1 < br.length; j++) s += wt * gradedIntegral(f, br[j], br[j + 1], Tg) / c;
        continue;
      }
      const p0 = -c / (2 * q1);
      let g = (p) => f(p) / (p - p0), extra = 0;
      if (p0 > -pm && p0 < pm) {
        const f0 = f(p0);
        g = (p) => (p === p0 ? 0 : (f(p) - f0) / (p - p0));
        extra = f0 * Math.log((pm - p0) / (p0 + pm));
        br.push(p0);
      }
      br.sort((u, v) => u - v);
      let I = extra;
      for (let j = 0; j + 1 < br.length; j++) I += gradedIntegral(g, br[j], br[j + 1], Tg);
      s += wt * I / (2 * q1);
    }
  }
  return -numerator(M, q1, k, L) / (2 * PI * L) * s;
}

export function phi(k, q1, M, mu, L, T = 0, phi_ = 0) {
  return phiVac(k, q1, M, L, phi_) + phiMatT(k, q1, M, mu, L, T, phi_);
}

// Phase T (M* = 0): Phi = ½ G(0⁺; mu, L, g, φ) + [loop at the regulator mass M_ZERO] (njl3/stab/formulas.py:
// phi_zero). The constant ½ G(0⁺) is added by the caller (gapG0 from lab/js/njl3hom.js).
export const M_ZERO = 1e-4;
