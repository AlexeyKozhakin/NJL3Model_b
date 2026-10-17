// njl3b0.js — фазовая диаграмма NJL3 на R²×S¹ при b = 0, phi = 0 (порт njl3/b0 на JavaScript).
//
// Обозначения: k_n = 2πn/L, m_n = sqrt(M² + k_n²), P_n = sqrt(mu² − m_n²),
// S_n = sqrt(mu² − k_n²) (порог моды n по M), C_0 = 1, C_{n≥1} = 2.
//
//   G(M)  = 1/g + M/π + (2/(πL)) [ln(1 − e^{−LM}) + Σ_{m_n<mu} C_n arccosh(mu/m_n)],  dΩ/dM = M·G
//   G(0⁺) = 1/g + (2/(πL)) [ln(2 mu L) + 2 Σ_{n=1}^{n_max} arccosh(mu/k_n)]
//   Ω(M) − Ω(0) = ∫_0^M m G(m) dm
//
// Гарантированный поиск корней: на [0, mu] G = 1/g + A(M) + B(M), A возрастает, B убывает,
// откуда вилка 1/g + A(p) + B(q) ≤ G ≤ 1/g + A(q) + B(p); на [mu, ∞) G возрастает.
// Эталон — Python-пакет njl3.b0.

const PI = Math.PI;

// ---------------------------------------------------------------------------
// Моды
// ---------------------------------------------------------------------------

export function modes(mu, L) {
  if (mu <= 0) return [];
  const nMax = Math.floor(L * mu / (2 * PI));
  const out = [];
  for (let n = 0; n <= nMax; n++) out.push({ k: 2 * PI * n / L, C: n === 0 ? 1 : 2 });
  return out;
}

export function thresholds(mu, L) {
  return modes(mu, L).map(({ k }) => Math.sqrt(mu * mu - k * k)).sort((a, b) => a - b);
}

export function filledMode(M, mu, L) {
  return Math.floor(L * Math.sqrt(mu * mu - M * M) / (2 * PI));
}

// ---------------------------------------------------------------------------
// Уравнение щели (устойчивая форма)
// ---------------------------------------------------------------------------

export function gapG0(mu, L, g) {
  if (mu <= 0) return -Infinity;
  let s = Math.log(2 * mu * L);
  const md = modes(mu, L);
  for (let i = 1; i < md.length; i++) s += md[i].C * Math.acosh(mu / md[i].k);
  return 1 / g + 2 / (PI * L) * s;
}

export function gapG(M, mu, L, g) {
  if (M === 0) return gapG0(mu, L, g);
  const x = L * M;
  let s;
  if (M < mu) {
    // ln(1 − e^{−x}) + ln((mu + P0)/M) = ln[(1 − e^{−x})/x] + ln[L(mu + P0)]
    const P0 = Math.sqrt(mu * mu - M * M);
    s = Math.log(-Math.expm1(-x) / x) + Math.log(L * (mu + P0));
  } else {
    s = Math.log(-Math.expm1(-x));
  }
  const md = modes(mu, L);
  for (let i = 1; i < md.length; i++) {
    const m = Math.sqrt(M * M + md[i].k * md[i].k);
    if (m < mu) s += md[i].C * Math.log((mu + Math.sqrt(mu * mu - m * m)) / m);
  }
  return 1 / g + M / PI + 2 / (PI * L) * s;
}

export function G2(mu, L) {
  let s = L * L / 24 - 1 / (4 * mu * mu);
  const md = modes(mu, L);
  for (let i = 1; i < md.length; i++) {
    const S2 = mu * mu - md[i].k * md[i].k;
    if (S2 <= 0) return -Infinity;
    s -= md[i].C * mu / (2 * md[i].k * md[i].k * Math.sqrt(S2));
  }
  return 2 / (PI * L) * s;
}

function partA(M, L) {
  if (M === 0) return 0;
  const x = L * M;
  return M / PI + 2 / (PI * L) * Math.log(-Math.expm1(-x) / x);
}

function partB(M, mu, L) {
  let s = Math.log(L * (mu + Math.sqrt(Math.max(mu * mu - M * M, 0))));
  const md = modes(mu, L);
  for (let i = 1; i < md.length; i++) {
    const m = Math.sqrt(M * M + md[i].k * md[i].k);
    if (m < mu) s += md[i].C * Math.acosh(mu / m);
  }
  return 2 / (PI * L) * s;
}

// ---------------------------------------------------------------------------
// Метод Брента
// ---------------------------------------------------------------------------

export function brent(f, a, b, xtol = 1e-15, maxIter = 200) {
  let fa = f(a), fb = f(b);
  if (fa === 0) return a;
  if (fb === 0) return b;
  if (fa * fb > 0) throw new Error('brent: нет смены знака');
  let c = a, fc = fa, d = b - a, e = d;
  for (let it = 0; it < maxIter; it++) {
    if (fb * fc > 0) { c = a; fc = fa; d = b - a; e = d; }
    if (Math.abs(fc) < Math.abs(fb)) { a = b; b = c; c = a; fa = fb; fb = fc; fc = fa; }
    const tol = 2 * Number.EPSILON * Math.abs(b) + 0.5 * xtol;
    const m = 0.5 * (c - b);
    if (Math.abs(m) <= tol || fb === 0) return b;
    if (Math.abs(e) >= tol && Math.abs(fa) > Math.abs(fb)) {
      let p, q, r;
      const s = fb / fa;
      if (a === c) { p = 2 * m * s; q = 1 - s; }
      else {
        q = fa / fc; r = fb / fc;
        p = s * (2 * m * q * (q - r) - (b - a) * (r - 1));
        q = (q - 1) * (r - 1) * (s - 1);
      }
      if (p > 0) q = -q; else p = -p;
      if (2 * p < Math.min(3 * m * q - Math.abs(tol * q), Math.abs(e * q))) { e = d; d = p / q; }
      else { d = m; e = m; }
    } else { d = m; e = m; }
    a = b; fa = fb;
    b += Math.abs(d) > tol ? d : (m > 0 ? tol : -tol);
    fb = f(b);
  }
  return b;
}

// ---------------------------------------------------------------------------
// Гарантированное отделение корней G
// ---------------------------------------------------------------------------

export function isolateRoots(mu, L, g, tol = 1e-10) {
  const G = (x) => gapG(x, mu, L, g);
  const ig = 1 / g;
  const roots = [];
  if (mu > 0) {
    const windows = [];
    const stack = [[0, mu, partA(0, L), partA(mu, L), partB(0, mu, L), partB(mu, mu, L)]];
    while (stack.length) {
      const [p, q, Ap, Aq, Bp, Bq] = stack.pop();
      if (ig + Ap + Bq > 0 || ig + Aq + Bp < 0) continue;
      if (q - p <= tol) { windows.push([p, q]); continue; }
      const c = 0.5 * (p + q), Ac = partA(c, L), Bc = partB(c, mu, L);
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
      if (p === 0 && q <= tol) continue;          // окрестность M = 0 — кандидат M = 0 отдельно
      const Gp = G(p), Gq = G(q);
      if (Gp < 0 && Gq > 0) roots.push({ M: brent(G, p, q), kind: 'min' });
      else if (Gp > 0 && Gq < 0) roots.push({ M: brent(G, p, q), kind: 'max' });
      else roots.push({ M: 0.5 * (p + q), kind: 'touch' });
    }
  }
  const a = Math.max(mu, 1e-300);
  if (G(a) < 0) {
    let b = 2 * a + 1;
    while (G(b) < 0) b *= 2;
    roots.push({ M: brent(G, a, b), kind: 'min' });
  }
  return roots.sort((u, v) => u.M - v.M);
}

// ---------------------------------------------------------------------------
// Ω(M) − Ω(0) = ∫_0^M m G(m) dm  (Гаусс–Лежандр, замена m = S − u² у порогов)
// ---------------------------------------------------------------------------

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
const [GL_X, GL_W] = gaussLegendre(48);

function integrateSegment(a, b, S, mu, L, g) {
  let total = 0;
  if (S === null || S - b >= b - a) {
    const h = 0.5 * (b - a), c = 0.5 * (a + b);
    for (let i = 0; i < GL_X.length; i++) { const m = h * GL_X[i] + c; total += GL_W[i] * m * gapG(m, mu, L, g); }
    return h * total;
  }
  const ua = Math.sqrt(S - b), ub = Math.sqrt(S - a);
  const h = 0.5 * (ub - ua), c = 0.5 * (ua + ub);
  for (let i = 0; i < GL_X.length; i++) {
    const u = h * GL_X[i] + c, m = Math.max(S - u * u, 1e-300);
    total += GL_W[i] * 2 * u * m * gapG(m, mu, L, g);
  }
  return h * total;
}

export function omegaRel(M, mu, L, g) {
  if (M === 0) return 0;
  const S = thresholds(mu, L);
  const edges = [0, ...S.filter((s) => s > 0 && s < M)];
  // за порогами (M > mu) функция гладкая; длинные куски режем по 1
  let last = edges[edges.length - 1];
  while (M - last > 1 && last >= (S.length ? S[S.length - 1] : 0)) { last += 1; edges.push(last); }
  edges.push(M);
  let total = 0;
  for (let i = 0; i + 1 < edges.length; i++) {
    const a = edges[i], b = edges[i + 1];
    if (b <= a) continue;
    const Snext = S.find((s) => s >= b);
    total += integrateSegment(a, b, Snext === undefined ? null : Snext, mu, L, g);
  }
  return total;
}

// ---------------------------------------------------------------------------
// Глобальный минимум и фаза
// ---------------------------------------------------------------------------

export function phaseLabel(M, mu, L) {
  if (M === 0) return 'T';
  if (M >= mu) return 'V';
  return 'I' + filledMode(M, mu, L);
}

export function findMinimum(mu, L, g, roots = null) {
  roots = roots || isolateRoots(mu, L, g);
  const minima = roots.filter((r) => r.kind !== 'max').map((r) => ({ M: r.M, omega: omegaRel(r.M, mu, L, g) }));
  let best = { M: 0, omega: 0 };
  for (const m of minima) if (m.omega < best.omega) best = m;
  return { M: best.M, omega: best.omega, minima, roots };
}

export function analyzePoint(mu, L, g) {
  const r = findMinimum(mu, L, g);
  return { mu, L, g, phase: phaseLabel(r.M, mu, L), M_star: r.M, omega_star: r.omega,
           G0: gapG0(mu, L, g), G2: G2(mu, L), roots: r.roots, local_minima: r.minima };
}

export function vacuumMass(L, g) {
  const f = (M) => 1 / g + M / PI + 2 / (PI * L) * Math.log(-Math.expm1(-L * M));
  let hi = 1;
  while (f(hi) < 0) hi *= 2;
  return brent(f, 1e-300, hi);
}

export function peak(n, g) {
  let s = Math.log(4 * PI * n);
  for (let m = 1; m < n; m++) s += 2 * Math.acosh(n / m);
  const L = -2 * g / PI * s;
  return { n, L, mu: 2 * PI * n / L };
}
