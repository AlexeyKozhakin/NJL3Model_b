// Сверка JS-ядра (web/b0/js/njl3b0.js) с эталоном Python (njl3.b0).
// Запуск: node web/b0/test/compare.js golden.json   -> печатает JSON с максимальными расхождениями.
import { readFileSync } from 'node:fs';
import { gapG, gapG0, G2, omegaRel, isolateRoots, findMinimum, phaseLabel, vacuumMass, peak } from '../js/njl3b0.js';

const golden = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const g = golden.g;
const err = { G: 0, G0: 0, G2: 0, omega_abs: 0, omega_rel: 0, roots: 0, root_count_mismatch: 0,
              M_star: 0, phase_mismatch: 0, M_vac: 0, peaks: 0 };

for (const p of golden.functions) {
  err.G = Math.max(err.G, Math.abs(gapG(p.M, p.mu, p.L, g) - p.G));
  err.G0 = Math.max(err.G0, Math.abs(gapG0(p.mu, p.L, g) - p.G0));
  if (Number.isFinite(p.G2)) err.G2 = Math.max(err.G2, Math.abs(G2(p.mu, p.L) - p.G2));
  const o = omegaRel(p.M, p.mu, p.L, g);
  err.omega_abs = Math.max(err.omega_abs, Math.abs(o - p.omega));
  err.omega_rel = Math.max(err.omega_rel, Math.abs(o - p.omega) / Math.max(Math.abs(p.omega), 1e-300));
}
for (const p of golden.points) {
  const roots = isolateRoots(p.mu, p.L, g);
  if (roots.length !== p.roots.length) err.root_count_mismatch++;
  else roots.forEach((r, i) => { err.roots = Math.max(err.roots, Math.abs(r.M - p.roots[i])); });
  const r = findMinimum(p.mu, p.L, g, roots);
  err.M_star = Math.max(err.M_star, Math.abs(r.M - p.M_star));
  if (phaseLabel(r.M, p.mu, p.L) !== p.phase) err.phase_mismatch++;
}
for (const v of golden.vacuum) err.M_vac = Math.max(err.M_vac, Math.abs(vacuumMass(v.L, g) - v.M_vac));
for (const v of golden.peaks) {
  const q = peak(v.n, g);
  err.peaks = Math.max(err.peaks, Math.abs(q.L - v.L), Math.abs(q.mu - v.mu));
}
console.log(JSON.stringify(err));
