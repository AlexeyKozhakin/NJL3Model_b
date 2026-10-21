// node web/lab/test/compare_hom.js — JS port njl3hom.js vs Python njl3.hom.
import { readFileSync } from 'node:fs';
import { findMinimum, gapG, omegaRel, phaseLabel } from '../js/njl3hom.js';

const rows = JSON.parse(readFileSync(new URL('./expected_hom.json', import.meta.url)));
let w = { G: 0, omega: 0, M: 0 }, bad = 0;
for (const r of rows) {
  const m = findMinimum(r.mu, r.L, r.g, r.phi);
  w.M = Math.max(w.M, Math.abs(m.M - r.M_star));
  if (phaseLabel(m.M, r.mu, r.L, r.phi) !== r.phase) { bad++; console.log('phase', r, m.M); }
  r.Ms.forEach((M, i) => {
    w.G = Math.max(w.G, Math.abs(gapG(M, r.mu, r.L, r.g, r.phi) - r.G[i]));
    w.omega = Math.max(w.omega, Math.abs(omegaRel(M, r.mu, r.L, r.g, r.phi) - r.omega[i]));
  });
}
console.log(`${rows.length} points, max |JS − Python|:`, w, 'phase mismatches:', bad);
if (bad || w.G > 1e-12 || w.omega > 1e-10 || w.M > 1e-9) { console.error('FAIL'); process.exit(1); }
console.log('OK');
