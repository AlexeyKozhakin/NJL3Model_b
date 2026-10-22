// node web/stab/test/compare.js — JS port vs Python (web/stab/test/expected.json).
import { readFileSync } from 'node:fs';
import { phiVac, phiMat, phi } from '../js/njl3stab.js';

const rows = JSON.parse(readFileSync(new URL('./expected.json', import.meta.url)));
let worst = { vac: 0, mat: 0, T3: 0 }, worstZ = { vac: 0, T3: 0 };
for (const r of rows) {
  const f = r.phi || 0;
  const v = phiVac(r.k, r.q1, r.M, r.L, f), m = phiMat(r.k, r.q1, r.M, r.mu, r.L, f), t = phi(r.k, r.q1, r.M, r.mu, r.L, 1e-3, f);
  worst.mat = Math.max(worst.mat, Math.abs(m - r.mat));
  // phase T rows (regulator mass 1e-4): Python's adaptive quad vs the graded fixed grid agree to ~1e-8
  const w = r.M < 1e-2 ? worstZ : worst;
  w.vac = Math.max(w.vac, Math.abs(v - r.vac));
  w.T3 = Math.max(w.T3, Math.abs(t - r.T3));
}
console.log(`${rows.length} rows, max |JS − Python|:`, worst, 'M = 1e-4 (phase T):', worstZ);
const ok = worst.vac < 1e-10 && worst.mat < 1e-10 && worst.T3 < 1e-6 && worstZ.vac < 1e-7 && worstZ.T3 < 1e-6;
if (!ok) { console.error('FAIL'); process.exit(1); }
console.log('OK');
