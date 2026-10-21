// node web/stab/test/compare.js — JS port vs Python (web/stab/test/expected.json).
import { readFileSync } from 'node:fs';
import { phiVac, phiMat, phi } from '../js/njl3stab.js';

const rows = JSON.parse(readFileSync(new URL('./expected.json', import.meta.url)));
let worst = { vac: 0, mat: 0, T3: 0 };
for (const r of rows) {
  const v = phiVac(r.k, r.q1, r.M, r.L), m = phiMat(r.k, r.q1, r.M, r.mu, r.L), t = phi(r.k, r.q1, r.M, r.mu, r.L, 1e-3);
  worst.vac = Math.max(worst.vac, Math.abs(v - r.vac));
  worst.mat = Math.max(worst.mat, Math.abs(m - r.mat));
  worst.T3 = Math.max(worst.T3, Math.abs(t - r.T3));
}
console.log(`${rows.length} rows, max |JS − Python|:`, worst);
const ok = worst.vac < 1e-10 && worst.mat < 1e-10 && worst.T3 < 1e-6;
if (!ok) { console.error('FAIL'); process.exit(1); }
console.log('OK');
