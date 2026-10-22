// node web/fast/test/compare_fast.mjs — JS fastmin.js vs Python njl3.fast (expected_fast.json); timing.
import { readFileSync } from 'node:fs';
import { setTable, PHIS } from '../../lab/js/njl3inh.js';
import { findMinimum } from '../js/fastmin.js';

for (const phi of PHIS) setTable(phi, JSON.parse(readFileSync(new URL(`../../lab/data/inh/s_table_phi${phi}.json`, import.meta.url))));
const rows = JSON.parse(readFileSync(new URL('./expected_fast.json', import.meta.url)));
let bad = 0, dl = 0, db = 0, dom = 0, tsum = 0, tmax = 0;
for (const p of rows) {
  const t = performance.now();
  const r = findMinimum(p.mu, p.L, p.g, p.phi, p.T);
  const dt = performance.now() - t; tsum += dt; tmax = Math.max(tmax, dt);
  if (r.phase !== p.phase) { bad++; console.log('phase', p.mu, p.L, p.g, p.phi, p.phase, r.phase, p.omega, r.omega_star); continue; }
  if (p.lnM !== null) dl = Math.max(dl, Math.abs(r.ln_M_star - p.lnM));
  if (p.b !== null) db = Math.max(db, Math.abs(r.b_star - p.b));
  dom = Math.max(dom, Math.abs(r.omega_star - p.omega) / Math.max(1e-12, Math.abs(p.omega)));
}
console.log(`${rows.length} points: phase mismatch ${bad}, max |Δln M| ${dl.toExponential(2)}, |Δb| ${db.toExponential(2)}, rel ΔΩ ${dom.toExponential(2)}; ` +
            `${(tsum / rows.length).toFixed(1)} ms/point (max ${tmax.toFixed(0)} ms)`);
if (bad || dl > 1e-4 || db > 1e-4) { console.error('FAIL'); process.exit(1); }
console.log('OK');
