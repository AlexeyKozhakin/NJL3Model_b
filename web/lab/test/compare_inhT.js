// node web/lab/test/compare_inhT.js — JS port njl3inhT.js vs Python njl3.fast.thermal / minimize_T.
import { readFileSync } from 'node:fs';
import { setTable, PHIS } from '../js/njl3inh.js';
import { omegaRelT, findMinimumT } from '../js/njl3inhT.js';

for (const phi of PHIS) setTable(phi, JSON.parse(readFileSync(new URL(`../data/inh/s_table_phi${phi}.json`, import.meta.url))));
const gd = JSON.parse(readFileSync(new URL('./expected_inhT.json', import.meta.url)));
const err = { omega: 0, phase_mismatch: 0, M_rel: 0, b: 0, omega_star_rel: 0 };
for (const p of gd.omega) err.omega = Math.max(err.omega, Math.abs(omegaRelT(p.M, p.b, p.mu, p.L, p.g, p.phi, p.T) - p.omega));
const t0 = Date.now();
for (const p of gd.points) {
  const r = findMinimumT(p.mu, p.L, p.g, p.phi, p.T);
  if (r.phase !== p.phase) { err.phase_mismatch++; console.log('phase', p, r.phase, r.M_star, r.b_star); continue; }
  if (p.M > 0) err.M_rel = Math.max(err.M_rel, Math.abs(r.M_star - p.M) / p.M);
  if (p.b !== null && p.M > 0) err.b = Math.max(err.b, Math.abs(r.b_star - p.b));
  if (p.omega !== 0) err.omega_star_rel = Math.max(err.omega_star_rel, Math.abs(r.omega_star - p.omega) / Math.abs(p.omega));
}
console.log(JSON.stringify(err), `${gd.points.length} points in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
if (err.omega > 1e-10 || err.phase_mismatch || err.M_rel > 1e-4 || err.b > 1e-4 || err.omega_star_rel > 1e-6) { console.error('FAIL'); process.exit(1); }
console.log('OK');
