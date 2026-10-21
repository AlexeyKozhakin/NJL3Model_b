// node web/lab/test/compare_inh.js — JS port njl3inh.js vs Python njl3.inh.
import { readFileSync } from 'node:fs';
import { setTable, sFunc, omegaRel, findMinimum, PHIS } from '../js/njl3inh.js';

for (const phi of PHIS) setTable(phi, JSON.parse(readFileSync(new URL(`../data/inh/s_table_phi${phi}.json`, import.meta.url))));
const gd = JSON.parse(readFileSync(new URL('./expected_inh.json', import.meta.url)));
const err = { s: 0, omega: 0, phase_mismatch: 0, lnM: 0, b: 0, omega_star_rel: 0 };
for (const p of gd.s) err.s = Math.max(err.s, Math.abs(sFunc(p.x, p.phi) - p.s));
for (const p of gd.omega) err.omega = Math.max(err.omega, Math.abs(omegaRel(p.M, p.b, p.mu, p.L, p.g, p.phi) - p.omega) / Math.max(1, Math.abs(p.omega)));
for (const p of gd.points) {
  const r = findMinimum(p.mu, p.L, p.g, p.phi, p.T);
  if (r.phase !== p.phase) { err.phase_mismatch++; console.log('phase', p, r.phase); continue; }
  if (p.omega !== 0 && Math.abs(p.omega) < 1e-10) continue;
  if (p.lnM !== null) err.lnM = Math.max(err.lnM, Math.abs(r.ln_M_star - p.lnM));
  if (p.b !== null) err.b = Math.max(err.b, Math.abs(r.b_star - p.b));
  if (p.omega !== 0) err.omega_star_rel = Math.max(err.omega_star_rel, Math.abs(r.omega_star - p.omega) / Math.abs(p.omega));
}
console.log(JSON.stringify(err));
if (err.s > 1e-14 || err.omega > 1e-12 || err.phase_mismatch || err.lnM > 1e-4 || err.b > 1e-4 || err.omega_star_rel > 1e-6) { console.error('FAIL'); process.exit(1); }
console.log('OK');
