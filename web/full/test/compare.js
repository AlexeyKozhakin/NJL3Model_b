// Сверка JS-ядра (web/full/js/njl3full.js) с эталоном Python (njl3.full).
// node web/full/test/compare.js golden.json  -> JSON с максимальными расхождениями
import { readFileSync } from 'node:fs';
import { init, sFunc, omegaRel, channelAsymptotics, findMinimum } from '../js/njl3full.js';

init(JSON.parse(readFileSync(new URL('../data/s_table.json', import.meta.url), 'utf8')));
const gd = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const g = gd.g;
const err = { s: 0, omega: 0, A: 0, lnM_asym: 0, phase_mismatch: 0, lnM_star: 0, b_star: 0, omega_star_rel: 0 };
for (const p of gd.s) err.s = Math.max(err.s, Math.abs(sFunc(p.x) - p.s));
for (const p of gd.omega) err.omega = Math.max(err.omega, Math.abs(omegaRel(p.M, p.b, p.mu, p.L, g) - p.omega) / Math.max(1, Math.abs(p.omega)));
for (const p of gd.channels) {
  const a = channelAsymptotics(p.S, p.C, p.mu, p.L, g);
  err.A = Math.max(err.A, Math.abs(a.A - p.A)); err.lnM_asym = Math.max(err.lnM_asym, Math.abs(a.lnM - p.lnM));
}
for (const p of gd.points) {
  const r = findMinimum(p.mu, p.L, g, p.T || 0);
  if (r.phase !== p.phase) { err.phase_mismatch++; continue; }
  // |Ω*| у шума (~1e-12): яма плоская, M* численно не определён — сверяется только фаза
  if (p.omega !== 0 && Math.abs(p.omega) < 1e-10) continue;
  if (p.lnM !== null) err.lnM_star = Math.max(err.lnM_star, Math.abs(r.ln_M_star - p.lnM));
  if (p.b !== null) err.b_star = Math.max(err.b_star, Math.abs(r.b_star - p.b));
  if (p.omega !== 0) err.omega_star_rel = Math.max(err.omega_star_rel, Math.abs(r.omega_star - p.omega) / Math.abs(p.omega));
}
console.log(JSON.stringify(err));
