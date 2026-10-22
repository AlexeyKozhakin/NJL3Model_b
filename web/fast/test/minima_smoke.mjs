// node web/fast/test/minima_smoke.mjs — all local minima at the "rich" points (timing, list).
import { readFileSync } from 'node:fs';
import { setTable, PHIS } from '../../lab/js/njl3inh.js';
import { allMinima, findMinimum } from '../js/fastmin.js';

for (const phi of PHIS) setTable(phi, JSON.parse(readFileSync(new URL(`../../lab/data/inh/s_table_phi${phi}.json`, import.meta.url))));
for (const [mu, L, phi] of [[3.1409, 10, 0], [3.1273, 4.8667, 0.5]]) {
  let t = performance.now();
  const mins = allMinima(mu, L, -1, phi);
  const tm = performance.now() - t;
  t = performance.now();
  const r = findMinimum(mu, L, -1, phi, 1e-3);
  console.log(`mu=${mu} L=${L} phi=${phi}: ${mins.length} minima (${tm.toFixed(0)} ms); findMinimum ${r.phase} M*=${r.M_star.toPrecision(5)} ` +
              `b*=${r.b_star.toPrecision(5)} Ω*=${r.omega_star.toExponential(4)} (${(performance.now() - t).toFixed(0)} ms)`);
  for (const m of mins.slice(0, 8)) console.log(`   ${m.phase.padEnd(4)} M=${m.M.toPrecision(5)} b=${m.b.toPrecision(5)} Ω=${m.omega.toExponential(4)}`);
  if (!mins.length || mins[0].omega < r.omega_star - 1e-9 * Math.abs(r.omega_star) - 1e-12)
    console.log('   NOTE: the dense scan found a deeper well than findMinimum');
}
