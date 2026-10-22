// Фоновые расчёты web/fast: строки карты, рельеф Ω(M, b) в точке, все локальные минимумы.
import { setTable, hasTable, checkPhi, omegaRel } from '../../lab/js/njl3inh.js';
import { findMinimum, allMinima } from './fastmin.js';

async function ensure(phi) {
  if (hasTable(phi)) return;
  const r = await fetch(new URL(`../../lab/data/inh/s_table_phi${phi}.json`, import.meta.url));
  setTable(phi, await r.json());
}

self.onmessage = async ({ data }) => {
  const { type, job } = data;
  try {
    const phi = checkPhi(data.phi || 0);
    await ensure(phi);
    const { g, T } = data;
    if (type === 'map') {
      for (const { i, L } of data.rows) {
        const n = data.mu.length, lnM = new Float64Array(n), b = new Float64Array(n), om = new Float64Array(n);
        const phase = new Array(n);
        const t0 = performance.now();
        for (let j = 0; j < n; j++) {
          const r = findMinimum(data.mu[j], L, g, phi, T);
          lnM[j] = r.ln_M_star; b[j] = r.b_star; om[j] = r.omega_star; phase[j] = r.phase;
        }
        self.postMessage({ type, job, i, lnM, b, om, phase, ms: performance.now() - t0 }, [lnM.buffer, b.buffer, om.buffer]);
      }
      self.postMessage({ type, job, done: true });
    } else if (type === 'point') {
      self.postMessage({ type, job, result: findMinimum(data.mu, data.L, g, phi, T) });
    } else if (type === 'landscape') {
      for (let i = 0; i < data.b.length; i++) {
        const row = new Float64Array(data.M.length);
        for (let j = 0; j < data.M.length; j++) row[j] = omegaRel(data.M[j], data.b[i], data.mu, data.L, g, phi);
        self.postMessage({ type, job, i, row }, [row.buffer]);
      }
      self.postMessage({ type, job, done: true });
    } else if (type === 'minima') {
      self.postMessage({ type, job, minima: allMinima(data.mu, data.L, g, phi) });
    }
  } catch (e) {
    self.postMessage({ type: 'error', job, message: String((e && e.stack) || e) });
  }
};
