// Фоновые расчёты вкладки «Неоднородный, T > 0»: минимум в точке, карта Ω_T(M, b), срезы.
import { setTable, hasTable, checkPhi } from './njl3inh.js';
import { findMinimumT, omegaRelT } from './njl3inhT.js';

async function ensure(phi) {
  if (hasTable(phi)) return;
  const r = await fetch(new URL(`../data/inh/s_table_phi${phi}.json`, import.meta.url));
  setTable(phi, await r.json());
}

self.onmessage = async ({ data }) => {
  const { type, job } = data;
  try {
    const phi = checkPhi(data.phi || 0);
    await ensure(phi);
    handle(type, job, { ...data, phi });
  } catch (e) {
    self.postMessage({ type: 'error', job, message: String((e && e.stack) || e) });
  }
};

function handle(type, job, d) {
  const { mu, L, g, phi, T } = d;
  if (type === 'point') {
    self.postMessage({ type, job, result: findMinimumT(mu, L, g, phi, T) });
  } else if (type === 'omega-map') {
    for (let i = 0; i < d.b.length; i++) {
      const row = new Float64Array(d.M.length);
      for (let j = 0; j < d.M.length; j++) row[j] = omegaRelT(d.M[j], d.b[i], mu, L, g, phi, T);
      self.postMessage({ type, job, i, row }, [row.buffer]);
    }
    self.postMessage({ type, job, done: true });
  } else if (type === 'slice') {
    const om = new Float64Array(d.M.length);
    for (let i = 0; i < d.M.length; i++) om[i] = omegaRelT(d.M[i], d.b[i], mu, L, g, phi, T);
    self.postMessage({ type, job, om }, [om.buffer]);
  }
}
