// Фоновые расчёты вкладки «Неоднородный случай»: минимум в точке, карта Ω(M, b), кривая канала, срезы.
import { setTable, hasTable, checkPhi, findMinimum, omegaRel, channelAsymptotics } from './njl3inh.js';

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
  const { mu, L, g, phi } = d;
  if (type === 'point') {
    self.postMessage({ type, job, result: findMinimum(mu, L, g, phi, d.T || 0) });
  } else if (type === 'omega-map') {
    for (let i = 0; i < d.b.length; i++) {
      const row = new Float64Array(d.M.length);
      for (let j = 0; j < d.M.length; j++) row[j] = omegaRel(d.M[j], d.b[i], mu, L, g, phi);
      self.postMessage({ type, job, i, row }, [row.buffer]);
    }
    self.postMessage({ type, job, done: true });
  } else if (type === 'slice') {
    const om = new Float64Array(d.M.length);
    for (let i = 0; i < d.M.length; i++) om[i] = omegaRel(d.M[i], d.b[i], mu, L, g, phi);
    self.postMessage({ type, job, om }, [om.buffer]);
  } else if (type === 'channel') {
    const asy = channelAsymptotics(d.S, d.C, mu, L, g, phi);
    const num = d.lnM.map((lm) => omegaRel(Math.exp(lm), d.S, mu, L, g, phi) / Math.exp(2 * lm));
    const as = d.lnM.map((lm) => asy.A + asy.c * lm);
    self.postMessage({ type, job, num, as, asy });
  }
}
