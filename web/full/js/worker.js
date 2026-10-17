// Фоновые расчёты: глобальный минимум в точке, карта Ω(M, b), кривая канала.
import { init, findMinimum, omegaRel, channelAsymptotics } from './njl3full.js';

const ready = fetch(new URL('../data/s_table.json', import.meta.url)).then((r) => r.json()).then(init);

self.onmessage = async ({ data }) => {
  const { type, job } = data;
  try {
    await ready;
    handle(type, job, data);
  } catch (e) {
    self.postMessage({ type: 'error', job, message: String(e && e.stack || e) });
  }
};

function handle(type, job, data) {
  if (type === 'point') {
    self.postMessage({ type, job, result: findMinimum(data.mu, data.L, data.g, data.T || 0) });
  } else if (type === 'omega-map') {
    const { mu, L, g, M, b } = data;
    for (let i = 0; i < b.length; i++) {
      const row = new Float64Array(M.length);
      for (let j = 0; j < M.length; j++) row[j] = omegaRel(M[j], b[i], mu, L, g);
      self.postMessage({ type, job, i, row }, [row.buffer]);
    }
    self.postMessage({ type, job, done: true });
  } else if (type === 'slice') {
    // значения Ω вдоль среза: пары (M[i], b[i])
    const { mu, L, g, M, b } = data;
    const om = new Float64Array(M.length);
    for (let i = 0; i < M.length; i++) om[i] = omegaRel(M[i], b[i], mu, L, g);
    self.postMessage({ type, job, om }, [om.buffer]);
  } else if (type === 'channel') {
    const { mu, L, g, S, C, lnM } = data;
    const asy = channelAsymptotics(S, C, mu, L, g);
    // [Ω(M, S_n) − Ω(0)]/M²: асимптотика A − c ln(1/M) — прямая в логарифмической шкале по M
    const num = lnM.map((lm) => omegaRel(Math.exp(lm), S, mu, L, g) / Math.exp(2 * lm));
    const as = lnM.map((lm) => asy.A + asy.c * lm);
    self.postMessage({ type, job, num, as, asy });
  }
}
