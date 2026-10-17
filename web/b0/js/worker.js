// Фоновый расчёт карты M*(L, mu): строки по L, гарантированный минимизатор из njl3b0.js.
import { findMinimum, phaseLabel } from './njl3b0.js';

self.onmessage = ({ data }) => {
  const { job, rows, mu, g } = data;
  for (const { i, L } of rows) {
    const M = new Float64Array(mu.length);
    const phase = new Array(mu.length);
    const omega = new Float64Array(mu.length);
    for (let j = 0; j < mu.length; j++) {
      const r = findMinimum(mu[j], L, g);
      M[j] = r.M; omega[j] = r.omega; phase[j] = phaseLabel(r.M, mu[j], L);
    }
    self.postMessage({ job, i, M, omega, phase }, [M.buffer, omega.buffer]);
  }
  self.postMessage({ job, done: true });
};
