// Фоновый расчёт карты M*(L, mu) при произвольных φ, g.
import { findMinimum, phaseLabel } from './njl3hom.js';

self.onmessage = ({ data }) => {
  const { job, rows, mu, g, phi } = data;
  for (const { i, L } of rows) {
    const M = new Float64Array(mu.length);
    const phase = new Array(mu.length);
    const omega = new Float64Array(mu.length);
    for (let j = 0; j < mu.length; j++) {
      const r = findMinimum(mu[j], L, g, phi);
      M[j] = r.M; omega[j] = r.omega; phase[j] = phaseLabel(r.M, mu[j], L, phi);
    }
    self.postMessage({ job, i, M, omega, phase }, [M.buffer, omega.buffer]);
  }
  self.postMessage({ job, done: true });
};
