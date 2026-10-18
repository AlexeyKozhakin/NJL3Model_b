// Timing of the JS core (web/b0/js/njl3b0.js) in Node.js, for the performance table.
// Usage: node web/b0/test/bench.js [N]   -> prints JSON: microseconds per call / per point.
import { gapG, findMinimum } from '../js/njl3b0.js';

const N = Number(process.argv[2] || 2000);
let seed = 12345;
const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
const pts = Array.from({ length: N }, () => [0.05 + 6.95 * rnd(), 0.1 + 9.9 * rnd()]);
const g = -1;

let t = performance.now(), s = 0;
for (let k = 0; k < 20; k++) for (const [mu, L] of pts) s += gapG(0.37 * mu, mu, L, g);
const gapUs = (performance.now() - t) * 1e3 / (20 * N);

for (const [mu, L] of pts.slice(0, 200)) findMinimum(mu, L, g);      // warm-up (JIT)
t = performance.now();
for (const [mu, L] of pts) s += findMinimum(mu, L, g).M;
const minUs = (performance.now() - t) * 1e3 / N;

console.log(JSON.stringify({ node: process.version, N, gapG_us: gapUs, findMinimum_us: minUs, checksum: s }));
