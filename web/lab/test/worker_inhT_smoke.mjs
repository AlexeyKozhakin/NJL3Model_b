// node web/lab/test/worker_inhT_smoke.mjs — the worker file itself (worker_inhT.js) with a self/fetch shim.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

globalThis.fetch = async (u) => ({ ok: true, json: async () => JSON.parse(readFileSync(fileURLToPath(u))) });
const out = [];
globalThis.self = { postMessage: (m) => out.push(m) };
await import('../js/worker_inhT.js');
const send = async (msg) => { out.length = 0; await self.onmessage({ data: msg }); return out.slice(); };
for (const [phi, g, T] of [[0, -1, 1e-3], [0, -1, 1e-2], [0.5, -1, 1e-2], [0.3, 1, 1e-3]]) {
  const t = Date.now();
  const [r] = await send({ type: 'point', job: 1, mu: 3.6, L: 5, g, phi, T });
  if (r.type === 'error') { console.error(r.message); process.exit(1); }
  console.log(`phi=${phi} g=${g} T=${T}: ${r.result.phase} M*=${r.result.M_star.toPrecision(4)} b*=${r.result.b_star} (${Date.now() - t} ms)`);
}
const t = Date.now();
const om = await send({ type: 'omega-map', job: 2, mu: 3.6, L: 5, g: -1, phi: 0, T: 1e-2,
                        M: Array.from({ length: 60 }, (_, i) => 0.02 * (i + 1)), b: Array.from({ length: 60 }, (_, i) => 0.07 * i) });
const sl = await send({ type: 'slice', job: 3, mu: 3.6, L: 5, g: -1, phi: 0.5, T: 1e-3, M: [0.1, 0.2], b: [3, 3] });
if (om.at(-1).done !== true || om.length !== 61 || sl[0].type !== 'slice') { console.error('FAIL', om.at(-1), sl); process.exit(1); }
console.log(`omega-map 60×60: ${Date.now() - t} ms`);
console.log('OK');
