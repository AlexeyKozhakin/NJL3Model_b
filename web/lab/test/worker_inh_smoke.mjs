// node web/lab/test/worker_inh_smoke.mjs — the worker file itself (worker_inh.js) with a self/fetch shim.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

globalThis.fetch = async (u) => ({ ok: true, json: async () => JSON.parse(readFileSync(fileURLToPath(u))) });
const out = [];
globalThis.self = { postMessage: (m) => out.push(m) };
await import('../js/worker_inh.js');
const send = async (msg) => { out.length = 0; await self.onmessage({ data: msg }); return out.slice(); };
for (const [phi, g] of [[0, -1], [0.5, -1], [0.3, 1]]) {
  const t = Date.now();
  const [r] = await send({ type: 'point', job: 1, mu: 3.6, L: 5, g, phi, T: 1e-3 });
  if (r.type === 'error') { console.error(r.message); process.exit(1); }
  console.log(`phi=${phi} g=${g}: ${r.result.phase} M*=${r.result.M_star.toPrecision(4)} b*=${r.result.b_star} (${Date.now() - t} ms)`);
}
const sl = await send({ type: 'slice', job: 2, mu: 3.6, L: 5, g: -1, phi: 0.5, M: [0.1, 0.2], b: [3, 3] });
const ch = await send({ type: 'channel', job: 3, mu: 3.6, L: 5, g: -1, phi: 0.5, S: 3.0, C: 2, lnM: [-5, -3] });
if (sl[0].type !== 'slice' || ch[0].type !== 'channel') { console.error('FAIL', sl, ch); process.exit(1); }
console.log('OK');
