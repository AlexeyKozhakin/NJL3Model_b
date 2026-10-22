// node web/fast/test/worker_smoke.mjs — the worker file itself (worker.js) with a self/fetch shim.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

globalThis.fetch = async (u) => ({ ok: true, json: async () => JSON.parse(readFileSync(fileURLToPath(u))) });
const out = [];
globalThis.self = { postMessage: (m) => out.push(m) };
await import('../js/worker.js');
const send = async (msg) => { out.length = 0; await self.onmessage({ data: msg }); return out.slice(); };
const base = { mu: 3.1409, L: 10, g: -1, phi: 0, T: 1e-3 };
let r = await send({ ...base, type: 'point', job: 1 });
if (r[0].type !== 'point') throw new Error(JSON.stringify(r[0]));
console.log('point', r[0].result.phase, r[0].result.M_star.toPrecision(5));
r = await send({ ...base, type: 'minima', job: 2 });
console.log('minima', r[0].minima.length);
r = await send({ ...base, type: 'landscape', job: 3, M: [0.5, 1, 2], b: [0, 0.5] });
console.log('landscape rows', r.filter((m) => m.row).length, 'done', !!r.at(-1).done);
r = await send({ ...base, type: 'map', job: 4, rows: [{ i: 0, L: 5 }], mu: [3.1, 3.2] });
console.log('map row', r[0].phase, 'done', !!r.at(-1).done);
console.log('OK');
