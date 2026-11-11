// node --test test/hybrid.test.js
const test = require('node:test');
const assert = require('node:assert/strict');
const H = require('../core/hybrid.js');
const model = H.loadModel(require('../data/weights.js'));
const imgs = require('../data/images.js').map(H.loadImage);
const cal = imgs.filter(i => i.split === 'calib' && i.w === 64);
const ver = imgs.filter(i => i.split === 'verify' && i.w === 64);
const calib = H.calibrate(model, cal);
const plan = require('../data/plan.json');
const HOT = { years: 10, tempC: 105 };

test('weights load with Zero-DCE shapes (79,416 parameters)', () => {
  const shapes = model.map(L => [L.cout, L.cin]);
  assert.deepEqual(shapes, [[32, 3], [32, 32], [32, 32], [32, 32], [32, 64], [32, 64], [24, 64]]);
  const n = model.reduce((s, L) => s + L.w.length + L.b.length, 0);
  assert.equal(n, 79416);
});

test('conv3x3 matches a naive reference on a random tensor', () => {
  const rnd = H.mulberry(1), cin = 3, cout = 2, Hh = 5, Ww = 4;
  const X = Float32Array.from({ length: cin * Hh * Ww }, () => rnd() - 0.5);
  const Wt = Float32Array.from({ length: cout * cin * 9 }, () => rnd() - 0.5);
  const Y = H.conv3x3(X, cin, Hh, Ww, Wt, cout);
  for (let co = 0; co < cout; co++) for (let y = 0; y < Hh; y++) for (let x = 0; x < Ww; x++) {
    let s = 0;
    for (let ci = 0; ci < cin; ci++) for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) {
      const yy = y + ky - 1, xx = x + kx - 1;
      if (yy < 0 || xx < 0 || yy >= Hh || xx >= Ww) continue;
      s += Wt[((co * cin + ci) * 3 + ky) * 3 + kx] * X[(ci * Hh + yy) * Ww + xx];
    }
    assert.ok(Math.abs(Y[(co * Hh + y) * Ww + x] - s) < 1e-5);
  }
});

test('FP32 forward brightens a low-light image (the network works)', () => {
  const out = H.forward(model, ver[0], null, 0, null, calib);
  const mIn = ver[0].data.reduce((a, b) => a + b, 0) / ver[0].data.length;
  const mOut = out.reduce((a, b) => a + b, 0) / out.length;
  assert.ok(mOut > mIn * 1.4, `mean ${mIn.toFixed(3)} -> ${mOut.toFixed(3)}`);
});

test('digital INT8 is deterministic and close to FP32; analog error grows with drift and temperature', () => {
  const ref = H.forward(model, ver[0], null, 0, null, calib);
  const d1 = H.forward(model, ver[0], model.map(() => 'D'), 1, HOT, calib, H.DEFAULTS, 1);
  const d2 = H.forward(model, ver[0], model.map(() => 'D'), 2, HOT, calib, H.DEFAULTS, 1);
  assert.deepEqual(Array.from(d1.slice(0, 50)), Array.from(d2.slice(0, 50)));
  assert.ok(H.psnr(H.mse(d1, ref)) > 50);
  const fresh = H.analogSigma(H.DEFAULTS, { years: 0, tempC: 25 }), hot = H.analogSigma(H.DEFAULTS, HOT);
  assert.ok(Math.abs(fresh - H.DEFAULTS.sigProg) < 1e-9 && hot > 1.8 * fresh);
  const aF = H.psnr(H.mse(H.forward(model, ver[0], model.map(() => 'A'), 3, { years: 0, tempC: 25 }, calib, H.DEFAULTS, 1), ref));
  const aH = H.psnr(H.mse(H.forward(model, ver[0], model.map(() => 'A'), 3, HOT, calib, H.DEFAULTS, 1), ref));
  assert.ok(aH < aF, `${aH} < ${aF}`);
});

test('two cells per weight reduce analog error', () => {
  const ref = H.forward(model, ver[1], null, 0, null, calib);
  let a = 0, a2 = 0;
  for (let c = 0; c < 4; c++) {
    a += H.mse(H.forward(model, ver[1], model.map(() => 'A'), 10 + c, HOT, calib, H.DEFAULTS, 1), ref);
    a2 += H.mse(H.forward(model, ver[1], model.map(() => 'A2'), 10 + c, HOT, calib, H.DEFAULTS, 1), ref);
  }
  assert.ok(a2 < a);
});

test('energy model: all-digital > hybrid > all-analog, and die crossings are charged', () => {
  const eD = H.energyPerPixel(model, model.map(() => 'D')).joules;
  const eA = H.energyPerPixel(model, model.map(() => 'A'));
  const mixed = H.energyPerPixel(model, ['A', 'D', 'A', 'D', 'A', 'D', 'A']);
  assert.ok(eD > mixed.joules && mixed.joules > eA.joules);
  assert.ok(eA.d2d > 0, 'input crosses to the analog die and r crosses back');
  assert.ok(mixed.d2d > eA.d2d);
});

test('search returns the cheapest plan meeting the surrogate target (checked by brute force)', () => {
  const t = 40, margin = 0.7;
  const best = H.search(model, plan.sens, plan.kappa, t, margin);
  let bf = null;
  for (const p of H.allPlans(7)) {
    if (H.psnr(H.surrogate(plan.sens, p, plan.kappa)) < t + margin) continue;
    const e = H.energyPerPixel(model, p).joules;
    if (!bf || e < bf) bf = e;
  }
  assert.ok(Math.abs(best.e - bf) < 1e-21);
  assert.equal(H.allPlans(7).length, 2187);
});

test('zero-failure sample size for 95%/95% is 59', () => {
  assert.equal(H.zeroFailureN(0.95, 0.95), 59);
});

test('binomial CDF matches direct summation, and the Clopper-Pearson bound inverts it', () => {
  const choose = (n, k) => { let c = 1; for (let i = 0; i < k; i++) c = c * (n - i) / (i + 1); return c; };
  for (const [k, n, p] of [[0, 59, 0.05], [1, 100, 0.05], [3, 40, 0.2], [7, 30, 0.31]]) {
    let s = 0; for (let i = 0; i <= k; i++) s += choose(n, i) * Math.pow(p, i) * Math.pow(1 - p, n - i);
    assert.ok(Math.abs(H.binomCdf(k, n, p) - s) < 1e-10);
  }
  const u = H.cpUpper(4, 150, 0.05);
  assert.ok(Math.abs(H.binomCdf(4, 150, u) - 0.05) < 1e-6);
  assert.ok(H.binomCdf(1, 100, 0.05) <= 0.05 && H.binomCdf(2, 100, 0.05) > 0.05); // 100 chips certify with <= 1 failure
});

test('frontier is Pareto: energy strictly decreases and predicted PSNR strictly decreases down the ladder', () => {
  const lad = H.frontier(model, plan.sens, plan.kappa);
  assert.equal(lad.length, plan.ladder.length);
  for (let i = 1; i < lad.length; i++) { assert.ok(lad[i].e < lad[i - 1].e); assert.ok(lad[i].pred < lad[i - 1].pred); }
  // no plan off the frontier dominates a frontier plan
  for (const p of H.allPlans(7)) {
    const e = H.energyPerPixel(model, p).joules, q = H.psnr(H.surrogate(plan.sens, p, plan.kappa));
    for (const f of lad) assert.ok(!(e < f.e - 1e-21 && q > f.pred + 1e-9));
  }
});

test('fixed-sequence test stops at the first non-certified plan and picks the last certified one', () => {
  const lad = [0, 1, 2, 3, 4].map(() => ({}));
  const chips = [Array(100).fill(50), Array(100).fill(45).map((x, i) => (i < 1 ? 30 : x)), Array(100).fill(41).map((x, i) => (i < 4 ? 30 : x)), Array(100).fill(60), Array(100).fill(60)];
  const r = H.fixedSequence(lad, (i) => chips[i], 40, 0.05, 0.05);
  assert.equal(r.chosen, 1);
  assert.deepEqual(r.rows.map((x) => x.tested), [true, true, true, false, false]);
  assert.equal(r.rows[2].k, 4);
});

test('fixed-sequence family-wise error stays below delta (simulation with bad plans after good ones)', () => {
  const rnd = H.mulberry(99), trueFail = [0.0, 0.01, 0.03, 0.05, 0.07, 0.12, 0.2], n = 100, trials = 4000;
  let wrong = 0;
  for (let t = 0; t < trials; t++) {
    const chips = trueFail.map((q) => Array.from({ length: n }, () => (rnd() < q ? 0 : 1)));
    const r = H.fixedSequence(trueFail.map(() => ({})), (i) => chips[i], 0.5, 0.05, 0.05);
    if (r.rows.some((x, i) => x.tested && x.ok && trueFail[i] > 0.05)) wrong++;
  }
  assert.ok(wrong / trials <= 0.05, `false certification rate ${wrong / trials}`);
});

test('committed certification chips reproduce exactly from the core (selected 40 dB plan, first 3 chips)', () => {
  const lad = plan.ladder, r = H.fixedSequence(lad, (i) => lad[i].chips, 40, plan.eps, plan.delta);
  assert.ok(r.chosen >= 0);
  const refs = ver.map((i) => H.forward(model, i, null, 0, null, calib));
  const got = H.evalPlan(model, ver, refs, lad[r.chosen].plan, 3, HOT, calib, H.DEFAULTS, 50000);
  got.forEach((x, k) => assert.ok(Math.abs(x.worstPsnr - lad[r.chosen].chips[k]) < 1e-3));
});
