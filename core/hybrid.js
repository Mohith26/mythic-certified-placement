/*
 * Hybrid analog/digital placement with a chip-population accuracy guarantee.
 * Unofficial concept by Mohith Gajjela. Not affiliated with Mythic.
 *
 * Model: Zero-DCE (the low-light network Mythic runs on-chip for Project Starlight), real public weights.
 * Each conv layer is placed on the analog die (flash compute-in-memory, 1 or 2 cells per weight) or on the
 * digital die (INT8). Analog error model: per-chip weight programming error, retention-drift and
 * temperature residuals after per-column gain calibration, per-inference read noise, 8-bit ADCs.
 * Energy model: per-MAC energy by mode plus die-to-die transfer of every activation tensor that crosses dies.
 *
 * Planner: measure each layer's error contribution by Monte Carlo over simulated chips, rank all 3^7
 * placements with an additive surrogate and keep the energy/quality Pareto frontier, then certify along that
 * frontier with a fixed-sequence binomial test on fresh simulated chips at the worst corner (Learn-then-Test):
 * the cheapest certified plan has at most 5% of chips below the target, with 95% confidence.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.HYB = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ------------------------------------------------------------------ network topology
  // Zero-DCE: x1=conv1(x), x2=conv2(x1), x3=conv3(x2), x4=conv4(x3), x5=conv5[x3,x4], x6=conv6[x2,x5], r=tanh(conv7[x1,x6])
  const TOPO = [
    { name: 'conv1', inputs: ['x'], out: 'x1' },
    { name: 'conv2', inputs: ['x1'], out: 'x2' },
    { name: 'conv3', inputs: ['x2'], out: 'x3' },
    { name: 'conv4', inputs: ['x3'], out: 'x4' },
    { name: 'conv5', inputs: ['x3', 'x4'], out: 'x5' },
    { name: 'conv6', inputs: ['x2', 'x5'], out: 'x6' },
    { name: 'conv7', inputs: ['x1', 'x6'], out: 'r' },
  ];
  const MODES = ['D', 'A', 'A2']; // digital INT8, analog 1 cell, analog 2 cells averaged

  const DEFAULTS = {
    // analog error model (fractions of each output column's max |w|), at reference 25 C, fresh
    sigProg: 0.010,      // programming (write-verify) error per cell
    sigDrift10y: 0.014,  // retention-drift residual after gain calibration, 10 years at 105 C
    driftEa: 0.6,        // eV, Arrhenius for charge loss
    sigTempPerC: 0.00012,// temperature-coefficient mismatch residual per degC away from 25 C
    sigGain: 0.002,      // per-column gain calibration residual
    sigRead: 0.004,      // read noise per conversion, fraction of column full scale
    adcBits: 8,
    actBits: 8,
    // energy model
    eMacA: 16.7e-15,     // J per MAC, analog (120 TOPS/W, 2 ops per MAC)
    eMacA2: 1.7 * 16.7e-15,
    eMacD: 0.15e-12,     // J per MAC, digital INT8 incl. local SRAM (assumption)
    eBitD2D: 0.25e-12,   // J per bit across the die-to-die link (assumption)
    inputDie: 'D',       // image arrives over MIPI on the digital die; curve application runs there too
  };

  // ------------------------------------------------------------------ RNG
  function mulberry(seed) {
    let s = seed >>> 0;
    return function () {
      s = (s + 0x6d2b79f5) >>> 0; let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function seedOf(a, b, c, d) {
    let h = 2166136261 >>> 0;
    for (const v of [a, b, c, d]) { h ^= (v | 0) + 0x9e3779b9; h = Math.imul(h, 16777619) >>> 0; h ^= h >>> 13; }
    return h >>> 0;
  }
  function gaussFill(arr, rnd) {
    for (let i = 0; i < arr.length; i += 2) {
      const u1 = rnd() + 1e-12, u2 = rnd(), r = Math.sqrt(-2 * Math.log(u1));
      arr[i] = r * Math.cos(2 * Math.PI * u2);
      if (i + 1 < arr.length) arr[i + 1] = r * Math.sin(2 * Math.PI * u2);
    }
    return arr;
  }

  // ------------------------------------------------------------------ data decoding
  function b64ToBytes(b64) {
    if (typeof Buffer !== 'undefined') return new Uint8Array(Buffer.from(b64, 'base64'));
    const s = atob(b64), u = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
    return u;
  }
  function loadModel(raw) {
    return raw.map((L, i) => {
      const wb = b64ToBytes(L.w), bb = b64ToBytes(L.b);
      const w = new Float32Array(wb.buffer, wb.byteOffset, wb.byteLength / 4).slice();
      const b = new Float32Array(bb.buffer, bb.byteOffset, bb.byteLength / 4).slice();
      const [cout, cin] = L.shape;
      const wmax = new Float32Array(cout);
      for (let co = 0; co < cout; co++) { let m = 0; for (let k = 0; k < cin * 9; k++) m = Math.max(m, Math.abs(w[co * cin * 9 + k])); wmax[co] = m; }
      return Object.assign({ w, b, cin, cout, wmax, macsPerPx: cin * cout * 9 }, TOPO[i]);
    });
  }
  function loadImage(img) { // -> Float32Array CHW in [0,1]
    const bytes = b64ToBytes(img.rgb), n = img.w * img.h, out = new Float32Array(3 * n);
    for (let p = 0; p < n; p++) for (let c = 0; c < 3; c++) out[c * n + p] = bytes[p * 3 + c] / 255;
    return { data: out, w: img.w, h: img.h, name: img.name, split: img.split };
  }

  // ------------------------------------------------------------------ conv
  // X: Float32Array [cin][H][W]; Wt: [cout][cin][3][3]; returns [cout][H][W] (no bias)
  function conv3x3(X, cin, H, W, Wt, cout) {
    const Hp = H + 2, Wp = W + 2;
    const P = new Float32Array(cin * Hp * Wp);
    for (let c = 0; c < cin; c++) for (let y = 0; y < H; y++) {
      const src = c * H * W + y * W, dst = c * Hp * Wp + (y + 1) * Wp + 1;
      for (let x = 0; x < W; x++) P[dst + x] = X[src + x];
    }
    const Y = new Float32Array(cout * H * W);
    for (let co = 0; co < cout; co++) {
      const yo = co * H * W;
      for (let ci = 0; ci < cin; ci++) {
        const pc = ci * Hp * Wp, wo = (co * cin + ci) * 9;
        for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) {
          const wv = Wt[wo + ky * 3 + kx];
          if (wv === 0) continue;
          for (let y = 0; y < H; y++) {
            const pr = pc + (y + ky) * Wp + kx, yr = yo + y * W;
            for (let x = 0; x < W; x++) Y[yr + x] += wv * P[pr + x];
          }
        }
      }
    }
    return Y;
  }

  // ------------------------------------------------------------------ operating condition
  // cond: { years, tempC }. Drift residual grows with log(1 + t_eff / 1 day), t_eff Arrhenius-scaled to 105 C.
  function analogSigma(cfg, cond) {
    const kB = 8.617e-5;
    const af = Math.exp(cfg.driftEa / kB * (1 / (105 + 273.15) - 1 / (cond.tempC + 273.15)));
    const days = cond.years * 365 * af;
    const ref = Math.log(1 + 10 * 365);
    const sDrift = cfg.sigDrift10y * Math.log(1 + days) / ref;
    const sTemp = cfg.sigTempPerC * Math.abs(cond.tempC - 25);
    return Math.sqrt(cfg.sigProg * cfg.sigProg + sDrift * sDrift + sTemp * sTemp);
  }

  // ------------------------------------------------------------------ forward pass
  // plan: array of 7 modes or null (FP32 reference). chip: integer id (analog error realisation).
  // calib: {actMax[], outMax[][]} from calibrate(). imageKey: integer for read-noise seeding.
  function forward(model, img, plan, chip, cond, calib, cfg, imageKey) {
    cfg = cfg || DEFAULTS;
    const H = img.h, W = img.w, n = H * W;
    const T = { x: img.data };
    const sig = plan ? analogSigma(cfg, cond) : 0;
    const qa = (1 << cfg.actBits) - 1, qadc = (1 << (cfg.adcBits - 1)) - 1;
    for (let l = 0; l < model.length; l++) {
      const L = model[l];
      let X;
      if (L.inputs.length === 1) X = T[L.inputs[0]];
      else { X = new Float32Array(L.cin * n); let off = 0; for (const k of L.inputs) { X.set(T[k], off); off += T[k].length; } }
      const mode = plan ? plan[l] : null;
      let Wt = L.w;
      if (mode) {
        // 8-bit activation inputs (DAC on analog, INT8 on digital)
        const am = calib.actMax[l], Xq = new Float32Array(X.length);
        for (let i = 0; i < X.length; i++) { const v = Math.min(Math.max(X[i], 0), am); Xq[i] = Math.round(v / am * qa) / qa * am; }
        X = Xq;
        Wt = new Float32Array(L.w.length);
        const K = L.cin * 9;
        if (mode === 'D') {
          for (let co = 0; co < L.cout; co++) { const s = L.wmax[co] / 127; for (let k = 0; k < K; k++) { const i = co * K + k; Wt[i] = Math.round(L.w[i] / s) * s; } }
        } else {
          const cells = mode === 'A2' ? 2 : 1;
          const rnd = mulberry(seedOf(chip, l, 7, cells));
          const z = gaussFill(new Float32Array(L.w.length), rnd);
          const zg = gaussFill(new Float32Array(L.cout), rnd);
          const s = sig / Math.sqrt(cells);
          for (let co = 0; co < L.cout; co++) {
            const g = 1 + cfg.sigGain * zg[co];
            for (let k = 0; k < K; k++) { const i = co * K + k; Wt[i] = (L.w[i] + L.wmax[co] * s * z[i]) * g; }
          }
        }
      }
      const Y = conv3x3(X, L.cin, H, W, Wt, L.cout);
      if (mode === 'A' || mode === 'A2') {
        const rnd = mulberry(seedOf(chip, l, imageKey | 0, 99));
        const z = gaussFill(new Float32Array(Y.length), rnd);
        for (let co = 0; co < L.cout; co++) {
          const R = calib.outMax[l][co], sR = cfg.sigRead * R;
          for (let p = 0; p < n; p++) {
            const i = co * n + p;
            let v = Y[i] + sR * z[i];
            v = Math.max(-qadc, Math.min(qadc, Math.round(v / R * qadc)));
            Y[i] = v / qadc * R;
          }
        }
      }
      const last = l === model.length - 1;
      for (let co = 0; co < L.cout; co++) {
        const bv = L.b[co];
        for (let p = 0; p < n; p++) {
          const i = co * n + p, v = Y[i] + bv;
          Y[i] = last ? Math.tanh(v) : (v > 0 ? v : 0);
        }
      }
      T[L.out] = Y;
    }
    // light-enhancement curves: 8 iterations x = x + r_k (x^2 - x), on the digital die
    const out = img.data.slice(), r = T.r;
    for (let k = 0; k < 8; k++) for (let c = 0; c < 3; c++) {
      const ro = (k * 3 + c) * n, o = c * n;
      for (let p = 0; p < n; p++) { const x = out[o + p]; out[o + p] = x + r[ro + p] * (x * x - x); }
    }
    return out;
  }

  // Calibrate activation and ADC ranges on calibration images (FP32 pass, 99.99th percentile, as a deployment tool would)
  function calibrate(model, images) {
    const actMax = new Array(model.length).fill(0), outMax = model.map(L => new Float32Array(L.cout));
    const actVals = model.map(() => []), outVals = model.map(L => Array.from({ length: L.cout }, () => []));
    for (const img of images) {
      const n = img.w * img.h, T = { x: img.data };
      for (let l = 0; l < model.length; l++) {
        const L = model[l];
        let X;
        if (L.inputs.length === 1) X = T[L.inputs[0]];
        else { X = new Float32Array(L.cin * n); let off = 0; for (const k of L.inputs) { X.set(T[k], off); off += T[k].length; } }
        for (let i = 0; i < X.length; i += 7) actVals[l].push(X[i]);
        const Y = conv3x3(X, L.cin, img.h, img.w, L.w, L.cout);
        for (let co = 0; co < L.cout; co++) for (let p = 0; p < n; p += 3) outVals[l][co].push(Math.abs(Y[co * n + p]));
        const last = l === model.length - 1;
        for (let co = 0; co < L.cout; co++) for (let p = 0; p < n; p++) { const i = co * n + p, v = Y[i] + L.b[co]; Y[i] = last ? Math.tanh(v) : Math.max(v, 0); }
        T[L.out] = Y;
      }
    }
    const pct = (a, q) => { const s = Float64Array.from(a).sort(); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; };
    for (let l = 0; l < model.length; l++) {
      actMax[l] = Math.max(pct(actVals[l], 0.9999), 1e-6);
      for (let co = 0; co < model[l].cout; co++) outMax[l][co] = Math.max(pct(outVals[l][co], 0.9999) * 1.1, 1e-6);
    }
    return { actMax, outMax };
  }

  function mse(a, b) { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return s / a.length; }
  function psnr(m) { return 10 * Math.log10(1 / Math.max(m, 1e-12)); }

  // ------------------------------------------------------------------ energy
  function energyPerPixel(model, plan, cfg) {
    cfg = cfg || DEFAULTS;
    let e = 0;
    const where = { x: cfg.inputDie };
    const sent = new Set();
    const breakdown = { mac: 0, d2d: 0 };
    for (let l = 0; l < model.length; l++) {
      const L = model[l], m = plan[l], die = m === 'D' ? 'D' : 'A';
      const eMac = m === 'D' ? cfg.eMacD : m === 'A2' ? cfg.eMacA2 : cfg.eMacA;
      breakdown.mac += L.macsPerPx * eMac;
      for (const k of L.inputs) {
        if (where[k] !== die && !sent.has(k + '>' + die)) {
          sent.add(k + '>' + die);
          const ch = k === 'x' ? 3 : model.find(M => M.out === k).cout;
          breakdown.d2d += ch * cfg.actBits * cfg.eBitD2D;
        }
      }
      where[L.out] = die;
    }
    if (where.r !== 'D') breakdown.d2d += 24 * cfg.actBits * cfg.eBitD2D; // curve maps go to the digital die
    e = breakdown.mac + breakdown.d2d;
    return { joules: e, mac: breakdown.mac, d2d: breakdown.d2d };
  }
  // power at a video format (W)
  function powerAt(model, plan, px, fps, cfg) { return energyPerPixel(model, plan, cfg).joules * px * fps; }

  // ------------------------------------------------------------------ Monte Carlo evaluation
  // returns per-chip worst-image PSNR (vs FP32 reference) and mean MSE
  function evalPlan(model, images, refs, plan, chips, cond, calib, cfg, chipBase) {
    const res = [];
    for (let c = 0; c < chips; c++) {
      const chip = (chipBase || 0) + c;
      let worst = Infinity, msum = 0, mmax = 0;
      const per = [];
      images.forEach((img, j) => {
        const o = forward(model, img, plan, chip, cond, calib, cfg, j + 1);
        const m = mse(o, refs[j]); msum += m; mmax = Math.max(mmax, m); per.push(m); worst = Math.min(worst, psnr(m));
      });
      res.push({ chip, worstPsnr: worst, meanMse: msum / images.length, worstMse: mmax, perImage: per });
    }
    return res;
  }

  // Sensitivity: excess MSE of putting one layer in an analog mode, everything else digital.
  // Per image, so the surrogate can track the worst image (the pass criterion is the worst image on each chip).
  function sensitivity(model, images, refs, chips, cond, calib, cfg) {
    const allD = model.map(() => 'D');
    const base = evalPlan(model, images, refs, allD, 1, cond, calib, cfg, 0)[0].perImage;
    const table = [];
    for (let l = 0; l < model.length; l++) {
      const row = {};
      for (const m of ['A', 'A2']) {
        const plan = allD.slice(); plan[l] = m;
        const r = evalPlan(model, images, refs, plan, chips, cond, calib, cfg, 1000 + l * 100);
        const perImage = base.map((b, j) => r.reduce((a, x) => a + Math.max(x.perImage[j] - b, 0), 0) / r.length);
        row[m] = { perImage, mean: perImage.reduce((a, b) => a + b, 0) / perImage.length };
      }
      table.push(row);
    }
    return { base, table };
  }

  // ------------------------------------------------------------------ planner
  function allPlans(nl) {
    const out = [], total = Math.pow(3, nl);
    for (let k = 0; k < total; k++) { const p = []; let v = k; for (let l = 0; l < nl; l++) { p.push(MODES[v % 3]); v = Math.floor(v / 3); } out.push(p); }
    return out;
  }
  // surrogate: predicted chip-population upper-percentile worst-image MSE
  //   = kappa * max_j (base_j + sum_l excess_{l,mode,j})   (errors from different layers add, to first order)
  function surrogate(sens, plan, kappa) {
    let worst = 0;
    for (let j = 0; j < sens.base.length; j++) {
      let m = sens.base[j];
      plan.forEach((mode, l) => { if (mode !== 'D') m += sens.table[l][mode].perImage[j]; });
      worst = Math.max(worst, m);
    }
    return kappa * worst;
  }
  // cheapest plan whose surrogate PSNR >= target + margin (plans in `exclude` already failed verification)
  function search(model, sens, kappa, targetDb, marginDb, cfg, exclude) {
    let best = null;
    for (const p of allPlans(model.length)) {
      if (exclude && exclude.has(p.join())) continue;
      const pp = psnr(surrogate(sens, p, kappa));
      if (pp < targetDb + marginDb) continue;
      const e = energyPerPixel(model, p, cfg).joules;
      if (!best || e < best.e - 1e-21 || (Math.abs(e - best.e) < 1e-21 && pp > best.pred)) best = { plan: p, e, pred: pp };
    }
    return best;
  }
  // zero-failure demonstration: n chips all pass => pass rate >= p with confidence conf (n = 59 for 95%/95%)
  function zeroFailureN(p, conf) { return Math.ceil(Math.log(1 - conf) / Math.log(p)); }

  // ------------------------------------------------------------------ certification (Learn-then-Test, fixed sequence)
  // Binomial CDF P(K <= k) for K ~ Bin(n, p), computed in log space so it is stable for any n.
  function lgamma(x) { // Lanczos, |err| < 1e-13 for x > 0
    const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
      12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
    x -= 1; let a = c[0]; const t = x + g + 0.5;
    for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }
  function binomCdf(k, n, p) {
    if (k < 0) return 0; if (k >= n) return 1;
    let s = 0;
    for (let i = 0; i <= k; i++) s += Math.exp(lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) + i * Math.log(p) + (n - i) * Math.log1p(-p));
    return Math.min(1, s);
  }
  // exact (Clopper-Pearson) one-sided upper confidence bound on a failure rate after k failures in n
  function cpUpper(k, n, alpha) {
    if (k >= n) return 1;
    let lo = 0, hi = 1;
    for (let it = 0; it < 80; it++) { const m = (lo + hi) / 2; if (binomCdf(k, n, m) > alpha) lo = m; else hi = m; }
    return hi;
  }
  // Pareto frontier of the 3^L placements: energy vs surrogate-predicted PSNR. Returned from the most
  // conservative (highest predicted PSNR) to the cheapest; this order is fixed before any verification chip is seen.
  function frontier(model, sens, kappa, cfg) {
    const pts = allPlans(model.length).map((p) => ({ plan: p, e: energyPerPixel(model, p, cfg).joules, pred: psnr(surrogate(sens, p, kappa)) }));
    pts.sort((a, b) => a.e - b.e || b.pred - a.pred);
    const out = []; let best = -Infinity;
    for (const x of pts) if (x.pred > best + 1e-9) { out.push(x); best = x.pred; }
    return out.reverse();
  }
  // Fixed-sequence test (Bauer 1991; Angelopoulos et al., "Learn then Test", 2021).
  //   H0(plan): more than eps of chips fall below the target. p-value = P(Bin(n, eps) <= failures).
  //   Walk the ladder from the safest plan; reject while p <= delta; stop at the first plan that cannot be rejected.
  //   Every rejected plan is certified simultaneously with family-wise error <= delta, so picking the cheapest
  //   certified plan keeps the guarantee no matter how many plans were tested on the same chips.
  // ladder: [{plan,e}] safest -> cheapest; chipsOf(i) -> array of per-chip worst-image PSNR for ladder[i]
  function fixedSequence(ladder, chipsOf, targetDb, eps, delta) {
    const rows = []; let chosen = -1, stopped = false;
    for (let i = 0; i < ladder.length; i++) {
      if (stopped) { rows.push({ i, tested: false }); continue; }
      const c = chipsOf(i), n = c.length, k = c.filter((x) => x < targetDb).length, p = binomCdf(k, n, eps);
      const ok = p <= delta;
      rows.push({ i, tested: true, n, k, p, ok, upper: cpUpper(k, n, delta) });
      if (ok) chosen = i; else stopped = true;
    }
    return { chosen, rows };
  }
  // Naive baseline: plan for the average chip (cheapest plan whose mean-surrogate PSNR meets the target, no margin).
  function naivePlan(model, sens, targetDb, cfg) { return search(model, sens, 1, targetDb, 0, cfg); }

  return { TOPO, MODES, DEFAULTS, loadModel, loadImage, conv3x3, forward, calibrate, mse, psnr, analogSigma,
    energyPerPixel, powerAt, evalPlan, sensitivity, allPlans, surrogate, search, zeroFailureN, binomCdf, cpUpper, frontier, fixedSequence, naivePlan, mulberry, seedOf };
});
