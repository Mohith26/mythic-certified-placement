// End-to-end planner run. Writes data/plan.json (and data/plan.js for file:// viewing).
//   node scripts/plan.js            (about 10 minutes on 6 cores)
// 1. per-layer sensitivity (calibration images, 12 simulated chips at 10 years / 105 °C)
// 2. kappa: 95th-percentile / mean ratio of worst-image error across 40 all-analog chips (calibration images)
// 3. surrogate sanity check on 10 random plans (held-out images, fresh chips)
// 4. Pareto frontier (energy vs predicted PSNR), evaluated on 100 fresh certification chips at the worst corner;
//    the page runs the fixed-sequence test on these results for any target
// 5. independent holdout: 150 more chips for the certified plan and the "average chip" plan at each target
// 6. corners: 40 chips at three operating conditions for the certified plans and the uniform placements
const fs = require('fs'), path = require('path'), { spawn } = require('child_process');
const { H, model, cal, ver, refsC, refsV, calib, HOT, CORNERS, CHIPS } = require('./common.js');
const cfg = H.DEFAULTS, t0 = Date.now(), WORKERS = +(process.env.WORKERS || 6);
const log = (...a) => console.log(((Date.now() - t0) / 1000).toFixed(0) + 's', ...a);
const N_CERT = 100, N_HOLD = 150, N_CORNER = 40, EPS = 0.05, DELTA = 0.05, TARGETS = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46];

// optional resume cache (CACHE=/path.json): finished job results are reused, so an interrupted run can continue
const CACHE = process.env.CACHE, cache = CACHE && fs.existsSync(CACHE) ? JSON.parse(fs.readFileSync(CACHE, 'utf8')) : {};
async function runJobs(all) {
  const ck = (j) => j.key + '@' + j.base + 'x' + j.n + '@' + j.corner.years + '/' + j.corner.tempC;
  const out = {}, jobs = [];
  for (const j of all) { if (cache[ck(j)]) out[j.key] = cache[ck(j)]; else jobs.push(j); }
  // small batches so the cache is written often
  for (let s = 0; s < jobs.length; s += WORKERS * 2) {
    const r = await runBatch(jobs.slice(s, s + WORKERS * 2));
    for (const j of jobs.slice(s, s + WORKERS * 2)) { out[j.key] = r[j.key]; cache[ck(j)] = r[j.key]; }
    if (CACHE) fs.writeFileSync(CACHE, JSON.stringify(cache));
  }
  return out;
}
function runBatch(jobs) {
  const parts = Array.from({ length: WORKERS }, () => []);
  jobs.forEach((j, k) => parts[k % WORKERS].push(j));
  const out = {};
  return Promise.all(parts.filter((p) => p.length).map((p) => new Promise((res, rej) => {
    const c = spawn(process.execPath, [path.join(__dirname, 'evalworker.js'), JSON.stringify(p)]);
    let buf = '';
    c.stdout.on('data', (d) => { buf += d; });
    c.stderr.on('data', (d) => process.stderr.write(d));
    c.on('close', (code) => { if (code) return rej(new Error('worker ' + code)); buf.trim().split('\n').forEach((l) => { const o = JSON.parse(l); out[o.key] = o.chips; }); res(); });
  }))).then(() => out);
}

(async () => {
  const sens = H.sensitivity(model, cal, refsC, 12, HOT, calib, cfg);
  log('sensitivity done');
  const allA = model.map(() => 'A');
  const ka = H.evalPlan(model, cal, refsC, allA, 40, HOT, calib, cfg, 9000).map((r) => r.worstMse).sort((a, b) => a - b);
  const kappa = ka[Math.floor(0.95 * (ka.length - 1))] / (ka.reduce((a, b) => a + b, 0) / ka.length);
  log('kappa', kappa.toFixed(3));
  const rnd = H.mulberry(4242), check = [];
  const checkPlans = Array.from({ length: 10 }, () => model.map(() => H.MODES[Math.floor(rnd() * 3)]));
  const ck = await runJobs(checkPlans.map((p, k) => ({ key: 'c' + k, plan: p, base: 70000 + k * 100, n: 10, corner: HOT })));
  checkPlans.forEach((p, k) => {
    const c = ck['c' + k].slice().sort((a, b) => a - b);
    check.push({ plan: p, predP95Db: H.psnr(H.surrogate(sens, p, kappa)), predMeanDb: H.psnr(H.surrogate(sens, p, 1)), mcMinDb: c[0], mcMedDb: c[c.length >> 1] });
  });
  log('surrogate check done');
  const ladder = H.frontier(model, sens, kappa, cfg);
  log('frontier', ladder.length, 'plans; certifying on', N_CERT, 'chips');
  const cert = await runJobs(ladder.map((x, i) => ({ key: 'f' + i, plan: x.plan, base: CHIPS.certify, n: N_CERT, corner: HOT })));
  ladder.forEach((x, i) => { x.chips = cert['f' + i]; x.powerMw = H.powerAt(model, x.plan, 1920 * 1080, 30, cfg) * 1e3; });
  log('certification done');
  // per-target selections (recomputed live in the page; stored here for the holdout and corner runs)
  const sel = TARGETS.map((t) => {
    const fs_ = H.fixedSequence(ladder, (i) => ladder[i].chips, t, EPS, DELTA);
    const naive = H.naivePlan(model, sens, t, cfg);
    return { target: t, chosen: fs_.chosen, plan: fs_.chosen >= 0 ? ladder[fs_.chosen].plan : null, naive: naive ? naive.plan : null };
  });
  const uniq = new Map();
  for (const s of sel) for (const p of [s.plan, s.naive]) if (p) uniq.set(p.join(), p);
  const hold = await runJobs([...uniq.values()].map((p) => ({ key: p.join(), plan: p, base: CHIPS.holdout, n: N_HOLD, corner: HOT })));
  log('holdout done', uniq.size, 'plans');
  const uniform = { allD: model.map(() => 'D'), allA, allA2: model.map(() => 'A2') };
  const cplans = new Map(Object.values(uniform).map((p) => [p.join(), p]));
  for (const s of sel) if (s.plan) cplans.set(s.plan.join(), s.plan);
  const cj = [];
  for (const [k, p] of cplans) CORNERS.forEach((c, ci) => cj.push({ key: k + '|' + ci, plan: p, base: CHIPS.corners, n: N_CORNER, corner: c }));
  const corn = await runJobs(cj);
  const corners = {};
  for (const [k] of cplans) corners[k] = CORNERS.map((c, ci) => ({ label: c.label, years: c.years, tempC: c.tempC, chips: corn[k + '|' + ci] }));
  log('corners done');
  const out = { generated: 'node scripts/plan.js', corner: HOT, eps: EPS, delta: DELTA, nCert: N_CERT, nHold: N_HOLD, nCorner: N_CORNER, targets: TARGETS,
    kappa, sens, check, ladder, holdout: hold, corners, uniform, verifyImages: ver.map((i) => i.name), calibImages: cal.map((i) => i.name) };
  const json = JSON.stringify(out);
  fs.writeFileSync(path.join(__dirname, '..', 'data', 'plan.json'), json);
  fs.writeFileSync(path.join(__dirname, '..', 'data', 'plan.js'), '// generated by scripts/plan.js (same content as plan.json, loadable from file://)\nwindow.PLAN = ' + json + ';\n');
  sel.forEach((s) => log('target', s.target, s.plan && s.plan.join(''), 'naive', s.naive && s.naive.join(''),
    s.plan ? 'hold fails ' + hold[s.plan.join()].filter((x) => x < s.target).length : '', s.naive ? 'naive hold fails ' + hold[s.naive.join()].filter((x) => x < s.target).length : ''));
  log('done');
})();
