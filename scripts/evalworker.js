// Worker: evaluates a list of jobs {key, plan, base, n, corner} on the held-out images; prints JSON lines.
const { H, model, ver, refsV, calib } = require('./common.js');
const jobs = JSON.parse(process.argv[2]);
for (const j of jobs) {
  const r = H.evalPlan(model, ver, refsV, j.plan, j.n, j.corner, calib, H.DEFAULTS, j.base);
  process.stdout.write(JSON.stringify({ key: j.key, chips: r.map((x) => +x.worstPsnr.toFixed(4)) }) + '\n');
}
