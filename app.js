/* Hybrid placement report UI. Everything on the page is computed in the browser by core/hybrid.js:
   the 3^7 search, the Pareto ladder order, the fixed-sequence certification for the chosen target, energy,
   and the image outputs. The per-chip PSNRs come from data/plan.json (written by scripts/plan.js with the same
   core) and the certified plan's 100 chips can be re-simulated here with one click. */
(function () {
  'use strict';
  const H = window.HYB;
  const $ = (id) => document.getElementById(id);
  const tick = () => new Promise((r) => setTimeout(r, 0));
  const HOT = { years: 10, tempC: 105 };
  const MODE_NAME = { A: 'Analog ×1', A2: 'Analog ×2', D: 'Digital' };
  const st = { target: 40, model: null, calib: null, plan: null, data: null, charts: {}, sel: null };
  const CERT_BASE = 50000;

  if (window.Chart) {
    Chart.defaults.font.family = 'Inter, system-ui, sans-serif';
    Chart.defaults.font.size = 12;
    Chart.defaults.color = '#3a342e';
    Chart.defaults.plugins.tooltip.backgroundColor = '#001f33';
    Chart.defaults.plugins.tooltip.cornerRadius = 10;
    Chart.defaults.plugins.tooltip.padding = 10;
  }
  const fmt = (x, d) => x.toFixed(d == null ? 1 : d);
  const pj = (j) => (j * 1e12).toFixed(1) + ' pJ';
  const key = (p) => p.join();
  const mw = (p) => H.powerAt(st.model, p, 1920 * 1080, 30, H.DEFAULTS) * 1e3;
  const badges = (p) => p.map((x) => `<i class="mm m-${x}" title="${MODE_NAME[x]}">${x === 'A2' ? 'A₂' : x}</i>`).join('');
  const pfmt = (p) => (p < 1e-4 ? p.toExponential(1) : p < 0.01 ? p.toFixed(4) : p.toFixed(3));

  // ------------------------------------------------------------ selection for a target (all live)
  function select(t) {
    const D = st.data, lad = D.ladder;
    const fsq = H.fixedSequence(lad, (i) => lad[i].chips, t, D.eps, D.delta);
    const naive = H.naivePlan(st.model, D.sens, t, H.DEFAULTS);
    return { t, fsq, chosen: fsq.chosen >= 0 ? lad[fsq.chosen] : null, naive: naive ? naive.plan : null };
  }
  const holdFails = (p, t) => { const h = p && st.data.holdout[key(p)]; return h ? { k: h.filter((x) => x < t).length, n: h.length } : null; };

  function render() {
    const t = st.target, D = st.data, m = st.model;
    const t0 = performance.now();
    const sel = select(t); st.sel = sel;
    const ms = performance.now() - t0;
    const c = sel.chosen; st.plan = c.plan;
    const allD = m.map(() => 'D'), pD = mw(allD), pPlan = mw(c.plan);
    const r = sel.fsq.rows[sel.fsq.chosen];
    const tested = sel.fsq.rows.filter((x) => x.tested).length;
    $('searchInfo').textContent = `${Math.pow(3, 7).toLocaleString()} placements, ${D.ladder.length} on the frontier`;
    $('verifyInfo').textContent = `${tested} plans walked on ${r.n} chips`;
    $('sPower').textContent = Math.round(pPlan);
    $('sPowerSub').textContent = `vs ${Math.round(pD)} mW all-digital INT8`;
    $('sX').textContent = (pD / pPlan).toFixed(1) + '×';
    $('sXSub').textContent = `than all-digital, certified at ≥ ${t} dB`;
    $('sChips').textContent = `${r.n - r.k}/${r.n}`;
    $('sChipsSub').textContent = `≥ ${t} dB after 10 years at 105 °C`;
    const nh = holdFails(sel.naive, t);
    $('sNaive').textContent = nh ? `${Math.round(100 * nh.k / nh.n)}%` : '–';
    $('sNaiveSub').textContent = nh ? `of ${nh.n} held-out chips below ${t} dB (${Math.round(mw(sel.naive))} mW plan)` : 'not simulated for this target';
    drawDiagram(c.plan);
    renderLayerTable(c.plan);
    renderLadder(sel);
    renderHist(c.chips, false);
    renderHoldout();
    renderCorners();
    drawFrontier();
  }

  // ------------------------------------------------------------ diagram
  function drawDiagram(plan) {
    const m = st.model, cfg = H.DEFAULTS;
    const W = 1160, yA = 88, yD = 238, bw = 98, bh = 54;
    const xs = [268, 378, 488, 598, 708, 818, 928];
    const die = { x: 'D' };
    m.forEach((L, i) => { die[L.out] = plan[i] === 'D' ? 'D' : 'A'; });
    const Y = (d) => (d === 'A' ? yA : yD);
    const pos = { x: 160 };
    m.forEach((L, i) => { pos[L.out] = xs[i]; });
    let s = '';
    s += `<rect x="8" y="${yA - 50}" width="${W - 16}" height="100" rx="18" fill="#e3f2fb"/>`;
    s += `<rect x="8" y="${yD - 50}" width="${W - 16}" height="100" rx="18" fill="#efe6d8"/>`;
    s += `<text x="26" y="${yA - 6}" font-size="12" font-weight="700" letter-spacing="2" fill="#018eda">ANALOG DIE</text><text x="26" y="${yA + 12}" font-size="11" fill="#3a5b72">GF 28nm flash</text><text x="26" y="${yA + 26}" font-size="11" fill="#3a5b72">ACE tiles</text>`;
    s += `<text x="26" y="${yD - 6}" font-size="12" font-weight="700" letter-spacing="2" fill="#001f33">DIGITAL DIE</text><text x="26" y="${yD + 12}" font-size="11" fill="#6b6259">TSMC N5A</text><text x="26" y="${yD + 26}" font-size="11" fill="#6b6259">SuperTile</text>`;
    const sent = new Set();
    let crossBits = 0;
    const edge = (from, toX, toDie, ch, skip, k) => {
      const fx = pos[from] + (from === 'x' ? 34 : bw / 2), fy = Y(die[from]);
      const tx = toX - bw / 2, ty = Y(toDie);
      const cross = die[from] !== toDie;
      const key = from + '>' + toDie;
      const first = cross && !sent.has(key);
      if (first) { sent.add(key); crossBits += ch * cfg.actBits; }
      const col = cross ? '#c2410c' : skip ? '#9b9084' : '#3a342e';
      let d, lx, ly;
      if (!skip) {
        d = `M${fx},${fy} C${fx + 30},${fy} ${tx - 30},${ty} ${tx - 2},${ty}`;
        lx = (fx + tx) / 2; ly = (fy + ty) / 2;
      } else if (!cross) {
        const up = toDie === 'A';
        const yy = up ? yA - 42 - k * 7 : yD + 42 + k * 7;
        const sx = pos[from], ex = toX - 14;
        d = `M${sx},${fy + (up ? -bh / 2 : bh / 2)} C${sx},${yy} ${ex},${yy} ${ex},${ty + (up ? -bh / 2 : bh / 2) - (up ? 2 : -2)}`;
      } else {
        const sx = pos[from] + 20, ex = toX - 20;
        d = `M${sx},${fy + (die[from] === 'A' ? bh / 2 : -bh / 2)} C${sx + 40},${(fy + ty) / 2} ${ex - 40},${(fy + ty) / 2} ${ex},${ty + (toDie === 'A' ? bh / 2 : -bh / 2) + (toDie === 'A' ? 2 : -2)}`;
        lx = (sx + ex) / 2; ly = (fy + ty) / 2;
      }
      s += `<path d="${d}" fill="none" stroke="${col}" stroke-width="${cross ? 2.2 : skip ? 1.3 : 1.6}" ${skip && !cross ? 'stroke-dasharray="4 4"' : ''} marker-end="url(#ar${cross ? 'c' : skip ? 'g' : 'k'})"/>`;
      if (first) s += `<text x="${lx}" y="${ly + 4}" font-size="11" font-weight="700" fill="#c2410c" text-anchor="middle" paint-order="stroke" stroke="#fffaf3" stroke-width="4">${ch * cfg.actBits} b/px</text>`;
    };
    let k = 0;
    m.forEach((L, i) => {
      for (const inp of L.inputs) {
        const prev = i === 0 ? 'x' : m[i - 1].out;
        const ch = inp === 'x' ? 3 : m.find((M) => M.out === inp).cout;
        edge(inp, xs[i], die[L.out], ch, inp !== prev, inp !== prev ? k++ : 0);
      }
    });
    pos.r = xs[6];
    edge('r', 1060, 'D', 24, false, 0);
    s += box(pos.x, yD, 68, 40, '#fffaf3', '#001f33', 'MIPI in', '3 ch');
    s += box(1060 + 2, yD, 118, bh, '#fffaf3', '#001f33', 'Curve LE ×8', 'x + r(x² − x)');
    m.forEach((L, i) => {
      const mode = plan[i];
      const fill = mode === 'A' ? '#018eda' : mode === 'A2' ? '#7fc8f0' : '#001f33';
      const txt = mode === 'A2' ? '#001d31' : '#faf1e5';
      s += box(xs[i], Y(die[L.out]), bw, bh, fill, txt, L.name, `${L.cin}→${L.cout} · ${MODE_NAME[mode]}`);
    });
    const defs = `<defs>${['k:#3a342e', 'g:#9b9084', 'c:#c2410c'].map((t) => { const [id, c] = t.split(':'); return `<marker id="ar${id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${c}"/></marker>`; }).join('')}</defs>`;
    $('diagram').innerHTML = defs + s;
    const e = H.energyPerPixel(m, plan, cfg);
    $('diagSum').innerHTML = `<span><i class="sw" style="background:#018eda"></i>Analog ×1</span><span><i class="sw" style="background:#7fc8f0"></i>Analog ×2 (two cells per weight)</span><span><i class="sw" style="background:#001f33"></i>Digital INT8</span><span><i class="ln"></i>die-to-die: <b>${crossBits} bits/px</b>, ${pj(e.d2d)}/px</span><span>compute ${pj(e.mac)}/px · total <b>${pj(e.joules)}/px</b></span>`;
  }
  function box(cx, cy, w, h, fill, txt, t1, t2) {
    return `<g><rect x="${cx - w / 2}" y="${cy - h / 2}" width="${w}" height="${h}" rx="14" fill="${fill}" stroke="${fill === '#fffaf3' ? '#cfc2b0' : 'none'}"/><text x="${cx}" y="${cy - 3}" text-anchor="middle" font-size="14" font-weight="700" fill="${txt}">${t1}</text><text x="${cx}" y="${cy + 15}" text-anchor="middle" font-size="10.5" fill="${txt}" opacity=".92">${t2}</text></g>`;
  }

  // ------------------------------------------------------------ layer table
  function renderLayerTable(plan) {
    const m = st.model, cfg = H.DEFAULTS, sens = st.data.sens;
    const tot = H.energyPerPixel(m, plan, cfg).joules;
    const dB = (ex, j) => { const b = sens.base; let w = 0; for (let k = 0; k < b.length; k++) w = Math.max(w, H.psnr(b[k]) - H.psnr(b[k] + ex[k])); return w; };
    $('layerTbl').querySelector('tbody').innerHTML = m.map((L, i) => {
      const mode = plan[i];
      const eMac = mode === 'D' ? cfg.eMacD : mode === 'A2' ? cfg.eMacA2 : cfg.eMacA;
      const e = L.macsPerPx * eMac;
      const lA = dB(sens.table[i].A.perImage), lA2 = dB(sens.table[i].A2.perImage);
      return `<tr><td><b>${L.name}</b> <span class="fine">${L.inputs.length > 1 ? 'cat[' + L.inputs.join(', ') + ']' : L.inputs[0]} → ${L.out}</span></td>
        <td class="mono">${L.cout}×${L.cin}×3×3</td><td>${L.macsPerPx.toLocaleString()}</td>
        <td>−${fmt(lA, 2)} dB · −${fmt(lA2, 2)} dB</td><td><span class="mode m-${mode}">${MODE_NAME[mode]}</span></td>
        <td>${pj(e)}</td><td><div class="bar"><i style="width:${Math.max(2, 100 * e / tot).toFixed(1)}%"></i></div></td></tr>`;
    }).join('');
    const ex = H.energyPerPixel(m, plan, cfg);
    $('tblNote').textContent = `Error if analog: worst-image PSNR lost (vs the all-digital pipeline) when only this layer is analog, averaged over 12 simulated chips at 10 years / 105 °C on the 3 calibration images. Die-to-die transfers add ${pj(ex.d2d)} per pixel (${(100 * ex.d2d / ex.joules).toFixed(0)}% of the total). Energy per MAC: analog ${(cfg.eMacA * 1e15).toFixed(1)} fJ, analog ×2 ${(cfg.eMacA2 * 1e15).toFixed(1)} fJ, digital INT8 ${(cfg.eMacD * 1e12).toFixed(2)} pJ (assumptions).`;
  }

  // ------------------------------------------------------------ certification ladder
  function renderLadder(sel) {
    const D = st.data, lad = D.ladder, t = sel.t;
    $('ladTbl').querySelector('tbody').innerHTML = sel.fsq.rows.map((r) => {
      const x = lad[r.i], chosen = r.i === sel.fsq.chosen;
      const res = !r.tested ? '<span class="pill-s gray">Not tested</span>' : r.ok ? `<span class="pill-s ${chosen ? 'blue' : 'green'}">${chosen ? 'Selected' : 'Certified'}</span>` : '<span class="pill-s red">Stop</span>';
      return `<tr class="${chosen ? 'sel' : ''} ${r.tested ? '' : 'dim'}" id="lad${r.i}"><td>${r.i + 1}</td><td class="pl">${badges(x.plan)}</td><td>${Math.round(x.powerMw)}</td><td>${fmt(x.pred)} dB</td><td>${r.tested ? `${r.k} of ${r.n}` : '–'}</td><td>${r.tested ? pfmt(r.p) : '–'}</td><td>${res}</td></tr>`;
    }).join('');
    const stop = sel.fsq.rows.find((r) => r.tested && !r.ok);
    const ch = sel.fsq.rows[sel.fsq.chosen];
    $('ladNote').innerHTML = `The ${lad.length} Pareto-optimal placements are ordered from safest to cheapest using only the calibration images, before any certification chip is simulated. Each plan runs on the same ${ch.n} fresh chips (3 held-out images, 10 years at 105 °C) and is certified if that many failures would be unlikely (p ≤ ${D.delta}) were more than ${Math.round(100 * D.eps)}% of all chips below ${t} dB. The walk stops at the first plan that can't be certified${stop ? ` (#${stop.i + 1}: ${stop.k} of ${stop.n} below target)` : ''}. Because the order is fixed in advance, every certified plan keeps its guarantee at once, so taking the cheapest one costs nothing statistically (fixed-sequence testing, as in Learn-then-Test).`;
    const wrap = document.querySelector('.ladder-wrap'), row = $('lad' + sel.fsq.chosen);
    if (wrap && row) wrap.scrollTop = Math.max(0, row.offsetTop - wrap.clientHeight / 2);
  }

  function renderHist(chips, live) {
    const t = st.target, sel = st.sel;
    const lo = Math.floor(Math.min(...chips, t) - 1), hi = Math.ceil(Math.max(...chips) + 1);
    const bins = []; for (let b = lo; b < hi; b += 0.5) bins.push({ b, n: 0 });
    for (const c of chips) { const k = Math.min(bins.length - 1, Math.floor((c - lo) / 0.5)); bins[k].n++; }
    if (st.charts.hist) st.charts.hist.destroy();
    st.charts.hist = new Chart($('histChart'), {
      type: 'bar',
      data: { labels: bins.map((x) => x.b.toFixed(1)), datasets: [{ label: 'chips', data: bins.map((x) => x.n), backgroundColor: bins.map((x) => (x.b + 0.5 <= t ? '#c2410c' : '#018eda')), borderRadius: 6, barPercentage: 0.92, categoryPercentage: 0.92 }] },
      options: { animation: false, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { title: (it) => `${it[0].label}–${(+it[0].label + 0.5).toFixed(1)} dB`, label: (it) => `${it.parsed.y} chips` } } },
        scales: { x: { title: { display: true, text: 'worst-image PSNR vs FP32 (dB), 10 years at 105 °C' }, grid: { display: false }, ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 12 } }, y: { title: { display: true, text: 'chips' }, beginAtZero: true, grid: { color: '#eadfce' }, ticks: { precision: 0 } } } },
      plugins: [{ id: 'target', afterDraw(c) { const xs = c.scales.x, k = (t - lo) / 0.5 - 0.5; const px = xs.getPixelForValue(k); const ctx = c.ctx; ctx.save(); ctx.strokeStyle = '#001f33'; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(px, c.chartArea.top); ctx.lineTo(px, c.chartArea.bottom); ctx.stroke(); ctx.fillStyle = '#001f33'; ctx.font = '600 12px Inter, system-ui, sans-serif'; ctx.fillText(`target ${t} dB`, px + 6, c.chartArea.top + 14); ctx.restore(); } }],
    });
    const n = chips.length, k = chips.filter((x) => x < t).length;
    $('histTitle').textContent = `${n} chips at the worst corner`;
    const h = holdFails(sel.chosen.plan, t);
    const up = h ? H.cpUpper(h.k, h.n, 0.05) : null;
    $('verNote').innerHTML = `${k} of ${n} certification chips fall below ${t} dB; the worst is ${fmt(Math.min(...chips))} dB and the median ${fmt(chips.slice().sort((a, b) => a - b)[n >> 1])} dB.${h ? ` On ${h.n} more chips it never saw, <b class="${h.k / h.n <= st.data.eps ? 'ok' : 'bad'}">${h.k} fall below</b> (${fmt(100 * h.k / h.n)}%, exact one-sided 95% bound ${fmt(100 * up)}%).` : ''}${live ? ' <b>Re-simulated in this browser.</b>' : ''}`;
  }

  async function rerun() {
    const c = st.sel.chosen, btn = $('rerunBtn');
    btn.disabled = true;
    const ver = st.images.filter((i) => i.split === 'verify' && i.w === 64);
    const refs = ver.map((i) => H.forward(st.model, i, null, 0, null, st.calib));
    const out = [], t0 = performance.now();
    for (let k = 0; k < c.chips.length; k++) {
      out.push(H.evalPlan(st.model, ver, refs, c.plan, 1, HOT, st.calib, H.DEFAULTS, CERT_BASE + k)[0].worstPsnr);
      if (k % 2 === 1) { $('rerunStatus').textContent = `chip ${k + 1} of ${c.chips.length}…`; await tick(); }
    }
    const same = out.every((x, k) => Math.abs(x - c.chips[k]) < 1e-3);
    $('rerunStatus').textContent = `${out.length} chips × 3 images in ${((performance.now() - t0) / 1000).toFixed(0)} s · ${same ? 'matches data/plan.json chip for chip' : 'differs from data/plan.json'}`;
    renderHist(out, true);
    btn.disabled = false;
  }

  // ------------------------------------------------------------ holdout, frontier, corners
  function renderHoldout() {
    const D = st.data;
    const rows = D.targets.map((t) => { const s = select(t); return { t, s, nh: holdFails(s.naive, t), ch: holdFails(s.chosen && s.chosen.plan, t) }; });
    st.holdRows = rows;
    const pct = (h) => `${h.k} of ${h.n} <span class="fine">(${fmt(100 * h.k / h.n)}%)</span>`;
    $('holdTbl').querySelector('tbody').innerHTML = rows.filter((r) => r.t % 2 === 0 || r.t === st.target).map((r) => `<tr class="${r.t === st.target ? 'sel' : ''}"><td><b>${r.t} dB</b></td>
      <td class="pl">${badges(r.s.naive)}</td><td>${Math.round(mw(r.s.naive))}</td><td class="${r.nh.k / r.nh.n > D.eps ? 'bad' : 'ok'}">${pct(r.nh)}</td>
      <td class="pl">${badges(r.s.chosen.plan)}</td><td>${Math.round(r.s.chosen.powerMw)}</td><td class="${r.ch.k / r.ch.n > D.eps ? 'bad' : 'ok'}">${pct(r.ch)}</td><td>${fmt(100 * H.cpUpper(r.ch.k, r.ch.n, 0.05))}%</td></tr>`).join('');
    const worst = rows.reduce((a, r) => (r.nh.k / r.nh.n > a.nh.k / a.nh.n ? r : a), rows[0]);
    const over = rows.filter((r) => r.nh.k / r.nh.n > D.eps).length;
    const cur = rows.find((r) => r.t === st.target), pD = mw(D.uniform.allD);
    $('holdNote').innerHTML = `The average-chip plan is the cheapest placement whose predicted mean error meets the target (per-layer sensitivity, no margin), which is what a sensitivity-ranked allocator does. It is cheaper, but it misses the ${Math.round(100 * D.eps)}% budget at ${over} of ${rows.length} targets; at ${cur.t} dB, ${fmt(100 * cur.nh.k / cur.nh.n)}% of held-out chips fall below. The certified plan costs ${Math.round(cur.s.chosen.powerMw - mw(cur.s.naive))} mW more and is still ${(pD / cur.s.chosen.powerMw).toFixed(1)}× below all-digital (${Math.round(pD)} mW, dotted). Held-out chip ids are disjoint from every chip either planner used.`;
  }
  function drawFrontier() {
    const D = st.data, rows = st.holdRows;
    const unif = [['All digital', D.uniform.allD, '#001f33'], ['All analog ×2', D.uniform.allA2, '#7fc8f0'], ['All analog ×1', D.uniform.allA, '#9b9084']];
    if (st.charts.front) st.charts.front.destroy();
    st.charts.front = new Chart($('frontChart'), {
      type: 'line',
      data: { labels: rows.map((r) => r.t), datasets: [
        { label: 'Certified plan', data: rows.map((r) => r.s.chosen.powerMw), borderColor: '#018eda', backgroundColor: '#018eda', pointRadius: rows.map((r) => (r.t === st.target ? 6 : 3)), borderWidth: 2.5, stepped: false, tension: 0 },
        { label: 'Average-chip plan', data: rows.map((r) => mw(r.s.naive)), borderColor: '#9b9084', backgroundColor: '#9b9084', borderDash: [5, 4], pointRadius: 2, borderWidth: 2 },
        { label: 'All digital', data: rows.map(() => mw(D.uniform.allD)), borderColor: '#001f33', pointRadius: 0, borderWidth: 1.5, borderDash: [2, 3] },
      ] },
      options: { animation: false, maintainAspectRatio: false,
        scales: { x: { title: { display: true, text: 'quality target (dB PSNR vs FP32)' }, grid: { display: false } }, y: { type: 'logarithmic', title: { display: true, text: 'mW at 1080p30' }, grid: { color: '#eadfce' }, ticks: { callback: (v) => ([50, 100, 200, 500, 1000].includes(v) ? v : '') } } },
        plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, usePointStyle: true } }, tooltip: { callbacks: { title: (it) => `target ${it[0].label} dB`, label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(0)} mW` } } } },
    });
  }
  function renderCorners() {
    const D = st.data, t = st.target, c = st.sel.chosen;
    const rows = [[`Certified plan (${t} dB)`, c.plan], ['All analog ×1', D.uniform.allA], ['All analog ×2', D.uniform.allA2], ['All digital INT8', D.uniform.allD]];
    $('cornerTbl').querySelector('tbody').innerHTML = rows.map(([lab, p]) => {
      const cs = D.corners[key(p)];
      return `<tr><td><b>${lab}</b> <span class="pl">${badges(p)}</span></td>${cs.map((x) => { const w = Math.min(...x.chips), k = x.chips.filter((v) => v < t).length; return `<td class="${k / x.chips.length > st.data.eps ? 'bad' : ''}">${fmt(w)} dB <span class="fine">· ${k} of ${x.chips.length} below</span></td>`; }).join('')}<td>${Math.round(mw(p))}</td></tr>`;
    }).join('');
    $('cornerNote').textContent = `Worst chip and count below ${t} dB; red where more than ${Math.round(100 * D.eps)}% of chips miss. Corner chips are a third disjoint sample (40 per condition). Retention drift follows an Arrhenius model (Ea 0.6 eV) referenced to 10 years at 105 °C; temperature-coefficient residuals grow with distance from 25 °C. Digital INT8 does not age in this model.`;
  }

  function paint(canvas, chw, w, h) {
    const ctx = canvas.getContext('2d'), im = ctx.createImageData(w, h), n = w * h;
    for (let p = 0; p < n; p++) for (let c = 0; c < 3; c++) im.data[p * 4 + c] = Math.max(0, Math.min(255, Math.round(chw[c * n + p] * 255)));
    for (let p = 0; p < n; p++) im.data[p * 4 + 3] = 255;
    ctx.putImageData(im, 0, 0);
  }
  async function renderImages() {
    const name = $('imgSel').value, chip = +$('chipSel').value;
    const img = st.images.find((i) => i.name === name && i.w === 192);
    $('imgNote').textContent = 'computing 3 forward passes…'; await tick();
    const t0 = performance.now();
    const ref = H.forward(st.model, img, null, 0, null, st.calib);
    const outA = H.forward(st.model, img, st.model.map(() => 'A'), chip, HOT, st.calib, H.DEFAULTS, 1);
    const outP = H.forward(st.model, img, st.plan, chip, HOT, st.calib, H.DEFAULTS, 1);
    const errMap = (o) => { const e = new Float32Array(o.length); for (let i = 0; i < o.length; i++) e[i] = Math.min(1, Math.abs(o[i] - ref[i]) * 16); return e; };
    paint($('cIn'), img.data, img.w, img.h); paint($('cRef'), ref, img.w, img.h); paint($('cA'), errMap(outA), img.w, img.h); paint($('cP'), errMap(outP), img.w, img.h);
    const pA = H.psnr(H.mse(outA, ref)), pP = H.psnr(H.mse(outP, ref));
    $('capA').innerHTML = `All-analog error ×16, chip #${chip - 300000}: <b class="${pA < st.target ? 'bad' : ''}">${fmt(pA)} dB</b>`;
    $('capP').innerHTML = `Certified plan error ×16, same chip: <b class="${pP < st.target ? 'bad' : 'ok'}">${fmt(pP)} dB</b>`;
    $('imgNote').textContent = `${img.w}×${img.h} · 10 years at 105 °C · three forward passes run in this browser`;
  }

  function exportPlan() {
    const c = st.sel.chosen, r = st.sel.fsq.rows[st.sel.fsq.chosen];
    const out = { model: 'zero_dce', target: 'vanguard (analog die + digital die)', quality_target_db: st.target, corner: '10y@105C',
      placement: Object.fromEntries(st.model.map((L, i) => [L.name, c.plan[i]])), mw_1080p30: +c.powerMw.toFixed(1),
      certification: { chips: r.n, below_target: r.k, p_value: r.p, eps: st.data.eps, delta: st.data.delta, method: 'fixed-sequence binomial test along the energy/quality Pareto frontier' } };
    const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' })); a.download = 'placement_zero_dce.json'; a.click();
  }

  async function boot() {
    st.data = window.PLAN || (await fetch('data/plan.json').then((r) => r.json()));
    st.model = H.loadModel(window.ZDCE_WEIGHTS);
    st.images = window.ZDCE_IMAGES.map(H.loadImage);
    st.calib = H.calibrate(st.model, st.images.filter((i) => i.split === 'calib' && i.w === 64));
    $('targetSel').innerHTML = st.data.targets.map((t) => `<option value="${t}" ${t === st.target ? 'selected' : ''}>≥ ${t} dB PSNR vs FP32</option>`).join('');
    $('imgSel').innerHTML = st.images.filter((i) => i.w === 192).map((i) => `<option value="${i.name}">${i.name.replace('_', ' ').replace(/\.(jpg|bmp)$/, '')} (${i.split === 'verify' ? 'held-out' : 'calibration'})</option>`).join('');
    const defImg = st.images.find((i) => i.w === 192 && i.name === 'LIME_6.bmp') || st.images.find((i) => i.w === 192 && i.split === 'verify');
    $('imgSel').value = defImg.name;
    const aChips = st.data.ladder.length ? (st.data.corners[key(st.data.uniform.allA)][2].chips) : [];
    const worstChip = aChips.indexOf(Math.min(...aChips));
    $('chipSel').innerHTML = Array.from({ length: 40 }, (_, k) => `<option value="${300000 + k}" ${k === worstChip ? 'selected' : ''}>#${k}${k === worstChip ? ' (worst all-analog)' : ''}</option>`).join('');
    render();
    await tick();
    await renderImages();
  }
  $('targetSel').addEventListener('change', async (e) => { st.target = +e.target.value; render(); await renderImages(); });
  $('imgSel').addEventListener('change', renderImages);
  $('chipSel').addEventListener('change', renderImages);
  $('rerunBtn').addEventListener('click', rerun);
  $('exportBtn').addEventListener('click', (e) => { e.preventDefault(); exportPlan(); });
  boot().catch((e) => { console.error(e); $('searchInfo').textContent = 'Error: ' + e.message; });
})();
