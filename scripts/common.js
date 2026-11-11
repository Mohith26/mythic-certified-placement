// Shared setup for the planning scripts: model, images, calibration, FP32 references.
const H = require('../core/hybrid.js');
const model = H.loadModel(require('../data/weights.js'));
const imgs = require('../data/images.js').map(H.loadImage);
const cal = imgs.filter((i) => i.split === 'calib' && i.w === 64);
const ver = imgs.filter((i) => i.split === 'verify' && i.w === 64);
const calib = H.calibrate(model, cal);
const refsC = cal.map((i) => H.forward(model, i, null, 0, null, calib));
const refsV = ver.map((i) => H.forward(model, i, null, 0, null, calib));
const HOT = { years: 10, tempC: 105, label: '10 years at 105 °C' };
const CORNERS = [{ years: 0, tempC: 25, label: 'New, 25 °C' }, { years: 5, tempC: 85, label: '5 years at 85 °C' }, HOT];
// chip-id ranges (disjoint): sensitivity 1000+, kappa 9000+, surrogate check 70000+, certification 50000+, holdout 200000+
const CHIPS = { certify: 50000, holdout: 200000, corners: 300000 };
module.exports = { H, model, cal, ver, calib, refsC, refsV, HOT, CORNERS, CHIPS };
