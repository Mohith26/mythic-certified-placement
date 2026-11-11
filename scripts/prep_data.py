# Builds data/weights.js and data/images.js from the Zero-DCE authors' public weights (Epoch99.pth,
# parsed without torch by scripts/convert_weights.py) and public low-light test images (DICM, LIME).
#   python3 scripts/prep_data.py <dir with zerodce_weights.json and images>
import sys, json, base64, struct, os
from PIL import Image
src = sys.argv[1]; out = os.path.join(os.path.dirname(__file__), '..', 'data')
w = json.load(open(os.path.join(src, 'zerodce_weights.json')))
layers = []
for i in range(1, 8):
    W = w['e_conv%d.weight' % i]; b = w['e_conv%d.bias' % i]
    layers.append({'name': 'e_conv%d' % i, 'shape': W['shape'],
                   'w': base64.b64encode(struct.pack('<%df' % len(W['data']), *W['data'])).decode(),
                   'b': base64.b64encode(struct.pack('<%df' % len(b['data']), *b['data'])).decode()})
open(os.path.join(out, 'weights.js'), 'w').write(
    '// Zero-DCE (Guo et al., CVPR 2020) weights from github.com/Li-Chongyi/Zero-DCE (Epoch99.pth), CC BY-NC 4.0.\n'
    '(function (r) { const d = ' + json.dumps(layers) + ';\n'
    'if (typeof module === "object" && module.exports) module.exports = d; else r.ZDCE_WEIGHTS = d; })(typeof self !== "undefined" ? self : this);\n')
# calibration images (sensitivity + ADC ranges) and held-out verification images
sets = {'calib': ['DICM_01.jpg', 'DICM_05.jpg', 'DICM_12.jpg'], 'verify': ['DICM_26.jpg', 'LIME_6.bmp', 'DICM_41.jpg']}
imgs = []
for split, names in sets.items():
    for n in names:
        im = Image.open(os.path.join(src, n)).convert('RGB')
        for size, tag in [((64, 48), 'small'), ((192, 144), 'view')]:
            t = im.copy(); t.thumbnail((10000, 10000))
            # center-crop to 4:3 then resize
            W0, H0 = t.size; tw = min(W0, H0 * 4 // 3); th = tw * 3 // 4
            t = t.crop(((W0 - tw) // 2, (H0 - th) // 2, (W0 - tw) // 2 + tw, (H0 - th) // 2 + th)).resize(size, Image.LANCZOS)
            imgs.append({'name': n, 'split': split, 'size': tag, 'w': size[0], 'h': size[1],
                         'rgb': base64.b64encode(t.tobytes()).decode()})
open(os.path.join(out, 'images.js'), 'w').write(
    '// Low-light test images from the DICM and LIME benchmark sets as distributed in github.com/Li-Chongyi/Zero-DCE, downscaled.\n'
    '(function (r) { const d = ' + json.dumps(imgs) + ';\n'
    'if (typeof module === "object" && module.exports) module.exports = d; else r.ZDCE_IMAGES = d; })(typeof self !== "undefined" ? self : this);\n')
print('ok', len(imgs))
