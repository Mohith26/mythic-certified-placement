# Certified hybrid placement for Mythic Vanguard (concept)

**Live demo:** https://mohith26.github.io/mythic-certified-placement/

Unofficial concept demo by Mohith Gajjela. Not affiliated with Mythic. The chip error and energy models are synthetic assumptions, not Mythic measurements.

## What it is

Mythic Vanguard pairs an analog compute-in-memory die with a digital die. A compiler for that platform has to decide, layer by layer, what runs in analog flash (cheap, but noisy and drifting) and what runs on the digital die (exact, but far more energy per MAC, plus die-to-die traffic).

This demo is a placement pass that answers that question with a **guarantee over the chip population at end of life**: it picks the cheapest placement for which *at most 5% of chips fall below the quality target after 10 years at 105 °C, with 95% confidence*.

It runs on a real network: **Zero-DCE**, the low-light enhancement model Mythic names for Project Starlight, with the authors' public weights and public low-light test images (DICM, LIME). Every number on the page is computed by `core/hybrid.js`, either live in the browser or by `scripts/plan.js` with the same code.

## Why it matters

Per-layer analog/digital allocation is usually done from a sensitivity ranking: measure how much each layer hurts accuracy when it goes analog, add a margin, pick a placement. That plans for the *average* chip. In this demo, the cheapest plan whose predicted mean error meets 40 dB is all-analog, and **49 of 150 held-out end-of-life chips (32.7%) fall below 40 dB**. The certified plan keeps one layer (conv2) digital: 170 mW at 1080p30, 4.3x less than all-digital INT8 (739 mW), and 1 of 150 held-out chips falls below 40 dB.

A part that ships to cars or cameras for ten years needs the second kind of statement.

## How it works

1. **Chip model** (`forward` in `core/hybrid.js`). Each simulated chip draws its own weight errors per analog layer: write-verify programming error, retention-drift residual after gain calibration (Arrhenius, Ea 0.6 eV, referenced to 10 years at 105 °C), temperature-coefficient residual, per-column gain residual, plus per-inference read noise and 8-bit DAC/ADC quantization with ranges calibrated on calibration images. Analog x2 averages two cells per weight. Digital layers are INT8 per-channel. Errors are seeded by (chip, layer), so every chip is reproducible.
2. **Energy model** (`energyPerPixel`). Per-MAC energy by mode (analog 16.7 fJ from 120 TOPS/W; digital INT8 0.15 pJ, an assumption) plus die-to-die transfer of every activation tensor that crosses dies, including the camera input and the curve maps that go back to the digital die.
3. **Sensitivity and surrogate.** Monte Carlo over 12 chips gives each layer's excess error per image in each analog mode. An additive surrogate predicts any placement's worst-image error; kappa (the 95th-percentile / mean ratio over 40 all-analog chips) turns it into a predicted 95th-percentile.
4. **Pareto ladder.** All 3^7 = 2,187 placements are scored; the 38 that are Pareto-optimal (energy vs predicted quality) are ordered from safest to cheapest. This order uses only calibration data.
5. **Certification (fixed-sequence testing, as in Learn-then-Test).** Every ladder plan is simulated on the same 100 fresh chips with 3 held-out images at the worst corner. For a target T, walk down the ladder: plan *i* is certified if P(Bin(100, 0.05) <= failures) <= 0.05, and the walk stops at the first plan that is not. Because the order is fixed in advance, all certified plans hold their guarantee simultaneously (family-wise error <= 0.05), so taking the cheapest one does not need a multiple-testing correction. The page re-runs this test live for any target.
6. **Independent checks.** 150 more chips (disjoint ids) check the certified and average-chip plans at every target; 40 chips per condition give a corner table (new at 25 °C, 5 years at 85 °C, 10 years at 105 °C).

## Run it

```
node --test test/              # 13 unit tests
node scripts/plan.js           # regenerates data/plan.json and data/plan.js (~10 min, 6 worker processes)
python3 -m http.server         # or just open index.html
```

`scripts/prep_data.py` and `scripts/convert_weights.py` show how the weights (Epoch99.pth, parsed without torch) and images were converted.

Tests cover: weight shapes, the convolution against a naive reference, the FP32 network brightening an image, INT8 determinism, analog error growing with drift and temperature, two cells reducing error, the energy ordering and die-crossing charges, the search against brute force, the binomial CDF and Clopper-Pearson bound, the Pareto property of the ladder, fixed-sequence stopping logic, a 4,000-trial simulation showing the family-wise false-certification rate stays under 5%, and exact reproduction of committed chip results from the core.

## Honest limitations

- **The chip model is synthetic.** Error magnitudes, drift law and energy per MAC are assumptions chosen to be plausible for flash CIM; they are not Mythic data. With Mythic's characterization data the same pipeline would run unchanged.
- **The guarantee is over chips, for these images.** It certifies the chip population on the 3 held-out images at one corner. It does not certify every possible input image.
- **The surrogate is optimistic for plans with many analog layers** (errors compound; e.g. the selected 40 dB plan is predicted at 44.5 dB p95 but its worst certification chip is 38.0 dB). Certification is what catches this; the surrogate only orders the ladder.
- **Fixed-sequence can stop early.** At 42 dB the walk stops at a plan with 4 failures even though a slightly cheaper plan further down would pass; the price is 6 mW. Graph-based multiple-testing procedures could skip around such a plan.
- Zero-DCE is tiny (79k parameters); a real Vanguard compile would also weigh tile capacity, latency and SRAM. Layers are placed whole, not split.
- Model weights and images are CC BY-NC 4.0 (Zero-DCE authors); this repo is non-commercial.

## Novelty note

Searched Mythic's site, newsroom, technology/workflow pages, the Vanguard and Starlight pages, the Videantis developer tools page and Mythic's Lever postings (Nov 2026) for automatic analog/digital layer partitioning, yield or end-of-life accuracy guarantees, and Monte Carlo verification; nothing like it is announced. The closest signal is a Mythic compiler job post about extending MLIR "across analog accelerators and digital subsystems", which is the infrastructure such a pass would sit on. Per-layer analog/digital allocation itself exists in research and patents (IBM's HILAL, DATE 2026; LIONHEART, IEEE TETC; Intel's application US 2026/0093972), and RACE-AIMC (arXiv 2609.03149) certifies selective inference on analog accelerators. None of these certify a placement's yield over the chip population at an end-of-life corner, or cost die-to-die traffic for a two-die package.

## License

Code: MIT. Data: see `data/` headers (CC BY-NC 4.0).
