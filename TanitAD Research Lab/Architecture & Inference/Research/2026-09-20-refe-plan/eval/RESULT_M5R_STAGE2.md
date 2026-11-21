# RESULT -- Measure 5r STAGE 2 (fresh-token confirmation; Amendment 5's 923 tokens, 43 logs)

*Pre-registration `eval/PREREG_MEASURE5R.md` §6 + ADDENDUM 1 (blob `2f32042b`; scope-prose correction `ff3fd2f3`). Applied as written. Numbers are written by `eval/m5r_write_result_stage2.py` from `eval/raw/m5r/readout_stage2.json`; none is typed.*

## Stage-2 decision (as registered): **ADOPT V3r**

Registered consequence: a DRAFTED pod kit for the adopted arm; it ships ONLY on the PI's go (nothing has been sent to the pod).

Gates: {"stage2_gate_context.json": true, "stage2_gate_eval.json": true} (context admissibility: `stage2_gate_context.json`; eval: `stage2_gate_eval.json`). Estimator: paired log-cluster bootstrap over the 43 logs, 10,000 resamples, 95 % percentile, seed 20260927; n = 923 tokens, 43 logs.

| arm vs V3 | E1 (>= +0.03, lo > 0, > floor) | seed floor | E3a (lo > -0.01) | E3b (lo > -2.0) | E2 (lo > -2.0) | verdict |
|---|---|---|---|---|---|---|
| V4r | +0.0315 [+0.0002, +0.0612] (separated) | 0.02652 | +0.0197 [+0.0090, +0.0300] (separated) | +1.79 [+0.08, +3.60] (separated) | +7.08 [+4.93, +9.13] (separated) | **PASS** |
| V3r | +0.0339 [+0.0146, +0.0528] (separated) | 0.02652 | +0.0141 [+0.0037, +0.0241] (separated) | +2.00 [+0.53, +3.57] (separated) | +6.54 [+4.59, +8.39] (separated) | **PASS** |

Named bar (both pass -> V3r unless): E1 V4r - V3r = -0.0024 [-0.0209, +0.0142], seed floor 0.02075 -> V4r beats V3r: **False**.

## All comparisons (reported)

| metric | pair | diff [95 %] |
|---|---|---|
| E1_masked | V4r_minus_V3 | +0.0315 [+0.0002, +0.0612] (separated) |
| E1_masked | V3r_minus_V3 | +0.0339 [+0.0146, +0.0528] (separated) |
| E1_masked | V4r_minus_V3r | -0.0024 [-0.0209, +0.0142] |
| E1_masked | V4r_minus_V4 | +0.0625 [+0.0356, +0.0893] (separated) |
| E3a | V4r_minus_V3 | +0.0197 [+0.0090, +0.0300] (separated) |
| E3a | V3r_minus_V3 | +0.0141 [+0.0037, +0.0241] (separated) |
| E3a | V4r_minus_V3r | +0.0055 [+0.0022, +0.0086] (separated) |
| E3a | V4r_minus_V4 | +0.0185 [+0.0072, +0.0295] (separated) |
| E3b | V4r_minus_V3 | +1.79 [+0.08, +3.60] (separated) |
| E3b | V3r_minus_V3 | +2.00 [+0.53, +3.57] (separated) |
| E3b | V4r_minus_V3r | -0.22 [-0.79, +0.31] |
| E3b | V4r_minus_V4 | +2.12 [+0.38, +3.98] (separated) |
| E2 | V4r_minus_V3 | +7.08 [+4.93, +9.13] (separated) |
| E2 | V3r_minus_V3 | +6.54 [+4.59, +8.39] (separated) |
| E2 | V4r_minus_V3r | +0.53 [-0.29, +1.43] |
| E2 | V4r_minus_V4 | +7.37 [+5.17, +9.62] (separated) |

## EP and NAVSIM sub-scores of the picks (x100; NC, DAC, EP, TTC, C, DDC)

* A0: E3b [92.199, 91.766, 73.385, 83.64, 97.183, 92.795]; E2 [95.179, 95.991, 61.315, 89.816, 78.657, 96.696]
* V3: E3b [95.414, 93.102, 76.904, 88.696, 97.58, 94.402]; E2 [98.267, 96.208, 61.341, 94.8, 76.923, 97.382]
* V3r: E3b [97.129, 92.958, 78.88, 91.116, 97.833, 95.739]; E2 [97.057, 93.355, 76.701, 90.755, 96.244, 95.684]
* V4: E3b [95.269, 92.741, 76.758, 88.01, 97.616, 94.113]; E2 [96.443, 95.197, 63.557, 90.466, 87.252, 97.201]
* V4r: E3b [96.93, 92.669, 78.814, 91.116, 97.833, 95.341]; E2 [96.334, 93.897, 77.757, 90.358, 97.436, 95.395]

| route | pair | EP | NC | DAC | TTC | C |
|---|---|---|---|---|---|---|
| E3b | V4r_minus_V3 | +1.91 [+0.14, +3.77] | +1.52 [+0.15, +3.04] | -0.43 [-1.50, +0.58] | +2.42 [+0.14, +4.75] | +0.25 [+0.04, +0.55] |
| E3b | V3r_minus_V3 | +1.98 [+0.47, +3.58] | +1.72 [+0.41, +3.16] | -0.14 [-1.01, +0.74] | +2.42 [+0.14, +4.72] | +0.25 [+0.04, +0.55] |
| E3b | V4r_minus_V3r | -0.07 [-0.68, +0.49] | -0.20 [-0.54, +0.12] | -0.29 [-0.80, +0.17] | +0.00 [-0.45, +0.47] | +0.00 [+0.00, +0.00] |
| E2 | V4r_minus_V3 | +16.42 [+13.55, +19.35] | -1.93 [-3.47, -0.60] | -2.31 [-4.48, -0.38] | -4.44 [-6.87, -2.31] | +20.51 [+14.05, +27.48] |
| E2 | V3r_minus_V3 | +15.36 [+12.64, +18.17] | -1.21 [-2.57, -0.03] | -2.85 [-5.11, -0.86] | -4.04 [-6.23, -2.04] | +19.32 [+13.46, +25.50] |
| E2 | V4r_minus_V3r | +1.06 [+0.13, +2.17] | -0.72 [-1.32, -0.15] | +0.54 [-0.44, +1.70] | -0.40 [-1.18, +0.37] | +1.19 [+0.00, +3.55] |

E2 selection mix: {"A0": {"original": 411, "copy_075": 191, "stop": 321}, "V3_s0": {"original": 406, "copy_075": 108, "stop": 409}, "V3_s1": {"original": 490, "copy_075": 94, "stop": 339}, "V3_s2": {"original": 406, "copy_075": 138, "stop": 379}, "V3r_s0": {"original": 757, "copy_075": 144, "stop": 22}, "V3r_s1": {"original": 771, "copy_075": 134, "stop": 18}, "V3r_s2": {"original": 723, "copy_075": 177, "stop": 23}, "V4_s0": {"original": 224, "copy_075": 398, "stop": 301}, "V4_s1": {"original": 268, "copy_075": 391, "stop": 264}, "V4_s2": {"original": 246, "copy_075": 453, "stop": 224}, "V4r_s0": {"original": 645, "copy_075": 272, "stop": 6}, "V4r_s1": {"original": 627, "copy_075": 293, "stop": 3}, "V4r_s2": {"original": 630, "copy_075": 281, "stop": 12}}

<!-- HAND-WRITTEN BELOW THIS LINE (kept on re-run) -->
## Interpretation, four families, declared departures

*Written by the coordinator 2026-10-01 after the M5r agent stopped at its API limit with this section empty. Numbers above are the agent's machine-written readout; the text below adds only reading, no new measurement.*

**What it says.** Both candidate scorers (V4r = repaired labels + 0.75x slowed copies; V3r = repaired labels only) beat the V3 control on the registered bars on 923 FRESH tokens / 43 logs (paired log-cluster bootstrap, 3 seeds per arm): slow-plan ranking E1 +0.034 [+0.015, +0.053] for V3r (+0.032 [+0.000, +0.061] for V4r), ranking among the 64 E3a, and the PDMS of the pick among the original 64 E3b +2.00 [+0.53, +3.57] (V3r) / +1.79 [+0.08, +3.60] (V4r). V4r does NOT beat V3r (E1 V4r - V3r -0.002 [-0.021, +0.014]), so the registered rule adopts the cheaper V3r.

**Why the headline E2 (+6.5 to +7.1) is not the clean number.** E2 picks among the 64 originals PLUS the slowed copies and STOP. The V3 control picks STOP on ~375 of 923 tokens (409 / 339 / 379 over its 3 seeds); V3r picks it on ~21 (22 / 18 / 23). The repaired labels remove a defect in how slow candidates were ranked, which is the mechanism the amendment was built to test, but the like-for-like effect on the original 64 is E3b, about +2 PDMS.

**Caveats.** E1's V4r lower bound is +0.0002 and E3b's +0.08, both thin. The arms are one fine-tune each of 3 seeds on the dev box GPU (not the pod's training run), so this says the labels help a scorer fine-tuned from the same checkpoint; it does not show a re-trained pod run would gain the same. The CI answers another draw of episodes; the 3 seeds give the training-variance floor (0.0265).

**Declared departures.** The addendum added V3r after Stage 1 (registered before any Stage-2 output existed, blob 2f32042b). Stage 1 read on exposed tokens; Stage 2 is the confirmation. Families (longitudinal / lateral / tactical, strategic UNAVAILABLE in NAVSIM) were NOT computed for Stage 2: a work item, not yet done. The drafted pod kit was NOT written; the pod run it would change has finished, so it now belongs to the next training version and still needs the PI's go.
