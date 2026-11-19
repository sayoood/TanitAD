# MM reading: PROBE-N1D-FEAT (N1DW card section 10, R5), 2026-11-19

**Card and run**
- Card: `PREREG_V8F_N1DW.md` (sha256 59f2b7f9) section 10, with Amendment 1 (sha256 bff20b4d), which was landed while the result folder was still empty.
- Run: on the pod, CPU only, zero training, 14:20Z to 14:36Z.
- Result: `result/PROBE_RESULT.json` (sha256 2b7d3631), with the printout `result/READ.txt` (sha256 871a4076) and the per-window out-of-fold correctness `result/oof_correct.npz` (sha256 c0900ef7).
- Features: banked on private HF, `Sayood/tanitad-v8f-strat1-runs`, `n1dw_probe/n1dwprobe_feats.tar` (record `result/BANK_N1DWPROBE.json`).

**Evidence class and estimator**
- MEASURED. Tier: T0 decodability of frozen features, never driving performance.
- Population: the D1-HO grid, 5,736 windows / 497 held-out episodes. Class counts 1,934 / 1,862 / 1,940; majority share 0.338215.
- Estimator: paired episode-cluster bootstrap, B 2,000, seed 0. It answers "another draw of EPISODES" only. Training variance is carried by the two N1D seeds. The tree probe has no seed variance (Amendment 1).
- Bar: 0.0383.

## Outcome as committed (tree probe decides): PROBE-PRESENT-UNUSED at both N1D seeds

| tree accuracy | F-TEXT | F-NAV | F-PLAN | F-PLAN+TEXT | F-VIS |
|---|---|---|---|---|---|
| N1D_s0 | 0.6044 | 0.5934 | **0.7028** | 0.7333 | 0.5875 |
| N1D_s1 | 0.6044 | 0.5910 | **0.6937** | 0.7249 | 0.5614 |

| contrast (tree) | N1D_s0 | N1D_s1 | reading |
|---|---|---|---|
| F-NAV - F-TEXT | -0.0110 [-0.0197, -0.0023] | -0.0134 [-0.0225, -0.0043] | separated, below the bar: not LOST-AT-TOKEN |
| F-PLAN - F-NAV | +0.1093 [+0.0831, +0.1373] | +0.1027 [+0.0765, +0.1285] | not LOST-AFTER-TOKEN |
| **F-PLAN - F-TEXT** | **+0.0983 [+0.0725, +0.1241]** | **+0.0893 [+0.0634, +0.1143]** | PRESENT-UNUSED |
| F-PLAN+TEXT - F-PLAN | +0.0305 [+0.0216, +0.0400] | +0.0312 [+0.0210, +0.0411] | below the bar |
| F-VIS - F-TEXT | -0.0169 [-0.0478, +0.0137] | -0.0431 [-0.0728, -0.0138] | VISION not held |

**What it says.** The roundabout decision is decodable from the input of O's plan head at NOW (0.70 / 0.69), well above what the route text alone gives (0.60), on episodes N1D never trained on. The information is not lost at the nav token, nor between the token and the plan head. It is lost downstream: in how the plan head decodes it into a 6 s track, or over the rollout.

For scale, N1D's realised T1-kin TAC-LAT on the same windows is about 0.62 (s0) and 0.65 (s1): N1R's 0.5905 / 0.6328 plus NAVRBD's deltas. The plan feature carries more than the trained arm delivers.

**Beside (logistic probe, never deciding):** PROBE-PRESENT-UNUSED and PROBE-VISION. Linear F-VIS - F-TEXT is +0.0563 [+0.0262, +0.0854] and +0.0532 [+0.0241, +0.0827]. But linear F-TEXT reads only 0.5159, against 0.6044 for the tree: a linear probe cannot read the one-hot exit columns well. So this VISION reading is largely the text's linear weakness. Under the card, N1D8 stays parked.

## Controls
| control | value | status |
|---|---|---|
| K-CONST | majority share 0.338215 = the card's; the fold-wise train-majority predictor reads 0.3040 | PASS |
| K-DET | features re-extracted bit-identical, both checkpoints | PASS |
| K-PARITY (informational) | extractor vs the banked N1D dumps: 5,736 / 5,736 windows, tick-0 plan diff 0.0 m, labels agree on all | PASS |
| Regression arm, binding (Amendment 1: all 15 columns zeroed) | -0.3004 [-0.3379, -0.2621], both seeds | PASS |
| Regression arm as first committed (c0 and the exit columns zeroed) | -0.1309 [-0.1533, -0.1088], both seeds | PASS |
| **SHUF** (training labels permuted within folds) | max abs(acc - 0.338215) = **0.0403**, against a band of 0.02. Out of band: 6 of 40 cells (0.3161 to 0.3785), all at permutation seed 1 | **FAIL, as committed** |

- **The first-committed regression arm passes at full size.** Amendment 1 said it "cannot remove the signal". That was too strong. c13 and c14 keep only the roundabout support; the exit columns carry which side, and zeroing them costs 0.13 at full size. The 400-window smoke was too small to separate it. The amended all-15 arm stays binding, and both pass.
- **SHUF fails, so this reading carries CONTROL STATUS: PROBLEM SHUF.**
  - HYPOTHESIS for the cause: under episode-grouped folds, class shares differ by fold. A label-free predictor scores the test fold's share of its training-fold majority class, which can sit away from the global share; the fold-wise train-majority predictor reads 0.304, already 0.034 off. If so, a band of 0.02 around the global share is narrower than this design's null.
  - The deciding contrasts (+0.089 to +0.098) are paired on the same windows and folds, and are 2.2 to 2.4 times the largest SHUF deviation. That does not make SHUF a pass.
  - Next, zero GPU: SHUF over 50 permutation seeds per cell, giving the permutation null, read beside. It does not move the committed band.

## Next (RULE ZERO)
1. The card makes this the lever after an RBW-NO-GAIN. It is zero training and cheap, and the coordinator asked that it not wait. So the **T0-against-T1 read of N1D on D1-HO** is pre-registered now and runs before any N1DW number exists: `PREREG_N1D_T0T1.md`. Its two outcomes are DECODE-LOSES and ROLLOUT-LOSES, and it is read from the banked dumps.
2. The SHUF permutation null (above).
3. N1DW is unaffected: its bars and outcomes do not depend on this probe. The N1DW build is in progress.
