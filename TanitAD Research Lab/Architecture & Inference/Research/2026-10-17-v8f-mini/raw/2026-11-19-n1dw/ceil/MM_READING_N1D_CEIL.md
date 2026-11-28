# MM reading: PREREG_N1D_CEIL. The route and the ego state determine no more of the roundabout decision than O's plan feature holds; all inputs together reach 0.73 (2026-11-19)

**Inputs**
- Card: `../PREREG_N1D_CEIL.md` (sha256 092bb25c), frozen and landed at e0185d36 before any ego feature was fitted.
- Result: `RESULT_CEIL.json` (sha256 af609542), printout `READ_CEIL.txt` (sha256 1e628da6), `oof_correct.npz` (sha256 a5370047), code in `code/`.
- Features: banked PRIVATE on HF `Sayood/tanitad-v8f-strat1-runs`, `n1dw_ceil/n1dwceil_feats.tar` (sha256 00740a5f, hub equal; record `BANK_N1DWCEIL.json`).

**Evidence and estimator**
- MEASURED. CPU, zero training, pod, launched about 16:01Z, probe stage 899 s.
- Tier: T0 decodability of inputs and frozen features, never driving performance.
- Population: D1-HO, 5,736 windows / 497 held-out episodes.
- Estimator: paired episode-cluster bootstrap, B 2,000, seed 0. It answers "another draw of EPISODES" only.
- Bar: 0.0383.
- **F-EGO** is the 8-column ego state O's ego token reads at NOW: `v`, `a`, `k`, `v^2`, `v*a`, `v*k`, `v^2*k`, `a*k`. These are speed, acceleration and curvature terms, from `p10_prior.kin_features`.

## Outcome as committed (tree probe decides): CEIL-AT-PLAN at both N1D seeds

| tree accuracy | F-EGO | F-TEXT | **F-TE** (text + ego) | F-PLAN | **F-ALL** (text + ego + plan) |
|---|---|---|---|---|---|
| N1D_s0 | 0.5579 | 0.6044 | **0.6855** | 0.7028 | **0.7306** [0.713, 0.748] |
| N1D_s1 | 0.5579 | 0.6044 | **0.6855** | 0.6937 | **0.7287** [0.711, 0.746] |

| contrast (tree) | N1D_s0 | N1D_s1 |
|---|---|---|
| **F-TE - F-PLAN** (decides) | **-0.0173 [-0.0361, +0.0029]**, not separated | **-0.0082 [-0.0278, +0.0109]**, not separated |
| F-TE - F-TEXT (what the ego state adds to the route) | +0.0811 [+0.0629, +0.0989] | same data |
| F-ALL - F-PLAN (what route and ego add to the plan feature) | +0.0279 [+0.0188, +0.0375] | +0.0350 [+0.0252, +0.0450] |
| F-ALL - F-TE | +0.0452 [+0.0278, +0.0624] | +0.0432 [+0.0273, +0.0583] |

**What it says**
- The 6 s roundabout decision is only partly determined at NOW:
  - the route text alone gives 0.60;
  - adding the ego's speed, acceleration and curvature gives 0.69;
  - O's plan feature gives 0.70 / 0.69, the same;
  - all three together give 0.73.
- This is a LOWER bound on determinability (card section 3). Vision beyond O's features is not in F-TE, and the probe is one tree probe.
- The ego state carries real decision information (+0.08 over the text). Plausibly that is timing: how much of the turn falls inside 6 s depends on the speed now. That reading is HYPOTHESIS.
- O's plan feature misses a little of route plus ego (+0.03 when they are added).
- **The logistic probe, beside, also reads CEIL-AT-PLAN.** Its F-TE is far lower (0.5746): route plus ego is not linearly separable for this label. A linear negative is not a negative about learnability.

## Controls: all card controls read
| control | value | status |
|---|---|---|
| K-CONST | 0.338215 | PASS |
| K-PARITY | F-TEXT 0.6044 and F-PLAN 0.7028 / 0.6937 reproduce the banked values; correct-vectors identical | PASS |
| K-EGO-DET | F-EGO bit-identical across both checkpoints and both extractions | PASS |
| Identity arm (binding): ego columns constant | 0 of 5,736 windows differ from F-TEXT's predictions | PASS |
| Regression arm (binding): text columns zeroed | -0.1276 [-0.1493, -0.1074] below F-TE; predictions identical to F-EGO's | PASS |

## What it means, and the next lever (RULE ZERO)
- **Committed consequence (CEIL-AT-PLAN):** the code-to-decoder card states its target as F-ALL, **0.73**, the best measured determinability at NOW from these inputs. Its start is the decoded tick-0 plan, 0.63.
  - The room is about +0.10 on D1-HO TAC-LAT, about 2.6 bars.
  - The rolled track (T1) reads 0.62 / 0.65.
- The chain of reads now places the loss at one step:
  - text 0.60;
  - plus ego 0.69;
  - T's code 0.67 / 0.68;
  - O's plan feature 0.70 / 0.69;
  - all inputs 0.73;
  - **decoded plan 0.63**.
  The decoder is the step that loses the decision; the inputs and the hierarchy carry it.
- **Not indicated:** an input-reading lever (CEIL-ABOVE-PLAN not held). A T-side vocabulary lever was not indicated either, by TCODE.
- N1DW, the first decoder-side lever, is training now on the pod (started 16:26Z). The code-to-decoder card is drafted next with this target; it needs a PI go.
