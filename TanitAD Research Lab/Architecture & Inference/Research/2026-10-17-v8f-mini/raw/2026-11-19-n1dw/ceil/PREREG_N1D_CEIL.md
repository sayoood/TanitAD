# PREREG_N1D_CEIL (FROZEN 2026-11-19 by the MM, before any number of this read exists): how much of the 6 s roundabout decision do the route and the ego state determine at NOW?

**Status:** frozen by the Master Mind (MM) before any ego-state feature is extracted or fitted.

**Seen before freezing:** the PROBE-N1D-FEAT, T0T1 and TCODE results. No probe on the ego state has been fitted.

**Parent:** `../tcode/MM_READING_N1D_TCODE.md`, outcome CODE-CARRIES. It named this test as a HYPOTHESIS:
* the route text is built from the ego's own future path, yet it reads only 0.6044;
* so the 6 s kinematic label may not be determined at NOW, and the ceiling the next O-side lever aims at is unmeasured.

The coordinator asked that this ceiling be measured before the code-to-decoder card is written, so that card can state it.

**Evidence class:** MEASURED once run. CPU, zero training. It measures what INPUTS determine, never driving performance (tier T0 decodability).

## 1. Data and features
* **Windows, label and folds:** the D1-HO grid as in PREREG_N1D_TCODE, 5,736 windows / 497 episodes. Same 3-class kinematic label, majority share 0.338215, and the same episode-grouped folds.
* **F-EGO** [N, 8]: the ego state O's ego token reads at NOW, `p10f` at the NOW index (`v8f_mini.py` `run_o`: `self._ego(self._at(b["p10f"], i))`), before standardisation. Its column meanings are printed from source. It is an inference input (measured ego motion), the same for both N1D arms.
* **F-TE** = F-TEXT ⊕ F-EGO (23 columns).
* **F-ALL** = F-TEXT ⊕ F-EGO ⊕ F-PLAN, per N1D seed.
* **F-TEXT and F-PLAN:** the banked TCODE features (`n1dw_tcode/n1dwtcode_feats.tar`, sha256 c93c25b4).

## 2. Probe and estimator
* **Probe:** the PROBE-N1D-FEAT tree probe, unchanged. It decides. The logistic probe runs beside.
* **Estimator:** paired episode-cluster bootstrap, B 2,000, seed 0, over the 497 episodes. It answers "another draw of EPISODES" only.
* **Bar:** 0.0383.

## 3. Outcomes, committed (each must hold at both N1D seeds; otherwise it is printed per seed with "one-seed")
| outcome | condition | consequence |
|---|---|---|
| **CEIL-AT-PLAN** | F-TE - F-PLAN not separated above 0 by at least 0.0383 | The route and the ego state determine no more of the decision than O's plan feature already holds. The code-to-decoder card states its target as the F-ALL accuracy (the best measured determinability at NOW from these inputs), with the decoded plan's 0.63 as the start |
| **CEIL-ABOVE-PLAN** | F-TE - F-PLAN separated above 0 by at least 0.0383 | The route and the ego state hold decision information that O's plan feature lacks. A lever that makes T and O read those inputs ranks beside the decoder path, and both are carded |
| **NEITHER** | any other result | printed with every number |

**Printed beside:**
* F-EGO alone;
* F-TE - F-TEXT (what the ego state adds to the route);
* F-ALL - F-PLAN (what the route and the ego state add to the plan feature).

**Scope, stated now:**
* This is a LOWER bound on how far the label is determined at NOW. Vision beyond what O's features hold is not in F-TE.
* The bound is for this tree probe on these inputs.

## 4. Controls (must read)
* **K-CONST:** the majority predictor reads 0.338215.
* **K-PARITY:** F-TEXT and F-PLAN are taken from the banked tar unchanged. F-TEXT's tree accuracy reproduces 0.6044 and F-PLAN's reproduces 0.7028 / 0.6937.
* **K-EGO-DET:** F-EGO extracted twice is bit-identical, and identical between the two N1D checkpoints. It is data, not a model output.
* **Identity arm (binding):** F-TE with the 8 ego columns replaced by a constant gives out-of-fold predictions identical to F-TEXT's, on every window. A tree cannot split on a constant column, so any difference is a pipeline defect.
* **Regression arm (binding):** F-TE with the 15 text columns zeroed (F-EGO alone plus constants) reads separated below F-TE, unless F-TE does not clear F-EGO. In that case it is printed as such.
