# PREREG_N1D_TCODE (FROZEN 2026-11-19 by the MM, before any number of this read exists): does N1D's tactical level hand O the roundabout exit decision?

**Status:** frozen by the Master Mind (MM) before any feature of this read is extracted.

**Seen before freezing:**
* the PROBE-N1D-FEAT result (`../probe/MM_READING_N1DWPROBE.md`);
* the T0-against-T1 result (`../t0t1/MM_READING_N1D_T0T1.md`);
* the model source at 134f5da1.

Nothing about T's code or T's trunk on any window has been seen.

**Parents:**
* PROBE-N1D-FEAT found PROBE-PRESENT-UNUSED: the roundabout decision is decodable from the input of O's plan head at NOW, at 0.7028 / 0.6937 against 0.6044 for the route text alone.
* PREREG_N1D_T0T1 found DECODE-LOSES: the tick-0 plan already reads 0.6266 / 0.6349.
* That reading names one next decoder-side lever after N1DW: an auxiliary route-decision head on O's plan feature.

**Why this read comes first.** The source shows a second path that reading did not weigh.
* O already reads a decision from T, through T's code (`v8f_mini.py` `run_o`: `self.e_code(code)`). The code has CODE_DIM = 60 columns:
  * the goal probabilities;
  * the softmax of T's 5 lateral classes;
  * the softmax of its 7 longitudinal classes;
  * 26 constraints, among them `turn_dyaw_deg` and `turn_exit_x_m / _y_m / _psi_deg`.
* So there are three different places the decision could be missing:
  * from T altogether;
  * from T's code only;
  * from neither, with O's decoder failing to use it.
* Each one points to a different lever. This read tells them apart at zero training.

**Evidence class:** MEASURED once run. CPU, zero training, frozen checkpoints. Tier: T0 decodability of frozen features, never driving performance.

## 1. Data
* **Windows.** The D1-HO grid, 5,736 windows / 497 held-out episodes (`eval_grid_x_feasible.json`, sha256 ca289d0f). These are the PROBE-N1D-FEAT windows, built the same way.
* **Arms.** The banked N1D_s0 and N1D_s1 checkpoints, as PROBE-N1D-FEAT used them.
* **Label.** The same 3-class kinematic factor of the GT 6 s track (`read_otband.labels`). Class counts 1,934 / 1,862 / 1,940; majority share 0.338215.

## 2. Features (NOW tick, `forward_infer`, CPU, fp32, the PROBE-N1D-FEAT extractor plus two captures)
| set | tensor | where |
|---|---|---|
| **F-CODE** [N, 60] | the code O actually read at NOW | `out["code"]` of `forward_infer` (`v8f_mini.py`: `out["code"] = code`, after every override; no override is active at eval) |
| **F-TTRUNK** [N, d] | the input of T's heads | `h = gelu(trunk_t(read_t(...)))`, captured by a forward-pre-hook on `h_lat` |
| F-TEXT, F-PLAN | as in PROBE-N1D-FEAT | re-extracted; must equal the banked features (K-PARITY) |

* Beside only, never deciding: **F-CODE-LAT** is the 5 lateral-softmax columns plus the 9 turn constraints (`turn_t_start_s` through `turn_exit_psi_deg`, `CONS_NAMES` order).
* The extractor passes no `code_override` and asserts that the model's code-mix and teacher-forcing paths are inactive in `forward_infer`.

## 3. Probe and estimator
* **Probe.** The PROBE-N1D-FEAT tree probe, unchanged: same settings and same episode-grouped folds. It decides. The logistic probe runs beside.
* **Estimator.** Paired episode-cluster bootstrap, B 2,000, seed 0, over the 497 episodes, on per-window out-of-fold correctness. It answers "another draw of EPISODES"; training variance is carried by the two N1D seeds.
* **Bar 0.0383.**

## 4. Outcomes, committed (each must hold at both N1D seeds; otherwise it is printed per seed with "one-seed")
| outcome | condition | next lever (each needs its own card and a PI go) |
|---|---|---|
| **CODE-CARRIES** | F-CODE - F-TEXT separated above 0 by at least 0.0383 | T hands O the decision, and O's decoder does not turn it into the track. The lever is O-side: N1DW (in build); beyond it, a direct code-to-decoder path (the code conditioning `head_o`, zero-init) |
| **T-HAS-CODE-LACKS** | F-TTRUNK - F-TEXT separated above 0 by at least 0.0383, and F-CODE - F-TEXT not | T forms the decision but its vocabulary does not pass it on. The lever is tactical: a roundabout-exit head in T whose softmax is appended to the code (the PI's tactical band, up to 6 s plus vocabulary) |
| **T-LACKS** | F-TTRUNK - F-TEXT not separated above 0 by 0.0383 | the decision forms only in O. The lever is the T0T1 reading's: an auxiliary route-decision head on O's plan feature |
| **NEITHER** | any other result | printed with every number; the lever is chosen with the coordinator |

* Printed beside: F-PLAN - F-CODE and F-PLAN - F-TTRUNK.

## 5. Controls (must read)
* **K-CONST:** the majority predictor reads 0.338215.
* **K-DET:** each checkpoint's features re-extracted twice are bit-identical.
* **K-PARITY:** the re-extracted F-TEXT and F-PLAN equal the banked PROBE-N1D-FEAT features (`n1dwprobe_feats.tar`, sha256 4a282ace) to 0.0.
* **Regression arm (binding):** F-CODE with all 60 columns zeroed must read separated below F-CODE, at both seeds. If F-CODE does not clear F-TEXT, the arm reads against F-TEXT instead.
* **SHUF (beside):** the PROBE-N1D-FEAT permutation, read against its own null where the SHUF-null run gives one. It moves no outcome. (PROBE-N1D-FEAT's committed SHUF band failed, as reported there.)

## 6. Admissibility
* T's code and T's trunk are outputs of the tactical level of the driving model. They are not the situation classifier, and they read only `INFERENCE_INPUT_KEYS` (`assert_no_situation_classifier_input`).
* Nav is an input at training and inference.
* No label enters any feature.
