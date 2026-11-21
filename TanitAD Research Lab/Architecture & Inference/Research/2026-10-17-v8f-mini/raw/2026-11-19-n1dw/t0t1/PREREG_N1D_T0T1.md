# PREREG_N1D_T0T1 (FROZEN 2026-11-19 by the MM, before any number of this read exists): is N1D's roundabout decision lost in decoding or over the rollout?

**Status:** frozen by the Master Mind (MM) before any T0-against-T1 number is computed.

**Parent:** `PREREG_V8F_N1DW.md` section 10. Its committed lever after PROBE-PRESENT-UNUSED is "a T0-against-T1 read of N1D on D1-HO (zero training) first". It runs now, because it is zero training and the coordinator asked that R5 not wait. Its result changes no N1DW bar or outcome.

**Evidence class:** MEASURED once run. CPU, zero training, read from banked dumps.

## 1. Question
PROBE-N1D-FEAT found the roundabout decision decodable from O's plan-head input at NOW: tree 0.7028 (s0) and 0.6937 (s1), against 0.6044 for the text alone. N1D's realised T1-kin TAC-LAT on the same windows is about 0.62 and 0.65. Is the decision already missing from the 6 s plan O decodes at t0 (DECODE), or is it in that plan and lost over the T1 rollout (ROLLOUT)?

## 2. Data and quantities
**Windows.** The D1-HO grid, 5,736 windows / 497 held-out episodes (`eval_grid_x_feasible.json`, sha256 ca289d0f).

**Arms.** N1D_s0 and N1D_s1, with N1R_s0 and N1R_s1 beside. All are read from the banked T1-kin dumps of the NAVRBD read (pod `/root/strat1/navrbd/out/t1/`, banked as `navrbd_read/navrbd_read_outputs.tar`, sha256 143f3b2a). No new dump is made.

**Per window:**
* **T1 TAC-LAT:** the kinematic factor (`read_otband.labels`) of the rolled 6 s track (m = 10, causal bucket) against the same factor of the GT track. This is exactly the NAVRBD D1-HO indicator.
* **T0 TAC-LAT:** the same factor of the **tick-0 open-loop plan**, O's full 30-knot plan at t0 from the logged history, against GT. It is the plan the dump stores as its open-loop tick-0 output. The PROBE build already matched it to its own tick-0 plan to 0.0 m.

**Estimator.** Paired episode-cluster bootstrap, B 2,000, seed 0, over the 497 episodes, on the per-window difference of the two indicators. Both come from the same checkpoint on the same windows. It answers "another draw of EPISODES"; training variance is carried by the two seeds.

**Bar 0.0383** (the rig floor of the roundabout cells).

## 3. Outcomes, committed (each must hold at both N1D seeds; otherwise it is printed per seed with "one-seed")
| outcome | condition | next lever |
|---|---|---|
| **ROLLOUT-LOSES** | T0 - T1 TAC-LAT separated above 0 by at least 0.0383 | the plan decoded at t0 carries the decision and the rollout loses it. The lever is rollout-side: the RB window weight applied to the DAgger (own-rollout) rows, or a re-plan consistency term. Each needs its own card and a PI go |
| **DECODE-LOSES** | T0 - T1 not above 0 by the bar, and T0 TAC-LAT at least 0.0383 below the probe's F-PLAN tree accuracy (a point difference between two estimators, printed as such) | the information sits in the plan feature and the plan head does not turn it into the track. The lever is decoder-side: N1DW's weight already targets it. Beyond N1DW, an auxiliary route-decision head on O's plan feature (kinematic label; labels may use ego, inference vision plus route) that conditions the decoder. Own card, PI go |
| **NEITHER** | any other result | printed with both numbers; the next lever is chosen with the coordinator |

* N1R's T0 - T1 is printed beside, with N1D - N1R at T0 (does the channel act at T0 and fade over the rollout?).

## 4. Controls (must read)
* **K1:** the reader's self-pair (N1D_s0 T1 against itself) reads exactly 0.0.
* **K-RE:** the T1 TAC-LAT accuracies recomputed from the dumps equal the banked NAVRBD D1-HO values (N1R_s0 0.5905, N1R_s1 0.6328, and the N1D values in `RESULT_NAVRBD`) to 1e-9.
* **K-CONST:** the majority predictor reads 0.338215.
* **K-T0:** the tick-0 plan the reader uses equals the probe extractor's tick-0 plan (0.0 m, already measured on 5,736 / 5,736 windows).
