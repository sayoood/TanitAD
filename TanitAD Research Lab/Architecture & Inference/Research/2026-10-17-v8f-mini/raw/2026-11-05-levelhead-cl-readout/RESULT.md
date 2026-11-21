# CL head read-out check: RESULT (PREREG_CL_READOUT_A1.md, sha256 1f96d833a9c5e32eb7f845d669d2f825485daee5d3b8fcd29bbec45a2523a8bb, matches FROZEN_SHA.txt)

**Committed labels, both seeds identical (no split): `PARTIAL` (tracking) and `O-INSENSITIVE` (sensitivity).
The addendum's branch for PARTIAL is the lead-aware head (lead presence and distance from the frame, CRAWL-LEAD), its own card before any run.**

Evidence class MEASURED, ours; tier **T0** (the tick-0 batch with logged history and frames; a head and world-model diagnostic, not a driving result). This is not a driving eval,
so the four-family table is not engaged. Zero training, dev-box CPU, 12 threads, batch 32. Raw: `raw/CL_READOUT.json` (per seed, merged from `raw/CL_READOUT_s{0,1}.json`); code: `code/cl_readout.py`.
Checkpoints `M2b_s{0,1}_recft_CL/ckpt_last.pt` sha256 verified against SHA256SUMS.txt (`b61d18d5...` / `0a5b427d...`); the checkpoint cfg says `level_head on`, `w_level 0.2027`.
Estimator: per-window values, episode-cluster bootstrap B = 2000, seed 0 (taniteval.ci's draw stream), 141 episodes, 2,399 windows per seed. It answers "would another draw of episodes say this";
it does not answer a training-run or inference-run question. The two checkpoints are the two seeds and agree on both labels.

## 1. Tracking: v_hat against the logged speed at the tick

| | s0 | s1 | rule |
|---|---|---|---|
| R2, all 2,399 windows | 0.780 [0.685, 0.845] | 0.780 [0.692, 0.842] | TRACKS needs >= 0.5; DOES-NOT-TRACK below 0.3 |
| MAE in the 3..8 m/s band (n = 586) | **1.898** [1.621, 2.202] | **1.829** [1.605, 2.072] | TRACKS needs <= 1.5; DOES-NOT-TRACK above 2.5 |
| MAE on M1 crawl windows of the C0 rollout (n = 330 / 326, 62 / 60 episodes) | 2.075 [1.725, 2.462] | 1.867 [1.630, 2.129] | not in the rule |
| label | **PARTIAL** | **PARTIAL** | |

* R2 clears its TRACKS bar by a wide margin; the band MAE sits between the two bars. The lower CI end of the band MAE (1.62 / 1.61) is above 1.5 at both seeds, the upper end (2.20 / 2.07) is below 2.5,
  so PARTIAL holds on the point estimate and on the interval.
* The band error is a one-sided over-estimate: mean v_hat minus logged speed in the band is +1.01 (s0) and +0.65 (s1) m/s (mean logged 5.68, mean v_hat 6.69 / 6.33): consistent with a pull toward the corpus mean at low speed (not tested here). Beside, all windows: bias -0.05 / -0.33 m/s, MAE 2.58 / 2.58 m/s; the 2,105 / 2,106 GT-fast windows read MAE 2.57 / 2.61.
* **Target.** The logged speed is the cache `v` at the tick; it equals `kin3[t0, 0]` exactly (max abs difference 0.0, the head's own training target), so one target, not two. `raw` carries the kin3 re-read; the label is the same.
* **M1 crawl windows (my definition, stated because the addendum does not fix it).** M1 per window is `rr_lib.m1_window`: the share of GT-fast pairs (GT pair speed >= 5.2 m/s) whose own speed is < 5.2 m/s (`dagreread/code/rr_lib.py:91`);
  the per-window C0 value is `a_18` of `dagland/recover_next/read/raw/arrays/C0_s{s}.npz` (sha256 in the JSON; INHERITED as the C0 rollout's M1 column, windows aligned by (clip_index, ws), v0 matches kin3 with max difference 0.0).
  A crawl window = `a_18 > 0`. Beside: the heavy-crawl set (`a_18 >= 0.5`; n = 133 / 155) reads MAE 1.875 [1.467, 2.325] / 1.694 [1.412, 2.064]; the whole GT-fast set (n = 2,105 / 2,106) reads 2.572 / 2.613. The C0 dumps under `rr_scratch` were not needed.

## 2. O's sensitivity to the level it is handed

| | s0 | s1 | rule |
|---|---|---|---|
| mean change of the plan's 0..3 s mean speed, v_hat + 3 m/s (m/s) | -0.00043 [-0.00047, -0.00039] | +0.00051 [+0.00045, +0.00056] | O-SENSITIVE needs |change| >= 0.5; O-INSENSITIVE below 0.1 |
| the same, v_hat - 3 m/s | +0.00043 [+0.00039, +0.00047] | -0.00051 [-0.00056, -0.00046] | |
| mean of per-window |change|, +3 m/s | 0.00043 [0.00039, 0.00047] | 0.00053 [0.00047, 0.00058] | the other reading of "mean |plan-speed change|": same label |
| ||e_level output|| / ||O's ego token||, mean over windows | 0.0037 [0.0034, 0.0040] (norms 0.022 / 7.32) | 0.0048 [0.0045, 0.0052] (norms 0.029 / 7.40) | |
| label | **O-INSENSITIVE** | **O-INSENSITIVE** | |

* The change is 3 orders of magnitude below the 0.1 m/s floor and the 95th percentile of the per-window |change| is 0.0012 / 0.0015 m/s (max 0.0031 / 0.0041). The mean plan speed over 0..3 s is 11.35 / 11.37 m/s over these windows.
* **Controls (all read their known values).** A shift of 0.0 changes the plan by EXACTLY 0.0 (max abs trajectory difference 0.0 m); the value e_level receives equals `v_hat/10` (max deviation 0.0) and, with the shift, `v_hat/10 +- 0.3` (max deviation 6e-8);
  the forward hook's v_hat equals the model's `level_v` output (0.0); the T0 tick-0 plan equals the banked CL rollout's tick-0 `plan_raw` for all 2,399 windows (max difference 3.1e-5 m s0, 1.5e-5 m s1, same (ci, t0) rows), so the batch is the rollout's tick-0 batch;
  the label function reads TRACKS for a perfect head, DOES-NOT-TRACK for a constant and a shuffled head, and the three sensitivity labels at their thresholds.
* **Reach control (instrument only, not a card measurement).** Adding a fixed random direction of norm 8 (the size of an ego token) to e_level's output moves the plan's 0..3 s mean speed by 0.118 (s0) / 0.099 (s1) m/s on average (max 0.81 / 1.06): the harness can read a change at that site,
  and O answers it weakly even at that size. Beside: replacing e_level's output by zeros at inference changes the plan speed by 0.0016 [0.0015, 0.0018] / 0.0019 [0.0016, 0.0021] m/s.
* e_level weights are not zero: abs sum 0.218 (s0) / 0.294 (s1) for the weight, 0.079 / 0.092 for the bias (0.297 / 0.386 together, as `CL_PRECHECK.json` reads): the lever exists and O's decision does not use it.
* Code fact for reading this: O's plan is `upsample_prior(P) + residual` with the residual bounded by `res_scale_m` (`v8f_mini.py` `run_o`), P computed from the measured ego history; a 0..3 s speed change has to come through that residual. Whether O's residual could carry a level at all was not measured here.

## 3. The committed branch

| tracking | sensitivity | addendum row | next arm |
|---|---|---|---|
| PARTIAL (both seeds) | O-INSENSITIVE (both seeds) | "DOES-NOT-TRACK or PARTIAL" | **lead-aware head** (lead presence and distance from the frame; CRAWL-LEAD: half the crawl sits behind a lead), its own card before any run |

The CLR rows of the addendum (TRACKS with O-INSENSITIVE / WEAK or O-SENSITIVE) are not taken: tracking did not reach TRACKS at either seed (band MAE 1.83-1.90 against the 1.5 bar).
No eval-141 number here trains anything, as the addendum commits.

**For the card that follows (HYPOTHESIS, flagged for the Master Mind; it does not change the committed branch).** O-INSENSITIVE was measured at both seeds independent of the tracking label: the level channel as built (`e_level` added to O's ego token, zero-initialised)
reaches O's plan at about 0.0005 m/s per 3 m/s. A lead-aware head that enters O the same way, in a model trained the same way, would meet the same ignoring. The A2 card for the lead-aware head should name how its output enters O and
what makes O use it (the CLR-style exposure to states where the history and the head disagree is the addendum's own candidate for that), and it should carry this reach control.

## Not done / UNVERIFIED
* The M1 crawl-window set is my reading of "an M1 crawl window of the C0 rollout" (`a_18 > 0`); three variants are reported and the labels do not depend on it (M1 MAE is outside the committed rules).
* v_hat's band error was read at tick 0 only; later ticks of the T1-kin rollout (where history is the model's own) were not read here.
* Thread count 12 against the rollout's 16: tick-0 plans agree to 3e-5 m (control above); no effect on any quoted figure.
