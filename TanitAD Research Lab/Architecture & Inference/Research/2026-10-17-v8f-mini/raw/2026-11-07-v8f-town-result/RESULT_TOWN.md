# RESULT_TOWN: does the tactical layer's speed decision follow the car's own speed on its own states?

Card `PREREG_V8F_TOWN.md`, frozen prefix 4,676 bytes, sha256 `707e29f3...` (holds); implementation choices `IMPL_CHOICES.md` sha256 `2c6fc876...`.
Tier: T1-kin for the own-state rows, T0 for the twin (logged-state) rows; the paired own-minus-logged difference is the card's statistic, no tier is pooled. Evidence class: MEASURED (ours; artifacts under town/raw).
Estimator: episode-cluster bootstrap, B 2,000, seed 0, over 141 episodes (taniteval/ci.py); slope refitted on every draw; paired for F1 and RMSE. n = 2,399 windows, 141 episodes, 3 replans per window (7,197 replans). C0 s0 / s1, 4 threads, CPU.

## Outcome: **MIXED**

| condition (card) | s0 | s1 |
|---|---|---|
| slope below 0 and separated (T-FEEDBACK) | False | False |
| slope not separated (T-STABLE) | True | True |
| T-lon macro-F1 own - logged separated-worse (T-FEEDBACK needs True, T-STABLE needs False) | True | True |

T-FEEDBACK: False. T-STABLE: False. 'at both seeds' applies to every condition of an outcome (each condition must hold at s0 AND s1).

MIXED here means: the slope is not separated from 0 at either seed, so T-FEEDBACK does not hold; T-lon is separated-worse on its own states at both seeds, so T-STABLE does not hold either. The card names no step for MIXED; the result names it (section Next).

## Controls (card section Controls, binding)

| control | s0 | s1 | what was read |
|---|---|---|---|
| 1 Identity (G-ID: dump run vs the bare pinned tool, 4 threads, full grid) | PASS | PASS | max abs delta of the executed tracks 0.0 / 0.0 (must be exactly 0.0); tick file and 141 episode dumps bitwise equal |
| 2 Twin identity at tick 0 (G-TWIN0; C0 and REG arms) | PASS | PASS | 2,399 tick-0 rows, swapped inputs and T outputs bitwise equal, max abs d at tick 0 = 0.0 under all three rank orders (must be exactly 0.0) |
| 2b Twin is the logged state (G-TWINL, G-OWN; supporting) | PASS | PASS | 4,798 of 4,798 later rows differ in the swapped inputs; logged p10f mismatch 0 |
| 3a Regression arm built as declared (G-REG) | PASS | PASS | T's ego token zeroed on 450 of 450 T calls (own + twin) at s0, 450 of 450 at s1; weights restored |
| 3b Regression arm slope not separated from 0 | PASS | PASS | REG slope +0.0041 [-0.0017, +0.0099] / -0.0081 [-0.0189, +0.0010] |
| integrity (not a numbered control) | PASS | | frozen prefix and implementation sha hold; 7,197 replans, 141 episodes, B 2,000, the pinned tool sha, 4 threads, in all four reads |

Control 1 is read against the 4-thread bare run of the pinned tool, as the card's Log (09:12, before any statistic) commits. Against the banked 12-thread runs (RIC0) the same comparison reads max abs delta 0.0815 m (s0) and 0.2570 m (s1), not 0.0; the Log declares that line informational. The card's sentence says 'the tool of record's' without saying which run; the Log fixes it, and it is applied as written.

### The identity line that read rc 5 at both seeds (informational, decides nothing)

| | s0 | s1 |
|---|---|---|
| banked run threads / chain threads | 12 / 4 | 12 / 4 |
| same checkpoint sha256 / same tool sha256 | True / True | True / True |
| episode dumps unequal | 141 of 141 | 141 of 141 |
| tick-file columns unequal | 16 (all float) | 16 (all float) |
| integer / bool columns unequal | 0 | 0 |
| per-episode max track delta: median / p90 / max (m) | 0.0008 / 0.0053 / 0.0815 | 0.0008 / 0.0045 / 0.2570 |
| own-state rows whose T-lon argmax differs from the banked tick file | 1 of 7,197 | 0 of 7,197 |
| T-lon logits max / mean abs difference | 0.0032 / 0.000022 | 0.0022 / 0.000023 |

Same checkpoint sha256 and same tool sha256; only the thread count differs (12 against 4). Float summation order changes the bits and the rollout drifts (per-episode max track delta in the table). The committed statistics are all paired within one run (own against logged twin), so they are statistics of the 4-thread rollout; the identity that bears on them is control 1, which reads 0.0. The own-state T outputs barely move between the two thread counts (argmax rows and logit differences above), which bounds how much the F1 and share rows could move; the twin side was run at 4 threads only, so whether the conclusions would hold on the 12-thread rollout is not tested here.

## The committed statistics, per seed (C0; own = the car on its own motion, logged = the same replan with the logged ego state)

| statistic | s0 | s1 | status s0 / s1 |
|---|---|---|---|
| FEEDBACK slope of d on dv (rank units per m/s; rows with abs dv >= 0.5) | -0.0009 [-0.0599, +0.0607] | -0.0078 [-0.0595, +0.0415] | not separated / not separated |
| rows / episodes in the slope | 2,729 / 139 | 2,745 / 138 | |
| T-lon macro-F1 own - logged (own, logged) | -0.0831 [-0.1155, -0.0396] (0.4031, 0.4862) | -0.0702 [-0.1055, -0.0287] (0.4023, 0.4725) | worse / worse |
| T-lat macro-F1 own - logged (own, logged) | -0.0828 [-0.1254, -0.0462] (0.6993, 0.7820) | -0.0938 [-0.1398, -0.0577] (0.6914, 0.7852) | worse / worse |
| speed-constraint RMSE own - logged, m/s (own, logged) | +0.963 [+0.761, +1.161] (3.688, 2.725) | +0.794 [+0.620, +0.993] (3.505, 2.711) | worse / worse |
| labelled replans in the quality rows (lon / lat) | 6,175 / 6,168 | 6,175 / 6,168 | |

Outcome logic uses the first row and the T-lon row only. The T-lat and constraint rows are reported as the card asks and decide nothing. F1 is higher-is-better (status worse = interval below 0); RMSE is lower-is-better (status worse = interval above 0).

## What the read says about T's own decisions on the model's own states

* T-lon's expected class index is not found to follow the car's speed deficit: the slope of the shift on the deficit is not separated from 0 at either seed (-0.0009 [-0.0599, +0.0607] and -0.0078 [-0.0595, +0.0415] rank units per m/s). The intervals allow slopes of about -0.06 to +0.06 rank units per m/s, i.e. at most about 0.06 of a class position per m/s of deficit. The mechanism in the form the card committed (a negative slope) is not found at this precision. Both signs of deficit are present: of the slope rows, 1,595 / 1,515 have the car slower than logged and 1,134 / 1,230 faster (s0 / s1).
* T does worse on the model's own states than on the logged states at both seeds, with the same camera frames and the same route in both calls (only the ego block is swapped): T-lon macro-F1 0.403 against 0.486 (s0) and 0.402 against 0.472 (s1); T-lat 0.699 against 0.782 and 0.691 against 0.785; speed-constraint RMSE 3.69 against 2.72 and 3.50 against 2.71 m/s.
* Beside, deciding nothing: the share of replans whose lon argmax is a slower class than the label's is 23.2 % on own states against 18.8 % on logged (s0) and 24.1 % against 20.7 % (s1).
* The two findings sit side by side: the degradation of T on its own states is measured at both seeds, and it is not carried by the linear speed-deficit channel the card tests.

## What it does not say

* It does not say T is the crawl source (T-FEEDBACK needs the slope) and it does not say T is cleared (T-STABLE needs T-lon F1 not worse on own states). The card's two clean outcomes are both unmet.
* One linear slope pooled over both signs of deficit and over all classes. A threshold, a class-specific or a nonlinear dependence on speed is not read.
* F1 and RMSE are scored against the label at the logged time (ego and future may be used for labels). On own states the logged driver's label is not necessarily the right target, so own - logged mixes degraded decisions with a target that fits the own state less well. It does not say which.
* The twin swaps the ego block only (`p10f, base, pose_now, pose_t, pose_s, limit, limit_ok`); frames, route countdown and hold ages are the same tensors. The measured drop is therefore a drop from the ego block, not from the images. HYPOTHESIS, untested here: T-lat loses about as much F1 as T-lon (T-lat -0.083 / -0.094, T-lon -0.083 / -0.070) and lat has no speed-deficit channel, so part of the drop may be a general own-state shift rather than a speed effect.
* No crawl share, along-track or any driving metric of these rollouts is read, and no T-to-O link is measured: this read stops at T's outputs.
* The estimator answers another draw of EPISODES. Training-run variance is read by the two seeds side by side (the outcome pattern is the same at both); the rollout is reproducible bit for bit at a fixed thread count (control 1: two separate processes, max abs delta 0.0), so there is no inference-seed variance to read, but its bits depend on the thread count (see the identity line above).
* STRATEGIC family: PAUSED by PI ruling (item 25). The S layer is not read here; this is not a pass.

## Beside (not committed, decides nothing)

| | s0 | s1 |
|---|---|---|
| slope, through origin | -0.0036 [-0.0622, +0.0576] (not separated) | -0.0097 [-0.0598, +0.0390] (not separated) |
| slope, deficit read at T's frame (j - age) | +0.0154 [-0.0496, +0.0750] (not separated) | -0.0004 [-0.0571, +0.0546] (not separated) |
| slope, FOLLOW / KEEP swapped | -0.0067 [-0.0691, +0.0583] (not separated) | -0.0279 [-0.0818, +0.0275] (not separated) |
| slope, three-group order (slow / middle / fast) | +0.0455 [+0.0210, +0.0716] (separated above 0) | +0.0374 [+0.0147, +0.0616] (separated above 0) |
| T-lon F1 own - logged, ticks 1-2 only | -0.1295 [-0.1788, -0.0657] | -0.1214 [-0.1791, -0.0505] |
| T-lat F1 own - logged, ticks 1-2 only | -0.1302 [-0.1928, -0.0730] | -0.1487 [-0.2168, -0.0903] |
| share lon argmax slower than label: own / logged, delta [CI] | 23.2 % / 18.8 %, +0.0437 [+0.0244, +0.0661] | 24.1 % / 20.7 %, +0.0342 [+0.0178, +0.0520] |
| constraint RMSE, 26 cells: worse / better / n.s. | 19 / 1 / 6 | 22 / 0 / 4 |
| mean deficit dv (m/s); share of replans with abs dv >= 0.5 | +0.087; 0.379 | +0.059; 0.381 |
| mean shift d over all replans (rank units) | -0.0319 | -0.0217 |

The three-group variant of the order reads separated ABOVE 0 at both seeds, the opposite sign to the hypothesis. It is one of three orderings the card lists as beside; it was not committed, the committed ordering reads not separated, and nothing is promoted from it.

### The regression arm (T's ego token zeroed inside T), beside the C0 rows

| | s0 | s1 |
|---|---|---|
| REG slope (committed form) [CI]; half-width | +0.0041 [-0.0017, +0.0099]; 0.0058 | -0.0081 [-0.0189, +0.0010]; 0.0100 |
| C0 slope half-width | 0.0603 | 0.0505 |
| REG slope through origin [CI] | -0.0017 [-0.0058, +0.0015] | -0.0106 [-0.0200, -0.0033] |
| T-lon F1 on logged states: REG against C0 | 0.260 against 0.486 | 0.244 against 0.472 |
| T-lon F1 own - logged, REG | -0.0013 [-0.0042, +0.0017] (n.s.) | +0.0011 [-0.0024, +0.0048] (n.s.) |
| mean deficit dv (m/s): REG against C0 | +1.217 against +0.087 | +0.879 against +0.059 |
| mean deficit dv at tick 2 (m/s): REG against C0 | +2.147 against +0.086 | +1.592 against +0.029 |

Control 3b passes with the committed with-intercept slope at both seeds; at s1 the upper bound is +0.0010, a narrow margin, and the through-origin form (beside) is separated below 0 at s1 and not separated at s0. The control as committed is the with-intercept slope. The REG interval is about 10x / 5x narrower than the C0 interval, so the probe resolves slopes of about 0.01 when T has no ego token; the C0 intervals (0.06 / 0.05) are the resolution that applies to the C0 result.

Observed beside, not pre-registered and not a result: with T's ego token zeroed the rollouts run slower than the logged state (mean deficit +1.22 / +0.88 m/s against +0.09 / +0.06 in C0) and T-lon F1 on logged states falls to 0.260 / 0.244 from 0.486 / 0.472. A path from T to the car's speed exists in this model. The arm also leaves ego routes into T open (S's checkpoints, the speed-limit bucket, the route countdown), as IMPL_CHOICES.md section 5 discloses.

## Next (Rule Zero: the read does not end at its bar)

Named by: this result (the card leaves MIXED to the result); not frozen, each step needs its own card written before data by the Master Mind.

MIXED. The card commits a next lever for T-FEEDBACK and for T-STABLE and leaves MIXED to the result; neither clean outcome fired, so both card steps stay open and are ordered by the measured effect.

1. Zero-training read at T1-kin: O fed T's LOGGED-state code (the card's interim read under T-FEEDBACK). It measures the share of the crawl that T's own-state code carries WITHOUT needing the slope; T's own-state quality is separated-worse at both seeds (T-lon macro-F1 -0.083 / -0.070, speed-constraint RMSE +0.79 to +0.96 m/s), so a T-side error exists whose crawl cost is not yet measured.

2. The twin read on O (own plan against O fed the logged ego history), the card's T-STABLE step. It is independent of step 1 and can run beside it; this result does not clear T, so it does not displace it.

3. Only if step 1 removes crawl excess: T trained on its own states (DT2 / RECOVER recipe on T's labels), under its own card.


## Families

* TACTICAL: read (T-lat, T-lon: shift d, macro-F1 own minus logged, speed-constraint RMSE)
* LONGITUDINAL / LATERAL driving metrics: not read by this card (mechanism read, not a driving claim)
* STRATEGIC: PAUSED (PI ruling, PI_DECISION_QUEUE item 25); not read here, decides nothing, not a pass

## Artifacts

* Inputs (sha256 recorded in `RESULT_TOWN.json`): `raw/READ_{C0,REG}_s{0,1}.json`, `raw/GATE_ID_C0_s{0,1}.json`, `raw/GATE_TWIN_{C0,REG}_s{0,1}.json`, `raw/GATE_ID_INFO_C0_s{0,1}_vs_banked_RIC0_s{0,1}.json`, `raw/receipts/` (copies of the run receipts and done markers), `raw/INFO_DRIFT_vs_banked.json`, `raw/INFO_DV_SIGNS.json`.
* Rollouts (not copied): `F:/TanitAD-scratch/motion_c/town/out/{BARE,C0,REG}_s{0,1}`. Banked runs compared: `F:/TanitAD-scratch/motion_c/ri/out/RIC0_s{0,1}`.
* Generated by `town_result.py`; the deliberate-regression tests are `tests/test_town_result.py`.

