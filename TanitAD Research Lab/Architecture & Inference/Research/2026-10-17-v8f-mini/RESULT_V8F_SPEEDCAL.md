# SPEEDCAL_READ: PREREG_V8F_SPEEDCAL executed as written

**Outcome (the PREREG table applied verbatim): INFORMATION-LIMITED.**

Tier: T1-class open loop (own held outputs; no recorded future action). CEILING is stamped T0. Estimator: paired episode-cluster bootstrap (taniteval/ci.py), B=2000, 141 episodes / 2399 windows; answers 'another draw of EPISODES' only; BLIND to training variance (H-ESTIM-SEED-1): the second seed is the replicate; a separated CI at two seeds is necessary, not sufficient, for a lever claim.

Committed next step, quoted from the PREREG: *"T's speed output carries no affine-recoverable information about the right speed. The speed lever moves to perception (what T sees: temporal context, encoder adaptation). That needs compute beyond box 2, so a PI go"*

| read | seed 0 | seed 1 | bar |
|---|---|---|---|
| PRIMARY ADE6 SPEEDCAL - X_s | +0.0374 [-0.0100, +0.0864] | -0.0058 [-0.0156, +0.0036] | <= -0.1207 and separated at BOTH |
| CEILING (T0 diagnostic) ADE6 eval-GT map - X_s | +0.0296 [-0.0267, +0.0862] | -0.0075 [-0.0174, +0.0025] | beside; routes the two 'no' rows |
| CAPTURE | -0.052 | 0.009 | beside |
| SECONDARY not separated-worse | False | True | both |

## The outcome table (verbatim from the PREREG)

| outcome | condition | next (committed) | this read |
|---|---|---|---|
| **HEAD BIAS PAYS** | PRIMARY and SECONDARY at both seeds | The calibration joins `forward_infer`: 10 buffers per model, deployable at zero training. The HSPEC dominance row is re-read on the calibrated model against P10. Then a T-side debiasing arm, so one model needs no post-map, under its own PREREG |  |
| **PARTIAL BIAS** | PRIMARY < 0 and separated at both seeds, but not ≤ −F at both | Report CAPTURE. Next: a non-linear (isotonic, per-regime) calibration under its own PREREG |  |
| **MAP DOES NOT TRANSFER** | PRIMARY not separated at either seed, AND CEILING ≤ −F at both seeds | The error is affine-recoverable but the 141-clip fit does not carry to eval. Next: the same map fitted on all 4,512 training clips' windows |  |
| **INFORMATION-LIMITED** | PRIMARY not separated at either seed, AND CEILING > −F at either seed | T's speed output carries no affine-recoverable information about the right speed. The speed lever moves to perception (what T sees: temporal context, encoder adaptation). That needs compute beyond box 2, so a PI go | **YES** |
| **MIXED** | anything else | reported per seed; seeds before windows |  |

## Controls

PASS: True
* seed 0: control 1 True (max dev 0.0), control 2 True (max dev 2.13e-14), control 3 True (max dev 4.31e-06 m), control 4 True (perm - own +5.462 m), guards True
* seed 1: control 1 True (max dev 0.0), control 2 True (max dev 1.78e-14), control 3 True (max dev 1.46e-05 m), control 4 True (perm - own +5.871 m), guards True

## Per cell: the map (a, b) and the RMSE to the GT, m/s (beside; decides nothing)

| seed | cell | a | b | train n | train RMSE before -> after | eval RMSE before -> after |
|---|---|---|---|---|---|---|
| 0 | `lon_v_target_ms` | 1.0113 | -0.9550 | 9534 | 2.934 -> 2.814 | 2.894 -> 2.886 |
| 0 | `v_a_ms` | 1.0986 | +0.1327 | 9582 | 1.770 -> 1.268 | 1.919 -> 1.325 |
| 0 | `v_end_ms` | 0.9968 | -1.1218 | 9446 | 2.913 -> 2.672 | 2.795 -> 2.651 |
| 0 | `v_lo_ms` | 1.0808 | -0.8636 | 9582 | 2.110 -> 2.029 | 2.257 -> 2.163 |
| 0 | `v_hi_ms` | 1.0476 | -0.2458 | 9582 | 2.105 -> 2.060 | 2.093 -> 2.008 |
| 1 | `lon_v_target_ms` | 0.9894 | -0.0900 | 9534 | 2.712 -> 2.703 | 2.747 -> 2.749 |
| 1 | `v_a_ms` | 1.0651 | -0.6175 | 9582 | 1.284 -> 1.200 | 1.302 -> 1.249 |
| 1 | `v_end_ms` | 1.0184 | -0.7938 | 9446 | 2.634 -> 2.564 | 2.692 -> 2.590 |
| 1 | `v_lo_ms` | 1.0040 | +0.1804 | 9582 | 2.031 -> 2.019 | 2.143 -> 2.117 |
| 1 | `v_hi_ms` | 1.0166 | -0.1532 | 9582 | 1.859 -> 1.856 | 1.802 -> 1.802 |

## TACTICAL / STRATEGIC (`v8f_paired_families.py`, SPEEDCAL - X_s; beside, decides nothing)

* seed 0: tool controls_ok True; native reproduction [{'arm': 'A', 'n_checked': 119, 'n_failed': 0, 'ok': True}, {'arm': 'B', 'n_checked': 119, 'n_failed': 0, 'ok': True}]; INSTRUMENT GAP none; substitute control PASS True (max |delta| 8e-07); separated cells [{'cell': 'tactical.goal_setting.goal_long_bias_m', 'delta': 0.5231, 'lo': 0.3387, 'hi': 0.7068, 'reading': 'A higher'}, {'cell': 'tactical.goal_setting.goal_lat_bias_m', 'delta': -0.1154, 'lo': -0.1445, 'hi': -0.0874, 'reading': 'A lower'}, {'cell': 'tactical.goal_setting.goal_range_ratio', 'delta': 0.0108, 'lo': 0.0069, 'hi': 0.0144, 'reading': 'A higher'}] / []
* seed 1: tool controls_ok True; native reproduction [{'arm': 'A', 'n_checked': 119, 'n_failed': 0, 'ok': True}, {'arm': 'B', 'n_checked': 119, 'n_failed': 0, 'ok': True}]; INSTRUMENT GAP none; substitute control PASS True (max |delta| 7e-07); separated cells [{'cell': 'tactical.goal_setting.goal_point_error_m', 'delta': -0.0304, 'lo': -0.0568, 'hi': -0.0049, 'reading': 'A better'}, {'cell': 'tactical.goal_setting.goal_long_bias_m', 'delta': 0.2526, 'lo': 0.2174, 'hi': 0.2856, 'reading': 'A higher'}, {'cell': 'tactical.goal_setting.goal_lat_bias_m', 'delta': -0.0104, 'lo': -0.0185, 'hi': -0.0027, 'reading': 'A lower'}, {'cell': 'tactical.goal_setting.goal_range_ratio', 'delta': 0.0032, 'lo': 0.0023, 'hi': 0.004, 'reading': 'A higher'}] / []

Reading: PREREG_V8F_SPEEDCAL: INFORMATION-LIMITED; PRIMARY (cal - own, ADE6 m): s0 +0.0374 [-0.0100, +0.0864] ns, s1 -0.0058 [-0.0156, +0.0036] ns vs -F = -0.1207; CEILING (T0) s0 +0.0296, s1 -0.0075; CAPTURE s0 -0.05, s1 0.01; SECONDARY not separated-worse: s0 False, s1 True.


## MM independent re-read (2026-10-26, Master Mind; descriptive; it changes no row of the read)

A build agent produced the read above. Before it was quoted anywhere, the Master Mind re-read the two decision rows with code that shares nothing with the agent's reader (`raw/2026-10-25-speedcal/mm_verify/verify_sc.{py,log,json}`):
* its own ADE6, computed from the banked plans against HSPEC's eval GT, with the window order checked;
* its own episode-cluster bootstrap (numpy, B = 2,000, rng 20261026).

The 12 eval npz it reads are byte-identical to the banked copies (`cmp`, log header).

| check | s0 | s1 |
|---|---|---|
| my ADE6 from the plans vs the banked per-window ADE6 (own, cal, ceiling, identity), max abs | 0.0 | 0.0 |
| identity arm vs own, ADE6 max abs (control 1) | 0.0 | 0.0 |
| the lever ran: calibrated plans vs own, max abs (m) | 6.80 | 1.56 |
| the lever ran: windows whose T code moved | 100 % | 100 % |
| PRIMARY ADE6, SPEEDCAL − X_s | +0.0374 [−0.0120, +0.0895] | −0.0058 [−0.0154, +0.0033] |
| CEILING (T0), eval-GT map − X_s | +0.0296 [−0.0262, +0.0893] | −0.0075 [−0.0171, +0.0023] |

* **Every point estimate agrees exactly.** The intervals agree to the bootstrap draw.
* **The outcome row stands: INFORMATION-LIMITED.** PRIMARY is not separated at either seed, and CEILING is above −F = −0.1207 at both.

**What the result is, and what it is not:**
* **It is a clean elimination.** No affine map of T's own five speed cells, not even one fitted on the eval GT, moves ADE6 by F. Yet the GT speed cells (T0) cut ADE6 by 0.72 / 0.67 m (`READ.json` `ade6_m`). So speed is still the largest lever, and the error O responds to is per-window, not a global head bias.
* ⚠ **The tier is T1-class open loop** (CEILING and the `gt` arm are T0). One estimator question is answered: another draw of episodes. The second seed is the replicate.
* ⚠ **Disclosed by the build agent:** the fit-set GT is the trainer's `join_train.npz` (the same v9 md5 as all four trainings), not `v9_labels_train.npz`, which sat only in directories the task could not read.
* **Committed next, quoted from the PREREG:** the speed lever moves to perception (temporal context, encoder adaptation), and that needs compute beyond box 2. The arm for it already exists and is gated: `PREREG_V8F_LORA.md`, whose AR-2 smoke PASSED in run 3 and projects 470 h per seed on box 2, so its venue rule names the A40. The go is the PI's: `D-V8F-SPEED-PERCEPTION` in `PI_DECISION_QUEUE.md`.
