# read_K2DC02s1 -- the K2DC02s1 EMA read (refcv9 E2), banked by MM 4a3d05 on 2026-11-22

K2DC02s1 = K2DC02's argv at training seed 1 (the PREREG_K2DC02 sec. 4.4 replicate; PI go 2026-11-20). One training run, 6,000 steps at
1f14f8d81657 (Thor runs/r9e2-K2DC02s1). Pre-registration: `../e2raw/PREREG_K2DC02S1_2026-11-20.md`. This is the **EMA** read (model_ema.pt);
the **TAIL** read (the E3-representative weights, PREREG_K2DC02S1 sec. 5: the tail outcome decides the dose reading) runs separately on Thor
(E2TAIL_K2DC02s1) and is banked on its own.

## E2 verdict (`read/E2_VERDICT.md`): **NO-GO** (rule unchanged; DEV windows, 70 episodes; paired episode-cluster bootstrap)

| clause | tier | K2DC02s1 | parent K2DC02 (read_K2DC02) |
|---|---|---|---|
| (a) map primary nf_swedge_prec_lv_5_20, K2 > K0 | T0 | **FAIL** 0.4602 vs K0 0.7505 (s0 = s1) | FAIL 0.4066 |
| (b) box AP@2 m not separated worse | T0 | PASS +0.005 [-0.0002, +0.0101] | PASS +0.0070 [+0.0015, +0.0127] |
| (c) S6c sidewalk rate, K2 < K0 | T1 | **FAIL** s0 0.1938 vs 0.1883, s1 0.1932 vs 0.1852 | FAIL s0 0.1889 vs 0.1883, s1 0.1907 vs 0.1852 |
| (d) no four-family row separated worse | T1 | **FAIL at s0 only**: LAT_heading_mae_deg +0.0315 [+0.0008, +0.0632], LAT_cross_mae_m +0.0039 [+0.0010, +0.0068], LAT_cross_abs_final_m +0.0090 [+0.0027, +0.0159]; s1 none | FAIL: LON_accel s0 + s1, LAT_curvature s0 |
| (e) W4 preview vs camera-blind | T1 | PASS | PASS |

* LONGITUDINAL: no row separated worse at either seed; LON_dk_min_ttc_s +0.239 [+0.061, +0.461] separated BETTER at s0 only.
* LATERAL: the three (d) rows above at s0; none at s1.
* TACTICAL: no row separated at either seed (lat / lon / man5 correct within +-0.0054).
* STRATEGIC (report-only, `strategic/`): the reader still stamps PAUSED, which is stale (the PI lifted the pause on 2026-11-16). Computed beside
  with the landed `k2dc02s1tail_strategic_navcomp.py` (tip d1a00859, blob-checked), labels s2_labels_v8_eval, B 2000, on the EMA planner
  dumps pulled + sha256-verified by the EMA dose step (`dose/inputs_*.sha256`). Route following vs the TRUE command (the E2 arms train
  `--no-strategic`): every K2DC02s1 - K0 row (plan / gstr / anchor, s0 and s1) is NOT separated, e.g. plan s0 +0.0000 [-0.0340, +0.0317],
  anchor s0 +0.0207 [-0.0184, +0.0592]. Controls: REF - REF reads exactly 0.0; the K2DC and K2DC02 rows reproduce `../read_K2DC02/` exactly.
* The separated-worse rows of the replicate and its parent do NOT overlap (parent: LON_accel + LAT_curvature; replicate: LAT heading / cross).
  The rig's replicate false-positive rate for a separated cell is ~9.5 % (H-ESTIM-SEED-1): read them against that, not as lever effects.

## EMA dose reading (`dose/`, `floor/`, `combined/`; PREREG_K2DC02S1 sec. 3 "read twice: EMA and tail")
* `dose/` (r9_e2_dose_verdict.py --arm K2DC02s1, comparator K2DC in K2's slot): **DOSE-VOID**. In-training validity VOID: 1 of 13 late probes
  (step >= 3,000) inside [0.01, 0.03] (PREREG_K2DC02 sec. 2 needs >= 2). Pre-registered rows reported anyway: LAT_cross_mae K2DC02s1 - K2DC
  +0.0007 [-0.0025, +0.0039] s0 / +0.0006 [-0.0020, +0.0032] s1 (the bar is <= -2.5 mm).
* `floor/`: the same tool with K2DC02's EMA in K2's slot -- the EMA floor, never a verdict (`FLOOR_NOT_A_VERDICT.txt`).
* `combined/`: the EMA reading = VOID (no dose claim). The TAIL read's outcome decides; this is reported beside it.
* Why DOSE-VOID: the dead-beat dose controller chased one-batch probe noise (register row H-R9-E2-DOSECTL; `../e2raw/dcreplay_2026-11-22/`).

## How it was read
Thor E2_READ_K2DC02s1 (`logs/ROLL_STATUS.txt`, AWAIT-S6C 17:12:26Z) -> dev-box pull 3,494 MB, 14,404 files sha256-equal (17:23:01Z) ->
S6c x4 / box-paired / inputs / verdict on code d1a00859 (18:32:37Z). The first scoring run ABORTED at box-paired (`logs/s6c_resume.log.boxexport1736`):
its wrapper never exported `stack/scripts/r9_e1_eval.py` + `stack/tanitad` (derived from the K2DC02 wrapper as it was BEFORE that wrapper's own
11-20 fix). Re-run with both exported from the same tip (343 files blob-equal). The EMA dose step ran in its own work root
(/f/refcv9_e3/k2dc02s1ema, code pinned 20179573), because the tail driver would run it inside the tail's work root, where it refuses.
Line endings: 31 files were written with CRLF on the dev box; the copies here are LF and `SHA256SUMS` is over these bytes.
One training run per arm: every interval answers another draw of EPISODES, not another training run (H-ESTIM-SEED-1).
