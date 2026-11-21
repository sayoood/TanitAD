# refcv9 E2 read of K2DC02 (map-hires share target 0.02) + its dose verdict

K2DC02 = K2DC with `--map-hires-share-target 0.02` and `--map-hires-w-bounds 0.1,25`; every other token is K2DC's
(`stack/ops/runs.d/r9e2_K2DC02.argv.json`). It trained at commit e454e96c6a84 on Thor, reaching DONE at 2026-11-19T23:01:16Z,
and passed the zerograd gate. Pre-registration: `e2/e2raw/PREREG_K2DC02_2026-11-18.md` (landed 75d9ca52, before any training step).

## How it was read
* **Thor rolls:** E2_READ_K2DC02, read tree read_tree_e454e96c6a84_k2dc. It was started by e2read_launch_k2dc02.sh after the
  roll re-pin c69b51a9. K0 was reused by hard link from the K2DC read. AWAIT-S6C was reached at 2026-11-20T01:59:36Z.
* **Pull to the dev box:** 14,404 files, sha256-equal to Thor (02:13:49Z).
* **S6c / box-paired / verdict:** run with r9e2_s6c_resume.sh on code af27d979, all files blob-verified. It took three runs:
  * The first was killed by a Windows Update restart at 02:33:33Z (log `logs/s6c_resume.log.reboot0233`).
  * The second aborted at BOX-PAIRED (log `logs/s6c_resume.log.boxexport0600`). The K2DC02 wrapper never exported
    stack/scripts/r9_e1_eval.py or stack/tanitad, because its template predates clause (b). Both were then exported at the same
    commit, with 330 files blob-verified.
  * The third ran clean (`logs/s6c_resume.log`).
* **Windows:** DEV (eval139 even half, 70 episodes / 2,397 windows); HOLD is never read.
* **Estimator:** paired episode-cluster bootstrap, B 2000. Each arm is ONE training run, so a separated interval answers
  "another draw of EPISODES" only (H-ESTIM-SEED-1).

## E2 verdict (pre-registered rule, unchanged): **NO-GO** (`read/E2_VERDICT.json`)

| clause | tier | seed 0 | seed 1 |
|---|---|---|---|
| (a) map primary nf_swedge_prec_lv_5_20, K2 > K0 (point) | T0 | FAIL 0.4066 vs 0.7505 | FAIL (same) |
| (b) box AP@2 m, not separated worse (paired) | T0 | PASS +0.0070 [+0.0015, +0.0127] | PASS |
| (c) S6c-DEV sidewalk rate, K2 < K0 (point) | T1 | FAIL 0.1889 vs 0.1883 | FAIL 0.1907 vs 0.1852 |
| (d) no four-family row separated worse | T1 | FAIL: LON_accel_mae, LAT_curvature_mae | FAIL: LON_accel_mae |
| (e) W4 preview vs camera-blind | T1 | PASS | PASS |

**Four families.** Every row's K2DC02 − K0 interval, per seed, is in `read/E2_VERDICT.json` under `four_families`.
* Separated worse:
  * LON_accel_mae_mps2: +0.0073 [+0.0007, +0.0135] at s0 and +0.0067 [+0.0004, +0.0134] at s1;
  * LAT_curvature_mae_1pm_valid: +0.0002 [+0.0001, +0.0003] at s0 only.
* LATERAL passes at s1, and TACTICAL passes at both seeds.
* STRATEGIC: the E2 reader still stamps PAUSED, which is stale. The PI lifted the pause on 2026-11-16. The rows are computed
  below, report-only.

## STRATEGIC family, report-only: route following vs the TRUE nav command (`strategic/`)
Tool: scratchpad `e2_strategic_navcomp.py` on the fixed reader (taniteval nav_compliance, 6e35e3b7). The refcv9 E2 arms train
`--no-strategic`, so the computable rows are route-following readouts.

| readout | K2DC − K0 s0 | K2DC − K0 s1 | K2DC02 − K0 s0 | K2DC02 − K0 s1 |
|---|---|---|---|---|
| plan | −0.0111 [−0.0543, +0.0330] | −0.0148 [−0.0494, +0.0178] | −0.0074 [−0.0422, +0.0246] | −0.0074 [−0.0361, +0.0202] |
| anchor | −0.0101 [−0.0542, +0.0263] | −0.0040 [−0.0359, +0.0283] | −0.0150 [−0.0502, +0.0155] | +0.0000 [−0.0300, +0.0290] |
| gstr | +0.0022 [−0.0177, +0.0179] | (same) | −0.0044 [−0.0272, +0.0197] | (same) |

* Nothing is separated.
* K0's plan compliance with the true command is 0.7037 [0.6259, 0.7778] at s0.
* The control (REF − REF) reads exactly 0.0 on every readout and seed.

## Dose verdict (PREREG_K2DC02 sec. 3, comparator K2DC in K2's slot): **INCONCLUSIVE** (`dose/DOSE_VERDICT.json`)
* **In-training validity HELD:** 3 of 13 late probes fall inside [0.01, 0.03].
* **K2DC02 − K2DC LAT_cross_mae:** −0.0014 [−0.0040, +0.0010] at s0 and −0.0003 [−0.0027, +0.0021] at s1. Neither reaches the
  −2.5 mm bar, and neither is separated worse.
* **Beside the rule, with no committed direction:**
  * map primary 0.4066 vs K2DC 0.4082;
  * box AP@2 m 0.3265 vs 0.3201;
  * sidewalk rate +0.0043 at both seeds.
* **Reading the dose axis:** cutting the trunk share from 0.10 to 0.02 moved no pre-registered row beyond its bar, and left the
  map primary unchanged (−0.0015).
* **Next step under sec. 4.4:** a replicate arm, same flags with seed + 1. That is the PI's decision. The PI's conditional K2SG go
  (sec. 4.2, SUPPORTED) did NOT fire. Check: scratchpad k2sg_condition.py, "NOT FIRED ... SEC4: 4.4".

Files: see `SHA256SUMS`. The 3.4 GB of dumps are not banked; Thor keeps E2_READ_K2DC02, and their sha256 lists are under `dose/`.
