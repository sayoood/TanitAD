# K2DC02-TAIL read: the share dose and the E2 rule on the tail-EMA weights (zero training)

* **Pre-registration:** `e2/e2raw/PREREG_K2DC02TAIL_2026-11-20.md` (d20cd6f3), with AMENDMENT 1 (387616b3). Both were written
  before any K2DC02 tail weights existed.
* **Kit:** `e2/e2raw/k2dc02tail/scripts` (387616b3, plus the env-order fix 7b26a8ac) and `devbox/` (ea503356).
* **Weights:** W_tail = (EMA_6000 − 0.81872 · EMA_5000) / 0.18128 for K2DC02. It carries no step-0 init, like an EMA at E3's
  73,280 steps (`thor/run_K2DC02/model_ema_tail5k6k.pt.provenance.json`, md5 6fc5423b…).
* **Read conditions:**
  * DEV windows: eval139 even half, 70 episodes.
  * Inference seeds 0 and 1.
  * Estimator: paired episode-cluster bootstrap, B 2000.
  * ONE training run per arm, so a separated interval answers "another draw of EPISODES" only (H-ESTIM-SEED-1).

## Validity: all items HELD
1. **Identity control (A1, tensor identity):** IDCTL-PASS-TENSOR. A K2DC tail rebuild was `torch.equal` on all 1,201 tensors.
   File bytes differ because `torch.save` is not byte-deterministic (`thor/BUILD_STATUS.txt`).
2. **Tail guard:** the stage A DEV dump records the tail md5 and step 6000 (`thor/stageA/STATUS.txt`).
3. **Deliberate regression:** REGRESSION-RED. The guard refused a dump header made from `model_ema.pt`.
4. **Comparators:** K0-tail / K2DC-tail planner rolls RE-ROLLED in one Thor environment (A2). Every K0-slot file reused from
   E2-TAIL's pull was sha256-equal to its Thor list (7,198 files).
5. **Reproduction check (A2 §2.5): REPRO-PASS.** The re-rolled K2DCt − K0t reproduces E2-TAIL's banked families exactly on every
   row, e.g. LAT_cross_mae +0.0126 / +0.0091 m, difference 0.0 (`verdicts/REPRO_CHECK.txt`).

## Primary: the dose rule (PREREG_K2DC constants, unchanged): **INCONCLUSIVE** (`verdicts/dose/DOSE_VERDICT.md`)
In the tool's labels, "K2DC" is K2DC02-tail and "K2" is K2DC-tail. All rows are T1.

| seed | K2DC02t − K0t LAT_cross_mae | K2DC02t − K2DCt LAT_cross_mae | K2DC02t − K2DCt cross_abs_final | jitter K2DC02t − K2DCt |
|---|---|---|---|---|
| 0 | +0.0075 [+0.0024, +0.0126], separated worse | **−0.0051 [−0.0095, −0.0011]**, separated better | −0.0115 [−0.0216, −0.0025] | +0.0044 [−0.0056, +0.0137] |
| 1 | +0.0083 [+0.0024, +0.0147], separated worse | −0.0008 [−0.0051, +0.0034] | −0.0005 [−0.0103, +0.0091] | +0.0046 [−0.0039, +0.0132] |

* SUPPORTED fails at both seeds: K2DC02t is still separated worse than K0t.
* REFUTED holds at seed 1 only, so the outcome is INCONCLUSIVE.
* Reading:
  * At seed 0 the 0.10 → 0.02 share cut removed about 5 mm of K2DC's 12.6 mm cross-track deficit, and that change is separated.
  * At seed 1 it removed about 1 mm.
  * At both seeds about 8 mm of deficit against K0 remains.

## Secondary: the E2 rule on tail weights, K2DC02t vs K0t: **NO-GO, clause (d) only** (`read/E2_VERDICT.json`)

| clause | tier | result |
|---|---|---|
| (a) map primary, own FIT thresholds | T0 | **PASS** 0.4742 vs 0.4665 (the EMA read had 0.4066) |
| (b) box AP@2 m | T0 | **PASS** +0.0108 [+0.0039, +0.0182] (K2DC-tail FAILED (b) in E2-TAIL) |
| (c) S6c sidewalk rate | T1 | **PASS** s0 0.1852 vs 0.1914, s1 0.1821 vs 0.1963 |
| (d) four families | T1 | **FAIL**, see below |
| (e) W4 preview | T1 | **PASS** |

**Four families: rows separated worse** (`verdicts/FAM_TABLE.md` has every row):
* **Seed 0:**
  * LAT_cross_mae +0.0075 [+0.0024, +0.0126];
  * LAT_cross_abs_final +0.0173;
  * LON_dk_headway_abs_err.
  * The LONGITUDINAL family FAILS on that row; TACTICAL PASSES.
* **Seed 1:**
  * LON_accel_mae +0.0140 [+0.0032, +0.0256];
  * LON_dk_headway_min and LON_dk_min_ttc;
  * LAT_heading +0.0759 [+0.0140, +0.1451];
  * LAT_yaw_rate;
  * LAT_cross_mae +0.0083 [+0.0024, +0.0147];
  * LAT_cross_abs_final.
  * TACTICAL FAILS at seed 1 (TAC_traj_lat_correct −0.0088, separated).

**STRATEGIC** (report-only, never a pass; `strategic/`, clip IDs replaced by their index):
* Plan compliance with the TRUE command, K2DC02t − K0t, at seed 1: −0.0630 [−0.1345, −0.0041], **separated worse**.
* At seed 0: −0.0074, not separated.
* The control (REF − REF) reads exactly 0.0.

## Committed next step
PREREG_K2DC02TAIL §5.3 (INCONCLUSIVE): no new evidence on the dose. The PREREG_K2DC02 §4.4 replicate (K2DC02 with seed + 1)
stays the committed next step, and its read must include this tail read.

Beside the rule, MM, with no committed direction: even the best seed leaves about 8 mm of separated lateral deficit against K0. A
SUPPORTED dose would therefore still leave (d) failing. The untested bundled lever is HSG (`PREREG_K2DC02NH_2026-11-20.md`,
c87b4dbb).

**Files:** see `SHA256SUMS`. The dumps are not banked; Thor keeps `runs/E2TAIL_K2DC02`.
