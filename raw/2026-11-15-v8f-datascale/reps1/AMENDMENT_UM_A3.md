# AMENDMENT UM-A3 to UM-A2 (frozen 2026-11-19 ~06:15Z, before any UE_s1 / UM_s1 number exists): two beside lines, no bar moves

Parent: `AMENDMENT_UM_A2.md` (sha256 3bc7968e, landed 40938dd8).
The v8f coordinator reviewed UM-A2 at 40938dd8 and accepted it as frozen, asking for these two additions. UE_s1 has been training since 05:45Z; no evaluation or summary number of either seed-1 arm has been read.

## R1. The two arms train on two commits (648d4ca3 against cdbff971): the evidence, and a direct control added
* 648d4ca3 is a child of cdbff971: two commits, 314 lines added and 5 removed, all in `stack/scripts/v8f_mini_train.py` (300 added, 0 removed) and `stack/tanitad/models/v8f_mini.py`.
* Every changed model line is behind the row weight `rw = batch.get("tac_w")`. With `rw is None`, the T-block goal, lat, lon and cons terms take the cdbff971 expressions unchanged (`v8f_mini.py` at 648d4ca3, the `rw is None` branches near lines 1167-1186 and 1276-1285).
* Already measured in the blockum prep (INHERITED from UM_s0's gate records, same trees):
  * C-INIT: UM's step-0 state equals UE's.
  * The fresh reach probe on tree_um: weight 1 gives `tac_lat` gradient 1642.025, equal to UE_s0's on the same batch, and weight 0 makes the tactical terms exactly 0.0.
* That evidence covers the step-0 state and one batch's loss. It does not cover the sampler or a full training step.
* **Added control K-TREE (beside, no bar, run before the read):**
  * tree_um with the three `--tac-src-*` flags absent, against tree, both at seed 1 with UE_s1's argv and 20 training steps, on the same device.
  * PASS = the 20 step losses are bit-identical.
  * If it fails, the UM - UE rows are printed as "lever plus commit".
  * It runs on the pod GPU after UM_s1 finishes, under the queue lock (minutes).

## R2. A floor on the paired delta, printed beside
* D2 = abs((UM_s1 - UE_s1) - (UM_s0 - UE_s0)) on the headline UM - UE TAC-LAT cell, printed beside F_cell, F_U and F2.
* A DILUTION-CONFIRMED-2SEED whose two-seed mean delta is below D2 is written "narrow" in its row.
* The labels, F2 and every bar stay as frozen in UM-A2.
