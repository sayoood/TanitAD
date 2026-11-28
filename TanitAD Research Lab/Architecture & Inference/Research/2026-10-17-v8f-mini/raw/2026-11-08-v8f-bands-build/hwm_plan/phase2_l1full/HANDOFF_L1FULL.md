# HANDOFF L1-FULL (phase 2 arm 1, AMEND_HWMPLAN_2), 2026-11-14: smoke passed, contrast script validated, full arm RUNNING

Tier T0 (world-model diagnostic, never driving performance). Code of record untouched. CPU only. STRATEGIC: PAUSED (PI item 25).

## State (MEASURED)
* Smoke (16 windows, both R35 seeds, `--stitch l1_full`, hwm_oracle, inference seed 0, run twice): plans, cost1, iters1, a1_raw finite; every array bit-equal across the two runs; geometry read from the built model `K1=10` (= ceil(30/3)), s2=3, s3=5, K2=2, T3=2, n_traj=30, n_op=10, stitch_mode l1_full. No metric read. 56-78 s per 16-window unit at 1 worker (two of the runs overlapped my validation run). Logs `phase2_l1full/smoke_s{0,1}.log`, outputs `x/smokeL1F_*` (284 KB each).
* Windows copied from phase 1, sha256 equal: s0 `windows.npz` 371c753b..., `windows_all.npz` 6453ab10...; s1 `windows.npz` 8833e292..., `windows_all.npz` 0198a248... (full hashes in the lead report/logs: `sha256sum` of `phase1_R35/s*/windows*.npz`).
* Contrast script `F:/TanitAD-scratch/hwm_plan/phase2_l1full/contrast_l1full.py` (sha256 b7661635aaf3c77a...) imports `run_hwm.load_ws`, `hwm_data.lead_for`, `hwm_eval.band_metrics/contrast/tactical_blocks/FAMILIES/BANDS`; B 2,000, seed 0.
  Validation: `hwm_oracle - flat_oracle` reproduces ALL 75 rows (39 contrasts + 36 replicate) of `RESULT_HWMPLAN_R35_s{0,1}.json` exactly (0 mismatches, both seeds, delta/lo/hi bit-equal). `hwm_oracle - hwm_oracle` reads exactly 0.0 [0, 0] on all 39 rows, both seeds. Outputs: `phase2_l1full/validate_hwm_minus_flat_s*.json`, `validate_self_s*.json`.

## Running
Waiter WINPID cmd 51524 (bash 26276/43992/42404 incl. job; the job started at 05:51:50 with RAM 10.14 GB, C: 25.6 GB, 2 workers). Job `phase2_l1full/phase2_l1full.sh` (sha256 409b01a9...): for s0, s1: `run_hwm.py plan ... --stitch l1_full --arms hwm_oracle --seeds {0,1} --workers 2`; 16 units of 64 windows per (seed, inference seed), 64 units in all. Log `phase2_l1full/phase2_l1full.log`; status `WAITER_STATUS.txt`; marker `phase2_l1full/PHASE2_L1FULL.done` = `rc=N <time>`. Plans: `phase2_l1full/s{0,1}/plans_hwm_oracle_s{0,1}.npz` (a re-run skips banked ones). If rc != 0, read the log tail, fix the cause, rerun the waiter via the WMI recipe.
ETA (ESTIMATED from the 16-window smoke, scaled 4x to a 64-window unit, not measured at full size): about 2.5-3 h at 2 workers (about 5 h at 1 worker).

## Control
Phase 1 HWM-ORACLE (`decode`): `phase1_R35/s{0,1}/plans_hwm_oracle_s{0,1}.npz`.

## Next step once the marker reads rc=0 (git-bash)
```
source /f/TanitAD-scratch/hwm_plan/common.sh; V=F:/TanitAD-scratch/hwm_plan/phase2_l1full; P1=F:/TanitAD-scratch/hwm_plan/phase1_R35
for S in s0 s1; do $PY $V/contrast_l1full.py --windows $P1/$S --a $V/$S/plans_hwm_oracle_s0.npz --b $P1/$S/plans_hwm_oracle_s0.npz --rep-a $V/$S/plans_hwm_oracle_s1.npz --rep-b $P1/$S/plans_hwm_oracle_s1.npz --out $V/CONTRAST_L1FULL_R35_$S.json > $V/CONTRAST_L1FULL_R35_$S.log; done
```
A = l1_full, B = decode; rows `A - B`, `A - HOLD`, `B - HOLD`, replicates of both, `tactical_kappa`. Read against the card's L1-FULL rows (ADDS: separated-better in 2-6 s, not separated-worse in 0-2 s, beyond the replicate), then the PLANNER-LIMITED row on the decode-free path. Use a `F:/` style path in python arguments, not `/f/`.

## Deletions
None (smoke dirs kept for banking, 1.1 MB total). Phase 1 files untouched.
