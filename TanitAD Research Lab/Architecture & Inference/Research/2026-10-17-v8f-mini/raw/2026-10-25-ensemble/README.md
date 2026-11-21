# Zero-GPU facts after HIST (2026-10-25): seed floors and the six-model plan average

EXPLORATORY: these decide nothing. They ranked the next lever and narrowed `RESULT_V8F_HIST.md` §3 (R66).

* **`ens_facts.{py,json,out}`:** averages the 6 s plans of the six banked M2b-family evals (M2b s0/s1, GRID, HIST, PGT, WC at seed 0). All are on eval-141's one grid, `--ages cycle`, kin_mode yaw.
  * Scored with `v8f_mini_eval.evaluate` and `ctx_from_args` (no re-implementation).
  * Control: the GT reproduces every arm's stored per-window ADE6 exactly (max error 0.0).
  * Run against an export of tip `211ab1bc`: `python ens_facts.py <export>`.
* **`lon_floor.{py,json,out}`:** the raw seed gap |M2b_s0 − M2b_s1| per longitudinal metric, next to GRID's and HIST's deltas. This is the floor for UNPAIRED comparisons.
* **`paired_swing.{py,json,out}`:** the swing of the one seed-paired contrast banked at two seeds (M2b − M1), on ADE6 and the longitudinal metrics. This is the floor for SEED-PAIRED deltas (n = 1).
* Inputs: the box-2 eval files under `C:/Users/Admin/box2_ship/logs/v8f_{mini,grid,hist,pgt,wc}/eval/`, banked in git under each arm's `raw/…/run/eval/` (M2b's under `raw/full/`).
