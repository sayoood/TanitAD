# PREREG_V8F_ROLLHIST, Amendment 4 (FROZEN 2026-11-21 04:48Z by the MM, before any RH or B0 run or number)

Card `PREREG_V8F_ROLLHIST.md` (781eea87) and Amendments 1 (87928487), 2 (15004dc9) and 3 (3c176fac) all stand. The commit of record stays **97fe0a56**. This amendment adds one file of that commit to the pod's tree copy. It changes no code, arm, argv, step, read or bar.

## What happened (MEASURED, pod)
* The chain on 97fe0a56 started at 02:08:58Z. It passed tree, inputs, render, G-SUITE (0 regressions, 0 new-test failures, 0 missing), the pilot, and `bank_s0` (finished 04:44:10Z, `done/bank_s0.done`).
* At 04:44:12Z the RH_s0 selfcheck (`v8f_mini_train.py --selfcheck-only`) refused with:
  `K-B3: the canonical eval-141 grid /root/rollhist/tree_cdb/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/eval_grid_windows.json does not exist`.
  The chain then ended `ZZ ROLLHIST FAIL ZZ step RH_s0`. No training step and no arm value exist.
* Cause: the trainer reads this grid from a repo-relative constant (`DAGGER_EVAL_GRID`, `stack/scripts/v8f_mini_train.py:54`), with no flag for it. The pod tree list `all_blobs_97fe0a56.txt` held only `stack/` and `taniteval/` (3,407 files), so the tree lacked the file. The bank step had passed K-B3 because it takes the grid as `--eval-grid /root/strat1/eval_grid_windows.json`.
* The refusal is the K-B3 control working as designed: a check that cannot run refuses.

## The change
* The file is in commit 97fe0a56 as git blob **c489b553fe9355cf50e6007bdf395bd4a07a6e9a** (21,050 bytes).
* The copies in `/root/dscale10k/tree` (the DSCALE10K tree, which its arms used) and `/root/strat1/eval_grid_windows.json` (the bank's `--eval-grid`) are the same blob.
* `fix_evalgrid.sh` copies it into `tree_cdb` at that path and checks the git hash of the copy. It appends that one line to `all_blobs_97fe0a56.txt` (copy kept as `*.bgrid`), then re-runs the chain's own tree check (blobcheck plus file count).
* `--p10` is explicit in every arm's argv, so the trainer's other repo-relative default (`PKG/raw/p10*/p10_prior.pt`) is never read. The tree needs no other file.
* The selfcheck is a first read under the failed-control rule: an environment repair made before any value, with the control form unchanged.

## Relaunch
* The same RH_READOUT=causal_g1x. The steps that are done are skipped; `bank_s0` is kept.
* RH_s0's partial selfcheck dir is moved aside by the chain's own `arm()`. Nothing is deleted.
* The N1DH launcher exited on the FAIL line as designed. It is restarted only after the new chain has written past that line.

## Lesson
Run the trainer's `--selfcheck-only` on the pod tree before the launch, not only G-SUITE and the pilot. The pilot exercised the bank and the readout, not the trainer's K-B3 path.
