# HANDOFF - refcv9 L1C (Architecture & Data agent, 2026-10-25 15:40 Berlin)

**State: everything that does not need the GPU is DONE and CHECKED; the two GPU rolls are QUEUED behind `refcv8-p4-gtmap`, which holds the shared dev-box GPU lock
(acquired 14:37:17+0100; its own budget probe projects ~21,500 s for seed 1 then seed 0 from cache => frees about 21:30-22:00). A foreign holder is never broken.**

What runs by itself (detached, survives this agent):
* `code/chain_l1c.sh` (pid in `chain.log` ZZL1CSTARTZZ) - waits for the lock (<= 12 h), rolls FRESH (10,456 windows, ~85-135 min) then the PLATFORM control (824 windows), releases the lock, then on CPU:
  panels -> `l1c_fit_score.py` -> `l1c_roc.py` -> `l1c_verdict.py` -> `l1c_next.py`. Markers `ZZL1C<NAME>ZZ` in `chain.log`: IDENT, COMMITWAIT, LOCKED, ROLL_fresh, ROLL_plat, GPUFREE, PANELS, FITTED, ROC, VERDICT, NEXT, END.
* `code/post_chain.sh` - after END writes `RESULT.md` mechanically (`l1c_report.py`); marker `ZZL1CREPORTZZ`.
* Outputs: `result/VERDICT.json` (mechanical outcome: VOID / O1 / O2 / O3), `result/VERDICT_TABLES.md`, `result/L1C_RESULT.json`, `result/L1C_ROC.json`, `result/L1C_NEXT.json`, `RESULT.md`.
  Heavy artifacts on F:/l1c (bank/ztac_fresh.npz, bank/ztac_plat.npz, tree/, kit_fresh/, kit_plat/). Raw clip ids exist only there.

If the chain dies: re-run `bash code/chain_l1c.sh` from `refcv9_l1c` (it skips a roll whose final bank already has the expected window count); the CPU stage alone: see the `cpu()` lines at the end of the chain script
(env: `R8_LAUNCH_TREE=<tree_fb47a58>`, `PYTHONPATH=<tree>/stack;<tree>/taniteval;<battery code>`, `CUDA_VISIBLE_DEVICES=-1`).

Evidence that the unattended part will work (all MEASURED, before the lock was won): `FEASIBILITY.md` sec. 3 (CPU smoke of the whole roll on the platform kit; bit-exact regression of the analysis chain on L1's own data,
`dryrun/REGRESSION_ON_L1_DATA.json`, 24/24 cells equal, K4 control reads RED when it should).

Decision for the Master Mind (not mine): let P4 finish (~21:30-22:00) and take the verdict ~23:30-00:00, or ask P4's owner to yield the lock earlier. Nothing else blocks.
