# PREREG_V8F_GRID run, seed 0 (box 2, 2026-10-22/23)

`code/box2_ops/chain_v8f_grid.sh` ran twice. Its committed rule stopped it at seed 0 (FAIL-NO-MECHANISM); seed 1 was not trained.

* **Run 1** (refreshed at tip `a8f2c89`; `box2_logs/chain_v8f_grid_run1.out`, `v8fgrid_guard_run1.out`, `v8fgrid10222155_*.log`):
  * base B = M2b, B-side check, cache check, G gate PASS;
  * G s0 trained 22:22–03:07 and evaluated to 03:11;
  * then the reader REFUSED on a lever-record schema difference (see `../a2/` and PREREG Amendment 2).
* **Run 2** (tip `cf5956f2`, Amendment 2; `box2_logs/chain_v8f_grid.out`, `v8fgrid_guard.out`, `v8fgrid10230322_*.log`):
  * the same checks and gate;
  * training and evals SKIPPED by the queue (their outputs existed);
  * the committed seed-0 read at 03:29.
* `grid_read_s0.json`: the committed read. `base.json`: the base decision from PGT's read. `check_inputs.json`: the B side, checked before training. `cache_check.json`: box 2's caches, every clip carrying 8×20.
* `eval/`: G s0's four evals. `runs/M2b_s0_8x20/`: G's run (summary, selfcheck, argv, train log).
* The checkpoint is not in git: `D:/Projects/TanitAD-artifacts/v8f_grid/M2b_s0_8x20/ckpt_last.pt` (sha256 `9562ceb7…`).
* `manifest_eval.jsonl` is banked. The train manifest (3.05 MB, sha256 `661f5d45…`) is summarised by `cache_check.json`.
