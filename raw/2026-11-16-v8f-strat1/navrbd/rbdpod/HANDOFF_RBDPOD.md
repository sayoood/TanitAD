# HANDOFF_RBDPOD (2026-11-18)

Launched 22:01:54Z on the LoRA pod, detached. Script: `/root/strat1/navrbd/rbdpod/rbdpod.sh` (local copy `F:/TanitAD-scratch/strat1/navrbd/rbdpod/stage/`, generator `gen.py`).
Log + sentinel: `/root/strat1/logs/rbdpod.log`, last line `ZZ RBDPOD DONE ZZ` or `ZZ RBDPOD FAIL ZZ <step and why>`. Expected run time about 1 h (G-SUITE ~25 min, train ~25 min, dump).

## Steps and checks
1. Waits for `ZZ NAVRBD DUMPS2 END ZZ` (already seen at launch); needs `out/t1/DONE_N1D_s1`, no `FAIL_N1D_s1`.
2. `tree_fix` = `cp -a tree` + 5 files from `rbdpod/files/`: all 5 by sha256 (`TREE_FILES_SHA256.txt`, = commit 5d474c91 blobs, verified against `git archive`); manifest `navrbd_tree_fix.sha256` (18 lines); `diff -rq tree tree_fix` shows exactly 5 entries; exactly one `eval_grid_windows.json`, blob c489b553; `v8f_t1_eval_rbd.py` sha identical in both trees.
3. `navrbd_gsuite_fix.sh` (log `logs/navrbd_gsuite_fix.log`, result `gsuite_result_navrbd_fix.json`, junit `junit/navrbd_fix`): base = `junit/navrbd` (0c3b374f, today), 8 shards, new test file placed in the smallest shard. Pass = 0 regressions, 0 new-test failures, 0 missing, `new_commit` = 5d474c91.
4. Moves `runs/N1Dx_s0` to `runs/N1Dx_s0.attempt_refused` (plus copies of its mint/pre_* logs), then `flock /root/gpu_queue.lock bash navrbd_chain_fix.sh` (log `logs/navrbd_fix.log`; ARMS="N1Dx:s0"). Needs `ZZ STRAT1 NAVRBD DONE ZZ`, `ARM_DONE`, `ckpt_last.pt`.
5. `dump_rbd_fix.sh N1Dx_s0 follow 0` (CPU) -> `out/t1/DONE_N1Dx_s0`, `out/t1/t1v8f_NAVRBD-HO-N1Dx_s0_m10_causal_heads.json`.

## Deviations from the brief (necessary, all in the copies; originals untouched)
- `navrbd_chain_fix.sh` was built from the POD's `navrbd_chain.sh` (LF). The local `b4/navrbd_chain.sh` is CRLF and stale.
- Besides the 6 requested edits, 3 more were required: `navrbd_tree.sha256` -> `navrbd_tree_fix.sha256` (3 of its files changed), `render_argv.py` -> `render_argv_fix.py` (hard-codes TREE=tree), tree-verify log renamed `navrbd_tree_verify_fix.log`.
- Dry run (DRY=1, steps 2-3 composition without pytest, plus N1Dx render + pair on the dry tree) passed; dry temp deleted (tree_fix_dry 454M, junit/tmp dry dirs, dry logs; by literal path).

## MM next
- Read the last line of `rbdpod.log`. On FAIL: the step is named; the pod logs above hold the cause. Nothing was deleted; the refused attempt is kept.
- On DONE: N1Dx_s0 dump is at `out/t1/`; the paired reading vs N1R_s0 / N1D_s0 is the MM's. Bank the checkpoint, argv, gate token and `gsuite_result_navrbd_fix.json`; the tree_fix files are commit 5d474c91.
- Chain-start check (step 4): `runs/mint_N1Dx_s0.log` must show the mint passing (the G-LIVE fix); a second refusal there means the fix is incomplete.

## Addendum (MM request): zero-variant dumps
`/root/strat1/navrbd/rbdpod/rbdzero.sh` (separate waiter, launched 22:0xZ, because rbdpod.sh was already running). It waits for the last line of rbdpod.log to be `ZZ RBDPOD DONE|FAIL ZZ`, then runs the ORIGINAL `dump_rbd.sh N1D_s0 zero 0` and `N1D_s1 zero 0` (tree 0c3b374f), even if the main script failed; skipped only if out/t1/DONE_N1D_s0/s1 are missing. Output `out/t1_zero/` (tags NAVRBD-HO-N1D_s{0,1}_zero, confirmed from dump_rbd.sh: SUF=_zero). Lines appended to rbdpod.log: `ZZ RBDPOD ZERO <arm> DONE|FAIL ZZ`, then the FINAL sentinel `ZZ RBDPOD ALL END ZZ main: <main sentinel>; zero dumps DONE|FAIL`. So the final line of rbdpod.log is ALL END, with the main result inside it (the main DONE/FAIL line sits above).
