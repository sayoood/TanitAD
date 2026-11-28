# AMENDMENT 3 to the P6 read (2026-11-08 19:23Z, Master Mind): how the navtest read is produced

**Written before any P6 value was read.** The chain's smoke is still queued on the GPU.

`AMENDMENT_2.md` said the navtest read would run through a path-corrected copy of `eval/eval_checkpoint.py`. The launch agent found that route cannot run here. Its score step passes 12,146 `--tokens`, a 230,812-character override that exceeds the Windows command-line limit (WinError 206, measured in `stage2_sft.py`). It also takes no GPU lock and needs code that is absent on this machine.

**Ruling: accepted.** The chain makes the two calls `stage2_sft.py` used for u400's own full-navtest read: the path-corrected `refe_navtest_seam.py`, then the banked pinned `score_e.py` with no `--tokens`, on the same full token list (`navtest_full_tokens.json`, md5 `3a8b3653...`, 12,146 tokens, 136 logs). This is the route that produced the u400 reference, so P6 and u400 are read the same way. Path changes are recorded in `F:/TanitAD_p6read_work/evalcode/PATH_DIFF.txt`: 17 lines in 6 scratch files, root paths only. No pinned file changes, and gates G6-1 and G6-3 still compare this seam with the FP32 dump.
