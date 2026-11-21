# LAUNCH_HANDOFF -- P6 read, 2026-11-08 20:19. STATE: CHAIN LAUNCHED (detached, WMI, stdin NUL). Nothing of P6 opened; no result seen.

Chain: `F:/TanitAD_p6read_work/chain_p6.py`. PIDs: cmd 56208 -> python 55336 (chain). Started 2026-11-08 20:19:16.
Stages: 0 GPU smoke (selc6_dump --smoke, own YAX lock; waits for the GPU) -> 1 navtest seam (shared lock via stage2_sft.take_lock) -> 2 W3 harness score, full navtest filter (score_e.py, no --tokens) -> 3 `p6_read.py`.
Expected: smoke 3 min + any lock wait (at launch the GPU was held by `strat_labels_ctx`, ~4.2 GB; the smoke had waited 14 min with no idle GPU); seam ~4.9 h (u400 full read took 17,497 s, MEASURED); score ~1-2 h; p6_read ~2.0-2.5 h. Total about 8-10 h plus waits.

## Markers
- `F:/TanitAD_p6read_work/markers/chain.DONE` (text = p6_read rc: 0 verdict issued, 2 instrument failure) / `markers/chain.FAILED` (text = traceback; a smoke failure carries the error text).
- Also `F:/TanitAD_p6read_work/run/markers/p6read.DONE|.FAILED` (written by p6_read.py), `smoke_PASSED.txt`, `CHAIN.log`, `chain_stdout.log`, `navtest_p6final/{seam.log,score_N.log,score_status.json}`, `run_stdout.log`.
- Wait: one Monitor until-loop on the two chain markers (not model turns).

## Inputs (done)
- `F:/TanitAD_p6read_work/ckpt/run/model_final.pt` 1,288,514,360 B, sha256 968a469d...a4d3d1 = AMENDMENT_2 (verified), `config.json` 4,577 B.
- `F:/TanitAD_p6read_work/train_logs.txt` 1,075 logs (one pod `ls` of `refe_navtrain/metric_cache`).
- RUNBOOK (a) control+pins: PASS (all gates True, all RED arms True). Output `F:/TanitAD_p6read_work/control_result.json`.
- Token list: `E:/Projects/TanitAD/data/refe_navtest/navtest_full_tokens.json`, md5 3a8b3653fd70891e50a230a37eb76ae8 = `stage3_full.FULL_PINS`, 12,146 tokens = the export's keys (136 logs); the u400 read (`stage2_sft4_B_u400_navtest_full`) used it. Name `p6final` -> seam `E:/Projects/TanitAD/data/refe_navtest/seams/refe_p6final.npz`, csv `.../score/refe_p6final/refe_p6final.csv`.

## DEVIATIONS (Master Mind to rule; no pinned file touched)
1. Smoke is stage 0 of the chain, not run beforehand: the foreground smoke was blocked on a busy GPU, and I may not wait alive. I stopped it by explicit PIDs (54868, 36732); it had loaded nothing. If it fails the chain writes FAILED with the error text and nothing else runs.
2. `eval_checkpoint.py` is NOT run. Its score step passes `--tokens` (12,146 tokens = 230,812-char Hydra override > Windows 32,767 limit: WinError 206, MEASURED in `stage2_sft.py` comments, six failures), scores with W3's D: `run_v1_fix1`, takes no GPU lock, and its floors/families steps need `C:/Users/Admin/ev6/...` (absent). The chain instead runs the same two calls `stage2_sft.py` used for the u400 full read: the scratch copy of `refe_navtest_seam.py` (default rule navsim_v1, as the dump) and the banked pinned harness `score_e.py`. Path-only copies of `eval_checkpoint.py` and `score_navtest_refe.py` are in evalcode/ but unused.
3. Scratch copy: `F:/TanitAD_p6read_work/evalcode/{eval,refe,code}` (siblings resolved via `__file__`: refe/ and code/ are `HERE/..`). `PATH_DIFF.txt`: 17 changed lines in 6 files, only `D:/Projects/TanitAD`->`E:/Projects/TanitAD` and `D:/Archive`->`F:/Archive` (targets verified to exist). `orig/` holds the unedited three eval files.

## Next steps
1. Monitor on the two markers. FAILED: read the text, fix the cause, delete `markers/chain.FAILED`, relaunch chain_p6.py the same way (it resumes: smoke, seam and score are skipped when done).
2. DONE: quote only from `run/p6_speed_result.json`. Then RUNBOOK sec. 4: register update, then delete (literal paths, log size) `F:/TanitAD_p6read_work/score`, `.../smoke` (if left), `.../ckpt`; keep markers and the navtest seam/csv. Pre-existing `F:/TanitAD_p6read_scratch` (previous agent) is not mine.
Deleted so far: nothing (the smoke dir is removed by the chain on pass, size logged in `smoke_PASSED.txt`).
