# HANDOFF_WMPOOL (code half done; nothing staged, committed or pushed)
State: lever coded and controls green (`CONTROLS_WMPOOL.md`). Tree `F:/TanitAD-scratch/wmpool/tree` = base `1faa1a06` + `wmpool.diff` (re-creatable: `git -c core.autocrlf=false archive 1faa1a06 stack taniteval`, `git apply wmpool.diff`).
Next, in order:
1. Master Mind lands `wmpool.diff` (3 files: `v8f_mini_train.py`, `v8f_mini.py`, `v8f_wm_windows.py`) and `zcells.py`; the `launch_gate.py` owner adds the G-DVB entry for `wm_pool`.
2. Box 2: ship `zcells.py` (needs `theory/ladder/zcache.py` beside it, or `ZCACHE_DIR`), then `zcells.py build --ckpt 0=<M2b_s0 ckpt> 1=<M2b_s1 ckpt> --out D:\tanitad\zcells`, `manifest` (same --ckpt), and `check3 --clips-file ladder\clips8.json --zmean-dir D:\tanitad\zcache --ckpt 0=.. 1=.. --out-json ZCELLS_CHECK3.json` (control 3). About 20 min, 2.36 GB per seed (ESTIMATED); box-2 D: has ~11 GB.
3. Chain for `argvs_wmpool.txt` (3 runs, patterned on EGOWM's chain; tree pinned to the landing commit).
4. Read side: `READSIDE_WMPOOL.md`.
