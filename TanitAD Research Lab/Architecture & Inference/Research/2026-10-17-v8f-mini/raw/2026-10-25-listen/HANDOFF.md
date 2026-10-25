# HANDOFF: PREREG_V8F_LISTEN (stopped by the coordinator on memory/pagefile pressure; nothing deleted)

OUT = `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/listen/`
Python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`; every call `CUDA_VISIBLE_DEVICES=-1`, torch 8 threads, `PYTHONUTF8=1`. Nothing staged or committed.

## State
* **Phase A DONE.** `listen.py` (the model's own run_s/run_t/run_o, ONE input of ONE layer swapped), `test_listen.py` (20 passed, `pytest_log.txt` is the 18-test run before the reading tests were added; rerun for the current log), `run_mutations.py` (9 source-level mutations, all RED, unmutated GREEN: `mutation_log.txt`).
* **Phase B inputs complete** (SFRESH `READ.json` exists). Controls already read: **reproduction PASS** (clean pass vs banked box-2 eval and SFRESH CPU re-run, s0: max metric diff 4.1e-5 vs the 1e-3 stop; `repro_s0.json`).
* **Finished runs** (`runs/*.npz`, each loads): `s0_d0` `s1_d0` `s0_d1` (the grid, donor seeds 0/0/1), `s0_id` `s1_id` (identity control), `s0_navc` `s0_tsoff` (structural-zero copies, seed 0).
* **Missing:** `runs/s1_navc.npz` (killed at 0%), `runs/s1_tsoff.npz`, and `analyze` (-> `READ.json`, then the figure and `LISTEN_RESULT.md`, neither written yet).
* No LISTEN number has been looked at. `listen_read.py` docstring fixes readings R1-R5 (argmax of unclipped n, "holds" needs the argmax cell to listen, donor-seed floor applied to both seeds, headline precedence, Q-NAV rule) BEFORE any number.

## Resume (exact)
```
cd <OUT>
LISTEN_LRU=8 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 PYTHONUTF8=1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe run_listen.py all > run_all2.log 2>&1
```
`all` skips existing run files, so it does only the two missing regress runs (~1 min each) and then `analyze` (~5 min of bootstrap, B=2000). Then `python fig_listen_grid.py` (reads READ.json; layout tested on a mock) and write `LISTEN_RESULT.md` (outcome first: both tables, controls, per-seed grid, Q-NAV, committed next steps from the PREREG; state deviation D1 below). `READ.json` also carries every number needed.

## Memory plan (the run held 2.2 GB working set, 3.1 GB private)
* Cause: `ClipStore(lru=64)` kept up to 64 eval clips (about 21 MB fp16 each, plus pinned float copies) resident, donor clips included. **Fixed in `run_listen.py`: lru now `LISTEN_LRU`, default 8.** Windows are consecutive within an episode and donors come from ONE other episode, so 8 clips cover a batch; expected peak < 1.5 GB private.
* If still above target: batch 32 (`--batch 32` does not change results beyond float noise; identity/regress controls compare inside one run), `python -X importtime` unnecessary; run the analysis step alone (`run_listen.py analyze`, it loads only the small npz plus `X`).
* Disk: C: had 0.4 GB free; runs write about 2.7 MB each, `READ.json` and the PNG under 2 MB. Do not write anything else big.

## Deviation D1 (state it in the result)
`v8f_mini_eval._f1_reducer` raises IndexError on this numpy when the bootstrap hands it float64 arrays; `listen_read.f1_reducer` wraps the eval's own reducer with an exact integer cast. No number changes.

## Manifest (all under OUT)
`listen.py` `listen_read.py` `run_listen.py` `fig_listen_grid.py` `test_listen.py` `run_mutations.py` `mutation_log.txt` `pytest_log.txt` `repro_s0.json` `run_all.log` `runs/` `banked/` (banked tac-shuffle evals extracted from git, for control 5) `code/` (tip export, 67 MB) `mock/` (figure test only, delete freely: PI-only rule applies, I did not delete it).
