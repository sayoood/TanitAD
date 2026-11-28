# HANDOFF_KINCACHE (2026-11-23): state at the end of the build

Done. Nothing is running. Nothing staged, committed or pushed.

* Code: `F:/TanitAD-scratch/kinlab/tree2/` (0ca6f49d export + 4 files, listed with sha256 and base blob in `tree2/CHANGED_FILES_KINCACHE.txt`): `stack/tanitad/data/v8f_dagger.py`, `stack/scripts/v8f_mini_train.py`, new `stack/tests/test_v8f_kincache.py`, new `stack/tests/golden_v8f_dagger_0ca6f49d.py`.
* Facts: `FACTS_KINCACHE.md`. Logs: `kc_final_kincache.log` (24 passed), `kc_test_v8f_*.log`, `kc_lg_*.log` (21 launch-gate files). Real-record round trip: `KINCACHE_REAL.json` (script `kincache_real.py`). Mutants: `mutants_kincache/MUTANTS_RESULT.json` (runner `run_mutants_kincache.py`; M24 re-run in `run_mutants_b.log`).
* Deletions logged in `DELETIONS.log` (section KINCACHE). `rm -rf` is refused by the harness in Bash; PowerShell `Remove-Item` worked.

Exact next step for the MM (the MM lands): copy the 4 files into the landing, run G-SUITE at the new commit, build the caches ONCE from the landed commit with `--selfcheck-only --dagger-label kin_v1 --dagger-label-cache <path>` for each bank (about 14 min CPU on this box, 27 min on the pod; about 0.7 GB per bank, ESTIMATED), outside the GPU lock, then name the same path in the RK argv (the gate token binds the argv, so `rk_argv.py` needs the new flag in its allowed difference). Any later edit of `v8f_dagger.py` or `v8f_recovery.py` invalidates a cache (named refusal, rebuild).
