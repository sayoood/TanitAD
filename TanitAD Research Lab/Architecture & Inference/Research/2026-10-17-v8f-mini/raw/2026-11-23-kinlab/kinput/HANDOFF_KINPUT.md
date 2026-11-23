# HANDOFF_KINPUT (2026-11-23): state at the end of the build

Done except the census. Nothing is running. Nothing staged, committed or pushed.

* Code: `F:/TanitAD-scratch/kinlab/tree4/` (fde8b716 export + 6 files, sha256 and base blobs in `tree4/CHANGED_FILES_KINPUT.txt`): `stack/tanitad/data/v8f_dagger.py`, `stack/scripts/v8f_mini_train.py`, `stack/tests/test_v8f_kinput.py` (new), `stack/tests/golden_v8f_dagger_fde8b716.py` (new), `stack/tests/test_v8f_kintb.py` and `test_v8f_kincache.py` (one line each).
* Facts + results: `FACTS_KINPUT.md`. Logs: `kif_*.log` (final-file test runs), `ki_lg_*.log` (21 launch-gate files), `ki_nb_*.log` (first neighbour pass; its failures were fixed, then everything re-run as `kif_*`), `mutants_kinput/MUTANTS_RESULT.json` (M0 green, 14/14 RED; `run_mutants_kinput.py`, `mutants_kinput_run.log`). Scripts: `run_tests4.sh`, `run_chain_ki.sh`, `run_final_ki.sh`, `census_kinput.py`. Deletions: `DELETIONS.log` (KINPUT line).
* Usage: `--dagger-p P --dagger-mode swap --dagger-base logged` (default `own`).

For the MM:
1. CENSUS NOT RUN: the pod pull of shard 0 of RH bank s0 was refused by the permission classifier. Place the shard and `native_tracks_union.pt` on F: (or allow the pull), then run `census_kinput.py` (command in FACTS_KINPUT.md; untested on real data), one python process behind `wait_commit.py 5.5 60`, and delete the pulled files afterwards.
2. Caches built at fde8b716 are refused by tree4 naming `[dagger_sha256]` (code binding; `v8f_dagger.py` was edited). The labeller is untouched.
3. Landing moves the sha256 of `v8f_dagger.py` and `v8f_mini_train.py` (gate token, trainer pins, G-SUITE not run). No test-pinned launch-gate fingerprint moved. The 21 LG files ran on the final code files.
4. Under `logged` O's prior sees the OWN p10f (8 kin features) with the HUMAN's kd_x base: that is the lever the card names. The G-LIVE probe was changed to require base unchanged under `logged`.
