# PREP_RK2: RK revision 2 (commit 1a8b11a1, Amendment 1), pod prep results (MEASURED, LoRA pod, CPU only, nice 19, 2 threads; pod busy with UE_s3 then ADD123)
Kit: pod `/root/rollhist/kit_rk/` (SHA256_RK2.txt, 15 files, verified on the pod). Nothing staged/committed; no GPU, lock or marker touched; the RK chain was NOT launched.

## 1. Tree (a): `rk_tree_prep.sh` -> `/root/rollhist/tree_rk`
tree_ra re-verified 3415 of 3415 blobs, bad 0; tree_rk 3420 of 3420 blobs (`git ls-tree -r 1a8b11a1 stack taniteval` + eval-grid row, list sha256 2a64eb92...), bad 0; file count equal. `ZZ RK TREE OK ZZ`. The DRY walk re-verified it (3420 blobs).

## 2. Label caches (b): trainer `--selfcheck-only`, `--device cpu` the only argv difference
| seed | cache | sha256 | size | wall | ok records | kin_emitted | share (bar 0.5) |
|---|---|---|---|---|---|---|---|
| 0 | `/root/rollhist/kin_cache/s0_kin_v1.npz` | c882733d15c2f6214d72a80b0ecec5337846b5adeb146c1a215e82924ca094ea | 705,974,763 B | 1,645 s | 1,490,036 | 953,301 | 0.6398 |
| 1 | `/root/rollhist/kin_cache/s1_kin_v1.npz` | 5a99afdcf7ba48b19946cd2fa9fe26e1112afa894d7e10f19f648173b928c4dd | 702,188,056 B | 1,645 s | 1,478,090 | 945,608 | 0.6397 |
Census (`kin_status_counts`, full bank), s0: kin_emitted 953,301; refused_not_merged 310,272; refused_infeasible 70,901; speed_refused_infeasible 22,624; refused_path_exhausted 115,777; speed_path_exhausted 17,161. Seed 1 in `kin_cache/s1_precompute.json`. Both equal the dry mint's 0.6398 / 0.6397. Bar met at both seeds. Peak container total_rss during a precompute 11.9 GB; child RSS ~4.2 GB. The selfcheck's built levers read the file's sha256 (`declared_vs_built` PASS). Files beside each cache: `s<K>_selfcheck_cpu.json`, `s<K>_precompute.json|.log`, `s<K>_argv_cpu.json`.

## 3. G-SUITE (c) at 1a8b11a1: `rk_gsuite.sh` (nice 19, waves 3/3/2)
0 regressions, 0 new-test failures, 0 missing (n_new_only 190; new 15,041 tests vs base 14,851). `gsuite_result_rk.json` new commit 1a8b11a1. Wall 04:52:53 to 05:30:25 = 37.5 min (waves 10:15, 16:16, 11:01; peak container rss growth 11/17/6 GB).

## 4. Chain DRY walk (`DRY=1 bash kit_rk/rh_rk_chain.sh`, writes nothing)
tree_prep and gsuite read as done/valid; both caches PRECHECK OK; RK argv = RL + three changes (66 tokens; s0 sha256 b33c3e9c..., s1 ef02f855...); `ZZ ROLLHIST-RK DRY PASS ZZ`.

## 5. Dev-box tests (whole files, logs beside this file)
`test_log_rk_argv.txt` (well-formed + 21 RK regressions incl. other seed's cache + 8 source regressions: PASS); `test_log_rk_gate_record.txt` (57 cases PASS); `mutants_rk_gate_record_result.txt` (18 code mutants all RED); `test_log_rk_checker.txt` (24 passed); `mutants_rk_checker_result.txt` (8 mutants all RED); `test_log_chain_dry_rk.txt` (well-formed + 8 refuse-before-lock mutants PASS).

## Not run / caveats
* The real chain, the GPU selfcheck (a cache HIT under the lock), training: not run. Hit cost (hash of 0.7 GB + records) is ESTIMATED seconds to a few minutes.
* G-SUITE ran 3 shard processes at once (as revision 1's design, 2 threads each, nice 19).
* Checker change beyond the brief's list: `dscale10k_check_rk.py` must admit the cache token pair, so it was patched (patch_rk2.py, 3 anchored edits); it now lives in kit_rk (not study/).
* Pod files written outside the four named paths: `/root/rollhist/logs/rk_gsuite.log`, `suite_compare_rk.log`, `junit_rk/` (the G-SUITE's own files).
