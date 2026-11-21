**Generated 2026-10-17 23:43 by `tools/post_queue.py`: 22 runs, 3 not clean.** Raw: `logs/qfinal/SUMMARY.txt` (patched, one line per run), `logs/qbase/SUMMARY.txt` (baseline re-run of each failure). Each test ran ALONE, at below-normal priority, CUDA not hidden, no bash shim except the row named `_shim`. Tree = the 2026-10-17 final set on a `git archive` of tip `3b8c39f`.

| test (queue id) | patched: rc / token / s | baseline (pristine tip + SO-A16), if it failed | verdict |
|---|---|---|---|
| test_sft4r_chain | rc 0 / `ZZTEST_PASS 31` / 5 s / skips 0 |  | **PASS** |
| test_suite_nesting | rc 0 / `ZZTEST_PASS 95` / 8 s / skips 0 |  | **PASS** |
| test_sft4r_gate | rc 0 / `ZZTEST_PASS 153` / 11 s / skips 0 |  | **PASS** |
| test_p2r_chain | rc 0 / `ZZTEST_PASS 139` / 7 s / skips 0 |  | **PASS** |
| test_p6_overlap | rc 0 / `ZZTEST_PASS 54` / 1 s / skips 0 |  | **PASS** |
| test_rl_s1b | rc 0 / `ZZTEST_PASS 18  (1 unmutated suite + 17 RED mu` / 11 s / skips 0 |  | **PASS** |
| test_rl_r0c | rc 0 / `ZZTEST_PASS 26  (1 unmutated suite + 25 RED mu` / 46 s / skips 0 |  | **PASS** |
| test_rls1_chain | rc 0 / `ZZTEST_PASS 522` / 166 s / skips 0 |  | **PASS** |
| test_s1c | rc 0 / `ZZTEST_PASS 220  (2 skipped, not counted)` / 117 s / skips 2 | rc 0 / ZZTEST_PASS 220  (2 skipped, not counted) | **PRE-EXISTING (fails on the baseline too)** |
| test_aux_s1_wiring | rc 0 / `ZZTEST_PASS 20  (1 unmutated suite + 19 RED mu` / 129 s / skips 0 |  | **PASS** |
| test_s1aux_chain | rc 1 / `NONE` / 58 s / skips 0 | rc 1 / AssertionError | **PRE-EXISTING (fails on the baseline too)** |
| test_aux_g12_bf16 | rc 1 / `NONE` / 17 s / skips 0 | rc 1 / AssertionError: the G12 bf16 checks FAIL on the real source: ['h1cuda/preflight_ran_to_a_G | **PRE-EXISTING (fails on the baseline too)** |
| test_gate_gpu_probes | rc 0 / `ZZTEST_PASS 16  (incl. the CUDA probe on this ` / 3 s / skips 0 |  | **PASS** |
| test_p6_cpu_probe | rc 0 / `ZZTEST_PASS 111` / 137 s / skips 0 |  | **PASS** |
| test_p6_levers | rc 0 / `ZZP6_PASS 156` / 212 s / skips 0 |  | **PASS** |
| test_p2rep_chain | rc 0 / `ZZTEST_PASS 353` / 191 s / skips 0 |  | **PASS** |
| test_rl_safeobj | rc 0 / `ZZTEST_PASS 61  (1 unmutated suite + 60 RED mu` / 345 s / skips 0 |  | **PASS** |
| test_gate_suite_prerun | rc 0 / `ZZTEST_PASS 144` / 609 s / skips 0 |  | **PASS** |
| test_p6_chain | rc 0 / `ZZP6CHAIN_PASS 341` / 780 s / skips 0 |  | **PASS** |
| test_rls1_queue | rc 0 / `ZZTEST_PASS 254` / 922 s / skips 0 |  | **PASS** |
| test_launch_gate_full_noshim | rc 0 / `ZZLG_FULL_PASS 719` / 441 s / skips 0 |  | **PASS** |
| test_launch_gate_full_shim | rc 0 / `ZZLG_FULL_PASS 719` / 343 s / skips 0 |  | **PASS** |
