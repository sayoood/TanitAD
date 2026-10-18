# SO-PICKCON test results (baseline = pristine 0a5958a export, patched = the files below)

Lanes complete: True. Regressions: ['test_dump_scores.py', 'test_safeobj_outcome.py']. Failing at baseline too: ['test_aux_navtest_forward.py', 'test_p6_chain.py', 'test_rls1_chain.py', 'test_rls1_queue.py'].

| test file (run in full form) | baseline rc | baseline token | patched rc | patched token | patched s | verdict |
|---|---|---|---|---|---|---|
| `test_aux_g12_bf16.py` | 0 | ZZTEST_PASS 7  (1 unmutated suite of 30 checks + 6 RED mutations) | 0 | ZZTEST_PASS 7  (1 unmutated suite of 30 checks + 6 RED mutations) | 36.9 | **PASS = PASS** |
| `test_aux_navtest_forward.py` | 1 | AssertionError: the head module is not the D: copy | 1 | AssertionError: the head module is not the D: copy | 2.2 | **FAILS AT BASELINE TOO (environmental)** |
| `test_aux_s1_wiring.py` | 0 | ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | 0 | ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | 250.0 | **PASS = PASS** |
| `test_dump_scores.py` | 0 | ZZTEST_PASS 16  (1 unmutated suite of 41 checks + 14 RED mutations + coverage) | 1 | AssertionError: the unmutated source FAILS its own checks: ['parser: exactly ONE new dest  | 4.6 | **REGRESSION** |
| `test_gate_gpu_probes.py` | 0 | ZZTEST_PASS 16  (incl. the CUDA probe on this GPU) | 0 | ZZTEST_PASS 16  (incl. the CUDA probe on this GPU) | 3.0 | **PASS = PASS** |
| `test_gate_suite_prerun.py` | 0 | ZZTEST_PASS 144 | 0 | ZZTEST_PASS 144 | 262.3 | **PASS = PASS** |
| `test_label_dir_refusal.py` | 0 | ZZTEST_PASS 6 | 0 | ZZTEST_PASS 6 | 4.7 | **PASS = PASS** |
| `test_launch_gate.py_static_no-banked` | 0 | ZZLG_STATIC_PASS 674 | 0 | ZZLG_STATIC_PASS 674 | 55.8 | **PASS = PASS** |
| `test_p2r_chain.py` | 0 | ZZTEST_PASS 139 | 0 | ZZTEST_PASS 139 | 5.1 | **PASS = PASS** |
| `test_p2rep_chain.py` | 0 | ZZTEST_PASS 353 | 0 | ZZTEST_PASS 353 | 257.1 | **PASS = PASS** |
| `test_p6_chain.py` | 1 | ZZGATE-FAIL-000000000000ZZ | 1 | ZZGATE-FAIL-000000000000ZZ | 146.6 | **FAILS AT BASELINE TOO (environmental)** |
| `test_p6_cpu_probe.py` | 0 | ZZTEST_PASS 111 | 0 | ZZTEST_PASS 111 | 125.2 | **PASS = PASS** |
| `test_p6_levers.py` | 0 | ZZP6_PASS 156 | 0 | ZZP6_PASS 156 | 349.0 | **PASS = PASS** |
| `test_p6_overlap.py` | 0 | ZZTEST_PASS 54 | 0 | ZZTEST_PASS 54 | 1.4 | **PASS = PASS** |
| `test_rl_epo.py` | 0 | ZZTEST_PASS 42  (1 unmutated suite + 41 RED mutations) | 0 | ZZTEST_PASS 42  (1 unmutated suite + 41 RED mutations) | 133.8 | **PASS = PASS** |
| `test_rl_pickcon.py` | None | (not run) | 0 | ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | 39.1 | **NEW TEST** |
| `test_rl_r0c.py` | 0 | ZZTEST_PASS 26  (1 unmutated suite + 25 RED mutations) | 0 | ZZTEST_PASS 26  (1 unmutated suite + 25 RED mutations) | 116.1 | **PASS = PASS** |
| `test_rl_reward.py` | 0 | ZZTEST_PASS 31  (1 unmutated suite + 29 RED mutations + coverage) | 0 | ZZTEST_PASS 31  (1 unmutated suite + 29 RED mutations + coverage) | 1.4 | **PASS = PASS** |
| `test_rl_s1b.py` | 0 | ZZTEST_PASS 18  (1 unmutated suite + 17 RED mutations) | 0 | ZZTEST_PASS 18  (1 unmutated suite + 17 RED mutations) | 32.6 | **PASS = PASS** |
| `test_rl_safeobj.py` | 0 | ZZTEST_PASS 61  (1 unmutated suite + 60 RED mutations) | 0 | ZZTEST_PASS 61  (1 unmutated suite + 60 RED mutations) | 408.0 | **PASS = PASS** |
| `test_rl_terms_label.py` | 0 | ZZTEST_PASS 39  (1 unmutated suite + 37 RED mutations + coverage) | 0 | ZZTEST_PASS 39  (1 unmutated suite + 37 RED mutations + coverage) | 3.4 | **PASS = PASS** |
| `test_rls1_chain.py` | 1 | AssertionError: MODEL_REGISTRY.md not found above the package | 1 | AssertionError: MODEL_REGISTRY.md not found above the package | 79.4 | **FAILS AT BASELINE TOO (environmental)** |
| `test_rls1_queue.py` | 1 | AssertionError: {'rc': 1, 'out': "ps: unknown option -- o\nTry `ps --help' for more inform | 1 | AssertionError: {'rc': 1, 'out': "ps: unknown option -- o\nTry `ps --help' for more inform | 65.9 | **FAILS AT BASELINE AND PATCHED, DIFFERENT TAIL** |
| `test_s1aux_chain.py` | 0 | ZZTEST_PASS 256 | 0 | ZZTEST_PASS 256 | 78.6 | **PASS = PASS** |
| `test_s1c.py` | 0 | ZZTEST_PASS 220  (2 skipped, not counted) | 0 | ZZTEST_PASS 220  (2 skipped, not counted) | 153.3 | **PASS = PASS** |
| `test_safeobj_outcome.py` | 0 | ZZTEST_PASS 133  (1 unmutated suite of 172 checks + the CLI end to end, 132 RED mutations) | 1 | AssertionError: the UNMUTATED tool fails: ['outcome:o1->O1', 'outcome:o1_ttc->O1', 'outcom | 71.5 | **REGRESSION** |
| `test_sft4r_chain.py` | 0 | ZZTEST_PASS 31 | 0 | ZZTEST_PASS 31 | 3.2 | **PASS = PASS** |
| `test_sft4r_gate.py` | 0 | ZZTEST_PASS 153 | 0 | ZZTEST_PASS 153 | 11.0 | **PASS = PASS** |
| `test_suite_nesting.py` | 0 | ZZTEST_PASS 95 | 0 | ZZTEST_PASS 95 | 8.5 | **PASS = PASS** |

<!--FIXPASS-->
## Regression fix, second pass (2026-10-18) -- supersedes the two REGRESSION rows above

Fix pass complete: True. Regressions after the fix: none. Fixed: ['test_dump_scores.py', 'test_safeobj_outcome.py']. Failing at baseline too (environmental): ['test_aux_navtest_forward.py', 'test_p6_chain.py', 'test_rls1_chain.py', 'test_rls1_queue.py']. Token mismatches on re-run files: none.

| test file (full form) | re-run in fix pass | baseline rc | baseline token | patched rc (first pass) | patched rc (fix pass / final) | patched token (final) | patched s | verdict |
|---|---|---|---|---|---|---|---|---|
| `test_aux_g12_bf16.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 7  (1 unmutated suite of 30 checks + 6 RED mutations) | 0 | 0 | ZZTEST_PASS 7  (1 unmutated suite of 30 checks + 6 RED mutations) | 36.9 | **PASS = PASS** |
| `test_aux_navtest_forward.py` | no (first-pass row stands) | 1 | AssertionError: the head module is not the D: copy | 1 | 1 | AssertionError: the head module is not the D: copy | 2.2 | **FAILS AT BASELINE TOO (environmental)** |
| `test_aux_s1_wiring.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | 0 | 0 | ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | 250.0 | **PASS = PASS** |
| `test_dump_scores.py` | yes | 0 | ZZTEST_PASS 16  (1 unmutated suite of 41 checks + 14 RED mutations + coverage) | 1 | 0 | ZZTEST_PASS 16  (1 unmutated suite of 41 checks + 14 RED mutations + coverage) | 28.8 | **FIXED (first pass REGRESSION) -> PASS = PASS** |
| `test_gate_gpu_probes.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 16  (incl. the CUDA probe on this GPU) | 0 | 0 | ZZTEST_PASS 16  (incl. the CUDA probe on this GPU) | 3.0 | **PASS = PASS** |
| `test_gate_suite_prerun.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 144 | 0 | 0 | ZZTEST_PASS 144 | 262.3 | **PASS = PASS** |
| `test_label_dir_refusal.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 6 | 0 | 0 | ZZTEST_PASS 6 | 4.7 | **PASS = PASS** |
| `test_launch_gate.py_static_no-banked` | yes | 0 | ZZLG_STATIC_PASS 674 | 0 | 0 | ZZLG_STATIC_PASS 674 | 42.8 | **PASS = PASS** |
| `test_p2r_chain.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 139 | 0 | 0 | ZZTEST_PASS 139 | 5.1 | **PASS = PASS** |
| `test_p2rep_chain.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 353 | 0 | 0 | ZZTEST_PASS 353 | 257.1 | **PASS = PASS** |
| `test_p6_chain.py` | no (first-pass row stands) | 1 | ZZGATE-FAIL-000000000000ZZ | 1 | 1 | ZZGATE-FAIL-000000000000ZZ | 146.6 | **FAILS AT BASELINE TOO (environmental)** |
| `test_p6_cpu_probe.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 111 | 0 | 0 | ZZTEST_PASS 111 | 125.2 | **PASS = PASS** |
| `test_p6_levers.py` | no (first-pass row stands) | 0 | ZZP6_PASS 156 | 0 | 0 | ZZP6_PASS 156 | 349.0 | **PASS = PASS** |
| `test_p6_overlap.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 54 | 0 | 0 | ZZTEST_PASS 54 | 1.4 | **PASS = PASS** |
| `test_rl_epo.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 42  (1 unmutated suite + 41 RED mutations) | 0 | 0 | ZZTEST_PASS 42  (1 unmutated suite + 41 RED mutations) | 133.8 | **PASS = PASS** |
| `test_rl_pickcon.py` | yes | None | (not run) | 0 | 0 | ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | 17.3 | **NEW TEST** |
| `test_rl_r0c.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 26  (1 unmutated suite + 25 RED mutations) | 0 | 0 | ZZTEST_PASS 26  (1 unmutated suite + 25 RED mutations) | 116.1 | **PASS = PASS** |
| `test_rl_reward.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 31  (1 unmutated suite + 29 RED mutations + coverage) | 0 | 0 | ZZTEST_PASS 31  (1 unmutated suite + 29 RED mutations + coverage) | 1.4 | **PASS = PASS** |
| `test_rl_s1b.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 18  (1 unmutated suite + 17 RED mutations) | 0 | 0 | ZZTEST_PASS 18  (1 unmutated suite + 17 RED mutations) | 32.6 | **PASS = PASS** |
| `test_rl_safeobj.py` | yes | 0 | ZZTEST_PASS 61  (1 unmutated suite + 60 RED mutations) | 0 | 0 | ZZTEST_PASS 61  (1 unmutated suite + 60 RED mutations) | 208.7 | **PASS = PASS** |
| `test_rl_terms_label.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 39  (1 unmutated suite + 37 RED mutations + coverage) | 0 | 0 | ZZTEST_PASS 39  (1 unmutated suite + 37 RED mutations + coverage) | 3.4 | **PASS = PASS** |
| `test_rls1_chain.py` | no (first-pass row stands) | 1 | AssertionError: MODEL_REGISTRY.md not found above the package | 1 | 1 | AssertionError: MODEL_REGISTRY.md not found above the package | 79.4 | **FAILS AT BASELINE TOO (environmental)** |
| `test_rls1_queue.py` | no (first-pass row stands) | 1 | AssertionError: {'rc': 1, 'out': "ps: unknown option -- o\nTry `ps --help' for more inform | 1 | 1 | AssertionError: {'rc': 1, 'out': "ps: unknown option -- o\nTry `ps --help' for more inform | 65.9 | **FAILS AT BASELINE TOO (environmental)** |
| `test_s1aux_chain.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 256 | 0 | 0 | ZZTEST_PASS 256 | 78.6 | **PASS = PASS** |
| `test_s1c.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 220  (2 skipped, not counted) | 0 | 0 | ZZTEST_PASS 220  (2 skipped, not counted) | 153.3 | **PASS = PASS** |
| `test_safeobj_outcome.py` | yes | 0 | ZZTEST_PASS 133  (1 unmutated suite of 172 checks + the CLI end to end, 132 RED mutations) | 1 | 0 | ZZTEST_PASS 136  (1 unmutated suite of 177 checks + the CLI end to end, 135 RED mutations) | 586.1 | **FIXED (first pass REGRESSION) -> PASS = PASS** |
| `test_sft4r_chain.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 31 | 0 | 0 | ZZTEST_PASS 31 | 3.2 | **PASS = PASS** |
| `test_sft4r_gate.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 153 | 0 | 0 | ZZTEST_PASS 153 | 11.0 | **PASS = PASS** |
| `test_suite_nesting.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 95 | 0 | 0 | ZZTEST_PASS 95 | 8.5 | **PASS = PASS** |
<!--/FIXPASS-->
