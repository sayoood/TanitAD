# P6 gate fix -- test battery (dev box, full form unless args shown; <= 2 concurrent)

FIX = c2d4042 + the 11 changed files (gate_checks/launch_gate/test_launch_gate 3-way merged onto c2d4042); BASE = pristine c2d4042 export. A row is a REGRESSION only if BASE passes and FIX fails.

| test | FIX rc | FIX token | FIX s | BASE rc | BASE token | verdict |
|---|---|---|---|---|---|---|
| `test_p6_chain.py@gateenv` | 0 | ZZP6CHAIN_PASS 351 | 332.4 | - | - | PASS |
| `test_p6_levers.py_fast` | 0 | ZZP6_PASS 165 | 211.1 | - | - | PASS |
| `test_p6_chain.py` | 0 | ZZP6CHAIN_PASS 351 | 412.3 | - | - | PASS |
| `test_p6_levers.py` | 0 | ZZP6_PASS 199 | 334.4 | - | - | PASS |
| `test_gate_suite_prerun.py` | 0 | ZZTEST_PASS 144 | 279.6 | - | - | PASS |
| `test_suite_nesting.py` | 0 | ZZTEST_PASS 95 | 7.8 | - | - | PASS |
| `test_sft4r_gate.py` | 0 | ZZTEST_PASS 153 | 10.3 | - | - | PASS |
| `test_sft4r_chain.py` | 0 | ZZTEST_PASS 31 | 2.5 | - | - | PASS |
| `test_rls1_chain.py` | 0 | ZZTEST_PASS 522 | 129.6 | - | - | PASS |
| `test_rls1_queue.py` | 1 |  | 2.7 | 0 | ZZTEST_PASS 254 | REGRESSION |
| `test_rl_r0c.py` | 0 | ZZTEST_PASS 26  (1 unmutated suite + 25 RED mutations) | 51.9 | - | - | PASS |
| `test_rl_s1b.py` | 0 | ZZTEST_PASS 18  (1 unmutated suite + 17 RED mutations) | 15.9 | - | - | PASS |
| `test_rl_pickcon.py` | 0 | ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | 17.1 | - | - | PASS |
| `test_field_wiring.py` | 1 |  | 0.6 | 1 |  | fails on BASE too (pre-existing / environment) |
| `test_switch_resume_m6b.py` | 1 | ZZSWITCH_RESUME_M6B_FAIL | 45.4 | 1 | ZZSWITCH_RESUME_M6B_FAIL | fails on BASE too (pre-existing / environment) |
| `test_launch_gate.py_static_no-banked` | 0 | ZZLG_STATIC_PASS 674 | 35.0 | - | - | PASS |
| `test_p6_cpu_probe.py` | 0 | ZZTEST_PASS 111 | 122.4 | - | - | PASS |
| `test_p6_overlap.py` | 0 | ZZTEST_PASS 54 | 1.5 | - | - | PASS |
| `test_p2r_chain.py` | 0 | ZZTEST_PASS 139 | 5.4 | - | - | PASS |
| `test_launch_gate.py` | 0 | ZZLG_FULL_PASS 719 | 508.2 | - | - | PASS |
| `test_p6_planner_command.py` | 1 |  | 2.1 | 1 |  | fails on BASE too (pre-existing / environment) |
| `test_label_dir_refusal.py` | 0 | ZZTEST_PASS 6 | 4.5 | - | - | PASS |
| `test_gate_gpu_probes.py` | 0 | ZZTEST_PASS 16  (incl. the CUDA probe on this GPU) | 3.0 | - | - | PASS |
| `test_aux_g12_bf16.py` | 0 | ZZTEST_PASS 7  (1 unmutated suite of 30 checks + 6 RED mutat | 17.1 | - | - | PASS |
| `test_aux_s1_wiring.py` | 0 | ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | 157.5 | - | - | PASS |
| `test_p2rep_chain.py` | 0 | ZZTEST_PASS 353 | 168.9 | - | - | PASS |
| `test_s1aux_chain.py` | 0 | ZZTEST_PASS 256 | 58.9 | - | - | PASS |
| `test_s1c.py` | 0 | ZZTEST_PASS 220  (2 skipped, not counted) | 184.7 | - | - | PASS |
| `test_rl_fan_dump.py` | 0 | ZZTEST_PASS 67  (1 unmutated suite of 157 checks + 65 RED mu | 120.6 | - | - | PASS |
| `test_rl_tlane_label.py` | 0 | ZZTEST_PASS 36  (1 unmutated suite of 42 checks + 34 RED mut | 36.0 | - | - | PASS |
| `test_rl_safeobj.py` | 0 | ZZTEST_PASS 61  (1 unmutated suite + 60 RED mutations) | 169.4 | - | - | PASS |
| `test_s1c_autoread.py` | 0 | ZZTEST_PASS 186 | 101.8 | - | - | PASS |
| `test_decision_logic.py` | 0 | ZZDECISION_PASS 45 checks, 15/15 mutants killed | 0.3 | - | - | PASS |
