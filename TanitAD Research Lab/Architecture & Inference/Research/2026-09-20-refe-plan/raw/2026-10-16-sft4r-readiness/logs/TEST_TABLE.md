| test | form | patched (rc / token / s / peak procs) | verdict | unpatched, if it failed |
|---|---|---|---|---|
| test_launch_gate--full | gate | rc 0 / `ZZLG_FULL_PASS 690` / 530.0 s / 8 | **PASS** |  |
| test_p6_chain | gate | rc 0 / `ZZP6CHAIN_PASS 341` / 510.2 s / 9 | **PASS** |  |
| test_gate_suite_prerun | gate | rc 0 / `ZZTEST_PASS 144` / 421.2 s / 17 | **PASS** |  |
| test_p2rep_chain | gate | rc 0 / `ZZTEST_PASS 353` / 237.6 s / 6 | **PASS** |  |
| test_p6_levers | gate | rc 0 / `ZZP6_PASS 145` / 344.7 s / 5 | **PASS** |  |
| test_rls1_chain | gate | rc 0 / `ZZTEST_PASS 512` / 137.1 s / 7 | **PASS** |  |
| test_s1c | gate | rc 1 / `NONE` / 97.0 s / 6 | **PRE-EXISTING (fails unpatched too)** | rc 1 AssertionError: (1, "-10-16T21:33:29 ZZS1AUX_UPSTREAM_NO_DONE rls1 (WA |
| test_aux_s1_wiring | gate | rc 1 / `NONE` / 67.1 s / 2 | **PRE-EXISTING (fails unpatched too)** | rc 1 AssertionError: the unmutated scorer_finetune.py FAILS its aux wiring  |
| test_aux_g12_bf16 | gate | rc 1 / `NONE` / 5.1 s / 2 | **PRE-EXISTING (fails unpatched too)** | rc 1 AssertionError: Invalid device id |
| test_s1aux_chain | gate | rc 1 / `NONE` / 63.0 s / 6 | **PRE-EXISTING (fails unpatched too)** | rc 1 AssertionError: ('data', 'navtest list', '-s1 H2-s1 N-s2 H2-s2 H1-s1 H |
| test_rl_r0c | gate | rc 0 / `ZZTEST_PASS 26  (1 unmutated suite + 25 ` / 80.5 s / 1 | **PASS** |  |
| e2e_suite_prerun_refe_sft4r | gate | rc 1 / `NONE` / 70.0 s / 13 | **NEW-TEST-FAILS (no unpatched counterpart)** |  |
| test_launch_gate--static | gate | rc 0 / `ZZLG_STATIC_PASS 645` / 46.5 s / 4 | **PASS** |  |
| test_launch_gate--static | alone | rc 0 / `ZZLG_STATIC_PASS 645` / 55.9 s / 4 | **PASS** |  |
| test_sft4r_gate | gate | rc 0 / `ZZTEST_PASS 152` / 20.3 s / 3 | **PASS** |  |
| test_sft4r_gate | alone | rc 0 / `ZZTEST_PASS 152` / 21.6 s / 3 | **PASS** |  |
| test_suite_nesting | gate | rc 0 / `ZZTEST_PASS 88` / 16.2 s / 13 | **PASS** |  |
| test_suite_nesting | alone | rc 0 / `ZZTEST_PASS 88` / 15.1 s / 13 | **PASS** |  |
| test_p2r_chain | gate | rc 0 / `ZZTEST_PASS 139` / 7.6 s / 5 | **PASS** |  |
| test_p2r_chain | alone | rc 0 / `ZZTEST_PASS 139` / 7.3 s / 5 | **PASS** |  |
| test_rl_s1b | gate | rc 0 / `ZZTEST_PASS 18  (1 unmutated suite + 17 ` / 19.6 s / 1 | **PASS** |  |
| test_p6_cpu_probe | gate | rc 0 / `ZZTEST_PASS 111` / 148.2 s / 8 | **PASS** |  |
| test_p6_overlap | gate | rc 0 / `ZZTEST_PASS 54` / 1.9 s / 3 | **PASS** |  |
| test_rls1_queue | gate | rc 0 / `ZZTEST_PASS 134` / 517.4 s / 8 | **PASS** |  |
| test_rls1_queue | alone | rc 0 / `ZZTEST_PASS 134` / 519.6 s / 7 | **PASS** |  |
| test_sft4r_chain | gate | rc 0 / `ZZTEST_PASS 20` / 1.9 s / 5 | **PASS** |  |
| test_sft4r_chain | alone | rc 0 / `ZZTEST_PASS 20` / 1.9 s / 5 | **PASS** |  |
| test_hf_egress | gate | rc 1 / `NONE` / 0.4 s / 1 | **PRE-EXISTING (fails unpatched too)** | rc 1 ZZTEST_FAIL 1/13 |
| test_hf_egress | alone | rc 1 / `NONE` / 0.5 s / 1 | **PRE-EXISTING (fails unpatched too)** | rc 1 ZZTEST_FAIL 1/13 |
| test_pdm_label_complete | gate | rc 0 / `ZZTEST_PASS 13` / 2.4 s / 1 | **PASS** |  |
| test_pdm_label_complete | alone | rc 0 / `ZZTEST_PASS 13` / 0.2 s / 1 | **PASS** |  |
| test_gate_gpu_probes | gate | rc 1 / `NONE` / 4.2 s / 1 | **PRE-EXISTING (fails unpatched too)** | rc 1 AssertionError: Invalid device id |
| test_gate_gpu_probes | alone | rc 1 / `NONE` / 4.1 s / 2 | **PRE-EXISTING (fails unpatched too)** | rc 1 AssertionError: Invalid device id |
