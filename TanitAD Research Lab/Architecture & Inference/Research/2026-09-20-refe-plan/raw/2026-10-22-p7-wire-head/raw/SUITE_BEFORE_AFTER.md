# before / after, verbatim

51 test scripts (50 at the tip + the new one). rc 0: before 37 / after 39. Identical rc AND final token on 47; differing: ['test_hnscreen_chain', 'test_rls1_queue', 'test_suite_nesting']; new: ['test_p7_head_wiring']. Sum of the `ZZ…PASS <n>` counts of the green tests: before 8,270 / after 8,486 (the difference is the new test's count).
Red BEFORE (the same env, the tip's own code): ['test_aux_drivable_head', 'test_aux_label_store', 'test_aux_s1_wiring', 'test_carve_feed', 'test_hn_bank', 'test_hn_screen_wiring', 'test_hnscreen_chain', 'test_p6_planner_command', 'test_rl_fan_dump', 'test_rls1_chain', 'test_rls1_queue', 'test_s1aux_chain', 'test_sft4r_chain']. Red AFTER: ['test_aux_drivable_head', 'test_aux_label_store', 'test_aux_s1_wiring', 'test_carve_feed', 'test_hn_bank', 'test_hn_screen_wiring', 'test_p6_planner_command', 'test_rl_fan_dump', 'test_rls1_chain', 'test_rls1_queue', 'test_s1aux_chain', 'test_sft4r_chain']. Causes of the pre-existing reds, read off their own output (ENVIRONMENT of this box / the minimal export, not the change): missing `shapely` / `nuplan` modules, `raw/` artifacts absent from the export, a dev-box lock file touched by a parallel agent during the run, and non-`.py` fixtures rewritten to CRLF by the export (`fixtures/scorer_finetune_pre_aux.py.txt` re-hashes to its pin only after CRLF -> LF: MEASURED).

| test (`refe/`) | BEFORE (tip) | before token | AFTER (this change) | after token |
|---|---|---|---|---|
| `test_aux_drivable_head` | rc 1 | ZZTEST_FAIL  1 | rc 1 | ZZTEST_FAIL  1 |
| `test_aux_g12_bf16` | rc 0 | ZZTEST_SKIP_NO_CUDA | rc 0 | ZZTEST_SKIP_NO_CUDA |
| `test_aux_label_store` | rc 1 | (no ZZ token) | rc 1 | (no ZZ token) |
| `test_aux_s1_wiring` | rc 1 | (no ZZ token) | rc 1 | (no ZZ token) |
| `test_carve_feed` | rc 1 | (no ZZ token) | rc 1 | (no ZZ token) |
| `test_carve_feed_gate` | rc 0 | ZZTEST_PASS 10 | rc 0 | ZZTEST_PASS 10 |
| `test_ctx_cache` | rc 0 | ZZTEST_PASS 15 | rc 0 | ZZTEST_PASS 15 |
| `test_d9_accept` | rc 0 | ZZTEST_PASS 16 | rc 0 | ZZTEST_PASS 16 |
| `test_dump_scores` | rc 0 | ZZTEST_PASS 16 | rc 0 | ZZTEST_PASS 16 |
| `test_field_loss` | rc 0 | ZZTEST_PASS 42 | rc 0 | ZZTEST_PASS 42 |
| `test_flock_rw` | rc 0 | ZZTEST_PASS 59 | rc 0 | ZZTEST_PASS 59 |
| `test_gate_gpu_probes` | rc 0 | ZZTEST_PASS 14 | rc 0 | ZZTEST_PASS 14 |
| `test_gate_suite_prerun` | rc 0 | ZZTEST_PASS 144 | rc 0 | ZZTEST_PASS 144 |
| `test_hf_egress` | rc 0 | ZZTEST_PASS 13 | rc 0 | ZZTEST_PASS 13 |
| `test_hn_bank` | rc 1 | ZZHNBANK_SY | rc 1 | ZZHNBANK_SY |
| `test_hn_content_fp` | rc 0 | ZZTEST_PASS 81 | rc 0 | ZZTEST_PASS 81 |
| `test_hn_ctx_anchor` | rc 0 | ZZTEST_PASS 72 | rc 0 | ZZTEST_PASS 72 |
| `test_hn_screen_wiring` | rc 1 | ZZTEST_FAIL 2 | rc 1 | ZZTEST_FAIL 2 |
| `test_hnscreen_chain` | rc 1 | ZZTEST_FAIL 1 | rc 0 | ZZTEST_PASS 31 |
| `test_hnscreen_ctx_chain` | rc 0 | ZZTEST_PASS 22 | rc 0 | ZZTEST_PASS 22 |
| `test_label_dir_refusal` | rc 0 | ZZTEST_PASS 6 | rc 0 | ZZTEST_PASS 6 |
| `test_launch_gate` | rc 0 | ZZLG_STATIC_PASS 1168 | rc 0 | ZZLG_STATIC_PASS 1168 |
| `test_p2r_chain` | rc 0 | ZZTEST_PASS 139 | rc 0 | ZZTEST_PASS 139 |
| `test_p2rep_chain` | rc 0 | ZZTEST_PASS 353 | rc 0 | ZZTEST_PASS 353 |
| `test_p6_chain` | rc 0 | ZZP6CHAIN_PASS 432 | rc 0 | ZZP6CHAIN_PASS 432 |
| `test_p6_cpu_probe` | rc 0 | ZZTEST_PASS 111 | rc 0 | ZZTEST_PASS 111 |
| `test_p6_levers` | rc 0 | ZZP6_PASS 165 | rc 0 | ZZP6_PASS 165 |
| `test_p6_overlap` | rc 0 | ZZTEST_PASS 54 | rc 0 | ZZTEST_PASS 54 |
| `test_p6_planner_command` | rc 1 | (no ZZ token) | rc 1 | (no ZZ token) |
| `test_p7_head_wiring` | - | (no output) | rc 0 | ZZP7HEAD_PASS 179 |
| `test_pdm_label_complete` | rc 0 | ZZTEST_PASS 13 | rc 0 | ZZTEST_PASS 13 |
| `test_r0e_b9b10` | rc 0 | ZZTEST_PASS 3601 | rc 0 | ZZTEST_PASS 3601 |
| `test_rl_epo` | rc 0 | ZZTEST_PASS 42 | rc 0 | ZZTEST_PASS 42 |
| `test_rl_fan_dump` | rc 1 | (no ZZ token) | rc 1 | (no ZZ token) |
| `test_rl_partition` | rc 0 | ZZTEST_PASS 20 | rc 0 | ZZTEST_PASS 20 |
| `test_rl_pickcon` | rc 0 | ZZTEST_PASS 20 | rc 0 | ZZTEST_PASS 20 |
| `test_rl_pool_available` | rc 0 | ZZTEST_PASS 10 | rc 0 | ZZTEST_PASS 10 |
| `test_rl_r0c` | rc 0 | ZZTEST_PASS 26 | rc 0 | ZZTEST_PASS 26 |
| `test_rl_reward` | rc 0 | ZZTEST_PASS 31 | rc 0 | ZZTEST_PASS 31 |
| `test_rl_s1b` | rc 0 | ZZTEST_PASS 18 | rc 0 | ZZTEST_PASS 18 |
| `test_rl_safeobj` | rc 0 | ZZTEST_PASS 61 | rc 0 | ZZTEST_PASS 61 |
| `test_rl_terms_label` | rc 0 | ZZTEST_PASS 39 | rc 0 | ZZTEST_PASS 39 |
| `test_rl_tlane_label` | rc 0 | ZZTEST_PASS 36 | rc 0 | ZZTEST_PASS 36 |
| `test_rls1_chain` | rc 1 | (no ZZ token) | rc 1 | (no ZZ token) |
| `test_rls1_queue` | rc 1 | ZZRLS1_SAFEOBJ_NO_SECOND_CONTROL | rc 1 | (no ZZ token) |
| `test_rls1e_chain` | rc 0 | ZZTEST_PASS 936 | rc 0 | ZZTEST_PASS 936 |
| `test_s1aux_chain` | rc 1 | (no ZZ token) | rc 1 | (no ZZ token) |
| `test_s1c` | rc 0 | ZZTEST_PASS 220 | rc 0 | ZZTEST_PASS 220 |
| `test_sft4r_chain` | rc 1 | (no ZZ token) | rc 1 | (no ZZ token) |
| `test_sft4r_gate` | rc 0 | ZZTEST_PASS 153 | rc 0 | ZZTEST_PASS 153 |
| `test_suite_nesting` | rc 0 | ZZTEST_PASS 112 | rc 0 | ZZTEST_PASS 118 |
