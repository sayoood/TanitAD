| script | pristine (rc, result line) | patched (rc, result line) | same |
|---|---|---|---|
| test_aux_g12_bf16.py | 0, ZZTEST_SKIP_NO_CUDA: no CUDA device -- the G12 feed probe was NOT exercised on the bf16 au | 0, ZZTEST_SKIP_NO_CUDA: no CUDA device -- the G12 feed probe was NOT exercised on the bf16 au | YES |
| test_aux_s1_wiring.py | 0, ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | 0, ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | YES |
| test_carve_feed.py | 1,   9 G7 re-reads the disjointness + probes the feed           refe/scorer_finetune.py:2585 | 1,   9 G7 re-reads the disjointness + probes the feed           refe/scorer_finetune.py:2585 | YES |
| test_carve_feed_gate.py | 0, ZZTEST_PASS 10  (1 unmutated suite + 9 RED mutations) | 0, ZZTEST_PASS 10  (1 unmutated suite + 9 RED mutations) | YES |
| test_div_gate.py | 0, ZZTEST_PASS | 0, ZZTEST_PASS | YES |
| test_dose_gate.py | 0, ZZTEST_PASS | 0, ZZTEST_PASS | YES |
| test_ego_yaw.py | 0, ZZTEST_SKIP real-data control: no nuPlan devkit in this interpreter | 0, ZZTEST_SKIP real-data control: no nuPlan devkit in this interpreter | YES |
| test_gate_gpu_probes.py | 0, ZZTEST_PASS 14  (SKIP_CUDA: the GPU check did NOT run) | 0, ZZTEST_PASS 14  (SKIP_CUDA: the GPU check did NOT run) | YES |
| test_gate_suite_prerun.py | 0, ZZTEST_PASS 144 | 0, ZZTEST_PASS 144 | YES |
| test_hn_content_fp.py | 0, ZZTEST_PASS 81  (62 arms on the real tree + 19 RED mutations) | 0, ZZTEST_PASS 81  (62 arms on the real tree + 19 RED mutations) | YES |
| test_hn_ctx_anchor.py | 0, ZZTEST_PASS 72  (38 arms on the real tree + 34 RED mutations) | 0, ZZTEST_PASS 72  (38 arms on the real tree + 34 RED mutations) | YES |
| test_hnscreen_chain.py | 0, ZZTEST_PASS 31  (1 unmutated suite of 14 checks + 30 RED mutations; the shared dev-box loc | 0, ZZTEST_PASS 31  (1 unmutated suite of 14 checks + 30 RED mutations; the shared dev-box loc | YES |
| test_hnscreen_ctx_chain.py | 0, ZZTEST_PASS 22  (10 chain arms + 12 RED mutations of hnscreen_devbox.py) | 0, ZZTEST_PASS 22  (10 chain arms + 12 RED mutations of hnscreen_devbox.py) | YES |
| test_launch_gate.py | 1, ZZTEST_SKIP banked generic gate not present: primitives not cross-checked here | 1, ZZTEST_SKIP banked generic gate not present: primitives not cross-checked here | YES |
| test_p2r_chain.py | 0, ZZTEST_PASS 139 | 0, ZZTEST_PASS 139 | YES |
| test_p2rep_chain.py | 0, ZZTEST_PASS 353 | 0, ZZTEST_PASS 353 | YES |
| test_p6_chain.py | 0, ZZP6CHAIN_PASS 431 | 0, ZZP6CHAIN_PASS 431 | YES |
| test_p6_cpu_probe.py | 0, ZZTEST_PASS 111 | 0, ZZTEST_PASS 111 | YES |
| test_p6_levers.py | 0, ZZP6_PASS 199 | 0, ZZP6_PASS 199 | YES |
| test_p6_overlap.py | 0, ZZTEST_PASS 54 | 0, ZZTEST_PASS 54 | YES |
| test_p7_chain.py | 0, ZZP7CHAIN_PASS 110 | 0, ZZP7CHAIN_PASS 110 | YES |
| test_p7_head_wiring.py | 0, ZZP7HEAD_PASS 182 | 0, ZZP7HEAD_PASS 182 | YES |
| test_p7_integration.py | 0, ZZP7INT_PASS 225 | 0, ZZP7INT_PASS 225 | YES |
| test_p7_ship_list.py | 0, ZZP7SHIP_TEST_PASS 33 | 0, ZZP7SHIP_TEST_PASS 33 | YES |
| test_rep_gate.py | 0, ZZTEST_PASS | 0, ZZTEST_PASS | YES |
| test_rl_pickcon.py | 0, ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | 0, ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | YES |
| test_rl_r0c.py | 0, ZZTEST_PASS 26  (1 unmutated suite + 25 RED mutations) | 0, ZZTEST_PASS 26  (1 unmutated suite + 25 RED mutations) | YES |
| test_rl_s1b.py | 0, ZZTEST_PASS 18  (1 unmutated suite + 17 RED mutations) | 0, ZZTEST_PASS 18  (1 unmutated suite + 17 RED mutations) | YES |
| test_rl_safeobj.py | 0, ZZTEST_PASS 61  (1 unmutated suite + 60 RED mutations) | 0, ZZTEST_PASS 61  (1 unmutated suite + 60 RED mutations) | YES |
| test_rls1_chain.py | 1, <no stdout> | 1, <no stdout> | YES |
| test_rls1_queue.py | 0, ZZTEST_PASS 254 | 0, ZZTEST_PASS 254 | YES |
| test_rls1e_chain.py | 0, ZZTEST_PASS 936  (1 unmutated chain + 16 RED mutations) | 0, ZZTEST_PASS 936  (1 unmutated chain + 16 RED mutations) | YES |
| test_s1aux_chain.py | 0, ZZTEST_PASS 253 | 0, ZZTEST_PASS 253 | YES |
| test_s1c.py | 0, ZZTEST_PASS 220  (2 skipped, not counted) | 0, ZZTEST_PASS 220  (2 skipped, not counted) | YES |
| test_s1z_gate.py | 0, ZZTEST_PASS | 0, ZZTEST_PASS | YES |
| test_s1z_preflight.py | 1, <no stdout> | 1, <no stdout> | YES |
| test_s1zsm_gate.py | 0, ZZTEST_PASS | 0, ZZTEST_PASS | YES |
| test_s1zsm_preflight.py | 1, <no stdout> | 1, <no stdout> | YES |
| test_scorer1_gate.py | 0, ZZTEST_PASS | 0, ZZTEST_PASS | YES |
| test_scorer1_mutation.py | 0, ZZTEST_PASS | 0, ZZTEST_PASS | YES |
| test_scorer1p6_gate.py | 0, ZZTEST_PASS | 0, ZZTEST_PASS | YES |
| test_scorer1p6_preflight.py | 1, <no stdout> | 1, <no stdout> | YES |
| test_session_reserve.py | 0, ZZSESSRES_PASS 43  (optional checks run: ['aux_label_store.finalize', 'filter over the rea | 0, ZZSESSRES_PASS 43  (optional checks run: ['aux_label_store.finalize', 'filter over the rea | YES |
| test_sft4r_chain.py | 0, ZZTEST_PASS 31 | 0, ZZTEST_PASS 31 | YES |
| test_sft4r_gate.py | 0, ZZTEST_PASS 153 | 0, ZZTEST_PASS 153 | YES |
| test_suite_nesting.py | 0, ZZTEST_PASS 250 | 0, ZZTEST_PASS 250 | YES |
| test_vdac_g12_bound.py | 0, ZZTEST_PASS 15 | 0, ZZTEST_PASS 15 | YES |
| test_vdac_gate.py | 0, ZZTEST_PASS | 0, ZZTEST_PASS | YES |
| test_vdac_hb.py | 0, ZZTEST_PASS 133 | 0, ZZTEST_PASS 133 | YES |
| test_vdac_hb_accum.py | 0, ZZTEST_PASS 40 | 0, ZZTEST_PASS 40 | YES |
| test_vdac_hbmem.py | 0, ZZTEST_PASS 83 | 0, ZZTEST_PASS 83 | YES |
| test_vdac_stage0_forward.py | 1, <no stdout> | 1, <no stdout> | YES |
