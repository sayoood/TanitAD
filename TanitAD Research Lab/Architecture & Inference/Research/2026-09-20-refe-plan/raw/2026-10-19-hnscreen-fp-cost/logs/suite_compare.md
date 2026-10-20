| file | base rc | mine rc | base count | mine count | skips b/m | verdict |
|---|---|---|---|---|---|---|
| refe/test_launch_gate.py:--static,--no-banked | 0 | 0 | 1167 | 1168 | 0/0 | MORE (new checks) |
| eval/test_hn_screen_read.py | 0 | 0 | 31 | 36 | 0/0 | MORE (new checks) |
| refe/test_hn_content_fp.py | 0 | 0 | 81 | 81 | 0/0 | SAME |
| refe/test_hn_ctx_anchor.py:--no-mutants | - | 0 | - | 38 | -/0 | NEW FILE (mine only) |
| refe/test_hnscreen_ctx_chain.py:--arms-only | - | 0 | - | 9 | -/0 | NEW FILE (mine only) |
| refe/test_suite_nesting.py | 0 | 0 | 111 | 112 | 0/0 | MORE (new checks) |
| refe/test_ctx_cache.py | 0 | 0 | 15 | 15 | 0/0 | SAME |
| refe/test_hn_screen_wiring.py | 0 | 0 | 120 | 120 | 0/0 | SAME |
| refe/test_sft4r_gate.py | 0 | 0 | 153 | 153 | 0/0 | SAME |
| refe/test_carve_feed_gate.py | 0 | 0 | 10 | 10 | 0/0 | SAME |
| refe/test_gate_gpu_probes.py | 0 | 0 | 14 | 14 | 0/0 | SAME |
| refe/test_aux_g12_bf16.py | 0 | 0 | - | - | 1/1 | SAME |
| refe/test_p2r_chain.py | 0 | 0 | 139 | 139 | 0/0 | SAME |
| refe/test_rl_s1b.py | 0 | 0 | 18 | 18 | 0/0 | SAME |
| refe/test_rl_r0c.py | 0 | 0 | 26 | 26 | 0/0 | SAME |
| refe/test_rl_pickcon.py | 0 | 0 | 20 | 20 | 0/0 | SAME |
| refe/test_rls1_chain.py | 0 | 0 | 522 | 522 | 0/0 | SAME |
| refe/test_s1aux_chain.py | 0 | 0 | 256 | 256 | 1/1 | SAME |
| refe/test_sft4r_chain.py | 0 | 0 | 31 | 31 | 0/0 | SAME |
| refe/test_rls1e_chain.py | 0 | 0 | 936 | 936 | 0/0 | SAME |
| refe/test_hnscreen_chain.py | 0 | 0 | 16 | 16 | 0/0 | SAME |
| refe/test_s1c.py | 0 | 0 | 220 | 220 | 2/2 | SAME |
| refe/test_rl_safeobj.py | 0 | 0 | 61 | 61 | 0/0 | SAME |
| refe/test_rls1_queue.py | 1 | 1 | - | - | 0/0 | FAILS ON BOTH (pre-existing in this layout/environment) |
| refe/test_p2rep_chain.py | 0 | 0 | 353 | 353 | 0/0 | SAME |
| refe/test_p6_chain.py | 0 | 0 | 351 | 351 | 0/0 | SAME |
| refe/test_p6_levers.py | 0 | 0 | - | - | 0/0 | SAME |
| refe/test_aux_s1_wiring.py | - | 0 | - | 20 | -/0 | MINE PASS; NO BASE COMPARISON (base run skipped: no disk) |
| refe/test_gate_suite_prerun.py | - | 0 | - | 144 | -/0 | MINE PASS; NO BASE COMPARISON (base run skipped: no disk) |
| refe/test_launch_gate.py | - | 0 | - | 1213 | -/0 | MINE PASS; NO BASE COMPARISON (base run skipped: no disk) |

NEW FAILURES: none; LOWER COUNTS: none; PENDING: none; NO BASE COMPARISON (base skipped, no disk): ['refe/test_aux_s1_wiring.py', 'refe/test_gate_suite_prerun.py', 'refe/test_launch_gate.py']
