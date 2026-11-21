# MANIFEST pilot_v2 (all `repo-external: E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v2/`; nothing staged, nothing committed; every item exists in this one place unless noted)
| artifact | path | note |
|---|---|---|
| env code (v2) | scorer_gym/gym_env.py, run_pilot.py, summarize_v2.py, determinism_child.py | pinned in CODE_PIN.json |
| tests | scorer_gym/tests/test_gym.py | 6 pilot tests + 7 v2 tests (+2 state-level item-1 tests) |
| offline diagnostics | scorer_gym/analysis_v2_*.py, analysis_v2_t0_drivable_diag.out | MEASURED on pilot raw data / 4 states |
| pre-registration | PREREG_M_GYM_1.md | unchanged copy |
| pin | CODE_PIN.json | |
| handoff | HANDOFF.md | runs finished; M_GYM_2_RESULT.md complete (Part B filled) |
| run outputs | run_main/, run_stall/, run_hold2/ | results.json, cost_rows.json, run_meta.json, states_used.json, states.jsonl.gz, summarize_*.out, run_log_tail.txt (whole log) |
| gates | tests_v2.out, det1.txt, det2.txt, progress.txt | tests green, determinism diff empty |
| scratch (LEFT; PI manual delete) | F:\tanitad_scratch_mgym2 | original run dirs and logs; small outputs copied to the folders above |
