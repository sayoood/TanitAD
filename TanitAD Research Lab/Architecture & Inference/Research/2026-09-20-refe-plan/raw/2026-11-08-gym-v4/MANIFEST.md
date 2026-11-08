# pilot_v4 manifest (M-GYM-4), written at launch 2026-11-08
| artifact | where | status |
|---|---|---|
| PREREG_M_GYM_4.md + .sha256 | repo-artifacts: E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v4/ | final, written before the fresh draw |
| PREREG_M_GYM_4_STATES.json + .sha256, draw_states.py | same | fresh 20-state list (seed 20261109) |
| scorer_gym/ (gym_env.py, run_v4.py, summarize_v4.py, chain_v4.py, determinism_child_v4.py, tests/test_v4.py, + v2 copies) | same | code; pinned in CODE_PIN.json by the chain after tests are green |
| explore/ (explore_corridor.py, prescreen_dist.py, ex_stall*.json, prescreen_dist.json, smoke_v4.py, launchers) | same | derivation on USED states |
| HANDOFF.md, launch_chain.py | same | next steps |
| run outputs (run_main, run_ttcoff, det*.txt, tests_v4_g*.out, results_v4.json, progress.txt) | F:/tanitad_scratch/mgym4/ ONLY until the next agent copies them | exists in ONE place until copied |
Nothing committed, pushed or landed. Nothing staged (git refuses on exFAT dubious ownership; the Master Mind lands).

## Update 2026-11-08 (result agent)
| artifact | where | status |
|---|---|---|
| M_GYM_4_RESULT.md | pilot_v4/ | final |
| raw/ (results_v4.json, summarize_v4.out, det1/2.txt, progress.txt, tests_v4_g0..3.out, run_main/{results,cost_rows,run_meta,states_used}.json + states.jsonl.gz + summarize_v2.out, run_ttcoff/{run_meta,states_used}.json + states.jsonl.gz, run_*.log.tail) | pilot_v4/raw/ (2,352,639 bytes) | copied, results/states files verified by cmp; now the only copy |
Deletion log: `F:/tanitad_scratch/mgym4` (2,661,358 bytes; remainder = explore logs, smoke, full run logs, chain.log, wait_all.sh; explore json already in pilot_v4/explore) deleted via Remove-Item -LiteralPath (result below). Not touched: F:/tanitad_scratch_mgym2.
Nothing committed, pushed or staged.
Deletion result: succeeded, path gone (Test-Path False).
