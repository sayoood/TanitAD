# E3 HANDOFF (written 2026-11-11 ~11:40)
State: two identical Test14-random NR runs of checkpoint_2400.pt, CPU, 2 ray workers each (4 procs total), low priority, OMP=1, started ~11:35.
Both runs use run_e3.sh (copy of F:/tanitad_scratch/tf_e3/t10/run.sh): same overrides as h0920 (REVIEW_4 s5) except device=cpu, no simulation-log callback, DB E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test (D: no longer exists), maps E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0, devkit+venv C:/Users/Admin/dz/DriveZero/DriveRL, C:/Users/Admin/venvs/driverl-eval.
10-scenario timing: 10 scen in 119 s wall on 4 workers (incl ~15 s warmup) => ~35-45 s/scenario/worker CPU; est. 261 scen at 2 workers ~ 1.5-2.5 h wall per run, both concurrent. Budget 12 CPU-h not exceeded (est. ~6 CPU-h total).
Root PIDs (cmd.exe): runA 44828, runB 35616 (children: bash, python, ray). Kill only these trees if needed.
CHECK: ls F:/tanitad_scratch/tf_e3/run{A,B}/DONE.txt  (contains DONE_RC=0 when finished); tail F:/tanitad_scratch/tf_e3/runA/run.log
SCORE (per run X in A,B): parquet = newest F:/tanitad_scratch/tf_e3/runX/out/aggregator_metric/*weighted_average*.parquet ; final = row scenario=='final_score' (num_scenarios notna), score*100. Verify 261 rows in runner_report.parquet, succeeded all True. Do NOT use the 10-scenario t10 run (its aggregator did not run; name fixed since).
Compare with 94.50 (Table A10). Spread |A-B| = noise unit. Outcome A (both within ~1 pt of 94.50) / B (both >=2 pts below) per TEACHER_REVIEW s6 E3; otherwise report "in between" honestly, no new bar.
Bank before delete: copy runX/out/aggregator_metric/*.parquet, runner_report.parquet, metrics/ to e3_identity/runX_bank/, then delete F:/tanitad_scratch/tf_e3 by literal path and log it. Write RESULT_E3.md.
Val14 NR (95.16): optional, DBSUB=trainval FILTER=driverl_val14, only if time.
