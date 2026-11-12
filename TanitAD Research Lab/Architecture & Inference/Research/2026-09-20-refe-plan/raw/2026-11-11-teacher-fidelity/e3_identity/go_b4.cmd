@echo off
rem E3 run B4 after the 2026-11-12 00:33Z reboot killed runB3 at scenario 5. The output path now CONTAINS the challenge name
rem closed_loop_nonreactive_agents, so nuPlan's MetricAggregatorCallback filter (metric_aggregator_callback.py:46) finds the metric
rem files natively; aggregate_posthoc.py is then run on the same files as a cross-check of the post-hoc score used for runA3.
set PATH=C:\Program Files\Git\usr\bin;C:\Program Files\Git\mingw64\bin;%PATH%
if not exist F:\tanitad_scratch\tf_e3\runB4\closed_loop_nonreactive_agents mkdir F:\tanitad_scratch\tf_e3\runB4\closed_loop_nonreactive_agents
"C:\Program Files\Git\usr\bin\bash.exe" F:/tanitad_scratch/tf_e3/t10/run.sh F:/tanitad_scratch/tf_e3/runB4/closed_loop_nonreactive_agents/out driverl_test14_random test 0 3 > F:\tanitad_scratch\tf_e3\runB4\run.log 2>&1 < NUL
echo DONE_RC=%ERRORLEVEL% > F:\tanitad_scratch\tf_e3\runB4\DONE.txt
