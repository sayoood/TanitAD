@echo off
"C:\Users\Admin\venvs\tanitad\Scripts\python.exe" C:\Users\Admin\pickcon\lane_runner.py C:\Users\Admin\pickcon\patched_full C:\Users\Admin\pickcon\patched_logs_fix C:\Users\Admin\pickcon\patched_summary_fix.json refe/test_dump_scores.py refe/test_rl_pickcon.py refe/test_launch_gate.py::--static+--no-banked refe/test_rl_safeobj.py eval/test_safeobj_outcome.py
echo LANE_DONE > C:\Users\Admin\pickcon\fix\patched_fix.done
