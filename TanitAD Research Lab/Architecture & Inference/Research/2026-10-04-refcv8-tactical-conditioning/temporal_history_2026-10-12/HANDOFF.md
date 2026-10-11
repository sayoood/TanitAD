# HANDOFF -- temporal-history experiment (written 2026-10-12, EvalFlyWheel)
State: G3 verdict DONE and staged (RESULT.md). The PREREG s6 O3 follow-up (full S6, 3,668 windows, same arms/analysis) is RUNNING in the background on the dev box GPU:
`E:/hold_temporal/work/run_full.sh` -> chains A (f_real_s0, f_nt), B (f_real_s1, f_hold), C (f_st) -> then `hist_analyze.py` -> `E:/hold_temporal/work/analysis_full`.
ETA ~1.7-2 h from 01:30 local. DONE when `E:/hold_temporal/work/chain_status.txt` contains `ZZFULLDONE` and `analysis_full.log` ends `ZZHISTANALYZE_DONE`
(per-arm roll status lines `ZZDONE_f_*_1ZZ`; a `_0ZZ` means that roll produced no manifest -> look at `roll_f_*.log`, likely OOM: 3 concurrent rolls use ~7.6 of 8.2 GB).
Then, with `PYTHONPATH=E:/hold_temporal/tip/stack;E:/hold_temporal/tip/taniteval`, `PYTHONIOENCODING=utf-8`, venv `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`:
`python code/make_tables.py E:/hold_temporal/work/analysis_full FULL-S6 > raw/tables_full.md`; copy `analysis_full/{cross_paired_*,lead_viol,verdict_s6}.json` to `raw/full_*`.
Read the PREREG s5 verdict in `analysis_full/verdict_s6.json`; apply s6 (O1 / O2 / O3) UNCHANGED -- no margin or metric edits after data. Report it as a SEPARATE section (G3 stays the registered primary).
If the roll crashed: rerun only the failed arm with `E:/hold_temporal/work/run_chain.sh "<arm>:<seed>:<tag>:s6full_windows.json"` (arms real/hold/st/nt; seeds as in run_full.sh), then re-run hist_analyze.py with the five f_* dump dirs.
Do not touch Thor; nothing is staged in git; GPU gate: nothing else of ours is on the 4060 besides the desktop.
