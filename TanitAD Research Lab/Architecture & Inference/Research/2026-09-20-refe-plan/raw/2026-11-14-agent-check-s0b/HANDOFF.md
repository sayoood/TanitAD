# HANDOFF S0-B (2026-11-14) -- built and smoked, FULL RUN NOT STARTED, no result read
State: all scripts done. Tests 15/15 green, mutations 7/7 RED. s0b_cid smoke n=2 PASS (cell agreement min 0.9996, truth nc/ttc reproduced exactly, MEASURED, cid_smoke_n2.json).
s0b_run smoke (1 log, 5 sets, counts only) OK: 0.224 s/set -> ESTIMATED 0.26 CPU-h full. Full-mode code path self-tested on one log (--fullpath, numbers not read, output deleted).
Artifacts (all in this hold): s0b_raster.py s0b_readout.py test_s0b.py s0b_cid.py s0b_run.py covered_sets.txt found_logs.txt FREEZE_S0B_DRAFT.md smoke_report.json cid_smoke_n2.json ref/ (tip 99ffcd3a exports: sel_common.py, refe/navsim_*.py, navtrain_scenarios.py).
NEXT (MM): review FREEZE_S0B_DRAFT.md, freeze, then run its 3 commands in order (cid n=64, then s0b_run with --cid-json). Do not start before the freeze.
Deviations / contradictions named:
- Covered set = 3,958 sets / 47 logs (1,343 held-out), not 3,976 / 48 / 1,361: log 2021.06.23.14.54.32_veh-16_00862_01000 (18 sets) had its DB only in live download staging, gone now; per instructions staging is ignored.
- Task text "statics held fixed" vs NAVSIM code: NAVSIM interpolates static centres like the others (only velocity 0). Mirrored the code.
- A float claim "15*0.1=1.5000000000000002" was checked and is FALSE; time grid is NAVSIM-literal anyway.
- Card C-ID said pod tokens; done on dev box per decisions. F: nuplan-devkit needs pytest (absent): C-ID uses the venv nuplan (DriveRL fork) + navsim 3e8291b.
- Outcome row computed for both columns; MM picks. C-SHUF "gain > 0 => INVALID" is applied to the point estimate exactly as written.
Deletions: scratch C:/Users/Admin/s0b/work/{repo 64 KB, tmp 0, smoke 4 KB, cid1 4 KB, cid2 4 KB, fullpath 32 KB} removed by literal path (own data).
