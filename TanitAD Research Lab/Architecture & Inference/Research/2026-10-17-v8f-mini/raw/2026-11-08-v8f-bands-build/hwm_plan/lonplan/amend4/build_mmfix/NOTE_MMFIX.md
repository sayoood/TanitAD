# AMEND_4 read: the two MM fixes to the readout's imports (2026-11-17)

The file landed in `build/amend4_read.py` is the builder's version. `build_mmfix/amend4_read.py` is the version that produced `readout/`. The planning code (`amend4_stage.py`, `amend4_run.py`, `l2img.py`) and every SCORE / TRAIN output are untouched by both fixes.

1. **Fix 1 (11:10Z).** `amend4_read.py train` raised `ModuleNotFoundError: hwm_eval`. The readout never put the code of record (`F:/TanitAD-scratch/actrsp/lonplan/build`) on `sys.path`; its sibling `amend4_run.py` does that through `lonplan_common`. The MM added `_code_of_record_on_path()` (lazy, before `import hwm_eval`). `test_amend4.py` passed 34. The resume (`run_amend4_resume.sh`) kept steps 0-2 and ran SCORE and readback for P / P23 / RS at both chassis (all rc 0).
2. **Fix 2 (15:10Z).** `amend4_read.py score` raised `ModuleNotFoundError: lonplan_common` at line 179. `score()` imported `lonplan_common` before the helper ran, because fix 1 placed the call before `hwm_eval` only. The helper is now the first line of `score()` and `sha()`. `test_amend4.py` passed 34, and a fresh process with the launcher's environment imports and hashes. `run_amend4_readout2.sh` ran the readout only and ended `ZZ AMEND4 READ DONE ZZ`.

Class: an entry point that lacks its sibling's path setup. Fix 1 put the setup at one import site and missed an earlier one. The probe after fix 1 called the helper by hand before importing, so it could not see the ordering. Fix 2 checks in a fresh process without a manual call.
