# HANDOFF -- D-R8-E1-REF (bars scored against ha0_ext_pose), EvalFlyWheel, state 2026-10-19 ~06:00 Berlin

PKG = `C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery` (unlanded qland hold on C:, NOT git-added, NOT committed; the orchestrator lands it).

## DONE (code final, tested on the primary tree)
* `code/compliant_reread_r8.py` (CLI of `read --tree T [--force]` UNCHANGED; chain_40k_r8.sh pid 1152752 calls it ~22:30): pose arm, `BAR-R8-E1/E3` vs `ha0_ext_pose` + `reference` field,
  `BAR-R8-E1_ha0_ext/E3_ha0_ext` beside, `os_minus_ha0_ext_pose` cell (ade_m), compliance arm `echo_ha0_ext_pose`, control `C4_pose_reference_valid` (inside ALL_PASS), new stages `series`, flag `--mutate-pose-to-ha0-ext`.
* `summarize_40k_r8.py`, `kdraw_e1_r8.py` (additive `pose_reference` block; pre-registered cells untouched), `test_chain_40k_r8.py` (+2 summarizer assertions), NEW `test_compliant_reread_e1ref_r8.py`, NEW `run_e1ref_reread_r8.sh`, `run_e1ref_all_r8.sh`.
* Pre-change snapshot (control b): `raw/compliant_reread/trees_pre_e1ref/` (10 tree jsons + MD5SUMS.txt, `assembled_pre/`, `code_pre/*.pre_e1ref`).
* EMA-30k first run: ALL_PASS, A0K0 control 8/8 cells exact, R0926 exact. (Superseded by the full run below, which re-reads with the frozen code.)

## RUNNING (detached, MSYS pid 1175054 `run_e1ref_all_r8.sh`; one heavy job at a time, FreeVirtualMemory >= 5 GB gate; ~45-60 min from 05:55)
Order: 10 tree re-reads -> mutation arm -> strict (EMA-30k) -> assemble + series -> pytest (e1ref, chain_40k, kdraw, a0k0).
CHECK: `cat raw/compliant_reread/logs/e1ref_series.log` -- markers `ZZE1REFTREEok|bad <tree>ZZ` (x10), `ZZE1REFSTAGE mutation|strict|assemble ok|badZZ`, `ZZE1REFPYTEST <file> :: <pytest last line>ZZ` (x4), terminal `ZZE1REFALLDONEZZ`.
Per-job logs: `raw/compliant_reread/logs/e1ref_<tree>.log`, `e1ref_mutation.log`, `e1ref_strict.log`, `e1ref_assemble.log`, `pytest_<file>.log`.
DO NOT edit `compliant_reread_r8.py` while it runs (the reads import it per tree).

## NEXT (fresh agent, after ALLDONE)
1. Any `ZZE1REFTREEbad`: read that tree's log; a C4/C1 failure prints which cell; re-run `sh code/run_e1ref_reread_r8.sh <tree>`.
2. Outputs to confirm: `trees/*.json` x10 (each `controls.ALL_PASS` true, `bars.E1_E3_reference`), `mutation/step30000ema_unc.json` (C4 PASS false = RED), `STRICT_CAP_step30000ema_unc.json`,
   `E1REF_SERIES.md/.json`, `COMPLIANT_REREAD.md/.json`; pytest counts from the markers (expect every file green; a red one: read `logs/pytest_<file>.log`).
3. Hand back to the orchestrator: the page builder `C:/Users/Admin/mmops/` reads the bar blocks -- NOT edited here; new keys: `bars.<compliant|all_window>.<capped|uncapped>.{BAR-R8-E1, BAR-R8-E3 (pose, "reference"), BAR-R8-E1_ha0_ext, BAR-R8-E3_ha0_ext}`.
4. Not done on purpose: `pair_read_r8.py` (the GPU battery's all-window uncapped PAIRED_READ) still states BAR-R8-E1 vs ha0_ext; it is C1/C1v's banked control input, so it is the registered read -- the ruling is implemented in the compliant instrument that decides.
