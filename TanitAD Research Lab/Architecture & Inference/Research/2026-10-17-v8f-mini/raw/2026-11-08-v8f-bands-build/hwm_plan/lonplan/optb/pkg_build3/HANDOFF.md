# ACTRSP Option B build: HANDOFF 3 (2026-11-17 ~23:50Z, build agent 3, ended to avoid waiting on long jobs)

## Done (all under F:/TanitAD-scratch/actrsp/optb/)
* MR11-MR15 all RED, no control red (`mutation_results_optb.json`). Regression on the new tree: 89 files, 55 rc 0, 34 rc != 0 (`regress_new.tsv`, LAST entry per file; list in `failing_files.txt`).
* `basetree/` built (tree + 3 base files, no test_v8f_optb.py).
* Checker stage 2 applied (`pkg/optb_check.py`, compiles; stage 1 had a CR-corrupted `\readout` string, fixed). Unit test `pkg/tests/test_optb_check.py`: UNIT ok=119 bad=0.
* Chain package forked in `pkg/`: `optb_chain.sh` (ship, inputs, pilot, ten runs in checker RUN_ORDER, chassis-0 failure stops before chassis 1, per-run HF bank), `stages/optb_{common,ship,inputs,run,hfbank}.sh`, `optb_hfbank.py`, `optb_waiter.sh` (snap before launch, wait), `launch_optb_chain.ps1` (WMI 0x08000200; NOT run), `jobs/`, `optb_refresh.py`, `optb_ship_tree.py`, `frozen/` (argvs, export list, inputs/ probes), `README_OPTB.md`, `tests/` (dry run forked and adapted).
* `optb.patch` re-derived: identical to the earlier one (776 lines, sha256 4212b0e2...), tree unchanged.

## RUNNING DETACHED (WMI)
1. `regress2.py` (pid via `regress2.ps1`): per-test junit for the 34 failing files on `tree` and `basetree` -> `regress_pertest.json`; writes `regress2.done` at the end; log `regress2.log` (blocks on wait_commit 6 GB).
2. `dry_bg.sh`: waits for `regress2.done`, then runs `pkg/tests/test_optb_dryrun.sh` -> `dryrun1.log` (ends `DRY-FINISHED`; summary line `DRYRUN ok=<n> failed=<m>`).

## Next steps
1. When `regress2.done` exists: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe compare_regress.py` -> `regress_base.tsv`, `regress_compare.json`. Bar 0 regressions (fail on new, pass on base). For each, diagnose from `msgs`; fix only inside the 3 Option B files or test_v8f_optb.py.
2. Read `dryrun1.log`. The adapted dry run has NOT run yet: fix failures in the forked tests (not the checker rules) or in the new stages (inputs, hfbank, chassis stop). Copy the run's final line to `pkg/tests/DRYRUN_RESULT.log`.
3. Generate `pkg/PACKAGE_MANIFEST.sha256` (sha256 of every pkg file except tests logs, __pycache__, *.before_stage2, fork_*.py).
4. Append the section to `BUILD_OPTB.md` (census, mutations, regression counts, pkg, dry run).
5. Deletions (literal paths, log them): `F:/TanitAD-scratch/actrsp/optb/tmp` (incl. 144 gate_reuse_* dirs, bt_*), `.../optb/basetree`, `.../optb/base`, `C:/Users/Admin/AppData/Local/Temp/optb_dry_run_a`, `pkg/fork_check2.old_unused`, `pkg/optb_check.before_stage2`. Keep tree/, pkg/, optb.patch.
