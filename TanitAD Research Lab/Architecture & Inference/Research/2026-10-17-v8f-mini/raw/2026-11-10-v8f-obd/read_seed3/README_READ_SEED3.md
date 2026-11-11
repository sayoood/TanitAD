# OBD seed-3 BAND read (`AMEND_OBD_SEED3.md`, sha256 `a8a5848f22a3f164e4069bd7b302e7201cb05de5995494d05cf92dfde426e314`, verified)

Built 2026-11-11 BEFORE any seed-10 / seed-11 run exists. CPU only (`CUDA_VISIBLE_DEVICES=-1`, 2 threads), zero training. The existing read package `../read/` (README_READ.md, `obd_read.py`, `run_dumps_obd.py`, `obd_geom.py`, `RESULT_OBD.json`) is REUSED by import and was not edited. Nothing here was launched on a real bank. Tier, estimator and cells are the existing read's: T1-kin, m = 10, causal bucket, `taniteval.ci.paired_episode_cluster_bootstrap` B 2,000 seed 0, 24 cells, floor G = |OBD-BAND_rep - OBD-BAND_s0| per cell.

## Files
| file | role |
|---|---|
| `seed3_common.py` | the four arms (tag, base arm, `--seed`, init), `init_of_tag` (`_s11` is init 1; the existing `endswith("_s1")` test is not used), `argv_problem` (bank argv equals the base arm's except `--seed` = 10 / 11 and `--out`) |
| `run_dumps_seed3.py`, `job_dumps_seed3.cmd`, `launch_wmi.ps1` | dump driver: imports `../read/run_dumps_obd.py`, extends its tag table with the four banks, same tree, same `t1` dump folder, same OTBAND `run()`; skips finished arms first; refuses a missing ckpt / manifest, wrong tag, wrong commit, sha mismatch (existing refusals) and an argv that differs beyond `--seed` / `--out` |
| `obd_read_seed3.py` | the reader: floor G as the existing read computes it (cross-checked cell by cell against `floors.G` of `RESULT_OBD.json`), per-seed labels of the s10 and s11 pairs by `obd_read.label_band` (the card's rules), s0 / s1 labels read from `RESULT_OBD.json` (sha256 asserted `e05e1f5f...`, recorded), the combination rule, the recipe of record, four-family tables, beside rows; writes `RESULT_OBD_SEED3.json` + `.md` |
| `test_obd_read_seed3.py`, `test_run_dumps_seed3.py`, `tests/run_mutation_initclause.py` | tests and mutation runner |
| `smoke_seed3.py`, `smoke_map_seed3.json`, `SMOKE_OBD_SEED3.json` / `.md`, `SMOKE_COMPARE_SEED3.json` | the instrument control on stand-in dumps (SMOKE, never a result) |
| `READ_SEED3_PINS.sha256` | sha256 of every file here |

## Rule implemented (amendment lines 21-31)
Four per-seed labels s0, s1 (existing), s10, s11 (init M2b s0, s1, s0, s1). The band outcome is the label held by at least 3 of the 4 with at least one at each init, else **BAND-UNSETTLED**. A per-seed label that could not be read (arm missing, no floor) gives NOT-YET, never UNSETTLED. Trigger: BAND-ADDS -> OBD-BAND's argv is the recipe of record; BAND-NEUTRAL / BAND-COSTS -> OBD-CTL's argv, the operative gain recorded as not carried; BAND-UNSETTLED -> OBD-CTL's argv (conservative), the band's operative-only gain recorded beside with the code-drift caveat (`VERDICT_OBD.md` section 3). STRATEGIC is reported PAUSED (PI item 25).

Observation for the Master Mind (a property of the rule, not a change to it): with two seeds per init, any 3 of the 4 labels already contain both inits, so the clause "at least one at each init" cannot bite on the real layout. The reader implements the clause as written (`combine4` takes (label, init) pairs) and tests it on a planted unbalanced layout.

## Run sequence once the four banks exist (the Master Mind launches; nothing here was started)
Shell: Git bash. `PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe`. For every python step: `export PYTHONPATH="F:/TanitAD-scratch/otband_read/tree/stack;F:/TanitAD-scratch/otband_read/tree/taniteval" CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`.
0. **Preconditions.** `C:/Users/Admin/t2v8f_pulls/obd_<tag>/` exists for `OBD-CTL_s10 OBD-BAND_s10 OBD-CTL_s11 OBD-BAND_s11` (ckpt_last.pt, BANK_MANIFEST.json with the full commit `cdbff971...`, selfcheck.json, argv.txt); the read tree `F:/TanitAD-scratch/otband_read/tree` and, in `F:/TanitAD-scratch/otband_read/t1`, the dumps of `OBD-BAND_rep`, `OBD-BAND_s0`, `PARITY_DAGOACC_s0`, `PARITY_DAGOACC_s1` (+ their done files) are present (they were on 2026-11-11; CHECK); free memory >= 4 GB (the driver waits).
1. **Dump** (ESTIMATED about 10 min per arm, 40 min total, from the existing driver): `powershell -NoProfile -ExecutionPolicy Bypass -File E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/obd/read_seed3/launch_wmi.ps1`. Marker `DUMPS_SEED3_RUN.done`, log `run_dumps_seed3.log`, summary `DUMPS_SEED3_<stamp>.json`. A `REFUSED:` line in the log means that arm was not dumped; fix the cause (never edit a manifest) and run the same job again: finished arms are skipped.
2. **Read** (B 2,000, about 20-60 s): `cd` to the `read_seed3` folder, `$PY obd_read_seed3.py --out RESULT_OBD_SEED3.json --md RESULT_OBD_SEED3.md`. The last lines print the outcome, the four labels, the recipe of record and the trigger. `INVALID` lists the failed controls (bank manifest sha vs the dump's done file, selfcheck read-back, argv, floor G vs the existing result, identity control).
3. **Verdict.** `RESULT_OBD_SEED3.json` keys `outcome`, `band.outcome`, `band.labels`, `band.recipe_of_record`, `band.trigger`, `band.beside` (four labels, per-cell spread, code-drift sizes). Register the outcome in `GOALS_AND_CLAIMS.md` the same turn.

## Evidence class
Tests, mutation and SMOKE below: MEASURED (ours, CPU, 2026-11-11). The eventual `RESULT_OBD_SEED3.json`: MEASURED (ours; dumps, zero training), tier T1-kin. The estimator answers "another draw of episodes"; "another training run" is carried by floor G and by the four-seed rule itself.

## Controls and tests (MEASURED)
* **Tests: 44 passed** (`tests/pytest_clean.log`): `test_obd_read_seed3.py` 35 cases incl. all 81 label patterns against an independent oracle, the named 4-0, 3-1 (each seed as the odd one out), 2-2 and 2-1-1 outcomes, NOT-YET, recipe per outcome, floor G gating a separated sub-floor delta, floor cross-check, the `RESULT_OBD.json` sha pin (and its refusal when smoke / INVALID), beside rows, rendering; `test_run_dumps_seed3.py` 9 with fake banks (tag table, `_s11` is init 1, good banks, the existing refusals, the argv refusals, finished arm skipped, `subprocess.run` forbidden, job file order and marker, launch pattern).
* **Mutation** (`tests/mutation_initclause.json|log`, `tests/run_mutation_initclause.py`): flipping the clause turns 14 tests RED; dropping it (`each_init = True`) turns exactly 1 RED, the planted unbalanced-layout test (expected: see the observation above); the clean file is green.
* **SMOKE** (`smoke_seed3.py` -> `SMOKE_COMPARE_SEED3.json`, `SMOKE_OBD_SEED3.json|md`; B 2,000, 17 s): the existing real dumps OBD-CTL/BAND s0 and s1 stand in for s10 and s11. **336 of 336** compared fields (delta, lo, hi, floor, gain, cost, separated, 24 cells x 7 fields x 2 seeds) equal the existing `RESULT_OBD.json` bit for bit; floor G equals the existing `floors.G` (0 problems); the four labels read ADDS, COSTS, ADDS, COSTS and the combination is BAND-UNSETTLED, recipe OBD-CTL. This is a code check on stand-ins and a statement about the existing arms, never an OBD seed-3 result.

## UNVERIFIED
* Everything on a real seed-10 / seed-11 bank: that the four banks follow the existing layout (`BANK_MANIFEST.json` with `commit` = the full code-of-record hash, `selfcheck.json`, `argv.txt`), that their argv differs only in `--seed` / `--out` (the driver and the reader both refuse otherwise), that the OTBAND tree loads those checkpoints (same trainer and tree as the existing arms), and the per-arm dump time.
* `launch_wmi.ps1` / `job_dumps_seed3.cmd` were not launched (copies of the existing pattern with paths and tags changed; the job-file test checks tags, order and marker).
* The driver's own end-to-end `RD.run()` path on a real checkpoint was not run (only the refusal / skip logic, on fake banks and on the real banks' absence: the four tags read `REFUSED: checkpoint missing`).
* `selfcheck.json` key layout for the new banks (an absent key is a note, a present-and-wrong key is INVALID), as in the existing read.
* The code-drift row and the NULL-parity state are read from the existing `RESULT_OBD.json` only; the new arms add no NULL arm.

## Temporary data
Scratch only under `F:/TanitAD-scratch/obd_seed3_read/` (pytest basetemp, mutation copies), deleted at the end of the build. `F:/TanitAD-scratch/otband_read/t1` was read, never written.
