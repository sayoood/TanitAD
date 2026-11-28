# HANDOFF_KINLAB2 (build of `--dagger-label kin_v1`; phase 2 of KINLAB). Zero training, no GPU, nothing on the pod, nothing staged or committed.

**State: built, tested, censused. Waiting on the MM.** Folder `F:/TanitAD-scratch/kinlab/` (74 MB after cleanup).

## Deliverable (the MM lands it as ONE commit on `agent/v8f-rollhist-cdb-20261120`)
`CHANGED_FILES.txt` lists path, sha256, base blob at df250a00: modified `v8f_recovery.py`, `v8f_dagger.py`, `v8f_mini_train.py` under `tree/stack/`; new `tests/test_v8f_kinlab.py`, `tests/golden_v8f_dagger_df250a00.py`, `tests/golden_v8f_recovery_df250a00.py`. Wiring with file:line: `FACTS_KINLAB_2.md`. A trainer edit moves the launch-gate profile fingerprints: run the launch_gate tests before the land.

## Results (CPU, dev box)
* New file, whole-file: 30 passed (`kinlab_tests.log`). All six dagger/recovery/rolladd/kinlab files together: 233 passed, 2 skipped (`final_all_tests.log`; the 2 skips are the same as before the change, 203 passed + 2 skipped at df250a00 + this edit's trainer wiring, `TESTLOG_existing_regression.txt`).
* Gold: human swap and speed_v1 labeller / bank labels / swap equal df250a00 bit for bit on planted records, on the fixture bank, and (census run) on 4,000 real shard records.
* Mutants: 17 of 17 RED (`mutants/MUTANTS_RESULT.json`; T_B, w, kappa_o, a_o, merge, human fallback, identity rule, friction clip, A_LIM clip, flag not passed, labeller not run, feasibility check, pursuit sign, counters, probe).
* Census shard 0 (`KINLAB_CENSUS_2.md` / `.json`): relabelled 27.4 % of ok (**below the 0.5 mark; reported, design unchanged**), refused_infeasible 61.5 % (accel term 220,689 of 229,147), path_exhausted 5.2 %, not_merged 3.2 %, speed refusals 2.8 %, identity 0.

## Open for the MM (do not skip)
1. The merge tail as built returns the speed to speed_v1's profile in one step at the merge; 98.6 % of the accel refusals sit on that step (`KINLAB_DIAG_REFUSAL.json`). `FACTS_KINLAB_2.md` section "Interpretations" lists this and five more readings of the brief; none was tuned.
2. The G-LIVE probe now REFUSES a kin_v1 probe batch with 0 kin labels (TR:2900-2904). Intended; say if you prefer a softer rule.
3. Not run: the card's T1 / training; no replicate of the gold on the other three shards; the launch-gate tests after the trainer edit (the MM's step).

## How to re-run
`bash run_tests.sh <log> /f/TanitAD-scratch/kinlab/tree stack/tests/test_v8f_kinlab.py`; mutants `python mutants/run_mutants.py` (copies stack/ + taniteval/taniteval into mutants/tree, then delete it); census needs a fresh read-only pull (shard 0 of `/root/rollhist/bank/s0`, tracks of `/root/dscale10k/merged`) then `run_census.sh`.
