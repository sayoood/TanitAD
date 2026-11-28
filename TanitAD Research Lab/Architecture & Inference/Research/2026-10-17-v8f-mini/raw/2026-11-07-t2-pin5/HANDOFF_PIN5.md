# HANDOFF PIN5 -- 2026-11-07 03:45. Agent PIN5 (Sonnet), dev box CPU. Nothing staged, committed, deleted. `code/read_t2_dominance.py` (pin 4, 7fd65ffd...) untouched.

## DONE (all under `motion_c/t2v8f/pin5/`)
* `read_t2_dominance_pin5.py` sha256 `56a91667df03523093ce9bb8b0e96282e9621919a8d63c97c58b1ca5574197de` (+ `.sha256.txt`, `PINS_PIN5.txt` for the card log, `PIN5_DIFF.txt` = diff -u pin 4 -> pin 5). Siblings are found in `code/` (`HERE`).
* `--arm-seeds {0,1 | 0}` (default `0,1`; ARM_SEEDS_FORM refuses any other). `0`: cells `<tc>|NAME_s0` only, outcome `SEED0-ONLY` (decision: n_seeds 1, counted_dominant false, statement "seed 0 only, not a counted DOMINANT", seed0_block), `seed_gap_beside = {computable: false, reason}`, GATE-FAIL kept for a failed gate.
  Default adds NO key to the json and no line to the md. Layout rule: stage 4 ran both seeds whatever is read (R-SCN unchanged); a stage > 4 under `--arm-seeds 0` must have run seed 0 only (124 cells), a two-seed stage there gives GATE-FAIL `RSCN_CELLS_PLANNED`.
* Identity on the REAL pulls (`run_real.sh`, the read of record's args): `logs/pin5_identity.json` identical **true** vs `stage4/score/DOMINANCE_T2.json` (unchanged `compare_dominance_reads.py`: decision, gates, both seeds' 19 rows, seed gap, r_scn, md equal); `logs/pin5_vs_pin4.json` identical true vs pin 4's read. Real seed-0 read: `score_seed0/` (SEED0-ONLY; s0 block equals the record's s0).
* Tests: `logs/pytest_pin5_final.log` 12 passed (8 synthetic `tests/test_pin5.py`, 4 real-data `tests/test_pin5_real.py`, 0 skipped); `logs/pytest_regression_on_pin5.log` 71 passed (frozen dominance, G2, R-SCN, arm Q, scene-exclusion tests run against pin 5 via `run_regression.py`).

## IN FLIGHT / NEXT (exact)
`run_mutations_pin5.py` (21 synthetic mutants + 2 real-data arms: seeds swapped in the default read; seed-1 records read as seed 0) runs detached with its scratch on E: (C: was at 0 bytes free, E: builds a synthetic world in about 45 s, so ~30 min):
`pin5/logs/mut_rc.txt` appears when done; read `pin5/logs/mutations_pin5.log`, expect `mutations RED: 23 of 23`, last line `ZZ PIN5-MUT-DONE ZZ`. If the process died (log stops growing), rerun: `cd t2v8f; PIN5_MUT_SCRATCH=<dir with room> python -I pin5/run_mutations_pin5.py` (about 3 min on a healthy C:).
First complete run (scratch C:, before the real arms were added) printed `mutations RED: 20 of 21`: the one GREEN was `seeds_swapped_in_the_default_read` against the identity test only (synthetic arrays ignore the records); it was RED against `test_main_arm_seeds_0_on_the_c0c_stage4_...`; its designation is now that test plus the real arm. The cap mutants (a seed-0 DOMINANT block naming DOMINANT-T2) and `seed1_records_read_as_seed0`, `g3_plan_loads_the_seed1_cells_silently` were RED.
Then: the MM enters the pin-5 sha in the card log (`PINS_PIN5.txt`) before any seed-0-only read is quoted.

## UNVERIFIED
No stage-5 package or `--seeds 0` driver exists: the 124-cell layout is the HANDOFF_LC assumption, tested on a synthetic world only. G4 (banked C0C arrays, both seeds) still runs under `--arm-seeds 0`: it is the reader's control, not an arm cell.
Cleanup for the PI: `E:/Projects/TanitAD-artifacts/scratch_pin5/`, `C:/Users/Admin/pytest_tmp_pin5/` (about 30 MB), `pin5/_aside_pycache/`.
