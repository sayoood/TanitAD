# HANDOFF: SEARCH-PESSIMISTIC Stage 0, amendment A1 (seed-averaged arms). Calibration RUNNING. (A1 build agent, 2026-10-22 15:14 Berlin)

**State.** A1 is built and tested in `C:/Users/Admin/par18/spess0a1/` (a copy of `spess0`; the withheld `spess0/raw/RESULT_STAGE0.json` was never opened or copied). The calibration (REAL features, SYNTHETIC labels only) is running. **The real read has NOT been run and must not be run before the steps below.**

| item | value |
|---|---|
| calibration launched | 2026-10-22 15:13:3x +02:00, detached via WMI, `run_calib_a1.cmd`, **12 workers** |
| PIDs | launcher `cmd.exe` **51512** -> venv stub `python.exe` **62080** -> master `python.exe` **39816** -> 12 workers (`multiprocessing.spawn`, no `calib_a1` in their command line) |
| log | `C:/Users/Admin/par18/spess0a1/raw/calib_a1.log` (one line per finished task, 112 tasks) |
| done marker | the LAST line is `ZZCALIB_DONE` (a crash prints `ZZCALIB_FAILED` and a traceback instead) |
| results | `raw/calib_a1.json` (written at the end), `raw/calib_a1.partial.jsonl` (appended per task) |
| ESTIMATED duration | **50 to 65 min wall** (about 16:05 to 16:20), up to 90 min on a loaded box |
| committed before launch | `DRAFT_AMENDMENT_A1.md` §4 (selection rule); hashes in `raw/launch_record.txt` |

**Check it:** `tail -3 raw/calib_a1.log`; `grep -c " done " raw/calib_a1.log` (of 112); `tasklist /FI "PID eq 39816"` (master). A task that raises is logged as `ERROR` and carried in `errors`; an S with any errored task cannot qualify. There is no resume: if the job died, relaunch `run_calib_a1.cmd` (same detached WMI command, written in its header); the partial file is for salvage only.

## What is built (all tests CPU, synthetic data)
* `code/stage0.py`: `N_SEEDS` (= 1 now, placeholder) and `REPL_SEED_BASE = 100` in the digested constants; `fit_all_seeds`, `ens_score` (mean of member COMPOSITE scores), `seed_sets()` (fails closed on overlap); every arm, control and the planted/mutated arms average S models. Working digest (S = 1): `529fc02d…`; digests for S = 3 / 5 / 8: `raw/constants_digest_by_S.json`.
* `code/calib_a1.py`, `run_calib_a1.cmd`; `code/golden_digests.py` (+ `raw/golden_unamended.json`, from the SEALED code).
* Tests: **74 passed** (`raw/pytest_final.txt`; 56 sealed-suite tests kept, 18 new `test_A1_*`). Mutation census on the changed lines: **33 of 33 killed** (`raw/MUTATION_CENSUS_A1.json`; one survivor there was a skipped-test artefact and was killed on re-run, `raw/MUTATION_CENSUS_A1_survivor_rerun.json`).
* **Identity at S = 1: bit for bit.** Synthetic worlds vs the sealed code: all digests equal. Real features, the four planted worlds vs the sealed run: max |diff| = 0.0 (`raw/preflight_identity.json`).

## Exact steps after `ZZCALIB_DONE` (fresh agent, Sonnet is enough)
1. `raw/calib_a1.json`: require `n_errors == 0` and `identity.pass == true`. Read `selection.verdict` (the script applied the committed rule) and check it by hand against the per-S table printed in the log.
2. **If `SELECT S=k`:** set `N_SEEDS = k` in `code/stage0.py` (line 49); `python code/stage0.py constants` gives the new digest (equal to the entry in `raw/constants_digest_by_S.json`). Finalise `DRAFT_AMENDMENT_A1.md` as `PREREG_STAGE0_A1.md`: replace the working digest by the chosen one (the test `test_draft_amendment_quotes_this_codes_constants_digest` enforces it), write S into §2, paste the calibration table, add the seal block. `sha256sum PREREG_STAGE0_A1.md > PREREG_STAGE0_A1.sha256`. The MM seals; then `run_stage0.cmd` (it runs the suite, then the read; ESTIMATED 4.5 to 11 min at S = 3 to 8, `DRAFT_AMENDMENT_A1.md` §6) and writes `raw/RESULT_STAGE0_A1.json`.
3. **If `INSTRUMENT-LIMITED`:** no real read. Route: `D-SESSION-RESERVE` (PI) or the decoder-level variant (~1.5 GPU-h).
4. Either way: update `D-REFE-SESSION-EXPOSURE-1` and `GOALS_AND_CLAIMS.md` in the same turn; keep `spess0/raw/RESULT_STAGE0.json` unread until the amended read concludes, then bank it.

## Integration (escalated)
**MM: land `C:/Users/Admin/par18/spess0a1/` with the scoped lander.** Nothing is staged, committed or pushed; it exists in ONE place (dev box).
