# HANDOFF_V44 (second agent, 2026-11-07 ~13:25 dev-box clock)

## Done (all under `E:/Projects/TanitAD-artifacts/dagland/report_v44/`)
* `sec_747_world_model.md` (first agent, untouched).
* `sec_748_renc_town.md`: 7.48.1 RENC (M-RENC-PASS, 3,000 steps, fa035232 stated), 7.48.2 C0C-RE (identity gate PASS cited as a dev-box file; T2 drive PENDING; card section 6 outcome table; 13:04 readings), 7.48.3 TOWN (MIXED; not comparable to 7.15 and why; TSWAP card as the named next step), 7.48.4 establishes / does not, 7.48.5 provenance. Banned-word scan 0, LF, no em or en dashes.
* `edits_v44.md`: blocks V44S, ABS1, NXT, S81, R83, ROW (rows 1bm to 1bt), I22, APP, C3A, C3B, C43S (2 hits in v43), C43B, C46A to C46D, NOTES. Every old text checked against the v43 REPORT.md copy (count 1, C43S 2). Scan 0.
* `check_numbers_v44.py`: 321 checks, all PASS (`python -I check_numbers_v44.py`). It also WRITES `NUMBERS_V44.md` (321 rows plus a derived / inherited / hypothesis table, 0 FAIL).

## Not done
* No script sweeps EVERY numeric token of sec_747, sec_748 and the edits against the sources (v43's `audit_v43.py` did). NUMBERS covers the numbers that decide a claim; interval brackets inside sec_747 tables were copied from sec_747 and only their point values are checked. A build agent should run an audit sweep.

## Flags for the Master Mind
* The tip moved during this run to `448a67b5` (TSWAP card landed 13:11:53, frozen prefix 3,642 bytes `fc775ca1`). It was not in the first handoff; 7.48.3, row 1bq and APP use it.
* MRATE (`c85eed44`) and LATSEM2 (`e8789391`) are on the tip; sec_747 7.47.7 and 7.47.9 still say "E: only". Update at the build (edits NOTES item 1).
* `G_ID_C0CRE.json` is the only cited file not on the tip. When the C0C-RE read lands it replaces the PENDING text (edits NOTES item 4).
* 7.15's 0.46 to 0.34 is `RESULT_V8F_T1_HEADS.md` (HACC, m = 10, causal bucket, tick 0 against ticks 1 to 2); TOWN's 0.40 is C0 own versus the logged twin: stated as not comparable.
* The RENC cosine is uncentered and the card has no mean-token companion: stated as a limit, not measured.
* Row 1bs says LATSEM2 "a read is running" (from the task brief); no source file states it. Confirm.
