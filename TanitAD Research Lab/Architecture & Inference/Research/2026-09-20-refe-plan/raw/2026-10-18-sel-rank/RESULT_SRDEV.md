# SRTTC-DEV: SR-TTC's dev gate, the navtest C2 reading, and the four families on the confirm seams

Date 2026-10-19. Brief `C:/Users/Admin/par18/srdev.md`. CPU only, one Python process at a time (one exception, section 8). No dev outcome and no navtest outcome was opened. No landed file, `refe/`, or `C:/Users/Admin/srttc_s0/` was touched: a byte comparison of both landed code directories against the tip export found 0 differing files and only the new files below.

Evidence classes: **MEASURED** = our run, log under `raw/`. **INHERITED** = read from another document, not re-run. Every test below ran on SYNTHETIC worlds unless it says "real inputs (header or shape only)".

## 1. What was built (all NEW files, LF, `-I` isolated runs)

| # | file | repo dir (REFE = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`) | lines | tests | mutation markers |
|---|---|---|---|---|---|
| 1 | `srttc_dev.py` (SR-TTC dev gate) | `REFE/raw/2026-10-18-sel-rank/code/` | 1052 | 72 OK (`test_srttc_dev.py`, 1395 lines) | 44 |
| 1 | `srttc_c2.py` (SR-TTC C2 shuffled-label fit) | same | 134 | in the 72 | 3 |
| 1 | `srttc_dev_inputs_check.py` (real inputs, header/shape only) | same | 91 | ran: all_pass | n/a |
| 2 | `navtest_c2.py` (navtest C2: `picks` before the look, `score` after the ledger) | `REFE/raw/2026-10-18-ep-r1/code/` | 277 | 29 OK (`test_navtest_c2.py`) | 11 |
| 3 | `confirm_families.py` (families6 on every confirm reading) | same | 420 | 25 OK (`test_confirm_families.py`) | 9 |
| 2+3 | `confirm_read.py` (the confirm READ entry point: order, post, verify, final report) | same | 226 | 20 OK + 1 gated, run once (`test_confirm_read.py`) | 9 |
| seam | `test_dev_to_confirm_seam.py` (REAL dev results feed the confirm) | same | 90 | 3 OK | n/a |

Every marker is mutated by a test (one source line edited, the check must go RED while the unmutated module is GREEN); a coverage test in each module fails if a marker is not mutated. Logs: `raw/srttc_dev/`, `raw/srdev/`. Code hashes: `code/CODE_SHA256_SRDEV.json` in each code dir.

## 2. (1) `srttc_dev.py`: SR-TTC's OWN dev gate

* Bar, each term with its file:line (the trace is in the result JSON under `registration_trace`, and `test_every_cited_registration_line_says_what_the_code_says_it_does` opens each cited line in the registered documents and requires the phrase): total paired gain >= +0.30 PDMS x100 on BOTH seeds, point estimates (`PREREG_SELRANK_ARM.md:93`); controls C0-C2, C4 (`:69-73`), C3 = two seeds, C5 = paired log-cluster bootstrap B 10,000 seed 20260927 (`:74`); c* and its sha256 written first (`:8-10`); the LC call before the gate, read iff FLAT or f = f_max (`amendment2.md:169-177,182`); the gate set and its four admissibility conditions (`amendment2.md:245-248`); outcome rows (`:98-103`, `amendment2.md:206-207`).
* Contract for `refe_head_confirm.read_dev_result`: `dev_gate.verdict` PASS/FAIL/VOID, `inputs.rows_sha256` per seed, `inputs.r1_result_sha256`. MEASURED: the landed reader accepts the output and refuses 5 corrupted copies; `load_arm_rows` binds the rows the gate applied and refuses a row swapped after the gate.
* One read: the marker `srttc_dev_READ_STARTED.json` is written exclusively BEFORE `pdms`/`sub` are opened; a crashed first attempt still consumed the read. A RED control before the outcomes refuses and leaves the read unspent. A table with no outcome members is used by every refusal test, so an early read would be a KeyError, not a refusal.
* Identity arm (MEASURED): the unchanged TTC row gives P2's pick on every token (bit-identical logits), gain exactly 0.0; and a no-op row makes C1's applier-side reach RED (with a negated-row positive control).
* Real inputs, header/shape only (MEASURED, `raw/srttc_dev/srttc_dev_inputs_check.json`): the three dev files hash to the landed pins; 1,705 tokens / 24 logs; the outcome members exist with shapes (1705,64), (1705,64,6), (1705,64), (6) read from the zip headers only; P2's head reproduces the stored logits within 1.45e-3 (a next-token control reads 15.2); C0 1705/1705; C4 proposals max abs diff 0.0 m; A2.10 all four conditions hold on the landed attestations.

## 3. (2) Navtest C2

Mirrors dev C2: the SAME shuffled-label rows, point gain <= 0.0 on EVERY seed of an arm, no tolerance (MM D7), same bootstrap draws as the real readings. `picks` (no outcome, before the look): re-validates the look gating, binds each C2 row to the one the DEV gate applied (`inputs.c2_result_sha256`, `inputs.c2_rows_sha256`), runs the landed bank pass with the C2 readings in the plan; a RED control refuses with the look unspent. `score`: its first act refuses when the ledger is absent (MEASURED: the outcome opener is never called), binds the ledger to the picks (table, bank manifest, the real rows, P2's pick), then opens the outcomes. Identity: C2 rows equal to P2's row are refused before the look.

## 4. (3) Four families

`confirm_families.py` calls `families6.py` exactly as `stage3_full.compute_families` does (argv equality is tested against its source) on every seam the landed read wrote; the tool is pinned (sha256 5dfd0cf1..., identical at the tip, on D: and on E:). LONGITUDINAL (three target-speed bands, speed MAE/bias, along-track, acceleration, intervals; distance-keeping UNAVAILABLE with the tool's reason and n = 0), LATERAL (heading, curvature, yaw-rate, cross-track, intervals), TACTICAL (lateral/longitudinal/5-way accuracy and kappa, goal-setting), STRATEGIC printed PAUSED with the PI ruling and `is_a_pass: false`. A missing item is MISSING and makes the reading incomplete; a reading the tool did not produce is NOT_COMPUTED, never a pass. The registered clause (`PREREG_EPR1.md:110`, `PREREG_SELRANK_ARM.md:94`) is `vs_p2.clears` = nothing separated adverse AND at least one component compared (zero compared = not evaluated). `fam_adverse` is ported from `proxy_eval.py` with an equivalence test that extracts the original from ITS SOURCE and compares 20 ordered pairs of banked families JSONs plus 200 random hand-built pairs: identical. The banked P2 families JSON reads as a complete report (control). The REAL tool and NavSim adapter run in the tests on synthetic seams.

## 5. The entry point and the verdict

`confirm_read.py read`: preflight (ledger absent; families tool resolvable) -> C2 picks -> landed `stage_read` (spends the look) -> C2 score -> families -> `confirm_final_report.json` (exclusive, last). MEASURED order test: the C2 picks run with the ledger absent; the C2 score and families only with it present. A failure after the look raises `PostLedgerError` ("THE LOOK IS SPENT"); `post` resumes; `verify` lists what is missing and checks the final report is bound to the files it summarises. The candidate verdict is the registered bar applied mechanically (INCOMPLETE > CONTROL_RED_RULING_NEEDED > CLEARS_ALL_REGISTERED_CLAUSES > DOES_NOT_CLEAR); the Master Mind signs it. Real-machine resolution (MEASURED, `raw/srdev/tool_resolution.json`): families6, the inputs export and the interpreter resolve with no overrides.

## 6. Gaps: the registration is silent, so nothing was invented

| id | gap | handling |
|---|---|---|
| G1 | SR-TTC has no registered C2 shuffle seed | `srttc_c2.py` uses 20261018 (EP-R1's value), recorded in the result |
| G2 | dev C2 has no tolerance and no replicate: a null fit has a gain of random sign, so C2 can read RED by chance and burn the one dev read | applied as written (D7); flagged. A false-RED rate was NOT measured here |
| G3 | the registration does not say whether a VOID after the outcomes opened consumes the one dev read | the attempt consumes it |
| G4 | `amendment2.md:207` calls non-FLAT + DEV-FAIL "only reachable when f_max < 100 %"; at f_max = 100 % it is reachable | not applied, surfaced in `outcome_rows.notes.a2_6_row3` |
| G5 | the launch-gate token's commit binding needs git | HMAC, verdict, profile, stage, fraction, srttc_r1.py and bank-manifest bindings are re-checked; the commit is not |
| G6 | no grid-edge rule in the SR-TTC documents | `c_star_at_grid_edge` reported only |
| GN1 | no navtest C2 rule or consequence is registered (`PREREG_EPR1.md:152` names "C2 and C4 reads" only) | mirrors dev C2; a RED C2 is reported as CONTROL_RED_RULING_NEEDED, never applied |
| GN2 | the C2 picks cost a second bank pass and the C2 score a second outcome open (after the ledger) | disclosed; no real reading changes |

## 7. Integration asks (escalated in the hand-back)

1. Use `confirm_read.py read` as THE read command. The landed `refe_head_confirm.py read` does not call the C2 or families pieces (its output says `families6: PENDING`); run directly it spends the look without them (`post` and `verify` recover and disclose it).
2. `srttc_r1_gate.py` has no `c2` stage, so `srttc_c2.py` needs `--no-gate-stage-ack`: add a c2 stage or rule that C2 needs no token (the dev gate requires the C2 files).
3. Rule GN1 (consequence of a RED navtest C2), G2/G3/G4 before the dev read, not after.
4. Distance-keeping stays UNAVAILABLE until lead-agent tracks reach `families6.py` (work item, `build_lead_tracks.py`).

## 8. Honest limits

* The subprocess runner (the production call) was executed ONCE (30 s, log `raw/srdev/test_confirm_read_subprocess.log`): parent and child were alive together, so that run broke the one-process rule for 30 s (the child ran the real families6 tool and the test passed). All other runs were one process.
* Nothing here has seen a real SR-TTC row, a real dev outcome or a real navtest outcome. `srttc_dev_inputs_check` could only run the no-outcome phases; navtest C2 and the families have no real-input check beyond tool resolution.
* The tests' false-RED rate for C2 on the real tables is unknown (G2).

## 9. Proposed register rows (not edited; ids assigned by the MM on the tip)

* `D-SRTTC-DEV`: SR-TTC dev-gate applier BUILT against SR-TTC's own registration (cites in `registration_trace`); 72 tests, 47 mutation arms; not yet run on rows (S0 running). Next lever: S0 -> R1 -> LC -> `srttc_c2.py` -> the dev gate.
* `D-NAVTEST-C2` and `D-FAMILIES-CONFIRM`: D8's two launch blockers BUILT; the read entry point is `confirm_read.py`.
* `R-GN1 / R-G2 / R-G3 / R-G4`: four registration silences awaiting a ruling (section 6).
* Class note for `RETRACTION_LOG.md` (not a retraction): a fifth instance of "built, tested, unreachable from its caller" was avoided by construction: the landed read did not call D8's pieces; `test_dev_to_confirm_seam.py` feeds REAL dev results to the confirm.

## 10. Rule Zero

Nothing is refuted here; the work is built. What blocks the next step: S0 (running), then R1/LC/C2 on the S0 bank, then the PI/MM rulings in section 7.3. Everything else in the three priorities is delivered.

## 11. Deliverable manifest

All files are staged under `C:/Users/Admin/par18/srdev/out/` mirroring the repo path; `out/paths.json` lists `[repo path, absolute source, "raw"]`. Each exists in ONE place until the Master Mind lands it. `out/RESULT.md` is a copy of this file.
