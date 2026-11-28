# HANDOFF_V45_BUILD (text agent, 2026-11-08). UPDATE (build agent, tip c69bc890): the document IS built (`out/`), PLANTEST (7.50.7, row 1ca) and the build facts (7.50.8) are inserted, the checks pass; run `bash build_v45.sh <tip>` to rebuild. The text below is the original text-agent handoff. Nothing staged, committed, pushed or branch-switched. CPU only.

Folder `E:/Projects/TanitAD-artifacts/dagland/report_v45/`. Python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`. Base: the landed v44 report (`R/report/2026-11-07-v44/`, built at tip `0423bb9d`; md at `E:/Projects/TanitAD-artifacts/dagland/report_v44/out/V8F_SCIENTIFIC_REPORT.md`, sha256 `a62b4327bb80149162d7ac462e00552f12e0a67903ceec3399c615d0693d829d`). `R` = `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/`.

## What is here
| file | what |
|---|---|
| `sec_749_imagination.md` | section 7.49 (what the world model imagines): VOE1B, IMAG1B, CAPLADDER, IMAG1C; T0; leads with the measured positives, then what did not work |
| `sec_750_horizon_bands.md` | section 7.50 (horizon bands, audit, per-band baseline, redesign as designed-not-trained); T1-kin baseline; STRATEGIC PAUSED |
| `edits_v45.md` | blocks V45S, ABS2, NXT, S82, R84, ROW (rows 1bv to 1bz), ROW1BU, I22N, R83N, IMPL, I23, APP, C3A, C3B, NOTES (same format as `edits_v44.md`: old text, a line `-->`, new text; insertion blocks have no old text) |
| `NUMBERS_V45.md` | written by `check_numbers_v45.py` (391 checks plus a derived / inherited / hypothesis table) |
| `check_numbers_v45.py` | `python -I check_numbers_v45.py`; exit 0 at writing, 391 checks, 0 FAIL |

## Build steps for the Master Mind (from v44; the v44 scripts are the template)
1. Copy `report_v44/{patch_v44.py,audit_v44.py,checks_v44.py,build_v44.sh,render_check_v44.js}` to `report_v45/` as `*_v45.*`. Change in them: the folder (`report_v44` to `report_v45`), the base md (`report_v43` landed copy to the v44 md above, base sha above), the section files (add `sec_749_imagination.md`, `sec_750_horizon_bands.md` after `sec_748_renc_town.md`), the edits file, `check_numbers_v44.py` to `check_numbers_v45.py` (and `NUMBERS_V44.md` to `NUMBERS_V45.md`), the version words (v44 to v45, date 2026-11-08), the expected base-sha check, the "only 8 carried lines changed" check (v45 changes the carried lines of blocks C3A, C3B, NXT, ROW1BU, I22N, R83N and IMPL: 7 blocks, IMPL on 3 sites, so count the sites, not the blocks), and the section-order check (7.49 and 7.50 after 7.48, before section 8).
2. Apply the blocks of `edits_v45.md` exactly as `NOTES` says. Anchors were each checked against the v44 md at writing (counts: ROW1BU 1, I22N 1, R83N 1, NXT 1, IMPL 3 (replace all three), C3A 1, C3B 1; insertion anchors `\n\n---\n\n## 8. Discussion` 1, `**Update (v44).**` 1, `### 8.2 What is not established` 1, `\n* **What is next.** The crawl set is still open` 1). The first `Nothing in v44 is a driving result.` (status line) takes V45S; the second stays.
3. Replace `{{TP}}` by the full 40-hex tip at the build. The v44 tip text is the literal `0423bb9d` (already in the edits).
4. Run `python -I check_numbers_v45.py` (must exit 0, 391 checks at writing; it reads the E: dev-box sources directly, so no tip is needed). Then run the v44-style audit sweep (`audit_v44.py` adapted) on the added lines: every number in the two sections should be found in its source; the scopes are the eleven files listed in `NUMBERS_V45.md`.
5. Run the standing scans: the banned-word list of the brief, the phrase for the looped-driving tier, em and en dashes, the word `drives`, CR characters. `check_numbers_v45.py` already runs these on the two sections and the edits (all 0 at writing).
6. Land as `R/report/2026-11-08-v45/` with the same layout as v44 (md, html, DIRECTIVES from `out/`; scripts, sections, edits, NUMBERS, checks, MANIFEST, handoffs from this folder). All source results are already on the tip with register rows (MM checked 2026-11-08: H-V8F-VOE-1, H-V8F-IMAG-1, H-V8F-CAPLADDER-1, H-V8F-IMAG-1C, D-HWM-BANDS-1, D-V8F-BAND-BASE-1). When the PLANTEST row goes in, the MM adds its own landing commit and claim row to the APP table.

## PLANTEST insertion point (MARKED PLACEHOLDER; no PLANTEST statistic is written anywhere in this folder)
PLANTEST (`MC/theory/plantest/`) was still running at writing; only `RESULT_PLANTEST_controls.md`, `IMPL_CHOICES_PLANTEST.md`, `HANDOFF_PLANTEST.md` and `TEST_PLANTEST.done` existed. When its result file exists:

`<<<PLANTEST-INSERT-BEGIN>>>`
* New subsection **7.50.7 PLANTEST (the required gate for a world-model planner, SPEC_HWM section 8)**, placed after 7.50.6, written from `plantest/RESULT_PLANTEST.md` only: tier of the read, estimator, controls, all four families (STRATEGIC PAUSED), the result with its intervals, and what it does not show.
* Edit 7.50.5 "Does not establish": replace the sentence "the PLANTEST read is not in this section" by the actual statement.
* Edit the `Not run` bullet of I23 and row 1bz in `edits_v45.md` ("PLANTEST is not part of this row").
* Add a row 1ca to the ROW block, and the PLANTEST folder to the APP table.
* Add its numbers to `check_numbers_v45.py` (source string AND text string, as the other blocks do) and rerun until exit 0.
`<<<PLANTEST-INSERT-END>>>`
The build must not ship while the text still says PLANTEST is "not part of" the report if the result has landed; if the result is not ready at build time, ship as is.

## Source conflicts and open points (not decided in the text)
1. **LATSEM2B (v44 7.47.8) against CAPLADDER / IMAG1C**: lead gap "encoded at no level" (TRAIN windows, 139 episodes) against readable at levels 2 and 3 (eval-141 frames with a lead, ridge probe). Both are printed; 7.49.6 and I23 state it; no bridging read exists.
2. **v44 7.47.5 against IMAG1B**: level 3 decodes to a future speed better than persistence (+0.233 / +0.223 at 2.4 s) against IMAG1B's pooled decision for speed COPIES at all nine cells. Different references and rules; both printed.
3. **IMAG1C versus IMAG1B lead_gap**: IMAG1B 9 of 9 UNREADABLE, IMAG1C 3 UNREADABLE / 6 COPIES; IMAG1C's attribution table shows the probe, not the frames, accounts for the change; stated in 7.49.5.
4. **Eval label count**: `AUDIT_DATA.md` notes the eval label file holds 147 records and the census says 141 clips; unreconciled by the audit; carried into 7.50.6 and I23.
5. **v44 7.47.9 / row 1bu** say IMAG1 and VOE1 are "running" or "no result". The reads that ran are VOE1B, IMAG1B, CAPLADDER, IMAG1C; IMAG1 and VOE1 themselves have no result file in the folders I read. ROW1BU, I22N, R83N and IMPL edit the v44 sentences accordingly.
6. `RESULT_CAPLADDER.md` says its weight check was skipped (`ModuleNotFoundError`): disclosed in 7.49.4 and I23.
7. `AUDIT_MODEL.md` open item 1 (does video extend beyond 20 s, "HYPOTHESIS no") is answered by `AUDIT_DATA.md` (video ends at 20.17 s, MEASURED); the text uses the data audit.
8. The three label-change percentages (5.3 / 7.3 / 15.4 %) are INHERITED in the source and marked [I] in 7.50.2.

## Evidence discipline
Every number in the sections carries its arm (M2b s0 / s1, DAG-OACC s0 / s1, P10), source path (the section's provenance paragraphs), class and tier (T0 for 7.49, T1-kin for the 7.50 baseline), and every interval its estimator (episode-cluster bootstrap B = 2,000 seed 0; refit permutation R = 199 for the two-part rule; paired for arm minus P10). T0 is never compared with T1-kin in the text. No exponent is quoted. No DOMINANT or driving claim is made.

## Deliverable manifest
All single copies on E: (nothing on a pod, staged or pushed): `E:/Projects/TanitAD-artifacts/dagland/report_v45/{sec_749_imagination.md, sec_750_horizon_bands.md, edits_v45.md, NUMBERS_V45.md, check_numbers_v45.py, HANDOFF_V45_BUILD.md}`. The sources are the dev-box artifacts under `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/` (`voe1b/`, `imag1b/`, `imag1c/`, `capladder/`, `horizon_review/`, `horizon_review/band_eval/`), also single copies and not on the tip.

## Deleted (own temp data)
`E:/Projects/TanitAD-artifacts/dagland/report_v45/run1.log` (about 60 KB, a re-creatable check log) and the three patch helper scripts `C:/Users/Admin/AppData/Local/Temp/fix1.py`, `fix2.py`, `fix3.py` (under 5 KB each, re-creatable, nothing depends on them).
