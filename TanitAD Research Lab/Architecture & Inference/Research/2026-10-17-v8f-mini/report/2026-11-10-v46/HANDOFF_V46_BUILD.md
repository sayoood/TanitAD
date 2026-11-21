# HANDOFF_V46_BUILD (build agent, 2026-11-10). Nothing staged, committed, pushed or branch-switched. CPU only. PDF dropped by the Master Mind (no PDF, no PDF_CHECK.json).

Folder `E:/Projects/TanitAD-artifacts/dagland/report_v46/`. Rebuild: `bash build_v46.sh <40-hex tip>`. Built at tip `47a3eeac8e10cea99f60e19c1fd6569f58270166` (the live remote tip at build time; a descendant of `5bb4400648c61d9496cb5e7cae004fd428bbb7ff`, the tip of the paper v1.8 the two sections come from; the only commit between them is "TanitAD-4B paper v3"). Exit 0, audit 0 unmatched, 23 checks PASS, `check_numbers_v46.py` 92 checks PASS with 0 UNSOURCED.

## Base and result
* Base: `report_v45/out/V8F_SCIENTIFIC_REPORT.md`, sha256 `324c200158fe27decacccb4aac1fbf2e0b6e1711fb34fb8b213ab32e556218b0` (pinned in patch and checks; the v45 html is asserted equal to the page template with that md).
* Out: `out/V8F_SCIENTIFIC_REPORT.md` and `out/v8f_report.html` (template `page_template_v30.html` at the tip with the md). Offline render: 315 tables, 68 h3, 0 ragged tables, 0 stray placeholders.
* Insertions: sections 7.51 and 7.52 after 7.50.8, before 8; status line (v46 of 2026-11-10); abstract bullets (two) plus a sentence on the What-is-next line; 8.1 item 36; 8.3 Update (v46); 8.4 rows 1cb to 1ce; 9 item 24; appendix block "Reads added in v46".
* Carried v45 lines changed: exactly 5 lines (status line with C3A/C3B/V46S; abstract What-is-next line; row 1bz; the Update (v45) "What is blocked" bullet; item 23 "Not run" bullet). Sections 7.49 and 7.50 are untouched dated statements.

## Section files (edits allowed by the brief only)
Originals in `orig_sections/`. Changes: the leading HTML comment line removed (the patch forbids raw markup); cross-references: `§7` to `§5.3` (H-ESTIM-SEED-1), "The cascade of §7.50 states it" to "The PI's rulings of §7.50.8 state it (R-T2O, SPEC_HWM.md section 9.1)", "R-OUT, §7.50" to "R-OUT, §7.50.1"; tags [M] on the bold lead-ins and [E] on the 35 % / 45 % sentence. No number or claim changed.

## Number findings (for the Master Mind)
* No number failed and none is UNSOURCED. Two doubts resolved: `0.702` (7.51 R^2) is in `decinfo/decn/fulln/RESULT_FULLN.md` (s1, best of four readers; PT3F's s1 ridge is +0.696); `0.0390` is the FLOOR_TABLE value 0.039 with a trailing zero.
* NOTE 1: the text says the band loss's operative ADE gain is "a fifth to a quarter" of the acceleration term's; the ratios are 0.19 (s1) and 0.27 (s0), so s0 is slightly above a quarter. Left as written.
* NOTE 2: "five to twelve times" matches the source's printed 3-decimal values (0.287 / 0.060 = 4.8, 0.287 / 0.024 = 12.0); on the unrounded 0.2869 / 0.0235 the upper ratio is 12.2.
* NOTE 3: 7.51 (i) says "Nothing here is driving performance"; the v45 wording guard on `driving performance` is relaxed for exactly that negated phrase (patch and checks), nothing else.
* The planner-gate and ladder card hashes are not restated in the report (the edits name the cards only).

## Not done / open
* PDF: dropped by instruction.
* The "landed" column of the appendix holds the latest commit touching each folder at tip `5bb44006`; the register rows exist at that tip.
* `check_numbers_v46.py` reads the register from tip `5bb44006` (constant `TIP`) and the dev-box E: sources otherwise; a later register edit is not re-read.

## Deleted (own temp data)
`F:/TanitAD-scratch/report_v46_build/` (about 6 MB: prototype and fix scripts, a register copy, a trial build copy; all re-creatable), including the checks mutation folder (also removed by checks_v46.py itself).
