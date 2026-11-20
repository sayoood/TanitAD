# Build note: the TanitAD-4B paper, version 11 (2026-11-20)

Source: `paper/v10/tanitad_4b_paper.tex` (the v10 files are not edited). Edits: `patch_tex_v11.py` (11 edits, each asserted to match exactly once; output `ok edits 11`) with the fragments `frag_integ.tex` (rows 100 to 107, 113: the integrated 10k line, the closing speed, the rule's parameters), `frag_cycle.tex` (rows 108 to 112, 114 to 116, 118, 119: replan cycle, acceleration read from the own rollout, second look), `frag_strat.tex` (rows 120, 121, 122, 125: strategic read along the rollout, heading bound and executor stamp, inference-side candidate, decision on card D16; markers for rows 126 and 127), `frag_scale.tex` (rows 117, 123, 124), `frag_fam7rows.tex` (four rows of the four-family table), `frag_t1row.tex` (one row of the headline table) and `frag_status_run.tex` (the running-or-queued paragraph). The established, not-met, strategic-family, closing and handed-over paragraphs are edited in the patch script itself. `frag_row126.tex` and `frag_row127.tex` do not exist: the patch inserts them at their markers when they exist and nothing when they do not (the two reads of cards ADD68 and ADD69 are named in the status paragraph as pre-registered, no result).

Numbers: only from the register rows H-V8F-PROOF-100 to 125 (`ROW100.md` to `ROW125.md`). Check: `check_numbers_v11.py ROW100.md ... ROW125.md`: 760 tokens checked, 2 not in the rows: `97` (the id of row 97, cited in the status paragraph; it is a row id of version 10) and `127` (the marker `%%ROW127%%` of the fragment, removed by the patch when its file is absent).

Engine: Tectonic 0.17.0 (`tectonic.exe --keep-logs --outdir build tanitad_4b_paper.tex`, in this folder, output in `build_v11.out`). Result: rc 0, 39 pages (v10: 34; +5), 0 overfull boxes, 0 undefined references (grep of `build/tanitad_4b_paper.log`). The only warnings are the font and hyperref notes of the engine, as in v10.

Float note (edit E11): in v10 the headline table (`[!t]`) stayed in the float queue behind its height from page 10 on and every later single-column float followed it, so all of them were printed on or after the references page. v11 changes its placement to `[!tp]`; the table is now on page 11, all single-column tables sit in the text, and the references are on page 39. Nothing else of the table changes except one compact row (row 121). The same fix, tried without the new row, gave the same placement.

Tiers: every new number carries T0 or T1-kin and its row id; row 100 is a training-log replay (no tier). Executor stamp: said once in the subsection of the heading bound (rows 100 to 125 were measured under the study wrapper; runs named `Hb` used the heading bound, 0.2 1/m unless another dose is named, all others the unbounded executor) with the decision of 2026-11-20 (row 125) that the bounded executor is the tier of record from the overlap read on and earlier rows stand as measured. STRATEGIC: stated as measured along the rollout by the study instrument (route checkpoints at 30, 50 and 80 m, rows 120 to 122), a decision between routes is still not scored (n = 0); the earlier reads keep their entry as written. The wording rule of the register was checked on the new text (grep, no hit); no figure and no camera frame was added; pages were not rendered to images.

What version 11 adds: the integrated candidate at 10k clips on the two frozen checkpoints (rows 100 to 107, 113); the 1.0 s cycle and the acceleration the plan reads from its own rollout, with its dose (rows 108 to 112, 114 to 116, 119); the rule's second look (row 118); the strategic read along the rollout (row 120); the inference-side candidate (row 121); the heading bound of the executor (row 122); the lead read against the number of clips (rows 117, 123, 124); the decision of the PI on card D16 (row 125). Status paragraphs (established, not met, running or queued, handed over) agree with the rows. Cards ADD68 and ADD69 are named as pre-registered, no result.

Files (folder `F:/TanitAD-scratch/v8proof/paper/v11/`): `tanitad_4b_paper.tex`, `tanitad_4b_paper.pdf` (copy of `build/`), `patch_tex_v11.py`, `check_numbers_v11.py`, `frag_*.tex` (seven), `BUILD.md`, `build_v11.out`, `build/` (engine output, left in place), `scratch_t1/` (a test build used to place the headline table; left in place, deletable). Nothing deleted.

## Update after rows 126 to 128 (2026-11-20)

Fragments added or changed: `frag_row126.tex` (row 126), `frag_row127.tex` (rows 127 and 128, inserted at the marker of row 127), `frag_status_run.tex` (items iv and v). `patch_tex_v11.py`: `ok edits 11`. Build: rc 0, 40 pages, 0 overfull boxes, 0 undefined references. Number check with `ROW100.md` to `ROW128.md`: 884 tokens checked, 0 not in the rows.

## Update after rows 129 and 130 (2026-11-20)

Fragments changed: `frag_row127.tex` (the paragraph of row 128 rewritten to the re-read of row 129, with the readings of row 128 that do not stand named as such; a new paragraph for row 130), `frag_status_run.tex` (item iv). `patch_tex_v11.py`: `ok edits 11`. Build: rc 0, 40 pages, 0 overfull boxes, 0 undefined references. Number check with `ROW100.md` to `ROW130.md`: 967 tokens checked, 0 not in the rows. The wording rule of the register was checked on the new text (grep, no hit).

PDF of record after rows 129 and 130: `tanitad_4b_paper_r130.pdf` (40 pages). The file `tanitad_4b_paper.pdf` on the branch is the build after rows 126 to 128 and quotes readings of row 128 that do not stand; the landing route rewrites text files only, so the rebuilt PDF is landed under the new name.

## Update after row 131 (2026-11-20)

Fragments changed: `frag_row127.tex` (a paragraph for row 131 appended), `frag_status_run.tex` (item iv). `patch_tex_v11.py`: `ok edits 11`. Build: rc 0, 41 pages, 0 overfull boxes, 0 undefined references. Number check with `ROW100.md` to `ROW131.md`: 997 tokens checked, 0 not in the rows. A first build of this update printed one macro of the new paragraph as text (a tab had replaced its first two characters when the paragraph was written through a shell); found by a check of the fragment for tab characters, repaired, rebuilt. PDF of record: `tanitad_4b_paper_r131.pdf`.

## Update after row 133 (2026-11-20)

Fragments changed: `frag_row127.tex` (the paragraphs `frag_row132.tex` and `frag_row133.tex` appended), `frag_status_run.tex` (item iv). Build: rc 0, 41 pages, 0 overfull boxes, 0 undefined references. Number check with `ROW100.md` to `ROW133.md`: 1177 tokens checked, 0 not found. PDF saved as `tanitad_4b_paper_r133.pdf` (sha256 71e6434486cb4c45).

## Update after row 135 (2026-11-20)

Fragments changed: `frag_row127.tex` (the paragraphs `frag_row134.tex` and `frag_row135.tex` appended), `frag_status_run.tex` (item iv). `patch_tex_v11.py`: `ok edits 11`. Build: rc 0, 42 pages, 0 overfull boxes, 0 undefined references. Number check with `ROW100.md` to `ROW135.md`: 1299 tokens checked, 0 not found. PDF saved as `tanitad_4b_paper_r135.pdf` (sha256 61be4db651d04a11); it holds the paragraphs of rows 132 to 135.

Fault of the build after row 133, found here: that build ran the typesetter without the step `patch_tex_v11.py`, which is the step that puts the fragments into the paper source. The file `tanitad_4b_paper_r133.pdf` (sha256 71e6434486cb4c45, 41 pages) therefore has the text of the build after row 131 and does not hold the paragraphs of rows 132 and 133, although the entry above says it was built after them. The number check reads the fragments, not the PDF, so it did not see this. From this build on the page count and the row ids in the paper source are checked after the patch step.

## Update after row 136 (2026-11-20)

Fragments changed: `frag_row127.tex` (the paragraph of row 135 replaced by `frag_row135b.tex`: card ADD45 has no name, rows 135 and 136), `frag_status_run.tex` (item iv). `patch_tex_v11.py`: `ok edits 11`; row ids 134, 135, 136 found once each in the paper source. Build: rc 0, 42 pages, 0 overfull boxes, 0 undefined references. Number check with `ROW100.md` to `ROW136.md`: 1335 tokens checked, 0 not found. PDF saved as `tanitad_4b_paper_r136.pdf` (sha256 cc337ca4f48d0c3a). The PDF `tanitad_4b_paper_r135.pdf` says in its paragraph of row 135 that the lead-gap read is flat with scale; by row 136 that name does not stand, and the PDF after row 136 is the one to read.

## Update after rows 137 and 138 (2026-11-20)

Fragments added: `frag_row137.tex` (card ADD76, the second-look arm), `frag_row138.tex` (card ADD77 without a name), both appended to `frag_row127.tex`; `frag_status_run.tex` item on this fork changed. `patch_tex_v11.py`: `ok edits 11`; row ids 136, 137, 138 found once each in the paper source. Build: rc 0, 42 pages (the page count of the build after row 136; the xdv byte count differs, see the log line below), 0 overfull boxes, 0 undefined references. Number check with `ROW100.md` to `ROW138.md`: 1520 tokens checked, 0 not found. PDF saved as `tanitad_4b_paper_r138.pdf` (sha256 f2c0127a9caf2846).

## Update after rows 139 and 140 (2026-11-20)

Fragment added: `frag_row139.tex` (card ADD78), appended to `frag_row127.tex`; `frag_row137.tex` and its copy in `frag_row127.tex` changed in one place (the at-age setting is a pre-roll of 30 steps, 6 s of own driving; the PDF after rows 137 and 138 says 30 s there, see row 140); `frag_status_run.tex` item on this fork changed. `patch_tex_v11.py`: `ok edits 11`; row ids 137, 138, 139 found once each in the paper source, the old wording 0 times. Build: rc 0, 43 pages, 0 overfull boxes, 0 undefined references. Number check with `ROW100.md` to `ROW140.md`: 1572 tokens checked, 0 not found. PDF saved as `tanitad_4b_paper_r140.pdf` (sha256 b1612da5e8d177b7).

## Update after rows 141 and 142 (2026-11-20)

Fragments added: `frag_row141.tex` (card ADD79, the estimated closing speed in the reader-fed rule) and `frag_row142.tex` (card ADD80, constant closing inputs, with the sign of the input that the card had wrong), both appended to `frag_row127.tex`; the item of this fork in `frag_status_run.tex` changed. `patch_tex_v11.py`: `ok edits 11`; row ids 141 and 142 found once each in `tanitad_4b_paper.tex`. Tectonic 0.17.0 build; number check `check_numbers_v11.py` on the row files 100 to 142: 1672 tokens, 0 not found. PDF copied to `tanitad_4b_paper_r142.pdf` (410587 bytes, sha256 8d8feb9cdf016c63).
43 pages, xdv 2,496,148 bytes (the build after rows 139 and 140: 43 pages, 2,448,976 bytes).

## Update after row 143 (2026-11-20)

Fragment added: `frag_row143.tex` (card ADD81, the token head as a read of the closing speed), appended to `frag_row127.tex`; the item of this fork in `frag_status_run.tex` changed (cards ADD82, ADD83, ADD84). `patch_tex_v11.py`: `ok edits 11`; row id 143 found once in the patched source. `check_numbers_v11.py`: 1712 tokens, 0 not found. PDF `tanitad_4b_paper_r143.pdf` (c0a6b89a181a9f34), 44 pages, xdv 2,514,452 bytes (the build after rows 141 and 142: 43 pages, 2,496,148 bytes).
