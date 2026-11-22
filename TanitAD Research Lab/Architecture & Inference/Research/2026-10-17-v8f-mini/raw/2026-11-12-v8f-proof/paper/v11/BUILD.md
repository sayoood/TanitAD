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

## Update after rows 144 and 145 (2026-11-20)

Fragments added: `frag_row144.tex` (card ADD83, the estimate of another clip as a control) and `frag_row145.tex` (card ADD82, constants of the cautious sign), both appended to `frag_row127.tex`; the item of this fork in `frag_status_run.tex` changed (card ADD84 running). `patch_tex_v11.py`: `ok edits 11`; row ids 144 and 145 found once each in the patched source. `check_numbers_v11.py`: 1837 tokens, 0 not found. PDF `tanitad_4b_paper_r145.pdf` (24ad9e119473d4d3), 44 pages, xdv 2,546,008 bytes (the build after row 143: 44 pages, 2,514,452 bytes).

## Update after row 146 (2026-11-20)

Fragment added: `frag_row146.tex` (card ADD84, the token head in the rule and the requirement curve of a closing read, with the control that did not pass), appended to `frag_row127.tex`; the item of this fork in `frag_status_run.tex` changed (card ADD85 frozen). `patch_tex_v11.py`: `ok edits 11`; row id 146 found once in the patched source. `check_numbers_v11.py`: 1923 tokens, 0 not found. PDF `tanitad_4b_paper_r146.pdf` (c11d4c7bac483913), 45 pages, 2572256 bytes (the build after rows 144 and 145: 44 pages, 2,546,008 bytes).

## Update after row 147 (2026-11-20)

Fragment added: `frag_row147.tex` (card ADD85, the requirement for a closing read with a smooth error, with the two statements of the other session), appended to `frag_row127.tex`; the item of this fork in `frag_status_run.tex` changed (cards ADD86 to ADD88 frozen). `patch_tex_v11.py`: `ok edits 11`; row id 147 found once in the patched source. `check_numbers_v11.py`: 2042 tokens, 0 not found. PDF `tanitad_4b_paper_r147.pdf` (a178cffe26020506), 45 pages, xdv 2599760 bytes (the build after row 146: 45 pages, 2572256 bytes).

## Update after rows 148 to 152 (2026-11-20)

Fragments added: `frag_row148.tex` (card ADD86, the full fit of the token head), `frag_row151.tex` (cards ADD88 to ADD90: the gap history and the ego speed, reported with failed controls, and what gap error a closing read tolerates, with the statements of the other session) and `frag_row152.tex` (card ADD91, the reader gap error over range), all appended to `frag_row127.tex`; the item of this fork in `frag_status_run.tex` changed (card ADD87 running, cards ADD86 and ADD88 to ADD91 read). `patch_tex_v11.py`: `ok edits 11`; rows 148 to 152 each once in the TeX; Tectonic rc 0, xdv 46 pages, 2645824 bytes; `check_numbers_v11.py` over rows 100 to 152: 2309 tokens, 0 not found. PDF `tanitad_4b_paper_r152.pdf` 08f72f0be45949e5 (429038 bytes).

## Update after row 153 (2026-11-20)

Fragment added: `frag_row153.tex` (card ADD92, what kind of gap error a closing read carries), appended to `frag_row127.tex`; the item of this fork in `frag_status_run.tex` changed (card ADD87 running, card ADD93 frozen, cards ADD86 and ADD88 to ADD92 read). `patch_tex_v11.py`: `ok edits 11`; row 153 once in the TeX; Tectonic rc 0, xdv 46 pages, 2658568 bytes; `check_numbers_v11.py` over rows 100 to 153: 2381 tokens, 0 not found. PDF `tanitad_4b_paper_r153.pdf` 16296e456906bd1c (430357 bytes).

## Update after row 154 (2026-11-20)

Fragment added: `frag_row154.tex` (card ADD93, the reader gap error split by clip, in the terms of the bar of row 153), appended to `frag_row127.tex`; the item of this fork in `frag_status_run.tex` changed (card ADD87 running, cards ADD92 and ADD93 read). `patch_tex_v11.py`: `ok edits 11`; row 154 once in the TeX; Tectonic rc 0, xdv 46 pages, 2667600 bytes; `check_numbers_v11.py` over rows 100 to 154: 2435 tokens, 0 not found. PDF `tanitad_4b_paper_r154.pdf` c560a8314e0c2722 (431368 bytes). The PDF of row 153 (`tanitad_4b_paper_r153.pdf`) was built and is not landed; this one takes its place in the hold.

## Update after rows 155 to 158 (2026-11-21)

Fragments added: `frag_row157.tex` (rows 155 to 157: card ADD87, the token head on the finer token grid; card ADD94, failed at a control before any rollout; card ADD95, the gap bar inside the lead rule) and `frag_row158.tex` (card ADD96, the cut of the reader gap error that the rule needs), both appended to `frag_row127.tex`; the item of this fork in `frag_status_run.tex` changed (cards ADD87 and ADD94 to ADD96 read, card ADD97 frozen). `patch_tex_v11.py`: ok edits 11. Build with Tectonic 0.17.0: 47 pages. `check_numbers_v11.py` on rows 100 to 158: 2595 tokens checked, 0 not found. PDF: `tanitad_4b_paper_r158.pdf` e95a9193db4ad564.

Rebuilt on 2026-11-21 after the other session read row 157: one scope sentence on the arm with a scale error of 20 percent was added to the paragraph of rows 155 to 157 (the rule's largest gap of 60 m truncates the planted error on 118 of 3,986 frames). `tanitad_4b_paper_r158.pdf` is now 576cb058eef0f217; the build before it, e95a9193db4ad564, was not landed. Numbers check: 2607 tokens, 0 not found.

## Update after rows 159 and 160 (2026-11-21)

Fragments added: `frag_row159.tex` (card ADD97: the mean of the reader seeds and a causal smoothing do not deliver the cut of the reader gap error) and `frag_row160.tex` (card ADD98: the part of the reader gap error that moves within a clip carries the contact), both appended to `frag_row127.tex`; the item of this fork in `frag_status_run.tex` changed. Build: `patch_tex_v11.py` (ok edits 11), Tectonic 0.17.0, `check_numbers_v11.py` over the rows 100 to 160 (2733 tokens, 0 not found). `tanitad_4b_paper_r160.pdf` 50764daae62c8953, 47 pages; the paragraph of row 160 is on page 46.

## Update after row 161 (2026-11-21)

Fragment added: `frag_row161.tex` (card ADD99: the pilot of a new read of the lead from camera-frame crops fails its bar on the fixed windows; the bound with the windows on the lead reaches the cut of the gap error), appended to `frag_row127.tex`; the status sentence of the fork in `frag_status_run.tex` now ends at row 161 with cards ADD100 and ADD101 frozen. Build: `patch_tex_v11.py` (ok edits 11), Tectonic 0.17.0, `check_numbers_v11.py` over rows 100 to 161 (ZZ NUMBERS-V11 DONE 0). `tanitad_4b_paper_r161.pdf` b6b0e7f4fb8e1752, 47 pages; the paragraph of row 161 is on page 46.

## Update after row 162 (2026-11-21)

Fragment added: `frag_row162.tex` (card ADD101: the lead rule reading the gap of the pilot; with the label marking the frames with a lead the fixed-window gap lies within half the gap to the label gap, at a price in the longitudinal and tactical reads), appended to `frag_row127.tex`; the status sentence of the fork in `frag_status_run.tex` now ends at row 162 with cards ADD100 and ADD102 frozen. Build: `patch_tex_v11.py` (ok edits 11), Tectonic rc 0, `check_numbers_v11.py` over the rows up to 162 (`ZZ NUMBERS-V11 DONE 0`). PDF copied as `tanitad_4b_paper_r162.pdf` dff3e8c44da8f6fa, 48 pages, row 162 on pages 40 and 46.

## Update after rows 163 and 164 (2026-11-21)

Fragments added: `frag_row163.tex` (card ADD100: the windows steered from the image do not reach the cut by 2 and are not separated from the fixed windows) and `frag_row164.tex` (card ADD102: the lead rule with no label at inference lies within half the gap to the label gap), appended to `frag_row127.tex`; the status sentence of the fork in `frag_status_run.tex` now ends at row 164 with card ADD103 frozen. Build: `patch_tex_v11.py` (ok edits 11), Tectonic rc 0, `check_numbers_v11.py` over the rows up to 164 (`ZZ NUMBERS-V11 DONE 0`). PDF copied as `tanitad_4b_paper_r164.pdf` a7afe6f620d5ba73, 48 pages, rows 163 and 164 on pages 41 and 47.

## Update after row 165 (2026-11-21)

Fragment added: `frag_row165.tex` (card ADD103: the replicate of the lead rule with no label at inference is not decided, the slope carries the cut), appended to `frag_row127.tex`; the status sentence of the fork in `frag_status_run.tex` now ends at row 165 with card ADD104 frozen. Build: `patch_tex_v11.py` (ok edits 11), Tectonic rc 0, `check_numbers_v11.py` over the rows up to 165 (`ZZ NUMBERS-V11 DONE 0`). PDF copied as `tanitad_4b_paper_r165.pdf` 41aab148673bff3f, 49 pages, row 165 on page 41, the status sentence on page 47. Figures beside the paper, not yet in the PDF: `figs/fig_leadrule.png` and `.pdf` (`figs/make_fig_leadrule.py`, every value read from the result files of cards ADD101 to ADD103) and `figs/fig_gaptrace.png` and `.pdf` (`figs/make_fig_gaptrace.py`, tier T0; its two input dumps under `figs/figdata/` are not landed).

## Update after rows 166 to 168 (2026-11-21)

Fragments added: `frag_row166.tex` (a count per window on the rollouts of row 164: of 62 contacts, 52 do not occur and 1 is new), `frag_row167.tex` (card ADD104: the form holds over five fits of the head) and `frag_row168.tex` (card ADD105: the mean gap of five fits lies within half the gap), appended to `frag_row127.tex`; the status sentence of the fork in `frag_status_run.tex` now ends at row 168 with card ADD106 frozen. Build: `patch_tex_v11.py` (ok edits 11), Tectonic rc 0, `check_numbers_v11.py` over the rows up to 168 (`ZZ NUMBERS-V11 DONE 0`). PDF copied as `tanitad_4b_paper_r168.pdf` b515ecc7546f573c, 49 pages, rows 166 to 168 on pages 41 and 42; pages that name card ADD106 and pages with the status sentence, as a text search finds them: [42, 48] []. Figures beside the paper, not yet in the PDF: `figs/fig_pairs_r168.png` (the figure of the paired differences with the arms of cards ADD104 and ADD105 added; a new file name, the file landed with hold 116 stays as it is), `figs/fig_rolltrace.png`, `figs/fig_leadrule.png`, `figs/fig_gaptrace.png`.

## Update after row 169 (2026-11-22)

Fragment added: `frag_row169.tex` (two counts on banked rollouts: where the form with no label falls behind and the label gap form does not; the first count reported as written with the flaw of its threshold). Status sentence of the fork replaced (now "Open on this fork after row 169"). Build: `patch_tex_v11.py` (ok edits 11), Tectonic 0.17.0, 50 pages (one more than the build after row 168); number check over the rows 100 to 169: 3,380 tokens checked, 0 not in the rows (a first build carried the count of new clips of card ADD106 in the status sentence, which is in no row yet; the sentence was reworded and the build repeated). PDF copy: `tanitad_4b_paper_r169.pdf` sha256 begins d70eb2e94224c17f; the paragraph of row 169 is on page 42 and the status sentence on page 48. Figure beside the paper, not inside the PDF: `figs/fig_fallpre.png` and `.pdf` from `figs/make_fig_fallpre.py`.

## Update after row 170 (2026-11-22)

Fragment added: `frag_row170.tex` (a T0 diagnostic: the head reads far leads too short; a filter over frames and the mean of fits ruled out). Status sentence of the fork replaced (now "Open on this fork after row 170"). Build: `patch_tex_v11.py` (ok edits 11), Tectonic 0.17.0, 50 pages; number check over the rows 100 to 170: 3,436 tokens checked, 0 not in the rows. PDF copy: `tanitad_4b_paper_r170.pdf` sha256 begins 3a73a833c534f9ec; the paragraph of row 170 is on page 42, the status sentence ends on page 49. Figure beside the paper, not inside the PDF: `figs/fig_gapbins.png` and `.pdf` from `figs/make_fig_gapbins.py`.

## Update after row 171 (2026-11-22)

Fragment added: `frag_row171.tex` (two T0 diagnostics: the bound arm with windows on the lead has about half the gap error; the fixed windows lose the lead when it is more than 8 degrees off the axis; card ADD107 frozen). Status sentence of the fork replaced (now "Open on this fork after row 171"). Build: `patch_tex_v11.py` (ok edits 11), Tectonic 0.17.0, 50 pages; number check over the rows 100 to 171: 3,503 tokens checked, 0 not in the rows. PDF copy: `tanitad_4b_paper_r171.pdf` sha256 begins 9e2703358700f1bb; the paragraph of row 171 is on page 42 and runs on to page 43, the status sentence ends on page 49. Figure beside the paper, not inside the PDF: `figs/fig_offcentre.png` and `.pdf` from `figs/make_fig_offcentre.py`.

## Update after row 172 (2026-11-22)

Fragment added: `frag_row172.tex` (card ADD106 read: the camera head on 1,629 fit clips; gap error cut at T0, the form holds at T1-kin, the difference to the 500-clip fits not separated, the price in falling behind stays). Status sentence of the fork replaced (now "Open on this fork after row 172"). Build: `patch_tex_v11.py` (ok edits 11), Tectonic 0.17.0, 50 pages; row 172 on page 43, status sentence on page 49; `check_numbers_v11.py` over rows 100 to 172: 3641 tokens checked, 0 not found. Output `tanitad_4b_paper_r172.pdf` 3e4ecf52f6d95aad. Figures beside the paper, not set into it: `figs/fig_scale_t0.png`, `figs/fig_scale_pairs.png` from `figs/make_fig_scale.py`.

## Update after row 173 (2026-11-22)

Fragment added: `frag_row173.tex` (card ADD107 read: a head on the tokens of the fixed windows finds the direction of the lead with no label; the gap error with windows at that direction is not measured and needs a new encode, a decision for the PI). Status sentence of the fork replaced (now "Open on this fork after row 173"). Build: `patch_tex_v11.py` (ok edits 11), Tectonic 0.17.0, 51 pages; row 173 on page 43, status sentence on page 49; `check_numbers_v11.py` over rows 100 to 173: 3711 tokens checked, 0 not found. Output `tanitad_4b_paper_r173.pdf` 329f0d49959bc166. Figure beside the paper, not set into it: `figs/fig_lookat.png` from `figs/make_fig_lookat.py`.

## r175 (2026-11-22T08:16:07Z)
- Rows 174 and 175 (cards ADD108, ADD109): fragments frag_row174.tex and frag_row175.tex appended to frag_row127.tex by the asserted script append_r175.py; status sentence now starts "Open on this fork after row 175:"; patch_tex_v11.py ok edits 11; Tectonic rc 0; check_numbers_v11.py over the rows: ZZ NUMBERS-V11 DONE 0 (3,833 tokens checked).
- PDF tanitad_4b_paper_r175.pdf sha256 begins c87550b815f29516, 51 pages (as r173: the added text fits the page count; the file grew from 466,792 to 469,696 bytes); rows 174 and 175 on page 43, status on page 49.
- Figure fig_lookgap (make_fig_lookgap.py, reads the result file of card ADD108) is beside the paper and not set into it.

## r176 (2026-11-22T09:57:26Z)
- Row 176 (card ADD110): fragment frag_row176.tex appended to frag_row127.tex by the asserted script append_r176.py; status sentence now starts "Open on this fork after row 176:"; one spec bullet added; the cost card is unchanged (no decision arises); patch_tex_v11.py ok edits 11; Tectonic rc 0; check_numbers_v11.py over the rows: ZZ NUMBERS-V11 DONE 0 (3,932 tokens checked).
- PDF tanitad_4b_paper_r176.pdf sha256 begins 35610100c11c28c0, 51 pages, 472395 bytes; the text "row 176" is found twice in the PDF text, the status sentence once.
- Figure fig_fallattr (make_fig_fallattr.py, reads the pulled result file of card ADD110) is beside the paper and not set into it.
- Row 177 (card ADD111): fragment frag_row177.tex appended to frag_row127.tex by the asserted script append_r177.py; status sentence now starts "Open on this fork after row 177:"; one spec bullet added; the cost card is unchanged (no decision arises); patch_tex_v11.py ok edits 11; Tectonic rc 0; check_numbers_v11.py over the rows: ZZ NUMBERS-V11 DONE 0 (count not printed).
- PDF tanitad_4b_paper_r177.pdf sha256 begins 4399e261231f2158, 52 pages, 475476 bytes; the text "row 177" is found twice in the PDF text.
- Figure fig_gaperr (make_fig_gaperr.py ade565e3b77e162e, reads the pulled result file of card ADD111) is beside the paper and not set into it.
- Rows 178 and 179 (cards ADD112 and ADD113): fragments frag_row178.tex and frag_row179.tex appended to frag_row127.tex by the asserted script append_r179.py; status sentence now starts "Open on this fork after row 179:"; two spec bullets added (the second is the table of decision D16-A for the lead rule); the cost card is unchanged (no decision arises); patch_tex_v11.py ok edits 11; Tectonic rc 0; check_numbers_v11.py over the rows: ZZ NUMBERS-V11 DONE 0.
- PDF tanitad_4b_paper_r179.pdf sha256 begins 8942a88e90cfefb0, 52 pages, 479502 bytes; the text "row 178" is found once and "row 179" twice in the PDF text.
- Figure fig_candset (make_fig_candset.py 7fd5f98cb3ad249b, reads the pulled result file of card ADD113) is beside the paper and not set into it.
