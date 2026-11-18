# CHANGES_V8: the TanitAD-4B paper, version 8 (2026-11-18)

Source: `paper/v7/tanitad_4b_paper.tex` (not edited). Edits: `patch_tex_v8.py`, 15 edits, each asserted to match exactly once (output `ok edits 15`), plus one word change in the output (`bias-corrected` to `bias-adjusted`). New text lives in `frag_tokprobe.tex`, `frag_dist.tex`, `frag_ruletrain.tex`, `frag_imag.tex`, `frag_fam7.tex`, `frag_status.tex`.
Numbers: copied from the register rows H-V8F-PROOF-56 to 83 (`rows_56_83.txt`, extracted from `GAC_ROWS_*.md`). Row 55 is the register row of paper version 7 itself (paper/v7/GAC_ROWS_41.md), so nothing is doubled. Rows 42 to 54 are covered by v7. No number is recomputed; no camera frame or image is added.
Build: Tectonic 0.17.0, `tectonic.exe --keep-logs --outdir build tanitad_4b_paper.tex`, rc 0, 29 pages (v7: 20), 0 overfull boxes, 0 undefined references. Table 13 sits alone on page 28.
Not machine-checked: the digit check of v7 (`make_numbers`) was not re-run for v8; the numbers were typed from the rows and read once against them.

## Rows added and where
| row | name | where |
|---|---|---|
| 56 | TOKPROBE | 8.3, new paragraph before 8.4 |
| 57 | RULEATTR | 8.4 (Table 8) |
| 58 | MARGIN | 8.4 (Table 8) |
| 59 | QCAL | 8.4 (Table 8) |
| 62 | NOCLOSE | 8.4 (Table 8) |
| 61, 63, 65 | RULETRAIN (2,000 / 6,000 steps / weight 4) | new 8.5 (Table 9) |
| 64, 66 | RULEOWN squared, linear | new 8.5 (Table 9) |
| 71, 74, 75 | LEADCOND, LEADSCR, PLANLEAD | new 8.5 (Table 9) |
| 60, 67, 72 | MOTION (RR, RES, ROLL, four seeds) | new 8.6 (Table 10) |
| 82 | TARGET (PF) | new 8.6 (Table 10) |
| 70, 73 | LATVIS-2, two and four seeds | new 8.6 |
| 68, 77 | IMAGLEAD (LS, RL) | new 8.6 (Table 11) |
| 78 | LOGASSOC | new 8.6 |
| 69, 76, 80 | IMAGACT (build, LS bases, RL bases) | new 8.6 (Table 11) |
| 79, 81 | IMAGPLAN tool controls, source cand-imagkin | new 8.6 |
| 83 | STRATEGIC addendum | 8.7 family paragraph, Table 13 |

## Sections touched
Abstract (one added block); contributions (one item); Status paragraph of section 4 (pointer); Table 2 (five rows); 8.1 premises intro (strategic stamp); 8.3 TOKPROBE sentence and new paragraph; 8.4 head-fed paragraph tail (ADD29 result in place of "running") and four new paragraphs with Table 8; new 8.5 and 8.6; 8.7 (was 8.5) strategic paragraph and Table 13; roadmap (rows for the distance rule, imagination and planning, world-model state); outlook paragraph; new subsection "Status of the study" (Established, Not met as written, Running or queued, operations fact of the 65 hours, Open decisions); conclusion (two sentences).

## Stamps
Every new read carries T0 or T1-kin; none is presented as driving performance. STRATEGIC: PAUSED for rows dated 2026-11-15 as written; NOT COMPUTED (n = 0) for rows 80 to 82 (dated 2026-11-18) with the reason of row 83.
Missed bars and failed controls stated as written: MARGIN-COSTS, QCAL-COSTS, OWN-RULE-COSTS, BEATS-MARGIN bars, MOTION-ADDS-NOTHING-4, p2b (TOKPROBE), k5 (LATVIS-2), q3 (LEADCOND), d4 (PLANLEAD), p5b (row 79 not met at +0.185; row 81 not met at +0.495 and +0.479 against 0.5), equals extrapolation (row 81), TARGET-FLAT (row 82).

## Temp files left (not deleted)
`ex.py`, `rows_56_83.txt` (kept as the number source), `png/` (page renders, `png/p222628.png` is empty), `build/`, `build_stdout.txt`.
