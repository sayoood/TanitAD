# Build note: the TanitAD-4B paper, version 7 (2026-11-15)

Source: `paper/v6/tanitad_4b_paper.tex` (sha256 8f1e987fb2e8951b53e1c6d003c17156d352989ede0d0481344d2d65d1e607b9, asserted by the patch; the v6 files are not edited). Edits: `patch_tex_v7.py`, 25 edits, each asserted to match exactly once; output `ok edits 25`.
Numbers: copied only from the register rows H-V8F-PROOF-42 to 54 on the mirror tip a5013b63 (`gac_tip.md`, the 14 lines `rows.txt`, row 41 is version 6 itself, so nothing is doubled). No number is recomputed.
Engine: Tectonic 0.17.0, `C:/Users/Admin/tools/tectonic-0.17.0/bin/tectonic.exe --keep-logs --outdir build tanitad_4b_paper.tex` (run in this folder). Result: rc 0, **20 pages** (v6: 16), **0 overfull boxes**, **0 undefined references**, 45 underfull boxes (narrow cells and justified lines; v6 had the same kind).
Pages 1, 2, 8 to 20 were rendered with `../v6/render_pages.ps1` into `png/`; pages 1, 15 to 19 were read (abstract, the lever and per-seed tables, the lead tables, the distance section, the family table, the outlook and conclusion). No broken table was found; the last page is 70 percent full in its right column.
Every measured value is a T0 or T1-kin (kinematic rollout on the model's own actions) diagnostic on logged frames. STRATEGIC family: PAUSED (PI item 25), stated in Tables 8 and in each new section.

## What changed, section by section
* Abstract: the "two levers at seed 0" sentence is replaced by the final read (flat at both seeds); new sentences for GRID-NO-COST (six seeds), lead supervision (gap and presence read, closing speed flat; control k5 failed as written, names read under two later notes), distance keeping (DISTANCE-EQUAL; U 0.1275), the distance rule (about 60 percent with labels, about 13 to 32 percent with the model's own head) and the final data-efficiency names (DATA-SATURATED, KNEE-1/8). The "preliminary, four of six seeds" quarter sentence is replaced by the six-seed read.
* Contributions: one new item (the lead vehicle and distance keeping).
* Table 2 (headline results): three new rows (distance rule; grid state with lead supervision; eighth and quarter of the clips).
* 8.1 premises: the intro sentence about reads still training is changed to "reads that do not yet exist"; the limits sentence of the six-seed replicate points to the headway read.
* 8.2 state: last sentence of the cell-predictor paragraph (family read now final, NO-COST); the paragraph "Two levers, seed 0 only" is replaced by "The two levers at both seeds, and the arms that exist" (rows 50) with Table 3; new paragraph "The grid state costs nothing on six paired seeds" (rows 45 and 42) with Table 4 (per-seed differences).
* New 8.3 "What the state holds of the lead's motion": OBJSLOT (row 43), LEADHEAD stage 0 and 0b (rows 44, 46, Table 5), LEADSUP (rows 47, 51, 52, Table 6, the card form first), GAPFILTER and TOKPROBE (row 53).
* New 8.4 "Distance keeping, and a rule on the plan": HEADWAY (row 48), the DISTRULE design and controls (rows 49, 51 notes), label-fed rule (row 54), head-fed rule (row 54, Table 7), lead supervision alone (row 54).
* 8.5 (was 8.3) data efficiency: the preliminary paragraph is replaced by the final read (rows 42 and 45); Table 8 (was Table 3): the five earlier-read rows are replaced by three rows of the final reads (state arms, fewer clips, chosen against random), the hold-rule emulation row is kept, six rows for the new reads are added.
* Roadmap: the state row is extended; one new row (distance rule). Outlook paragraph rewritten; one conclusion sentence extended.
* Layout: top and bottom margins 20/22 mm to 18/19 mm, new floats `[!tp]`, roadmap table in footnotesize, bibliography in scriptsize (all to hold 20 pages).
* Wording: the four v6 sites that carried a negated phrase about performance now read "a diagnostic on logged frames" (the banned-word grep over the tex returns 0).

## Rows placed
| row | where |
|---|---|
| 42 DATAEFF final, SPATIAL Q3 final | 8.5 data paragraph, Table 2, Table 8 rows 1 to 3; 8.2 grid paragraph |
| 43 OBJSLOT | 8.3 first paragraph |
| 44, 46 LEADHEAD | 8.3 second paragraph, Table 5 |
| 45 SPATIALREP, one-eighth addendum | 8.2 grid paragraph, Table 4, abstract; 8.5 data paragraph |
| 47 LEADSUP build | 8.3 LEADSUP paragraph (build sentence); the six-arm result is row 52 |
| 48 HEADWAY | 8.4 first paragraph, Table 4 row 2 |
| 49 DISTRULE card, M3-BC | 8.4 second paragraph |
| 50 MOTION, TARGET | 8.2 levers paragraph, Table 3, Table 8 |
| 51 LEADSUP seed 0, notes N1, ADD26_N1 | 8.3 (N1) and 8.4 (ADD26_N1) |
| 52 LEADSUP six arms | 8.3, Table 6, Table 8 |
| 53 GAPFILTER, TOKPROBE | 8.3 last paragraph, Table 6 |
| 54 DISTRULE result | 8.4, Tables 4, 7 and 8, abstract, Table 2, outlook |

Where a row says a control failed as written (k5 of card ADD24, f2 of card ADD28, p2b of card ADD27) the paper says so first and gives the read as "under notes N1 and N2" or as the numbers of the card run with no name. Outcome names are copied exactly.

## Number check
`make_numbers_v7.py` (adapted from `make_numbers_v6.py`): every decimal or comma-formatted token on every line that v7 adds or changes against v6, looked up literally in the register rows; **619 matched, 0 unmatched** (`NUMBERS_V7.tsv`: value, tex line, register row, artifact path named by the row, 80-character snippet; `numbers_unmatched_v7.txt` is empty). A second check on signs: every signed token on a changed line appears with the same sign in the rows, except three v6 values on line 989 (-0.103, +0.122, -0.037) that sit on a line v7 touches and come from the v6 rows.
Limit of the check: a value is matched to the row where it occurs with the most companions from its sentence; it proves the digits exist in the register, not that each is attached to the right quantity. Integers (seeds, counts of windows below 1,000, "six of six", "about 60 percent", "13 to 32 percent") are copied by hand from the same rows and not scripted. Rounded values: none are rounded in v7 except the abstract's "about 60 percent" and "about 13 to 32 percent", which are the wording of row 54.

## Known gaps
* The restatement of the register rows in prose is not machine-checked beyond the digit check; the name-to-quantity pairing for PR and PE in the levers paragraph follows the order of the row title.
* Table 8 keeps the hold-rule emulation row of v6 unchanged. The v6 rows for G minus baseline, P minus G and the preliminary quarter, half and eighth reads are removed (replaced by the final reads); their numbers are not in the paper.
* Rows 43 and 44 give T0 reads only; no four-family cell exists for them and none is invented.
* Control x1 of the lever arms and the per-arm gate tokens were not checked in the source (stated in the paper).
* PNG pages 2, 8 to 14 and 20 were rendered but not each read; the pages with changed tables were.
* Strategic family PAUSED everywhere it is listed; no strategic number exists.

## Files
`tanitad_4b_paper.tex`, `build/tanitad_4b_paper.pdf`, `NUMBERS_V7.tsv`, `numbers_unmatched_v7.txt`, `patch_tex_v7.py`, `make_numbers_v7.py`, `rows.txt`, `gac_tip.md`, `build_stdout.txt`, `png/`.
