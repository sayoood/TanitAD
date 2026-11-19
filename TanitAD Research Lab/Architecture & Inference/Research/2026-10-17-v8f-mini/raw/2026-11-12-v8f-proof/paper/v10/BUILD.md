# Build note: the TanitAD-4B paper, version 10 (2026-11-19)

Source: `paper/v9/tanitad_4b_paper.tex` (the v9 files are not edited). Edits: `patch_tex_v10.py "86 to 99"` (10 edits, each asserted to match exactly once; output `ok edits 10`) with the fragments `frag_level.tex` (row 96), `frag_new.tex` (rows 95, 94, 93, 98; two new subsections), `frag_row99.tex`, `frag_row97.tex` and `frag_status_ii.tex`.

Numbers: only from the register rows H-V8F-PROOF-93 to 99 (`ROW93.md` to `ROW99.md` of the study folder). Check: `check_numbers_v10.py ROW93.md ... ROW99.md ROW92.md` takes every decimal, comma-formatted or multi-digit number of the fragments and looks it up in the row files: 297 tokens checked, 1 not in the rows (45, the number of card ADD45).

Engine: Tectonic 0.17.0 (`tectonic.exe --keep-logs --outdir build tanitad_4b_paper.tex`, in this folder). Result: rc 0, 34 pages (v9: 32), 0 overfull boxes, 0 undefined references.

Tiers: every new number carries T0 or T1-kin and its row id; row 98 is a cost measurement with no tier. STRATEGIC stays as stated in the gap-head subsection: NOT COMPUTED (n = 0). No figure and no camera frame was added. Pages were not rendered to images.

What version 10 adds: the two interior weights of the level term and the end of the calibration fallback (row 96); the planner on the full imagined rollout (row 95); where the forecast error sits (row 94); the trained gap head inside the planner (row 99); the replicate of the finer token grid, not named (row 97); the mutation lists handed over with the levers (row 93); the latency proxy of the part after the encoder (row 98); the status paragraphs, with the coordination decision on the integration base.

Files: `tanitad_4b_paper.pdf` (copy of `build/`), `tanitad_4b_paper.tex`. Nothing deleted; `build/` left in place.
