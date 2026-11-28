# Build note: the TanitAD-4B paper, version 9 (2026-11-19)

Source: `paper/v8/tanitad_4b_paper.tex` (v8 files not edited; `v8_base.tex` is a copy). Edits: `patch_tex_v9.py` (11 edits, each asserted to match once; output `ok edits 11`) with the fragments `frag_qhead.tex`, `frag_dose.tex`, `frag_level.tex`, `frag_replan.tex`, `frag_status.tex`. Run as `capmem.py 300 patch_tex_v9.py` (OMP_NUM_THREADS=1). `fix1.py` repaired a tab character in a fragment (a sed slip) and is kept.

Numbers: only from register rows H-V8F-PROOF-84 and 86 to 91 (`rows_84_91.txt`, commit 85f85844) and, for the level term, the three files `actlevel/read/NAMES_RLVa.json`, `NAMES_RLVb.json`, `NAMES_RLAh.json`. `ROW92.md` did not exist, so ROW92.md was NOT used; the text says "register row 92 in preparation". Check: `make_numbers_v9.py` takes every decimal or comma-formatted token on a line that v9 adds or changes against v8 (tokens already on the old line skipped) and finds it literally in a row or as a rounded JSON value: 401 matched (`NUMBERS_V9.tsv`), 0 unmatched. The check matches tokens, not their neighbours; a token can match by chance (the level weights 0.002 and 0.004 are not in any row: they come from note ADD37_N7 as given in the task).

Engine: Tectonic 0.17.0 (`C:/Users/Admin/tools/tectonic-0.17.0/bin/tectonic.exe --keep-logs --outdir build tanitad_4b_paper.tex`, in this folder, `build` created first). Result: rc 0, 32 pages (v8: 29), 0 overfull boxes, 0 undefined references. The first build had 3 overfull boxes (the level table in one column) and 1 slight one; fixed by a full-width table and a narrower column.

Tiers: every new number carries T0 or T1-kin and its row id; STRATEGIC is stated once (start of the gap-head subsection) as NOT COMPUTED (n = 0 ...). No figure and no camera frame was added. Pages were not rendered to images.

Files: `tanitad_4b_paper.pdf` (copy of `build/`), `tanitad_4b_paper.tex`. Nothing deleted; `build/` left in place.

Lead patches after the agent build (2026-11-19): `patch_row92.py` (4 edits: register row 92 cited, family and headway line of the level arms, control v5 line) and `patch_row92b.py` (1 edit: the three contact-rate differences of row 90 get their reads), then the same Tectonic command: rc 0, 32 pages, 0 overfull boxes, 0 undefined references. `tanitad_4b_paper_pre92.tex` is the agent build before these patches.
