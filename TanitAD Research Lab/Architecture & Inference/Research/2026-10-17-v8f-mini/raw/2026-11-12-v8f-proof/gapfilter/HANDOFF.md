# HANDOFF: GAPFILTER reader (card PREREG_V8F_PROOF_ADD28)

Tier T0 (a diagnostic on logged frames, never driving performance). STRATEGIC: PAUSED (PI item 25). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read. Evidence class MEASURED (ours, T0).
No git, pod, ssh, box 2 or GPU used. Nothing deleted or moved.

## State
Reader, definitions, tests and the run on the six arms are done. Card run result: control f2 FAILED, so no outcome is named (FILTER-* names withheld). f1, f3, f4, f5, f6 and the added alignment control x1 pass for all six arms.

## What ran
* Tests: `cd F:/TanitAD-scratch/v8proof/gapfilter && CUDA_VISIBLE_DEVICES=-1 /c/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_gapfilter_read.py --basetemp F:/TanitAD-scratch/v8proof/gapfilter/tmp/pt/<new name>`: 13 passed.
* Card run (about 25 s): `python gapfilter_read.py run --out-dir F:/TanitAD-scratch/v8proof/gapfilter/out` (inputs: spatial/leadsup/read/in, states in C:/Users/Admin/v8proof_c/s1states). Outputs `out/RESULT_GAPFILTER.json`, `out/VERDICT_GAPFILTER.md`. Re-run gave an identical result.
* Post-hoc variant, names nothing (same-lead step rule 3.0 m, written after f2 failed, see DEFS addendum A1, A2): `--variant-jump-m 3.0 --out-dir .../out_jump`. All controls pass there; no name.

## Verified and not
Verified: M1 of every arm reproduces the N1 result (to 1e-4) and DELTA of the head's own closing output too; k1 label path 0.7499068; f3 exactly 0; f4 permuted M1F between -0.088 and -0.32; f5 selection identical under noise labels; f6 records and layouts; f2 diagnosed (410 of 413 windows exact, 3 lead changes).
Not verified: pod and dev-box forward are the N1 reader's k2 (not re-run here); nothing on the T1 rollouts; the card has no same-lead rule, so the card's E1/E2 fit across lead changes (that is why f2 fails).

## Hashes (sha256)
* gapfilter_read.py b876a1bf075af2e5c60baec8cd9c6641049722ee35261eb281f9a1690e7eb30a
* test_gapfilter_read.py eb76ca6b0eb58dac873518c66530d74bb543a514c7b51436d337c2c24188ea01
* GAPFILTER_DEFS.md 816e99e47ea07ce1f9017075c5801201fb623f8a6013598918423605264eb39e
* out/RESULT_GAPFILTER.json a10c6dbfa04f6b15d9cff9b9a06d2349d8c596338efa47f4cb04df830bcdabc1
* out/VERDICT_GAPFILTER.md 393180952edc6116749a698f715df7de2969c6d71223c75c84ef9c31ba85e7dd
* out_jump/RESULT_GAPFILTER.json 90505247d42ec689aa5ff4a4d38f971056bda1b95e0931ed197c36e162a0cc9d

## Scratch (all under F:/TanitAD-scratch/v8proof/gapfilter/tmp, about 180 KB, nothing deleted)
`tmp/pt/a1 a2 a3` (pytest basetemps), `tmp/smoke/` (one-arm smoke run of the card path), `tmp/diag_f2.py`, `tmp/gapfilter_read.before_jump.py` (reader before the variant edit), `tmp/RESULT_prev.json`. The deletes are the lead's to make.
