# Build note: the TanitAD-4B paper, version 3 (2026-11-10)

Source: raw/2026-11-08-paper-pdf-v2/tanitad_4b_paper.tex (sha256 fb1d7ae8..., asserted by the patch). Edits: patch_tex_v3.py, six edits, each asserted to match exactly once.
Engine: Tectonic 0.17.0, `tectonic --keep-logs --outdir build tanitad_4b_paper.tex`. Result: 9 pages (v2: 8), 0 overfull boxes, 0 undefined references; one font-lookup warning (TeXGyreTermesX-BoldItalic) that resolves to the same face, as the log shows.
Pages 1, 7 and 8 were rendered to PNG with the Windows PDF API and read: the new paragraph, the three headline rows and the roadmap rows set without overflow.

## What changed (all MEASURED, tier T1-kin unless marked)
* Abstract: one sentence on the per-band reading (DAgger data carries 2-6 s, the acceleration term 0-2 s; the band loss adds a reproduced 0-2 s gain; the top three tactical manoeuvres contain the logged one in 90 % of windows).
* Results, Table 1: three rows (per-band attribution of the adopted base; the band loss beyond a same-recipe replicate; top-3 headroom, an oracle).
* Results: new paragraph "Which lever carries which band".
* Roadmap, Table 2: world-model planner (its first gates on the M2b latents do not yet beat holding the logged action, T0); band retraining (read, two seeds); new row tactical selection (designed); perception adaptation (in training, seed 0; about 184 h per seed at the measured 34.5 s per step on the 20 GB pod GPU, 19,164 steps: ESTIMATED from the measured step time).
* No number of v2 changed.

## Sources of the new numbers
* `raw/2026-11-09-v8f-otband/read/out/READ_FULL.json` (bars, ladder), `out/FLOOR_TABLE.json` (replicate floors), `out/TOPK_FULL.json` (top-3), register row H-V8F-OTBAND-1, programme paper section 7.24 (64 of 64 numbers checked against these files on 2026-11-10).
* Rounding: 0.0168 / 0.0149 -> 0.017 / 0.015; 0.0393 / 0.0267 -> 0.039 / 0.027; tactical replicate 0.2869 -> 0.29; top-1 0.572-0.585 -> 57-59 %; top-3 0.895-0.901 -> 90 %; best of three -0.3537 to -0.3904 -> -0.35 to -0.39.
* LoRA step time: the pod's train log, steps 3,000-6,300 (2026-11-09 23:18 UTC).

## Not re-verified
* As v2: the count "2^7 corners" and the PUBLISHED statements of the cited papers.

| file | sha256 |
|---|---|
| tanitad_4b_paper.tex | 1b783e91bc86ce772617f06f5d3ff8c4912692bce00dab759a40e3145b0ebbaa |
| tanitad_4b_paper.pdf | c92a117228cd56ad1245e46afce1637c847ec79619133e1a1f6c4475131affa2 |
| patch_tex_v3.py | 0b669b42aff7f7eb3a5b2821e41ebcba6931ca572c0fdea1426cc7651d3bd625 |
