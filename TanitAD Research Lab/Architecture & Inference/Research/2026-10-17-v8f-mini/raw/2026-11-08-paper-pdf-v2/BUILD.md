# Build note: the TanitAD-4B paper, version 2 (2026-11-08)

Source: raw/2026-11-06-paper-pdf/tanitad_4b_paper.tex (sha256 ad88ed32...). Edits: patch_tex.py, every edit asserted to match exactly once.
Engine: Tectonic 0.17.0, `tectonic --keep-logs --outdir <dir> tanitad_4b_paper.tex`. Result: 8 pages, 0 overfull boxes, 0 undefined references.
Pages 2 and 3 (Figure 1, the new section 3.2) were rendered to PNG with the Windows PDF API and read.

## What changed
* PI rulings of 2026-11-08 (REQ_HORIZONS.md, R-EGO, R-T2O, R-OUT): new section 3.2 (bands, every level reads the current ego state, tactical goals and constraints as the action conditioning the operative level, output of one or more trajectories up to 6 s, strategic extension, world-model planner, status).
* Predictors of Eq. 1 and Figure 1 are action-conditioned and goal-free; goals condition through the readers and the planner cost (review against H-JEPA, REVIEW_HJEPA_CONSISTENCY.md).
* Budget: at most 800 M parameters (PI 2026-10-17) in place of 300 M.
* Mathematics: horizon argument (per-level bounds sum back to the undivided bound at equal per-step error); recombination needs independence; 15 s strategic roll is three steps only at 0.2 Hz; Cramer-Wold stated for standard normal projections and finite slices; band limit stated for the cosine factor; DAgger bound under its no-regret assumptions.
* Status: T2 first reads on 48 scenes (report v45 section 7.48) in place of "in preparation"; strategic rows are measurements, their bars paused (PI item 25).
* No measured result of the 11-06 version changed.

## Checked and correct, left as written
Epps-Pulley statistic, heteroscedastic imagination loss, chain rule and Lipschitz decomposition, data-processing inequality, law of total variance, floor identity and ramp, code dimension 22+5+7+26=60.

## Not re-verified
* The count "2^7 corners" in the interface-support section: no source located; left unchanged.
* PUBLISHED statements of the cited papers were not re-read against their primaries.

| file | sha256 |
|---|---|
| tanitad_4b_paper.tex | fb1d7ae82815856e988fa604bbdf98edc1717de47f8168500607e67995c46404 |
| tanitad_4b_paper.pdf | 5777d29679bd162437ac6a8ee1af9786cc728483f284d82a191fa03082a19775 |
