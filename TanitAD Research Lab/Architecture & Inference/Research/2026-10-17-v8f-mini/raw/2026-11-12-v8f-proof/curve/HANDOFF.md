# HANDOFF RESCURVE (PREREG_V8F_PROOF_ADD7), 2026-11-13. Tier T0, CPU only, zero training. STRATEGIC: PAUSED (PI item 25).
## State
Code and job file are done. The full read has NOT been run. Nothing staged, committed, pushed; nothing written outside F:/TanitAD-scratch/v8proof/curve/.
Files: CURVE_DEFS.md (definitions, deviations R1-R9), rescurve.py, test_rescurve.py, job_rescurve.cmd, this file.
Order inside `rescurve.py run`: c1 (rank 256, both grids, 20 banked P1 values, 1e-6; exit 2 before any read if it fails) and c2 -> read A -> read B -> read C (with c3). Checkpoints after c1/c2, A, B (per grid), C (per grid): RESULT_RESCURVE.partial.json and OOF_RESCURVE.partial.npz. Finals: RESULT_RESCURVE.json, OOF_RESCURVE.npz, VERDICT_RESCURVE.md. Exit 2 if c1/c2/c2_full/c3 fails.
## Launch (orchestrator, detached via WMI, cmd with stdin from NUL as in the box notes)
`cmd /c F:\TanitAD-scratch\v8proof\curve\job_rescurve.cmd < NUL`  (waits until free commit >= 12 GB, polling every 2 min, gives up after 12 h with RESCURVE_STOP.txt; log curve\rescurve_job.log; ends with RESCURVE_DONE.txt `ZZ RESCURVE-DONE` or RESCURVE_STOP.txt `ZZ RESCURVE-STOP <reason>`).
Watch the sentinel files, not the log text.
## Runtime and memory (ESTIMATED, not measured at full size)
Peak about 3 to 4 GB commit (one 350 MB float32 Gram, float64 train kernel 450 MB with 3 to 4 transient copies in the full-kernel solve, rank-256 scores of both grids kept, about 200 MB). Runtime about 3 to 6 h: read A dominated by the rank-1,024 kernel PCA and the full-kernel solves (9 Cholesky of about 6,000 per target and fold), reads B and C by about 150 and about 480 subset kernel PCA fits. Smoke figure for scale only: 1,474 frames, 22 episodes, 4 threads, 179 s end to end (MEASURED, smoke).
## Smoke and tests
Unit tests: 6 passed, 0.4 s (full kernel equals the rank-n ridge, ego block in the kernel equals concatenation, `fit_p1x` is bit-identical to `resprobe.fit_p1`, nested and held-out-disjoint subsets, selection budget/whole episodes/no labels, paired last-doubling gain).
Smoke run (`--smoke 1500`, 1,474 frames / 22 whole episodes, B 300, 2 draws in B, 4 in C): all three reads, all controls and the verdict writer ran without error; c2, c2_full, c3 PASS. c1 CANNOT be checked on a subsample (banked values are for the 9,321-frame set): the smoke printed it as not checkable. c1 is checked only in the full run, as its first act. Smoke numbers are not results (22 episodes).
Not tested: the full-size run, memory and runtime at 9,321 frames, c1 at 1e-6 (it follows from using the same `kpca_fit`, folds, alphas and threads as RESPROBE; a mismatch is a stop, not a deviation).
## Deletions (own temporary data, literal paths)
F:/TanitAD-scratch/v8proof/curve/smoke/ (5 files, 2,324,923 bytes); F:/TanitAD-scratch/v8proof/curve/smoke_run.log (2,713 bytes). No __pycache__ was created (bytecode writing disabled; none under res/).
