# HANDOFF: BIASEDIT (PREREG_V8F_PROOF_ADD4.md, frozen, not edited)
Everything in `F:/TanitAD-scratch/v8proof/bias/`. Nothing staged, committed or pushed; no GPU, ssh or pod. wsp/ untouched (imported read only).

Facts: STOP_POINT = index 6 of h_goal (v8f_mini.py:46 STOP_I; stopbal_read.py:56). Key `h_goal.bias`. W1 4.5616 (BAL-G), W2 20.81 (BAL-G2), identical in s0 and s1 argv.txt. ln W1 1.51767, ln W2 3.03543.

Built (DONE): `bias_build.py` -> `ckpt/GE{0,1,2}_s{0,1}` + `BUILD_REPORT.json`; control e1 passed (one scalar differs, delta equals ln W within 2 eps float32; GE0 none; strict-load round trip).
Self-test (DONE): GE2_s0 strict load + 4 windows, rc 0, 20.2 s (`selftest/`).
QUICK_OFFSET.md (DONE): logit-offset arithmetic on banked STOPDEC, a preview only.
NOT run: `bias_read.py` (needs real dumps; only imports and helper functions tested, no end-to-end smoke), `job_bias.cmd`.

Launch (Master Mind), only after the WSP job wrote `wsp/WSP_DONE.txt` (the job refuses otherwise):
`powershell -File F:\TanitAD-scratch\v8proof\bias\launch_bias.ps1`
Order: 9 dumps (GE2_s0 GE1_s0 GE0_s0 GE*_s1 NULL_s1 G_s1 G2_s1, 2 threads) -> STOPDEC of the same 9 -> bias_read.py -> `RESULT_BIASEDIT.json`, `VERDICT_BIASEDIT.md`. Markers `BIAS_DONE.txt` / `BIAS_STOP.txt`. Resumable (done files). ESTIMATED 1.2 to 1.5 h.

Reading choices (bias_read.py docstring): replicate gap is the seed-0 pair, used for both seeds; EQUALS needs all four (dose, seed) pairs; MATCHES needs all four; mixed pattern printed MIXED.
Risk: if bias_read fails on first real run, fix in bias_read.py only.
Temp (deletable by literal path after the read is banked): `selftest/`, `ckpt/` (6 x ~29 MB).
