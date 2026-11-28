# YF3 smoke results (2026-11-07; 110 A5 tokens over 11 logs, 4-step arms; tier: NAVSIM v1.1 open-loop plan vs logged agents)
Attempt 1 (18 tokens, `smoke_attempt1_18tok_FAILED/`) found 2 faults; attempt 2 (`smoke_run2/`, 110 tokens) is the pipeline of record. Numbers from smoke arms are mechanics only; no accuracy is claimed.
| step | result (MEASURED) |
|---|---|
| Task 1, `test_yaw_ft.py` rerun (synthetic cache, real u400 weights) | 19/19 as required, `test_yaw_ft_rerun.out` |
| Task 1, `real_cache_gates.py` on the REAL u400 cache (all gates that read it, each with its RED arm) | 10/10 as required: G2 corr(omega, ay/v)=0.9569 (n 2,499; bar 0.85), corr(omega, yaw@4s)=0.7387; negated omega RED. G3 fingerprint 160b61bc == u400 == model_final, frames 4,000, 0 visual, C1 0.0; ep015 cache e155e527 RED. G4+G5 green (512 frames, grad_table 0.1076, control 0.0); `default_col` RED, `table_not_applied` RED; G9 2 % tokens dropped -> TableRefused. `gates_real/real_cache_gates.json` |
| stats + rule (`test_yf3_analyze.py`) | 15/15: estimator == `proxy_eval.boot` on the banked SFT-4 contrast (mean 3.041874, CI [1.209549, 4.865133]) to 1e-9; per-window curvature/heading/yaw/cross == banked families6 (0.01882 over 5,760 pairs, 1.9235, 2.6396, 0.2761); ADOPT / NOT PROVEN / REFUTED / VOID branches incl. swapped-arms RED |
| G6 (A0 through the YT launcher path) | 110/110 plans AND scores bit-identical to the banked seam/CSV |
| G7 live (R1_B_s0) | shim installed once, 110 calls / 110 tokens, ego[4] == float32(10 x navtest table) on 110/110, 100 % non-zero, n_incoming_nonzero 0, scale 10.0, navtest table sha f5ebe24f..., train sha 517c6413...; A0: shim NOT installed, 0 calls |
| G8 | CPU: B s0 rerun `torch.equal` (12 threads); eval: 28-token seam rerun bit-identical plans and ego[4] |
| identity | C == A0 state_dict (R1), B0 == A0 decode (omega 0 and != 0), trained decode == composed-ckpt decode; R2 C and B composed + identity OK |
| RED arms eval/identity side (`smoke_reds_nonlive.out`) | 11/11: G6 wrong ckpt RED; G7 absent shim / yaw arm judged non-yaw / one token x0.5 / all-zero ego[4] RED; B ckpt presented as C RED; B0 default column RED |
| R1 analysis | produced `analysis/R1.md`: NOT PROVEN (CI straddles 0, ~0 effect after 4 steps), all 9 gate lines GREEN, 4 families + tier + estimator printed; chain advanced to R2 (Rule Zero) |
| R2 analysis branch (Ref = same-seed C) | run on a stand-in (A0 copied as the C eval) in `_scratch/r2_analysis_test/`: works; the real smoke C eval is still queued |
| FAULTS found and fixed (AMENDMENT_1 B.3, B.4) | (1) `build_yaw_ckpt.py` K4 was mis-specified for arm C -> the C checkpoint could not be composed; fixed in `build_yaw_ckpt_yf3.py`. (2) the adapter's statistical frame guard fires at n=18; kept ON for n >= 100. Also: harness RAM guard retry, ascii-safe failure print, label tag |
| NOT yet run | live G7 refusal in the real seam (`yf3_smoke_reds.py --live`), smoke R2 C eval, smoke R3 train/ckpt/identity (queued behind the GPU lock) |
