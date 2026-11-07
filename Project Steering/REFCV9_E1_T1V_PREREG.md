# T1v — supervision-only history mask: pre-registration (frozen 2026-11-07 11:15Z by the clock, before any T1v code, build or data)

Written by the Master Mind before T1v exists in any form (no flag, no argv, no record, no run). The bar below is T1h's, unchanged.

## Why (each input MEASURED, with its artifact)
1. **T1h's gain is the SUPERVISION, not the history content.** H-R9-E1-T1H (GOALS 7b8bccf3): T1h vs C0 `nf_prec_0_10` +0.2951 [+0.2337, +0.3514]
   (KEEP, clauses 1-3; `e1/verdicts/T1h/`); clause 4 NOT MET: T1hs (deranged donor-episode history, the same supervision set, fuse and init) vs
   T1h +0.0152 [+0.0027, +0.0274], CONTROL-READ (`e1/verdicts/T1hs/VERDICT_T1hs.json` md5 6c3e53f5).
2. **The fuse was barely used:** T1h's final |gate| mean 0.0264, max 0.0598, 0 / 64 channels > 0.1 (`e1/t1_fuse_gate/RESULT_T1h_model_ema.json`).
3. **The warp is the cost.** Thor run dirs `r9e1/runs/r9e1-<arm>/`, wall = summary.json mtime - config.json mtime over 4,000 steps (in-run evals
   included, one run each, same rig): C0 4.416 s/step, C1 4.416, T1 5.013, **T1h 5.086**, T1hs 5.012. The T1 path (past-frame lift of the K = 3
   window's s8 features + warp + fuse) costs +0.60 to +0.67 s/step (+13.5 % to +15.2 %); T1h's supervision adds ~0.07 on top.
4. **The supervised set is GEOMETRY ONLY.** `bev_warp.hist_valid_of` builds `hist_valid` from the poses `T`, the camera's own lift `valid` and
   the lags' `has` (`stack/tanitad/models/bev_warp.py:134-135, 203-205` at 7b8bccf3); the warped CONTENT never enters it. The past-frame encode /
   lift (`bev_warp.py:187`) exists only to produce the content that (1) and (2) say is not what moved the primary.

## Arm
| arm | argv | what it isolates |
|---|---|---|
| **T1v** | C0 + `--r9-hist-supervise-geom` | T1h's supervision set (lift_valid_now OR hist_valid, at the 0.25 m lift grid, the same lags 0.5 / 1.0 s, the same `has` gating) with NO past-frame encode or lift, NO warp of features, NO fuse, NO gate |

Implementation contract (checked before any Thor time, else T1v does not run): on the same batch, T1v's `hist_valid` equals T1h's BIT FOR BIT (a
test builds both); the T1v build carries no `bev_warp` parameters; with the flag absent every arm's build, loss and gradients are bit-identical
to the tip's. One training run (seed as T1). Rig, windows, estimator, floor = E1's (REFCV9_DESIGN 4.3): T0 label level, 2,308 DEV windows /
70 episodes, paired episode-cluster bootstrap B 2000 seed 0, floor = |C1 - C0| (one seed pair).

## Keep rule vs C0 (T1h's, NOT relaxed)
T1v is KEPT iff, against **C0**: (1) `nf_prec_0_10` separated in the good direction, AND (2) |delta| > 2 x floor (2 x 0.0457598 = 0.0915196),
AND (3) no other perception primary separated worse beyond 2 x its floor.

## Replacement read vs T1h (decides K2's composition; committed now)
T1v REPLACES the T1h bundle in K2 iff T1v is KEPT (1-3 vs C0) AND, paired vs **T1h** on the same windows: (R1) `nf_prec_0_10` T1v - T1h is NOT
separated worse beyond 2 x floor (it fails R1 only if the CI's upper end is < 0 AND |delta| > 0.0915196), AND (R2) no other perception primary
T1v - T1h is separated worse beyond 2 x its floor. Reported beside (not bars): `nf_prec_0_5`, `nf_prec_5_10`, box AP@1 / 2 m by range, and
**the cost read**: T1v s/step by the same mtime method, against C0 4.416 and T1h 5.086 (expected near C0's: HYPOTHESIS until measured).
Four families: LONGITUDINAL / LATERAL / TACTICAL not computable on the rig (no planner, n = 0); STRATEGIC PAUSED (PI 2026-09-19).

## Committed outcomes and next levers
* **KEPT and replaces (1-3, R1-R2)** -> K2 composes `--r9-hist-supervise-geom` INSTEAD of the four T1h flags (the full-family gate admits it;
  the T1h bundle stays admitted, unused). The full-model saving is MEASURED at K2's explore smoke, never assumed from this rig.
* **KEPT, fails R1 or R2** -> the fused content / copy init / fill-only carries part of the gain despite the shut gate: K2 keeps the T1h bundle;
  T1v is logged as the supervision share of T1h's gain.
* **FAILS 1-2 vs C0** -> the supervision set alone does not carry the gain, which contradicts T1hs: diagnose (mask identity vs T1h on the
  trained run's batches; the init / fill-only path) before any claim; K2 keeps the T1h bundle.
* **FAILS 3** -> the vetoing row is named; K2 keeps the T1h bundle.

## Schedule (committed)
T1v trains in the SAME chain segment as the selected M3 arm (PREREG_M3: M3 or M3N20 by M1N20's committed reading), its records bound in the
same queue, so K2 is composed from E1-read levers only (the full run inherits K2's config; swapping T1v in after E2 would leave the full run
un-screened). Cost: its own box record (~1.4 h) + map record or reuse (the reuse rule is the gate's, decided by the code) + ~4.4 h training
+ the read. Code lands only AFTER M1N20's chain restart (a closure-module land before it voids queue 14's records:
memory land-closure-change-voids-bound-records).

## Caveats committed now
* One training seed per arm; the floor is ONE seed pair (H-ESTIM-SEED-1): separated is necessary, not sufficient.
* The banded K2 variants need the same geometry-only mask inside the banded loss (`bev_warp_flag.supervised_banded_unobs`, c2a2eeb6): a code
  item of the same build, tested the same way.
* The cost numbers above are wall time including in-run evals (the same cadence for every arm), not a profiler read.
