# E2-RAW stage A (map only): K2DC and K0 re-read on their RAW step-6,000 weights (2026-11-16)

Pre-registered before any raw number: `PREREG_E2RAW_2026-11-16.md` (this folder, landed 232f8738). Diagnosis: `../diag_k2family/`.
Zero training. Thor, read tree `read_tree_9697fa4df004_k2dc` (fitter / dump / eval md5 = the 11-12 diag tree). DEV = eval139 even half
(`DEV_HOLD_SPLIT.json`), 70 clips. Tier **T0** (perception). All numbers MEASURED from the files named; CI = `r9_e1_eval.py` bootstrap,
B 2000, seed 0, per arm (NOT paired). One training run per arm (`H-ESTIM-SEED-1`).

## Clause (a): `nf_swedge_prec_lv_5_20` at each arm's OWN FIT thresholds (design 2.1; E1's protocol)

| arm | EMA (`model_ema.pt`, E2's read) | RAW (`ckpt.pt`) | EMA -> RAW |
|---|---|---|---|
| K2DC | 0.3940 [0.3184, 0.4669] (`ema_control_K2DC/EVAL_K2_fitthr.json`) | **0.4912** [0.4037, 0.5697] (`stageA/EVAL_K2raw_fitthr.json`) | **+0.0971** |
| K0 | 0.4776 [0.3953, 0.5491] (`ema_control_K2DC/EVAL_K0_fitthr.json`, the 11-12 K0 diag, same dump inode) | **0.4729** [0.3919, 0.5443] (`stageA/EVAL_K0raw_fitthr.json`) | -0.0047 |

* **Clause (a) on raw weights: PASS** (point rule as pre-registered): K2DC 0.4912 vs K0 0.4729, +0.0182. The two
  per-arm CIs overlap almost entirely; no paired interval was computed, so this is a point result, not a separated one.
* **Mechanism prediction (CONFIRMED)**: K2DC gains +0.0971 (bar >= +0.03) while K0 moves -0.0047 (bar |.| <= 0.01). The EMA read
  (decay 0.9998 over 6,000 steps, shadow started from the step-0 weights: 0.301 of the init remains) under-read K2's new modules and not K0's.
* Default cut (reported, decides nothing): K2DC raw 0.4874 [0.4014, 0.5643], K0 raw 0.7532 [0.6480, 0.8330]
  (`stageA/EVAL_*raw_default.json`). At the default cut K0 still leads by a wide margin: the two heads sit at different operating points.
* FIT thresholds (`stageA/thr_*raw_pswpedgeF.json`): K2DC raw psw/pedge from STATUS.txt; K0 raw pedge [6, 7, 5] vs K2DC raw [111, 155, 137].

## Provenance

* Raw guard: every raw `dump.json` records `weights_are_ema: false`, step 6000 (K2DC ckpt md5 6848ca80fa7e2e7d4784982ecab81ede, K0
  aed728c6765725b94b5a67ae8bf0f04d). Scripts: `thor_scripts/e2raw_stageA.sh` (stage A), `e2_fitthr_diag_k2dc.sh` (EMA control),
  `e2raw_stageB_launch.sh` (stage B, refuses unless (a) passed).
* Pulled 2026-11-16 ~18:10Z; the EVALs, thresholds, STATUS and dump headers are sha256-equal to Thor. Rasters / perc_det.pkl not banked
  (re-creatable, ~GBs; Thor `runs/E2RAW_K2DC/stageA/`).
* Next (pre-registered): stage B, the planner clauses (c) S6c, (d) four families, (e) W4 on raw `ckpt.pt`, seeded with these DEV dumps and
  FIT-threshold EVALs; started 2026-11-16 18:06:29Z. A GO goes to the PI: the original E2 NO-GO verdicts stand as written.
