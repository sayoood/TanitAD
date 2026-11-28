# PREREG_L1 — AMENDMENT 1 (control K4 at inference seed 0 was mis-specified; sealed BEFORE any head is fit or scored)

*2026-10-25 ~11:30 Berlin. Base prereg `PREREG_L1.md` sha256 41ba4b2ff2443520c334a6f713acf1dcc1894adfbd3e134f2d91de8d6d91c8cf (sealed 10:10).
At the time of writing, the eval seed-0 bank exists; the eval seed-1 and train banks are still rolling; NO head has been fit and NO arm
has been scored on 48,852 data. Nothing below changes an arm, a bar, a split or a grid.*

## What happened (MEASURED)

* K4 as written (bank OWN plan == the banked battery `dump_s0` `nav_true.traj`, ≤ 1e-5 m) **FAILS at seed 0**: max |diff| **35.308 m**,
  2.6 % of windows within 1e-5, `sel_idx` agreement 0.749 (`bank/ztac_eval_s0.npz` vs `step48852ema_pf/dump_s0/refcv6_extras.npz`).
* It fails **by construction**, and this was documented before L1 existed: `r8oracle/PREREG_R8_ORACLE.md` §C1 — *the banked `dump_s0`
  came from the FULL arm-set roll (3 fed rows + a nav-null call per window: a different batch shape and sampler-stream consumption),
  so an os-only seed-0 roll is a fresh inference draw*; r8oracle measured the identical max |diff| 35.308 m at seed 0 and made seed 1
  its binding reproduction control. I did not read that clause before sealing — a prereg error (wrong reference artifact), not a defect
  in the bank.
* **The bank reproduces the os-only seed-0 child exactly**: OWN plan vs r8oracle's os-only seed-0 dump
  (`r8oracle/run_48852ema_pf/dump_s0/refcv6_extras.npz`, same os-only child argv, same seed, same windows): max |diff| **0.0 m**,
  `sel_idx` agreement **1.000**, n 4,754.
* Batch-shape numerics: the tactical logits of the os-only forward vs the full-roll forward differ by ≤ 6.5e-4 (lat 5.9e-4, lon 6.5e-4);
  `z_tac` sits upstream of the sampler, so its seed-0 value differs from the full roll's nav_true row only at this bf16 batch-shape level.

## The amendment

1. **K4 at seed 0** := the bank's OWN plan equals the **os-only** seed-0 roll of the same child (`r8oracle/run_48852ema_pf/dump_s0`),
   max |diff| ≤ 1e-5 m, aligned on (clip_index, ws). K4b (bank slots 0–3 == panel os) at seed 0 is reported, not binding (the panel's os
   at seed 0 is the full-roll draw).
2. **K4 at seed 1 stays as written** (bank OWN plan == banked `dump_s1` ≤ 1e-5 m, and K4b ≤ 1e-5 m) and is the binding battery anchor,
   exactly as in r8oracle.
3. **K6 becomes binding**: `z_tac` from the two os-only eval rolls (seeds 0 and 1, same windows) must agree to max |Δ| / rms(z) ≤ 1e-3
   (z_tac is deterministic upstream of the sampler). If it fails, each seed is scored with its own z_tac (already the design) and the
   failure is reported beside; the verdict is unaffected.
4. **Scoring is unchanged**: each seed's arms re-time that seed's **banked** os plan (the SHIP-R8 reference draws; K1 anchors H0 to the
   banked `FORWARD_TRAINFIT` cell). At seed 0 the head reads the os-only roll's z_tac beside the full-roll plan; the latent mismatch is
   the batch-shape numerics above (≤ 1e-3-level on logits), stated as a caveat.

Code changes implementing this (no definition changed): `code/l1_fit_score.py` bank-control block (K4 reference per seed, K6 flag),
`code/l1_verdict.py` (K6 in the control set). Logged in `DEVIATIONS.md`.
