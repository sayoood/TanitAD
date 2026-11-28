# PREREG_RETIME -- ARC-LENGTH RE-TIMING of refcv8's own plan (zero training, CPU, banked panels only)

Written 2026-10-19, **before any real-data number of the new arms (RETIME_XF, GATED_RETIME_XF and their _IS / _WF controls) was computed.**
Only synthetic fixtures (pytest) had been run when this file was written.
Instrument: `PKG/code/a0k0_headroom_retime_r8.py` (a COPY of `a0k0_headroom_lon_r8.py`; the LON script, `a0k0_headroom_r8.py` and their tests are untouched provenance).
Predecessors: `raw/a0k0_headroom/PREREG.md` Amendments 2-3 (RS = residual scaling on refcv8's own causal prior `ha0_ext_pose`); `raw/a0k0_lon/PREREG_LON.md` (longitudinal-only).
Tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**. Checkpoint: banked step 30,000 (EMA = headline `step30000ema_unc`, raw `step30000_unc` beside).
Estimator: paired episode-cluster bootstrap (`pair_read_r8.paired_cell` -> `taniteval.ci`, B 2000, seed 0, models fixed) = **"another draw of EPISODES" only** -- NOT another training run
(single training seed), NOT another inference run (seed 1 is the replicate beside seed 0), NOT another fold assignment (fold-seed sensitivity reported separately).

## DISCLOSURE (binding)
This arm is motivated by TWO reads seen before this registration, so its hypothesis is NOT independent of the data it will be scored on:
1. `raw/a0k0_headroom/A0K0_HEADROOM.json` (RS_XF, t0-frame residual scaling): clears the E1 form on S2 but is separated WORSE than `os` on S6 heading (+0.1773 deg s0).
2. `raw/a0k0_lon/A0K0_LON.json` (RS_LON_XF / GATED_RS_LON_XF, along-only scaling in the t0 frame, cross gain fixed at 1): **it FAILED its pre-registered ACC-LAT.**
   EMA-30k, S6, ceiling-compliant (n 3,251 windows / 138 episodes), arm - os, error metrics (lo > 0 = worse): RS_LON_XF heading +0.2346 deg [+0.1220, +0.3707] (s0) /
   +0.2326 [+0.1218, +0.3682] (s1); curvature +0.0007 [+0.0003, +0.0012] / +0.0007 [+0.0003, +0.0013]; yaw rate +0.0021 [+0.0009, +0.0035] / +0.0020 [+0.0009, +0.0035].
   GATED_RS_LON_XF and RS_LAT1_XF also fail on the same three rows. The cross-track rows read exactly 0 there (identity: p_y == os_y), so they carried no information.
   **Diagnosis (mechanism, not yet a measured attribution):** heading / curvature / yaw-rate are computed from the STEP GEOMETRY of the plan's positions at the S-grid instants
   (`four_families._seq_geometry`); scaling only the t0-frame x-component of each instant's residual moves each point OFF the path os drew, so the chords change direction.
The hypothesis below is a lever estimate fit on EVAL episodes (cross-fitted); it is never a bar pass. A deployable version fits g on TRAIN.

## Hypothesis
Keeping refcv8's PATH exactly (every re-timed point lies ON the os polyline) and changing only the PROGRESS along it (arc length) keeps the longitudinal / E1-form gain of the RS arms
without the S6 lateral regression.

## Definition (the code is held to this text; deviations are listed in "Where the geometry forced a choice")
For one window, instants t_1..t_S (S2: 0.5, 1.0, 1.5, 2.0 s; S6: 1..6 s), ego frame at t0, planned positions `os_k` in R^2.
* Vertices V_0 = (0,0), V_k = os_k (k = 1..S). **C** = the polyline V_0 V_1 ... V_S. Cumulative arc length L_0 = 0, **L_k = s_os(t_k) = sum_{j<=k} |V_j - V_{j-1}|**.
* The causal prior `ha0_ext_pose` (`kinematic_prior.prior_path`, the SAME base as the RS arms, evaluated at the same instants): W_0 = (0,0), W_k = prior_k;
  **M_k = s_prior(t_k) = sum_{j<=k} |W_j - W_{j-1}|**. Both arc lengths are chord sums on the S-grid, so they are measured identically (same sampling).
* **Re-timed progress** `s'_k = s_prior(t_k) + g_k (s_os(t_k) - s_prior(t_k))`, evaluated in the algebraically identical form `s'_k = L_k + (1 - g_k)(M_k - L_k)`
  so that g_k == 1 gives s'_k == L_k BIT FOR BIT. Clip: `s'_k >= 0` (lower). No other clip: the gain box g in [0, 2] bounds the over-run past L_k to (L_k - M_k)+.
* **Position** `p_k = C(s'_k)`. If s'_k < L_S: find j = max{ j <= S-1 : L_j <= s'_k } and `p_k = V_j + (s'_k - L_j) u_j`, u_j = (V_{j+1} - V_j)/|V_{j+1} - V_j|
  (linear interpolation along the polyline). If s'_k >= L_S: `p_k = V_S + (s'_k - L_S) u_end`, u_end = unit direction of the LAST POSITIVE-LENGTH segment, and +x (the t0 heading) if
  the whole polyline has zero length (counted and reported as `fallback_dir`).
* g_k is a SCALAR per instant (one gain, not one per axis).

## Arms (g fitted to minimise mean ADE on the FIT folds; separable per instant)
* Fit: F_k(g) = mean over fit windows of ||C(s'_k(g)) - G_k||, g in [0, 2]. **F_k is NOT convex in general** (the polyline bends), so the minimiser is a coarse grid (step 0.05)
  followed by a bounded scalar refinement in the best grid cell (scipy `minimize_scalar`, bounded, xatol 1e-9); the refinement is accepted only if it beats the grid minimum by > 1e-12;
  ties go to the g closest to 1. The pytest checks the fitted loss against a 0.002 brute-force grid (<= brute + 1e-6) and recovery of a planted g.
* 5 EPISODE folds (rng 20261018, fold = rank mod 5), fit on 4 folds, applied to the 5th; never fit on the scored episodes. Fit set = the scored window set (compliant fits on compliant).
* **RETIME_XF** -- one g per instant. **GATED_RETIME_XF** -- one g table per v0 bin (`s6_lon_decomp.V0_EDGES`); a bin with < 30 fit windows falls back to the fold's ungated gains.
* **Controls:** `RS_XF` (Amendment-2 arm, recomputed with the unchanged L-BFGS machinery: its cells must equal the banked `A0K0_HEADROOM.json` cells within 1e-9),
  `RETIME_IS` / `GATED_RETIME_IS` (in-sample; known sign: mean ADE <= os, because g == 1 is inside the box) and `RETIME_WF` / `GATED_RETIME_WF` (per-WINDOW folds = leaky).
* **Cells:** both trees (EMA-30k headline, raw-30k beside), both inference seeds, S2 and S6, ceiling-compliant set (A-R8-CLAMP-2 mask, sha `4ebc6b86...`) PRIMARY, all windows beside.
  Every `pair_read_r8.METRICS` row (families ADE, LONGITUDINAL, LATERAL, TACTICAL; STRATEGIC NOT APPLICABLE, n = 0, layer off).

## Controls (all pinned by `code/test_a0k0_headroom_retime_r8.py`, each with a deliberate-regression arm; (a), (b) also gate the real-data write)
* (a) g == 1 reproduces os: bit-equal after the cast to os's dtype (and <= 1e-6 m before it); on real data the g == 1 cells equal the banked os cells EXACTLY.
* (b) the re-timed path lies ON the os polyline (union its final extension ray): max point-to-path distance <= 1e-6 m over all windows, on the float64 positions. After the cast to float32 the
  bound is the float32 rounding (<= 2^-23 * max|p|), reported beside.
* (c) on a fixture circle of radius R sampled finely, every re-timed point is at distance R from the centre up to the chord sagitta R(1 - cos(dtheta/2)) (exactly R at the vertices).
* (d) DELIBERATE REGRESSION: the t0-frame along-only scaling of the LON script FAILS (b) on the same fixture (distance >> 1e-6 m).
* (e) g == 0 puts every point on the os polyline at arc length s_prior (independent loop-based locator in the test).
* Cross-fit hygiene: gains for fold f are unchanged when fold f's ground truth is perturbed; the _WF (leaky) arm must come out <= the _XF arm in mean ADE.

## Diagnostics reported (NOT bars; they describe the geometry, they do not move any verdict)
`frac_nonmonotone_windows` (s'_k decreasing in k for some k -- possible because g is free per instant), `frac_extrapolated_points` (s' >= L_S), `frac_fallback_dir_windows`,
max |g - 1|, and the fitted g per instant / per v0 bin.

## Where the geometry forces a choice (disclosed, not adjusted after data)
1. **Chord geometry.** On the S-grid the polyline has only S chords (S2: 4, S6: 6). A re-timed point is ON the polyline but generally BETWEEN vertices, so the chord between two re-timed
   points is NOT an os chord: the heading / curvature / yaw rate computed from the re-timed points can still change even though path membership is exact. "Shape preserved" therefore means
   path membership (control (b)), not invariance of the discretely-sampled heading. This is exactly why ACC-LAT below is a real test and not an identity.
2. **The cross-track rows are no longer structural zeros.** Moving along a CURVED path changes the lateral coordinate of the point. ACC-LAT keeps all five LATERAL rows.
3. **Arc length is the chord sum on the S-grid** for both os and the prior (the prior is not densified), so the two progress curves are measured on the same sampling.

## Bars for these LEVER ESTIMATES (verdict mechanics = `pair_read_r8.bar_E` / `bar_NR_metric` / `rollup_NR`, both seeds, arm's own inference-seed floor, "NOT PROVEN" inside 2x the floor)
* **LE-E1**  arm - `ha0_ext` (registered reference), S2 ADE 0-2 s: < 0 and separated at both seeds.  **LE-E1P** arm - `ha0_ext_pose`, S2 ADE, same.
* **LE-E3**  arm - `ha0_ext`, S6 ADE 0-6 s: < 0 and separated at both seeds.
* **LE-NR**  arm - refcv7-50,400, every METRICS row, S2 and S6: headline-row rollup per family (no NR headline row separated WORSE at either seed); all rows printed.
  **LE-OS**  arm - refcv8 `os`, same rows.
* **ACC-LAT (the ACCEPTANCE CONDITION, identical to PREREG_LON):** on **S6, ceiling-compliant set, EMA-30k**, NO LATERAL-family row (`LAT_heading_mae_deg`,
  `LAT_curvature_mae_1pm_valid`, `LAT_yaw_rate_mae_radps_valid`, `LAT_cross_mae_m`, `LAT_cross_abs_final_m`) of arm - os is **separated WORSE (error metric: lo > 0) at EITHER inference seed**.
  All-windows and raw-30k are reported beside and are not part of the condition. S2 LATERAL rows vs os are printed; a S2 row separated worse is reported as a regression and does not by
  itself fail ACC-LAT. **Vacuity guard (committed now):** an ACC-LAT HOLD counts as a finding only if the arm actually moved (LE-OS: arm - os ADE on S6 better and separated at both seeds);
  otherwise it is reported as "holds vacuously".
* An arm that misses ACC-LAT FAILED it and is reported so; the goalposts are not moved after the data.

## Committed readings -- BOTH outcomes, written before the data
* **ACC-LAT holds AND LE-E1 separated at both seeds** -> progress-only re-timing keeps an E1-form gain without the S6 lateral regression: the deployable-lever candidate. The frozen
  gains (below) go forward to the 40k / 48,852 test. Still a LEVER estimate (calibration fit on eval-30k).
* **ACC-LAT holds, LE-E1 NOT separated** -> the E1-form gain of RS_XF was carried by the lateral/cross components, not by progress along the path; report the share RETIME - os retains of
  RS_XF - os (point ratio, s0, descriptive).
* **ACC-LAT FAILS (any LATERAL row separated worse at either seed on S6 compliant EMA-30k)** -> **the next step is NOT another post-hoc arm.** It is the PI decision on a learned
  calibration (refcv8.x / refcv9): a calibration trained on TRAIN with the lateral rows in its loss, which a per-instant convex gain fit on eval episodes cannot honestly stand in for.
  No fourth hand-crafted rescaling variant (tangent-direction scaling, tied gains, smoothed polyline, densified prior, monotone projection) will be run to rescue this family.
  A negative from this per-instant scalar gain is a weak lower bound on a learned calibration (a negative from a restricted function class is not a negative about learnability).
* Not separated anywhere -> progress-only re-timing buys nothing measurable at this n.
* An effect inside 2x the inference-seed floor is NOT PROVEN. Seed-0/seed-1 agreement is inference replication only.

## Frozen gains + forward test (same method as Amendments 3 / PREREG_LON; written before any frozen number exists)
`FROZEN_GAINS_RETIME_ema30k.json` (written by `a0k0_headroom_retime_r8.py freeze`): `RETIME` (g per instant) and `GATED_RETIME` (per v0 bin; bin < 30 unique windows -> RETIME table), fit on
**EMA-30k, ALL windows of the surface (S2 4,754 / S6 3,668, all 139 eval episodes), BOTH inference seeds POOLED**, one table per surface grid, with fit set, n, input sha256s and script
sha256. `forward --tree T` applies them UNCHANGED (prior rebuilt from that tree's own panel decisions; the plan is that tree's own os). Run in this deliverable on `step30000ema_unc`
(in-sample by construction) and `step30000_unc` (a different checkpoint of the SAME run: a path check, NOT an independent test, NOT a bar). Bars FZR-E1, FZR-E1P, FZR-E3, FZR-NR as
LE-* above and **FZR-LAT** = ACC-LAT on S6 (arm - os, either seed). **The genuine forward test (EMA 40k / 48,852) is NOT run here**: no such tree exists on this box; the frozen file is the
input it will take.

## Compute protocol
CPU only (`CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=4`), ONE compute at a time, gated on `Win32_OperatingSystem.FreeVirtualMemory >= 6 GB`. Fold-seed sensitivity (10 fold seeds,
points) is computed for RETIME_XF and (5 seeds) GATED_RETIME_XF on EMA-30k inference seed 0 only. Order: compute -> freeze -> forward step30000_unc -> forward step30000ema_unc -> render -> brief -> pytest.
