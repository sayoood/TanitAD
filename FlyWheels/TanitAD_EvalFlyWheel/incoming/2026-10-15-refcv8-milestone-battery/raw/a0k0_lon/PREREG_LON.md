# PREREG_LON -- LONGITUDINAL-ONLY residual scaling for refcv8 (zero training, CPU, banked panels only)

Written 2026-10-19, **before any number of the new arms (RS_LON_XF, GATED_RS_LON_XF, RS_LAT1_XF) was computed on real data.**
Instrument: `PKG/code/a0k0_headroom_lon_r8.py` (a COPY of `a0k0_headroom_r8.py`; the original and its test are untouched provenance).
Parent registration: `raw/a0k0_headroom/PREREG.md` Amendments 2-3 (RS = residual scaling on refcv8's own causal prior `ha0_ext_pose`).
Tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**. Checkpoint: banked step 30,000 (EMA = headline
`step30000ema_unc`, raw `step30000_unc` beside). Estimator: paired episode-cluster bootstrap (`pair_read_r8.paired_cell` ->
`taniteval.ci`, B 2000, seed 0, models fixed) = **"another draw of EPISODES" only** -- NOT another training run (single training seed),
NOT another inference run (seed 1 is the replicate beside seed 0), NOT another fold assignment (fold-seed sensitivity reported).

## DISCLOSURE (binding)
This arm was **motivated by the RS_XF read on S6, seen before this registration** (`raw/a0k0_headroom/A0K0_HEADROOM.{json,md}`,
EMA-30k compliant): RS_XF - os is better on longitudinal (S6 ADE -0.1627 [-0.2464, -0.0763] s0) but separated WORSE on lateral
(heading +0.1773 deg [+0.0644, +0.3123], curvature +0.0007, yaw-rate +0.0019 at s0; vs refcv7-50,400 heading +0.3047 deg
[+0.0891, +0.5420]). The fitted RS gains were S2 along 1.00->0.41, cross 0.23->0.68. The hypothesis below is therefore NOT independent of
the data it will be scored on; it is a lever estimate, never a bar pass. A deployable version fits g on TRAIN, not on eval episodes.

## Hypothesis
Shrinking ONLY the along-track residual keeps the longitudinal / E1-form gain without costing turns.

## Arms (plans in the ego frame at t0; x = along, y = cross; prior = `ha0_ext_pose` exactly as in Amendment 2)
`p_k,c = g_k,c * os_k,c + (1 - g_k,c) * prior_k,c`, per instant k and axis c.
* **RS_LON_XF** -- along gain g_k,x in [0, 2] FITTED per instant; **cross gain g_k,y == 1.0 EXACTLY** (box (1,1)), so the lateral position
  p_k,y == os_k,y bit for bit. Fit = continuous minimiser of mean window ADE (separable per instant) by L-BFGS-B with analytic gradient,
  verified against a 0.1-step grid over the box (identical machinery to Amendment 2); 5 EPISODE folds (rng 20261018, fold = rank mod 5),
  fit on 4 folds, applied to the 5th; never fit on the scored episodes. Fit set = the scored window set (compliant fits on compliant).
* **GATED_RS_LON_XF** -- the same, one along-gain table per v0 bin (`s6_lon_decomp.V0_EDGES`); a bin with < 30 fit windows falls back to the
  fold's ungated LON gains.
* **RS_LAT1_XF** (optional; included -- same machinery, one more bounds spec) -- along gain free in [0, 2], cross gain in **[1, 2]** (never
  shrinks the lateral residual).
* **Control arms:** `RS_XF` (Amendment 2 arm, recomputed here: its three ADE cells must equal the banked `A0K0_HEADROOM.json` cells within
  1e-9), `RS_LON_IS` / `GATED_RS_LON_IS` (in-sample, known sign: <= prior and <= os in ADE) and `RS_LON_WF` / `GATED_RS_LON_WF` (per-WINDOW folds =
  leaky).
* **Cells:** both trees (EMA-30k headline, raw-30k beside), both inference seeds, S2 (0-2 s) and S6 (0-6 s), ceiling-compliant set
  (A-R8-CLAMP-2 mask, sha `4ebc6b86...`) PRIMARY and all windows beside. Every `pair_read_r8.METRICS` row (families ADE, LONGITUDINAL,
  LATERAL, TACTICAL; STRATEGIC NOT APPLICABLE, n = 0, layer off).

## Bars for these LEVER ESTIMATES (verdict mechanics = `pair_read_r8.bar_E` / `bar_NR_metric` / `rollup_NR`, both seeds, arm's own inference-seed floor, "NOT PROVEN" inside 2x the floor)
* **LE-E1**  arm - `ha0_ext` (registered reference), S2 ADE 0-2 s: < 0 and separated at both seeds.  **LE-E1P** arm - `ha0_ext_pose`, S2 ADE, same.
* **LE-E3**  arm - `ha0_ext`, S6 ADE 0-6 s: < 0 and separated at both seeds.
* **LE-NR**  arm - refcv7-50,400, every METRICS row, S2 and S6: headline-row rollup per family (no NR headline row separated WORSE at either
  seed); all rows printed.  **LE-OS**  arm - refcv8 `os`, same rows.
* **ACC-LAT (the ACCEPTANCE CONDITION for this arm, committed now):** on **S6, ceiling-compliant set, EMA-30k**, NO LATERAL-family row
  (`LAT_heading_mae_deg`, `LAT_curvature_mae_1pm_valid`, `LAT_yaw_rate_mae_radps_valid`, `LAT_cross_mae_m`, `LAT_cross_abs_final_m`) of
  arm - os is **separated WORSE (error metric: lo > 0) at EITHER inference seed**. This is stricter than the "both seeds" listing used in
  the RS_XF summaries. All-windows and raw-30k are reported beside and are not part of the condition. S2 LATERAL rows vs os are printed; a
  S2 row separated worse is reported as a regression but does not by itself fail ACC-LAT (defined on S6, where RS_XF failed).
* An arm that misses ACC-LAT FAILED it and is reported so; the goalposts are not moved after the data.

## Structural identities (not estimates; verified by control, any seed gives the same)
* With the cross gain == 1: `LAT_cross_mae_m` and `LAT_cross_abs_final_m` vs os read **exactly 0.0000 [0, 0]** (position y == os y). That is an
  identity, not evidence of "no lateral cost".
* Heading / curvature / yaw-rate rows are **derived from the plan's positions** (finite differences of x AND y): rescaling x at the
  instants changes the tangent even with y == os. So ACC-LAT is genuinely open for those three rows; they are the rows that decide it.

## Committed readings -- BOTH outcomes, written before the data
* **ACC-LAT holds AND LE-E1 separated at both seeds** -> longitudinal-only residual scaling keeps an E1-form gain without the S6 lateral regression:
  the deployable-lever candidate. Frozen gains (below) go forward to the 40k / 48,852 test. Still a LEVER estimate (calibration fit on eval-30k).
* **ACC-LAT holds, LE-E1 NOT separated** -> the RS_XF E1-form gain was carried by the CROSS shrink (S2 cross gain 0.23-0.68), not by the
  longitudinal residual; "longitudinal-only" does not clear the E1 form. Report the share RS_LON - os retains of RS_XF - os (point ratio, s0, descriptive).
* **ACC-LAT fails (any LATERAL row separated worse at either seed)** -> the S6 lateral regression is NOT only the cross gain: x-scaling alone
  rotates the plan tangent that heading / curvature / yaw-rate are read from. NEXT LEVER (named, not run here): scale the residual along the PRIOR's
  local path tangent instead of the t0 x-axis (or tie g_x across adjacent instants), + RS_LAT1 as the cheap beside; its cost is the same CPU
  machinery (~10 min per full tree set).
* **RS_LAT1_XF**: if its lateral rows are clean while RS_LON_XF's are not, the lateral cost is in the cross SHRINK; if they fail alike, in the
  along gain. An effect inside 2x the inference-seed floor is NOT PROVEN. Seed-0/seed-1 agreement is inference replication only.
* Not separated anywhere -> residual scaling restricted to the along axis buys nothing measurable at this n; a negative from a convex
  linear per-instant gain is a weak lower bound on a learned use (CLAUDE.md), not a statement that the longitudinal residual is worthless.

## Frozen gains + forward test (same method as Amendment 3; written before any step-40,000 / 48,852 data exists)
`FROZEN_GAINS_LON_ema30k.json` (written by `a0k0_headroom_lon_r8.py freeze`): `RS_LON` (along gains, cross == 1), `GATED_RS_LON` (per v0 bin;
bin < 30 unique windows -> RS_LON table), `RS_LAT1` (cross >= 1) -- fit on **EMA-30k, ALL windows of the surface (S2 4,754 / S6 3,668, all 139
eval episodes), BOTH inference seeds POOLED**, one table per surface grid, with fit set, n, input sha256s and script sha256. The forward test
(`forward --tree <EMA 40k / 48,852 tree>`) applies them UNCHANGED, prior rebuilt from that tree's own panel decisions; bars **FZL-E1, FZL-E1P,
FZL-E3, FZL-NR** as LE-* above and **FZL-LAT** = ACC-LAT on S6 (arm - os, either seed). Path checks run now (not bars): raw-30k (different checkpoint
of the same run, not independent) and EMA-30k (in-sample by construction).
