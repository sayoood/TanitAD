# PREREG_REFCV9_INMODEL_PA_AMENDMENT1 -- K-OFF / K-EQ made executable; the R9 RNG defect fixed (Master Mind, sealed 2026-10-26 14:08:29 Berlin)

Amends PREREG_REFCV9_INMODEL_PA (67d17860...). Written after the executor's pre-GO checks on **70 CALIBRATION windows only**; no eval139
R9 window output exists. **Controls only -- every bar (PA-1a / 1b / 2 / G / R), arm, population and estimator is unchanged.**
(A first draft of this file, sha 4009e063..., was garbled by shell expansion and is superseded unused; this text replaces it.)

## What the calibration checks MEASURED (executor, dev-box GPU, seed 0)
* (a) The tip tree's planner (no R9 head) selects a different path from the fb47a58-rolled bank on 70 / 70 windows (max 7.24 m), with z_tac
  bit-equal and the run deterministic. Planner-path code changed between fb47a58 and the tip (`kinematic_prior.py`, `refc_v3.py`,
  `refcv6_selection.py`, `refc.py`). Investigated separately (H-R9-TREEDRIFT); NOT an R9 effect.
* (b) DEFECT in the banked R9 code: `enable_refcv9` calls `torch.manual_seed` inside `fork_rng(devices=[])`, which re-seeds the CUDA RNG and
  does not restore it; merely building the head moved the selected path on 70 / 70 windows (max 8.10 m). Fix:
  `torch.default_generator.manual_seed` (CPU generator only) -> flags-on path == flags-off path bit for bit (0 / 70 differ).
* (c) K-EQ same-input max |d| = 1.53e-5 m at 128-256 m coordinates = one float32 ULP; 100 % decision agreement. The sealed 1e-5 m is
  below float32 resolution there.

## Changes
1. **The R9 code is the banked patch + the one-line RNG fix (b).** The fixed file's diff and sha256 are pinned in `FIX_R9_RNG.sha256` before
   any eval output; the composition refuses to run if it does not re-hash.
2. **K-OFF (amended):** on ONE tree (tip + patch + fix), flags-ON (head built, decision applied) vs flags-OFF: the pre-re-timing selected
   path (`traj_preretime`) and every non-R9 output are bit-equal on the K-EQ subset (>= 500 eval windows, both inference seeds). The sealed
   clause "== the banked os-only plan" is DROPPED because of (a), which is a cross-tree difference, not an R9 one.
3. **K-EQ (amended):** same-input form -- the in-forward R9 plan vs the offline composition on the forward's OWN selected path;
   max |d| <= 2 x ulp_float32(max |coordinate|) per window, AND 100 % decision agreement, on >= 500 eval windows.
4. **The bars are computed by the offline composition on the BANKED fb47a58 paths** (the same paths as every reference arm:
   H2CN_DEPLOY8 / FACTORED_DEPLOY8 / RETIME_DEPLOY8), exactly as the sealed s1 allows. The in-forward roll serves K-OFF / K-EQ only.
5. Disclosed beside every result: (a) and (b); and the executor's early fit-side flag that R9PF-CONSTZ (no vision) has a better
   out-of-fold CV objective than R9PF (1.83-1.85 vs 1.95-1.97) -- K-CONSTZ and the INERT branch decide it on eval, as sealed.
