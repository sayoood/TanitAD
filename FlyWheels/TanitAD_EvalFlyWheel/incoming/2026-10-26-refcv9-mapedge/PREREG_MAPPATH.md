# PREREG_MAPPATH -- the own map chooses the PATH; the speed stays the native plan's (Master Mind, 2026-10-26 ~04:31 CET)

Sealed BEFORE any MAPPATH pick or bar exists. 0 GPU: the banked seed-0 MAPSEL trace + rasters.

## 0. What is known (disclosed)
* MAPSOFT-G2 (`PREREG_MAPSOFT_G2.md` af1626e9): FAIL (M4, M5) -- own 0.1710 vs deranged 0.1714 sidewalk; both soft costs shorten
  the plan (path -0.08 / -0.11 m, separated), so M5 fails and a deranged map earns the same gain.
* Descriptive (`diag/violation_geometry.py`, every 4th window, 1,144 windows with mixed candidates): the own map's gated (x >= 4 m)
  max sidewalk probability separates GT-violating from clean candidates at **per-window AUC 0.905 vs 0.712 for a deranged map** -- the
  own map IS window-specific. Violations are deep (median penetration 1.2 m) and far (63 % at 40-100 m, 28 % at 20-40 m). Reading:
  a fan candidate couples a path with a speed profile, so a map cost can be met by a SLOWER candidate that never reaches the far
  sidewalk -- which any map rewards. Same coupling `H-R9-SSGUARD-1` found for the lon head (heading cost of selection changes).

## 1. The only change: the fan is progress-matched before anything reads it
For every window, every candidate c: `fan'[c] = at_progress(fan[c], arc(fan[sel]))` (RT polyline machinery, linear extrapolation past
a candidate's end) -- each candidate's PATH sampled at the NATIVE emitted plan's per-instant arc length. The trace is rewritten with
`fan'` and everything else byte-identical, and `mapsoft_g2.py`'s procedure runs on it unchanged (same gate grid 0/2/4/6/10/20 m, same
cross-fit, objective, deranged control, arms GT-T1 / OWN-T1 / MAPSOFT-G-XF / DER-MAPSOFT-G-XF). Every arm's emitted plan therefore has
the native speed profile; only its path can differ. Deployable: the fan, the native selection and the own map are all model outputs.

## 2. Controls (a failed control voids the bars)
* **K-ID:** `fan'[sel] == fan[sel]` bit-for-bit on every window (the native plan is unchanged).
* **K-BASE:** this run's BASE sidewalk / edge / lane rates equal the registered MAPSEL BASE rates exactly (replaces MAPSOFT-G's S-K0
  reproduction, which by design cannot hold for the other arms -- their fan changed).
* **K-PROG:** for every arm, per-instant arc length of the emitted (pre-stage) plan equals the native plan's within 1e-4 m on >= 99 % of
  windows (extrapolation can differ only where a candidate path is degenerate; the count is reported).
* S-K1..S-K3 as MAPSOFT-G.

## 3. Bars (MAPSOFT-G's, verbatim; PASS = M1-M5)
M1 sidewalk vs BASE separated below; M2 edge vs BASE separated below; M3 no four-family row worse; M4 the DERANGED-map arm NOT
separated below BASE on sidewalk or edge; M5 the progress / STOP guard. ⚠️ M5 is near-structural here (progress is matched by
construction) -- it is kept, and its reading is stated as such, not counted as evidence.
Beside (descriptive): GT-T1 on the progress-matched fan = the PATH-ONLY ceiling; OWN minus DER point estimates; per-fold fits.

## 4. Committed next steps
* **PASS** -> the map-in-planner lever is the path/speed factorisation (no training): roll MAPSEL at inference seed 1 for the
  replicate, then compose with the factored lon decode (path from the map-aware selection, speed from the self-sharpened lon decode)
  as ONE pre-registered arm; SHIP-R8-style four-family read before any PI request.
* **FAIL M4** (deranged still matches) -> the own map's 0.905 AUC does not survive progress matching: the authorised map-only
  fine-tune is the lever, read with this instrument.
* **FAIL M1/M2** with the GT-T1 path-only ceiling ALSO small -> the sidewalk defect is a SPEED/reach effect, not a path effect: the lever
  moves to the lon side (the factored decode's speed) -- name it.
