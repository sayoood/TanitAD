# SPEC addendum S2-A2 — E2′ (speed leak against the ROAD-LEVEL route, certified instrument) and E3′ (lateral leak per support class)

*Written 2026-10-04 by the Data FlyWheel at the Master Mind's instruction, after Stage 2 and BEFORE any E2′/E3′
number exists. Basis: the Master Mind's provisional ruling (to be confirmed by the PI): **road-geometry
(curvature-ahead) speed information is admissible inside a route input; lane-level ego choice (within-lane offset,
nudges, the ego's own speed) is not.** ⇒ the leak reference is the heavy road-level route, not NAV. sha256 in
`raw/SPEC_SHA256.txt`; the Master Mind registers it in the programme register.*

## Inputs under test (unchanged from SPEC §6, read from the same builder)
* **RC_noised(v)** — the σ = 8 m checkpoint of variant v ∈ {A50 (adopted, tentative), B (runner-up), A30, A80}, with
  the training noise applied in the route-tangent frame: along-track N(0, 2.0 m), lateral N(0, 0.75 m), seed
  20261005, heading unchanged.
* **HEAVY(v)** — the σ = 25 m road-level route at the SAME arc length (RC-H), (x, y, ψ). It is the reference: it
  carries road geometry and no lane-level detail.

## Population and the instrument (fixed now)
* **Population:** all 4,369 TRAIN clips (≥ 2,000 required), **one window per clip per 2 s**: the windows whose dataset
  index t satisfies t mod 20 = 0 (NOW spaced 2.01 s), with the 6-s future valid in the log and every input valid.
  Expected n ≈ 35,000–38,000 windows (ESTIMATED from 171 windows/clip / 20).
* **Model:** `HistGradientBoostingRegressor(max_iter=200, max_depth=6, learning_rate=0.05, random_state=0)`, 5-fold
  clip-grouped OOF, targets v(NOW+τ), τ = 1…6 s; ρ = (R²(F) − R²(F0)) / (1 − R²(F0)), mean over τ; F0 = {v0}.
* **Certification BEFORE any reading (both must pass, else nothing below is read):**
  (C1) deranged control — F0 + RC_noised(A50) taken from a seeded clip derangement (seed 20261005): **|ρ| ≤ 0.01**;
  (C2) target itself — F0 + v(NOW+τ): **ρ ≥ 0.99**.
  Positive control (reported, threshold fixed now from the Stage-2 readings 0.28–0.38): F0 + O1 oracle ρ ≥ 0.20.
* **If C1 or C2 fails:** one pre-registered fallback instrument — the same trees with `min_samples_leaf=200,
  l2_regularization=1.0`, re-certified on C1/C2. If that fails too: E2′ is BLOCKED ("no certified nonlinear
  instrument on this corpus"), reported with the control values; the OLS reading is printed beside it but decides
  nothing.

## E2′ — the bar
For each variant v: **ρ(F0 + NAV + RC_noised(v)) − ρ(F0 + NAV + HEAVY(v)) ≤ 0.01** under the certified instrument.
Also reported (no bar): ρ(F0 + RC_noised(v)) vs ρ(F0 + HEAVY(v)); the clean σ = 8 m RC against HEAVY; the OLS
readings. PASS ⇒ the noised checkpoint carries no more future-speed information than the road-level route.

## E3′ — lateral leak per support class (the S2-A1 classes), on the CLEAN checkpoint
The information content is measured on the CLEAN σ = 8 m point (the training noise only adds jitter and is not part of
the leak question). δ = signed offset of RC(v) from HEAVY(v) at the same arc; populations eval139 (all windows) and
the E2′ train population.
* **Straight-support windows** (HEAVY's heading range ≤ 10° over its kernel support): **p90 |δ| ≤ 1.0 m** (below the
  ≥ 2.5 m of a lane change and below a lane-level map + GNSS error).
* **Lane-level leak on straight support:** median |δ| on windows with a lane-change-scale excursion (D2-style detrended
  ≥ 1.5 m, |net Δψ| < 15° over [NOW, NOW+8]) **≤ 0.5 m**.
* **Curved-support windows: no bar** — the σ = 25 m reference cuts corners by metres there (Stage 2: p90 4.6–6.9 m),
  so geometry and lane choice cannot be separated with it. Reported: p90 |δ| and the LC-scale vs no-excursion median
  difference within 10°-wide curvature bins of the support range (descriptive).

## Decision (fixed now)
RC-A50 noised is confirmed as the training input iff E2′ PASSES for A50 and E3′'s two bars PASS for A50; else RC-B
under the same rule; else the RC returns to the Master Mind with the numbers.
