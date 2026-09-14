# Semantic-BEV calibration: the idea works, and it refutes the premise that blocked us

**Evidence class: MEASURED (ours, synthetic)** — `bev_calib.py`, `test_bev_calib.py`, raw output in
`raw/`. Nothing here has yet touched the real recording; §7 says exactly what that would change.

---

## 1. The idea (Sayed, 2026-09-13)

> Use a strong pixel-level semantic model (SAM3) to extract ground features — asphalt, road markings,
> painted symbols — then treat calibration as an **optimisation**: with the vehicle's motion known,
> the correct calibration is the one under which the same static ground features, seen from many
> frames, land on top of each other in a common BEV.

## 2. Why this targets the actual failure

Every automatic calibrator on the 2026-08-08 recording declined or disagreed, and they all failed
the *same* way — too few reliable ground correspondences:

| estimator | result | why it declined |
|---|---|---|
| `plane_calib` | height 1.691 m | only **14** homographies, spread ±0.56 m |
| `scale_calib` | f·h 1608.1 | 1576 tracks but **62 %** spread |
| FOE vs lane VP | pitch −2.93° vs horizon row 523.4 | **2.29°** apart |

Semantic masks replace weak corner association on low-texture asphalt with dense, *identifiable*
structure. That is the correct thing to fix.

## 3. The method

`bev_calib.py`. Front-end agnostic on purpose: it consumes per-frame arrays of ink pixel coordinates,
so a SAM3 mask, a classical top-hat threshold, or hand labels all drop in unchanged. The projection
is `trajlib.camera`'s, **verified to 0.000000000 px** against it — calibrating against a different
camera model than the one that renders would be silent and fatal.

## 4. ⚠️ Two design claims of mine, both refuted by the test before any real data

This is why the synthetic harness was built first.

**Refutation 1 — "metric ego-motion kills the collapse degeneracy".** I argued that shrinking the
height could not win, because the pose transform translates by *unscaled* metric motion. The first
objective was plain concentration, `Σ(Σᵢmᵢ)²`. MEASURED:

```
truth                      2.313118e-05
height -> 0.05 (collapse)  3.612014e-04     <-- 15x BETTER than the truth
```

Wrong because concentration is maximised by shrinking **each frame's own footprint**, whether or not
frames agree. Fix: exclude the self-term, scoring only cross-frame overlap.

**Refutation 2 — "excluding the self-term is enough".** It was not; collapse still won **3.8×**
(7.69e-05 vs 2.02e-05). Diagnosed rather than guessed again:

```
truth            on-grid  14.3%   per-frame x-span 16.20 m
collapse h=0.05  on-grid  96.7%   per-frame x-span  5.54 m    (frames 3.0 m apart)
```

Collapse compresses the whole 4–140 m scene into 5.5 m, so consecutive frames' dense blobs overlap
heavily. The mass is identical — merely denser — and a **raw** inner product rewards exactly that.

**The corrected objective** is density-invariant: mean pairwise **normalised** cross-correlation,
scaled by the fraction of ink that landed on the grid.

```
score = mean_{i<j} <Hᵢ,Hⱼ> / (‖Hᵢ‖‖Hⱼ‖)  ×  on_grid_fraction
```

With unit-normalised maps this has the closed form `(‖ΣᵢHᵢ‖² − K) / (K(K−1))` — one pass, not K².
The `on_grid_fraction` factor blocks the opposite cheat (fling the ink off-grid until a
self-consistent handful remains). Now the objective peaks at the truth:

```
truth                      8.771210e-02   <-- TRUTH
yaw +2 deg                 2.103608e-02
pitch +1 deg               1.199587e-02
height x1.3                1.755178e-02
height -> 0.05 (collapse)  2.061090e-02
fx x1.3                    1.217340e-02
```

## 5. Recovery from a deliberately wrong start

Curved path (1/300 m⁻¹), 14 frames, 3 m apart, five free parameters:

| parameter | true | start | recovered | error |
|---|---|---|---|---|
| yaw | −6.500° | 0.000° | −6.491° | **+0.009°** |
| pitch | −1.200° | 0.000° | −1.331° | **−0.131°** |
| height | 1.280 m | 1.170 m | 1.278 m | **−0.002 m** |
| roll | 0.800° | 0.000° | −0.046° | ⚠️ −0.846° |
| lateral | −0.120 m | 0.250 m | 0.282 m | ⚠️ +0.402 m |

⇒ **yaw, pitch and height recover essentially exactly. Roll and lateral do not.**

That split is structural, not a tuning failure: a constant mount translation is **common to every
frame**, so it very nearly cancels in a cross-frame consistency objective — only the lever arm during
a turn makes it visible, and 1/300 m⁻¹ is gentle. **This is precisely why the lateral offset has
resisted every estimator in this pipeline**, and it says the fix for lateral must come from
somewhere else (the ego's position between the markings, or an external measurement).

## 6. ⭐ The headline: f and h ARE separable — the premise that blocked us is a small-angle approximation

`scale_calib.py` states, and I repeated all through 2026-08-08, that *"the ground plane can never
separate f from h on its own"*. I predicted this method would inherit that on straight paths. **Both
the prediction and the premise are wrong.** Scanning the objective **along** the f·h = const manifold —
the supposedly flat direction:

```
   h [m]    f [px] |     straight   curved R=150
    0.80    2365.3 |   9.7682e-03     6.2084e-03
    1.10    1720.2 |   3.6240e-02     2.8062e-02
    1.28    1478.3 |   8.7956e-02     8.7566e-02   <-- TRUE
    1.60    1182.6 |   2.4589e-02     1.6875e-02
    2.00     946.1 |   9.8108e-03     4.9671e-03

  straight  peak at h = 1.28 (true 1.28)   max/min =  9.00x   PEAKED
  curved    peak at h = 1.28 (true 1.28)   max/min = 17.63x   PEAKED
```

Falsified as an artifact of my own grid bounds first: widening the BEV to y ∈ [−40, 40] m, so the
grid cannot supply a lateral scale, leaves the straight-path result **identical** (h err +0.006 m,
f err −18.8 px) and *improves* the curved one (f err −4.6 px).

**Why the premise fails.** `x = f·h/(v − v_h)` is a **small-angle approximation**. The ray is
`[(u−cx)/f, (v−cy)/f, 1]`; changing f scales only the first two components, producing a genuinely
*different* ray direction rather than a scaled one. Over a 66° field of view with ink spread across
the image, that difference is observable. Mount misalignment strengthens it further but is not
required — MEASURED, straight path:

```
  yaw −6.5°, roll +0.8° (real mount)   peak at true h, max/min 8.97x
  yaw  0.0°, roll  0.0° (aligned)      peak at true h, max/min 3.33x
```

⇒ The degeneracy is real for a **single frame under the paraxial approximation**. It is **not** real
for cross-frame BEV agreement with metric ego-motion over a wide FOV. That reopens the one question
that has blocked this calibration since 2026-08-08.

## 7. ⚠️ What this does NOT yet show

Everything above is **noise-free synthetic data with perfect ego-motion**. Before any of it is
quoted about the real recording:

1. **Ego-motion error.** Real poses carry 0.70 m position / 0.53° heading hold-out RMS. Relative
   motion over a 2 s window is far better than absolute, but it is not zero, and it blurs the BEV
   exactly like a calibration error would. **Use short baselines, and treat the pose as a nuisance
   parameter rather than as truth.**
2. **EIS.** The S21 FE applies electronic stabilisation to video, so the mount rotation is **not
   constant across frames** — the one assumption the whole method rests on. The mp4 carries no EIS
   flag (only `com.android.version=16`), so this has to be measured, not assumed. A per-frame
   rotation residual is the natural extension; if EIS is active, this is the single biggest threat.
3. **Rolling shutter** at 22 m/s: each row is exposed at a different instant.
4. **Non-planar road** — the 14-19-54 clip is a hilly D-road; the flat-ground assumption fails over
   40 m, which is also a plausible contributor to `plane_calib`'s ±0.56 m spread.
5. **Roll and lateral stay weakly observable** (§5) whatever the front end does.

## 8. Where SAM3 fits, and why the front end is still the point

The optimiser above ran on synthetic ink. On real frames the binding constraint is **data
association**: classical thresholding finds bright pixels, including guardrail, sunlit rock and
specular asphalt — the same weakness that gave `plane_calib` 14 usable homographies. SAM3's value is
that it labels *what* a region is, so the objective can be restricted to genuinely static painted
ground, and different semantic classes can be weighted differently (dashes are sharp and
well-localised; a large asphalt region contributes almost nothing to alignment).

Two externally-standardised quantities become usable once markings are segmented as markings, and
they are conveniently orthogonal in exactly the way f and h are:

| feature | direction | pins |
|---|---|---|
| painted line **width** (~0.15 m) | lateral → `y = (u−cx)h/(v−v_h)` | **h alone, no f** |
| **dash pitch** (standardised dash + gap) | longitudinal | **f·h** |

These are independent of §6 and would cross-check it — which, given §6 overturns a premise this
programme has been operating on, is worth having.

## 9. Next steps, in order

1. **Run it on the real recording** with a classical marking front end first — it de-risks the
   optimiser before any SAM3 dependency, and the container has no GPU, no torch and no HF access.
2. **Measure whether EIS is active** (per-frame rotation residual across a static-scene segment).
   If it is, §7.2 dominates everything else.
3. **Then** swap in SAM3 masks and measure the improvement in peak sharpness — the honest metric for
   whether the semantic front end is worth its cost.
4. Add the line-width and dash-pitch constraints as independent cross-checks on §6.
5. Leave roll and lateral to an external reference; do not expect this method to supply them.

---

# PART 2 — ON THE REAL RECORDING (2026-09-13)

**Evidence class: MEASURED (ours, real).** `raw/real_run_log_v2.txt`,
`raw/real_bev_calib_v2.json`. Front end = `lane_calib._ridge_points`, the pipeline's OWN marking
detector, so no new unvalidated perception component is in the loop. 9 anchors × 7 frames,
176,435 ink points.

## 10. ⭐ It works on the extrinsics, and independently corroborates August

```
nominal (shipped, yaw 0)      4.963e-02
yaw -7.01                     6.409e-02    +29%
yaw -7.01, pitch -0.64        7.020e-02    +41%   <- BEST
yaw -7.01, pitch -2.93 (FOE)  6.560e-02    worse than -0.64
height 0.05 (collapse)       -4.9e-18      correctly rejected
```

Two results matter here, and neither was available before:

1. **The yaw correction is confirmed by a third, independent physics.** August's −7.01° came from
   lane vanishing points and translational flow. This is BEV cross-frame consistency — a different
   measurement entirely — and it prefers −7.01° over the shipped 0.0° by **+29 %**.
2. **The lane-VP pitch beats the FOE pitch.** −0.64° (from the measured horizon row 523.4) scores
   above −2.93° (from the FOE). That is exactly the conclusion reached on 2026-08-08 by VP fitting,
   now reproduced by an unrelated method. It also supports the earlier call that the FOE pitch was
   the wrong quantity to drive the ground homography.

## 11. ⚠️ It does NOT yet determine height or focal on real data

```
f*h = const manifold:  peak at h = 0.95 m,  max/min = 2.09x
                       (synthetic gave 9.00x straight / 17.63x curved)
joint 4-parameter fit: ran to the bound — fx 2588 of a 2600 limit, height 2.064 m
```

The f·h direction that was sharply peaked in synthetic is **4–8× weaker on real ink**, peaks at
0.95 m against the 1.03–1.21 m from the independent focal-free lane-width method, and the
unconstrained joint fit runs away to its bounds. ⇒ **The extrinsics are usable; the intrinsics and
height are not.** Do not quote h = 0.95 m as a measurement.

## 12. Four bugs, three of mine, all found by measurement

| # | bug | how it showed |
|---|---|---|
| 1 | objective v1 scored **concentration** | collapse beat the truth **15×** (§4) |
| 2 | objective v2 used **raw** cross-correlation | collapse still won **3.8×** (§4) |
| 3 | **anchor span 2.0 s = ZERO ground overlap** | see below — this was the real-data killer |
| 4 | `pgrep -f` **self-matched** my own command | the first real run silently never started |

**Bug 3 is the instructive one.** At 22 m/s a 2.0 s span is **44 m of travel** against a ~35 m
visible ground band, so the anchor and the last frame share **no ground at all**:

```
   dt [s]  travel [m]  overlap [m]  % of band
     0.10         2.2         32.8        94%
     0.50        11.0         24.0        69%
     1.00        22.0         13.0        37%
     2.00        44.0          0.0         0%     <- what the first run used
```

With zero overlap the objective scores noise, and the joint fit duly returned a **54.7° yaw**. The
BEV picture is what exposed it — the stacked view showed the same structure repeated once per
frame instead of superimposed. **Default span is now 0.6 s.**

Bug 4 is the `pgrep -f` trap already in `CLAUDE.md` ("self-matches your own command"): the waiter's
own command line contained the pattern it was grepping for, so it looped forever and the run never
launched — while every status check reported "RUNNING" for the same reason.

## 13. Hypotheses of mine that the data refuted

Recorded because each was stated before being tested, and the corrections are the useful part:

| hypothesis | verdict |
|---|---|
| metric ego-motion kills the collapse degeneracy | **NO** — collapse won 15× until the self-term was removed |
| removing the self-term is sufficient | **NO** — still 3.8×; needed NCC |
| f and h are degenerate on a straight path | **NO** — peaked 9×; the degeneracy is a *paraxial approximation*, not a fact |
| mount misalignment is what breaks that degeneracy | **HALF** — it strengthens the peak (3.3×→9.0×) but an aligned camera is still peaked |
| non-ground ink (guardrail, rock) is the binding constraint | **NO** — restricting to central image columns made discrimination *worse*: +41% → −6% → −16%. Yaw needs lateral leverage, and the wide ink carries it. |

That last one matters for the SAM3 question: **the evidence does not currently support "the classical
front end is the bottleneck".** Removing suspected junk removed signal instead.

## 14. ⚠️ EIS remains UNVERIFIED — and the time sync could not be confirmed

The single-mount-rotation assumption could not be checked, because the instrument available for it
is too noisy on this clip. Camera-vs-gyro roll: magnitudes nearly equal (2.04 vs 2.06 °/s) but
**r = 0.048**. On yaw, with the pipeline's own band-pass (0.15–4.0 Hz), a full lag scan gives:

```
best |r| = 0.312 at lag +4.40 s        (the pipeline uses +3.30 s)
r at the pipeline's lag   = -0.193      (negative)
|r| over all lags: p50 0.111  p95 0.243  max 0.312   ->  max/p95 = 1.28x
```

**No distinct peak.** ⚠️ This does NOT prove the sync is wrong — the pipeline refines on the 3-vector
across three bands with an agreement test, a stronger estimator than this scalar correlation, and
its `sync_score` is not a Pearson r. What it establishes is that the sync **cannot be independently
confirmed from this channel**, so the BEV method should estimate the offset rather than trust it.
`scan_time_offset.py` does exactly that and is the next thing to run.

## 15. Revised next steps

1. **Run `scan_time_offset.py`** — turn the unconfirmable sync into a measured quantity.
2. **Constrain the f·h direction with an external metric** rather than hoping the optimiser finds
   it: painted line width (lateral → h alone) and dash pitch (longitudinal → f·h). §11 shows the
   free fit will not get there on its own.
3. **Sweep the anchor span** (0.2–1.0 s) and the number of anchors; §12 shows the result is very
   sensitive to overlap, and 0.6 s was chosen by arithmetic, not optimised.
4. **SAM3 — but with a measurement first.** §13 refutes the motivating assumption that non-ground
   ink is the limiter. Before taking the dependency, test whether restricting ink to *semantically
   confirmed markings* (hand-label a few frames as a proxy) actually sharpens the f·h peak. If it
   does not, the limiter is elsewhere (ego-motion error, non-planar road, EIS) and SAM3 will not fix it.

---

# PART 3 — SOLVED. AND THE FOCAL WAS NEVER THE PROBLEM (2026-09-13, later)

## 16. ⭐ The answer

| parameter | shipped | **measured here** | how |
|---|---|---|---|
| horizon row | 464.4 (FOE) / 523.4 (lane VP) | **437–485 px** | row flow **437.4** [431.4, 443.4]; 2-D flow **454.0**; lane-width range-consistency **~485** |
| `f·h` | 1478.3 × 1.17 = **1729** | **2240–2800 px·m** | row flow **2668** [2547, 2798]; 2-D flow **2243** |
| height | 1.17 m | **~1.65–1.86 m** | lane width at horizon 465, assuming a 3.50 m lane |
| focal | 1478.3 px (HFOV 66°) | **~1400–1530 px** | `f = f·h / h` — **the shipped focal was roughly right** |

**The shipped focal was approximately correct and the shipped HEIGHT was wrong by ~45 %.**
This whole line of work was a hunt for a focal error. Sayed's *"i think also the 1.6 m height are
very plausible"* is **corroborated by the measurement**, not contradicted by it.

Rendered with `f = 1438 px`, `h = 1.70 m`, horizon `465`, yaw `−7.01°`, lateral `−0.12 m`
(Sayed's estimate), which sits centrally in every interval above.

## 17. ⛔ RETRACTION — I had the horizon backwards (logged: `R-2026-09-13-horizon`)

Earlier this session I ruled the lane-VP horizon **523.4 px** correct and the pipeline's FOE
**464.4 px** wrong, a "2.29° pitch error", and defaulted four modules to `--horizon 523.4`.
**It is the other way round.** Two probes orthogonal to both candidates put it at **437–454**
(ego motion) and **~485** (lane width held range-independent), straddling the FOE. At 523.4 the
reconstructed lane width **grows 41.5 %** from the 8–13 m slab to the 18–25 m slab — a tilted
plane, which is exactly what a wrong horizon does:

| assumed horizon | 8–13 m | 13–18 m | 18–25 m | far/near |
|---|---|---|---|---|
| 464.4 | 1.980 | 2.070 | 1.710 | 0.864 |
| **485.0** | 1.980 | 2.370 | 2.070 | **1.045** |
| 523.4 | 1.950 | 2.685 | 2.760 | **1.415** |

Root-cause class **C6**: I adjudicated between two *image-derived* estimates on plausibility
instead of finding a criterion orthogonal to both. A vanishing point measures where lane lines
converge; the horizon is where the **ground plane** vanishes, and on a crowned, curving road those
are not the same row.

## 18. Three estimators, two diagnosed failures, one that works

**`lag_scale.py` — BEV cross-frame lag against the odometer. Validated, then REFUSED.**
Synthetic self-test recovers an injected `f·h` to within 5 % across a 3.3× range *and* recovers a
±10 px horizon error to ±2.3 px (where a naive whole-map read would be off by +21 %/−15 %). On real
data it declines, for a measured reason: at `D = 2.99 m` the correlation **decays monotonically from
`r = 0.211` at lag 0** and is only `r = 0.062` at the true displacement. Content static in the
*image* back-projects to the same range in every frame, so it correlates at lag 0 and outweighs the
moving ground. **Before the static-reference guard added here, it reported `f = 3645 px` with a tight
bootstrap** — confidently precise and wrong.

**`flow_scale.py` 2-D fit — failed on the front end, not the model.**
`goodFeaturesToTrack` does not find the road, it **avoids** it: corners live where texture is and
asphalt is smooth, so 6807 tracks gave **2.3 % inliers at a 400 px median reprojection**. Masking to
the pipeline's own ridge detector and cutting the row band at 0.77 H fixed it (median 11.7 → 7.6 px,
inliers 17.7 %).

**`flow_scale.row_flow_fit` — the one that works.** Vertical flow only, two parameters:

```
dv = D q² / (A − D q)        q = v − v_h,  A = f·h
A  = D q q' / (q' − q)       inverted per point
```

and the horizon is found by **consistency** — the value at which the per-point `A` stops trending
with image row — not by fitting a curve.

## 19. ⚠️ The row band is a measurement, not a choice

Below row ~830 the tracked flow **collapses to ~0** while the plane model predicts 59–135 px, and
the tracking rate falls to 15–20 %. That region is static in the image (bonnet / windscreen
reflection) and it supplied **6294 of the seeds**. Rows 580–830 follow the model closely — at
730–780, measured **27.46 px** against a predicted **27.27**.

| rows | seeds | tracked | rate | measured dv | model dv |
|---|---|---|---|---|---|
| 580–630 | 3605 | 3117 | 86.5 % | 8.81 | 3.15 |
| 680–730 | 1981 | 1312 | 66.2 % | 19.37 | 16.35 |
| 730–780 | 1436 | 615 | 42.8 % | **27.46** | **27.27** |
| 830–880 | 2859 | 428 | 15.0 % | **1.02** | 58.90 |
| 880–930 | 3435 | 713 | 20.8 % | **−0.60** | 80.15 |

## 20. Three hypotheses of mine, all REFUTED by measurement rather than assumed away

1. **Tracking selection inflates `f·h`** (points that move far fail LK, so survivors are biased
   slow). Across steps of 1/2/3 frames the tracking rate falls **61.8 % → 24.8 % → 8.8 %** — a 7×
   change in selection pressure — while the fitted horizon holds at **423.7 / 432.5 / 426.6 px**.
   A selection artefact cannot survive that.
2. **A per-pair constant row offset (pitch jitter / EIS).** Eliminating one by within-pair de-meaning
   made the robust cost **worse** (1.348 vs 1.340) and sent the horizon to a degenerate 120 px.
   Not modelled, on evidence.
3. **My "overlap normalisation" diagnosis** of the lag estimator's shrink-toward-unity bias: it
   changed `f·h × 1.40` from `s = 1.233` to `1.234`. **No effect.** The real cause was in the *test* —
   injecting a focal error at fixed pitch **also moves the horizon**, so `s = factor` was never the
   right expectation. The estimator was convicted of a bias it did not have.

## 21. Why the height needs a ruler, measured rather than asserted

`observability()` differentiates the actual reprojection on the actual tracks:
singular values **[233.3, 85.1, 33.5, 5.0]**, least-observable direction **height −0.82, yaw +0.57**.
And the height scan is the clean demonstration: forcing `h` from 0.90 → 1.60 m moves the fitted
`f·h` only **2244.2 → 2237.2 (0.4 %)** while the cost spread is **0.2 %**. `f·h` is the observable;
`h` is not, at any optimiser setting.

## 22. ⚠️ What is still an assumption

- **`h` scales linearly with the assumed lane width.** At 3.50 m (French motorway) `h ≈ 1.7 m` and
  `f ≈ 1438`; at 3.00 m, `h ≈ 1.47` and `f ≈ 1663`. Both keep `f` inside the 1356–1628 device band,
  so the band does **not** discriminate between them.
- **The painted-line-width ruler is UNUSABLE with this front end.** It returns 0.29–0.51 m FWHM at
  *every* horizon — that is the ridge detector's response plus along-range smear, not paint. Reported
  and discarded, not quietly averaged in.
- **The two flow fits differ**: `f·h` 2668 (row flow) vs 2243 (2-D). 16 % apart, both far above the
  shipped 1729. The interval, not the midpoint, is the result.

## 23. ⭐ Where SAM3 actually earns its place — and it is NOT where I first said

In Part 1 I motivated a semantic front end with *"non-ground ink is the bottleneck"*. **That was
REFUTED by measurement** (§13): restricting to the central road columns made the extrinsics contrast
*worse*, +41 % → −6 % → −16 %, because the wide ink carries the lateral leverage yaw needs.

The real bottleneck is the opposite one, and it is now measured. **The ridge detector does not find
enough marking ink in the FAR field**, and that is exactly what blocks the last two parameters:

- the 16–22 m range slab yields a lane-line pair in **9 of 140 frames**;
- whole-profile cross-correlation between the 8–12 m and 16–22 m slabs — a method that identifies no
  peaks at all, so the selector instability of §22 cannot occur — locks in only **14–18 of 150
  frames** at `r > 0.45`.

Residual yaw and lateral offset are separable **only** by range leverage: a lateral offset displaces
the lane by the same amount at every range, a yaw error displaces it in proportion to range. With no
usable far slab there is no leverage, and the two stay confounded. That is a **perception** limit,
not a geometry one — and it is the first evidence-backed case for a stronger marking segmenter.

⚠️ Note what this does **not** say. It does not say SAM3 would improve the calibration that is now
settled: `f·h` and the horizon came from tracked points at 8–18 m, where ink is plentiful, and they
are done. The claim is narrower and testable: **a segmenter that recovers far-field markings would
separate yaw from lateral offset, which the classical front end cannot.**

---

# PART 4 — THE HEIGHT WAS STILL 19 % TOO HIGH (2026-09-13, after Sayed: *"the quality is still not ok"*)

## 24. He was right, and the failing ruler was the one §22 already flagged

Motion settled the longitudinal geometry (`f·h`, horizon). The **height** came from
`lane_width.py`, which pairs adjacent peaks in a lateral profile — the selector §22 recorded as
unstable. It was. It gave 1.65–1.86 m; the truth is ~1.43.

`probes/lane_residual.py` replaces it with the test that actually matters: **project the model's
lane edges into the image and measure, in metres, how far the nearest paint is.** No peak is
identified, so there is nothing to mis-pair; association is bounded to ±1.1 m so a residual cannot
be manufactured by locking onto the next line over; the unassociated fraction is printed.

At the rendered `h = 1.70` the painted lane measured **4.09 m** wide. That is not a lane.

## 25. The fit, with `f·h` and the horizon held

|  | height | lateral | yaw | cost |
|---|---|---|---|---|
| rendered | 1.700 | −0.120 | −7.010 | 1.436 |
| **fitted** | **1.427** | **−0.126** | **−7.299** | **0.924** (−35.7 %) |

`f = f·h / h = 1713 px`. Three successive fits converge: **1.482 → 1.448 → 1.427**. The lateral
offset lands at **−0.126 m** against Sayed's own estimate of **−0.12 m**.

## 26. ⚠️ Two confounds found, both of which read as yaw

1. **Road curvature.** On a curve the lane centre genuinely moves laterally with range — the exact
   signature of a mount-yaw error. This clip reaches `|k| = 0.0034 1/m` (radius 294 m) = **0.38 m of
   displacement over the 8–15 m window**, the same size as the residual being fitted. Restricting to
   `|yaw rate| ≤ 0.6 °/s` dropped the apparent residual yaw from **+2.83° to +1.76°**.
2. **The right lane line is DASHED.** On straight frames the LEFT edge residual is flat —
   −0.087, −0.075, +0.007, +0.055, +0.037, +0.040 m across 8→22 m — while only the RIGHT drifts.
   **A yaw error moves both edges together, so an asymmetry is not yaw.** The left line is solid and
   associates at every range; the right is dashed, and beyond ~10 m the associator often has no dash
   and locks onto the shoulder. The residual "+1.76°" is that artefact. Ranges restricted to 8–12 m,
   where both edges associate above 60 %.

## 27. ⚠️ What the height still rests on

`h` scales linearly with the assumed lane width, and **nothing here measures it**:

| assumed lane | height | focal | vs the 1356–1628 device band |
|---|---|---|---|
| 3.50 m (French motorway) | 1.427 | 1713 px (HFOV 58.5°) | outside |
| 3.65 m (the pipeline's own `--lane-width` default) | 1.488 | 1643 px | at the edge |

**I am not picking the assumption that lands inside the band.** The coupling is stated; the render
uses the 3.50 m standard. Resolving it needs a lane width measured on the ground, or a device FOV
measured in video mode rather than inferred from the stills spec.

## 28. Part of what he saw was mine

The video sent was **downscaled 1.6×** to fit a 30 MiB upload limit. At full resolution the same
frame is markedly cleaner, and the range ticks check out against the calibration (25 m tick predicted
at panel row 333, observed ~345; horizon 465 → 276 predicted, ~272 observed). A compressed frame is
not evidence about a calibration.

## 29. ⭐ Yaw, from the flatness of the SOLID line — and the final set

A yaw error makes the lane residual grow with range; the right yaw makes it **flat**. Scanning yaw
against the LEFT edge only (the solid line — the dashed right one mis-associates past 10 m), on
near-straight frames, 130 frames:

| yaw | 8 m | 10 m | 12 m | 15 m | 20 m | slope | mean abs |
|---|---|---|---|---|---|---|---|
| −7.90 | +0.147 | +0.143 | +0.242 | +0.327 | +0.403 | +0.0235 | 0.252 |
| −7.30 | +0.071 | +0.069 | +0.105 | +0.180 | +0.187 | +0.0114 | 0.122 |
| **−6.80** | **+0.049** | **−0.020** | **+0.005** | **+0.048** | **+0.012** | **−0.0001** | **0.027** |
| −6.30 | −0.104 | −0.096 | −0.105 | −0.083 | −0.175 | −0.0055 | 0.113 |
| −5.80 | −0.147 | −0.193 | −0.210 | −0.223 | −0.356 | −0.0161 | 0.226 |

A clean interior minimum: **mean |residual| 2.7 cm across 8–20 m**, slope −0.0001 m/m. It sits
between the pipeline's FOE yaw (−6.31°) and its lane-VP yaw (−7.01°).

### The final calibration

| parameter | value | from |
|---|---|---|
| `f·h` | 2444.6 px·m | row flow vs the odometer |
| horizon row | 465.0 px | row flow + lane-width range-consistency |
| height | 1.427 m | projected-lane-vs-paint width (assumes a 3.50 m lane) |
| focal | 1713 px | `f·h / h` |
| yaw | **−6.80°** | flatness of the solid left line over 8–20 m |
| lateral | −0.126 m | same fit — and Sayed's own estimate was −0.12 |
| roll | 0.0° | not separately measured |

Verified at these values: left-edge residual −0.005, −0.031, +0.035, +0.079, +0.128 m over 8→20 m;
painted lane 3.60 m against the assumed 3.50.

⚠️ The `CENTRE offset / residual yaw` line printed by `lane_residual.py` at these values reads
**+4.04°** and is **not trustworthy** — it is corrupted by a single right-edge value of +0.979 m at
15 m with 16 % association, which is the dashed-line mis-association of §26.2. The left-edge scan
above is the admissible yaw estimate.

---

# PART 5 — WHY NO CONFIGURATION FITS EVERY FRAME (Sayed, 2026-09-13)

## 30. The camera: 26 mm-equivalent, and the "81°" is right

Both S21 FE cameras are **26 mm-equivalent**, so which one was used barely moves the FOV
(main: 12 MP f/1.8, OIS; front: 32 MP f/2.2, Sony IMX616, fixed focus). 26 mm-equiv is
**79.5° diagonal** — Sayed's 81° is that figure, rounded on a spec sheet.

| | |
|---|---|
| 79.5° diagonal on a 4:3 sensor | HFOV **67.3°**, VFOV 53.1° |
| at 1920 px wide, full sensor | **f ≈ 1442 px** |
| the pipeline's shipped nominal | 1478 px (HFOV 66°) — within 2.5 % |
| **measured here** | **1713 px** (HFOV 58.5°) = a **1.19× crop** |

A 1.19× crop is exactly what video stabilisation takes. ⚠️ And the choice matters: `f` is what
turns the measured `f·h = 2444.6` into a height. At `f = 1442`, `h = 1.695 m`; at `f = 1713`,
`h = 1.427 m` — and the lane-width measurement independently favours ~1.43, which is only
consistent with the cropped focal. **So this is a real, answerable question: was video
stabilisation on?**

## 31. ⭐ The camera's pitch relative to the ROAD moves by 1.75° during the clip

Sayed could not find one configuration satisfying every frame. He is right, and it is a property
of the recording, not a fitting failure.

Per-frame residual slope of the solid left line, 39 straight frames:

| | sd | p10 | p90 |
|---|---|---|---|
| with a **fixed** horizon (465) | 0.0404 m/m | −0.0120 | +0.0353 |
| with a **per-frame** horizon | 0.0165 m/m | −0.0027 | +0.0049 |

**A per-frame horizon removes 59 % of the spread.** The best-fit horizon per frame runs
**449.8 → 502.0 px (p10–p90 = 52 px = 1.75° of pitch)**, sd 20 px. A 52 px horizon swing moves the
range of a point at 20 m by ~11 m.

## 32. ⚠️ It is NOT the driver steering — tested, because the two look identical

A vehicle yawed θ to the lane is *crossing* the lane at `v·sin θ`, so the attitude swing makes a
prediction: `d(lateral)/dt = v · (residual slope)`. Both sides measured on consecutive frames:

- measured lateral drift **0.156 m/s rms**
- predicted from the attitude swing **0.512 m/s rms**
- regression **−0.05×**, **r = −0.164**

The attitude swing predicts **3× more lane-crossing than actually happens**, with no correlation.
So it is not yaw from steering — it is pitch, which produces the same residual slope and no lane
crossing. Consistent with a hilly motorway: vertical curvature and suspension move the camera
relative to the road plane ahead.

⇒ **A per-clip constant pitch is the wrong model for this recording.** The horizon has to be
tracked per frame, or the overlay will be right on average and wrong on individual frames — which
is exactly what Sayed observed.

## 33. Two probes that failed here, kept with their refutations

- **`eis_vs_gyro.py`** — race distant image motion against the gyro; the slope would be the
  stabilisation strength. **Inconclusive**: all correlations |r| < 0.04, because rows 0.12–0.38 H
  on this road hold *near* hillside and roadside trees, not distant content, so translation
  dominated (measured 15 px rms against a 2.4 px gyro prediction). The probe reports the weak
  correlation and refuses rather than quoting a slope.
- **`per_frame_vp.py`** — per-frame vanishing point from the two lane lines. **Unusable**: the solid
  left line yields a median of 19 samples per frame and fits to 1.39 px, the dashed right line
  yields 8. Three frames of 260 produced a vanishing point.

## 34. ⛔ Per-frame horizon tracking: built, validated, and NOT deployable

The obvious fix for §31 is to track the horizon per frame. Built (`probes/horizon_track.py`):
with `f·h` known, the row-flow relation inverts in closed form per tracked point,
`v_h = [(v+v') − √((v+v')² − 4(v v' − A dv/D))]/2`, so it needs road texture, not paint —
**259 frames at 412 votes each**, against the lane-based measurement's 39 frames at 5 samples.

It corroborates the magnitude: p10–p90 **47.1 px = 1.57°**, against the lane estimate's 1.75°.
But frame by frame the two agree only at **r = +0.336**.

Decomposing the flow series against itself settles which part is real. Lag-1 autocorrelation
**+0.416** (decaying to +0.203 at 2.4 s) on a raw sd of 19.7 px gives

| | |
|---|---|
| true horizon swing | sd ≈ **12.8 px ≈ 0.43°** |
| estimator noise | sd ≈ **15.0 px** |

**The noise is larger than the signal.** So §31's "1.75°" was itself inflated by the lane
estimator's own noise; the genuine swing is about 0.43° sd.

The deployment test, judged by the lane residual (which shares no features or model with the
tracker), 13 straight frames:

| horizon source | median abs slope | sd |
|---|---|---|
| **fixed 465 px** | **0.0265** | 0.0180 |
| flow track, raw | 0.0445 | 0.0439 |
| flow track, 0.9 s | 0.0382 | 0.0452 |
| flow track, 2.1 s | 0.0378 | 0.0409 |
| flow track, 3.9 s | 0.0277 | 0.0340 |

Even at 3.9 s smoothing it is **4 % worse than the constant**. ⇒ **Do not deploy.** Chasing a
0.43° signal with a 15 px-noise estimator makes the overlay worse, not better. The constant
horizon stays, and the per-frame wobble is a stated limitation rather than a fixable one with
the instruments available.

---

# PART 6 — CLOSED. EIS CONFIRMED BY THE OPERATOR (2026-09-13)

## 35. ⭐ Three independent routes now agree

Sayed, asked directly: *"Yes stabilisation was on, I checked the phone parameters."*

| route | focal |
|---|---|
| 26 mm-equivalent lens, 79.5° diagonal, full 4:3 sensor | **1442 px** |
| × the 1.19 EIS crop that confirmation implies | **1713 px** |
| `f·h = 2444.6` (ego motion) ÷ `h = 1.427` (lane paint) | **1713 px** |

The chain closes. **The final calibration for `2026-08-08_14-19-54-android`:**

```
--focal-px 1713  --cam-height 1.427  --horizon-row 465.0
--cam-yaw -6.80  --cam-roll 0.0
--lateral-offset -0.126 --lock-lateral  --mount-longitudinal 2.1
```

against the shipped `f 1478.3, h 1.17, horizon 464.4, yaw −7.01, lateral +0.25`.
**The focal was never really wrong — it was the uncropped lens value, 16 % low, and the ground
plane absorbed that into the height.**

## 36. The trap, stated where it will be read

`camera.py:nominal_camera` now writes `"nominal HFOV=… (UNCROPPED lens; EIS crops, so this is a
LOWER BOUND on the recorded focal)"` into the provenance, with the measurement in its docstring and
a test (`test_nominal_focal_is_labelled_as_the_uncropped_lens`) pinning it. 11/11 pass.

**Why it hid for so long.** The ground plane sees only the product `f·h`
(`scale_calib.py`'s own derivation), so a 16 % low focal is absorbed by a 22 % high height and
*every downstream number stays self-consistent*. Nothing contradicts anything; the overlay is
simply wrong. The only way out is an external metre-stick on the lateral axis, because that is the
one axis where `h` appears without `f`.

⇒ **Same family as the traps in `CLAUDE.md`:** a quantity that is correct about what it describes
(the lens) read as an answer to a different question (the recorded video).

---

# PART 7 — ⛔ THE HEIGHT MEASUREMENT WAS CIRCULAR. h = 1.427 IS WITHDRAWN.

## 37. What the automatic run exposed

Re-running the pipeline with no operator overrides (only `--lane-width 3.50`, which Sayed
confirmed and no image can measure):

- **`flow_calib` worked**: `f·h = 2872.7` px·m, 95 % CI [2644, 2909], **87,285 tracked points over
  204 frame pairs**, and its new guard fired — *"horizon from flow 427.4 px vs the camera's current
  464.4 px (−37.0 px) … LARGE disagreement"*.
- **`scale_calib.solve` then declined**: *"implied HFOV 41.6 deg is outside 55-85 deg for a phone
  camera"* — because it split `f·h` with `lane_calib`'s width (3.60 m **at h = 1.17**).
- The pipeline fell back to the shipped nominal: `f 1478.3, h 1.17, horizon 464.4`.

⇒ **The automatic render reproduces the original, rejected overlay.**

## 38. ⛔ And it exposed a defect in my own instrument

`lane_calib` says the lane is **3.60 m at h = 1.17**. My `lane_residual` said **3.55 m at
h = 1.427**. Lateral scale is linear in `h`, so those cannot both be true — 3.55 m at 1.427 is
4.33 m at 1.17. Run at both calibrations:

| assumed height | painted lane reported |
|---|---|
| 1.17 m | **3.53 m** |
| 1.427 m | **3.57 m** |

**The measured lane width barely moves with the assumed height** — ratio 1.011 where 1.22 was
required. The instrument has ~zero sensitivity to the quantity it claims to measure: the ±1.1 m
association window finds whatever paint is nearest the projection, so it **confirms any height fed
to it**. That is exactly the censoring defect I criticised in `lane_calib` (§*no width censoring*),
rebuilt in my own probe and not noticed because it always returned a plausible number.

⇒ **`h = 1.427 m` and `f = 1713 px` are WITHDRAWN as measurements.** They remain the best
*rendering* values tried — the BEV-agreement check ranked them +59.4 % over shipped, and that
instrument shares no machinery with the lane association — but the height itself was never measured.

## 39. What survives, and the honest bracket

| quantity | status |
|---|---|
| `f·h` | **2440–2880 px·m** — MEASURED, several flow variants, tight CIs, synthetic self-test passes |
| horizon | 427–448 (flow) vs 465–485 (paint) — **unresolved, ~40 px apart** |
| lens | 26 mm-equiv → **uncropped f = 1442 px**; EIS confirmed on, so the true `f` is larger |
| **height** | **NOT MEASURED.** From `f·h` and a crop of 1.0–1.2×: **h ∈ [1.41, 2.00] m** |

A third attempt at a non-circular height — `probes/height_from_row_gap.py`, which needs only the
pixel gap between lane lines in one row (`h = W·(v − v_h)/du`, no focal, no projection, no
window) — **also failed**: 48–62 % spread across rows at every horizon, because the ridge peaks in
a single row are not reliably lane lines. It is kept with that result recorded.

⇒ **The height cannot be settled from this recording with this front end.** It needs either a
marking segmenter good enough to identify the two ego-lane lines per row, or one physical
measurement of the camera height with a tape.

---

# PART 8 — THE VEHICLE IS A VW CADDY, AND IT DOES WHAT NO IMAGE COULD

## 40. One fact from Sayed replaced an unmeasurable parameter

The height could not be measured from this recording (§37-39). It did not have to be: the mount
photo shows a windscreen cradle at rear-view-mirror height, and the vehicle is a **VW Caddy**
(1.797 m tall). A Caddy windscreen-header mount puts the camera at roughly **1.50-1.80 m**, against
~1.25 m for a saloon — and that band, crossed with the device's focal limits, is decisive.

`f·h` across **every** flow variant measured this session:

| variant | f·h |
|---|---|
| rowfit pooled (bootstrap) | 2664 |
| `row_flow_fit`, 218 pairs | 2668 |
| `flow_calib`, 204 pairs (in the pipeline) | 2873 |
| step-1 only | 2906 |
| direct 2-parameter fit | 2444 |
| 2-D flow fit | 2243 |
| **median** | **2666** (range 2243-2906) |

| h | f = f·h/h | HFOV | crop vs the 1442 px uncropped lens |
|---|---|---|---|
| 1.43 *(rendered earlier)* | 1864 | 54.5° | **1.29× — too large for EIS** |
| 1.55 | 1720 | 58.3° | 1.19× |
| **1.60** | **1666** | **59.9°** | **1.16×** |
| 1.70 | 1568 | 62.9° | 1.09× |

⇒ **h = 1.427 is excluded**: it demands a 1.29× crop. Sayed's own early estimate — *"i think also
the 1.6 m height are very plausible"* — was right from the start.

## 41. BEV agreement, on arms that share an f·h

All the Caddy arms hold `f·h = 2666`, so the objective's known f·h bias cannot order them:

| set | vs shipped |
|---|---|
| h 1.60, **horizon 485** | **+79.6 %** |
| h 1.60, horizon 465 | +74.5 % |
| h 1.55, horizon 465 | +74.4 % |
| h 1.70, horizon 465 | +73.3 % |
| rendered (h 1.427) | +62.4 % |
| h 1.60, horizon 437 *(the flow horizon)* | +51.5 % |

**The paint-family horizon wins and the flow horizon loses**, on an instrument that saw neither.

## 42. ⚠️ And where the same objective must NOT be believed

Scanning further at fixed `f·h`:

- **horizon** — interior optimum at **485** (475 and 495 both score lower). Admissible, and it
  agrees with the independent lane-width range-consistency test, which also said ~485.
- **height** — monotone to the boundary (1.45 best of 1.45-1.75, still rising). **Rejected.**
- **yaw** — monotone to the boundary (−6.00 best of −7.6 to −6.0, still rising). **Rejected.**

A scan that runs to its boundary is the exact pathology that opened this whole investigation (the
focal scan, §Part 2). Adopting `h = 1.45, yaw = −6.00` from it would repeat the original mistake
with a different parameter. Yaw stays at **−6.80** from the left-line flatness test, which reads
range *structure* rather than a level and is not affected by the association circularity.

## 43. Rendered

```
--focal-px 1666  --cam-height 1.60  --horizon-row 485.0
--cam-yaw -6.80  --cam-roll 0.0
--lateral-offset -0.126 --lock-lateral  --mount-longitudinal 2.1
```

⚠️ **`h = 1.60` is CHOSEN, not measured** — the midpoint of the band the Caddy geometry and the
device FOV jointly allow, and `f` follows from it as `f·h / h`. A tape measure from the ground to
the lens still collapses the remaining ±0.15 m, and with it the ±100 px on the focal.

## 44. `--lock-height` verified end to end

The fix was committed on a source-level test; run against the real recording it now actually fires:

```
override: focal = 1666.0 px (HFOV 59.9 deg) (operator)
override: horizon row = 485.0 px -> pitch -1.891 deg (operator)
override: camera height locked at 1.600 m
```

`calibration.json`: `f 1666.0  h 1.600  horizon 485.0  f·h 2665.6`, with
`height_m.source = "OPERATOR OVERRIDE --lock-height (not measured)"`. Before the fix the same
invocation emitted **1.622**, because `scale_calib.solve` recomputed the height after
`--cam-height` had been applied.

The delivered Caddy video was rendered at 1.622 rather than 1.600 — a 1.4 % difference, inside the
±0.15 m the height is uncertain by anyway, so it was not re-rendered.
