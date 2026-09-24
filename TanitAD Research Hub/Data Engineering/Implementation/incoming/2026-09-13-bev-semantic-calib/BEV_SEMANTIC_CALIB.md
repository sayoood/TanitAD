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

---

# PART 9 — SYNC, AND WHY THE CORRIDOR CUTS MARKINGS AT RANGE (Sayed, 2026-09-14)

*"Did you look also at the synchronization between frame and ego data? … the overlay corridor cuts
road markings at large distances."*

## 45. Sync: looked at three times, never confirmed — and now the reason is known

| attempt | result |
|---|---|
| `scan_time_offset.py` — BEV agreement vs a pose offset | **FLAT**, max/min 1.05×. Refused. |
| image yaw rate vs the raw gyro | no distinct peak; best \|r\| 0.312 at +4.40 s vs the pipeline's +3.30 s, r **negative** at the pipeline's own lag |
| `eis_vs_gyro.py` | all \|r\| < 0.04 — the "distant" rows held near hillside, so translation dominated |
| **`sync_yawrate.py`** (new) — image rotation vs the *trajectory*, central columns only | **cannot resolve it, and says why** |

Raw, it produced a textbook false positive: r ramping monotonically from −0.284 at −0.97 s to
−0.587 at +0.97 s, **still rising at the search bound** (peak/p95 = 1.01×). That is the
boundary-running pathology from Part 2 in a third costume, and the probe now carries a guard that
refuses it.

Band-passed (both series share the road's overall curvature, and a shared trend correlates at
*every* lag), the correlation goes **flat**: peak \|r\| 0.237, peak/p95 **1.06×**. The reason is
visible in the numbers — after removing the trend the trajectory's yaw rate rms is only
**0.190 °/s**. **The trajectory is too smooth to time-align against.** No method can resolve the
sync from this channel on this clip; the signal is not there.

What the pipeline says about its own sync: `t_video_start = 3.3043 s`, source
*"gyro-xcorr (bands agree to 2 ms)"*, `sync_score 0.47`.

## 46. ⭐ But sync is not needed to explain it — the trajectory's heading error already does

The corridor follows the **future path**, so a heading error rotates it about the car and the
lateral error **grows linearly with range**. From the pipeline's own hold-out validation on this
recording (n = 82 folds, each holding out every 5th GNSS fix ≈ a 5 s outage):

```
heading   mean 0.43   rms 0.53   p95 0.97   max 1.00  deg
position  mean 0.59   rms 0.70   p95 1.36   max 1.71  m
```

Corridor half-width 0.90 m inside a 1.75 m half-lane leaves **0.85 m of margin** each side:

| range | rms 0.53° | p95 0.97° | verdict |
|---|---|---|---|
| 10 m | 0.09 m | 0.17 m | clear |
| 20 m | 0.19 m | 0.34 m | clear |
| 30 m | 0.28 m | 0.51 m | clear |
| **40 m** | **0.37 m** | **0.68 m** | touching the line |
| **50 m** | **0.46 m** | **0.85 m** | margin fully consumed |

⇒ **Beyond ~30 m the p95 heading error alone eats the margin.** Fine near, cutting markings far,
growing with distance — exactly the reported symptom, with no calibration or timing error required.

⚠️ This is an **ego-trajectory accuracy limit, not a calibration one.** No further work on `f`, `h`,
the horizon or the yaw will move it. It is bounded by GNSS/IMU heading over a 1–2 s horizon, and the
levers are better heading (RTK, or fusing the image), or drawing the corridor shorter than ~30 m and
saying why.

## 47. ⛔ PART 9's DIAGNOSIS IS REFUTED. It is a stale yaw, and it is fixable.

Part 9 blamed the trajectory's heading error for the corridor cutting markings at range. That was an
inference from a hold-out statistic, never a measurement of the symptom.
`probes/corridor_vs_lane.py` measures it: the distance from the ribbon to the left painted line, by
range, computed twice — once for the **trajectory** ribbon that is actually drawn, once for a
**straight** ribbon.

| range | trajectory ribbon | straight ribbon |
|---|---|---|
| 10 m | −0.380 m (n 238) | −0.357 m (n 235) |
| 14 m | −0.392 m (n 159) | −0.389 m (n 158) |
| 18 m | −0.411 m (n 53) | −0.411 m (n 55) |
| 22 m | −0.515 m (n 25) | −0.528 m (n 26) |
| 26 m | −0.468 m (n 14) | −0.548 m (n 14) |
| **angle** | **1.24°** | **1.18°** |

**The trajectory adds +0.06°.** A straight ribbon drifts just as much, so the heading error is not
the dominant term and Part 9 is wrong. The drift is **calibration**, and at 1.2° it is 0.85 m at
40 m — the whole symptom.

**The cause: a stale yaw.** Yaw was fitted at horizon 465 (§29). The horizon was then moved to 485
(§41-42) and the yaw was never refitted — and the two are coupled in this residual. Refitting at the
current `f 1666, h 1.622, horizon 485`, over 190 straight frames:

| yaw | 9 m | 12 m | 15 m | 18 m | 21 m | slope | at 40 m |
|---|---|---|---|---|---|---|---|
| −7.20 | −0.365 | −0.396 | −0.347 | −0.446 | −0.500 | −0.0107 | 0.43 m |
| **−7.80** | −0.341 | −0.252 | −0.225 | −0.273 | −0.273 | **+0.0038** | **0.15 m** |
| −8.40 | −0.212 | −0.155 | −0.097 | −0.079 | −0.034 | +0.0145 | 0.58 m |
| *−6.80 (rendered)* | −0.337 | −0.473 | −0.512 | −0.640 | −0.688 | −0.0290 | **0.85 m** |

A clean interior minimum. **Yaw −7.80° cuts the far-field drift 0.85 m → 0.15 m at 40 m, 5.7×.**

⚠️ The residual LEVEL (≈ −0.27 m at −7.80) is deliberately **not** zeroed into the lateral offset.
It conflates the camera's mount offset with the driver's average position in the lane, and the
corridor is supposed to follow the *car*, not the lane centre. Only the slope is a calibration error.

⚠️ **Lesson, and it is the session's recurring one:** every time a parameter moved, the ones coupled
to it needed refitting and did not get it. Yaw↔horizon here; earlier f↔h. Fitting them one at a time
and freezing each is what left a 1° error in a set that every individual test called good.

---

# PART 10 — the instrument that was deciding things could not see the defect

## 48. ⛔ §47's yaw −7.80 is WITHDRAWN. It made the video worse, measurably.

`probes/overlay_far.py` measures the **delivered video**: corridor-left-edge to painted-line
distance, in metres, at ranges out to the far field. Over 220 frames, straight frames only:

| arm | drift, straight frames | at 40 m | gap 10 m → 50 m |
|---|---|---|---|
| `out_caddy`, yaw **−6.80** | **−0.0037 m/m = 0.21°** | 0.15 m | 1.12 → 0.96 m |
| `out_yaw78`, yaw **−7.80** | −0.0192 m/m = **1.10°** | 0.79 m | 0.99 → **0.16 m** |

Yaw −7.80 walks the corridor onto the left line by ~50 m. That is Sayed's *"the corridor leaves
the ego lane"*, and it is the arm he was looking at. **§47's table had the sign of the correction
right and the magnitude wrong because its residual was read through an association window centred
on a prediction that moves with the parameter under test.**

## 49. ⭐ THE VALIDITY CHECK THAT SHOULD HAVE EXISTED FROM THE START

The two arms differ by **exactly 1.000° of yaw and nothing else**. A camera-yaw error `δ` shifts the
drawn corridor by `f·δ` px at every row, which is a lateral error of `δ·x` metres at range `x` — so
**the residual slope IS the yaw error in radians**, and the two arms' slopes MUST differ by
0.01745 m/m. That is free ground truth, available at any moment, and it grades the instrument:

| instrument | difference recovered | verdict |
|---|---|---|
| per-range window nearest the prediction | 0.0038 (**22 %**) | ⛔ absorbed 78 % of a known 1° |
| `track_line_up` (local tracker) | — | ⛔ random-walks: a relative threshold in a 10 px window ALWAYS returns a candidate, so it never reports a miss (IQR to +9.4 m, slopes of 27°) |
| **RANSAC line, global threshold, identity gate only** | **0.0155 (89 %)** | ✅ usable |

⇒ **RULE: when an experiment imposes a known change, MEASURE THAT CHANGE BACK before trusting the
instrument on the unknown one.** Two renders differing by one parameter are a free calibration
standard, and neither `corridor_vs_lane.py` nor `joint_fit.py` was ever held to it.

## 50. ⭐⭐ CAMERA HEIGHT, MEASURED AT LAST — and it needs no focal length

`probes/lane_width_far.py`. Eliminating range between `u = cx + f(y−lat)/x` and `v = v_h + f·h/x`:

    Delta_y  =  Delta_u · h / (v − v_h)          ← f is GONE

so the lateral metric scale on the ground is `h/(v − v_h)` and nothing else. Then the lane width
cannot depend on range, and **only the true `v_h` makes it range-independent**; Sayed's 3.5 m turns
the scale into `h`. This is the same degeneracy that blocked the session, used as a tool.

It only became possible once the far field was reachable — `lane_calib._ridge_points` ramps its
operator width by **the row's rank in the band it is handed**, not by the paint's apparent width, so
in a narrow far band it puts `w = 2` where the paint is 7.5 px wide and the operator samples
entirely *inside* the line. Fixed in `overlay_far.ridge_width_px`.

**The separation histogram is the internal validation, and it was not assumed:**

| Δy/h | n | reading |
|---|---|---|
| 0.00–0.25 | 19 | the two edges of one painted line |
| **2.25–2.50** | **30** | **one lane** |
| 4.50–5.00 | 32 | two lanes — **exactly 2× the first cluster** |

⚠️ The first pass took the two STRONGEST lines and called them the lane. They are the median edge
and the right-hand edge — **two** lanes — and calling that 3.5 m returned h = 0.82 m and a 35° HFOV.
Rescuing the number by reinterpreting the pair afterwards would have been the plausibility-over-
measurement failure of R-2026-09-13-horizon; the histogram is what makes the reading a measurement.

## 51. The horizon: two zero-trend criteria, fitted together, arbitrated by the lens

`probes/horizon_joint.py`. Both instruments define the horizon the same way — drive a range-trend to
zero — on physics that share nothing:

| criterion | uses | prefers |
|---|---|---|
| row flow `q' = qA/(A−Dq)`, 62,928 pts / 218 pairs | ego motion, **no paint** | **438** |
| lane width range-independence | paint, **no focal length, no ego motion** | **470** |
| **joint minimum of both trends** | — | **448** |

and the arbiter is orthogonal to both — **the lens**. 26 mm-equivalent is ≈ 1442 px at 1920 wide and
EIS can only ever crop *in*:

| solution | h | f | crop |
|---|---|---|---|
| flow-only, 438 | 1.637 m | 1622 px | 1.12× |
| **joint, 448** | **1.586 m** | **1533 px** | **1.06×** |
| width-only, 470 | 1.468 m | 1345 px | **0.93× — ⛔ IMPOSSIBLE** |

The paint-only horizon is **excluded by the optics**: it would need the recorded image to be wider
than the lens. This is the third-criterion rule from R-2026-09-13-horizon actually applied.

**ADOPTED:** `horizon 448.4 px · h 1.586 m · f 1533 px (HFOV 64.1°, EIS crop 1.06×) · f·h 2431 px·m`.
h = 1.586 m sits inside the 1.50–1.80 m the VW Caddy mount allows; crop 1.06× is what a mild video
EIS does. Against what was being rendered (`f 1666, h 1.622, horizon 485`) the horizon was **37 px
low** and f·h **11 % high**.

## 52. What is left is VARIANCE, not bias — and one part of it is not an error at all

At yaw −6.80 the *median* frame is already right (drift 0.21°, gap 0.87–0.93 m against an ideal
0.85 m). The **90th-percentile** frame is off by **0.023 m/m ≈ 1.3° ≈ 0.9 m at 40 m**, on straight
frames, and that is what "no config satisfies all frames" means.

⚠️ **Part of that spread is the car, not the calibration.** The corridor is the *predicted path*,
not the lane. A driver drifting 0.3 m/s inside the lane genuinely puts the path 0.4 m closer to the
line at 40 m — a 0.01 m/m "drift" that is the overlay being CORRECT. Any further optimisation that
drives the per-frame spread to zero would be fitting the calibration to the driver's line.

**Not measured:** how much of the remaining spread is EIS. `probes/road_vp.py` was written to answer
it (per-frame vanishing point from road-parallel segments; its pooling argument is sound — every
road-parallel line passes through `cx + f·tanθ` at `v = v_h` regardless of lateral position) but it
**did not complete** — 35 min without output, HoughLinesP on dilated ridge masks is too slow. It is
banked unrun. The MP4 carries **no optical metadata** (`ffprobe`: no focal, no lens, only
`com.android.version=16`), and SensorLogger records through Camera2, so whether the Samsung camera
app's stabilisation setting reaches this capture is **not established from the file** — the 1.06×
crop implied by the joint fit is the only evidence, and it is indirect.

## 53. A second defect in the same function, found and NOT fixed — on purpose

`lane_calib.estimate` reads its ridge points from rows **0.55–0.86 H** and then keeps
back-projected points with **X ∈ (7, 40) m**. With the adopted calibration those rows are::

    row 594 (0.55 H) -> q = 145.6 -> x = 2431/145.6 = 16.7 m
    row 928 (0.86 H) -> q = 479.6 -> x =  5.1 m

**The band tops out at 16.7 m. The filter asks for 40 m. The upper 58 % of the requested range
does not exist in the input** — the yaw fit has never seen it on this recording, and the width
schedule fix cannot conjure rows that are not read.

The physical band would be `r0 = v_h + f·h/40`, i.e. row **509** here. I have **not** made that
change. It alters behaviour on every recording, the regression that would catch a mistake is a
full pipeline re-run per corpus, and this container cannot run `pytest -q` at all (125 collection
errors, every one `No module named torch`). Changing a band I cannot regression-test, in the same
commit as a fix I can, is how a good change and a bad one become indistinguishable later.

⇒ Banked as a work item with its measurement, not as a silent edit.

---

# PART 11 — the yaw was never the parameter that was wrong first

## 54. ⭐⭐ CAMERA HEIGHT AGAIN, WITH NO HORIZON IN IT AT ALL

§50 measured `h` by holding the lane width range-independent, which needs `v_h`. There is a
**stronger form of the same algebra that needs nothing**. For any road-parallel line,

    u = cx + f·tanθ + y·(v − v_h)/h      ⇒      du/dv = y/h

so the image SLOPE of a lane boundary is `y/h`, and the **difference of the two boundaries'
slopes is 3.5/h — with no horizon, no focal length and no ego motion anywhere in it.**

    MEASURED, 125 adjacent line pairs from straight frames:
        median |slope difference| = 2.2247
        h = 3.5 / 2.2247 = 1.573 m     frame-cluster bootstrap 95% CI [1.511, 1.628]

⚠️ Weaker than §50's histogram in one respect and it must be said: the slope-difference
histogram does **not** show the crisp 1-lane / 2-lane pair of clusters that the width histogram
does, so the 1.6–3.2 single-lane band here is a **prior, not a read-off**.

What makes it decisive is the agreement. §50 gives h = 1.586 m **through** the horizon; this gives
1.573 m [1.511, 1.628] **without** it. Inverting: h = 1.573 requires `r(v_h) = 3.5/1.573 = 2.225`,
and the measured `r` curve reaches that at **v_h ≈ 451** — against the joint fit's **448**.

⇒ **Horizon 448, h 1.58, f 1533 is now confirmed by a route that shares no parameter with the
route that produced it.** It also kills the remaining candidates: h = 1.427 (withdrawn earlier as
circular) and h = 1.468 (the paint-only horizon) are both outside the CI.

## 55. ⛔ EVERY YAW USED THIS SESSION WAS ~1.3° TOO NEGATIVE

With the horizon settled, the yaw can be read straight off the paint. The left lane line's column
**at the horizon row** is `cx + f·tan(yaw)`, because at `v = v_h` the `y/h` term vanishes — so that
one number is the angle between the camera axis and the lane, and on straight road that is the
mount yaw. **No trajectory, no ego motion, no flow.**

    MEASURED over 54 straight frames:   yaw = −5.30°,  robust sd 1.01°,  10–90% [−7.05, −3.91]

Against everything that was rendered: **−7.01, −6.80, −6.59, −7.80, −8.52.** All too negative,
because they were all fitted through a horizon that was 37–75 px too low.

## 56. ⭐ A YAW SCAN THAT NEEDS NO RENDER, AND THE CHECK IT PASSES

A yaw change `δ` displaces the drawn corridor by `f·δ` px at **every** row — its image line moves
in intercept only. So once the corridor's edge line and the lane line are fitted in a delivered
video, **every candidate yaw can be scored on the frames already in hand.** The objective is not
"fewest crossings" (over-rotating the other way maximises that); it is that the two lines should
**meet AT the horizon**, which is what parallel-on-the-ground means.

| video rendered at | scan's answer | median crossing row at the answer |
|---|---|---|
| −6.59 | **−5.34°** | 449.1 vs horizon 448 |
| −6.80 | **−5.30°** | 446.2 |
| −7.80 | **−5.55°** | 448.1 |

Three different images, spread 0.25°. And against the paint-only reading of §55, **−5.30 vs −5.34
— 0.04° apart.**

⭐ **That agreement is also a decomposition.** §55 measures the CAMERA. §56 measures what makes the
corridor — which comes from the TRAJECTORY — parallel to the lane. Had the trajectory carried a
systematic heading bias the two would differ by it. They do not, to 0.04°, so
**the trajectory's heading is unbiased and the entire far-field error was the mount yaw.**
Part 9's diagnosis (trajectory heading) is refuted a second time, now by a measurement that could
have shown the opposite.

## 57. ⚠️ AND THE METRIC I HAD BEEN RANKING ARMS WITH IS HORIZON-DEPENDENT

`out_caddy` measured a drift of **0.21°** on its own grid and **1.22°** on the common grid — same
video, same probe. The algebra says why: with a metric horizon `v_m` and focal `f_m`,

    residual(x) = (P + Q·v_m)·x/f_m + Q·f·h_m/f_m

so **the slope carries `v_m` inside it.** Ranking arms by drift in m/m compares interpretations as
much as overlays, and each arm read through its own wrong horizon flattered itself.

⇒ The ranking statistic must be the one that is a fact about the image: **the row where the
corridor's left edge crosses the painted line.** Two lines meeting at a pixel is not an
interpretation. On the common grid, at the rendered yaws: `caddy` 496, `joint2` 492, `yaw78` 524 —
all far below the horizon at 448, i.e. all three cross the paint on the visible road.

**ADOPTED AND RENDERED:** `horizon 448.4 · h 1.586 · f 1533 · yaw −5.35 · lateral −0.126`.

## 58. ✅ VERIFIED ON THE DELIVERED VIDEO — `out_final2`

`horizon 448.4 · h 1.586 · f 1533 · yaw −5.35 · lateral −0.126`, measured over 220 frames:

| range | corridor left edge → painted line | IQR | n |
|---|---|---|---|
| 10 m | **+1.24 m** | [+1.07, +1.36] | 88 |
| 15 m | +1.22 m | [+1.01, +1.38] | 88 |
| 20 m | +1.22 m | [+0.96, +1.42] | 88 |
| 25 m | +1.21 m | [+0.87, +1.46] | 88 |
| 30 m | +1.20 m | [+0.80, +1.51] | 88 |
| 40 m | +1.28 m | [+0.68, +1.94] | 69 |
| 50 m | +1.15 m | [+0.32, +1.97] | 76 |

**Slope −0.001 m/m.** Straight frames **+0.0023 m/m = 0.13°**, against **1.11°** at yaw −6.59 and
**2.14°** at yaw −7.80 on the same grid. Median crossing row **447.6** against the horizon at
**448.4**. **65 % of straight frames: the corridor never reaches the paint on the visible road**;
when it does, the median is 66 m.

⭐ **The loop closes.** The yaw scan run on THIS render answers **−5.35, i.e. +0.00 from what was
rendered**. The correction predicted from the previous video was applied and the new video needs
none — which is the check every earlier fit failed.

### Why the level is 1.22 m and not 0.85 m, and why nothing should be done about it

Parallel geometry forces it: with `Q = du/dv` for the corridor-minus-lane difference, the metric gap
is `Q·h` at EVERY range, so making the corridor parallel *determines* the level. It says the
corridor's centre sits **0.38 m right of lane centre**.

MEASURED, with no calibration at all — `offset = 1.75·(m_L + m_R)/|m_L − m_R|`, in which `h` cancels
against `|m_L − m_R| = 3.5/h`:

    camera offset from lane centre = 0.245 m, frame-cluster bootstrap 95% CI [0.075, 0.414] m
    per-frame robust sd 0.473 m  (real lane wandering)

**0.38 m is inside that interval.** The car genuinely drives right of lane centre, and the corridor
is supposed to follow the car. Zeroing the level into `--lateral-offset` would be fitting the
calibration to the driver's line — the confound §40 already refused once.

### What is NOT fixed

- **Per-frame spread.** 90th-percentile |slope| on straight frames is **0.0388 m/m ≈ 2.2° ≈ 1.55 m
  at 40 m**. Part of that is real (the path genuinely approaches the line when the driver drifts);
  how much is EIS is **unmeasured** — `probes/road_vp.py` was written for it and did not complete.
- **The steering panel uses the wrong car.** `pipeline.py:994` defaults `--wheelbase` to the
  **Audi A6 e-tron's 2.946 m**; Sayed's car is a **VW Caddy** (≈2.68 m gen-4, ≈2.76 m gen-5). This
  changes the reported steering-wheel angle and rate, and nothing else — the overlay geometry does
  not use it. Left as an operator flag, not guessed.

## 59. ⭐ The ridge-width fix shows up in the pipeline's OWN automatic calibration

The fix in §50 was made for the probes. Its effect on `lane_calib` itself, read out of the
`out_caddyv` run's log (`--horizon-row`/`--cam-height` are applied AFTER these estimators, so
these are the pipeline's own unaided numbers):

    LaneCalib(yaw=-6.86 deg, lateral=declined, lane_width=3.40 m,
              frames=59, segments=981, yaw_spread=0.31 deg)
    pitch cross-check: lane VP row is 7.9 px from the FOE row

| | before the fix | after |
|---|---|---|
| lane VP vs FOE row | **~59 px** apart (523.4 vs 464.4 — the R-2026-09-13-horizon dispute) | **7.9 px** |
| reconstructed lane width | grew **41.5 %** from the 8–13 m slab to the 18–25 m slab | **3.40 m** against a true 3.5 m |

The horizon dispute that cost a retraction was, in part, **a detector artefact**: an operator sized
by the row's rank in its band lost the far paint, and the lane VP was fitted to whatever near-field
segments survived.

⚠️ **NOT resolved, and recorded as open:** the pipeline's own yaw is still **−6.86° (lane VP) /
−6.31° (FOE)** against the **−5.30°** measured in §55. The relation `dyaw/dv_h = m/f ≈ 0.029°/px`
says a 54 px horizon difference would account for it exactly, which points at the horizon
`lane_calib` uses internally rather than at the yaw estimator. Not chased here — it does not touch
the delivered overlay, which takes the operator's overrides.

## 60. A label must not outlive the values it describes

Overriding `--wheelbase 3.105` printed:

    Steering [Audi A6 e-tron, L=3.105 m, ratio 15.9:1]

The wheelbase was applied correctly; only the **name** was stale — and the remaining fields
(steering ratio, understeer gradient, lock, track) really are still the Audi's, so the line asserts
more than the run knows. **Same failure as the yaw gate calling a nominal placeholder "the FOE":
the defect was never the number, it was a label that made a placeholder look like a measurement.**

`--vehicle-name` added; without it, overriding either dimension relabels the vehicle
`unnamed vehicle (wheelbase overridden; other Audi A6 e-tron values retained)` — which also says
what is still inherited, rather than trading a wrong name for a claim that nothing is.

## 61. ✅ DELIVERED — `out_caddyv`, VW Caddy V Maxi

Sayed: *"ist caddy last generation and the long version"* ⇒ **Caddy V (SB, 2020–) Maxi**.

| flag | was | now | what it touches |
|---|---|---|---|
| `--wheelbase` | 2.946 m (Audi A6 e-tron default) | **3.105 m** | the steering read-out **only** — `apply_drivability` is off by default and `pipeline.py` never calls it, so neither the path nor the corridor depends on it |
| `--vehicle-width` | 1.8 m (generic) | **1.855 m** (body, excl. mirrors) | the drawn ribbon **and** the BEV panel |

Verified on the delivered video, 220 frames, calibration unchanged:

| 10 m | 15 m | 20 m | 25 m | 30 m | 40 m | 50 m |
|---|---|---|---|---|---|---|
| +1.22 | +1.20 | +1.19 | +1.16 | +1.17 | +1.30 | +1.16 m |

**Drift +0.0000 m/m.** Crossing row **447.6** vs horizon **448.4**; **69 % of straight frames never
reach the paint**. Yaw scan on this render: **−5.35, +0.00 from rendered.**

⚠️ Two Caddy numbers are still **not** the Caddy's and are flagged rather than guessed:
- **steering ratio 15.9:1** is the Audi's. The panel now says so (§60).
- **`--mount-longitudinal 2.1 m`** — the phone's distance ahead of the rear axle. With a 3.105 m
  wheelbase the published dimensions suggest ~2.3–2.5 m, but that is arithmetic on a bonnet length
  I do not have. On a mostly-straight motorway clip it is second-order (it only separates the
  phone's path from the vehicle's through turns), so it is left at the value already validated.
  **A tape measure from the rear-axle centreline to the phone would settle it.**

⛔ **The RIGHT-side clearance is NOT measured.** A mirrored version of the left-edge probe returned
**3 usable frames of 220** — the global ridge threshold is set by the brighter left half of the
image and buries the right-hand line. Arithmetic gives 3.5 − 1.22 − 1.855 ≈ **0.43 m**, but that
assumes the left reading and a 3.5 m lane and is **not an independent measurement**.

---

# PART 12 — SLAM: the right family, already in the pipeline, and measurably broken

Sayed: *"Back to my idea of using a sort of slam/optimization by looking for which parameter
configuration lead to the best match of static features? Do visual slam help here?"*

## 62. ⛔ First, §54 is WITHDRAWN — see `R-2026-09-14-slopeheight`

The "no-horizon" height (1.573 m) came from the median of a **band I chose from expectation**. The
histogram inside that band is **flat** (n = 14/8/8/4/6/12/9/9/10/10/5/3/5/7/9/5 across 1.6→3.2), and
the estimate moves with the frame sample (1.573 → 1.504 m). It was reporting my prior back to me.
§50's lane-WIDTH measurement survives — its histogram really does show a one-lane cluster and a
two-lane cluster **exactly 2× apart** — so `h ≈ 1.59 m` stands on one route, not two.

## 63. The idea is right, and the literal version of it is degenerate

**"Optimise the parameters for the best match of static features"** has a gauge problem: scene scale,
camera height and the focal length trade against each other exactly. Bundle adjustment over static
features alone has a 7-DoF similarity freedom, and for a camera translating along its own optical
axis there is a further near-degeneracy between focal length and depth. **A dashcam driving forward
is close to the worst case for monocular SfM**: the road ahead — precisely where the corridor lives —
sits near the epipole and has almost no parallax. That is the same `f·h` degeneracy this whole
document has been fighting, in a new coordinate system; it does not dissolve by adding more
features, only by adding a **metric anchor** (the odometer baseline, or a known lane width).

## 64. ⭐ But the right member of that family is ALREADY IN THE PIPELINE — and it is silently dead

`plane_calib.py` decomposes the inter-frame road-plane homography `H = R + (t/d)·nᵀ`. The normal `n`
**is** the camera's roll and pitch relative to the road **per frame pair**, and with the metric
baseline from the trajectory, `t/d` gives the **camera height**. It sidesteps the forward-motion
degeneracy by using the known planar structure instead of general structure-from-motion. On the
08-11 session it worked: 93 usable pairs, height 1.167 m [1.100, 1.223].

On this recording it prints one line and stops:

    WARN  plane calibration produced too few usable homographies

**MEASURED (`probes/plane_frontend.py`, 120 pairs), where it actually dies:**

| front end | features | tracked | RANSAC inliers | usable pairs |
|---|---|---|---|---|
| `goodFeaturesToTrack` (shipped) | 500 | 96 | 16 | **5/120** |
| + CLAHE | 500 | 88 | 12 | 11/120 |
| ridge-seeded (paint) | 894 | **192** | **42** | **0/120** |
| ridges + features | 1394 | **282** | **47** | **0/120** |

⇒ **The front end is NOT the bottleneck.** Ridge seeding triples the inliers and *still* yields
nothing: 120/120 die at `no admissible decomposition`.

## 65. ⛔ The bonnet is driving the homography

`collect_road_tracks` masks a corridor of `x ∈ (5, 32) m`, half-width 3.2 m, projected with `cam`.
MEASURED: that polygon spans **source rows 529–1079**. Row 832 (0.77 H) is the bonnet line
`flow_scale` already measured — below it the image is **static** — so **45 % of the mask's rows are
not road at all.** RANSAC fits the homography to the static part:

| mask | inliers | recovered normal | pitch | roll | height | admissible |
|---|---|---|---|---|---|---|
| as shipped | 54 | [+0.03, −0.94, **+0.31**] | **+17.8°** | +1.7° | **42.0 m** | **0/90** |
| bonnet cut at 832 | 16 | [−0.22, −0.88, +0.04] | +2.3° (IQR −7.2…+8.5) | −12.7° | 2.35 m | 12/73 |
| *a level road wants* | | *[0.00, −1.00, −0.06]* | *−3.4°* | *~0* | *~1.59 m* | |

A **42 m** height is the signature: `|t|/d → 0` because the homography is near-identity, which is
what static content gives. **This is the same class as the ridge-width bug and the panel-width bug —
a front end quietly measuring something other than the road.**

⚠️ **The bonnet cut is necessary but NOT sufficient.** With it, pitch scatters ±8° per pair and roll
lands at −12.7°. The roll is not a surprise: `plane_calib`'s own docstring already records that
**roll is geometrically unobservable this way** (`roll ≈ n_x`, the normal's lateral component). The
pitch scatter is the real blocker — over a 4-frame baseline at 80 km/h the camera travels 2.8 m, and
a real road is not planar over that distance. Tightening the gates (RANSAC 1.0 px, ≥20 inliers)
leaves 0–3 pairs of 110. **The 08-11 session was slower; motorway speed makes the same frame gap a
much longer baseline, and the planarity assumption pays for it.**

## 66. ⚠️ And the tempting shortcut is circular

The per-frame quantity is reachable straight from the paint — yaw from the lane line's column at the
horizon row, per frame, already measured at **robust sd 1.01°**. **Using it to correct the camera
per frame would be wrong**, and the reason matters: that sd is *camera jitter plus the vehicle's real
yaw relative to the lane*. Driving it to zero would force the corridor to be parallel to the lane in
every frame — which deletes exactly what the overlay exists to show, namely that the car is changing
lane, drifting, or turning. **A per-frame correction has to come from an instrument that measures the
CAMERA (homography, VO, gyro), never from one that measures the LANE.**

⇒ **The honest state:** per-frame attitude is the right fix for the residual spread (90th pct 2.2°,
1.55 m at 40 m), the right instrument for it is in the repo, its mask bug is diagnosed and fixable,
and its conditioning at motorway speed is an open problem — not a matter of turning on SLAM.

## 67. ✅ THE FIX, AND IT TURNS A DEAD ESTIMATOR INTO AN INDEPENDENT CONFIRMATION

`ground_calib._drop_static` — added to `collect_road_tracks`. It needs no bonnet row: a point on the
road must move by roughly what the **known vehicle displacement** predicts, so anything moving far
less is not on the road, whatever it is. The gate is deliberately **one-sided** (it only rejects
points that move too little); a two-sided version would be a fit against `cam`, which is the thing
the estimator exists to measure.

MEASURED, 110 pairs, 4-frame baseline:

| front end | inliers | admissible | pitch | roll | height |
|---|---|---|---|---|---|
| as shipped | 23 | 9/110 | **+13.66 ± 1.80** | −6.41 ± 12.99 | 1.602 ± 0.539 m |
| + motion filter | 16 | 2/110 | — | — | — |
| **+ motion filter, ridges + features** | 14 | **8/110** | **−3.07 ± 1.66** | −5.30 ± 6.87 | **1.592 ± 0.388 m** |
| *independent target* | | | *−3.42°* | *~0 (unobservable)* | *~1.59 m* |

⭐ **Pitch −3.07° against −3.42°, height 1.592 m against 1.586 m** — from the inter-frame plane
homography plus the odometer baseline, a route that shares **nothing** with the lane-width
range-constancy and row-flow pair that produced the adopted set. This is the third criterion
`R-2026-09-13-horizon` demands, and it lands on the adopted values. Roll is consistent with zero and
with `plane_calib`'s own note that roll is geometrically unobservable this way.

⚠️ **Corroborates, does not tighten.** Only 8 of 110 pairs are admissible and the per-pair MAD on
height is ±0.39 m. It is a check, not a replacement for the lane-width measurement — and the yield
has to come up before this can deliver the PER-FRAME attitude that the residual spread actually
needs.

## 68. ⭐ The residual spread is REAL — the instrument contributes 3 % of it

Before attributing the per-frame spread to EIS, the trajectory or the driver, ask how much of it is
the probe. The lane line was fitted **twice per frame on interleaved rows** (odd vs even — same
range span, independent noise), and the two yaws compared, over 66 straight frames:

    yaw, full fit          median −5.33°   robust sd 2.045°
    split-half difference  median +0.001°  robust sd 0.126°   ⇒ σ_meas = 0.063°

**σ_meas = 0.063°, σ_observed = 2.045°: 100 % of the variance is real.** At 40 m that is 1.43 m of
real frame-to-frame variation against 0.04 m of my own noise. The spread is not an artefact of the
measurement — which is the first thing every earlier "spread" in this session turned out to be.

## 69. And it is not road curvature aliasing into a straight-line fit

A chord across a 10–50 m span on a 2100 m radius (the |yaw rate| ≤ 0.6 °/s gate) is already tilted
~0.8° from the tangent, so the obvious suspect is my own straight-line lane model. Measured against
both knobs:

| fit span | \|yaw rate\| ≤ | n | median yaw | robust sd | corr. with yaw rate |
|---|---|---|---|---|---|
| 10–50 m | 0.6 | 183 | −4.98° | 2.284° | −0.188 |
| 10–50 m | 0.2 | 190 | −5.18° | 2.443° | −0.197 |
| 10–25 m | 0.6 | 148 | −5.12° | 1.773° | −0.163 |
| 10–18 m | 0.2 | 89 | −5.40° | 1.485° | −0.150 |

**It does not collapse.** Shortening the span 10–50 → 10–18 m moves it 2.28 → 1.49° and tightening
the yaw-rate gate does not help at all. Correlation with the vehicle's yaw rate is −0.07…−0.20, so
it is not the driver steering either.

⚠️ **Half of it is line misidentification in this ungated probe.** `overlay_far`, which carries an
identity gate, reports **1.01°** on the same quantity against 2.04° here — so the defensible figure
is **~1° of real per-frame camera-to-lane yaw variation ≈ 0.70 m at 40 m**.

⛔ **NOT YET SEPARATED, and this is the crux:** that ~1° splits between the vehicle's genuine yaw
relative to the lane (the lateral-position probe puts per-frame wander at robust sd 0.473 m, which
at 22 m/s implies roughly 0.6°) and actual camera rotation (EIS and/or suspension). Yaw RATE cannot
separate them — a constant yaw *offset* relative to the lane does not appear in a rate. The
instrument that can is the **gyro**, compared against image rotation measured in the far field where
`H = K·R·K⁻¹` exactly.

---

# PART 13 — STEP 1: it is NOT the camera. And my residual metric is noisier than I said.

## 70. Two instrument designs failed first, and both are recorded

`probes/image_rot_vs_gyro.py` estimates camera rotation depth-free: for a rigid scene,
`flow = A(x)·ω + s·e`, so the flow component **perpendicular to the FOE-radial direction is purely
rotational at any depth**. Elegant, and it did not work.

| design | 0–0.5 Hz corr. with the gyro | why |
|---|---|---|
| FOE taken from the calibration | **0.09** | a near-road point moves ~100 px/frame; a 20 px FOE error at radius 300 px injects 6.7 px of false "rotation" against a real signal of 1.5 px — **4× the signal** |
| FOE solved for, alternating | **0.22** | a yaw rotation and a horizontal FOE shift are near-degenerate; the re-fit absorbs the rotation |

Neither reproduced the vehicle's own turning, which both the image and the gyro **must** see. The
gyro side was validated first and passes: against the trajectory's yaw rate, **r = +0.992, slope
+0.992** below 0.5 Hz. So the fault was mine, and no EIS conclusion was read off either.

⚠️ **The broadband validation gate I first wrote (|r| > 0.8) was unmeetable by construction.** The
trajectory's yaw rate is a 0.6 s fit, i.e. the low-passed gyro, so `r = sd(traj)/sd(gyro) =
0.0219/0.0540 = 0.405` — and the measurement was 0.404. A gate that a correct instrument cannot
pass is not a validation.

## 71. A third design, and a correlation that was an artefact

Comparing the camera-to-lane angle `θ(t)` against the gyro-integrated vehicle heading on a straight
run gave **corr 0.73–0.79, slope 4–5** — which read as "the camera over-rotates 4× relative to the
car". It is an artefact: **44 % of frames had no lane fit and were interpolated**, and interpolation
manufactures smooth structure that correlates with any smooth signal. Re-run on fitted frames only,
with the gyro integrated at its own 105 Hz before sampling: **corr 0.075 / −0.008 / 0.050 / 0.007**
across every detrending window. The 4× slope evaporates with it.

## 72. ⭐⭐ THE COMMON-MODE CONTROL — and it answers Step 1

Fit the lane **twice per frame on disjoint range bands**, near (10–16 m) and far (16–40 m). Both look
through the *same* camera, so **a camera rotation moves both identically**; a wobble that is the fit
or the paint does not. 744 frames on the longest straight run:

    near-band theta   detrended rms  1.393 deg
    far-band  theta   detrended rms  2.373 deg
    correlation near vs far          -0.072
    differential rms 2.837 deg  =>  per-band noise 2.006 deg
    common-mode  rms 1.332 deg  (noise alone would give 1.419 deg)

**The two bands are uncorrelated.** Common-mode camera rotation is consistent with **zero**, with a
2σ upper bound of about **0.25° rms ≈ 0.17 m at 40 m.**

⇒ **PRE-REGISTERED OUTCOME, taken as committed: the camera is not the lever. Do NOT build Step 2.
Go to Step 3.**

## 73. ⚠️ And it undercuts §68's noise estimate, which I must correct

§68 put the instrument noise at **0.063°** from an odd/even row split. That is too optimistic:
interleaved rows are adjacent samples of the *same* paint over the *same* range span, so they share
every systematic the fit has and differ only in pixel noise. The disjoint-band split says a fit over
a shortened span carries **~2°**. The full-band fit's true noise lies between the two and is **not
0.063°** — so the "100 % of the variance is real" claim in §68 is **not supported at that
precision**, and the ~1° per-frame residual is part lane-fit variability, part real motion, in a
proportion I have not yet measured.

**Consequence, and it points the next move:** before chasing the trajectory, the residual metric
needs an honest noise floor — two *independent full-span* estimates of the same frame's lane angle
(e.g. left line vs right line, or alternating RANSAC seeds on disjoint point subsets over the whole
span). If the floor turns out to be ~0.5°, most of the "residual" is my ruler and the overlay is
already better than the numbers I have been quoting.

## 74. ⭐ THE NOISE FLOOR — 0.088°, and the hopeful reading is refuted

`probes/noise_floor.py`, 628 frames of the longest straight run, span 10–40 m, **no continuity
gate** (the delivered metric's gate constrains each frame toward the previous one and would drive
the floor artificially low).

**The structure function** `D(τ) = rms[θ(t+τ) − θ(t)]`. Independent per-frame noise contributes a
constant `√2·N` at *every* lag; anything real must grow from zero, because nothing moves in 33 ms.

| lag | 0.033 | 0.067 | 0.10 | 0.20 | 0.33 | 0.50 | 1.0 | 1.5 | 2.0 | 3.0 s |
|---|---|---|---|---|---|---|---|---|---|---|
| D(τ)° | 0.125 | 0.138 | 0.166 | 0.226 | 0.294 | 0.425 | 0.791 | 1.177 | 1.395 | 1.950 |

A clean floor that grows monotonically and saturates near 3 s.

    NOISE FLOOR   N = 0.088 deg  = 0.06 m at 40 m  = 0.0015 m/m
    REAL          S = 1.376 deg  = 0.96 m at 40 m

⇒ **The residual is real.** `overlay_far`'s 90th-percentile |slope| of 0.039 m/m is **25× the
floor**. The hopeful possibility I raised — *"if the floor is ~0.5°, most of the residual is my
ruler"* — is **refuted**.

⇒ **§73 over-corrected §68 and is itself corrected here.** §68's odd/even estimate was 0.063°; the
true floor is 0.088°, a factor of 1.4 away, not the factor of 30 that §73's disjoint-band figure
implied. The ~2° from short bands never transferred to the full span — a badly conditioned fit is
not a noise floor for a well conditioned one.

### ⛔ Estimator A (left line vs right line) is NOT a noise estimate here

It returned N = 1.827°, and the structure function refutes it in one line: **if per-frame noise were
1.8°, consecutive frames would differ by √2·1.8 = 2.6°. They differ by 0.125°.** The left-right
disagreement is systematic, not per-frame — the right boundary carries 55 median inliers against the
left's 136 and is frequently not the lane edge at all.

⭐ **Its median part is a measurement, though, of something else.** With
`θ = atan((m·v_h + b − cx)/f)`, `dθ/dv_h = m/f`, and the two boundaries have **opposite** slopes, so
a horizon error moves them in opposite directions. With `m_L − m_R = −3.5/h`, the measured offset of
**+1.125°** implies `Δv_h = −13.6 px`, i.e. **horizon 434.8** rather than the adopted 448.4 — and
row flow alone preferred **438**. Two independent routes now sit at 435–438 while lane-width
range-constancy sits at 470 and the joint fit at 448. *(Flagged, not acted on: the right-boundary
contamination that kills A as a noise estimate also weakens it as a horizon estimate, and the lens
constraint admits the whole 435–458 range.)*

### What the timescale says

`D(τ)` half-saturates around **1 s** and saturates by **3 s**. That is not camera shake; it is the
timescale of driver lane-keeping and of gentle road curvature. Together with §72 (camera common-mode
≤ 0.25°), the source of the residual is the **vehicle and/or the trajectory**, not the camera and
not the instrument.

⚠️ **And the reframing that follows.** θ varying over seconds is not by itself an overlay error — the
road curves and the car moves, and the corridor is *supposed* to follow the car. The open question
is sharper than "how big is the residual":

> **Does the corridor's predicted deviation actually come true?** When the corridor says at time `t`
> that the car will be 1 m left of lane centre in 2 s, is the car measurably there at `t+2 s`?

The lateral-position probe answers it with **no calibration in it at all** —
`offset = 1.75·(m_L + m_R)/|m_L − m_R|`, in which `h` cancels. If the prediction verifies, the
overlay is right and the rejected frames are correct. If it does not, the trajectory's heading is
the defect. **That is Step 3, and it is now a one-experiment question.**

---

# PART 14 — STEP 3: the angle is partly real driving, and the rest is MY ruler

## 75. Does the predicted deviation come true? Partly — and the lag structure says which part

`probes/does_it_come_true.py`, 595 frames of the longest straight run with a valid single-lane pair.
Both inputs come from the paint and **neither needs a calibration**: `ψ_rel = θ_L − median(θ_L)`
(the median removes the mount yaw — it came out **−5.38°** against the adopted −5.35, measured not
assumed) and `L = 1.75·(m_L + m_R)/|m_L − m_R|`, in which `h` cancels.

| lag | pairs | predicted rms | measured rms | slope | corr | skill |
|---|---|---|---|---|---|---|
| 0.25 s | 547 | 0.102 m | 0.345 m | 0.28 | 0.083 | 0.01 |
| 0.50 s | 477 | 0.206 m | 0.376 m | 0.21 | 0.117 | 0.01 |
| 1.00 s | 477 | 0.427 m | 0.387 m | 0.19 | 0.206 | 0.04 |
| 1.50 s | 466 | 0.642 m | 0.432 m | 0.24 | 0.363 | 0.13 |
| 2.00 s | 459 | 0.884 m | 0.470 m | 0.19 | 0.358 | 0.13 |
| 3.00 s | 436 | 1.350 m | 0.497 m | 0.16 | 0.426 | 0.18 |

Neither pre-registered extreme. **The correlation GROWS with lag, 0.08 → 0.43** — which is the
committed signature of *real heading*: a shared per-frame fit artefact would be largest at the
shortest lag and fade, and this does the opposite. So the angle carries genuine vehicle heading.

**But the slope is 0.16–0.28, not 1.** Regression dilution reads that directly: `slope =
var(true)/var(measured)`, so **only ~19 % of ψ_rel's variance predicts where the car actually goes.**
With ψ_rel at 1.19° rms that is **≥ 0.52° of real heading and ≤ 1.07° of error** (a lower bound on
the real part, since a driver who corrects realises less than the full heading).

⭐ **And the error is MINE, not the overlay's.** `ψ_rel` is measured purely from the paint — the
trajectory never enters it. So the ~1.07° that fails to predict the car's motion is error in my
**image measurement of the lane angle**: line identification wandering, paint irregularity, camber.
The residual metric is built on that same lane angle, so **it inherits ~1.07° of my own ruler**.

⇒ **This reconciles §74 with §73.** The structure function's 0.088° floor is the *fast* noise only;
lane-fit error that drifts over ~1 s is invisible at a 33 ms lag and is an order of magnitude larger.
**The per-frame "residual" I have been quoting is dominated by my lane measurement, not by the
overlay.** The MEDIAN statistics — which ranked the arms, closed the yaw loop at +0.00, and gave the
flat 1.15–1.28 m profile — average this down and are unaffected.

## 76. ⭐ And this explains the rejected frames without any error at all

The corridor is drawn 1.855 m wide around the car's path. Two things measured across the whole clip
(426 frames, both from the paint):

| time | lane width | ego offset from lane centre |
|---|---|---|
| 13–22 s | 3.66 m | −0.54 m |
| 22–41 s | 3.58–3.63 m | −0.36 m |
| **41–50 s** | **3.27 m** | **−0.73 m** |
| 50–59 s | 3.41 m | **−0.90 m** |
| 59–68 s | 3.42 m | −0.66 m |

**The road narrows to 3.27 m and the driver sits up to 0.90 m off centre.** Clearance on the tight
side for a 1.855 m ribbon at the median 0.48 m offset:

    lane 3.5 m -> +0.34 m      lane 3.2 m -> +0.19 m
    lane 3.0 m -> +0.09 m      lane 2.8 m -> -0.01 m

**29 % of frames read narrower than 3.2 m.** In those, a geometrically correct corridor has ~0.1–0.2 m
of clearance and *looks* as though it is on the line or the kerb — which is exactly the frame Sayed
rejected last. **No angle error is needed to produce it.**

⚠️ The absolute widths inherit the 3.5 m that calibrated `h` on this same corpus, so read them as
**ratios**: the road varies by **1.25×** between its narrowest and widest, and that variation is
measured independently of the assumption.

---

# PART 15 — No, not satisfied. And the unmeasured parameter is named.

## 77. ⛔ The lateral PLACEMENT of the corridor is not measurable with this detector

`probes/does_it_cut_paint.py` set out to measure Sayed's actual symptom. It cannot, and the way it
fails is the result:

| gate on the lane pair | frames kept | "corridor − vehicle" |
|---|---|---|
| slope band only (1.6 < \|m_L−m_R\| < 3.2) | 140/300 | **−1.22 m** |
| + range-constancy of the implied width | **18/300** | **+0.93 m** |

**The answer swung 2.15 m and changed sign on a gate choice. Neither value is a measurement and both
are withdrawn.** The first gate's own output shows why: it reported a "lane" of **6.41 m at 10 m
growing to 18.28 m at 40 m**. A lane does not do that — the right-hand line was not a lane boundary,
and every placement number built on it was nonsense. Adding the correct gate (two road-parallel lines
converge at the horizon, so the implied width must be range-independent) leaves **18 frames of 300**.

⇒ **On this recording the RIGHT-hand boundary cannot be detected reliably enough to locate the car in
its lane.** 55 median inliers against the left line's 136; 94 % of frames rejected by a parallelism
test. Every probe in this document that needed both boundaries — the lane-width height, the ego
offset, this placement check — inherits that weakness, and only the ones using the LEFT line alone
are solid.

## 78. What IS solid, and what it implies about the symptom

Measured with the left line alone (reliable), on the delivered render:

    corridor left edge -> left painted line:  1.15 - 1.28 m, FLAT from 10 m to 50 m
    drift 0.00 m/m; crossing row 447.6 against a horizon of 448.4

For a **1.855 m** ribbon centred in a **3.5 m** lane that gap should be **(3.5 − 1.855)/2 = 0.82 m.**
It is **1.2 m**, so the corridor sits about **0.38 m right of lane centre**, which puts its right
edge roughly **3.06 m** from the left line — i.e. **0.2–0.45 m from the right-hand markings**
depending on the local lane width, which itself varies 3.27–3.66 m across the clip.

**That is the symptom, and the angle is not causing it.** The corridor is parallel to the lane; it is
sitting too far to one side.

## 79. ⭐ The one parameter never measured — and its sign, now pinned

`--lateral-offset = −0.126 m` (the camera's distance from the vehicle centreline, + = left) is
**inherited and has never been measured**. It is the only calibration parameter in the set that has
not been, and it is exactly the one that decides §78.

**Sign convention verified end to end** (single-frame render, corridor edges read at the 20 m row):

| `--lateral-offset` | corridor centre | shift |
|---|---|---|
| −0.426 | 764 px | −0.31 m |
| **−0.126** (shipped) | 788 px | 0.00 m |
| +0.174 | 810 px | **+0.30 m** |
| +0.474 | 834 px | **+0.60 m** |

Exactly 1:1 and linear: **a more positive offset moves the corridor right, a more negative one moves
it left.** So the correction is a single number applied directly.

⛔ **I am NOT setting it to whatever centres the corridor.** That fits the calibration to the
assumption that the car drives centred — the confound §40 refused, and the same shape as the
circularity that killed `lane_residual.py`. The corridor is supposed to follow the *car*.

⇒ **The decisive input is a tape measure in the car: the horizontal distance from the vehicle's
centreline to the phone's lens, and which side.** With it the corridor moves by exactly that amount
and the right-hand clearance becomes a prediction testable against the left line alone.

---

# PART 16 — ⛔ THE PAINT DETECTOR HAS BEEN DETECTING THE ROADSIDE BANK

See `R-2026-09-14-foliage`. `ridge_cols` takes the brightest narrow features in each image row at
the row's 99th percentile; on this footage the sunlit vegetation and pale rock bank to the RIGHT are
brighter and more textured than the markings, so the detections land overwhelmingly on the bank.
Drawn on the image, the fitted "lane pair" sits in the bushes.

**Withdrawn:** the camera height (§50), the horizon's lane-width branch (§51), the lane-width
variation and ego offsets (§75–76), and everything in this turn's pair search.

**Not implicated:** `overlay_far`, which gates every candidate line to within 0.75 m of a predicted
position near the corridor — far inside the bank — and therefore the flat 1.15–1.28 m profile, the
crossing row, the render-less yaw scan and the loop closing at +0.00. Left-line measurements are
largely intact; the contamination is on the right of the carriageway. The delivered renders are
unaffected.

## 80. Four warnings, each explained away

| the detector said | I read it as | it meant |
|---|---|---|
| right boundary: 55 inliers vs the left's 136 | "the dashed line is sparse" | this is not a line |
| 94 % of frames fail a parallelism test | "the detector is weak here" | the pair is not a lane |
| implied lane 6.41 m at 10 m → **18.28 m** at 40 m | "the gate is wrong" | not a lane at all |
| two detectors agree on the left line in **0 of 260** frames | — | *this* is what made me look |

⇒ **A physically impossible intermediate value halts the chain.** An 18 m lane, a 42 m camera
height (§65), a 0.82 m camera height (§50) — each was treated as a threshold to tune. Each was the
measurement saying it was measuring something else.

⇒ **RULE, and it is the cheapest one in this document: before the first statistic, draw the
detector's INPUT on the image and look.** Not the fit, not the residual — the raw detections. Two
minutes would have saved a day, and all four warnings above were cheaper to obtain and none of them
was conclusive.

## 81. The fix, and what has to be redone

`ground_calib.collect_road_tracks` already masks to the projected road corridor. The probes in this
directory do not. Every one of them needs the same mask, and then:

1. re-measure the lane width, and with it **h** — currently the weakest link in the adopted set;
2. re-measure the ego's lateral position, which is what decides whether the corridor's placement is
   a defect or the truth;
3. re-run the horizon joint fit with a clean lane-width branch.

Until then the adopted calibration stands on: `f·h` from row flow (no paint), the horizon from row
flow plus the lens constraint, the yaw from `overlay_far`'s gated fit and its render-less scan, and
the loop closure on the delivered video. **`h` itself is the number now without support.**

---

# PART 17 — the road mask, and h re-measured behind it

## 82. The mask, and the preview that is now part of the procedure

`probes/road_mask.py`. A corridor projected from the calibration, cut at the bonnet row (832),
**asymmetric on purpose**: a symmetric ±3 m corridor previewed with its LEFT edge exactly on the ego
lane's left boundary while its RIGHT edge overran the shoulder — most slack on the side with the
contaminating bank. Widened left (4.2 m), trimmed right (2.6 m).

Running the file writes an annotated preview: **green = detections kept, red = rejected**. On this
recording every red point is on the bank and the green ones sit on the paint. **That preview is not
optional — it is the step whose absence produced `R-2026-09-14-foliage`.**

## 83. h re-measured — and it was 5 % LOW, not high

Masked ridge points + the constrained pair search (both boundaries share a vanishing point, so a
non-parallel "lane" is not expressible). 199 pairs from 300 frames, and the histogram is finally
**unimodal and tight** where the unmasked one was flat:

    1.9-2.0  n 28   |   2.0-2.1  n 61   |   2.1-2.2  n 71   |   2.2-2.3  n 22

    |m_R - m_L| = 2.095  ->  h = 3.5 / 2.095 = 1.668 m
    frame-cluster bootstrap 95% CI [1.655, 1.685] m     (withdrawn §50 value: 1.586 m)

**Converged under iteration** — the mask is projected with h, so it was re-run with the new value:
1.586 → 1.668 → 1.668.

## 84. ⚠️ But h ALONE IS MEANINGLESS — it is a curve against the horizon

`|m_R − m_L| = Δu/(v − v_h)`, so a higher horizon makes the separation larger and h smaller. With
`f·h` from the row flow (ego motion, no paint) at each pinned horizon:

| horizon | pairs | \|Δm\| | **h** | f·h | **f** | crop | |
|---|---|---|---|---|---|---|---|
| 440 | 126 | 2.034 | **1.721** | 2610 | 1517 | 1.05 | ⚠ very high for a screen mount |
| **448** | 127 | 2.098 | **1.668** | 2431 | **1457** | **1.01** | |
| 456 | 129 | 2.178 | 1.607 | 2260 | 1406 | 0.98 | |
| 464 | 120 | 2.262 | 1.547 | 2094 | 1353 | 0.94 | |
| 472 | 104 | 2.346 | 1.492 | 1935 | 1297 | 0.90 | |
| 480 | 84 | 2.404 | 1.456 | 1779 | 1222 | 0.85 | |

⚠️ **Correcting my own printed note in that run:** it said f is "flat across the scan". **It is not** —
f runs 1517 → 1222, a 24 % spread. It is *flatter* than either component because h and `f·h` move
together, but the **lens constraint still discriminates**: EIS can only crop in, so `f ≥ ~1442` and
the admissible band is **horizon ≲ 450 ⇒ h = 1.67–1.72 m, f = 1457–1517 px, crop 1.01–1.05×.**

⭐ **f lands essentially on the uncropped lens.** Crop ≈ 1.0 means **EIS is not cropping this
capture** — consistent with SensorLogger recording through Camera2, where the phone's own
stabilisation setting need not apply. That is the first direct evidence either way.

## 85. The horizon, measured a third way — and excluded a third time

Two UNCONSTRAINED lines fitted to the masked points intersect at row **467.7**, robust sd 9.0,
bootstrap CI **[466.2, 469.0]**. That is the lane vanishing point, now from clean data.

It is **excluded by the lens** (it needs crop 0.93×), exactly as the paint-only horizon was before —
and exactly as `R-2026-09-13-horizon` warns: *a vanishing point measures where lane lines converge;
the horizon is where the GROUND PLANE vanishes, and on a crowned road those are not the same row.*
Three routes now: row flow **438**, lens-admissible band **≲450**, lane VP **468**.

## 86. ⛔ Still open, and it is the one that moves the corridor off the paint

The camera yaw from the masked pair's `u_vp` is **−6.20° to −6.51°** (robust sd ~1.7 per frame,
~200 pairs) against the rendered **−5.35°** and `overlay_far`'s gated left-line fit at **−5.30°**.

The two are not the same measurement and the difference is the size a ~30 px horizon error produces
(`∂yaw/∂v_h = m/f ≈ 0.039 °/px`). **Going more negative moves the corridor LEFT — away from the
right-hand markings — by 0.4 m at 20 m and 0.6 m at 30 m.** That is the direction the symptom needs,
which is a reason to be *more* careful with it, not less.

## 87. ✅ The yaw scan, re-run behind the mask — and it CONFIRMS the rendered value

`overlay_far` now takes the road mask (`--mask`, default on). Its identity gate already gave it
partial protection, and the re-run shows that protection was real: **the numbers barely move, and
the ones that do, improve.**

| | unmasked | **masked** |
|---|---|---|
| coherent association | 88/220 | **146/220** |
| yaw from the paint alone | −5.30° | **−5.29°** (sd 1.03, n 74) |
| yaw scan optimum | −5.34° | **−5.10°** |
| profile 10→30 m | 1.24 … 1.20 m | **1.19 / 1.15 / 1.15 / 1.12 / 1.13 m** |
| never-crosses at the optimum | 65 % | **70 %** (straight frames) |

**Rendered at −5.35, the scan's optimum is −5.10 — a difference of 0.25°, i.e. 0.17 m at 40 m,**
well inside the per-frame spread. The delivered render's yaw is right.

## 88. ⛔ And that withdraws the −6.20/−6.51° yaw from §86

The masked pair search reported the camera yaw at **−6.20 to −6.51°**, against `overlay_far`'s
**−5.29°**. Both are masked, so the disagreement is not contamination — it is **methodological, and
the pair search is the one that is wrong here.**

`lane_pair` *constrains both boundaries to intersect at the assumed horizon* and reports that
intersection as `u_vp`. But §85 measured, with two UNCONSTRAINED lines, that the boundaries actually
meet at **467.7 px** — not at the assumed 448.4. Forcing the intersection 19 px away from where the
data puts it tilts both fitted lines and drags `u_vp` sideways; `∂yaw/∂v_h ≈ 0.039 °/px` × 19 px ≈
**0.75°**, which is the size of the disagreement.

⇒ The single-line reading is the trustworthy one for "the column at row 448.4": it fits a line to
the paint and evaluates it, with nothing imposed. **The pair's `u_vp` is not a yaw measurement
unless the horizon it is given is the row where the lines really meet.**

⚠️ This does NOT rescue the horizon. The boundaries meeting at 467.7 while the lens admits only
≲450 is the standing tension of `R-2026-09-13-horizon`, now with clean data on both sides: a
vanishing point is where lane LINES converge, and on a crowned road that is not where the ground
plane vanishes.

## 89. What is left of the symptom

With the angle confirmed, the corridor-to-left-line distance is **flat at 1.12–1.19 m from 10 to
30 m**. For a 1.855 m ribbon centred in a 3.5 m lane it should be **0.82 m**. The corridor is
therefore ~0.33 m to one side — **a LEVEL, not an angle** — and that is the `--lateral-offset`
question, which no image measurement on this recording has been able to settle (§77) and which a
tape measure from the vehicle centreline to the phone lens settles in thirty seconds.

---

# Part 18 — the lateral placement, and why three detectors in a row measured the wrong thing

## 90. ⛔ The right-hand "lane boundary" was a TAR SEAM

`does_it_cut_paint.py` is the probe that answers Sayed's actual complaint. It was rebuilt this
turn with the road mask, and it ran cleanly: 183/285 frames with a valid lane pair, a placement
defect of **−0.32 m** (−0.38 m by the h-free route), `clear R` going **negative** at 20–30 m, and
a boundary-convergence row of **466.1 ± 9.4**. Every one of those numbers is **VOID.**

The table diagnosed itself and I nearly missed it:

| range | clear L | clear R | lane w | ribbon w |
|---|---|---|---|---|
| 10 m | 1.19 | +0.30 | 3.33 | 1.87 |
| 20 m | 1.18 | −0.02 | 3.07 | 1.90 |
| 30 m | **1.19** | **−0.28** | **2.83** | **1.85** |

`clear L` is flat to **0.02 m** while `clear R` drifts **0.58 m**, and the "lane" shrinks 0.50 m
while the *drawn* ribbon reads flat at 1.87. A range-dependent error that appears on exactly one
side is not a calibration error — **it is one of the two lines not being a line.**

So I drew the fit on the image (`--preview`, new). The right-hand fit runs up a **tar seam in the
middle of the ego lane**; the real right boundary — **dashed** — is well beyond it.

**THE MECHANISM, and it is not bad luck.** RANSAC scores a line by **inlier count**. The left
boundary here is SOLID, the right is DASHED, and a continuous seam offers more inliers over the
same rows than a dashed line does. The detector was not confused; **it answered the question it was
asked.** Mask out the bank (`R-2026-09-14-foliage`) and the detector moves to the next-brightest
continuous non-paint structure available. Same class, one layer down.

⚠️ **A selection step cannot report that it selected badly.** RANSAC by inlier count, a Hough peak,
a seeded association — all three have now returned something that is not paint, and each returned
it with a confident support count. That is the argument for the histogram in §92.

## 91. ⛔ And the detector's threshold was set by pixels it then throws away

Found by reading the call, not the output: the masked probes call `ridge_cols(row, w)` **with no
window**, and that sets `thr = max(28, percentile(c, 99))` over the **WHOLE ROW** — on this
recording the sunlit bank and the concrete barrier, both **outside** the mask. The mask is applied
*after* thresholding, so structures we discard were setting the bar the paint had to clear, and
**masking made detection worse rather than better.**

MEASURED: with the mask on and no window, **three of four preview frames found no lane pair at
all**, while the left boundary is plainly bright paint in every one. `ridge_cols` has taken
`lo`/`hi` for exactly this purpose since it was written; nothing was passing them. Fixed in
`_cols_factory`, together with a **paint-brightness gate** (a detection must clear the row's own
masked median by `k` robust sd — the threshold then follows sun, shadow and exposure).

⚠️ Same family as `df` on a pod, `free`/`tegrastats` on Thor and cgroup `usage_in_bytes`:
**a statistic aggregated over the wrong scope, read as an answer.** Fourth costume.

## 92. The instrument that cannot pick a wrong winner: histogram, don't fit

New: `--hist` converts every detection to a vehicle-frame lateral (exactly — the ground projection
is affine in `y` at fixed range, so two probe points per row invert it) and **counts**. No line
fit, no pair constraint, no parallelism gate. The lane boundaries, the seam and the gravel all
appear as separate peaks **labelled in metres**, and a non-paint feature is *visible* instead of
*chosen*.

**MEASURED** (200 frames, `f 1533 · h 1.586 · v_h 448.4`, near band 9–13 m where lateral
resolution is 0.0065 m/px):

* **LEFT boundary: a clean, isolated peak at +1.62 m** (centroid over +1.25…+2.05). It is genuinely
  isolated — the bins from +0.3 to +0.9 carry 1–30 counts against a mode of 2725. Per-frame
  re-referencing sharpens it to **±0.10 m**, so it is a real narrow feature, not a smear.
* **RIGHT: no isolated peak exists.** A continuous ramp from −0.9 m into a mode at **−2.15 m**
  (5848) — the **gravel apron at the foot of the barrier** — with a plateau step around −1.35 m
  that is the dashed line riding the gravel's flank.

⚠️ **My de-smearing assumption was wrong and I am recording it as wrong.** I expected per-frame
re-referencing to the left line to blur the shoulder (its distance from the paint should vary) and
sharpen the lane. It did not: on this road the shoulder is **constant-width**, so it moves rigidly
with the paint. The re-referenced peak at **+3.99 m** is exactly `1.747 + 2.25` — the gravel again.
The technique is right; the road defeated it.

## 93. ⚠️ SUPERSEDED BY §97 — I attributed the lane measurement to the wrong parameter

*(Kept as written, with its error marked, because the error is the instructive part.)*

Taking +1.62 m and the −1.35 m step: the ego lane is **2.97 m** at ~~`f·h = 1533 × 1.586 = 2431`~~.

⛔ **THE SCALING IS `h/(v − v_h)`, NOT `f·h`.** With `u = cx − f(y−lat)/x` and `x = f·h/(v−v_h)`,
the focal **cancels**: `y = −(u − u₀)·h/(v − v_h)`. So the histogram's laterals are set by the
**height and the horizon**, and are entirely **free of `f`**. Everything below that reasons about
`f·h` is therefore mis-attributed; the arithmetic in row 2 is also simply wrong (`f = 2431/1.45 =
1676`, not 1976). **§97 redoes it correctly, and the corrected version is a much stronger result.**

| if… | then |
|---|---|
| the lane really is 3.5 m | `h = 3.5/1.873 = 1.87 m` — **impossible** on a Caddy windscreen ✓ *(this row survives: it never used `f`)* |
| ~~the lane really is 3.5 m and `h = 1.45 m`~~ | ~~`f = 1976`, a 1.29× EIS crop~~ — **wrong arithmetic and wrong parameter** |
| the adopted `h = 1.586` is right | the lane is **2.97 m** — narrow, but this section has a concrete barrier hard against the shoulder and yellow markings in places, i.e. a **roadworks/contraflow layout**, where 2.75–3.0 m is normal |

⛔ **`h = 1.668 m` (§86) is not admissible**, and neither is the `crop ≈ 1.0 ⇒ EIS is not cropping`
conclusion that rested on it. Both came from a pair search whose pair has not been shown to be the
lane.

⚠️ What does NOT change: the **angle**. `overlay_far`'s yaw is read from the corridor against the
LEFT line, which §92 confirms is real, isolated and narrow — so §87's "rendered −5.35 vs optimum
−5.10" stands, and so does the flat 1.12–1.19 m profile. **The delivered render's direction is
right; its lateral SCALE is in doubt, and its lateral OFFSET is still unmeasured.**

## 94. The horizon: a null result, with its resolution stated

A real lane boundary sits at the same lateral at 10 m and at 30 m — that is what "road-parallel"
means — and the recovered lateral depends on the range, hence on `v_h`. So the `v_h` that removes
the drift is the horizon, measured from the strong solid left line alone, with **no pair, no
vanishing point and no lens argument**. Scanned 370→470 px in 2 px steps, on 160 frames:

| v_h | y(9–13 m) | y(13–20 m) | y(20–30 m) | drift |
|---|---|---|---|---|
| 418 | 1.379 | 1.454 | 1.410 | **0.076** |
| 424 | 1.434 | 1.439 | 1.516 | 0.082 |
| 446 | 1.614 | 1.695 | 1.643 | 0.081 |
| 448 | 1.647 | 1.748 | 1.671 | 0.101 |
| 450 | 1.661 | 1.759 | 1.723 | 0.098 |

⛔ **This does NOT measure the horizon.** The basin is broad and noisy — 0.076 at 418 against 0.081
at 446 is not a discrimination — and a first pass that scanned only ±30 px returned **418.4, its
own lower bound**, which is a boundary solution and not a measurement. **Quotable conclusion: the
estimator constrains `v_h` to roughly 418–450 and does not resolve the standing
438 / ≲450 / 466 disagreement. The rendered 448.4 sits inside its indifference band and is not
refuted by it.**

*(The near band `y(9–13 m)` rises smoothly and monotonically 1.202 → 1.848 across the scan. That is
the signal; the far band is what is noisy. A version of this estimator with a longer range lever
and a proper CI is the obvious next instrument, and it would settle `v_h` without the lens.)*

## 95. Deliverable manifest

| artifact | where |
|---|---|
| `does_it_cut_paint.py` — rebuilt: road mask, sign fix, horizon-free gate, `fit_pair` shared by measurement and preview, `--preview` with a lateral ruler, `--hist`, `--scan`, paint-brightness gate, in-mask thresholding | repo `incoming/2026-09-13-bev-semantic-calib/probes/` |
| fit previews (fit on image, three mask/threshold variants) | session scratchpad — **regenerate with `--preview`, they are reproducible** |
| histogram JSON (`hist_wide`, `hist_rel`, `hist_horizon2`, `hist_near`) | session scratchpad |
| §90–§94 | this document |
| `R-2026-09-15-seam`, `R-2026-09-15-scope` | `Project Steering/RETRACTION_LOG.md` |

## 96. What is actually blocking, and what is not

1. ⛔ **`f·h` (the lateral scale).** Blocking every lateral number. Settled by ONE of: the true lane
   width on this road; a tape from the ground to the phone lens; or a metre-stick in frame.
2. **`--lateral-offset`.** Still unmeasured (§89), still a tape measure from the vehicle centreline
   to the lens. **Independent of 1** — it is a level, not a scale.
3. **NOT blocking:** the yaw (§87, confirmed), the fade, the vehicle geometry, the render pipeline.

---

# Part 19 — the 3.5 m lane was the assumption that was wrong

## 97. ⛔⛔ RETRACTED BY §100 — the ratio it used was 24 % low (`R-2026-09-15-longitudinal`)

*(Kept in full. The METHOD below is right and is reused in §100; the INPUT `lane/h = 1.873`
was corrupted by a coordinate-frame bug, and with the corrected `2.333` the headline
conclusion **inverts**: a 3.5 m lane is not refuted, it is comfortably admissible.)*

## 97. ⭐⭐ THE CIRCULARITY, AND HOW IT BREAKS

Every `h` in this programme descends from one sentence in §16: *"height ~1.65–1.86 m — lane width
at horizon 465, **assuming a 3.50 m lane**"*. From there `f = f·h / h` closed the loop. So the
chain has been:

```
  row flow (odometer, NO paint)  ->  f·h
  paint + ASSUMED 3.5 m lane     ->  h
  f = f·h / h                    ->  f          ->  checked against the lens bound
```

The middle line is the one §90 retracts: the separations being divided into 3.5 m were a bank, a
seam and a gravel apron, never the lane. **But the loop can be run the other way, and then the lane
width stops being an input and becomes the thing under test.**

**THE THREE INPUTS, each independent of the others:**

| | value | provenance |
|---|---|---|
| **lane / h** = \|Δm\| | **1.873** | MEASURED — near-band histogram, §92. **Free of `f` and free of `f·h`**: the focal cancels out of `y = −(u−u₀)·h/(v−v_h)`. |
| **`f·h`** at horizon 448 | **2431 px·m** | MEASURED — row flow against the odometer, §84. **No paint at all.** |
| **`f` ≥ 1442 px** | lens bound | PUBLISHED — 26 mm-equiv at 1920 px wide, and **EIS can only crop IN**, never out. |

Combining them, with the lane width `W` as the free variable:

| W (lane) | h = W/1.873 | f = 2431/h | crop = f/1442 |
|---|---|---|---|
| 2.60 m | 1.388 | 1751 | 1.21× |
| 2.75 m | 1.469 | 1655 | 1.15× |
| **3.00 m** | **1.602** | **1517** | **1.05×** |
| 3.25 m | 1.736 | 1401 | **0.97× ⛔** |
| 3.50 m | 1.869 | 1301 | **0.90× ⛔** |

⛔ ~~**THE LENS BOUND CAPS THE LANE AT W ≤ 3.16 m — so the 3.5 m assumption is REFUTED by the
optics.**~~ **WRONG — see §100. The true cap is W ≤ 3.93 m.** A 3.5 m lane here would require the recorded image to be *wider than the lens*, which is
the same impossibility that excluded the paint-only horizon in §51. The assumption that has been
propagating since Part 3 is the thing that was wrong.

⭐ **And the adopted calibration survives its own justification being retracted.** `h = 1.586,
f = 1533` corresponds to `W = 2.97 m`, comfortably inside the admissible band — and a 2.97 m lane
is exactly what the frames show: a concrete barrier hard against the shoulder, yellow markings in
places, a contraflow/roadworks cross-section. **The numbers were right; the reason given for them
was not.** That is a weaker claim than "confirmed" and it is the one the evidence supports:
**not excluded, and consistent.**

⚠️ **The soft input is the RIGHT boundary at −1.35 m**, read as a step on the gravel's flank rather
than a peak (§92). At ±0.15 m on it the ratio runs 1.78–1.97 and the cap runs `W ≤ 3.0–3.3 m`. The
conclusion "not 3.5 m" is robust across that whole range; the exact width is not.

## 98. ⛔ The dash-burstiness discriminator failed — and it failed by measuring my own sampling

The right-hand boundary is **dashed** and the gravel apron is **continuous**, which no amount of
masking or brightness gating can see. So: watch one lateral bin across frames — a dashed line
should be **bursty** (a dash enters the range window, then a gap), a shoulder **steady**. Occupancy
and the Fano factor should separate them.

**MEASURED, 240 frames, near band: they do not.** Fano is **15–75 in every bin** and occupancy
2–79 % in every bin — *including the left SOLID line* (+1.65 m: occupancy **51 %**, Fano **25.6**).

**ROOT CAUSE: the frames are sampled ~9 s apart**, spread across the whole recording to get
coverage. Between two samples the road section, the sun, the exposure, the curvature and the car's
lateral position have all changed, and that variance swamps any dash pattern by an order of
magnitude. **The statistic measured my sampling scheme, not the road.** Burstiness is only
meaningful on *consecutive* frames, where the dash period is the only thing changing.

⚠️ Logged rather than fixed: the result in §97 does not depend on it, and a consecutive-frame
version is a cheap future instrument. **Same family as everything else in Parts 18–19 — a statistic
computed over the wrong scope** — which is now the fifth costume, after `df`, Thor's `free`, cgroup
`usage_in_bytes` and the `ridge_cols` threshold of §91.

## 99. What this changes, and what it does not

**CHANGES:**
- ⛔ *"The lane is 3.5 m"* is **retired as an input** everywhere in this programme. It is now a
  **derived quantity, bounded above at ~3.16 m by the optics.**
- `f·h` is **not** the least-determined quantity (that was §93's error). It is measured by the row
  flow with no paint. The least-determined quantity is the **right lane boundary**, and through it
  the split of `f·h` into `f` and `h`.

**DOES NOT CHANGE:**
- The adopted set `horizon 448.4 · h 1.586 · f 1533 · yaw −5.35`, which is inside every admissible
  band above. The delivered renders stand.
- The yaw (§87), the fade, the vehicle geometry, the render pipeline.
- The `--lateral-offset` question (§89, §96 item 2) — still a **level**, still unmeasured, still a
  tape measure. Nothing in Part 19 touches it.

---

# Part 20 — a coordinate-frame bug that manufactured a horizon drift and a refutation

## 100. ⛔⛔ `project_ground`'s `x` is measured from the REAR AXLE, not the camera

`RR.NOMINAL` carries **`longitudinal: 2.1`**, and `CameraModel.t_v = [longitudinal_m,
lateral_m, height_m]` is *"the camera position expressed in the vehicle frame"*. So
`project_ground([[10, y]])` is a point 10 m ahead of the **trajectory origin** — which is
**7.80 m ahead of the camera**.

Every probe here that inverted a row to a range did it with `x = f·h/(v − v_h)`, which is the
range **from the camera**, and then handed that number to `project_ground` as a range **from the
rear axle**. The lateral scale used was therefore `f/(x − 2.1)` where `f/x` was required:

| nominal x | camera range `project_ground` actually used | lateral error |
|---|---|---|
| 8 m | 5.82 m | **−27 %** |
| 10 m | 7.80 m | **−21 %** |
| 20 m | 17.69 m | −12 % |
| 30 m | 27.59 m | −7 % |

**HOW IT WAS CAUGHT.** Two probes disagreed about the *same* line: the histogram said the left
boundary was at **+1.62 m**, `where_is_the_corridor` said **+2.01 m**. A 0.39 m gap between two
readings of one feature is not noise, and the renderer self-consistency test had already left the
clue — `project_ground` at "10 m" produced a lateral scale matching a **7.89 m** camera range, a
number with no business appearing unless something was subtracting ≈2.1 m.

With the `+LON` fix the two agree: histogram **+2.05 m**, `where_is_the_corridor` **+2.01 m**.

**⭐ AND IT EXPLAINS THE HORIZON SCAN.** §94's estimator measures a lane boundary's lateral in
three range bands and looks for the `v_h` that removes the drift. But the bug's error is
**range-dependent** — −21 % at 10 m, −7 % at 30 m — so it injects a spurious drift of ~0.14× the
lateral, about **0.28 m**, which is the same order as the drifts being scanned. **§94 was measuring
this bug, not the horizon.** That is why its basin was broad and noisy and why a first pass ran to
its own boundary. A re-run with the fix is the obvious next instrument and is now unblocked.

⚠️ **Not everything is affected.** `clear_L` and the yaw scan use `mpp = x/f` with `x` from the
camera and never touch `project_ground`, so §87 and the flat 1.12–1.19 m profile stand. The
renderer is unaffected: it works in vehicle coordinates throughout, which is why its
self-consistency check passes at **±0.04 m**.

## 101. ⭐ Redone: the 3.5 m lane is ADMISSIBLE, and the cap is 3.93 m

Corrected near-band peaks (200 frames, `h 1.586 · v_h 448.4`, `+LON` fixed):

* **left boundary +2.05 m** (mode 2.00–2.10, n 2303) — clean and isolated, as before;
* right-hand candidates **−1.65 m** (1272) and **−2.25 m** (2962), with the **gravel mode at
  −2.75 m** (4832). Taking −1.65 as the boundary gives `lane/h = 2.333`.

| W | h = W/2.333 | f = 2431/h | crop |
|---|---|---|---|
| 3.25 m | 1.393 | 1745 | 1.21× |
| **3.50 m** | **1.500** | **1620** | **1.12×** |
| **3.70 m** | **1.586** | **1533** | **1.06×** ← the adopted `h` |
| 3.93 m | 1.685 | 1443 | 1.00× |
| 4.10 m | 1.757 | 1383 | **0.96× ⛔** |

⭐ **The lens caps the lane at W ≤ 3.93 m.** A 3.5 m lane is comfortably inside it, at
`h = 1.500 m, f = 1620 px, crop 1.12×` — a very ordinary windscreen-mount height and a very
ordinary EIS crop. The adopted `h = 1.586` corresponds to a **3.70 m** lane, which is also an
entirely standard width. **Both are admissible and the optics no longer discriminate between
them.** §97's refutation is withdrawn.

⚠️ The open ambiguity is now the **right boundary: −1.65 m or −2.25 m** (lane 3.70 m or 4.30 m).
−2.25 m is excluded by the lens for any `h ≥ 1.30`, so **−1.65 m is the boundary and −2.25 m is
shoulder** — the optics do still discriminate *there*. Worth confirming with a consecutive-frame
dash test (§98's instrument, fixed).

## 102. What Sayed actually sees, and whether it is a defect

Measured on the delivered render, ~~straight frames only (|yaw rate| < 1 °/s,~~ n 106–167 per range):

⛔ **THE STRAIGHT-FRAME FILTER NEVER RAN** — see §110. These records carry no `yaw_rate_dps`, so
`r.get("yaw_rate_dps", 0.0)` returned the default for every frame and the filter passed 2156 of
2216. The table is over ALL frames. Its numbers stand; the label on them did not.

| range | corridor centre (should be 0.00) | left line | gap, ribbon edge → line |
|---|---|---|---|
| 10 m | **+0.004 m** | +2.01 | +1.09 m |
| 20 m | **−0.002 m** | +2.04 | +1.12 m |
| 30 m | **+0.008 m** | +1.95 | +1.02 m |

1. **The renderer is not misplacing the corridor.** It draws the ribbon within **0.01 m** of where
   `project_ground` puts it, at every range. There is no drawing bug.
2. **There is over a metre of clearance to the left line**, flat with range. The corridor is not
   cutting the left boundary in straight driving.
3. The vehicle centreline sits at `(2.05 − 1.65)/2 = +0.20 m` from lane centre — the car runs
   slightly right of centre, which is ordinary on a two-way road.

⇒ **The remaining candidates for the symptom are (a) CURVES, where the ribbon correctly follows the
future path and legitimately crosses a line, and (b) the per-frame spread** — the left line's robust
sd is **0.23 m at 10 m rising to 0.54 m at 30 m**, so individual frames sit far closer than the
median. A median cannot show either. **The next measurement is the per-frame minimum clearance
distribution and its correlation with yaw rate**, which separates "correct prediction of a turn"
from "wrong geometry" — and it needs no new calibration.

## 103. ⭐ The mount offset, from the car's own bodywork — no tape measure

The bonnet is symmetric about the vehicle centreline, so for a symmetric pair at depth `D` the
midpoint column is `cx + f·tanψ + f·lat/D`; against the road vanishing point,
`m − u_vp = f·lat/D`. At `D ≈ 1.8 m` and `f = 1533`, a 0.126 m mount offset is **107 px** — not a
subtle signal.

MEASURED by mirror correlation on the illumination-flattened vertical-gradient map of a 120-frame
median (`u_vp = cx + f·tan(−5.35°) = 816.4`):

| source rows | symmetry axis | corr | `m − u_vp` | lat at D = 1.3…2.1 m |
|---|---|---|---|---|
| 830–900 | 950 | **0.149** | +134 | +0.11…+0.18 ⚠ weak |
| 900–960 | 746 | 0.243 | −70 | −0.06…−0.10 |
| 930–1000 | **520 ⛔** | 0.378 | — | **boundary solution, discarded** |
| 960–1040 | 739 | **0.684** | −77 | −0.07…−0.11 |
| 830–1040 | 740 | **0.677** | −76 | −0.07…−0.11 |

⇒ **`lat ≈ −0.05 to −0.11 m`** from the two well-correlated bands — same sign as the rendered
**−0.126 m** and the same order. ⚠️ Not precise enough to *correct* it (the depth `D` is known only
to ~±20 %, and the bands disagree), but decisive on the question that mattered: **the mount offset
is small, so it cannot account for a placement error of several tenths of a metre.** The
tape-measure blocker of §96 is therefore **not blocking** — it was never the explanation.

⚠️ One band ran to the edge of its own scan window and was discarded. That is the third boundary
solution in two days (§94's first pass, the `--scan` default, this). **A scan whose optimum sits at
an end of its range is not a measurement** — it is now checked for explicitly wherever a scan is run.

---

# Part 21 — the horizon estimator works now, and it moves the disagreement

## 104. ⭐ With `+LON` fixed, the lateral-drift scan has a real minimum

§94 reported this estimator as a null: a broad noisy basin, a first pass that ran to its own
boundary, no discrimination. §100 explains why — the `+LON` bug injects a **range-dependent**
lateral error (−21 % at 10 m, −7 % at 30 m), which is precisely the signature the estimator
attributes to `v_h`. **It was measuring the bug.**

Re-run with the fix, 200 frames, scan 400→500 px in 2 px steps:

| v_h | y(9–13 m) | y(13–20 m) | y(20–30 m) | drift |
|---|---|---|---|---|
| 400 | 1.605 | 1.289 | 1.236 | 0.369 |
| 438 | 1.939 | 1.893 | 1.660 | 0.278 |
| 448 | 2.023 | 1.997 | 1.834 | 0.190 |
| 464 | 2.184 | 2.181 | 2.222 | 0.041 |
| **476** | 2.328 | 2.349 | 2.362 | **0.034** |
| 482 | 2.403 | 2.431 | 2.449 | 0.046 |
| 498 | 2.570 | 2.819 | 3.010 | 0.440 |

⭐ **A clean V with an interior optimum: drift falls monotonically from 0.37, bottoms in a basin
at 460–484, and rises again to 0.44.** Nothing like §94's flat noise. **Horizon = 460–484 px,
best ≈ 476.**

## 105. ⚠️ Two paint routes now agree — but they share a confound

| route | horizon | uses |
|---|---|---|
| lateral drift (§104) | **460–484** | paint, flat road |
| lane-boundary convergence (§85) | **466–468** | paint, flat road |
| row flow vs odometer (§16) | **437.4** [431.4, 443.4] | ego motion, **no paint**, flat road |
| lens bound | **≲450** | optics + an assumed `h` |

The two paint routes agreeing is **weaker evidence than it looks**: both assume the road is a
plane. On a downgrade, road-parallel lines converge **below** the true horizon and the recovered
lateral also drifts — so a grade pushes **both** paint routes the same way, and they would agree
while both being wrong. **They agree because they share a bias, not necessarily because they are
right.** The 30 px gap to the row flow corresponds to `atan(30/1533)` = **1.1° of average
downgrade**, which is unremarkable on a road that visibly runs through a rock cutting.

⇒ **The discriminating experiment is a grade-aware estimator, or the same estimators restricted to
a verifiably level stretch.** Pre-registered, both outcomes committed: if the paint routes move
toward 438 on level ground, grade is the explanation and the row flow wins; if they stay at ~470,
the row flow has a bias of its own and the flat-road assumption is not the culprit.

## 106. What it does to `h` and `f` — the lens bound now bites the OTHER way

Near-band peaks re-measured at each candidate horizon (160 frames, h 1.586), with the row-flow
`f·h` at the same pinned horizon from §84:

| horizon | `f·h` | left | right | lane/h | `f` if W = 3.5 m | crop |
|---|---|---|---|---|---|---|
| **448** | 2431 | +2.05 | −1.65 | 2.333 | **1620** | 1.12× |
| 464 | 2094 | +2.15 | −1.65 | 2.396 | 1433 | 0.99× ⚠ |
| 476 | ~1857 | +2.35 | −1.85 | 2.648 | 1405 | 0.97× ⚠ |

⚠️ **At the horizon the drift scan prefers, a 3.5 m lane needs `f` ≈ 1405–1433 — just below the
1442 px uncropped bound.** Marginal, not decisive: "26 mm equivalent" is a rounded spec, and
25–27 mm spans `f` = 1386–1497. So the lens no longer excludes the high horizons cleanly, and it
no longer picks a winner. **What it does say is that at horizon ≈ 470 the capture is essentially
uncropped, while at 448 it is cropped ~1.12×.**

## 107. ⛔ And the right-hand boundary is still NOT established

At every horizon tried, the right-side peaks come out **evenly spaced by exactly 0.60 m**
(−1.85 / −2.45 / −3.05 at v_h = 476; −1.65 / −2.25 / −2.85 at 464). That regularity is not road
structure — it is **the peak-picker's own 0.5 m minimum-separation rule slicing a broad
continuous mass**, which is what §92 already found on the right side and what §98's burstiness
test failed to resolve.

⇒ **Any "lane width" in the table above is provisional on the right boundary**, and the numbers
are published with that attached rather than quietly averaged. The instrument that can settle it
is the consecutive-frame dash test (§98's idea, with its sampling defect fixed) — the right line
is dashed and the shoulder is not, and at a 1-frame stride the dash period is the only thing
changing.

---

# Part 22 — the symptom, found: it is real, it is beyond 30 m, and it is the prediction

## 108. ⭐⭐ The corridor DOES cross the left line — in 12.9 % of frames, all beyond 30 m

Sayed: *"i see the trajectory cutting road markings"*. Every previous answer quoted a **median**,
and a median with a metre of clearance is exactly the statistic that cannot show a visible
minority. **A complaint about what is visible is a complaint about the TAIL.**

Per-frame **minimum** clearance from the ribbon's left edge to the left painted line, 210 frames,
all speeds, **all** steering angles, on the delivered fade render:

| range | n | median | p5 |
|---|---|---|---|
| 10 m | 404 | +1.08 | **+0.62** |
| 20 m | 391 | +1.10 | **+0.68** |
| 30 m | 232 | +1.03 | **+0.42** |
| 40 m | 170 | +1.09 | **−0.18** |
| 50 m | 111 | +1.24 | **−0.87** |

Frame-worst percentiles: p0 **−1.06 m**, p5 −0.29, p10 −0.11, p50 +0.94, p100 +2.13.
**27 of 210 frames (12.9 %) have the ribbon on or past the line.**

⭐ **Restricted to ≤30 m, it never happens: 0 of 228 frames below +0.25 m clearance, worst case
+0.28 m.** The symptom is entirely a far-field phenomenon, and the boundary is sharp — p5 flips
sign between 30 m (+0.42) and 40 m (−0.18).

## 109. ⭐ And it is the PREDICTION, not the geometry

The ribbon is drawn about the **future path**, so the discriminating variable is the path's own
predicted lateral — which the trajectory records carry directly (`x`, `y` arrays), needing no image
at all. At 50 m the predicted lateral has median −0.10 m and a **5–95 % span of [−1.49, +2.14] m**:
over a 2.4 s look-ahead the car genuinely goes somewhere else.

| statistic | at 30 m | at 50 m |
|---|---|---|
| `r(path lateral, worst clearance)` | −0.360 | **−0.789** |

⭐ **r = −0.789.** The far-field excursion is dominated by the predicted path, so **the ribbon is
crossing the line because the car is about to.** Crossings also rise with steering — the fraction
below zero goes 8.6 % → 10.2 % → 10.6 % → **24.0 %** across |steering| bands 0–1°, 1–2°, 2–4°,
4–7°.

⇒ **This is not a calibration defect and must not be "fixed" geometrically.** Forcing the ribbon
to stay inside the current lane would make the overlay *wrong* — it would stop showing where the
car is going, which is the entire point of it. The geometry is corroborated independently: the
renderer draws the ribbon within **0.01 m** of `project_ground` (§102), and inside 30 m the
clearance never drops below 0.28 m over the whole recording.

**WHAT IS ACTUALLY WRONG IS THE PRESENTATION.** At 50 m a 2 m lateral swing is drawn as crisply as
a 10 m one, while its uncertainty is far larger and is not shown at all. Three candidate fixes, in
increasing order of honesty:

1. bring the fade in — it currently runs 30→55 m, and **every crossing is beyond 30 m**, so a
   25→40 m fade would put them all in the faint region;
2. **widen the ribbon with range**, so the drawn band carries the prediction's growing lateral
   uncertainty instead of implying a precision it does not have;
3. draw the far field as a centre-line with a confidence envelope rather than a hard-edged corridor.

(2) is the one that is *true* rather than merely tidy, and it is a `viz.py` change, not a
calibration change.

## 110. ⛔ `yaw_rate_dps` does not exist, so the "straight frames" filter never ran

`per_frame`'s first run put **228 of 228 frames into a single yaw-rate band** — a result with only
one explanation. The records carry `frame · pts_s · speed_ms · steer_wheel_deg · steer_valid ·
standstill · t · x · y · yaw · v · pos_sigma`. There is **no `yaw_rate_dps`**, so
`r.get("yaw_rate_dps", 0.0)` returned the default everywhere.

⚠️ **It had already shown itself and I read past it.** `main()` reported *"2156 straight frames
(|yaw rate| < 1.0 deg/s)"* out of 2216 — 97 % of a recording on a bending road classified as
straight. **A filter that rejects almost nothing is a filter that is not running**, and §102's
"straight frames only" label is corrected above on that basis. The numbers in it stand (they are
over all frames); the claim attached to them did not.

⇒ The replacement is better than the original intent: `steer_wheel_deg` is real, and the path's own
`y` interpolated at each range is not a *proxy* for the cause but **the cause itself**.

⚠️ Same family as `R-2026-09-15-longitudinal`: **a default silently substituted for a fact.**
`.get(key, default)` on a schema you have not enumerated is the dictionary version of reading a
statistic over the wrong scope.

---

# Part 23 — the far field is TRUE, and §109's proposed fix was for a quantity that does not exist

## 111. ⛔ CORRECTING MY OWN RECOMMENDATION: there is no prediction

§109 proposed widening the ribbon with range *"to carry the prediction's growing lateral
uncertainty"*. **There is no prediction.** The records carry `t: [-3.0 … +5.0]` with `x` running
**−64 m to +114 m** — the **ACTUAL reconstructed trajectory**, past and future. The overlay replays
where the car really went.

⇒ Widening by a prediction uncertainty would have drawn **a quantity that does not exist**, and it
would have been worse than drawing nothing, because it would have looked like rigour. The
recommendation is withdrawn before implementation.

⚠️ The right question is therefore not *"how uncertain is the far field"* but ***"is it TRUE"*** —
and that has a closed-loop test needing no new calibration, no right-hand boundary and no new
assumption.

## 112. ⭐⭐ The same road, seen far and then near — the gap is physical, so both views must agree

A fixed piece of road sits at 50 m now and at 10 m **1.76 s later** (58 frames at 29.94 fps). The
corridor-to-line clearance there is a **physical gap**, so the two views must report the same
number. They share almost nothing: different image rows, different ranges, different parts of the
fade, ribbons drawn from trajectory windows 58 frames apart, and paint measured from different
source frames. The lag is computed **per frame from that frame's own `(t, x)` path**, because the
car varies 19.7–23.3 m/s and a fixed offset would smear the pairing by metres.

| far range | lag (frames) | n pairs | far clear | near clear | **far − near** | per-pair r |
|---|---|---|---|---|---|---|
| 20 m | 14 | 261 | +1.07 | +1.05 | **−0.01 m** | +0.71 |
| 30 m | 29 | 149 | +0.98 | +1.13 | **−0.09 m** | +0.56 |
| 40 m | 43 | 112 | +1.07 | +1.12 | **−0.07 m** | +0.51 |
| 50 m | 58 | 71 | +1.30 | +1.11 | **+0.03 m** | −0.01 |

⭐ **The bias is within 0.09 m at every range out to 50 m.** The far-field corridor is telling the
truth, so **§108's crossings are the car genuinely going there** and the overlay is correct.

⇒ **No `viz.py` change is warranted on geometric grounds.** Combined with §102 (the renderer draws
within 0.01 m of `project_ground`) and §108 (inside 30 m the clearance never drops below +0.28 m
over the whole recording), **the corridor's geometry is now verified end to end**: renderer
self-consistency, near-field absolute placement, and far-field truthfulness against an independent
view of the same road.

## 113. ⚠️ What the decaying correlation does and does NOT say

The per-pair `r` falls +0.71 → +0.56 → +0.51 → **−0.01**. Two readings, and they are not equally
supported:

* **the far-field RENDER stops corresponding frame by frame** — this would indict the overlay;
* **my own 50 m MEASUREMENT is noise-dominated** — n drops to 71, the paint is 7 px wide at 50 m,
  and the corridor is deep in the fade.

⭐ **The second is the supported one, and the table itself discriminates.** The near (10 m)
measurement is common to every row, and it correlates at **+0.71** with the 20 m view — so the near
measurement is demonstrably good. If both members of the 50 m pair were sound, `r` could not
collapse to zero while the *median* bias stays at +0.03 m. **A noise-dominated estimator loses
correlation while keeping its centre**, which is exactly the pattern observed.

⇒ **`r → 0` at 50 m is a limit of this instrument, not evidence against the render**, and it is
recorded as such rather than quoted as a finding. Distinguishing them properly would need a second
independent far-field measurement, which is not worth building for a question already answered by
the bias.

## 114. Where the calibration now stands

| question | status |
|---|---|
| does the corridor cut markings? | **YES, 12.9 % of frames, all beyond 30 m** (§108) |
| is that a defect? | **NO** — bias ≤ 0.09 m out to 50 m (§112); the car went there |
| renderer placement | **verified**, within 0.01 m of `project_ground` (§102) |
| near field (≤ 30 m) | **verified**, clearance never below +0.28 m over the recording (§108) |
| mount offset `lateral` | **≈ −0.05…−0.11 m** from the bodywork (§103), small; no tape needed |
| yaw | **verified**, rendered −5.35° vs optimum −5.10° (§87) |
| horizon | **460–484** by lateral drift (§104), vs row flow 438 — 1.1° of grade would explain it |
| `h` / `f` split | **OPEN** — 1.500/1620 or 1.586/1533, both admissible (§101) |
| right lane boundary | **OPEN** — no isolated peak; the 0.60 m spacing is the peak-picker's own rule (§107) |

**The two open items are the same item.** The `h`/`f` split is set by `lane/h`, and `lane/h` is set
by the right boundary. Neither changes the delivered render: `f·h` is fixed by the row flow, the
lateral scale at a given row depends on `h` alone, and the difference between the two candidates is
**5.7 % of ribbon width** — 0.05 m on a 1.855 m ribbon at the near field, far below the 0.28 m
clearance margin.

---

# Part 24 — the right boundary, settled by periodicity; and the closing state

## 115. ⛔ The first dash test failed its own control — and that is why it had one

`R-2026-09-15-burstiness` failed because the frames were **9 s apart**. The fix is stride 1, and at
29.94 fps there is a better discriminator than burstiness: **a dashed line is PERIODIC**, which is
a structural claim rather than a variance claim and survives a detector that misses half the
dashes.

**The first run flagged EVERY bin as dashed — including the SOLID left line — all at lag 6, which
was `--lag-lo`, the scan's own lower bound.** Two red flags at once, and the control is what made
it undeniable: consecutive frames 0.2 s apart are nearly identical, so the **detector's own
temporal smoothness** dominates the autocorrelation at short lags and looks exactly like a period.

⇒ High-pass first (subtract a 9-frame moving average), and exclude lags below any physical dash
cycle. ⚠️ **Fourth boundary solution in this work** — §94's first pass, `--scan`'s default, the
bonnet band, and now this. **A scan whose optimum sits at an end of its range is not a measurement**,
and the check is now explicit everywhere a scan runs.

## 116. ⭐⭐ Settled: the right boundary is at −1.10 m, pitch 13 m

Detrended, three **independent** 700-frame stretches:

| stretch | flagged bins | lag | pitch |
|---|---|---|---|
| 744–1443 | −1.15, −1.05 | 19 | 13.8 m |
| 1500–2199 | −1.45 … −0.75, centred **−1.10** | 18–19 | 12.4–13.1 m |
| 100–799 | −1.15 | 18 | 12.4 m |

**Controls pass:** the solid left line (+2.05) peaks at acf **+0.12**, the gravel (−2.95, −2.55) at
**+0.13** — no periodicity in either. Stray left-side flags at +1.55 and +1.95 appear in one
stretch each with inconsistent lags (17, 35) and acf 0.27–0.35, at the threshold; the right-hand
detection appears in **all three** at lag 18–19. That asymmetry is the evidence, not the flag count.

⭐ **The pitch is CALIBRATION-FREE.** It is `speed × lag / fps` — GPS speed and frame rate only,
no `f`, no `h`, no horizon. A dash reappears in *any* fixed image region at the pitch divided by
the speed, wherever that region is. **12.4–13.8 m against the French T'1 standard of 3 m mark +
10 m gap = 13 m.** A fact about the road, established without the camera.

## 117. ⭐ And that closes `h`/`f` — 3.5 m is excluded, on a MEASURED boundary this time

`lane/h` = (2.05 − (−1.10))/1.586 = **1.986**. §101 used 2.333 on a right boundary §107 itself
recorded as unestablished; it is now measured.

| W | h = W/1.986 | f = 2431/h | crop |
|---|---|---|---|
| 3.00 m | 1.510 | 1609 | 1.12× |
| **3.15 m** | **1.586** | **1533** | **1.06×** ← the adopted set |
| 3.25 m | 1.636 | 1486 | 1.03× |
| 3.35 m | 1.687 | 1441 | 1.00× |
| 3.50 m | 1.762 | 1380 | **0.96× ⛔** |

⭐ **The lens caps the lane at W ≤ 3.35 m, so a 3.5 m lane is excluded** — the conclusion §97
reached, then lost to the `+LON` bug, and now regains on properly measured inputs. **The adopted
`h = 1.586 · f = 1533` implies a 3.15 m lane**, an ordinary width for a French route
départementale, and it sits mid-band rather than at an edge.

⚠️ Stated at its true strength: this rests on the left line at **+2.05** (clean, two independent
probes agreeing to 0.04 m) and the right at **−1.10** (three stretches, controls passing). The band
`W ∈ [2.9, 3.35]` is the honest interval; 3.15 m is its centre, not a point measurement.

## 118. CLOSING STATE — the 2026-08-08 calibration

| quantity | value | evidence |
|---|---|---|
| **horizon** | **448.4 px** (adopted) | row flow 437.4; lateral drift 460–484; ~1.1° of grade reconciles them (§105). **The one genuinely unresolved item.** |
| **`f·h`** | **2431 px·m** | row flow vs the odometer, no paint |
| **`h`** | **1.586 m** | `f·h` / `f`, with `lane/h` = 1.986 measured (§117) |
| **`f`** | **1533 px** | HFOV 64.1°, EIS crop 1.06× |
| **yaw** | **−5.35°** | optimum −5.10°, i.e. 0.17 m at 40 m (§87) |
| **`lateral`** | **−0.126 m** | bodywork symmetry gives −0.05…−0.11 m (§103) — small, and not the explanation for anything |
| **lane width** | **3.15 m** [2.9, 3.35] | left +2.05, right −1.10 |
| **marking** | French T'1, 13 m pitch | calibration-free (§116) |

**The corridor's geometry is verified end to end:** the renderer draws within **0.01 m** of
`project_ground` (§102); inside 30 m the clearance to the left line never drops below **+0.28 m**
across the whole recording (§108); and the far-then-near consistency check agrees to within
**0.09 m** out to 50 m (§112).

⭐ **Sayed's complaint is answered: the trajectory DOES cut road markings, in 12.9 % of frames, all
beyond 30 m — and it is correct to do so.** The overlay replays the actual reconstructed
trajectory, `r(path lateral, clearance) = −0.789` at 50 m, and the far view agrees with an
independent near view of the same road. **The car went there.** No `viz.py` or calibration change
is warranted.

**What would still be worth doing, none of it blocking:** the horizon's 438-vs-470 split needs a
grade-aware estimator or a verifiably level stretch (§105, pre-registered with both outcomes); and
`W` could be tightened from [2.9, 3.35] with a second marking-standard anchor.

---

# Part 25 — Sayed was right: the corridor IS misplaced, by 0.28 m to the right

## 119. ⛔⛔ §114's "geometry verified end to end" was wrong about LATERAL PLACEMENT

Sayed, on the re-render: *"no improvements, the trajectory still leaving the road, knowing that ego
[is] driving between the road markings."*

He is right, and the defect was in my own published numbers the whole time. §116/§117 measured the
left boundary at **+2.05 m** and the right at **−1.10 m** from the vehicle centreline. **If the car
drives centred between the markings those must be symmetric. They differ by 0.95 m.** I read that
asymmetry as *the car sitting right of centre* rather than as *the drawing being displaced*, and
never questioned it because the one input that discriminates — where the driver actually is — was
never in the data.

**MEASURED on the delivered render, 584 distinct frames, 2335 reads at 7–12 m:**

| | left | right |
|---|---|---|
| clearance, ribbon edge → line | **+1.09 m** (sd 0.24) | **+0.51 m** (sd 0.30) |

⭐ **The ribbon centre sits 0.28 m RIGHT of the lane centre**, frame-cluster bootstrap 95 % CI
**[0.26, 0.29] m**. On frame 522 — the frame Sayed showed — it is worse: right clearance **+0.12 m
at 7 m and +0.03 m at 9 m**, i.e. the drawn edge is *on* the line while the left side has 0.64–1.03 m.

## 120. ⛔ WHY EVERY STATISTIC I RAN HID IT

**The right boundary is DASHED.** Whenever a dash had a gap at the sampled range, "the nearest paint
to the right" was not the lane line — it was the **shoulder edge line at −2.5 m**. That single
substitution turned a 0.5 m clearance into a reported metre, and it did so in exactly the frames
where the ribbon was closest to the paint.

This is `R-2026-09-15-seam` for the fourth time: **a selection step, handed an incomplete feature,
silently returns the next thing out.** The fix is not a better peak-picker but a **declared gate** —
the ego lane's right boundary is within 2.2 m, anything beyond is shoulder — and with that gate the
lane width stabilises at **3.40 m (robust sd 0.25)** instead of the 2.5–4.4 m my earlier passes
produced.

⚠️ **And I compounded it.** Every probe in Parts 22–23 measured the **LEFT** line, because it is the
clean one. A one-sided instrument cannot see a placement error at all — it sees a clearance, and a
clearance is consistent with any placement once you allow the car to be off-centre. **Measuring the
easy side is not measuring.**

## 121. The fix, and what it implies physically

The measured lateral shifts **1:1** with the `lateral` parameter: `measured_Y = −(u−c_x)·h/(v−v_h) +
lat`, so the correction is the offset itself.

    --lateral-offset   −0.126 m   →   −0.41 m

The camera sits ~0.41 m **right** of the vehicle centreline — a phone mounted right of the mirror,
which for a 1.855 m vehicle is unremarkable. ⭐ **The pipeline's own default is −0.35 m.** The
override to −0.126 was the error, and the default was closer to the truth than the "measurement"
that replaced it.

⛔ **§103's bonnet-symmetry estimate (−0.05…−0.11 m) is REFUTED, not merely imprecise.** I described
it as "not precise enough to correct, but decisive on what mattered: the mount offset is small".
**It was not small, and that sentence licensed exactly the wrong conclusion.** Its correlations were
0.68 at best, its bands disagreed in sign, and one ran to a scan boundary — three published warnings
that should have made it inadmissible rather than merely soft.

## 122. What survives, and what re-opens

**SURVIVES** — neither depends on `lateral`:
* §102's renderer self-consistency: the drawn ribbon is **1.88–1.91 m** at every range from 5 m to
  18 m (ratio 1.02–1.03 to `project_ground`, the excess being the edge stroke). The renderer is not
  flaring and is not misdrawing the width.
* §112's far-then-near consistency (bias ≤ 0.09 m out to 50 m) — it compares the same quantity in
  two views, so a common lateral bias cancels.
* The yaw, the fade, `f·h` from the row flow.

**RE-OPENS:**
* ⛔ §114's *"the corridor's geometry is verified end to end"* — withdrawn as to lateral placement.
* The lane width, now **3.40 m** rather than §117's 3.15 m, because the right boundary moved from
  −1.10 to −1.44 once the shoulder was gated out. That gives `lane/h` = 2.144, and with the row-flow
  `f·h` = 2431 and the lens bound the cap becomes **W ≤ 3.61 m** — so **3.5 m is admissible again**.
  ⚠️ This figure has now read 2.97 / 3.15 / 3.40 across three passes, each time moving with which
  right-side feature was used. **The `h`/`f` split is NOT settled and §117's closure of it is
  withdrawn**; re-deciding it a fourth time on the same weak boundary would be the error, not the fix.

## 123. Two coding errors of mine this turn, both caught in-session

1. A window expression `int(mid-3.0/w*w/mpp*0+mid-460)` evaluated to `2·mid−460`, so a "read the
   whole row" diagnostic searched **only the right half** and returned nothing but negative offsets.
   Caught because *every* detection had the same sign — an impossible result for a lane.
2. The correction print was `lateral − offset` where the algebra gives `lateral + offset`, so it
   reported **+0.152 m** when the answer is **−0.41 m**. Caught by re-deriving the 1:1 relation
   instead of trusting the line.

⚠️ Both were found by a result being *impossible* rather than merely surprising. That is the only
reliable check available when the instrument and the analyst share an assumption.

---

# Part 26 — the near clip is in the wrong frame, and that is what "leaves the road" means

## 124. ⛔ The ribbon is painted on the BONNET, and `near_clip_m` was supposed to stop that

Sayed, on the corrected render: *"still the same problem, the trajectory leaves the road while the
ego vehicle is not doing this in the future."* The BEV panel in his frame shows the future path as a
**straight vertical line at y = 0** — so it is not the trajectory, and it is not the 0.28 m lateral
offset fixed in Part 25.

Locating his exact frame (the HUD's `t` is `t_session_s`, not `pts_s` — **frame 298**, speed
19.87 m/s, path straight to ±0.03 m over 50 m) and measuring it:

| range | clear L | clear R |
|---|---|---|
| 7 m | +1.23 m | **+0.22 m** |
| 9 m | +1.37 m | +1.61 m |

At mid-field the corridor is inside the lane. **The part that reads as "on the shoulder" is at
source rows ~870–1078 — which is the BONNET.** Tracing the drawn ribbon edges straight out of the
render, it runs down to **row 1078**, while the road stops being visible at row ~830 (6.37 m camera
range; `lk_failure_vs_static` settled this — a stronger tracker finds no motion below it).

⇒ **~250 rows, a quarter of the frame, painted on sheet metal** — at the widest part of the ribbon,
immediately beside the visible shoulder. That is what it looks like when the corridor leaves the road.

## 125. ⛔⛔ AND THE CODE ALREADY INTENDED TO PREVENT IT — IN THE WRONG FRAME

`viz.draw_trajectory_on_image` has a `near_clip_m` whose docstring says exactly this:

> *"`near_clip_m` drops the first few metres of the future path … that near strip is under the
> bonnet and not actually visible."*

But it is applied as `s_fwd >= near`, and **`s_fwd` is arc length from the TRAJECTORY ORIGIN — the
rear axle** — while "under the bonnet" is a fact about the **lens**. The camera sits
`longitudinal_m` = 2.10 m forward, so `near_clip_m = 4.0` started the ribbon at

    4.0 − 2.10 = **1.9 m ahead of the lens**

against a bonnet that hides the road to **6.37 m**. The stated intent was never achieved, in any
recording, since the parameter was written.

⭐ **Same class as `R-2026-09-15-longitudinal`, and the second instance in two days: a quantity
documented in one frame and applied in another, with nothing to type-check it.** The first cost a
retracted lane-width conclusion; this one cost a quarter of every rendered frame.

**FIX:** `near += cam.longitudinal_m`, so `near_clip_m` means what it says, plus a
`--near-clip-m` pipeline flag (it had none — the default was unreachable from the CLI). Rendered
here at **6.5 m**, which clears the 6.37 m bonnet.

## 126. What this does and does not explain

**EXPLAINS:** the near-field sprawl, which is where the corridor visually crosses onto the shoulder
and which no amount of lateral correction could fix, because the geometry there was never wrong —
the pixels simply should not have been drawn.

**DOES NOT EXPLAIN, and is separate:** the residual ~0.2 m rightward bias still visible on
individual frames (frame 298 at 7 m: clear L +1.23 vs clear R +0.22). Part 25's correction moved the
fleet median from 0.28 m to 0.06 m, but per-frame spread remains and the right boundary is still
the weakest measurement in this programme.

⚠️ **AND IT COST ME A WRONG DIAGNOSIS FIRST.** I read Sayed's screenshot as a far-field lean,
computed a yaw error from it, and was about to chase the yaw — until tracing the ribbon's own edges
showed them converging at source column 812 against the calibration's 816, i.e. **the ribbon is
drawn exactly as the calibration specifies**. Reading pixel positions off a compressed screenshot
has now produced a wrong answer three times in this document. **Locate the frame, render it at full
resolution, measure it.**

---

# Part 27 — it was the YAW all along, and I fitted it with the lateral

## 127. ⛔⛔ THE ERROR GROWS WITH RANGE — that is a yaw, not an offset

Sayed: *"fix the residual 0.2 m right bias. still no solution, trajectory leaving the road boundary."*

Measuring the corridor-to-lane-centre offset **as a function of range** on the v3 render
(yaw −5.35, lateral −0.41), 1953 reads:

| range | offset | clear L | clear R | lane |
|---|---|---|---|---|
| 9 m | +0.05 | +0.86 | +0.75 | 3.43 |
| 12 m | +0.08 | +0.84 | +0.67 | 3.37 |
| 16 m | +0.25 | +0.98 | +0.44 | 3.29 |
| 20 m | **+0.51** | +1.22 | **+0.24** | 3.27 |

`offset = −0.411 + 0.0438·x` ⇒ **a residual yaw of 2.51°**, which extrapolates to **+1.34 m at
40 m**. *That* is "the trajectory leaves the road boundary."

⛔ **AND IT MEANS PART 25's FIX WAS WRONG IN KIND.** I measured the offset only over **7–12 m**,
and over a 5 m window a yaw error and a lateral offset are **indistinguishable**. Setting
`--lateral-offset` to −0.41 nulled the error at ~10 m and made it **worse at every longer range** —
which is precisely why each "fix" left the far field still leaving the road. **A single-range
measurement cannot separate a level from an angle, and I never varied the range until now.**

## 128. ⭐ `clear_L` is the diagnostic, because parallel means FLAT

The left boundary is the one feature every instrument in this programme agrees on. If the ribbon is
parallel to the road, the gap from its left edge to that line is **constant with range** — no
lane-width assumption, no right-hand line, no horizon.

| render | clear L, 9 m → 18 m | implied yaw error |
|---|---|---|
| v3, yaw −5.35 | 0.86 → 1.07 | **+1.40°** ⇒ −6.75° |
| v4, yaw −7.39 | 0.82 → 0.76 | **−0.38°** ⇒ **−7.01°** |

Two renders, corrected independently, land on **−6.75°** and **−7.01°**.

⭐⭐ **−7.01° is EXACTLY what `lane_calib` measured on this recording in the first place** — and the
pipeline threw it away with *"yaw declined: −7.01 deg is 7.0 deg from the FOE, not credible"*, where
the FOE fit had already failed so the gate was comparing against a **nominal 0.0**.
`tests/test_trajrecon_yaw_gate.py` exists in this repo **because that rejection was a bug**.

## 129. ⛔ Every independent estimate pointed away from −5.35, and I discarded them one at a time

| source | yaw | what I did with it |
|---|---|---|
| `lane_calib`, this recording | **−7.01°** | rejected by the broken FOE gate (pre-existing) |
| independent VP fit, 154 frames / 6689 segments | **−6.05°** [−6.28, −5.83] | not carried forward |
| §88 pair-Hough | **−6.20…−6.51°** | ⛔ **I withdrew it** in favour of −5.29° |
| road-VP vs ribbon-VP, this part, n=294 | **−6.26°** | — |
| `clear_L` slope, v3 and v4 | **−6.75 / −7.01°** | adopted |

**Five estimates spanning −6.0 to −7.0, and the render used −5.35.** §87's "rendered −5.35 vs
optimum −5.10, the delivered render's yaw is right" was the single-line reading, and it is the only
one that agreed with the shipped value — so I kept it and dropped the rest.

⚠️ **ROOT CAUSE: the single-line yaw read-out evaluates one fitted line AT the assumed horizon row.**
It therefore measures the yaw *conditional on the horizon being right*, and the horizon is the one
parameter still unresolved (§105: row flow 438 vs drift 460–484). A conditional measurement that
happens to agree with the status quo is the easiest thing in the world to keep.

## 130. ⚠️ What is still not clean

* The measured **lane narrows with range** — 3.43 m at 9 m to 3.27 m at 20 m in v3, 3.38 → 3.26 in
  v4. That is a horizon signature, and it implies `v_h` is ~8 px low (≈456 rather than 448.4),
  in the direction §104's drift scan already pointed (460–484).
* The offset's residual slope in v4 (+0.98°) **disagrees** with `clear_L`'s (−0.38°). The difference
  is the right-hand line: its detection count falls from 540 at 9 m to 70 at 20 m and `clear_R`
  shrinks with it, so the "lane centre" drifts inward at long range. **`clear_L` is the trustworthy
  one; the offset-vs-range slope inherits the right boundary's weakness**, which is the same
  weakness that has now defeated six instruments.
* ⇒ The yaw is settled to about **±0.3°**; below that the horizon has to be settled first.

---

# Part 28 — one metric, every frame, with a known target

## 131. ⭐⭐ ROAD CONTAINMENT — and why "we know the future" is what makes it a goal

Sayed: *"you need to introduce a metric which represents non-leaving-the-road, which [is] a
clear simple verified goal, since we know the future."*

Right, and the diagnosis of five failed renders is that **no single quantity was ever made to answer
for the whole calibration**. `R-2026-09-15-oneside` measured only the left boundary, so a placement
error was invisible; `R-2026-09-16-yawnotlateral` measured only 7–12 m, so a yaw error was fitted
with the lateral parameter. Same failure twice: a bespoke instrument, a narrow window, no target.

**THE METRIC.** The ribbon is drawn around the car's **actual recorded future path**, and the car
did not leave the road. So:

> **on-road rate** — the fraction of (frame, range) samples where BOTH ribbon edges land on
> drivable surface — **must be ~100 % for a correct calibration.**

⭐ That is what *"since we know the future"* buys, and it is the property every earlier instrument
lacked: **a known right answer.** A clearance of 0.5 m could always be explained away as the driver
sitting off-centre. 78 % on-road cannot.

**THE SURFACE.** Drivable road is near-grey. MEASURED: asphalt **S = 8**, lane paint **S = 4**, pale
shoulder **S = 79**, bank **S = 95**, sky **S = 139**. One saturation threshold separates the
carriageway from the shoulder the corridor was spilling onto, and puts the paint on the road side
where it belongs. Verified by drawing it on six frames before any statistic.

## 132. ⭐ It ranks every render this document has shipped

300 frames × 7 ranges (10–40 m), both edges required on road:

| calibration | on-road | 20 m | 30 m | **40 m** |
|---|---|---|---|---|
| shipped v1 `−5.35 / 448.4 / −0.126` | **65.1 %** | 70.0 | 44.8 | **53.6** |
| v3 `−5.35 / 448.4 / −0.41` (the lateral "fix") | 76.3 % | 90.0 | 60.8 | **58.8** |
| v5 `−7.01 / 448.4 / −0.088` | 90.9 % | 94.0 | 90.0 | **78.8** |
| **v6 `−7.75 / 472 / −0.15`** | **98.0 %** | 97.7 | 95.7 | **95.0** |

⚠️ **Read the v3 row.** Part 25's lateral correction raised the near field (20 m: 70 → 90 %) and
barely moved the far field (40 m: 53.6 → 58.8 %). That is the signature of fitting an angle with a
level, visible at a glance in a table that did not exist when I made the error.

## 133. ⛔ The metric is DEGENERATE in the horizon, and in `f·h` — measured, not assumed

Raising the horizon moves samples toward the wide near field **and shrinks the ribbon relative to
the road**, so containment climbs without bound. MEASURED, at fixed yaw/lateral:

| horizon | on-road | ribbon/road width at 10 m | at 40 m |
|---|---|---|---|
| 440 | 89.1 % | 0.197 | 0.091 |
| 472 | 97.1 % | 0.185 | 0.077 |
| 500 | **98.0 %** | 0.176 | **0.068** |

The ratio falls monotonically while the score rises ⇒ **the score is being bought by drawing a
smaller ribbon.** ⇒ **containment cannot determine the horizon, and cannot choose between `f·h`
candidates either.** It is used only for `yaw` and `lateral`, which do not change the ribbon's size
and where the optimum is genuinely interior.

⭐ **This is the guard the five previous instruments never had**, and it is why the optimum is
reported with its degeneracy rather than as a measurement. An optimum at the edge of its scan is not
a measurement — a rule this document has now had to learn six times.

## 134. The optimisation, and its honest resolution

With the horizon fixed at **472** (from §104's left-line drift scan, basin 460–484, which is a
SHAPE constraint and therefore not degenerate), scanning `yaw × lateral`:

* binary containment: optimum **yaw −7.50, lateral 0.00 → 96.7 %**, interior ✓ — but the 1 %-down
  contour spans the whole grid, so it pins the calibration only to ±1° and ±0.35 m;
* **margin** (signed distance to the road edge, in metres, via a distance transform) is far sharper:

| calibration | on-road | **p5 margin** | median |
|---|---|---|---|
| shipped v1 | 81.9 % | **−0.25 m** | +0.48 |
| v3 | 88.0 % | −0.19 m | +0.57 |
| v5 | 95.4 % | +0.04 m | +0.62 |
| **v6 `−7.75 / 472 / −0.15`** | **98.0 %** | **+0.29 m** | **+0.94** |

⭐ **A positive p5 margin is the goal stated properly: 95 % of all samples sit at least 0.29 m
inside the road.** The shipped calibration was at **−0.25 m** — i.e. its 5th percentile was a
quarter of a metre *outside*.

## 135. ⚠️ What the metric cannot settle, stated plainly

Four self-consistent `(f·h, horizon)` triples, each optimised independently:

| | optimum | p5 margin | on-road |
|---|---|---|---|
| `f·h 2431, hz 448.4` (adopted) | −7.50 / −0.10 | +0.07 m | 96.0 % |
| **`f·h 2431, hz 472`** (drift scan) | **−7.75 / −0.15** | **+0.29 m** | **98.0 %** |
| `f·h 1935, hz 472` (row flow at 472) | −8.00 / −0.25 | +0.17 m | 97.4 % |
| `f·h 2431, hz 448.4, f at lens bound` | −8.00 / −0.10 | +0.10 m | 96.4 % |

The chosen triple scores best, **but its margin over the row-flow-consistent one is inside the
degeneracy** (a larger `h` draws a smaller ribbon). ⇒ **`f·h` and the horizon remain unresolved**,
and the practical consequence is that the **range labels** on the ticks carry that uncertainty:
at horizon 472 the row flow would want `f·h ≈ 1935`, which would make the labelled ranges ~26 %
shorter. **The corridor's placement is settled to the metric; its range calibration is not.**

⚠️ The yaw is now consistent across every non-conditional estimate: `lane_calib` −7.01, `clear_L`
slope −6.75/−7.01, containment −7.50, margin −7.75. The shipped −5.35 is outside all of them.

## 136. ⭐ Verified ON THE RENDER, and the model agrees with it

The optimisation scored a *predicted* projection. Closing the loop — reading the **drawn** ribbon out
of the composite and testing it against the drivable mask built from the source frame:

| render | on-road | p1 | **p5 margin** | median | n |
|---|---|---|---|---|---|
| v1 `−5.35 / 448.4 / −0.126` | 84.3 % | −0.25 | **−0.13 m** | +0.33 | 15598 |
| v5 `−7.01 / 448.4 / −0.088` | 97.2 % | −0.20 | +0.12 m | +0.63 | 7554 |
| **v6 `−7.75 / 472 / −0.15`** | **97.9 %** | −0.16 | **+0.27 m** | **+0.85** | 7552 |

**Predicted 98.0 %, measured on the render 97.9 %** — the projection model and the renderer agree,
which is what makes the optimisation admissible rather than circular.

⚠️ v1's sample count is **twice** the others' because it drew the ribbon down over the bonnet
(`R-2026-09-15-nearclip`); the near-clip fix removed those rows, and they were the widest and worst.

## 137. ⭐⭐ THE PIPELINE HAD THE ANSWER. THE OVERRIDE WAS THE BUG.

From the v6 run's own calibration stage, with no input from this document:

    LaneCalib(yaw=-6.86 deg, lane_width=3.40 m, frames=59, segments=981, yaw_spread=0.31 deg)
    yaw: FOE -6.31 deg -> lane VP -6.86 deg (-0.55 deg, 0.38 m at 40 m)

* `lane_width = 3.40 m` over 981 segments — **identical to the independent left-anchored
  measurement (3.38–3.40 m, robust sd 0.18)**;
* the **FOE fitted this time** (−6.31°), so the credibility gate had a real reference and would have
  **accepted** −6.86°.

Scored with the metric (horizon 472, lateral −0.15):

| yaw | source | on-road | p5 |
|---|---|---|---|
| **−5.35** | operator override (shipped) | 82.1 % | **−0.25 m** |
| −6.31 | pipeline FOE | 96.9 % | +0.13 m |
| **−6.86** | **pipeline's own lane VP** | 97.7 % | **+0.23 m** |
| −7.01 | `lane_calib` v1 | 97.9 % | +0.24 m |
| −7.75 | this document's optimum | 98.3 % | +0.29 m |

⭐ **My hand-tuned value beats the pipeline's own measurement by 0.06 m of p5 margin — inside the
noise.** Six renders of parameter-chasing recovered what `lane_calib` reports unaided. **The entire
defect was `--cam-yaw -5.35`, an operator override that bypassed a working measurement.**

⚠️ **AND I MISREAD THE GATE WHILE CHECKING THIS.** I read `lane_calib.py:282` — which still compares
against `cam.yaw` and still says *"from the FOE"* unconditionally — and reported the gate as unfixed.
It is fixed, **in the caller**: `pipeline.py:583` sets `max_yaw_correction_deg = 4.0 if foe_measured
else 15.0`, exactly as `test_trajrecon_yaw_gate.py` asserts. ⇒ **Reading a guard without reading its
call site is reading half the code**, and it is the same mistake as `R-2026-09-15-nearclip`, where
the docstring was right and the application wrong.

⇒ **STANDING RECOMMENDATION: do not override `--cam-yaw` on this corpus.** Let `lane_calib` measure
it. An override is admissible only with a measurement that beats the metric by more than its noise,
and this one did not.

---

# Part 29 — three instruments, three different questions, and only the third one mattered

## §138 The complaint contained the specification and I had scored half of it

Sayed, seven renders in, on the frame at `t = 33.63 s`:

> *"the trajectory still [is] leaving the road, **knowing that ego [is] driving between the
> road markings**"*

`road_containment.py` (Part 28) scores **"is the ribbon on drivable asphalt?"** and gave the v6
render 96 % of frames clean. That answer is true. It is also **permissive in exactly the direction
of the error**, because this is a **two-lane carriageway** — the asphalt continues across the
left-hand line for another ~3.5 m, so a ribbon drifting toward the adjacent lane never leaves the
mask and never costs a point.

**MEASURED** on his frame (source 908), scale `h/(v−v_h)`, no other calibration:

| range | clear to LEFT line | clear to RIGHT line | lane | centred clear would be |
|---|---|---|---|---|
| 10 m | **+0.43 m** | +1.29 m | 3.59 m | 0.87 m |
| 12 m | **+0.31 m** | +1.39 m | 3.58 m | 0.86 m |

Both positive ⇒ road-containment sees nothing. Yet the ribbon is **0.44 m left of lane centre at
10 m and 0.55 m at 12 m**, and **a bias that grows with range is residual yaw**.

## §139 Two of the three instruments were also simply broken

**`corridor_edges` locked onto roadside foliage** (`R-2026-09-19-greenhue`). It selected the drawn
ribbon by hue alone. Composite frame 165 row 340 → columns **834–845 at BGR (28,99,77)**, dark
foliage, while the ribbon sits at **470–521 in pale (206,229,208)**. A 16 px "1.855 m ribbon" gives
**0.112 m/px, 20× the true scale**, so ordinary pixel distances became **−6.72 / −8.08 / −10.68 m**;
and where both share a row, min/max spans both → (470, 835) for a ribbon ending at 521. This
manufactured *"19.8 % of frames leave the road"* on frames whose ribbon is squarely between the
painted lines. ⇒ **a colour test selecting a DRAWN overlay must gate on brightness, not only hue.**

**`drivable_mask` called shadowed asphalt "not road"** (`R-2026-09-19-hsvshadow`). Shadow here is
sky-lit, so it is blue, and `S = (max−min)/max` makes a dark blue-grey pixel read as saturated:
frame 1860 shadowed asphalt BGR (98,79,58) → **S = 104** against a threshold written for sunlit
asphalt (**S = 8**). Replaced with CIELAB chromaticity, **MEASURED** on frame 1860:

| region | L\* | a\* | b\* | kept by `\|a*\|<8, −22<b*<6` |
|---|---|---|---|---|
| sunlit asphalt | 119 | −1.3 | −3.0 | 99.8 % |
| **shadowed asphalt** | 76 | −1.3 | −12.1 | **100.0 %** (was lost) |
| lane paint | 155 | −2.6 | +0.6 | 96.5 % |
| vegetation bank | 107 | −7.9 | +23.6 | 0.7 % |
| concrete barrier | 193 | +2.0 | +21.3 | **0.0 %** (was kept) |
| sky | 183 | −6.5 | −33.2 | 0.0 % |

## §140 `lane_containment.py` — offset from the LANE CENTRE, whose correct value is zero

The car drove between the markings, so the ribbon drawn around its own recorded future path must be
**centred between them**. That is the metric, and the target is **0.000 m**, not "on the road".

⚠️ **Its own first version could not return a failure.** It searched for paint strictly *outside*
the ribbon edges, so a sample where the ribbon sits ON a line was **dropped for want of a detection
instead of recorded as a crossing** — it reported `cross 0.0 %` for **every** arm, including v1,
whose ribbon is over a metre off centre at 30 m. Anchoring both searches on the ribbon's CENTRE
gives v1 **8.2 %** and v6 **4.0 %**. *(Caught by a control, not by inspection.)*

## §141 The answer: yaw −6.40°, lateral −0.320 m, horizon 472

Joint scan over yaw × lateral, 300 frames, scoring the **worst per-range median** so a calibration
cannot trade near field against far — which is how the yaw error hid for three renders. Interior
optimum, not on a scan edge.

| calibration | worst-range \|offset\| | per-range offsets, 8→30 m |
|---|---|---|
| v6 shipped (−7.75, −0.150) | 0.356 m | −0.020 +0.097 +0.143 +0.190 +0.227 +0.278 **+0.356** +0.348 |
| **v7 (−6.40, −0.320)** | **0.038 m** | −0.038 +0.007 +0.028 +0.001 −0.030 −0.021 −0.031 +0.031 |

v6 climbs monotonically with range (a yaw error); **v7 is flat**. Per frame, over 400 frames:

| calibration | p50 | frames >30 cm off | cross rate |
|---|---|---|---|
| v1 (−5.35, −0.126) | −0.415 | 65.3 % | 7.8 % |
| v5 (−7.01, −0.088) | −0.058 | 36.6 % | 3.3 % |
| v6 (−7.75, −0.150) | +0.180 | 46.8 % | 4.2 % |
| pipeline LaneCalib (−6.86, −0.088) | −0.081 | 40.4 % | 3.3 % |
| **v7 (−6.40, −0.320)** | **+0.027** | **32.8 %** | **3.1 %** |

⭐ **−6.40° agrees with every independent estimate and with the pipeline itself** — `LaneCalib`
−6.86 (981 segments), FOE −6.31, VP fit −6.05, pair-Hough −6.20…−6.51, road-VP −6.26. **−7.75 agreed
with none of them**, and it was mine.

## §142 What the remaining spread is — and what it is not

**MEASURED**, splitting v7's per-frame offsets by how much the road curves ahead:

| \|path lateral @ 30 m\| | n | median offset | robust sd |
|---|---|---|---|
| **0–0.5 m (straight)** | **274** | **+0.001 m** | 0.349 |
| 0.5–1.5 m (curving) | 55 | +0.149 m | 0.244 |

**On straight road the calibration is correct to 1 mm of median offset.** The residual correlates
with curvature (r = +0.301) and steering (r = +0.269) — a **path/heading effect in bends**, not a
static camera parameter, and it is worth ~0.15 m.

⛔ **THE PER-FRAME SPREAD IS NOT ALL CALIBRATION AND MUST NOT BE READ AS SUCH.** The metric's premise
is that the car drives centred — true on average, false in any given frame, where the driver is a
few centimetres off and correcting. Only the **central value over many frames** is a calibration
claim; the spread bounds how much any single frame can be trusted. On frame 908 itself v7 still
reads +0.43 m at 12 m, about 1.2 robust-sd from the straight-road median — **an unremarkable frame
for this spread, and not separable from real driving with one frame.**

## §143 The bend residual is the DRIVER, not a parameter — two pre-registered tests

The +0.15 m left in bends (§142) is the only thing left after v7. Two hypotheses, each with the
signature it would have to produce written down **before** the run.

**H1 — a time lag between the ego trace and the camera.** Signature: drawing the path recorded at
frame `f+k` onto the image at frame `f` would **null the bias at exactly one k** and not at the
others. **REFUTED.** MEASURED over 140 curving frames (`|path lateral @30 m| > 0.5 m`), n ≈ 530
samples per shift:

| shift | −6 | −4 | −2 | −1 | 0 | +1 | +2 | +4 | +6 |
|---|---|---|---|---|---|---|---|---|---|
| Δt (s) | −0.200 | −0.133 | −0.067 | −0.033 | 0 | +0.033 | +0.067 | +0.133 | +0.200 |
| median offset (m) | +0.116 | +0.116 | +0.112 | +0.109 | +0.108 | +0.107 | +0.105 | +0.104 | +0.106 |

**Flat across ±0.2 s, no null anywhere.** There is no lag to find. *(Testing this on all frames
would have diluted it to nothing — a lag is invisible on a straight road, which is why the frame set
is curving-only.)*

**H2 — the driver cuts the corner.** Signature: the bias **flips sign with turn direction**, sitting
toward the INSIDE of the bend either way. **CONFIRMED.** MEASURED:

| group | frames | n | median offset | robust sd |
|---|---|---|---|---|
| LEFT bend (`lat30 > +0.5`) | 281 | 590 | **+0.164** | 0.211 |
| straight (`\|lat30\| < 0.5`) | 1839 | 625 | −0.032 | 0.374 |
| RIGHT bend (`lat30 < −0.5`) | 96 | 237 | **−0.457** | 0.472 |

`+` is left of lane centre, so **both bends put the ribbon toward the inside** — the car really does
drive toward the inside of a bend, and the ribbon is drawn around the car's own recorded path.

⛔ **THIS IS THE METRIC'S PREMISE FAILING, NOT THE CALIBRATION.** `lane_containment` assumes the car
is centred between the markings. That holds on average and on straight road (median −0.032 m over
1839 frames) and **is simply false in a bend**. ⇒ **The calibration claim is the straight-road
median; the bend numbers are a measurement of driving.** Anyone tuning a camera parameter against
the bend figure would be fitting the driver's line into the mount geometry — the same shape of
error as `R-2026-09-16-yawnotlateral`, where a driving/geometry effect was absorbed into the wrong
parameter.

⚠️ The right-bend group is only 96 frames with sd 0.472 (against 281 and 0.211 for left bends), so
its **−0.457 is the weakest number in the table** and the asymmetry between the two directions is
not established — only the sign flip is.

## §144 The horizon, at last — measured by a signature that needs no known lane width

The horizon has been the one parameter no instrument could pin, because `road_containment` is
**degenerate** in it (§137: the ribbon/road width ratio falls 0.197→0.176 as the horizon rises
440→500 *while the score rises*) and because every direct estimate needed `f·h`, which is itself
degenerate.

⭐ **THE SIGNATURE THAT WORKS: THE LANE IS THE SAME WIDTH AT 8 m AND AT 30 m.** The lateral scale is
`h/(v − v_h)`, so a WRONG `v_h` makes the *measured* width **trend with range**. The horizon that
makes it **flat** is the right one — and the flatness condition **does not involve the width's
value**, so `h` and `f` drop out entirely. Exactly the shape of the `clear_L` test
(`R-2026-09-16-yawnotlateral`): *parallel means flat*.

**MEASURED**, 200 frames, yaw −6.40, lateral −0.320:

| horizon | 440 | 448 | 456 | **463** | 464 | 472 | 480 | 488 | 496 |
|---|---|---|---|---|---|---|---|---|---|
| slope (m/m) | +0.0154 | −0.0053 | −0.0068 | **−0.00015** | +0.0004 | +0.0079 | +0.0146 | +0.0091 | +0.0147 |
| mean width (m) | 3.580 | 3.410 | 3.480 | 3.618 | 3.640 | 3.805 | 3.951 | 4.017 | 4.096 |

⭐ **`v_h = 463`**, with a broad interior minimum over **460–464** (|slope| ≤ 0.00065). At 464 the
per-range widths are 3.610 3.659 3.661 3.643 3.634 3.595 3.659 3.656 — **sd 0.024 m across a 3.75×
range span.** Not on a scan edge.

**WHY THIS MATTERS EVEN THOUGH THE LANE-CENTRE METRIC BARELY MOVES.** The optimum yaw is **−6.40 at
both** 472 and 463; only the lateral shifts (−0.340 → −0.260) and the worst-range offset is
unchanged (0.068 → 0.061). That is expected — the offset is a *difference* of clearances, so a scale
error cancels in it. **The horizon is invisible to the centre metric and very visible on screen:**

    ribbon width drawn at v_h = 472 vs the truth at 463 = (v − 463)/(v − 472)
      at 10 m  (row 715):  252/243 = 1.037   →  3.7 % too wide
      at 30 m  (row 553):   90/81  = 1.111   →  11.1 % too wide

⇒ **the ribbon FLARES OUTWARD WITH RANGE**, its edges ~0.10 m further out each side at 30 m. A
symmetric flare does not move the centre, so `lane_containment` cannot see it — **and a flare is
precisely what "the trajectory is leaving the road" looks like.** This is the third instrument in
this part to be blind in exactly the direction of the complaint.

⚠️ **A ROAD GRADE IS ABSORBED HERE AND THAT IS CORRECT, NOT A CONFOUND.** Over 8–30 m a constant
grade acts as a horizon offset, and the quantity we need is the one that makes the drawing land on
*this* road — so fitting it in is the right behaviour. It does mean **`v_h = 463` is an effective
horizon for this recording, not a claim about the camera's mounting pitch**, and it must not be
quoted as one.

⚠️ **AND IT STILL DOES NOT BREAK `f·h`.** The mean width at the flat point is 3.618 m *given*
`h = 1.586`; the pipeline's `LaneCalib` reads 3.40 m from 981 segments. Those differ by 6.4 %, which
is the `f·h` degeneracy showing up as a height question, not a horizon one. **The flatness result is
independent of it; the width VALUE is not.** Do not quote 3.618 m as a measured lane width.

## §145 v8 — the numbers that go with the render

**MEASURED**, 400 frames, `lane_containment`:

| calibration | p50 | p5 | p95 | frames >30 cm off | cross rate |
|---|---|---|---|---|---|
| v1 first shipped (−5.35, −0.126, 448.4) | −0.212 | −0.651 | +0.394 | 45.1 % | 8.0 % |
| v6 last sent (−7.75, −0.150, 472.0) | +0.180 | −0.517 | +0.734 | 46.8 % | 4.2 % |
| **v8 (−6.40, −0.260, 463.0)** | **−0.003** | −0.653 | +0.556 | **35.7 %** | 4.6 % |

**Per-range median offset (m), + = ribbon left of lane centre:**

| calibration | 8 | 10 | 12 | 15 | 18 | 22 | 26 | 30 |
|---|---|---|---|---|---|---|---|---|
| v1 | −0.263 | −0.228 | −0.179 | −0.207 | −0.148 | −0.149 | −0.103 | −0.266 |
| v6 | −0.022 | +0.055 | +0.145 | +0.186 | +0.280 | +0.300 | +0.382 | +0.373 |
| **v8** | **−0.081** | **−0.021** | **+0.041** | **−0.012** | **−0.022** | **−0.012** | **+0.010** | **−0.037** |

v6 climbs monotonically — a yaw error. **v8 is flat within 0.081 m over an 8–30 m span.** On straight
road alone (333 of 400 frames, the calibration claim — §143) the median is **−0.040 m**, against
v6's +0.124 and v1's −0.211.

⚠️ **THE CROSSING RATE DID NOT IMPROVE (4.2 % → 4.6 %) AND THAT IS REPORTED, NOT BURIED.** Crossings
are dominated by frames where the car genuinely runs close to a line, which no camera parameter
fixes. The centring and the flatness are what moved.

⭐ **A SCALE-FREE CONFIRMATION OF THE HORIZON**, in pure pixels — ribbon width ÷ lane width at the
same row, which needs no `f`, no `h` and no horizon to evaluate:

| | 8 m | 10 m | 12 m | 15 m | 18 m | 22 m | 26 m | 30 m |
|---|---|---|---|---|---|---|---|---|
| v6 (hz 472) | 0.507 | 0.499 | 0.497 | 0.492 | 0.489 | 0.487 | 0.485 | **0.480** |
| v8 (hz 463) | 0.519 | 0.512 | 0.518 | 0.513 | 0.517 | 0.525 | 0.515 | **0.511** |

v6 declines monotonically — **that is the flare**. v8 is flat to sd 0.005.

⚠️ **This is the SAME CONDITION as §144's lane-width test re-expressed, not independent evidence for
it** — same detections, same underlying requirement that the drawn 1.855 m tracks the lane across
range. It is worth stating separately only because it contains no metres at all.

⚠️ **The implied lane width of 3.59 m is CONDITIONAL ON `h = 1.586`** and is not a free measurement:
`ratio = 1.855·h_true /(W_true·h)`, so the value pins `W_true·h`, not `W_true`. The `f·h` degeneracy
is untouched by any of this. `LaneCalib` reads 3.40 m.

## §146 The one assumption that cannot be tested from inside this metric

`lane_containment` sets the lateral by forcing the ribbon to the lane centre **on average**, so it
**absorbs the driver's mean lane position into the camera's lateral offset**. The two are exactly
degenerate here and no amount of data separates them — if the driver habitually sits 10 cm left,
the fitted `lateral` is 10 cm wrong and the metric reads zero.

The only check available is a physical one, and it passes: the fitted **−0.260 m** sits close to the
nominal windscreen-mount value of **−0.35 m**, so the assumption is not being forced to buy an
absurd geometry. ⇒ **quote `lateral = −0.260 m` as "the value that centres the ribbon on this
recording", never as a measured mount offset.**

## §147 The flow horizon and the paint horizon disagree by 37 px — and the pipeline said so

The v8 run's own log, unprompted:

```
f*h from ground flow:  f*h = 2872.7 px*m  95% CI [2644, 2909]
                       horizon row = 427.4 px  95% CI [425.7, 438.5]
WARN  horizon from flow 427.4 px vs the camera's current 464.4 px (-37.0 px)
WARN  ⚠️ that is a LARGE disagreement. Flow and paint measure different things
      when the road is not planar; do not adopt either silently. Validate
      against the markings ... before trusting f*h to set a height.
ScaleCalib(f*h=2873 from 87285 tracks, spread 0%, focal=1687 px, height=1.703 m)
```

**`horizon_calib` IS that validation against the markings**, and it says **463.9**.

**They cannot both describe one planar road.** `flow_calib` measures longitudinally
(how fast the ground flows past, against the odometer); `horizon_calib` measures laterally
(how wide the lane subtends). A grade or crest separates them, which is precisely the
caveat the log names — and it is the same caveat `horizon_calib` carries in its own docstring
about absorbing grade.

⭐ **FOR THIS DELIVERABLE THE PAINT HORIZON IS THE RIGHT ONE, and the reason is the
specification, not a preference.** The thing being asked for is a ribbon that sits between
the lane markings. That is a LATERAL requirement, so it must be served by the scale that was
measured laterally. Adopting 427 would restore the flare the render was remade to remove:
the flatness check (§145) is decisively against it — at horizon 440 the width-vs-range slope
is already **+0.0154 m/m**, and 427 lies further out in the same direction.

⛔ **WHAT IS NOT SETTLED, AND MUST NOT BE WRITTEN UP AS IF IT WERE.** `f·h` remains degenerate,
and this disagreement is now its clearest expression:

| source | f·h | horizon | implied h | implied lane at that h |
|---|---|---|---|---|
| flow + scale_calib | 2873 | 427.4 | 1.703 m | 3.91 m (too wide for a lane) |
| in use for v8 | 2431 | 463.0 | 1.586 m | 3.64 m |
| if the lane is the 3.5 m standard | — | 463.9 | 1.524 m | 3.50 m |

`horizon_calib` returns **W/h = 2.2969**, a ratio, so it constrains the *product* and never
splits it. ⇒ **Quote no camera height and no lane width from this work.** What is measured is
the horizon (laterally, 463.9, R² 0.984) and the yaw (−6.40, agreeing with five independent
estimates). The height, the focal and the lane width are still one equation short, and
`ScaleCalib`'s "spread 0 %" is a fit statistic, not evidence that the split is right.

## §148 The frame Sayed sent, settled — a constant parameter cannot vary with time

v8 improves his frame (source 908, t = 33.63 s) but does not null it, and that needed an answer
rather than an excuse. **MEASURED** on the source frame, scale `h/(v−v_h)`:

| | 8 m | 10 m | 12 m | 18 m | median off-centre |
|---|---|---|---|---|---|
| v6 clear L / R | *(dropped)* | +0.42 / +1.66 | +0.31 / +1.40 | **−0.02** / +1.72 | **+0.62 m** |
| v8 clear L / R | +0.55 / +1.27 | +0.47 / +1.11 | +0.39 / +1.15 | +0.16 / +1.29 | **+0.37 m** |

v6's left clearance goes **negative at 18 m** — the ribbon is over the line, which is exactly what he
reported. v8 is positive at every range. But +0.37 m remains, against a straight-road median of
−0.04 m.

⭐ **THE ARGUMENT THAT SETTLES IT COSTS NOTHING AND I SHOULD HAVE REACHED FOR IT SEVEN RENDERS AGO:
a camera parameter is CONSTANT, so it produces a CONSTANT offset. Anything that varies with time is
the car.** **MEASURED**, v8, every 6th frame through his moment:

| t (s) | 31.6 | 32.0 | 32.4 | 32.8 | 33.2 | **33.6** | 34.0 | 34.4 | 35.0 | 35.6 | 36.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| steer (°) | −3.00 | −2.65 | −3.06 | −3.10 | −1.93 | **−2.77** | −1.62 | −0.32 | +0.61 | +0.49 | −0.28 |
| off-centre (m) | −0.29 | −0.17 | +0.10 | +0.15 | +0.33 | **+0.37** | +0.55 | +0.74 | +0.74 | +0.66 | +0.72 |

The offset **drifts smoothly from −0.29 m to +0.74 m over about three seconds**, tracking the
steering from −3.0° to +1.1°: the driver steers left, the car moves left in the lane. **No
calibration can do that.** ⇒ the residual at t = 33.63 s is **the car's own lane position**, and the
frame was sampled mid-drift.

⚠️ **THE ONE WAY THIS ARGUMENT COULD FAIL, CHECKED:** measurement noise is random and would not
drift smoothly for 3 s while correlating with an independent channel (the steering). It does both.

⛔ **WHAT THIS DOES NOT LICENSE.** It explains ONE frame; it does not retire the global numbers. The
straight-road median (−0.040 m over 333 frames) is still the calibration claim, and the per-frame
robust sd of 0.379 m — which this stretch shows is largely real driving — is still the reason a
single frame can never settle a calibration question in either direction. **That cuts both ways: it
was not evidence against v6 either, and v6 was wrong for reasons measured over 400 frames.**

---

# Part 30 — the camera moves during the clip: EIS, measured at last

## §149 The PI was right, and the evidence against him was pooled

Frame 932 (t = 34.43 s), v8 render. Both lane lines fitted as **whole Hough lines** — top-hat to isolate
paint, `HoughLinesP`, segments grouped per side and fitted as one line `u = a + b·v` weighted by segment
length — and **drawn onto the frame and checked to lie on the paint before any number was read**
(`scratchpad/vp_check.py`, image `vp_932.png`). They meet at **(834, 473)**. The drawn ribbon's
straight-ahead direction meets that row at **787**. The ribbon is rotated **1.75 deg left of the lane**,
its left edge crossing the solid line at ~20–30 m. Scanning the drawing yaw on that frame nulls the
angle at **−4.65 deg**, ~1:1 in yaw.

## §150 It drifts, with time, not with steering

Gated straight frames only (`|path lateral @ 40 m| < 0.3 m`; both lines found; slopes, vanishing-point
row and lane width at row 720 all physically plausible), every 3rd frame, n = 128:

| t (s) | 0–5 | 5–10 | 10–15 | 15–20 | 20–25 | 25–30 | 30–35 | 35–40 | 40–45 | 50–55 | 60–65 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lane VP column | 757 | 747 | 746 | 766 | 773 | 762 | **821** | **805** | **807** | 760 | 798 |
| lane VP row | 475 | 463 | 468 | 483 | 484 | 458 | **490** | **489** | 479 | 467 | 463 |

Column p10–p90 spread **80 px ≈ 3 deg**, lag-1 autocorrelation **+0.64**; row spread **37 px**,
autocorrelation **+0.54** — slow drifts, not noise. Against the angle: steering **r = +0.09**, path
curvature r = −0.35, **time r = −0.41**; steering is flat across bins (−1.1…+0.6 deg) while the angle
moves ~4 deg. **The car is not turning. The image is.** EIS (operator-confirmed, Part 6) moves the crop
window, and a moved crop is a moved camera.

⭐ **This is the risk §2 named on day one** — *"the mount rotation is not constant across frames — the
one assumption the whole method rests on"* — and §1041 left as *"not measured"*. Every constant yaw since
was a pooled average of a drifting quantity, which is why each render fixed one stretch and broke
another: **v6's −7.75 fits t < 30 s; the 30–50 s stretch wants ≈ −4.9; v8's −6.40 fits neither end.**

## §151 The fix: a per-frame attitude track, for DRAWING only

`stack/tanitad/data/trajrecon/attitude_track.py` + `pipeline.py --attitude-track`: a
`(t_session_s, yaw_deg[, horizon_row])` track interpolated per rendered frame. The trajectory is
camera-independent and does not move. Track v9: running median ±3 s (widened to ≥ 3 samples, max ±8 s),
then a 1.5 s Gaussian; yaw **−7.92…−5.62 deg**, horizon **456…489 px**.

⚠️ **What it absorbs.** Fitting to the lane VP assumes the car heads along the lane on average over the
window, so heading changes slower than ~6 s are absorbed as "camera". Real lane-keeping wobble is
shorter and survives. **The track is a measurement of effective pointing for this clip, not a mount
calibration**, and the constant `--cam-yaw` stays in the record only as the fallback.

## §152 Held-out check of track v9 — yaw passes (halfway), horizon FAILS

Evaluated on **337 straight frames that were NOT used to build the track** (fit on every 3rd frame,
evaluated on the frames offset by one). Two pre-registered tests: the path-vs-lane angle must fall
toward 0 in every time bin, and the painted line must read a constant ~0.15 m wide.

| t (s) | 0–10 | 10–20 | 20–30 | **30–40** | **40–50** | 60–70 | pooled |
|---|---|---|---|---|---|---|---|
| angle, v8 constant −6.40 | +1.38 | +0.61 | +1.12 | **−1.34** | **−1.40** | +0.03 | med +0.57, sd 1.20 |
| angle, yaw track | −0.09 | −0.40 | +0.17 | **−0.55** | **−0.70** | +0.41 | med −0.22, sd 0.92 |
| paint width, horizon 463 | 0.246 | 0.156 | 0.162 | 0.144 | 0.150 | 0.146 | med 0.156, sd 0.019 |
| paint width, horizon track | 0.256 | 0.168 | 0.156 | 0.164 | 0.164 | 0.147 | med 0.162, sd 0.021 |

- **Yaw track: adopted, but it only halves the error in the PI's stretch** (−1.4 → −0.6 deg at 30–50 s):
  the ±3 s median + Gaussian blurs a fast jump (the VP column moves ~60 px in 5 s at t ≈ 30 s).
- ⛔ **Horizon track: REJECTED by its own test.** Paint width gets less consistent (sd 0.019 → 0.021)
  and moves further from 0.15 m exactly where it was meant to help. The straight-line VP **row** is a
  poor horizon estimator — a crest or dip ahead bends the fitted lines vertically — so its 458–490 px
  drift is not established as a camera movement. **Horizon stays at 463.**
- The 0–10 s bin reads 0.25 m under both horizons: a different (wider) marking there, not a scale
  error — that bin cannot be used for scale.

## §153 Per-frame camera pointing from the CAR itself — the cowl edge (PI's suggestion)

A smoothed lane-based track lags the fast stabiliser jumps, and a per-frame lane estimate is too noisy.
**Anything rigid to the car moves in the image only when the image moves relative to the car** — which
is exactly what the overlay needs, per frame, independent of the road.

Three car-fixed references, registered by phase correlation on gradient magnitude against one fixed
reference frame (`probes/carfixed_layer.py`, `probes/cowl_track.py`):

| reference | 5-s bins vs bonnet | note |
|---|---|---|
| windscreen sticker (top-left, translucent) | agrees to t≈45 s, then drifts off by up to 10 px | trees show through it |
| glossy bonnet (rows 860–1080, ±15-frame median) | — | r = 0.990 with the sticker in dx overall |
| **windscreen/bonnet COWL edge** (rows 1022–1080) — *the PI's suggestion* | **within 1–2 px, both axes, whole clip** | opaque, non-reflective: registers PER FRAME (response 0.77, 0.31 px jitter at ±4 frames) |

27 of 2216 cowl frames had low confidence (response < 0.3 or > 10 px off the local median) and are
interpolated. Result: **yaw −7.80…−5.86 deg, horizon 450–482 px, max frame-to-frame yaw step 0.15 deg.**

## §154 The acceptance test: did the car end up where the ribbon said?

`probes/predict_vs_outcome.py` — the PI's original specification ("since we know the future"). At
frame F the ribbon's centre at range X, as a fraction of the lane width between the lane lines fitted
in F; at frame F′, when the car has travelled X − 8 m, the ribbon's centre at 8 m (where a yaw error
barely matters) between the lane lines fitted in F′. Same patch of road. **The difference is the
drawing error in metres — assuming nothing about the car being centred or parallel.** Control: the
same test on v8's constant yaw.

| error at 30 m (m) | 0–10 s | 10–20 s | 20–30 s | 30–40 s | robust sd |
|---|---|---|---|---|---|
| v8 constant −6.40 *(control)* | +0.42 | +0.44 | +0.27 | **−0.32** | 0.50 |
| **cowl per-frame** | +0.08 | −0.03 | −0.02 | **−0.15** | 0.29 |

At 20 m the cowl track is within ±0.06 m in every bin (v8: up to 0.24). **The test discriminates** (the
control fails it), and the cowl-referenced camera passes. ⇒ In 30–40 s, the ribbon pointing toward the
left line is **mostly the car's own drift within the lane** — the residual drawing error there is
≤0.15 m at 30 m.

Horizon from the cowl: paint-width spread between 10-s bins **0.026 → 0.018 m** (30–40 s: 0.140 → 0.156
m); per-frame sd 0.011 → 0.013. **Passes modestly; adopted.**

⚠️ **Coverage:** no prediction/outcome pairs after ~50 s (both lane lines must be fitted in F and F′,
and the late clip is curves); the 40 m row is too noisy to use (sd ≈ 0.65 m). The cowl correction is
applied everywhere, but it is **validated only on 0–50 s**.

## §155 The 10-second bin hid the PI's 1.5 seconds — again

§154's "−0.15 m at 30 m in 30–40 s" is a **bin median**. Frame by frame around the PI's frame, the cowl
track still leaves **−0.45 … −0.77 m at 30 m over t = 33.9–35.4 s** (−0.54 m at 34.43 s; v8 there:
−0.70). The same pooled-vs-specific error as `R-2026-09-24-constantyaw`, one level down — in time bins
instead of the whole clip. ⇒ **When the PI names a moment, report that moment's pairs, not its bin.**

## §156 It is not the trajectory — the gyro says so

Hypothesis (stated to the PI as a suspicion): the reconstructed path is too straight during a small
steering correction, because the error grows faster than linearly with range. **REFUTED.** The phone's
gyro projected on gravity (landscape mount, gravity along +x) is the car's true yaw rate; dead-reckoning
30 m ahead from gyro + speed alone reproduces the recorded path **to within 0.04 m at every sample from
33.0 to 36.5 s** (e.g. 34.53 s: path +0.02, gyro +0.02). The trajectory is right; the car went nearly
straight.

## §157 So it is the camera — and the car-fixed references under-read it

At frame 932 two independent lane-based measures agree on the yaw the drawing needs: the lane vanishing
point (**−4.65°**) and prediction-vs-outcome (−6.11 + 0.54/22 rad = **−4.70°**). The cowl says **−6.11°**.
Across the clip the lane VP moves **1.34× (column) / 1.43× (row)** as far as the car-fixed references.
Pure EIS warps shift near and far content equally, so a pure-EIS model predicts 1.0.

**HYPOTHESIS (not measured):** the phone also moves in its holder. The cowl, the bonnet and the sticker
all sit ~1–1.5 m from the lens, so a sideways translation of the phone shifts them by `f·δ/Z` while it
does not move the distant lane at all, and a pivoting holder couples that translation to the rotation.
That would make every car-fixed reference a biased measure of the rotation the drawing needs, and it is
consistent with sticker and cowl agreeing with each other (similar Z) but not with the lane.

⇒ **v12: fuse.** The cowl for fast changes (precise, lag-free); prediction-vs-outcome against the lane
lines for the absolute level, low-passed over ~2 s. Fit on half the prediction frames, tested on the
other half.

## §158 v12 rendered — the PI's frame is parallel to the lane

Frame 932 (t = 34.43 s), measured against both lane lines fitted as whole lines:

| render | yaw here | path-vs-lane | ribbon off lane centre, near → far (lane widths) |
|---|---|---|---|
| v8 constant | −6.40 | **−1.75°** | −0.175 → −0.340 |
| v11 cowl | −6.11 | −1.40° | −0.156 → −0.291 |
| **v12 cowl + outcome level** | **−4.87** | **−0.16°** | **−0.085 → −0.106** |

The residual offset is now the same near and far (≈0.3 m left): a constant position in the lane, i.e.
where the car actually was, not a rotation. Render: `out_v12`, 2216/2216 frames, track applied per frame.

⚠️ Validated mainly on 5–40 s. After ~40 s there are almost no prediction/outcome pairs, so v12 there is
the cowl track plus a constant (+0.13°) — v11 behaviour.
