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
