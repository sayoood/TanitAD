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
