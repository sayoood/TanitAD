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
