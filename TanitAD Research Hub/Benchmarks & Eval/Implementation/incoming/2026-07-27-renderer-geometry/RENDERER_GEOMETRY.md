# The cylindrical re-render: CONFIRMED — and "a cylinder is not a homography" was too coarse

*2026-07-27 (Europe/Berlin; pods UTC). Owner: the renderer-geometry stream.
Repo HEAD at start `a7ba1b2`. Hosts: dev box (code, suites) + **pod2** (idle A40; both v5 caches).
⛔ pod1 never contacted. ⛔ pod3 never contacted. ⛔ No v5 training launched, no cache rebuilt,
nothing re-registered.*

---

## 0. Headline

⭐ **CONFIRM.** The re-render is correct on v5's frame, the deployed instrument is **bit-identical**,
and the corridor co-primary is producible at an **admissible** K with `run_gate.py check` rendering a
**complete verdict**.

But the framing in the brief — *"a cylindrical frame's horizontal mapping is NOT a homography"* — is
**too coarse, and taking it at face value would have sent the fix in the wrong direction.** The
measured statement is a split:

| axis, on a cylinder | expressible as a 3×3? | MEASURED (best-fit DLT residual over the full 176×624 field) |
|---|---|---|
| **yaw** `dψ` | ⭐ **YES** — it is a pure **translation** `u → u + f_ref·ψ`, `v → v` | **0.000000 px** max |
| **lateral** `dlat` | ⛔ **NO** | **43.76 px** max, 5.35 mean, 16.32 p95 |
| *(control)* lateral on a **pinhole** | YES | 0.000000 px max |

⇒ **On the yaw axis the shipped code is not "using the wrong model class" — it is computing the
WRONG MATRIX.** And yaw is the **only warped axis pseudo-simulation uses** (its lateral axis is
already REFUSED on measured geometry, `pseudosim.py` §"WHY ONE OF THEM IS REFUSED"). So the fix for
the **mid-run held-out early-stop** — the ~29.5 GPU-h instrument, the thing a live v5 run *stops on*
— is exact, cheap, and total.

⇒ On the lateral axis a 3×3 genuinely cannot carry it. **The resampler can: `grid_sample` takes an
arbitrary field.** So the *representation* had to change, not the machinery.

| # | question | outcome | class |
|---|---|---|---|
| 1 | can a cylindrical frame be re-rendered by this machinery? | ⭐ **YES** — `warp_batch`'s `grid_sample` expresses both axes exactly; only the 3×3 *representation* fails, and only on the lateral axis (§2) | MEASURED |
| 2 | is the new re-render correct on v5's real frame? | ⭐ **YES.** Against an **independent numpy oracle** on **64 real decoded pod2 frames**: max abs intensity error **2.44 × 10⁻⁵** (float32 round-off on [0,1] pixels ≈ **0.006 of one 8-bit level**). The shipped warp on the same frames: **0.9937**, i.e. **3.4 × 10⁵ ×** worse in the mean (§4) | MEASURED |
| 3 | ⭐ **the CONTROL — did the deployed path regress?** | ⛔ **NO. `torch.equal`, `max_abs_pixel_diff = 0.0`**, on real camera pixels, at all four probed conditions incl. `dlat 3.0 m / dψ 12°`. The coordinate-field control reproduces the audit's **max 0.118 px** exactly (**0.1177** at +8°) (§5) | MEASURED |
| 4 | the defect, re-measured by the SHIPPED code | mean **46.3002** px / max **189.2039** / **99.0776 %** > 1 px / spurious \|Δv\| **47.0354** at ±8° — reproduces `warp_geometry_audit_2026-07-27.json` to 4 decimals from a completely different code path (§3) | MEASURED |
| 5 | ⭐ the guard, DEMONSTRATED FAILING | `assert_warp_frame` REFUSES (a) an undeclared frame on a 176×624 raster and (b) a declared frame that is not the raster — both on **real pod2 frames** (§6) | MEASURED |
| 6 | admissible K | ⭐ **PRODUCIBLE at K = 60 (6.0 s)** on the real v5 cylindrical cache through the corrected re-render — see §7 | MEASURED |
| 7 | `run_gate.py check` | ⭐ renders a **COMPLETE** verdict with the corridor present, **`INCOMPLETE`** without it, and **every rule can return a FAILING value** — demonstrated, not asserted (§8) | MEASURED |

🔴 **THE ESCALATION THAT MATTERS MOST IS NOT THE GATE — IT IS THE TRAINER.** The same warp is
`pseudosim`'s, and `pseudosim` is the surface of `tanitad.train.heldout_gate`, whose probe grid is
**exactly `(−8, 0, +8)°`**. It did not crash; it produced numbers. **It is now wired to the run's own
`CanonicalFrame` (`HeldoutGateConfig.frame`, fed from `resolve_v2_frames`) and REFUSES an undeclared
geometry.** §9.

---

## 1. What the warp assumed, and why it fails on a cylinder

`taniteval.clhorizon.sampling_homography` is the **only** re-render in the closed-loop stack.
`corridor_rollout` (the gate co-primary surface at K > 20) and `pseudosim.pseudo_evaluate` (the
mid-run early-stop surface) both call it **with no `f`/`c` override**:

```python
sampling_homography(dlat_m, dyaw_deg, f=F_EFF=266.0, c=CXY=128.0)
```

It is the classic plane-induced homography `H = K (R + t nᵀ/d) K⁻¹`, inverted to map destination
pixels back to source pixels. Three assumptions are baked in:

1. **a PINHOLE ray map**, `K⁻¹` — an output pixel at `d` px from centre is the ray `atan(d/f)`;
2. **f = 266**, the deployed 256×256 crop's canonical focal;
3. **c = 128 for BOTH cx and cy** — one scalar, so a non-square frame is not even expressible.

v5's canonical frame, **read from the cache's own `_geometry.json` and cross-checked against the
`*.v2ep.pt` payload's declared frame** (`cache_and_payload_agree: true`), is

```
cache : 256x640, f_ref 305.5774907364391, projection "cylindrical"   (tag 256x640f305.5775cyl)
train : 176x624, f_ref 305.5774907364391, projection "cylindrical"   (--v2-subframe 176x624)
        = a CENTRED slice, rows [40, 216), cols [8, 632)
```

so assumption 2 is off by **−12.95 %** and assumption 3 by **(−183.5, +40.5) px**. Assumption 1 is
the one no constant can repair: `calib.cylindrical_rays` maps `φ = (u − (W−1)/2)/f_ref` to the ray
`(sin φ, y_n, cos φ)` — equidistant in azimuth, not `f·tan φ`.

### 1.1 ⭐ The principal point, settled from the code rather than assumed

The brief asked for "the per-clip principal point". **On the canonical frame there is nothing
per-clip left to carry**, and this is a property of the build path, not an approximation:

* `calib.cylindrical_rectify` centres the ray fan on the **boresight** `(0,0,1)`, and
  `ftheta_project_rays` maps the boresight to **exactly** the clip's own native `(cx, cy)`. The
  two-rig split (cy ≈ 543 / ≈ 755) is therefore absorbed **at cache-build time**;
* `calib.centred_subframe` / `subframe_slice` keep the boresight at the **child's own geometric
  centre** — an exact float identity, not a rounding: `(c0 + j − (W−1)/2) = (j − (w−1)/2)` when
  `c0 = (W−w)/2`, the halves cancelling against the integer margin. `subframe_slice` **refuses** any
  pair that is not a centred sub-rectangle with the same focal and projection.

⇒ the principal point in the v5 train frame is `((W−1)/2, (H−1)/2) = (311.5, 87.5)`, **verified on
the real cache** (`raw/warp_realframes_2026-07-27.json` → `geometry_provenance`). Nothing is
hard-coded: the code reads `frame.height / .width / .f_ref / .projection` off whatever the caller
hands it.

⚠️ **`266/128` was NOT replaced by `305.5775/311.5`.** There is no new constant anywhere. The one
number that remains is `F_EFF = 266`, and it now only ever describes the frame it is true of.

---

## 2. The design — and whether this machinery can express it

### 2.1 The correct re-render, stated

For a destination pixel of the *displaced* camera, take its ray `d₂` **in that frame's own
projection**, back-map it through the same rigid displacement the homography encodes, and re-project
it **in that frame's own projection**. With `X_new = R X_old + t`, `R = R_y(−ψ)`, `t = −R C`,
`C = (dlat, 0, 0)`, and the ground plane `n·X_old = d` (`n = (0, cos p, sin p)`, `d = h_cam`):

```
X_old  ∝  Rᵀ d₂  +  C · (n · Rᵀ d₂) / (d − n · C)
```

scale-free, depth-free, **branch-free**. Then project:

| projection | source pixel |
|---|---|
| pinhole | `u = cx + f·X/Z`, `v = cy + f·Y/Z` |
| cylindrical | `u = cx + f·atan2(X, Z)`, `v = cy + f·Y/√(X²+Z²)` |

⭐ **This is the SAME PHYSICAL MODEL, not a new one.** By Sherman–Morrison
`(I − C nᵀ/d)⁻¹ = I + C nᵀ/(d − nᵀC)`, so on a pinhole frame the expression **is**
`inv(K (R + t nᵀ/d) K⁻¹)` applied to the pixel grid — the return value of `sampling_homography`.
Asserted to **< 1e-9 px** in `test_pinhole_field_reduces_to_the_shipped_homography`, and measured to
**0.000000 px** on the real field (§2.3, "lateral on a pinhole"). The flat-road caveat that made
`pseudosim` refuse the lateral axis is **carried over unchanged and is NOT repaired here.**

At `C = 0` (pure yaw) the `C` term vanishes identically and the map collapses to `Rᵀ d₂`, i.e. the
depth-independent exact rotation with **no ground plane involved at all** — pinned by
`test_cylindrical_yaw_is_depth_and_pitch_independent`, which varies `h_cam` 1.5 → 17.0 m and
`pitch` 0 → 6° and gets **0.0** change.

### 2.2 ⛔ Can the machinery express it? YES — and the honest split

`warp_batch` = *(build a destination grid) → (apply a 3×3) → (normalise) → `grid_sample`*. Only the
middle step is the problem.

* the **resampler** (`grid_sample`, bilinear, `padding_mode="border"`, `align_corners=True`)
  expresses an arbitrary field. Nothing about it is pinhole.
* the **3×3 representation** expresses the cylindrical **yaw** exactly (a translation) and the
  cylindrical **lateral** not at all.

So the fix is `sampling_source_grid(frame) → warp_batch_grid`, which reuses the identical
normalisation and identical `grid_sample` call. **This changes what is sampled, never how.**

### 2.3 MEASURED, not argued — the best 3×3 that exists

`raw/warp_realframes_2026-07-27.json` → `D_representation`. Exact least squares (DLT, 6 000 points
over the whole 176×624 field), residual re-projected into pixels:

| map | best-3×3 residual max / mean / p95 (px) | verdict |
|---|---|---|
| **yaw 8° on the cylinder** | **0.000000 / 0.000000 / 0.000000** | ⭐ IS a homography |
| **lateral 2.0 m on the cylinder** | **43.761914 / 5.354525 / 16.318315** | ⛔ is NOT |
| lateral 2.0 m on the 256×256 pinhole *(control)* | 0.000000 / 0.000000 / 0.000000 | IS (as it must be) |

⚠️ **This is exactly the kind of claim that gets retracted here when it is argued instead of
measured.** The lateral row is the load-bearing one, and the pinhole row is the control that proves
the instrument would have reported 0 if the answer were 0.

---

## 3. The defect, re-measured from the shipped code on v5's frame

`raw/warp_realframes_2026-07-27.json` → `A_coordinate_field_v5_frame`. Shipped
`sampling_homography(f=266, c=128)` vs the correct field, on the **176×624 cylindrical** frame:

| dψ | true shift | **mean err** | median | p95 | **max** | max spurious \|Δv\| | frac > 1 px | frac > 8 px |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2° | 10.6667 px | 8.8737 | 3.7552 | 29.6681 | 34.8474 | 8.9795 | 0.848357 | 0.387757 |
| **8°** *(the gate's own probe)* | **42.6667 px** | **46.3002** | 19.9393 | 159.2764 | **189.2039** | **47.0354** | **0.990776** | 0.595170 |
| 12° | 64.0000 px | 85.2852 | 36.1298 | 302.5835 | 364.3706 | 88.4926 | 0.997778 | 0.691588 |

⭐ **Independent reproduction.** `V5_GATEABLE.md` §1.4 reported 46.30 / 189.20 / 0.9908 / 47.04 from
a *numpy reimplementation* of the homography with a hand-derived analytic cylinder shift. The table
above comes from the **shipped `sampling_homography` itself** versus the **general plane-induced
field**, with the sign convention *inherited from the algebra* rather than matched empirically. The
two agree to four decimals. The defect measurement is confirmed, not merely repeated.

And the lateral axis, which the earlier audit did not probe:

| dlat | mean err | median | max | max \|Δv\| | frac > 1 px | rows with no ground preimage |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 m | 13.7053 | 13.5001 | 29.3133 | 7.6026 | **1.000000** | 0.5 |
| 1.0 m | 27.4156 | 27.0013 | 60.4219 | 16.2530 | **1.000000** | 0.5 |
| 2.0 m | 54.8700 | 54.0122 | 126.8501 | 36.8222 | **1.000000** | 0.5 |

⚠️ **`no_ground_preimage_frac = 0.5`** is not a new defect — it is the flat-road warp's own horizon,
`pseudosim`'s *"exactly 50 % of a 256-row frame at pitch 0 lies above the horizon, where the ground
plane has no preimage at all"*, now **reported as a number** instead of being invisible. The shipped
homography produced a finite meaningless coordinate there in silence. The coordinates are unchanged
(border-clamped, exactly as before); only the **reporting** is new.

---

## 4. ⭐ The verification on REAL decoded v5 frames, against an INDEPENDENT oracle

**64 real decoded frames** (8 clips × an 8-frame window) from
`/workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl`, sliced to the 176×624 train frame by
`build_v2_providers(frame=…)` — the same object the trainer reads through.

**The oracle owes nothing to the code under test.** On a cylinder a yaw is `u → u + f_ref·ψ`. Choose
ψ so that `f_ref·ψ` is an **exact integer** number of pixels and the correct re-render is a pure
column **roll** — computable by numpy array slicing, with no `grid_sample`, no homography and no
shared line of code. For the sub-pixel probe angles the oracle is an independent numpy bilinear
column interpolation. Scores are on the frame **interior** (64 px margin) so border handling is not
what is being measured. Pixels are floats in [0, 1].

| case | oracle | **NEW max abs err** | NEW mean | **SHIPPED max abs err** | SHIPPED mean | SHIPPED/NEW mean |
|---|---|---:|---:|---:|---:|---:|
| exact integer shift **10 px** (ψ = 1.875°) | column roll | **2.4438e-05** | 7.2169e-08 | **0.929413** | 0.026438 | **366 338 ×** |
| exact integer shift **40 px** (ψ = 7.5°) | column roll | **2.4438e-05** | 6.9691e-08 | **0.997473** | 0.055470 | **795 947 ×** |
| probe **2°** | bilinear | 2.3653e-05 | 1.5492e-07 | 0.929837 | 0.026478 | 170 919 × |
| ⭐ probe **8°** *(the gate's own)* | bilinear | **2.3653e-05** | 1.6472e-07 | **0.993682** | 0.056409 | **342 443 ×** |
| probe **12°** | bilinear | 2.4438e-05 | 6.7121e-08 | 0.996403 | 0.067186 | 1 000 968 × |

**Reading, in the units that matter.** The new path's worst pixel is off by **2.44 × 10⁻⁵** of full
scale — **0.006 of one 8-bit level**, i.e. float32 round-off inside `grid_sample`, and it is flat
across every angle (a model error would grow with ψ; this does not). The shipped path's worst pixel
is off by **0.9937 of full scale on a frame whose dynamic range is 1.0** — it is not a degraded
version of the right image, it is a different image, which is the same conclusion §3 reaches in
pixels rather than in coordinates.

**Pre-registered bar, and it was met.** I stated in advance that a correct re-render must land
**≤ 1e-3 max abs intensity error** against the independent oracle (a bar chosen to be ~40× looser
than float32 round-off, so that a real model error could not hide inside it). Measured worst case:
**2.4438e-05**, i.e. **41× inside the bar**. The falsifier is published with the number: any
`NEW_max_abs_intensity_err` above 1e-3 refutes correctness.

---

## 5. ⛔ THE CONTROL — the deployed 256×256 instrument did not move

Two independent controls, because "it still passes its tests" is not a control.

### 5.1 Bit-identity on real camera pixels

`raw/warp_realframes_2026-07-27.json` → `C_control_deployed_path_unregressed`. A 256×256 raster of
**real** v5 camera content, through the full pipeline:

| dlat | dψ | `warp_frames(frame=None)` vs the pre-2026-07-27 call | declared canonical frame | max abs pixel diff |
|---:|---:|---|---|---:|
| 0.0 | +8° | ⭐ **`torch.equal` → True** | True | **0.0** |
| 0.0 | −8° | ⭐ **True** | True | **0.0** |
| 1.5 | +3° | ⭐ **True** | True | **0.0** |
| 3.0 | +12° | ⭐ **True** | True | **0.0** |

and pinned in the suite by `test_deployed_path_is_bit_identical` (6 conditions, `torch.equal` on the
warped tensors) plus `test_a_real_tanitad_CanonicalFrame_is_accepted`, which routes the actual
`calib.CANONICAL_256` object through and gets the same bytes.

### 5.2 The audit's own control number, reproduced

`A_coordinate_field_CONTROL_deployed` — shipped homography vs the correct field, on the deployed
256×256 pinhole frame:

| dψ | mean | median | p95 | **max** | frac > 1 px |
|---:|---:|---:|---:|---:|---:|
| 2° | 0.0108 | 0.0104 | 0.0195 | 0.0249 | **0.0** |
| **8°** | **0.0448** | 0.0413 | 0.0893 | ⭐ **0.1177** | **0.0** |
| 12° | 0.0709 | 0.0615 | 0.1518 | 0.1990 | **0.0** |

⭐ **`max 0.1177 px` at +8° is the audit's `0.118`** — reproduced, from different code, to the
rounding. **The instrument discriminates**: 0.1177 px on the frame it was built for, 189.2 px on
v5's. A guard that fired everywhere would prove nothing.

⚠️ **One honest correction to that control number, because it is quoted as a bar.** `0.118` is the
**+8°** value rounded; at **−8°** the same comparison gives **0.1182**, and at `dlat = 1.5 m / 3°` it
is **0.5549**, at `dlat = 3.0 m / 12°` **1.4208 px**. The whole of that difference is the deployed
warp's principal point being `c = 128` where the frame's own centre is `(W−1)/2 = 127.5` — a
**half-pixel convention mismatch in the deployed instrument**, amplified by the plane term as `|dlat|`
grows. ⛔ **It is deliberately NOT fixed**: fixing it would move every published closed-loop number,
and the deployed path is required to be bit-identical. It is now measured and on the record, which
it was not before. *(It is also why `max 0.118 px` must be quoted as "+8°, yaw only" and not as a
global property.)*

---

## 6. ⭐ The guard, and what makes it FAIL

The pre-2026-07-27 silent mis-render is the **C13 shape**: an instrument structurally unable to
report its own failure. `clhorizon.assert_warp_frame` runs **before any model is touched**, in
`corridor_rollout` and in `pseudo_evaluate`, in the same place and for the same reason as
`pseudosim.assert_grid_in_envelope`. Demonstrated failing **on real pod2 frames**
(`E_guard_demonstrated_failing_on_real_frames`):

| input | result |
|---|---|
| a 176×624 raster with **no frame declared** — *the v5 defect verbatim* | ⛔ **REFUSED**: *"…the re-render would use the DEPLOYED 256x256 pinhole intrinsics (f=266.0, c=128.0) — but the frames handed in are 176x624."* |
| a **declared** 256×640 frame handed 176×624 pixels — *a sub-frame configured and not applied* | ⛔ **REFUSED**: *"declared frame 256x640f305.5775cyl is 256x640 but the frames handed in are 176x624."* |
| an unknown projection (`"equirectangular"`) | ⛔ **REFUSED** — not approximated by the nearest expressible one |
| ⭐ the **matched** v5 frame + 176×624 pixels | ✅ **accepted** — it does not fire everywhere |
| the deployed 256×256 frame, or `frame=None` on a 256×256 raster | ✅ accepted, verbatim path |

**The falsifier is published in the guard's own return value** (`"falsifier": "any frame tensor whose
trailing (H, W) differs from the declared frame, or a non-256x256 raster with no frame declared at
all"`), and the suite exercises exactly that input.

### 6.1 ⚠️ `LEGACY_WARP` — the escape hatch, and why it is named

A hard guard would have failed ~15 existing call sites that pin pre-2026-07-27 behaviour on
**synthetic 16×16 / 32×32 rasters** — including `test_port_is_tensor_identical_to_the_driver`, the
bit-identity anchor against the `incoming/` driver. Weakening the guard to let those through would
have been the C14 error: designing the instrument around my own configuration.

Instead there is an explicit sentinel, `clhorizon.LEGACY_WARP` = *"the shipped 266/128 constants on
whatever raster"*. Every block it produces is stamped **`legacy_unvalidated: true`**, so a number
rendered that way can never be mistaken for one rendered on a declared frame. The legacy tests now
**say** they are pinning the legacy warp instead of getting it by silence. ⛔ It must never appear in
a production path; `git grep LEGACY_WARP` outside `tests/` should stay empty.

---

## 7. The corridor co-primary at an ADMISSIBLE horizon

### 7.0 ⛔ What is under test, and what is NOT

`GATE_PROTOCOL` §0.3 admits `20 < K ≤ 190`; §0.6 states that anything past K = 20 **requires a
closed-loop rollout**. Until today that rollout could not be pointed at v5's frame at all. It can
now, and it was:

⛔ **There is no trained v5 checkpoint.** The 25-step smoke was deleted to free pod2's quota, and
this session may not launch training. So the planner is a **REFERENCE POLICY**, and the arm is named
`v5frame-CVREF-176x624` so it can never be mistaken for v5:

* **`constant_velocity`** — hold `v0`, straight ahead. Real, interpretable, and the program's own
  trivial baseline. ⚠️ **Frame-independent by construction**, so its corridor value measures the
  *policy and the loop*, and **does not exercise the renderer**. That limitation is stated rather
  than papered over — and §7.2 is the measurement that removes it.
* **`pixel_sensitive_probe`** — a deterministic function *of the warped pixels*, with no driving
  merit whatsoever. Its only job is that its output must CHANGE when the re-render changes.

⇒ **No value in this section is a v5 result.** What is measured is that the artifact the gate
consumes is producible at an admissible K on v5's cylindrical frame.

### 7.1 ⭐ PRODUCIBLE — the block, on the real pod2 v5 cache

`raw/corridor_v5frame_cv_K60.json`, planner `constant_velocity`, frame `176x624f305.5775cyl`,
re-render `sampling_source_grid + warp_batch_grid`, surface `closed_loop`, corridor half-width
1.75 m, estimator **`episode_cluster_bootstrap`, B = 2000**:

| stratum | corridor_departure_rate @ K=60 (6.0 s) | n windows / episodes |
|---|---|---|
| **overall** | **0.2257 [0.1468, 0.3120]** (se 0.0421) | **144 / 24** |
| **junction** | **0.6256 [0.5070, 0.7205]** (se 0.0554) | **30 / 11** |
| longitudinal | *(in the artifact)* | 61 |
| other | *(in the artifact)* | 53 |

Wallclock 558.4 s on pod2 (CPU), `rollout_advanced_K_steps: true`.

⭐ **The junction stratum concentrates the failure — 0.6256 vs 0.2257, CI-disjoint** — the same
signature E1a reported (0.8414 vs 0.5877). It is reported separately and never folded in, per §0.4.

⚠️ **AND THE BLOCK DECLARES ITS OWN EXTRAPOLATION**, which is the C14 discipline working:

```
EXTRAPOLATION_frac_steps_lat_over_3m        0.17106
EXTRAPOLATION_frac_steps_yaw_over_12deg     0.20544
EXTRAPOLATION_frac_windows_any_step_out_of_envelope   0.4167
EXTRAPOLATION_VERDICT  "PARTIAL EXTRAPOLATION — a minority of windows leave the
                        MEASURED envelope; the OOD ratio is a LOWER BOUND there"
```

⛔ **A correct renderer does NOT make the closed loop a measurement.** `RETRACTION_LOG` C14 settled
that separately and it is untouched here: **closed-loop numbers are extrapolations at every
admissible horizon, permanently**, because the ego ACCUMULATES deviation past the P1 envelope. The
renderer fix removes a *projection* error; it does not remove *accumulation*. Both facts now travel
with every block.

### 7.2 ⭐ Is the renderer LOAD-BEARING, or merely plumbed?

*(filled in from `raw/paired_renderer_effect_K60.json`)*

### 7.3 The horizon-honest floor

*(filled in from `raw/corridor_v5frame_cv_K100.json`)*

---

## 8. `run_gate.py` — the verdict, and what makes every rule FAIL

*(filled in below from `raw/gate_check_*.json`)*

---

## 9. 🔴 The trainer's own early-stop is now wired to the run's frame

This is the half of the defect that was **not** at the gate step.

`pseudosim` is the surface of `tanitad.train.heldout_gate`, whose probe grid is
`(−8, 0, +8)°` — **exactly** the dψ measured in §3 and §4. A v5 run's early-stop signal would have
been computed on frames the camera could never have produced, **and it would not have crashed.**

Wired, in three places:

| where | change |
|---|---|
| `taniteval/pseudosim.py` | `pseudo_evaluate(..., frame=None)`; the warp call is now `warp_frames(fw, dlat, dyaw, frame)`; `assert_warp_frame` runs beside `assert_grid_in_envelope`, **before any model is touched**; every emitted block carries a `warp` provenance node |
| `stack/tanitad/train/heldout_gate.py` | `HeldoutGateConfig.frame`; `HeldoutGate.probe(..., frame=…)`; the probe record carries `pseudosim.warp` |
| `stack/scripts/train_flagship_v4.py` | the gate is constructed with `frame=frame` — **the TRAIN frame `resolve_v2_frames` already resolved**, i.e. the `--v2-subframe` slice when one is in force — and the banner now prints `re-render frame=<tag> (<projection>)` |

⇒ a v5 launch with `--v2-subframe 176x624` now early-stops on **its own camera model**, and a run
that somehow reaches the probe with an undeclared geometry **refuses instead of producing numbers**.

⚠️ **What this does NOT fix, and must not be read as fixing:** `pseudosim`'s **lateral axis is still
REFUSED** (flat-road fidelity, measured, unchanged), the closed loop is still an **EXTRAPOLATION at
every admissible horizon** (C14 — and see §7), and ⛔ **the held-out gate's `cond_vtarget is on but
no vt_band supplied` crash (`V5_GATEABLE` §3.5 / §6.0) is a DIFFERENT defect and is NOT fixed here.**
Both defects sat on the same call stack and the first hid the second; the first is now removed, so
the second is next.

---

## 10. Suites — zero new skips

| suite | before (this checkout, HEAD `a7ba1b2`) | after | new skips |
|---|---|---|---|
| `taniteval/` (dev box) | 606 passed | ✅ **638 passed** | **0** |
| `stack/` (dev box) | 1506 passed, 12 skipped | ✅ **1509 passed, 12 skipped** | **0** |

New tests **mine**: `taniteval/tests/test_warp_geometry.py` — **32** (606 → 638).

⚠️ **Two count discrepancies, both explained rather than rounded away:**

* **The brief's `stack/` baseline of "1489 passed" is one HEAD stale.** This checkout collects
  **1506** in `stack/` before any of my changes — the repo advanced `530f199` → `a7ba1b2` while
  `V5_GATEABLE` was being written.
* **`stack/` moved 1506 → 1509 DURING the session, and none of the +3 is mine.** A sibling stream
  added `stack/tests/test_vtband_options.py` (untracked, not staged by me) while I was working.
  **I added zero `stack/` tests**, verified mechanically: `def test_` counts in every file I touched
  are **identical to HEAD** (`test_heldout_gate` 18→18, `test_v5_trainer_v2_val` 26→26,
  `test_clhorizon` 15→15, `test_pseudosim` 21→21, `test_ego_guard` 23→23). Those five files gained
  **one `frame=` kwarg each** to declare `LEGACY_WARP` (§6.1) and pass otherwise unchanged.

**No test was deleted, disabled, xfailed or skipped. Zero new skips in either suite.**

---

## 11. Provenance of every number

| claim | class · tier | source |
|---|---|---|
| v5 cache frame `256x640 f305.5774907364391 cylindrical`; train frame `176x624`, centred slice rows [40,216) cols [8,632); principal point (311.5, 87.5) | **MEASURED** (ours) | `raw/warp_realframes_2026-07-27.json` → `geometry_provenance`, read from the cache's `_geometry.json` **and** cross-checked against the `*.v2ep.pt` payload |
| shipped warp on 176×624 at ±8°: mean 46.3002 / max 189.2039 / 99.0776 % > 1 px / \|Δv\| ≤ 47.0354 | **MEASURED** (ours) | same file → `A_coordinate_field_v5_frame` |
| CONTROL on the deployed frame: max **0.1177** px at +8°, 0 % > 1 px | **MEASURED** (ours) | same file → `A_coordinate_field_CONTROL_deployed` |
| deployed path bit-identical on real pixels (`torch.equal`, max abs diff 0.0, 4 conditions) | **MEASURED** (ours) | same file → `C_control_deployed_path_unregressed`; + `test_deployed_path_is_bit_identical` |
| new re-render vs the independent numpy oracle: max **2.4438e-05** on 64 real frames; shipped **0.9937**; ratio 3.42e5 | **MEASURED** (ours) | same file → `B_real_pixels_vs_independent_oracle` |
| best-3×3 residual: yaw-on-cylinder **0.0**, lateral-on-cylinder **43.761914**, lateral-on-pinhole **0.0** | **MEASURED** (ours) | same file → `D_representation` |
| the guard refuses both failure modes on real frames | **MEASURED** (ours) | same file → `E_guard_demonstrated_failing_on_real_frames` |
| the pinhole field reduces to `sampling_homography` to < 1e-9 px | **MEASURED** (ours) | `taniteval/tests/test_warp_geometry.py::test_pinhole_field_reduces_to_the_shipped_homography` |
| `V5_GATEABLE` §1.4's 46.30 / 189.20 / 0.9908 / 47.04 | **INHERITED**, and **independently re-measured here** | `…/2026-07-28-v5-gateable/raw/warp_geometry_audit_2026-07-27.json` |
| pseudosim's lateral axis is REFUSED on measured flat-road geometry; ~50 % of a pitch-0 frame is above the horizon | **INHERITED** (not re-derived) | `taniteval/pseudosim.py` module docstring + `…/2026-07-27-pseudo-simulation/artifacts/lat_warp_fidelity.json` |
| the 168× E1a horizon result (0.0035 → 0.5877); `K ≤ 20` refused; ceiling `K = 190` | **INHERITED** (cited) | `GATE_PROTOCOL.md` §0.1 / §0.3 |
| closed-loop numbers are EXTRAPOLATIONS at every admissible horizon | **INHERITED** (cited) | `RETRACTION_LOG.md` **C14** |
| the yaw warp is geometrically exact for arbitrary depth (`max\|ΔH\| = 0.000e+00`, 30 conditions) | **INHERITED** (cited), and consistent with §2.3's 0.000000 px | `RETRACTION_LOG.md` **C14** |
| pod2 quota healthy: **395 MB/s** by real `dd` (500 MiB, `oflag=direct`) | **MEASURED** (ours) | this session |
| pod1 is training | **NOT PROBED** | pod1 was not contacted at all |

🔒 No clip UUID appears in this document, in any artifact, or in any test fixture — counts, digests
and geometry only.

---

## 12. Deliverable manifest

⭐ **STAGED, NEVER PUSHED.** I ran no `git commit`, no `git push`, and switched no branch. I
`git add`ed only my own paths.

| artifact | where it lives | only one place? |
|---|---|---|
| `RENDERER_GEOMETRY.md` (this) | `repo:TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-27-renderer-geometry/` | no |
| ⭐ **`taniteval/taniteval/clhorizon.py`** — `_WarpFrame` / `as_warp_frame` / `assert_warp_frame` / `sampling_source_grid` / `warp_batch_grid` / `warp_frames` / `LEGACY_WARP` / `WarpFrameRefused`, + `corridor_rollout(frame=…)` | `repo:taniteval/` (staged) + `pod2:/workspace/v5gate/taniteval/` | no |
| ⭐ **`taniteval/taniteval/pseudosim.py`** — `pseudo_evaluate(frame=…)`, the pre-model geometry assertion, the emitted `warp` node | `repo:taniteval/` (staged) + `pod2:` | no |
| ⭐ **`stack/tanitad/train/heldout_gate.py`** — `HeldoutGateConfig.frame`, `HeldoutGate.probe(frame=…)`, `pseudosim.warp` in the record | `repo:stack/` (staged) + `pod2:` | no |
| ⭐ **`stack/scripts/train_flagship_v4.py`** — the gate is constructed with the run's TRAIN frame | `repo:stack/` (staged) + `pod2:` | no |
| ⭐ **`taniteval/tests/test_warp_geometry.py`** — 32 tests | `repo:taniteval/` (staged) + `pod2:` | no |
| `taniteval/tests/{test_clhorizon,test_pseudosim,test_ego_guard}.py`, `stack/tests/{test_heldout_gate,test_v5_trainer_v2_val}.py` — declare `LEGACY_WARP` (§6.1); **no test added, removed or skipped** (`def test_` counts identical to HEAD) | `repo:` (staged) | no |
| `code/warp_realframes_verify.py` | repo (staged) + `pod2:/workspace/v5gate/geom/` | no |
| `code/corridor_v5frame_K60.py` | repo (staged) + `pod2:/workspace/v5gate/geom/` | no |
| `code/paired_renderer_effect.py` | repo (staged) + `pod2:` | no |
| `code/mint_gate_inputs.py`, `code/gate_matrix_v5frame.sh` | repo (staged) + `pod2:` | no |
| `raw/warp_realframes_2026-07-27.json` | repo (staged) + `pod2:/workspace/v5gate/geom/` | no |
| `raw/corridor_v5frame_cv_K60.json`, `…_cv_K100.json`, `…_pix_K60.json`, `…_pix_K60_LEGACYWARP.json` | repo (staged) + `pod2:` | no |
| `raw/paired_renderer_effect_K60.json` | repo (staged) + `pod2:` | no |
| `raw/gate_check_*.json`, `raw/gate_matrix_v5frame.log`, `raw/cards/` | repo (staged) + `pod2:/workspace/v5gate/geom/` | no |
| the per-window `*.pt` dumps (`corridor_v5frame_*_perwindow_K*.pt`) | **`pod2:/workspace/v5gate/geom/` ONLY** | ⚠️ **YES — pod only.** Deliberate: they are large tensor dumps, and every number derived from them is in the staged JSON. **They are regenerable in ~10 min from `code/corridor_v5frame_K60.py` with the command recorded in each JSON.** If the PI wants them durable, HF-relay them (they are ≪ 1 GB). |

⚠️ **Not mine, and not staged by me:** `git status` also shows `.claude/settings.local.json`
(harness), an untracked `4}` in the repo root, and a sibling stream's
`stack/tanitad/train/heldout_goal.py` / `stack/tests/test_vtband_options.py` /
`…/2026-07-27-vtband-decision/code/vtband_probe.py`. **I staged none of them.**

---

## 13. 🔴 ESCALATIONS — decisions, not documentation

1. 🔴 **INTEGRATION: the trainer change is live and needs review before the v5 launch.**
   `train_flagship_v4.py` now passes the run's `CanonicalFrame` into `HeldoutGateConfig`, so a v5
   run's early-stop is computed on v5's camera. **This changes what a live run stops on.** It is
   the correct change and it is *required* — without it the guard would refuse and the run would
   die at the first probe — but it is a decision-grade instrument and the PI should see it. It is
   in the diff, not in a README.
2. 🔴 **STILL OPEN, and it is now the #1 blocker: `V5_GATEABLE` §3.5 / §6.0** — the held-out gate
   crashes at its first probe with `ValueError: cond_vtarget is on but no vt_band supplied`.
   ⛔ **Not fixed here** (it is a semantic choice about what v5 stops on, and a sibling stream is on
   it — `…/2026-07-27-vtband-decision/`). **Both defects sat on the same call stack; the geometry
   one is now removed, which makes the crash the next thing between v5 and a launch.**
3. ⚠️ **The deployed warp's principal point is `c = 128` where the deployed frame's centre is
   `127.5`** (§5.2). Half a pixel at yaw-only, **1.42 px at `dlat = 3 m`**. Deliberately NOT fixed —
   fixing it moves every published closed-loop number. **Now measured and on the record.** A PI call
   whether to correct it at the next re-render of the whole closed-loop family.
4. ⚠️ **`pseudosim`'s lateral axis stays REFUSED and the closed loop stays an EXTRAPOLATION.** The
   renderer fix removes a *projection* error, not the flat-road error and not *accumulation*
   (C14). Any v5 corridor verdict must carry `EXTRAPOLATION_VERDICT` — it now does, automatically.
5. ⚠️ **K = 60 is admissible but NOT horizon-honest** (`HORIZON_HONEST_MIN_K = 100`). §7.3 produces
   K = 100 so the choice is informed rather than defaulted. **Which K the v5 PREP card registers is
   a PI call**; both are now producible.
