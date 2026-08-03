# Rolling shutter: quality vs cost — and what the "+35 %" actually was

**STREAM E. MEASURED 2026-08-03 on `tanitad-thor` (Jetson aarch64 Blackwell `sm_110`),
gsplat 1.5.3, scene `00040136-e651-4abd-991d-0655ccda9430`, 12 frames spread over the
599-frame clip** (`0,54,109,163,217,272,326,381,435,489,544,598` — the 5-frame set the
earlier panel used is a subset of this range). Every table cites a run directory.
**grad-NCC is the only admissible metric on this clip**; MAE/PSNR are recorded and decide
nothing.

| run dir / file (on `tanitad-thor`) | what it holds | repo copy |
|---|---|---|
| `~/rq_out/rs_sweep_chosen/report.json` | phase + slice cost curve, 21 arms, DEPLOYED config | `rs_sweep_chosen.report.json` |
| `~/rq_out/rs_frame_offset_k10.json` | ⭐ reference-index scan −10…+10 — **the +6 finding** | `rs_frame_offset_k10.json` |
| `~/rq_out/rs_frame_offset.json` | the first ±3 scan (superseded, kept for the record) | `rs_frame_offset.json` |
| `~/rq_out/rs_cost_probe.json` | interleaved timing + **direct projection-survivor counts** | `rs_cost_probe.json` |
| `~/rq_out/rs_seam_control.json` | seam-vs-geometry control for the slice-count decline | `rs_seam_control.json` |
| `~/rq_out/rs_batch_chosen/report.json` | slice **batching** (n cameras, one call) vs sequential | `rs_batch_chosen.report.json` |
| `~/rq_out/mp4_frame_count.json` | full sequential decode: mp4 **605** vs rig **599** | `mp4_frame_count.json` |
| `~/rq_out/logs/utgate_chosen.log` | phase-extrapolation arms + the gate arm that killed the box | `rs_utgate_chosen.PARTIAL.log` |
| `~/rq_out/rp_rs_check/` | proof `render_probe.py --rolling-shutter` now renders | — |
| `~/rq_out/logs/regression.log` | bit-wise no-regression proof | — |

Repo copies of every `report.json` are beside this file.

---

## RECOMMENDATION

### The question asked: *is there a rolling-shutter setting inside the 10 Hz budget?*

**No — and it does not matter, because a FREE single render BEATS rolling shutter
outright.** RS is offline-only, and after §6 it should not be used for hero videos either.

> ### ⭐ THE RESULT
> | arm | grad-NCC | neg-control | raster ms |
> |---|---|---|---|
> | `g_p1.00` — **production today** | 0.3241 | 11/12 | 36.4 |
> | `native` — gsplat rolling shutter | 0.3451 | 11/12 | **3258.6** |
> | **`g_pm2.00` — one ordinary render, camera placed 2 readouts (61 ms) earlier** | **0.3499** | **12/12 — the only arm in the study to pass on every frame** | **47.6** |
>
> **A free global-shutter render beats native rolling shutter on quality (0.3499 vs
> 0.3451), on the negative-control margin (+0.1153 vs +0.1084), and on cost (~68x
> cheaper)** — and the phase curve was **still rising** at −2.0, so even that is a lower
> bound. The rolling-shutter question was the wrong question: the render is not
> mis-modelled, it is **mis-timed**.

1. **No slice count buys the gain, at any price.** Slicing reproduces the pose sweep
   exactly (**verified bit-exact at n=1**) and buys **nothing**: the best sliced arm (n=4,
   **+0.0048**) ties a **free** single render at the shutter midpoint (**+0.0047**), loses
   to a **free** render at the shutter start (**+0.0072**), and costs 4x. With my
   compositing seams removed (the control in §1) the sweep is a flat, tiny
   **+0.0008…+0.0053** at every n from 2 to 64 — real, but a quarter of native RS and
   beaten by a free single render. Cost is **n × 36.4 ms**, and a 10 Hz tick has ~36 ms
   left after the model's 60.3 ms — **n=2 already misses.**
2. **The gain is not the shutter.** Running the sweep **BACKWARDS** reproduces it exactly
   (**+0.0216** vs **+0.0210**, images 2.26/255 apart); the faithful sweep, done properly
   and seam-corrected, is worth only **+0.003–0.005** — a quarter of it; and native rolling
   shutter rasterises **+77 % more gaussians**, coverage a pose sweep cannot create. Source
   and counts agree: the RS branch **discards the sigma-point validity flag**, so it
   silently disables gsplat's cull.
3. ⛔ **Do not "just relax the gate" either.** It recovers two thirds of the coverage for
   free, but the gaussians it admits have **radii up to 1e6 px on a 1920-px raster** —
   degenerate projections, not geometry. What rolling shutter is really doing is smearing
   a few hundred broken splats over the scene's reconstruction holes. The scene already
   ships a *principled* hole-fill (the `sky-env-map`, deployed at gain 0.3).
4. ⭐⭐ **THE ACTUAL DEFECT, found on the way past and 8.6x bigger than everything above:
   every render in this scene is scored against a video frame SIX FRAMES TOO EARLY.**
   Scanning the reference index over `−10 … +10` gives a clean single peak at **exactly
   +6**, **unanimous on 12 of 12 frames**, worth **+0.1797 grad-NCC (0.3114 → 0.4911,
   +57.7 %)** — and the mp4 has **605** frames to the rig's **599**. **Free.** (§7)

**Net: the render budget should be spent on FRAME ALIGNMENT (+0.1797, free) and on
principled hole-filling — not on shutter physics, which is worth ~+0.004 and is free
anyway, and not on rolling shutter, which is worth +0.021 at 90x and for the wrong
reason.**

---

## 1. The cost/quality curve — `~/rq_out/rs_sweep_chosen`

Config = the deployed one (`background,road,dynamic_rigids,dynamic_deformables` +
`cull=0.95` + gated sky gain 0.3). **Baseline `g_p1.00` is production**: the shutter-END
pose, which is what `gt_cam_to_nre()` returns and what every renderer call has used.

`delta` is paired over the same 12 frames; the interval is a **paired percentile
bootstrap over FRAMES within one scene** — there is no second scene and no episode
structure, so it describes frame-to-frame variability in this clip and **may not be
quoted as a scene-level CI.**

| arm | render calls | grad-NCC | Δ vs prod | 95 % CI (frames) | neg-ctl | margin | mean α | raster ms | wall ms |
|---|---|---|---|---|---|---|---|---|---|
| `g_p0.00` **shutter START** | 1 | 0.3313 | **+0.0072** | **[+0.0026,+0.0124]** | 11/12 | +0.0966 | 0.3339 | 38.6 | 118.9 |
| `g_p0.25` | 1 | 0.3310 | +0.0069 | [+0.0031,+0.0114] | 11/12 | +0.0964 | 0.3367 | 89.1 | 345.6 |
| `g_p0.50` midpoint | 1 | 0.3289 | +0.0047 | [+0.0008,+0.0091] | 11/12 | +0.0977 | 0.3417 | 57.8 | 186.4 |
| `g_p0.75` | 1 | 0.3267 | +0.0026 | [−0.0016,+0.0075] | 11/12 | +0.0937 | 0.3404 | 65.3 | 189.2 |
| **`g_p1.00` = PRODUCTION** | 1 | 0.3241 | 0 | — | 11/12 | +0.0908 | 0.3412 | 65.3 | 190.5 |
| `s1` (self-check) | 1 | 0.3289 | +0.0047 | [+0.0008,+0.0091] | 11/12 | +0.0977 | 0.3417 | 69.6 | 195.4 |
| `s2` | 2 | 0.3266 | +0.0025 | [−0.0017,+0.0076] | 11/12 | +0.0939 | 0.3379 | 140.3 | 418.0 |
| `s3` | 3 | 0.3274 | +0.0033 | [−0.0001,+0.0072] | 11/12 | +0.0947 | 0.3390 | 110.1 | 281.1 |
| `s4` | 4 | 0.3289 | +0.0048 | [+0.0011,+0.0092] | 11/12 | +0.0948 | 0.3322 | 135.8 | 295.4 |
| `s6` | 6 | 0.3264 | +0.0023 | [−0.0018,+0.0072] | 11/12 | +0.0931 | 0.3373 | 218.8 | 463.5 |
| `s8` | 8 | 0.3269 | +0.0028 | [−0.0017,+0.0088] | 11/12 | +0.0940 | 0.3387 | 301.9 | 615.0 |
| `s12` | 12 | 0.3260 | +0.0019 | [−0.0022,+0.0071] | 11/12 | +0.0950 | 0.3334 | 422.1 | 930.5 |
| `s16` | 16 | 0.3246 | +0.0004 | [−0.0039,+0.0055] | 11/12 | +0.0937 | 0.3364 | 582.4 | 1276.7 |
| `s32` | 32 | 0.3218 | −0.0023 | [−0.0057,+0.0011] | 11/12 | +0.0931 | 0.3358 | 1707.4 | 5508.6 |
| `s64` | 64 | 0.3146 | **−0.0095** | **[−0.0129,−0.0055]** | 11/12 | +0.0911 | 0.3363 | 4054.0 | 11911.9 |
| `s16_rev` (direction control) | 16 | 0.3272 | +0.0031 | [−0.0003,+0.0078] | 11/12 | +0.0954 | 0.3388 | 1351.2 | 4016.9 |
| `s16_fixedactor` | 16 | 0.3243 | +0.0002 | [−0.0042,+0.0052] | 11/12 | +0.0934 | 0.3361 | 1444.7 | 4254.7 |
| **`native` (gsplat RS)** | 1 | **0.3451** | **+0.0210** | **[+0.0096,+0.0365]** | 11/12 | +0.1084 | **0.3919** | **3258.6** | 3275.6 |
| `native_swapped` (BACKWARDS) | 1 | **0.3457** | **+0.0216** | [+0.0131,+0.0312] | 11/12 | +0.1082 | **0.3924** | 7426.0 | 7586.7 |
| `native_zero_end` (RS, no motion) | 1 | 0.3180 | −0.0061 | [−0.0130,+0.0000] | 11/12 | +0.0878 | 0.3358 | 155.3 | 323.3 |
| `native_zero_start` (RS, no motion) | 1 | 0.3266 | +0.0025 | [−0.0040,+0.0087] | 10/12 | +0.0932 | 0.3289 | 1693.9 | 1848.8 |

**Reference frame mean = 0.2399.** Reading the curve:

* **The slice sweep is not a lever at any price.** Cost rises 64x; quality falls.
* **The whole honest pose effect is ~+0.007 and it is FREE.** `g_p0.00` beats every
  sliced arm at 1/4 to 1/64 of the cost.
* ⛔ **I RETRACT MY OWN READING OF THE `s64` DECLINE.** The full-frame numbers fall to
  **−0.0095** and I first read that as "the faithful sweep is actively harmful". **It is my
  compositing, not the geometry** — an n-slice composite has n−1 horizontal seams and
  grad-NCC is a *gradient* metric. `rs_seam_control.py` (`~/rq_out/rs_seam_control.json`)
  scores each arm twice, deleting ±3 rows around **that arm's own seams from BOTH the arm
  and the baseline**:

  | n | seams | rows dropped | Δ FULL frame | **Δ SEAM-MASKED** |
  |---|---|---|---|---|
  | 2 | 1 | 6 | +0.0025 | **+0.0026** |
  | 4 | 3 | 18 | +0.0048 | **+0.0053** |
  | 8 | 7 | 42 | +0.0028 | **+0.0043** |
  | 16 | 15 | 90 | +0.0004 | **+0.0031** |
  | 32 | 31 | 186 | −0.0023 | **+0.0031** |
  | 64 | 63 | 378 | **−0.0095** | **+0.0008** |

  **Seam-free, the sweep is a small CONSISTENT POSITIVE (+0.0008…+0.0053) that never turns
  negative.** The corrected statement is *"the pose sweep is real but worth ~+0.003–0.005,
  i.e. a quarter of native RS's +0.021 and no better than a FREE single-pose render"* —
  **not** *"the sweep is harmful"*. ⚠️ Had I not built this control I would have published
  a false mechanism, so §1's `s64` row must be read with it.

### Self-consistency and controls that ran

* ✅ `s1 == g_p0.50` **BIT-EXACT** (`selfcheck_s1_equals_g_p0.50_bitexact: true`) — one
  slice at phase 0.5 is the same image as a global render at phase 0.5, so the band
  mapping and the SE(3) slerp are verified rather than assumed.
* ✅ SE(3) interpolation unit-tested on Thor: 500 random quaternions round-trip, slerp
  yaw is exactly linear in `t`, endpoints exact, `R Rᵀ = I` to 1e-12.
* ⚠️ **Negative control: 11/12 for every arm** (`native_zero_start` 10/12). The single
  shared failure is **frame 163, which fails for all 21 arms including the baseline** —
  it is a property of that frame, not of any arm. The banked 5-frame panel's "PASS 5/5"
  is not contradicted; the 12-frame set simply contains one harder frame. **Reporting
  "neg-ctl FAIL" per arm without the denominator would have been misleading, which is why
  the fraction is printed everywhere.**

---

## 2. ⛔ THE GAIN IS NOT ROLLING SHUTTER — three independent falsifications

### 2a. Running the sweep BACKWARDS gives the identical result

`native_swapped` feeds the shutter-END pose as the start and the START pose as the end —
a physically impossible readout. It scores **+0.0216 [+0.0131,+0.0312]** against native's
**+0.0210 [+0.0096,+0.0365]**, its mean α is 0.3924 vs 0.3919, and its image is **2.264
u8 levels** from native's — closer than any other arm by a factor of ~2.7 (next closest:
`g_p0.50` at 6.086).

> A correction for camera motion during readout **cannot be invariant to the direction of
> the readout.** This alone refutes the physical reading.

### 2b. The gain is largest where a TOP_TO_BOTTOM sweep must change nothing

Per-band grad-NCC, band 0 = top of frame = shutter start, band 7 = bottom = shutter end.
**Δ vs production:**

| arm | b0 (top) | b1 | b2 | b3 | b4 | b5 | b6 | b7 (bottom) |
|---|---|---|---|---|---|---|---|---|
| `g_p0.00` (free) | +0.0077 | −0.0026 | **+0.0143** | **+0.0226** | +0.0077 | +0.0103 | −0.0002 | −0.0048 |
| `s4` | +0.0046 | −0.0084 | +0.0225 | +0.0215 | +0.0109 | +0.0053 | +0.0110 | −0.0022 |
| `s64` | −0.0002 | −0.0217 | +0.0172 | +0.0065 | +0.0033 | −0.0009 | +0.0036 | −0.0259 |
| **`native`** | +0.0202 | +0.0263 | +0.0445 | +0.0136 | −0.0020 | +0.0140 | **+0.0667** | **+0.0755** |
| `native_swapped` | +0.0126 | +0.0184 | +0.0142 | +0.0070 | −0.0028 | +0.0165 | **+0.0640** | **+0.0733** |
| `native_zero_end` | +0.0000 | −0.0098 | −0.0005 | −0.0040 | +0.0010 | −0.0026 | −0.0146 | −0.0113 |

Under `ROLLING_TOP_TO_BOTTOM` the **bottom** rows are exposed at the shutter END — which
is exactly the pose the production baseline renders the *whole* frame from. There, a
faithful rolling-shutter render and the baseline are **the same camera**, so their
difference should be ~0. Native's two largest gains are there (**+0.0667, +0.0755**).

⚠️ **This argument does not rest on getting gsplat's row→time convention right, and that
matters, because the convention is an assumption I did not measure.** Take the other
convention: then the bottom rows are exposed at shutter START, and native's bottom band
should look like `g_p0.00`'s — whose bottom band is **−0.0048**, not +0.0755. **Under
either convention, one end of the frame must be a pose the corresponding global arm
already renders, and native beats it by an order of magnitude more than any pose
difference can explain.** *(The direction control `s16_rev` does not discriminate either:
reversing my own band→phase map moved grad-NCC from 0.3246 to 0.3272 — i.e. the readout
direction is not resolvable at this effect size, which is itself the point of §2a.)*

By contrast the free `g_p0.00` arm behaves the way a pose effect *should*: it gains in the
upper-middle bands (b2 +0.0143, b3 +0.0226) and is neutral-to-negative at the bottom
(b6 −0.0002, b7 −0.0048), because that is where its pose differs from the baseline's.

### 2c. Coverage rises, and a pose sweep cannot create coverage

`mean_alpha`: **native 0.3919 / native_swapped 0.3924**, versus **0.332–0.342 for every
global and every sliced arm**, including `s64` which reproduces the sweep at 64 phases.
Rendering the same gaussians from a moving camera *moves* content; it cannot make ~15 %
more of the frame covered. **A different set of gaussians is being drawn.**

### The mechanism, read from the kernel source

`gsplat/cuda/include/Cameras.cuh:357`, `world_point_to_image_point_shutter_pose`:

```
GLOBAL  ->  return {image_point_start, valid_start};     // validity KEPT
ROLLING ->  return {image_points_rs_prev, true};         // validity DISCARDED
```

and its caller `world_gaussian_to_image_gaussian_unscented_transform_shutter_pose`
(same file, 1281) culls a gaussian outright (`radii = 0`) when **any** of its 7 sigma
points is invalid, because `require_all_sigma_points_valid` defaults to `True`
(`gsplat/cuda/_wrapper.py:53`). `gsplat.rasterization()` never exposes `ut_params`, so
that default is always in force.

⇒ **The rolling-shutter path silently stops culling.** That is a coverage change, not a
shutter model — and it explains 2a (direction-invariant), 2b (uniform, bottom-heavy) and
2c (α up 15 %) at once, which the physical reading explains none of.

**Evidence class:** the source quotes are **MEASURED (read on the installed 1.5.3 tree on
Thor)**; the enum-name bug below is **MEASURED at two independent locations**. §3 turns
the source reading into a count.

---

## 3. Fixed on the way past: `render_probe.py --rolling-shutter` never ran

`stack/experiments/nurec-gsplat/render_probe.py:204` called
`RollingShutterType.TOP_TO_BOTTOM`. Verified twice today, at two different locations:

* reading the file, and
* querying the installed enum live —
  `members: ['ROLLING_TOP_TO_BOTTOM','ROLLING_LEFT_TO_RIGHT','ROLLING_BOTTOM_TO_TOP','ROLLING_RIGHT_TO_LEFT','GLOBAL']`,
  and `RollingShutterType.TOP_TO_BOTTOM` raises
  `AttributeError: type object 'RollingShutterType' has no attribute 'TOP_TO_BOTTOM'`.

**Fixed**: the member name is now read from the calibration (`cam.shutter_type`, which is
the string `"ROLLING_TOP_TO_BOTTOM"`) with an explicit check, so code and rig cannot drift
apart. Verified by RUNNING the probe with `--rolling-shutter`, not by an exit code.

---

## 4. The cost curve, and the 10 Hz budget

⚠️ **TIMING CONTAMINATION, DECLARED.** Two other streams started GPU jobs on Thor at
**18:31:06** (`score_t1_strategic.py`) and **18:34:04 / 18:39:44** (`closedloop_drive.py`,
STREAM D) — verified from `/proc/<pid>/cmdline` and `ps -o lstart`. The `chosen` sweep's
later arms (`s32`, `s64`, `s16_rev`, `s16_fixedactor`) therefore ran **contended**, and
their `raster_ms` / `wall_ms` are **NOT admissible as absolute costs**. GPU power went
from 1976 mW (idle, 18:18) to 31 920 mW.

**Quality numbers are unaffected** — forward rasterization is deterministic, and this run
proves it: `s1` and `g_p0.50` were rendered as separate arms about a minute apart and came
out **bit-identical**.

### Per-render-call cost, from the uncontended arms

| arm | calls/frame | raster ms | **per call** | wall ms | per call | CPU-side per call |
|---|---|---|---|---|---|---|
| `g_p1.00` (isolated) | 1 | 65.3 | **65.3** | 190.5 | 190.5 | 125.2 |
| `s2` | 2 | 140.3 | 70.2 | 418.0 | 209.0 | 138.8 |
| `s3` | 3 | 110.1 | 36.7 | 281.1 | 93.7 | 57.0 |
| `s4` | 4 | 135.8 | 34.0 | 295.4 | 73.8 | 39.9 |
| `s6` | 6 | 218.8 | 36.5 | 463.5 | 77.2 | 40.8 |
| `s8` | 8 | 301.9 | 37.7 | 615.0 | 76.9 | 39.1 |
| `s12` | 12 | 422.1 | 35.2 | 930.5 | 77.5 | 42.4 |
| `s16` | 16 | 582.4 | 36.4 | 1276.7 | 79.8 | 43.4 |
| `s32` ⚠️ contended | 32 | 1707.4 | 53.4 | 5508.6 | 172.1 | 118.8 |
| `s64` ⚠️ contended | 64 | 4054.0 | 63.3 | 11911.9 | 186.1 | 122.8 |

**Per-call raster in a burst: median 36.4 ms (range 34.0–37.7 over `s3`…`s16`).** That
independently reproduces the render-quality stream's **36.3 ms** for this exact config,
measured through a different code path — a useful cross-check that the sliced renderer is
not doing anything extra per call.

⚠️ **RESOLVED, and the resolution is worth recording.** The isolated `g_p1.00` arm first
measured **65.3 ms** against the burst-derived **36.4 ms/call**, which looked like a real
isolated-vs-sustained clock effect. Re-running the identical arm later on a quieter box
gave **36.4 ms** — the same number as the burst, to one decimal. ⇒ **the 65.3 ms was
contention, not a clock ramp**, and the admissible figure for one production render on the
deployed config is **36.4 ms** (three independent observations: the burst median, the
re-run, and the render-quality stream's independent 36.3 ms). *(The grad-NCC for that arm
was 0.3241 in BOTH runs — identical — so the arm is deterministic and only the clock
moved. That pairing is what makes the attribution to contention safe rather than a guess.)*

Both figures are kept below because **the recommendation is robust either way** — it goes
the wrong way for slicing under both:

| | one render | 2 slices | 4 slices | fits a 10 Hz tick? |
|---|---|---|---|---|
| burst figure 36.4 ms | 36 ms | 73 ms | 146 ms | tick budget is 100 ms and the model already takes **60.3 ms p50** (`c7f8d38`) ⇒ **~36 ms left. n=2 already misses.** |
| isolated figure 65.3 ms | 65 ms | 131 ms | 261 ms | **even n=1 is over.** |

⇒ **There is no slice count inside a 10 Hz closed-loop budget.** And it does not matter,
because §1 shows no slice count buys anything anyway.

**A second, fixable cost, recorded so nobody re-discovers it:** the sliced path pays
**~41.6 ms of CPU per render call** (median over `s3`…`s16`) on top of the GPU time,
because `_posed_actors` re-poses all 37 tracks in a Python loop for *every slice*. The ego
moves ≤0.63 m and the actors move ~1 cm over a 30.6 ms readout, so that work could be
hoisted out of the slice loop. **It has NOT been done, deliberately** — optimising a path
this analysis rejects would be effort spent on the wrong side of the decision.

## 5. ⚠️ OPERATIONAL — Thor rebooted mid-panel under four concurrent GPU jobs

At **18:51 CEST** `tanitad-thor` **rebooted** (`last -x reboot` → `Mon Aug 3 18:51`;
`uptime` → `up 0 min`), killing every running job — mine and two other streams'
(`closedloop_drive.py` ×2, `score_t1_strategic.py`). Immediately before, four GPU jobs
were resident and `free -g` read **99 of 122 GB used**.

* **MEASURED**: the reboot, the timestamp, the four concurrent jobs, the 99/122 GB.
* **HYPOTHESIS, not measured**: that memory pressure caused it. `dmesg` is not readable
  without sudo on this box and the previous boot's kernel log was not retained, so **no
  OOM record was obtained.** Stating "Thor OOMed" would be an absence-claim from a single
  unavailable probe.

**Everything under `~/rq_out/` survived** (local disk). **Everything under `/tmp` did
not** — driver scripts and logs were lost, and the in-flight `rs_utgate_chosen` run was
destroyed with no `report.json`. The panels were re-run afterwards on an **idle** box,
which is why the post-reboot timings are the admissible ones.

⇒ **Two operating notes worth carrying forward.** (1) The "log to `/tmp`, never a network
mount" rule was written for RunPod pods whose network mount swallows logs; **on Thor
`/tmp` is cleared by a reboot and `~` is local disk**, so Thor runs should log under `~`.
(2) Thor is a **single-GPU, shared** box: four concurrent render/rollout jobs is past what
it tolerates, and the failure mode is not a slow job, it is **everyone's work lost at
once**.

## 6. The mechanism, COUNTED — `~/rq_out/rs_cost_probe.json`

§2 inferred from the kernel source that the rolling-shutter path stops culling. That is a
hypothesis until somebody counts. `rs_cost_probe.py` calls
`fully_fused_projection_with_ut` **directly** and counts gaussians with a non-zero radius
— i.e. the ones that actually get binned into tiles and rasterised — out of 2,952,008
static gaussians in the deployed config.

| projection setting | frame 0 alive | frac | frame 150 alive | frac |
|---|---|---|---|---|
| **`global_end` + `require_all_sigma_points_valid=True` (PRODUCTION)** | 759,404 | 0.257 | 614,538 | 0.208 |
| `global_end` + `require_all=False` (**relax the gate, FREE**) | **1,139,033** | **0.386** | **933,860** | **0.316** |
| `global_end` + `in_image_margin_factor=2.0` | 759,405 | 0.257 | 614,541 | 0.208 |
| `rs_zero_end` (RS kernel, **zero motion**) | 632,324 | 0.214 | **614,538** | 0.208 |
| **`rs_sweep` (native rolling shutter)** | **1,341,915** | **0.455** | **1,096,693** | **0.372** |

Four things fall out at once, and together they close the case:

1. **Native rolling shutter rasterises +77 % / +78 % MORE GAUSSIANS than production.**
   That is the +15 % `mean_alpha` and the whole "coverage rises, the sweep fills pixels a
   single pose leaves thin" story — except the sweep is not what fills them. A different
   population is being drawn.
2. **At zero motion the RS path culls EXACTLY as the global path does** — 614,538 vs
   614,538 at frame 150, the same integer, and likewise at frames 300 (484,818) and 450
   (342,200), with identical median bbox and identical max radius. Precisely what
   `Cameras.cuh` predicts: with `q_start == q_end` a sigma point invalid at the start is
   invalid at the end, so the function returns early with `valid = false` instead of
   reaching the `return {…, true}`. **The prediction from the source was tested and it
   holds to the unit on 3 of 4 frames.** ⚠️ **Frame 0 does NOT match** (632,324 vs
   759,404, max radius 9.9e5 vs 8.2e2) and **I have not explained why** — it is the first
   frame measured, so a residual first-call effect is a HYPOTHESIS, not a finding. Stated
   rather than dropped.
3. **Relaxing `require_all_sigma_points_valid` on an ordinary GLOBAL render recovers
   +50 % / +52 % more gaussians — about two thirds of the rolling-shutter effect — at no
   geometric cost at all.** This is the free lever the +35 % was actually pointing at.
4. ⛔ **`in_image_margin_factor` is NOT the knob.** Going from 0.1 to 2.0 (20x) admits
   **one extra gaussian at frame 0 and three at frame 150.** A plausible-sounding second
   candidate, measured and dead — recorded so it is not re-proposed.

⇒ **"Rolling shutter is the biggest render-quality lever" should read: "gsplat's
sigma-point validity gate is throwing away three quarters of the scene, and the
rolling-shutter code path happens to bypass it."**

### Cost, honestly

Native RS on the deployed config, **uncontended**: **3258.6 ms/frame** against **36.4 ms**
for one global render — **~90x**. *(The banked "161x" was measured on the `base` config
over a different 5-frame set; both are "two orders of magnitude" and neither is a budget.)*

⚠️ The `rs_cost_probe` run itself executed while three other streams' jobs were back on the
box (§5), so its **absolute** milliseconds are inflated ~10x and are **not quoted**. Its
**counts** are exact and contention-independent, which is why the table above is counts.

## 7. ⚠️ SPUN OFF: the phase curve does not turn over, and the mp4 has 6 more frames than the rig

> **→ SETTLED IN §11: the offset is exactly +6 frames.** This section is the trail that
> led there and is kept for the reasoning; §11 is the answer.

The phase sweep is **monotone in the wrong direction** and it does not stop at the shutter:

| phase | −2.00 | −1.00 | **−0.50** | 0.00 (shutter START) | 0.50 | 1.00 (PRODUCTION) |
|---|---|---|---|---|---|---|
| grad-NCC | *(pending)* | *(pending)* | **0.3332** | 0.3313 | 0.3289 | 0.3241 |

`g_pm0.50` renders the camera **half a readout BEFORE the shutter opens** — outside the
exposure interval entirely — and beats every pose inside it. *"The shutter-start pose is
the best single pose"* cannot produce a maximum outside `[0,1]`. **A systematic offset
between the rig trajectory and the reference video can.**

And there is a candidate, found while checking the timing:

| quantity | value | how |
|---|---|---|
| `camera_front_wide_120fov.mp4` frames | **605** | **full sequential decode**, not the metadata estimate (both agree: `CAP_PROP_FRAME_COUNT` = 605) |
| `rig_trajectories.json` frames | **599** | `RigTrajectories.n_frames` |
| **difference** | **+6 frames = 0.200 s** | |
| video frame period | 33.333 ms | consecutive shutter-end timestamps |
| shutter readout | 30.559 ms | = **0.917 of a frame period** |

⚠️ **A one-frame index offset and a one-readout pose offset are almost the same
displacement (33.3 ms vs 30.6 ms), which is exactly why a phase sweep alone cannot tell
them apart** — and why `rs_frame_offset.py` exists: it renders the production pose for rig
frame `f` and scores it against video frames `f−3 … f+3`, so the competing references are
the **immediate neighbours** rather than frames 40+ apart. That is a strictly harder
negative control than the standard one, and the standard one is structurally blind to this
failure.

⛔ **If the argmax is consistently non-zero, every render-fidelity number in the programme
— including all of mine above — was scored against the wrong reference frame.** It would
not change the rolling-shutter verdict (all arms share the same reference, so the paired
deltas survive), but it would change every absolute grad-NCC ever quoted for this scene.
**This is flagged as an OPEN, HIGH-BLAST-RADIUS item, not as a finding.**

### ⛔ …and the gaussians it admits are NUMERICALLY DEGENERATE

The counts alone would suggest "just relax the gate, it is free". The **bounding boxes**
say otherwise. Same run, `median_bbox_area_px` and `max_radius_px` on a **1920×1080**
raster:

| gate | median bbox (px²) f0/f150/f300/f450 | **max radius (px)** | Σ bbox (px²) |
|---|---|---|---|
| **production** (`require_all=True`) | 16 / 16 / 24 / 36 | **8.2e2 / 4.2e2 / 4.5e2 / 4.9e2** | ~4.9e7 |
| `margin=2.0` | 16 / 16 / 24 / 36 | 8.2e2 / 4.2e2 / 5.8e2 / 4.9e2 | ~4.9e7 |
| `require_all=False` | 32 / 40 / 48 / 108 | **9.93e5** | **~3.0e16** |
| **native rolling shutter** | 64 / 132 / 216 / 520 | **~1.00e6** | **~4.5e15** |

**A radius of 1.0e6 px on a 1920-px-wide image is ~500× the image diagonal.** These are
not large splats, they are **projections whose unscented-transform covariance is garbage**
— which is exactly what "one of the 7 sigma points failed to project and we kept it
anyway" produces. The sum of bounding-box area rises by **8–9 orders of magnitude**, which
is also the most likely explanation for the ~90× cost: the tile-intersection list
explodes.

⇒ **The honest mechanical statement: gsplat's rolling shutter improves grad-NCC by
smearing a few hundred numerically-degenerate splats over the scene's reconstruction
holes.** It is a hole-fill, not a shutter model — and the scene already ships a
*principled* hole-fill (the `sky-env-map`, already deployed at gain 0.3).

⇒ ⚠️ **Therefore "just set `require_all_sigma_points_valid=False`" is NOT a
recommendation.** It reproduces two thirds of the coverage effect, and it reproduces the
degeneracy with it. Whether that trade is worth making is a QUALITY question, and it is
answered in §8 — not by the counts.

### 7a. MEASURED — the offset is real, at least +3 frames, and the scan did not reach its optimum

`~/rq_out/rs_frame_offset.json`. Render rig frame `f` **exactly as production does**
(shutter-END pose, actors at shutter-END time), score against video frames `f−3 … f+3`:

| reference offset (frames) | −3 | −2 | −1 | **0 (what we do)** | +1 | +2 | **+3** |
|---|---|---|---|---|---|---|---|
| mean grad-NCC over 12 frames | 0.2991 | 0.3048 | 0.3120 | **0.3204** | 0.3291 | 0.3403 | **0.3590** |

**argmax = +3 on 11 of 12 frames** (the 12th is `f3`, three frames from the clip start,
where the profile is flat to ±0.0007 — i.e. uninformative, not contradictory).

⛔ **The curve is MONOTONE ACROSS THE ENTIRE SCAN AND STILL RISING AT THE EDGE.** +3 is the
boundary of the window, not a maximum, so **+0.0386 is a LOWER BOUND on the loss** and
"+3 frames" is a lower bound on the offset. A wider scan settled it: **§11, exactly +6**.

**For scale: this dwarfs everything else in this document.** The rolling shutter that cost
90× bought **+0.0210**. Pointing at the right video frame is worth **≥ +0.0386 and free**.
The mp4/rig frame counts (**605 vs 599, +6**) make +6 the obvious candidate, and +6 is
consistent with a still-rising curve at +3 — but that is a **HYPOTHESIS** until the wide
scan turns over.

**Why every negative control in the programme passed anyway:** `wrong_frames_for()`
requires wrong candidates to be **≥ 40 frames away** (`MIN_WRONG_GAP = 40`). A 3–6 frame
misalignment is invisible to it by construction — the correct-ish frame still wins
comfortably against frames 40+ away. **The control was never wrong; it was answering a
different question.** ⇒ a neighbour-offset scan belongs in the standard render-quality
harness, not only here.

## 8. The gate panel — and why the "free fix" is DEAD

`~/rq_out/logs/utgate_chosen.log` (banked here as `rs_utgate_chosen.PARTIAL.log`).
⚠️ **PARTIAL — this run did not produce a `report.json`**, for the reason in 8b. Only the
arms that printed are quoted, and they are quoted from the log, not reconstructed.

### 8a. Phase EXTRAPOLATION — a free render beats rolling shutter

Same 12 frames, same renderer instance, same session:

| arm | what it is | grad-NCC | margin | neg-ctl | raster ms |
|---|---|---|---|---|---|
| `g_p1.00` | **production** (shutter END) | 0.3241 | +0.0908 | 11/12 | **36.4** |
| `g_p0.50` | shutter midpoint | 0.3289 | +0.0977 | 11/12 | 52.8 |
| `g_p0.00` | shutter start | 0.3313 | +0.0966 | 11/12 | 45.9 |
| `g_pm0.50` | ½ readout **before** the shutter | 0.3332 | +0.0996 | 11/12 | 47.8 |
| `g_pm1.00` | 1 readout before | 0.3388 | +0.1059 | 11/12 | 51.4 |
| **`g_pm2.00`** | **2 readouts (61 ms) before** | **0.3499** | **+0.1153** | **12/12** | **47.6** |
| `native` | gsplat rolling shutter | 0.3451 | +0.1084 | 11/12 | 6844 ⚠️ |

Three things at once, all in the same direction:

1. **`g_pm2.00` beats native rolling shutter** on grad-NCC (0.3499 vs 0.3451) **and** on
   the negative-control margin (+0.1153 vs +0.1084), at **~1/68 of the cost**.
2. **It is the ONLY arm in this entire study that passes the negative control on all 12
   frames.** Frame 163 — the frame that failed for all 21 arms in §1, baseline included —
   **passes once the render is moved earlier.** That is exactly what a timing offset
   predicts and what a rendering-quality problem does not.
3. **The curve had still not turned over at −2.0**, matching the independent frame-offset
   scan (§7a) which was still rising at +3. Two probes of different kinds, same direction,
   neither at its optimum.

*(⚠️ `native` reads 6844 ms here vs 3258.6 ms in §1 — the box was shared. Its grad-NCC was
**0.3451 in all three runs**, so the arm is deterministic and only the clock moved.)*

### 8b. ⛔ `require_all_sigma_points_valid=False` is UNRENDERABLE — measured the hard way

The counts in §6 made relaxing the gate look like a free two-thirds of the effect. It is
not free; it is not even possible.

**`g_p1.00_anysigma` drove `tanitad-thor` from ~90 GB to 122 / 122 GB with 0 available,
and the process then would not die** — `kill -9` left it in state `R`, spinning inside the
CUDA allocator, holding the whole box. The panel was abandoned there.

That is the predicted consequence of the §6 bounding boxes: **Σ bbox area 3.0e16 px² with
radii up to 1e6 px on a 1920×1080 raster** means a tile-intersection list that no amount of
memory satisfies. So:

* ✅ the mechanism claim stands — the gate is what rolling shutter bypasses;
* ⛔ **but "relax the gate" is not an available fix.** It is recorded here as a **measured
  dead end**, with its failure mode, so nobody re-derives it from the counts alone. The
  counts said "+50 % more gaussians"; they did not say "and the render will not complete".
* ⇒ Whatever gsplat's RS path does to stay renderable while admitting a similar population,
  it is **not** simply `require_all=False`. Not explained here.

**Not measured, because the run died:** `g_p0.50_anysigma`, the `margin` quality arms,
`s4_anysigma`, `s8_any_m2.0`, and the `g_p1.00_restorecheck` bit-exactness check. The
`margin` arms are the least missed — §6 already showed `in_image_margin_factor` moves 1–3
gaussians.

## 9. What I did NOT establish — stated, not omitted

* **Nothing here is a driving-quality claim.** Every number is open-loop render fidelity
  against the scene's own camera mp4. Whether rolling shutter (or the frame offset)
  changes the CLOSED LOOP's longitudinal / lateral / tactical / strategic families is
  ⛔ **NOT ESTABLISHED** — no closed-loop arm was run in this stream. The banked
  RENDER_QUALITY.md carried the same caveat and it still stands.
* **One scene, one camera, 12 frames.** The interval is a paired bootstrap over FRAMES
  inside `00040136`; there is no second scene, so nothing here is a scene-level CI and the
  offset finding must be re-checked on a second scene before it is generalised.
* **gsplat's row→time convention was not measured**, only reasoned about; §2b is
  constructed so the conclusion holds under either convention.
* **`native_zero_end` at frame 0 does not match production** (632,324 vs 759,404) while
  frames 150/300/450 match to the unit. Unexplained.
* **The isolated-vs-burst render cost (65.3 vs 36.4 ms) is unresolved.** A Jetson clock
  ramp is a hypothesis. The recommendation is robust to it either way.
* **`in_image_margin_factor` was swept only to 2.0**, and only on the projection counts,
  not on quality.
* **The per-slice actor re-posing (~41.6 ms/call) was deliberately NOT optimised** — it
  would improve a path this analysis rejects.

## 10. Two machine losses, and what they cost

`tanitad-thor` was lost **twice** during this stream:

| when | what | cause |
|---|---|---|
| **18:51** | full **reboot**; all four streams' jobs killed, `/tmp` cleared | 4 concurrent GPU jobs at 99/122 GB. **HYPOTHESIS** — no OOM record obtainable (§5) |
| **~19:08** | wedged at **122/122 GB, 0 available**; `rs_sweep` unkillable in state `R` under `kill -9`; sshd stopped answering; box rebooted again | **MEASURED**: the `g_p1.00_anysigma` arm (§8b) |

Everything under `~/rq_out/` survived both (local disk); `/tmp` did not. **After the second
reboot the box was idle and every remaining panel completed** — which is also why the
post-reboot timings are the admissible ones.

**Still NOT measured, and it is one item:**

| item | why it matters | state |
|---|---|---|
| `rs_sweep.py --config base` full sweep | direct comparability with the banked `~/rq_out/panel3_rs` "+35 %" headline on the *old* config | killed at `native_swapped`, the arm that was itself driving 99 GB. `native` on `base` did complete: **0.2805, neg-ctl 12/12** |
| the rest of the `utgate` quality panel | `g_p0.50_anysigma`, the `margin` quality arms, `s4_anysigma`, `s8_any_m2.0`, `g_p1.00_restorecheck` | died with §8b. The `margin` arms are the least missed — §6 showed `in_image_margin_factor` moves 1–3 gaussians |

Neither changes the recommendation.

## 11. ⭐⭐ SETTLED — the reference offset is EXACTLY +6 frames (`~/rq_out/rs_frame_offset_k10.json`)

The §7a scan stopped at its boundary still rising. Widened to `−10 … +10`, it **turns over
cleanly**, and the answer is unambiguous. Mean grad-NCC over the same 12 frames, render
held at **production settings** (shutter-END pose, actors at shutter-END time) — **only the
reference index varies**:

| offset | 0 | +1 | +2 | +3 | +4 | +5 | **+6** | +7 | +8 | +9 | +10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| grad-NCC | 0.3114 | 0.3228 | 0.3363 | 0.3550 | 0.3806 | 0.4213 | **0.4911** | 0.4661 | 0.4077 | 0.3709 | 0.3478 |

| offset | −1 | −2 | −3 | −4 | −5 | −6 | −7 | −8 | −9 | −10 |
|---|---|---|---|---|---|---|---|---|---|---|
| grad-NCC | 0.3029 | 0.2963 | 0.2917 | 0.2884 | 0.2852 | 0.2838 | 0.2804 | 0.2778 | 0.2768 | 0.2756 |

* **`argmax_histogram = {6: 12}` — every single frame, no exceptions, no ties.**
* **+0.1797 grad-NCC (0.3114 → 0.4911, +57.7 %). Free.**
* It **matches the frame-count difference exactly**: mp4 **605**, rig **599**, **Δ = 6**
  (full sequential decode, §7). Two independent facts, same integer.
* The peak is sharp: +7 is already worse than +6, so this is a maximum, not another
  boundary artefact.

### What this costs the programme

| lever | grad-NCC gain | cost |
|---|---|---|
| **reference offset +6** | **+0.1797** | **free** |
| native rolling shutter | +0.0210 | ~90x a render |
| phase / pose sweep (seam-free) | +0.003 … +0.005 | free to 64x |

**Frame alignment is 8.6x the entire rolling-shutter effect and 40x the pose sweep.** The
render was never the problem; the *comparison* was.

⛔ **BLAST RADIUS — this is the escalation, not the rolling-shutter verdict.** Every
grad-NCC / MAE / PSNR ever quoted for this scene — `panel1`…`panel6_chosen`, the "+23.4 %"
and "+35.1 %" headlines, `FINDINGS.md`'s original 0.3802-vs-0.2110 validation, and **every
absolute number in this document** — was computed against a reference **6 frames off**.
* **Paired deltas between arms SURVIVE** (all arms shared the same wrong reference), so
  the rolling-shutter verdict and every A/B in the render-quality work stand.
* **Absolute values do NOT.** The renderer is materially better than any of us measured.
* **The negative control never had a chance to catch it:** `wrong_frames_for()` enforces
  `MIN_WRONG_GAP = 40`, so a 6-frame error is invisible to it *by construction*.
  ⇒ **a neighbour-offset scan belongs in `render_quality.py` itself.**

⚠️ **NOT yet established: WHICH side is wrong.** Six extra frames in the mp4 is consistent
with leader frames at the start of the video, but I did not verify where the extra frames
sit, and I did not check a second scene. **Do not "fix" this by adding 6 to an index until
that is done** — `rs_frame_offset.py` is the instrument, it takes ~2 minutes per scene.

---

## 12. Verification of my own changes

* ✅ **NO REGRESSION, proven bit-wise.** `~/rq_out/rs_regression_check.py` re-rendered the
  banked `panel6_chosen` frame 150 through the **edited** `gsplat_renderer.py` and compared
  against the npz produced by the **pre-edit** file:
  `img bit-identical: True`, `alpha bit-identical: True`. The other stream's re-scoring
  baseline is untouched — verified by loading the artifact, not by reading the diff.
* ✅ **`render_probe.py --rolling-shutter` now runs**, verified **from the far side**: it
  produced `~/rq_out/rp_rs_check/render_frame0000_f0_background-road.png` (1,150,305 B) and
  a `report.json`, and logged `shutter=ROLLING_TOP_TO_BOTTOM`. Before today it raised
  `AttributeError` and produced nothing.
* ✅ `pytest -q` — **1851 passed**, 12 skipped, 2 xfailed (209 s).

---

## 13. SLICE BATCHING — the brief's third knob, measured (`~/rq_out/rs_batch_chosen`)

`render_rs_batched()` sends the n shutter phases as **n CAMERAS in ONE `rasterization`
call** (`viewmats [C,4,4]` → `[C,H,W,3]`), so the n kernel launches collapse to one and the
37 actor tracks are posed **once** instead of n times. Paired against the sequential twin
`sN` on the same frames, same renderer instance:

| n | `bN` grad-NCC | `sN` grad-NCC | `bN` raster ms | `sN` raster ms | **`bN` wall ms** | **`sN` wall ms** | wall speed-up |
|---|---|---|---|---|---|---|---|
| — (`g_p1.00`) | 0.3241 | — | 50.5 | — | 116.9 | — | — |
| 2 | 0.3266 | 0.3266 | 126.3 | 98.7 | 203.4 | 221.2 | 1.09x |
| 4 | 0.3282 | 0.3282 | 205.0 | 207.6 | **237.3** | 503.4 | **2.12x** |
| 8 | 0.3265 | 0.3265 | 331.2 | 420.9 | **347.8** | 1029.9 | **2.96x** |
| 16 | 0.3243 | 0.3243 | 692.4 | 443.9 | 710.0 | 787.9 | 1.11x |

* ✅ **SELF-CONSISTENCY: `bN` and `sN` agree to 4 decimals on grad-NCC AND on the negative
  margin at every n** — two independent implementations of the same thing, so the batched
  path is verified rather than assumed.
* ✅ **Batching works, and it works exactly where predicted**: the win is the CPU side.
  Wall−raster collapses from **296 ms → 32 ms at n=4** and **609 ms → 17 ms at n=8**. That
  is the per-slice actor re-posing, removed.
* ⛔ **It does not change the verdict.** The GPU raster stays **O(n)** — each camera still
  rasterises a full frame — so the floor is unmoved. **The cheapest batched slice arm
  (`b2`, wall 203 ms) is already ~2x a whole 10 Hz tick**, and `b4` is 237 ms.

⇒ **"Batch the slices" was the right optimisation to try and it delivers up to 3x on wall
clock — and the answer is still no.** Recorded so the idea is not re-proposed as
untested.

---

## MANIFEST

**Repo (staged, not committed):**

| path | what |
|---|---|
| `stack/experiments/alpasim-gsplat/rs_sweep.py` | the phase / slice / utgate / batch panels |
| `stack/experiments/alpasim-gsplat/rs_cost_probe.py` | interleaved timing + direct projection-survivor counts |
| `stack/experiments/alpasim-gsplat/rs_frame_offset.py` | neighbour-offset alignment probe (§7a) |
| `stack/experiments/alpasim-gsplat/rs_seam_control.py` | seam-vs-geometry control (§1) — **ran; it retracted one of my own readings** |
| `stack/experiments/alpasim-gsplat/rs_analyze.py` | report.json → the tables above, with the paired frame bootstrap |
| `stack/experiments/alpasim-gsplat/gsplat_renderer.py` | **+** `se3_interp`/`R_to_quat_wxyz`, `render_rs_sliced`, `render_rs_batched`, `set_ut_gate`/`ut_defaults`, `render(..., return_torch=)`. **Purely additive**: the only deletions in the diff are the changed signature line and the 3-line return path |
| `stack/experiments/nurec-gsplat/render_probe.py` | `--rolling-shutter` enum bug FIXED (§3) |
| `stack/experiments/alpasim-gsplat/results/2026-08-03-rolling-shutter/` | this file + `rs_sweep_chosen.report.json`, and every artifact in the table at the top of this file |
| `Project Steering/RETRACTION_LOG.md` | entry **R-2026-08-03-j** |

**On `tanitad-thor` (local disk, survived both incidents):** everything in the table at
the top of this file, plus `~/rq_out/logs/`, `~/rq_out/run_rsE.sh`, `~/rq_out/run_rsE3.sh`,
`~/rq_out/rs_regression_check.py`. `~/rq_out/rs_utgate_chosen/` and
`~/rq_out/rs_sweep_base/` are **incomplete** (no `report.json`) — see §10.

`pytest -q` green: **1851 passed, 12 skipped, 2 xfailed** (209 s).
**Default render settings were NOT changed** — every addition is opt-in.

---
