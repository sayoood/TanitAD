# v5 176×624 — geometry, intrinsics and extrinsics VALIDATED, then launched

**PI 2026-07-29: "accept the small regression and go with 117 degree, validate your image
interpretation and extrinsics/intrinsics."** Validation run BEFORE committing GPU-days; v5 launched
on pod2 only after all six checks passed.

⚠️ **Why this was not ceremony.** This program has shipped geometry errors twice: the
geometric-centre crop was **~215 px wrong for rig B** (the front-wide camera has TWO rigs with
principal points ~211 px apart), and a fixed 100° HFOV was refuted by GeoCalib on fresh video. A
wrong frame or pose convention does not crash — it trains silently and yields a plausible wrong
number.

## 1. ✅ Frame arithmetic — derived independently from `f_ref` alone

`f_ref = 305.5774907364391`, projection **cylindrical**.

| quantity | formula | computed | card |
|---|---|---|---|
| HFOV, full cache | **w / f** (arc length ÷ focal) | 640/f = **120.0000°** | 120 |
| HFOV, v5 slice | w / f | 624/f = **117.0000°** | **117.000** |
| VFOV, v5 slice | **2·atan(h/2 / f)** | 2·atan(88/f) = **32.1306°** | 32.131 |
| tokens @ patch 16 | (176/16)×(624/16) | 11 × 39 = **429** | 429 |

⭐ **The two axes use DIFFERENT formulas and that is the easy thing to get wrong.** Horizontal is
cylindrical (arc/focal); vertical is pinhole-like. Reading the horizontal as pinhole would give
**92.6°**, not 117° — a 24° error that would have looked superficially reasonable.

**The slice is genuinely centred:** rows [40:216] leaves 40 above / 40 below; cols [8:632] leaves
8 left / 8 right. Symmetric on both axes.

⭐ **Independently confirmed by the trainer at launch**, which computes this through its own
`resolve_v2_frames` path and printed: *"SUB-FRAME 176x624 (HFOV 117.000deg / VFOV 32.131deg) =
rows [40:216], cols [8:632] … a pure pixel slice (same f_ref, same cylindrical)"*. Two independent
derivations agree.

## 2. ✅ Decode matches declaration

`PIL` decode of a real `*.v2ep.pt` frame: declared `image_h=256, image_w=640` → decoded
`(256, 640, 3)`. Slice `a[40:216, 8:632]` → **(176, 624, 3)**.

## 3. ⛔→✅ Masking: my FIRST pass was wrong, twice. The population answer is what stands.

**First pass (WRONG):** one frame of one clip, reporting 22.5 % zeros inside the slice.
Wrong on two counts — (a) **one clip**, which is precisely the probe that produced this program's
`observed_frac = 1.0` error (`_assert_geometry_deliverable` sampled `clips[0]`, which happened to be
rig A); and (b) it counted **per-channel** zeros, not fully-black **pixels**. The frame was also a
night scene (mean 26.5), inflating dark values.

**Population pass (60 clips, stride 40 over 2400), fully-black pixels:**

| region | mean black frac | max |
|---|---|---|
| full 256×640 frame | **0.0768** | 0.1998 |
| **v5 slice** | **0.0063** | 0.0927 |

⭐ **12× reduction in masked pixels**, and ~70 % of sampled clips have **zero** black pixels in the
slice — more than rig A's 26.75 % share, so the slice clears rig B's band for most clips too.

**Row profile — the mask sits exactly BELOW the cut:**

| rows | mean black | worst clip | |
|---|---|---|---|
| 40–216 (all bands) | ≤ 0.013 | **0.347** | inside v5 |
| 208–223 | 0.040 | 0.566 | below |
| 224–239 | 0.358 | **1.0000** | fully masked on ≥1 clip |
| 240–255 | 0.764 | **1.0000** | fully masked on ≥1 clip |

⇒ **No clip has a majority-masked row inside the v5 slice.** The rig-clean cut is correctly placed.
Rig assignment is unambiguous: *"native principal-point row cy, boundary 650.872 (largest gap
193.35 px over 3,000 clips; no clip ambiguous)"*.

## 4. ✅ Pose / action semantics — read in CODE, then cross-checked numerically

```
physicalai.py:17,641   poses   = (x, y, yaw, v)   clip-local frame
physicalai.py:632      actions = (steer, accel)
```

| column | measured range (one clip) | consistent with |
|---|---|---|
| poses x | −0.41 → 90.95, monotonic | forward metres |
| poses y | −6.47 … 3.79 | lateral metres |
| poses yaw | −0.395 … 0.132 | radians (±23°) |
| **poses v** | 2.92 … 7.20, first **4.6748** | **m/s** |
| actions steer | −0.070 … 0.037 | — |
| actions accel | −1.95 … 1.32 | m/s² |

⭐ **Independent cross-check:** per-step ‖Δ(x,y)‖ = 0.4661 m at 10 Hz ⇒ **4.66 m/s**, against the
declared `v` first value of **4.6748**. The speed channel and the finite difference of the position
channels agree — so the pose units, the frame, and the 10 Hz rate are mutually consistent.

## 5. ✅ Extrinsics posture

The v2 cache carries only `frame = {height, width, f_ref, projection}` — **per-clip intrinsics and
extrinsics are consumed at BUILD time** into the cylindrical rectification, so there is no runtime
extrinsic to mis-apply. Poses are **clip-local metres**, consistent with the settled finding that
`egomotion` carries no lat/lon/GNSS.

## 6. Launch

pod2, `flagship-v5-w120-30k`, from-scratch, 30k steps, batch 16 × accum 4, `--require-parity`
(both caches sha256-VERIFIED against the committed manifest, skip-hash `f09e44db`),
`--anchors-dense`, `--heldout-gate`.

⚠️ **Standing, and unchanged by this validation:** v5 trains at **117°, not 120°**, and **v1's
0.4271 is not a valid comparator** — v1's encoder was trained at 51.4°, so wide frames are OOD for
it in either direction. The PI has explicitly accepted the ego-yaw-rate regression (−0.03546 R²)
that was the pre-registered hold.
