# PHASE 1 — TanitAD's input pipeline, established by MEASUREMENT from source

**Date:** 2026-08-30 · **Stream:** Research Lab special topic (Architecture & Inference)
**Part of:** the PI-commissioned INPUT-PIPELINE DESIGN REVIEW. This file is the *facts* layer.
The field comparison and the verdicts live in `RESULT.md` beside it.

**Why this file exists separately.** The PI supplied a specification of our own setup and
instructed that **none of it be taken on trust** — two of his own specs this week carried facts
that were wrong precisely because they were not re-derived (RETRACTION_LOG MM-C5, MM-C6). Every
row below is re-derived from source or from a config, with `file:line`. Where I found the PI's
belief to be **wrong**, it is flagged ⚠️ **CORRECTION**. Where I found something he did not
state, it is flagged ➕ **NEW**.

**Evidence classes used here:** `MEASURED-SOURCE` = read out of the named source file this
session · `MEASURED-COMPUTE` = computed this session from formulas *in* our source, script
banked under `raw/` · `MEASURED-REGISTRY` = `MODEL_REGISTRY.md` or a raw eval JSON ·
`ESTIMATED` = derived with a stated assumption · `UNVERIFIED` = could not establish.

---

## 0. The scorecard against the PI's specification

| PI's belief | verdict | the measured fact |
|---|---|---|
| 256 (H) × 640 (W) | ✅ **CONFIRMED** | `train_v6_staged.py:6613-6614` defaults `--frame-h 256 --frame-w 640` |
| cylindrical projection | ✅ **CONFIRMED** | `calib.py:1244-1248` `PHYSICALAI_WIDE120_256x640(..., projection="cylindrical")` |
| f_ref 305.5774907364391 | ✅ **CONFIRMED** | same line, to all 16 digits |
| 120° horizontal FOV | ✅ **CONFIRMED** | computes to **120.0000°** exactly from `calib.py:149-151` |
| cache tag `256x640f305.5775cyl` | ✅ **CONFIRMED** | `calib.py:196-199` `tag()`; corpus name `physicalai-train-e438721ae894-**w120-256x640cyl**` (`MODEL_REGISTRY.md` §13.0) |
| codec PNG (lossless) | ✅ **CONFIRMED** | `parity_manifest.json:2514-2536` — `"codec": "png"`, `"quality": null`, on **both** the train and val `…-w120-256x640cyl` corpora. The loader selects on that field (`v2_dataset.py:127`), not on the misleading `jpeg_buf` buffer name |
| patch 16 → 16×40 grid, 640 tokens | ✅ **CONFIRMED** | `config.py:43-46` `token_grid()`; `train_v6_staged.py:6615` `--patch 16` |
| **readout pools to 4 × 8** | ⚠️ **CORRECTION — it is arm-dependent, and the DEFAULT is 4 × 4** | `train_v6_staged.py:6636-6637`: `--readout-grid 4`, `--readout-grid-w` **`None`**; `readout.py:78` `gw = grid if grid_w is None else grid_w`. The v7 line *does* pass `--readout-grid-w 8` (that is what the arm name `rdw8` means). **See §4 — this is a real fork in the programme, not a typo.** |
| `--window 6` | ✅ **CONFIRMED** | `train_v6_staged.py:6647`; `config.py:54` |
| dt 0.1 s (10 Hz) | ✅ **CONFIRMED** | `train_v6_staged.py:7015` |
| hz_op 10.0 / hz_tac 2.0 / hz_str 0.5 | ✅ **CONFIRMED** | `v6.py:3706-3708`, with a consistency assert at `v6.py:4198-4199` |
| **horizon 20 steps = 2.0 s dense** | ⚠️ **CORRECTION — the horizon is 60 steps = 6.0 s. 2.0 s is only the OPERATIVE BAND.** | `v6.py:148-156`: `PLAN_STEPS = 60`, `DT = 0.1`, `HORIZON_S = 6.0 s`, `OP_BAND_S = (0.0, 2.0)`, `TAC_BAND_S = (2.0, 6.0)`. The block is headed *"THE BINDING HORIZON SPEC (PI 2026-08-11, HIERARCHY_VOCABULARY.md)"* and its own comment reads *"Emission k scales **20 → 60**; roll selection rolls to 6 s"* — i.e. **20/2.0 s is the superseded value**. See §5. |
| two rigs, cy ~543 / ~755 | ✅ **CONFIRMED** | `calib.py:1250-1254`, and the refusal message at `calib.py:1189` keys rig B on `cy >= 650` |
| rig B ~91 % observed (masked ≈ 0.0905) | ✅ **CONFIRMED, and sharpened** | `calib.py:1244-1246`: **8.897 %** masked on rig B, **0.0017 %** on rig A, n = 3,000 |
| rig A ~100 % | ⚠️ **CORRECTION — rig A is NOT fully observed either** | 0.0017 % masked. `calib.py:1178-1180` says this explicitly: *"less obviously, on rig A too — the 120° request over-runs some clips' horizontal field"* |
| single camera, no LiDAR, no map | ✅ **CONFIRMED** (unchanged from the pinned read-set test) | |

➕ **NEW facts the specification did not contain, and that matter:**
- **VFOV is 45.4556°** and is **PINHOLE, not cylindrical** (`calib.py:154-157`: `half_angle_y_rad = atan((H/2)/f_ref)`). Ours is a *cylinder*, i.e. equidistant in azimuth and rectilinear in elevation — not a sphere.
- **The encoder has no temporal mechanism at all.** Time enters *only* as input channels (§3).
- **The frame build point-samples a ~3× decimation with no anti-alias prefilter** (§7). This is the one candidate defect I found that is invisible at the architecture level.
- **Rig B is 72.9 % of the corpus** (2,188 / 3,000, `calib.py:1250-1253`), so the rig-correlated mask is the *majority* condition, not an edge case.

---

## 1. Frame geometry — exact

`stack/tanitad/data/calib.py:79-215` defines one object, `CanonicalFrame`, that flows
config → cache build → model. The deployed instances (`calib.py:1244-1263`):

| constant | H × W | projection | HFOV | VFOV | masked (rig A / rig B) |
|---|---|---|---|---|---|
| `PHYSICALAI_WIDE120_256x640` | 256 × 640 | cylindrical | **120.0000°** | **45.4556°** | 0.0017 % / **8.897 %** |
| `PHYSICALAI_RIG_CLEAN_176x624` | 176 × 624 | cylindrical | 117.000° | 32.131° | **0.000000 % / 0.000000 %** |
| `PHYSICALAI_RIG_CLEAN_128x576` | 128 × 576 | cylindrical | 108.000° | 23.658° | 0 / 0, and tiles a 4×4 readout exactly |

`MEASURED-COMPUTE` (raw: `raw/proj_measure.py`), from `calib.py:88-94` and `:149-165`:

```
256x640 cylindrical, f_ref = 305.5774907364391
  HFOV 120.0000°   VFOV 45.4556°
  horizontal angular resolution  0.18750 °/px  (11.250 arcmin/px) — UNIFORM across the width
  vertical   angular resolution  0.17756 °/px at the centre row
  patch 16 → 16 × 40 = 640 tokens, 3.000° per token column
```

⚠️ The **pinhole** FOV formula applied to this frame returns **92.64°** and looks entirely
plausible. It is wrong here; the corpus is cylindrical and the column is linear in azimuth.
This trap is already in `CLAUDE.md` and it caught nothing new this session — the code is right.

---

## 2. The projection — what it actually does to the scene

`calib.py:906-935` (`cylindrical_rays`) is the definition:
> ray = `(sin φ, y_n, cos φ)` with `φ = (u − (W−1)/2)/f_ref` and `y_n = (v − (H−1)/2)/f_ref`

so `u = f·φ` (equidistant in azimuth) and `v = f·tan(ε)` (rectilinear in elevation).

**2a. Local scale distortion** `MEASURED-COMPUTE`. This is the quantity that the entire
distortion-aware-vision literature (SphereNet, PanoFormer, tangent images) exists to fix, so it
is the right axis on which to ask whether we inherit their problem:

| projection | worst local anisotropy over OUR frame |
|---|---|
| **ours (cylindrical, 120° × 45.5°)** | **1.1755×** — at the top/bottom rows, `sec²(22.648°)` |
| pinhole at the *same* 120° field | **4.0000×** — at the frame edge, `sec²(60°)` |
| equirectangular at ±60° latitude (the regime the 360° papers target) | 2.0000× |
| equirectangular at ±85° latitude | 11.47× |

⭐ **Cylindrical is 3.40× LESS locally distorting than a pinhole rectification of the same
field**, and our elevation span (±22.6°) keeps us below the regime where the 360°-vision
literature's problem begins. This is a geometric fact, not a claim that cylindrical trains
better — `calib.py:893-896` says the same thing in the same words and defers the training
question to an experiment.

**2b. Straight-line bowing** `MEASURED-COMPUTE` (raw: `raw/proj_measure2.py`). Pinhole's one
real advantage is that world straight lines stay straight. Sagitta = max perpendicular deviation
from the chord, over the in-frame extent:

| world line | cylindrical sagitta | as % of frame H | pinhole |
|---|---|---|---|
| vertical pole | **0.00 px** | 0.00 % | 0.00 |
| lane marking, ego lane edge (X = 1.75 m) | 1.45 px | 0.57 % | 0.00 |
| lane marking, next lane (X = 3.5 m) | 7.75 px | 3.03 % | 0.00 |
| kerb / barrier top, X = 5 m | 2.33 px | 0.91 % | 0.00 |
| building roofline 8 m, X = 12 m | 5.91 px | 2.31 % | 0.00 |
| stop line across the road at Z = 8 m | 28.48 px | 11.13 % | 0.00 |
| **overhead gantry 5 m at Z = 15 m** | **35.54 px** | **13.88 %** | 0.00 |

So *globally* the bow is real and, for near-field transverse structure, large.

**2c. …but it is invisible at patch scale.** `MEASURED-COMPUTE`. The ViT does not see global
lines; it sees 16 × 16 patches. Worst-case sagitta of any of those same world lines **within one
16-px patch**:

| line | local sagitta inside one patch |
|---|---|
| overhead gantry 5 m, Z = 15 (the global worst case) | **0.021 px** |
| stop line, Z = 8 | 0.017 px |
| every other case tested | ≤ 0.011 px |

and analytically the horizontal mapping has **`d²u/dφ² = 0` exactly** — the cylinder is
*perfectly* linear in azimuth, so there is **zero** horizontal second-order distortion anywhere
in the frame. (Pinhole is not: `2f·sec²φ·tanφ`, which at 55° is 0.909 px of bow across half a
patch — 43× ours.)

⇒ **The cylindrical map is affine to within 0.02 px inside every patch.** What the projection
changes is the *arrangement* of patches, not their content — and arrangement is exactly what a
learned APE (`encoder.py:125`) plus 2D axial RoPE (`encoder.py:359-368`) is built to absorb.

**2d. Apparent-size variation with azimuth** `MEASURED-COMPUTE`. A 1.8 m car face at 25 m:

| azimuth | ours (cyl), width vs boresight | pinhole, width vs boresight |
|---|---|---|
| 0° | 100.0 % | 100.0 % |
| 30° | 86.6 % | 115.5 % |
| 45° | — | 141.4 % |
| 59° (frame edge) | **51.6 %** | (off-frame; would be 4× density at 60°) |

The cylindrical column is **exactly `cos φ`** — which is the *true* foreshortening a physical
observer sees. Cylindrical reproduces real angular size; pinhole magnifies the periphery. Our
edge "distortion" is physics, not an artefact. A residual **6.17 %** vertical-edge skew at 59°
is the genuine projection artefact and it is small.

---

## 3. The encoder — architecture, and how past frames are fed

**3a. What the "9" is.** `MEASURED-SOURCE`, resolved from the loader as instructed:

- `train_v6_staged.py:6606` — `--in-channels` default **9**
- `config.py:17` — `in_channels: int = 1 # 1 = BEV toy; 9 = camera (3-frame stack, D-015)`
- `comma2k19.py:633-641` — `stack_frames(vid_u8, n_stack=3)` → `[T−(n−1), 3n, S, S]`, docstring:
  *"n_stack=3 at 10 Hz → the encoder sees `[t−200ms, t−100ms, t]` in one 9-channel input …
  Oldest frame first, current frame in the LAST 3 channels."*
- `v2_dataset.py:29-36` pins the same contract for the compressed cache, and `_decode_stacked`
  decodes raw rows `[a : b+n_stack−1]` to build it.

⇒ **9 = 3 consecutive RGB frames × 3 channels, CHANNEL-stacked. It is not a sequence dimension.**
The epcache's `[199, 9, 256, 640]` is `[T_out, 3·n_stack, H, W]`.

**3b. Consequences, and they are not neutral:**
- `encoder.py:123-124` — a **single** `nn.Conv2d(9, d_model, kernel_size=16, stride=16)`. All
  three timesteps are mixed **in the first linear op**, before any nonlinearity or attention.
- **There is no temporal attention in the encoder.** `forward()` (`encoder.py:152-172`, and
  `:385-406` for ViT-5) is `[B,C,H,W] → patch → +pos → blocks → norm`. Time is channels, full stop.
- The 3 stacked frames are **NOT ego-motion-compensated** — `stack_frames` is a pure
  `torch.cat` of raw pixel tensors (`comma2k19.py:640-641`). Frame `t−200ms` is *not* warped into
  the `t` camera frame.
- The patch-embed weight is shared across the 3 timesteps only in the trivial sense that it is
  one 9→D matrix; it has no structure tying channel `k` and channel `k+3` to the same colour.

**3c. Which ViT.** Two implementations, both in `stack/tanitad/models/encoder.py`, **both trained
from scratch** (`torch.zeros` / `trunc_normal_` init — no pretrained weights anywhere in the file):

| | `ViTEncoder` (`:97`) | `ViT5Encoder` (`:319`) |
|---|---|---|
| patch embed | `Conv2d(in_ch, d, k=16, s=16)` | same |
| positional | learned **absolute** `[1, N, D]`, trunc_normal 0.02 | **joint** learned APE **+ 2D axial RoPE** (θ=100 patches / 10000 registers) |
| norm | LayerNorm | RMSNorm |
| attention | `nn.MultiheadAttention` | QK-Norm, bias-free qkv/proj, SDPA |
| extras | — | LayerScale 1e-5; **4 register tokens, stripped before return** (`:406`) |
| MLP | GELU, ratio 4.0 | GELU, ratio 4.0 (**SwiGLU deliberately rejected**, `:190-193`) |
| CLS token | **none** — all patch tokens returned | none |

Defaults `--enc-dim 384 --enc-depth 8 --enc-heads 6` (`train_v6_staged.py:6616-6618`).

⚠️ Both encoders **hard-refuse** a geometry that does not match the config (`encoder.py:154-160`,
`:386-390`) because the APE is a table sized for exactly one grid. RoPE makes the encoder
*extrapolatable* but the APE keeps the guard. This is correct and it has already caught one
mis-launched run (`encoder.py:199-201`).

---

## 4. ⚠️ The readout — the PI's spec and the register row are BOTH right, about different arms

`readout.py:38-113` (`SpatialGridReadout`). Pooling is `AvgPool2d((th//grid, tw//gw))` when the
grids tile, else an ONNX-exportable matmul pair. On a 16 × 40 token grid:

| arm | flags | readout | pooling kernel | azimuth bins over 120° | **°/bin** | `state_dim` |
|---|---|---|---|---|---|---|
| **default** (`train_v6_staged.py:6636-6638`) | `--readout-grid 4`, `grid_w=None`, `--readout-dim 128` | **4 × 4** | `AvgPool2d((4, 10))` | **4** | **30.00°** | 4·4·128 = **2048** |
| **`rdw8p30k`** (parity-scale, `MODEL_REGISTRY.md:3879`) | `--readout-grid 4 --readout-grid-w 8 --readout-dim 64` | **4 × 8** | `AvgPool2d((4, 5))` | **8** | **15.00°** | 4·8·64 = **2048** |

➕ **NEW, and it is the elegant part of the design:** `state_dim` is **2048 in both**. `rdw8`
buys 2× the azimuth resolution by halving the per-cell channel width, at *constant* compact-state
size — so the whole downstream stack (predictor, tactical/strategic policies, every grounding
head) loads unchanged. `readout.py:52-60` warns that `grid_w` "breaks the firewall"; with the
compensating `--readout-dim 64` it does not.

`MEASURED-COMPUTE` — what those bins mean on the ground:

| readout | °/azimuth bin | bin width at 10 m | at 30 m |
|---|---|---|---|
| 4 × 4 (default) | 30.00° | 5.36 m | **16.08 m** |
| 4 × 8 (`rdw8`) | 15.00° | 2.63 m | 7.90 m |
| *nuScenes-convention BEV cell, 0.5 m at 30 m* | *0.955°* | — | 0.5 m |

Elevation is 11.36°/bin in both (4 rows over 45.46°).

⇒ The PI's "4 × 8" is correct for the v7/`rdw8` line. The register's "4 azimuth bins, 30°/bin"
is correct for the default/v6 line. **A blanket statement about "our readout" is unsafe; the
axis must be reported per arm.** Which geometry the *scaled* ~336.5 M arm uses is being
established by a parallel probe and is not asserted here.

---

## 5. Temporal context — the honest arithmetic

| quantity | value | source |
|---|---|---|
| frames inside one encoder input | 3, at 100 ms spacing | `comma2k19.py:636-638` |
| ⇒ encoder's own temporal span | **0.2 s** | computed |
| predictor window | 6 steps | `train_v6_staged.py:6647`, `config.py:54` |
| step spacing | 0.1 s | `train_v6_staged.py:7015` |
| ⇒ span of the 6 step-anchors | 0.5 s (6 anchors ⇒ 0.6 s inclusive) | computed |
| ⇒ **total distinct wall-clock coverage** | **0.7 s** — earliest frame `t₀−0.2`, latest `t₀+0.5` | computed |
| ⇒ **distinct frames the stack ever sees per window** | **8** | computed |
| predictor multi-horizon targets | `(1, 2, 4)` steps = 0.1 / 0.2 / 0.4 s | `train_v6_staged.py:6648` |
| hierarchy cadences | op 10 Hz, tac 2 Hz (every 5 ticks), str 0.5 Hz (every 20) | `v6.py:3706-3708`, `:4228`, `:4233` |
| **plan horizon** | **`PLAN_STEPS = 60` → `HORIZON_S = 6.0 s`** | `v6.py:152-154` |
| operative band | 0.0 – 2.0 s | `v6.py:155` |
| tactical band | 2.0 – 6.0 s | `v6.py:156` |

⚠️ **CORRECTION, and it changes the field comparison.** The PI's spec said *"prediction horizon
20 steps at dt 0.1 = 2.0 s dense"*. That is the **operative band**. The binding horizon —
under a heading that names the PI's own 2026-08-11 decision — is **60 steps = 6.0 s**, and the
source comment states the transition explicitly: *"Emission k scales **20 → 60**; roll selection
rolls to 6 s."* The `--plan-steps` CLI default resolves to this constant (`train_v6_staged.py:7014`
→ `v6.py:152` via the import at `:123`).

⇒ **Our history : horizon ratio is 0.7 s : 6.0 s = 0.117**, not 0.6 : 2.0 = 0.3. This is the
number that must be compared against the field's conventions (nuScenes 2:6 = 0.333, Argoverse 2
5:6 = 0.833, Waymo Motion 1.1:8 = 0.1375 — all to be verified from primaries in Phase 2). The
correction cuts **both** ways: it makes our history look *relatively* less anomalous against
Waymo, and it makes the **absolute** shortfall worse, because 0.7 s of evidence is now being
asked to support a **6.0 s** extrapolation rather than a 2.0 s one.

⚠️ The PI's "0.6 s of history" is the *step-anchor* span. The **distinct-frame** span is 0.7 s
and the **encoder's** span is 0.2 s. These are three different numbers and reports have been
conflating them; the one that should be compared against the field's history conventions is
**0.7 s**, and the one that governs what a *single* encoder forward can observe (velocity yes,
acceleration marginally, jerk no) is **0.2 s / 3 frames**.

⚠️ **`--plan-steps` is a module constant `PLAN_STEPS`, not a literal** (`train_v6_staged.py:7014`).
The "20 steps = 2.0 s" claim is **not confirmed in this file** and is marked `UNVERIFIED` here
rather than repeated. `CLAUDE.md`'s own DE-C152 corollary applies: a derived constant silently
changes the experiment when its input changes.

---

## 6. The rig asymmetry — known, instrumented, and NOT automatically fixed

`calib.py:1029-1040` (retraction class **C26**) and `calib.py:1167-1205`:

- The **deployed crop** replicate-**pads** rows outside the sensor: 0.0017 % on rig A,
  **8.897 % on rig B**, n = 3,000. *"~73 % of training frames carry fabricated rows the rest do
  not, in a pattern that identifies the rig."*
- `cylindrical_rectify` (`calib.py:960-1001`) removes the **fabrication** — unobserved pixels
  become **honest black zeros** (`padding_mode="zeros"`, then `out = out * mask`) — but **not the
  asymmetry**. A rig-correlated black region is still a free rig label.
- `assert_fully_observed` (`calib.py:1167`) is a **zero-tolerance guard** that REFUSES
  256 × 640, and `largest_fully_observed_subframe` computes the fix: **176 × 624**, a pure
  `[40:216, 8:632]` slice, 0.000000 % masked for all 3,000 clips.
- `require_per_clip=True` (`calib.py:981-988`) REFUSES a corpus-median principal point, because
  its `cy` is a rig-B value and centring a ray fan on the wrong rig is the ~215 px D-016 error.

⇒ The **instrumentation is excellent**; the open question is purely **whether the live arms pass
`--v2-subframe 176x624`**. Corpus tags say the *cache* is 256 × 640; the slice happens at load.
A parallel probe is resolving this per arm. **If it is not passed, ~9 % of the input on 72.9 % of
clips is a constant black region that identifies the rig — a shortcut of exactly the kind
`CLAUDE.md`'s leak doctrine forbids.**

---

## 7. ⭐ THE ONE CANDIDATE DEFECT I FOUND: the build point-samples a 3× decimation

**7a. The operator.** `MEASURED-SOURCE`:

- `physicalai.py:544-552` — frames are decoded from the mp4 at **native** resolution
  (`physicalai.py:524` names *"a 604-frame 1080p clip"*), then batched into `_remap_batch`.
- `physicalai.py:520` / `:577-579` — `_remap_batch` dispatches to `cylindrical_rectify`.
- `calib.py:989-993` — the one line that does the work:
  ```python
  grid, mask = cylindrical_grid(intr, h, w, frame, device=vid.device)
  out = F.grid_sample(vid.float(), grid.expand(t, -1, -1, -1), mode="bilinear",
                      padding_mode=padding_mode, align_corners=False)
  ```

`grid_sample(mode="bilinear")` has a **fixed 2-source-pixel support**. It is a *reconstruction*
filter, not a *decimation* filter: it does not widen with the sampling ratio. There is **no
`antialias=` anywhere in the remap path** (the only two `F.interpolate` calls in `calib.py`,
at `:291` and `:553`, belong to the *crop* paths and are also plain bilinear).

**7b. The ratio.** `ESTIMATED` on the native pixel count, exact given it:

```
native  ~1920×1080 over ~120°  → 0.06250 °/px
output   640×256  cylindrical  → 0.18750 °/px (x),  0.17756 °/px (y at centre)
DECIMATION  x 3.000×   y 2.841×
```
⚠️ The 1080p figure is a **source comment**, not a manifest read — this is the ONE number in
this file that must be verified against a real mp4 header before anyone acts on it. If the
source is 3848×2168 (a common NVIDIA AV geometry) the ratio is worse, not better.

**7c. What the operator does — control-verified.** `MEASURED-COMPUTE`, raw:
`raw/alias_measure.py`. Controls: a DC image reads **exactly 0.00000** change under both arms;
a well-below-Nyquist sine reads **0.01996 vs 0.01994** (the two resamplers agree to 0.1 %), so
the panel measures aliasing and not passband differences.

A structure at **1.5× the output Nyquist** (one cycle per 4 native px) passes through bilinear
at its **full 0.45 amplitude** when the correct output amplitude is **0.15** — a **3.0× spurious
response**, and its phase depends on the sampling offset.

**7d. …and here is where I corrected myself.** `MEASURED-COMPUTE`, raw: `raw/alias_measure2.py`.
My first pass measured the frame-to-frame consequence on a **white-noise-textured** synthetic
frame and got **2.98×** excess. That frame is not a camera frame. Repeating it across controlled
`1/f^α` spectra with a lens/OLPF roll-off — the standard natural-image model — gives:

| frame spectrum | % power above output Nyquist | excess Δframe (bilinear ÷ prefiltered) |
|---|---|---|
| **CONTROL** — band-limited below Nyquist | 0.00 % | **1.02×** ✅ (must be 1.00) |
| soft image, 1/f²·⁴ + strong lens MTF | 0.00 % | 1.02× |
| **natural, 1/f² + lens MTF 0.20** | 0.10 % | **1.11×** |
| **natural, 1/f² + lens MTF 0.35** | 1.85 % | **1.31×** |
| crisp, 1/f¹·⁶, no roll-off | 40.97 % | 3.13× |
| white-ish, 1/f⁰·⁸, no roll-off | 76.75 % | 4.16× |

⇒ **RETRACTED within this session: the "≈3×" headline was an artefact of an unrealistic test
image.** The defensible figure for a realistic camera frame is **1.11–1.31× excess frame-to-frame
change**, i.e. **10–30 % of the per-frame image delta the world model is trained to predict is
sampling phase rather than scene** — real, but second-order.

⛔ **A third panel I ran — excess as a function of thin-feature width — is DISCARDED, not
reported.** It returned values *below* 1.0× (0.19–0.38×) and was non-monotonic in feature width.
Both are impossible for an aliasing measure, and the cause is in my own construction (feature
spacing `9 × width` is commensurate with the decimation ratio 3 for several rows). Per the
programme's probe doctrine — *a control that reads a wrong value invalidates the panel* — the
per-structure claim is **UNMEASURED**, and the thin-structure question stays open.

**7e. Why it is nevertheless worth a line item.** Aliasing is *phase-dependent*: as the ego
moves, the sub-pixel phase of every scene structure changes, so aliased content changes
frame-to-frame in a way that is **not a function of the scene**. A predictor cannot predict it,
so its loss-optimal output for that component is its **conditional mean** — a mechanism that
pushes a world model toward reproducing the marginal. That is the programme's headline failure
mode, arriving from the input rather than the objective. At **1.11–1.31×** it is a contributing
factor at most, not the cause; and the fix (a box/Gaussian prefilter before `grid_sample`) is
one code change plus a cache rebuild that **breaks parity**, which is the real cost.

---

## 8. Open items this file could not close

**All but two were closed by parallel probes within the same session. The answers are in `RESULT.md`; recorded here so this file is not read as still-open.**

| # | item | status |
|---|---|---|
| 1 | native mp4 resolution **and frame rate** | 🟡 **PARTIAL.** `FThetaIntrinsics` defaults `width=1920, height=1080` (`calib.py:349-350`) agrees with the `physicalai.py:524` comment — two probes, but still not a header read. Frame rate ~**30.2 fps** is inferred from that same comment ÷ 20 s; the adapter declares **no** fps constant (3 probes) ⇒ a ~3× *temporal* decimation, `ESTIMATED`. One `ffprobe` closes both. |
| 2 | codec in the parity caches | ✅ **CLOSED — PNG lossless.** `parity_manifest.json:2514-2536` `"codec": "png"`, `"quality": null`, both train and val. The consumer selects on that field (`v2_dataset.py:127`), not on the `jpeg_buf` name; ⚠️ its fallback default is `"jpeg"`, so a payload *missing* the field would be silently mis-decoded. |
| 3 | whether live arms pass `--v2-subframe 176x624` | ⛔ **CLOSED — THEY DO NOT.** `v2_subframe: null` on the 336.5 M config E, `rdw8p30k`, `champ30k` and every arm evaluated today; it was live only for v5f. **The C26 rig shortcut is live.** See `RESULT.md` §3.12 / §4.0. |
| 4 | geometry + readout of the scaled ~336.5 M arm | ✅ **CLOSED.** `v6F-SW-30k.config.json`: 256×640, patch 16, readout **4×4**×128, window 6, in-ch 9, **`--vit5-encoder` with 4 registers**, enc 768/12/12, **336,542,025** params (encoder 87,284,736 = 25.94 %). ⚠️ **Not in `MODEL_REGISTRY.md`** — see `RESULT.md` §5 item 9. |
| 5 | augmentation / normalisation constants | ✅ **CLOSED.** Augmentation exists but is off and never touches the supervised path; normalisation is a bare `/255.0` (`_contract.py:51-56`); ImageNet constants are used only for the frozen DINOv3 teacher. |
| 6 | any recorded **inference** latency | ✅ **CLOSED, and it cuts both ways.** Thor end-to-end **60.30 ms p50** inside a 100 ms budget exists for **v1/v5f** — but **zero** measurements exist for any **v6/v7** arm. |
| ~~7~~ | ~~`PLAN_STEPS` value~~ | ✅ **CLOSED — `v6.py:152`, `PLAN_STEPS = 60`, horizon 6.0 s. See §5.** |

`MEASURED-REGISTRY`, the one hardware number I can quote: arm `rdw8p30k` ran **30,000 steps at
batch 8 in 29,758 s on a Jetson Thor** (`MODEL_REGISTRY.md` §13.0, `summary.json done: true`)
= **0.992 s/step**, ≈ **8.1 windows/s**. That is **training** throughput for a 19.3 M-param arm;
it is **not** an inference latency and must not be quoted as one.

---

## 9. Reproduction

All three measurement scripts are banked beside this file under `raw/` and are pure-Python /
pure-torch, no data and no corpus access:

| script | what it establishes | runtime |
|---|---|---|
| `raw/proj_measure.py` | FOV, angular resolution, lane-line bowing, apparent-size vs azimuth, object px vs range, readout bin footprints | < 1 s |
| `raw/proj_measure2.py` | worst-case bowing sweep, per-patch locality, pinhole comparison | < 1 s |
| `raw/alias_measure.py` | decimation ratio, shift-induced change vs frequency, amplitude error | ~10 s |
| `raw/alias_measure2.py` | the spectrum-hardened retraction of 7d, with the band-limited control | ~30 s |

Run with `C:/Users/Admin/venvs/tanitad/Scripts/python.exe` (the repo venv). No GPU used; the
RTX 4060 was checked free (2,049 MiB / 8,188 MiB, 8 % — display only) and Thor was not touched.
