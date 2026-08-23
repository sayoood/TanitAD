> **See also `STRATEGIC_FAMILY.md`** (2026-08-03) — the map/junction half of this scene, and the
> survey of all 1607 NuRec scenes for a junction traversal with an actual branch. Headline: the
> night clip is **inside a junction for 46 of its 202 poses**, so "no junction" was FALSE — but
> every traversal has exactly **one** admissible continuation, so "no junction-scale decision" was
> TRUE. **141 of 1607 scenes** contain a real junction turn; the recommended one is downloaded to
> Thor.

# ⭐ CRACKED: NuRec scenes render on the Jetson Thor with gsplat

**MEASURED 2026-08-02, Thor (aarch64 Blackwell `sm_110`).** The one open unknown blocking
AlpaSim-on-Thor — *does the NuRec gaussian payload map onto gsplat's parameterisation?* — is
**ANSWERED YES**. We can render NVIDIA's neural reconstructions ourselves, on the edge device,
without the closed x86-only NRE binary.

![render vs reference](sbs_frame0_render_vs_reference.png)

*Left: our gsplat render. Middle: NVIDIA's shipped `camera_front_wide_120fov.mp4` frame 0.
Right: difference.*

## ⛔⛔ 2026-08-03 — EVERY SCALAR BELOW THIS LINE IS RETRACTED (wrong quaternion layout)

**Root-cause class: a headline table copied from a superseded run directory.**
All numbers in the two tables below came from `thor:~/nurec_work/out_xyzw/report.json`, the run
with `quat_layout = xyzw` — **the layout the scene's own geometric self-test REJECTED**. The
definitive run is `final/report.json` (`quat_layout = wxyz`); the two are trivially separated by
`render_mean` (0.2404 vs 0.2762) and nobody checked. Corrected values:

| quantity | RETRACTED (`out_xyzw`) | **CORRECT (`final`, wxyz)** |
|---|---|---|
| grad-NCC vs correct frame 0 | 0.2719 | **0.3802** |
| best WRONG frame | 0.1913 (f300) | **0.2110** (f300) |
| negative-control margin | +0.0806 | **+0.1692** |
| PSNR (context only) | 16.758 | 17.012 |
| render mean | 0.2404 | 0.2762 |
| affine gain R/G/B | 0.485 / 0.446 / 0.434 | **0.463 / 0.445 / 0.444** full-frame; **0.698 / 0.462 / 0.719** on covered pixels |

The verdict is unchanged — the negative control PASSES either way — but the margin is **2.1×
larger** than reported. ⇒ *Quote a run directory, not a number.*

**⛔ ALSO RETRACTED: "the residual is the scene's trained per-frame ISP" (the "next step 1" below).**
The PPISP parameters were found at `.post_processings.0.ppisp.*`, decoded and applied
(`ppisp.py`, `isp_experiment.py`). `exposure` is **exactly 0 for all 3594 views**, `color` is
**identical for all 3594 views**, vignetting `max|α| = 0.0047` — combined effect **`mean|Δ| =
0.0018` (0.18 %)**, because the config sets `per_frame_ppisp_enabled: false`. **The scene ships
no per-frame photometry at all.** And the "near-equal per-channel gain" is an artifact of the
**79–81 % of the absolute error that lives in pixels no gaussian covers**: restricted to covered
pixels the channels spread 1.55× at frame 0 and 3.4× at frame 450 (`0.960 / 0.286 / 0.471`).
Full corrected write-up: the 2026-08-03 agent report; measured numbers in `isp_report.json`.

## ⛔ RETRACTION: PSNR IS NOT A VALID METRIC ON THIS CLIP (this one STANDS)

**MEASURED 2026-08-03, negative control.** Our ONE render of frame 0 was scored against the correct
reference frame **and against four wrong ones**. If the mapping were right and PSNR meaningful, the
correct frame must win. It does not:

| ref frame | PSNR (dB) | NCC | **grad-NCC** |
|---|---|---|---|
| **0 — CORRECT** | 16.758 | 0.704 | **0.2719** |
| 60 (wrong) | 17.062 | 0.727 | 0.1436 |
| 150 (wrong) | **17.457** ← *beats the correct frame* | 0.767 | 0.1737 |
| 300 (wrong) | 16.756 | 0.708 | 0.1913 |
| 450 (wrong) | 17.073 | **0.782** ← *beats it too* | 0.1163 |

⛔ **PSNR ranks a WRONG frame first (150: 17.457 > 16.758), and so does plain NCC (450: 0.782 >
0.704).** Every frame of this clip is a dark night street, so ~17 dB measures *"both images are
dark"*, not *"the pose is right"*.

⇒ **I retract the earlier framing of "PSNR 16.758 → 20.689 after colour fit" as evidence of render
quality.** It is not evidence of anything on this clip. The affine-fit number is doubly inadmissible.

✅ **What DOES validate the mapping: gradient-NCC.** It picks the correct frame —
`argmax_grad_ncc = 0`, margin **0.0806** over the best wrong frame (0.2719 vs 0.1913, ~1.42×).
Structure — edges, the lamp post, the road vanishing point — correlates with the right frame and not
with the others. **That, plus the visual side-by-side, is the whole of the evidence that the payload
decodes correctly.** It is enough to proceed; it is not a photometric result.

⭐ **Transferable rule: on a low-dynamic-range corpus (night, fog, tunnel), pick the discriminating
metric by running the negative control FIRST.** A metric that cannot tell the right frame from a
wrong one cannot certify anything, however reasonable its value looks.

## The falsifier, and what it says

The scene ships its own reference video, so a wrong mapping cannot hide behind a plausible-looking
image. Frame 0, `background + road` layers:

| metric | value |
|---|---|
| gaussians rendered | **3,105,514** |
| render time | 224.98 ms (frame 0, unoptimised, 1920×1080) |
| mean alpha | 0.5145 |
| dropped gaussians | background 348, road 0 |
| **PSNR** | **16.758 dB** |
| MAE | 0.1174 |
| ref mean / render mean | 0.2659 / 0.2404 |
| **PSNR after affine colour fit** | **20.689 dB** |
| affine gain/bias per channel | R (0.485, 0.167) · G (0.446, 0.166) · B (0.434, 0.137) |

**Geometry, camera pose and scene content are unmistakably correct** — the same street lamp in the
same place, the same road vanishing point, the same buildings, the same lit signage. That is the
claim the reference exists to test, and it passes by inspection, not just by a scalar.

⚠️ **It is NOT yet a pixel match, and the numbers say where the gap is.** A near-identical affine
gain on all three channels (**0.485 / 0.446 / 0.434**, i.e. ~0.45) is the signature of a **colour-space
transfer error, not a geometry error** — almost certainly linear-vs-sRGB, and the file itself
declares `config/background/composite_in_linear_space: False`. Visible artifacts: a black band top-left
(FOV/coverage), a magenta smear bottom-left (a bad gaussian cluster or a missing layer), and general
haze (the sky env-map cubemap is not yet composited — `--sky` was off).

⇒ **Do not quote 16.758 dB as "the render quality".** It is the quality of a render with an
un-corrected colour transform and no sky. The honest statement is: *the payload decodes and the
scene reconstructs; photometric agreement is not yet established.*

## What the loader established about the format

- `volume.nurec` = **gzip → plain MessagePack**. No proprietary container.
- Every array carries an **explicit `<key>.shape` list** in the file — shapes are read, not guessed.
- Dtype is **float16**, *proven* by cross-checking every declared shape against its raw byte count
  (`nurec_loader.py`) rather than assumed.
- Band-0 SH constant `C0` with the standard `colour = 0.5 + C0·dc` convention (gsplat applies the
  `+0.5` internally).
- Layers present: `background`, `road`, `dynamic_rigids`, `dynamic_deformables`. **Only the first
  two are rendered here — dynamic actors are out of scope for this pass.**

## Next, in order

| # | step | note |
|---|---|---|
| 1 | Apply the scene's **trained per-frame ISP** (exposure / vignetting / colour / CRF) | `render_probe.py:259` — the ~0.45 affine gain is most likely THIS, not linear-vs-sRGB. Find its parameters in the msgpack. |
| 2 | ⛔ **Sky compositing MADE IT WORSE — do not just switch it on.** | MEASURED: `--sky` moves render mean **0.240 → 0.391** against a reference of **0.266**, and PSNR **16.76 → 15.32**. The env map overfills the ~49 % of pixels no gaussian covers (`mean_alpha` 0.5145). Gate the sky on depth/alpha, or the night sky gets painted bright. |
| 3 | Investigate the magenta smear + black band | localised, not global |
| 4 | Wrap as a `sensorsim.proto` gRPC service, **front camera only** | PI's explicit steer |
| 5 | `alpasim_wizard … renderer=<ours> driver=<tanitad/refc/flagship-v1>` | adapters already exist |
| 6 | LONG closed-loop videos: front camera + overlays + BEV | the deliverable |

⚠️ **Perf note:** 225 ms/frame at 1920×1080 with 3.1 M gaussians is **~4.4 FPS**, not the 492 FPS
measured on the 20 k-gaussian synthetic probe. Closed loop at 10 Hz needs work — but the front
camera alone at a lower raster (the PI's steer) is a large reduction, and this figure includes
first-call overhead. **Re-measure before treating it as the deployment number.**

## Evidence class

| claim | class |
|---|---|
| every metric in the table | **MEASURED (ours)** — `render_probe.py` on Thor, artifact `out/sbs_frame0000_f0_background-road.png` |
| "geometry and pose are correct" | **MEASURED by inspection** of the banked side-by-side |
| "the residual is a colour-space transfer" | ⚠️ **HYPOTHESIS** — strongly supported by the near-equal per-channel gains + the file's own `composite_in_linear_space: False`, **not yet confirmed by a corrected re-render** |
| 4.4 FPS is the deployment rate | ⛔ **NOT ESTABLISHED** — one unoptimised frame at full 1080p with sky off |
