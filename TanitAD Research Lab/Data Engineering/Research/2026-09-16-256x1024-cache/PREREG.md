# PRE-REGISTRATION — 256x1024 cylindrical v2ep cache, B1 EVAL 139

**Written 2026-09-16, BEFORE any comparison was measured.** Metrics and
tolerances below are fixed. If a check fails, the failure is reported; the
tolerance is not moved to make it pass.

## 0. What is being validated

The PI's decision: raise azimuth resolution from `256x640` cylindrical over 120
deg to `256x1024` over the same 120 deg, for all future trainings. Validated on
the 139 B1 EVAL clips.

`f_ref` is solved from the field, not copied:
`HFOV = 2*(W/2)/f_ref` (cylindrical is LINEAR in azimuth) so

| frame      | f_ref (derived)  | HFOV      | deg/column | centre column |
|------------|------------------|-----------|------------|---------------|
| 256 x 640  | 305.5774907364   | 120.000000| 0.187500   | 319.5         |
| 256 x 1024 | 488.9239851783   | 120.000000| 0.117188   | 511.5         |

## 1. Exact grid correspondence (derived analytically, before measuring)

`calib.cylindrical_rays`: `phi = (u - (W-1)/2)/f_ref`, `y_n = (v - (H-1)/2)/f_ref`
— the SAME `f_ref` on both axes.

* **Columns.** 1024-col `k` and 640-col `j` see the same azimuth iff
  `k = 1.6*j + 0.3`. A 1/1.6 resize with `align_corners=False` maps output
  column `c` to input column `1.6c + 0.3`, so **output column c == 640 column c,
  over the full width.**
* **Rows.** 1024-row `i` and 640-row `i'` see the same elevation iff
  `i = 127.5 + 1.6*(i' - 127.5)`. The 1024 frame's rows 0..255 therefore span
  640-frame rows **47.8125 .. 207.1875**, and a 1/1.6 resize maps output row `r`
  to 640-frame row **48 + r** exactly.

⇒ **The identity comparison is `resize(frame1024, 1/1.6) [160x640]` against
`frame640[48:208, :]`** — integer-aligned on both axes, no interpolation of the
reference.

⇒ **Consequence, stated here because it is a decision, not a detail:** holding
H = 256 while f_ref rises 1.6x shrinks the VERTICAL field
**45.4556 deg -> 29.3415 deg** (a 35.4 % loss). 408x1024 would restore it
(45.2960 deg). See RESULT.md §ESCALATION.

## 2. Checks, metrics, tolerances (all fixed before measuring)

### G0 — builder parity vs the DEPLOYED 640 cache
Rebuild, with this driver at `--width 640 --height 256`, >= 2 clips that are in
Thor's deployed `physicalai-b1-w120-256x640cyl`, and compare to Thor's banked
payload.
* `n_frames` **must be equal** (exact).
* `poses` and `actions`: `max |delta| <= 1e-6` (exact-arithmetic path).
* pixels: **MAE <= 1.0 level**, **PSNR >= 40 dB** over all sampled frames.
  A non-zero floor is allowed because Thor is aarch64 and this box is x86-64:
  libavcodec h264 output and `grid_sample` float reduction may differ in the
  last ulp. Byte-identity is not claimed and not required.

### G1 — downscale identity, GEOMETRY (primary, pass/fail)
On each sampled frame, integer-shift cross-correlation (mean-subtracted, search
window +-12 px in both axes) between `resize(frame1024,1/1.6)` and
`frame640[48:208,:]` must peak at **(dy, dx) = (0, 0)**. Any mirrored azimuth,
unscaled `f_ref`, or off-by-N crop moves this peak.

### G2 — downscale identity, PHOTOMETRIC (primary, pass/fail)
On each sampled frame, over the region observed in BOTH images:
* **median |delta| <= 2.0 levels**
* **MAE <= 4.0 levels**
* **PSNR >= 28 dB**

A non-zero tolerance is pre-registered with its reason: the two images are not
the same operation. The 640 payload is a bilinear POINT sample of the 1920x1080
source on a 640-column grid (no prefilter); the downscaled-1024 image is an
antialiased average of a 1024-column bilinear sample. Same geometry, different
kernel, so high-frequency content must differ. **Exact equality would be
evidence of a bug** (the 1024 build silently falling back to 640).

### G3 — deliberate regressions; these MUST FAIL G1/G2
Guards are proven by mutation, not inspection.
* **M1 unscaled f_ref**: build one clip at W=1024 with `f_ref` left at
  305.5774907 (the "forgot to scale the focal" defect). Must fail.
* **M2 wrong row crop**: compare the downscaled 1024 against `frame640[0:160,:]`
  instead of `[48:208,:]`. Must fail (expected peak at dy = -48).
* **M3 mirrored azimuth**: compare against a horizontally flipped 1024 frame.
  Must fail.

### G4 — frame / label alignment survives
For every clip with a SAM3 map (135 of 139 at
`D:/Projects/TanitAD-artifacts/sam3-maps-eval/`), recompute the episode time grid
from the staged `timestamps.parquet` exactly as `v2_compressed._resampled` does
and require, **exactly**:
* `t_query == npz.t_query_us`
* `frame_idx == npz.cam_frame_idx`
* `t_frames[frame_idx] == npz.t_img_us`
* `payload n_frames == len(npz.t_img_us)`

And against the banked token index
`C:/Users/Admin/tanitad-caches/bevhead-20260913/tokens/index.npz`, for all 139:
* `index.clip_sha12` set == the 139 built sha12
* per clip, `max(raw_frame) + 1 == payload n_frames`
* per clip, `n_rows == payload n_frames - (n_stack - 1)` and
  `stacked_row == raw_frame - (n_stack - 1)`

### G5 — geometry end to end, through `tanitad/data/rig_projection.py`
* A point straight ahead (azimuth 0) must land on column **511.5** at W=1024
  (tolerance `1e-6` px) and on **319.5** at W=640.
* The **mirrored** azimuth must FAIL the same check
  (`R-2026-09-08-wpa-mirror` defect class): asserting a sign-flipped azimuth
  lands on the same column must raise / return False.

### G6 — token geometry, MEASURED not derived
Run a real `timm` `resnet101` `features_only` model on one real 256x1024 frame
and one real 256x640 frame and read the feature-map shapes off the tensors:
* stride 16 -> **16 x 64 = 1024 tokens**, 120/64 = **1.875 deg/column**
  (today: 16 x 40 = 640, 3.000 deg/column)
* stride 32 -> **8 x 32 = 256 tokens**, 120/32 = **3.750 deg/column**
  (today: 8 x 20 = 160, 6.000 deg/column)

## 3. Sampling plan (fixed before measuring)

* **>= 5 clips, >= 20 frames** for G1/G2. Actual: **6 clips x 5 frames = 30**.
* Clips: the first 6 of `eval_sha12.txt` in file order (no selection on content).
* Frames per clip: `numpy.linspace(0, n_frames-1, 5)` rounded — deterministic,
  spread over the clip, not chosen after looking at the data.

---

# ADDENDUM — 408x1024, the field-preserving alternative

**Written 2026-09-17, BEFORE the 408 comparison was measured.**

## A1. Why 408 is not integer-aligned to the 640 grid, and what follows

To be an exact 1.6x supersample of 256x640 in BOTH axes the height must be
**256 x 1.6 = 409.6** — not an integer. So no integer height is simultaneously
field-exact and integer-aligned. At f_ref 488.9239852:

| H | VFOV | resize H/1.6 | 640-row of output row 0 |
|---|---|---|---|
| 256 | 29.3415° | 160 | **+48.0** (integer) |
| 400 | 44.4952° | 250 | **+3.0** (integer) |
| **408** | **45.2960°** | **255** | **+0.5** (HALF-PIXEL) |
| 416 | 46.0921° | 260 | **-2.0** (integer) |

⚠️ The half-pixel offset at H=408 is a property of the **comparison**, not a
defect of the cache: the 408 geometry is perfectly well-defined and nothing
downstream cares. It only means the identity check cannot be integer-aligned,
so the check must pay for one half-pixel resample of the reference.

## A2. The control that prices the half-pixel — measured BEFORE judging 408

**C1 (half-pixel penalty).** Re-run the ALREADY-PASSED 256x1024 comparison with
the reference shifted by +0.5 px vertically
(`0.5*(ref640[48:208] + ref640[49:209])`) and record the degradation in median /
MAE / PSNR against the unshifted numbers. That delta is the price of the
half-pixel, measured on a case already known to be geometrically correct.

## A3. G1/G2 for 408x1024 (tolerances fixed here)

Comparison: `resize(frame408,(255,640))` against the half-pixel-interpolated
reference `0.5*(frame640[0:255] + frame640[1:256])`, which is the band the 408
frame actually corresponds to.

* **G1 geometry**: x-corr peak at **(0,0)**, same +-12 window. Unchanged.
* **G2 photometric**: base tolerance **widened by the C1-measured penalty**, i.e.
  `median <= 2.0 + dC1_median`, `MAE <= 4.0 + dC1_mae`,
  `PSNR >= 28.0 - dC1_psnr`. The widening is justified by a measurement, is
  fixed before the 408 numbers are seen, and is the ONLY widening permitted.
* Same sampling plan: 6 clips x 5 `linspace` frames = 30.
* **G3 mutations re-run at 408**: wrong row crop and mirror must both still FAIL.

## A4. What each geometry keeps of the source — stated plainly

Expressed as rows of today's 256x640 frame (`v' = 127.5 + (v-(H-1)/2)/1.6`):

| geometry | 640-rows covered | keeps |
|---|---|---|
| 256x640 | 0 … 255 | everything it has today |
| 256x1024 | 47.81 … 207.19 | **drops the top 47.8 and bottom 48.8 rows** |
| 408x1024 | -0.31 … 254.69 | **the whole band** (0.3 row short at each end) |

## A5. Activation cost — what will be measured and what will NOT

Measured: summed bytes of every leaf module's output tensor, one forward,
batch 1, `in_chans = 9` (K=3), fp32, **on CPU** (the GPU belongs to another
agent). This is NOT a CUDA `max_memory_allocated` figure and must not be
compared to one as if it were the same quantity; the decision-relevant output is
the **ratio** between geometries.
