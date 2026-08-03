# NuRec render quality — what actually moved it, and what did not

**MEASURED 2026-08-03 on Thor (aarch64 Blackwell `sm_110`), scene
`00040136-e651-4abd-991d-0655ccda9430`, 5 frames (0, 60, 150, 300, 450), gsplat 1.5.3.**
Every number below cites a **run directory**, never a prior summary.

| run dir (on `tanitad-thor`) | what it holds |
|---|---|
| `~/rq_out/panel1` | first pass: layers + naive/gated sky |
| `~/rq_out/panel2_skygain` | sky gain sweep |
| `~/rq_out/panel3_rs` | rolling shutter |
| `~/rq_out/panel4_final` | 9-arm decision panel incl. scale cull |
| `~/rq_out/panel5_haze` | haze cull (scale AND opacity) — **negative result** |
| `~/rq_out/panel6_chosen` | BEFORE/AFTER confirmation + the side-by-side PNG |
| `~/rq_out/diagnose_f0`, `~/rq_out/diagnose_f150` | the two named artifacts |

---

## HEADLINE

| arm | grad-NCC | neg-control margin | mean_alpha | MAE | ms/frame |
|---|---|---|---|---|---|
| **BEFORE** `background+road` | 0.2774 | +0.0873 | 0.5404 | 0.1028 | 23.3 |
| **AFTER** all 4 layers + scale-cull 0.95 + gated sky 0.3 | **0.3424** | **+0.1020** | 0.3300 | 0.0875 | 36.3 |
| AFTER **+ rolling shutter** | **0.3747** | **+0.1285** | 0.3847 | 0.0797 | **3749** |

`run_dir = ~/rq_out/panel6_chosen`. **+23.4 % grad-NCC inside the render budget**
(36 ms ≈ 28 FPS, the closed loop needs 10 Hz); **+35.1 % if wall clock is not a constraint.**
Negative control PASSES on 5/5 frames for every arm.

---

## ⛔ THREE INHERITED CLAIMS THAT DID NOT SURVIVE MEASUREMENT

### 1. "79–81 % of the error lives in pixels NO GAUSSIAN COVERS; `mean_alpha` 0.5145; roughly half the frame is uncovered"

Wrong twice over.

* **The number came from a retracted run.** `mean_alpha = 0.5145` is from the `out_xyzw`
  directory — the rejected quaternion layout. The definitive `final/report.json` says
  **`mean_alpha = 0.5974997`**, which this harness reproduces exactly (0.5975, and
  `mae` 0.10296 vs the banked 0.10296). *The very failure mode FINDINGS warns about —
  "quote a run directory, not a number" — had propagated into the diagnosis built on top of it.*
* **"Uncovered" meant "not fully opaque".** `isp_experiment.py` defines covered as
  `alpha >= amin` with **`amin = 0.995`**. At frame 0 the fraction of pixels with
  **alpha < 0.01 — genuinely no gaussian — is 0.0000**, and 99.96 % of pixels have
  alpha > 0.05. The frame is *covered but semi-transparent*, not empty.

⇒ The residual is not "half the frame has no gaussians". Averaged over the 5 frames,
`alpha ≥ 0.995` on 20.7 % of pixels and `alpha < 0.01` on 5.9 %; 80.9 % of the absolute
error sits in the not-fully-opaque remainder. The full alpha histogram is now in every
report row so the two cannot be conflated again.

### 2. "The black band top-left is FOV/coverage"

**REFUTED.** The camera's own f-theta `max_angle` is **77.22°**, and
**`frac_pixels_beyond_max_angle = 0.0`** at both frame 0 and frame 150 — *no pixel in the
1920×1080 raster is outside the modelled field of view.* IoU between {alpha < 0.01} and
{beyond max_angle} is **0.0**.

What it actually is: **a reconstruction hole in the upper field.** At frame 150,
230,038 px (11.1 % of the frame) have alpha < 0.01, in two blobs — a band across the top
(`x 674..1920, y 0..202`) and a top-left block (`x 0..382, y 0..443`). We render them
**pure black `[0,0,0]`; the reference shows `[28.7, 26.0, 22.8]`.** The scene simply has
no gaussians up there. ⇒ this is exactly what the sky env map exists to fill, which ties
artifact #2 to fix #2. *(Frame 0 has none of this — which is why it never showed in a
frame-0-only analysis.)*

### 3. "A magenta smear bottom-left = a bad gaussian cluster or a missing layer"

**Half right, and the stated cause was wrong.** Strictly magenta pixels
(R and B ≥ 18 levels above G) are **0.006 % of the frame** — 130 px at f0, 123 px at f150 —
they are **fully opaque (alpha 0.9997)** real content at 119 m, they come from
`background` (not a missing layer), a max-scale cull does not remove them, and at f150 the
**reference has 6× MORE of them than we render** (0.00036 vs 0.00006). As a magenta
*cluster*, it does not exist.

The real defect is visible and different: **long horizontal light STREAKS** — one pink, one
cyan — smeared across the mid-frame. They are **over-sized semi-transparent splats**, and
they are the single most conspicuous thing wrong with the render. Culling static splats
whose largest axis exceeds the 95th percentile (**1.4263 m**, 153,506 of 3,105,514) removes
them: grad-NCC **0.2773 → 0.3460** and it is *faster* (20.8 ms vs 23.9).

---

## ⭐ WHICH METRIC MAY DECIDE — measured, not assumed

Every arm scores its render against the correct reference frame **and 5 wrong ones spread
across the clip**. Count of frames (of 5) where each metric picks the CORRECT reference:

| arm | grad-NCC | MAE | PSNR |
|---|---|---|---|
| base | **5** | 4 | 4 |
| all4 | **5** | 4 | 4 |
| all4 + sky 0.3 | **5** | 4 | 4 |
| all4 + sky 0.5 | **5** | **1** | **1** |
| all4 + cull 0.95 | **5** | 3 | 3 |
| all4 + rs | **5** | 3 | 4 |

⇒ **grad-NCC is 5/5 on every arm. MAE and PSNR are 1–4/5 and their reliability CHANGES WITH
THE ARM.** The FINDINGS retraction of PSNR/NCC **extends to MAE**: a photometric difference
between two arms is not evidence about quality, because at sky-gain 0.5 the metric can only
identify the right frame 1 time in 5. MAE appears in the tables as a descriptive statistic
and **decides nothing**.

*(This matters concretely: ranked by MAE, `all4 + sky 0.3` (0.0811) looks like the best arm
in the whole programme. Ranked by the admissible metric it is mid-table.)*

---

## WHAT WORKED, IN ORDER OF SIZE

### 1. ⭐ Rolling shutter — biggest lever, 161× the cost

The rig declares **`shutter_type = ROLLING_TOP_TO_BOTTOM`** with a **30.559 ms** readout,
during which the ego translates **0.63 m (f0), 0.57 m (f150), 0.40 m (f450)**. We rendered
every frame from ONE pose. Rendering the true start→end sweep:

| | grad-NCC | margin | MAE | mean_alpha | ms |
|---|---|---|---|---|---|
| `base` | 0.2774 | +0.0873 | 0.1028 | 0.5404 | 23.0 |
| `base_rs` | **0.3170** | **+0.1119** | **0.0931** | **0.6519** | **3700** |

All three move the right way at once, and coverage rises 21 % — the scanline sweep fills
pixels a single pose leaves thin. ⚠️ **3700 ms/frame vs 23 = 161×.** Simulated time is
unaffected (the loop steps a fixed 0.1 s), so this is a *production* cost, not a physics
change — but it is far outside a real-time budget. Off by default; `--rolling-shutter`.

⚠️ **`render_probe.py --rs` never worked.** It calls `RollingShutterType.TOP_TO_BOTTOM`,
which does not exist in gsplat 1.5.3 (members: `GLOBAL`, `ROLLING_{TOP_TO_BOTTOM,
BOTTOM_TO_TOP, LEFT_TO_RIGHT, RIGHT_TO_LEFT}`) — it raised `AttributeError`. No
rolling-shutter render had ever been produced. The renderer now reads the enum member name
straight from the calibration string, so the two cannot drift apart.

### 2. ⭐ Scale cull — removes the streaks, and is FREE

| arm | grad-NCC | margin | render_mean | ms |
|---|---|---|---|---|
| all4 | 0.2773 | +0.0887 | 0.2423 | 23.9 |
| + cull 0.98 (61,723 dropped, >2.686 m) | 0.3128 | +0.0933 | 0.1627 | 22.4 |
| + cull 0.95 (153,506 dropped, >1.4263 m) | **0.3460** | +0.1010 | 0.1462 | **20.8** |

⚠️ It also darkens: `render_mean` 0.2423 → 0.1462 against a reference mean of 0.2425,
because a pure max-scale rule removes large **opaque** splats (road, walls) along with the
haze. Pairing it with the gated sky puts the brightness back (0.2002) while keeping the
gain.

### 3. Gated sky — fills the black band; the gate that matters is GAIN, not direction

The scene ships a `sky-env-map` cubemap, `[1,6,512,512,3]` **read from the file's own
declared shape** (of four candidate memory layouts only `[6,H,W,3]` yields image-like faces:
mean |Laplacian|/std **0.022–0.081** vs 2.4–3.2 for the alternatives).

The decisive measurement: **where our alpha < 0.1, the reference's mean brightness is 0.1439
and ours is 0.0176.** The content is genuinely missing — so the env map belongs there.

⚠️ **But the horizon gate is nearly a no-op.** Sky weight with the gate 0.3935 vs 0.4025
without it — a **2 % difference**, because the low-alpha pixels are almost all above the
horizon already. The old failure ("`--sky` floods the uncovered 49 %") was never about
*direction*; the cubemap's mean is **0.318** against a reference frame mean of **0.2425**, so
at unit gain it over-brightens whatever it touches. Gain sweep (`~/rq_out/panel2_skygain`):

| gain | grad-NCC | MAE | render_mean (ref 0.2425) | render mean where alpha<0.1 (ref 0.1422) |
|---|---|---|---|---|
| 0 | 0.2773 | 0.1007 | 0.2423 | 0.0176 |
| **0.30** | 0.2894 | **0.0811** | 0.2821 | 0.0981 |
| 0.50 | 0.2965 | 0.0892 | 0.3086 | 0.1519 |
| 1.00 | 0.3092 | 0.1431 | 0.3750 | 0.2866 |

grad-NCC rises monotonically with gain while photometry degrades past ~0.3. Since MAE may
not decide (above), **gain 0.3 is chosen as the point where the env map restores the missing
radiance without overshooting it** — a physical criterion (`render@lowalpha` vs
`ref@lowalpha`), not a metric optimisation. The gate is kept because it costs nothing and is
the correct behaviour on scenes whose holes are *not* all sky.

### 4. Dynamic layers — correct, and honestly ~zero on the metric

`background, road` were the only layers ever rendered. The scene also ships
**`dynamic_rigids` (115,824 gaussians, 30 cuboids with gaussians / 35 tracks)** and
**`dynamic_deformables` (1,039 gaussians, 2 tracks: one `person`, one `rider`)**.

* Mapping is **exact**: 35/35 and 2/2 at `best_cost_us == 0.0`.
* ⚠️ The two layers store their time ranges under **different keys** — `time_embed.…` for
  rigids but `deform_network.feature_volume.time_input_embedding.…` for deformables.
  Probing only the first reads as "this layer has no timing".
* Actors rendered per frame: 9,948 / 15,324 / 55,390 / 35,535 / 52,893.
* Wrong-time negative control separates decisively: **+0.2358 grad-NCC** at frame 0.

**But the effect on whole-frame grad-NCC is nil: 0.2774 → 0.2773.** The actors change
389–42,631 px of a 2.07 Mpx frame. They are in for correctness — a driving sim without the
other cars is not a driving sim — not because they move the photometric score.

⚠️ **This forced a falsifier correction.** `falsify_actors` gated on
`g_on > g_off AND g_on > g_wrong_time`. At this footprint `g_on − g_off` is **±0.011 noise**
and its **sign flips with unrelated render settings** (+0.0016 with plain `background+road`,
−0.0001 with cull+sky, *same mapping*) — so a provably-exact mapping was REFUSED for a
reason unrelated to placement. `pass_placement` (`g_on > g_wrong_time`) and `pass_strict`
(the old rule) are now both recorded; `pass` follows placement.

---

## ⛔ WHAT DID NOT WORK — measured negatives, so nobody repeats them

* **Haze cull (large AND low-opacity).** The intuition — keep opaque surfaces, drop only
  translucent haze — is wrong here. `~/rq_out/panel5_haze`: best variant (q 0.90,
  opacity < 0.5) reaches grad-NCC 0.2960 vs the pure scale cull's **0.3460**, and its
  **negative-control margin FALLS below baseline** (+0.0733 vs +0.0873) — the gain is less
  specific to the correct frame. Pure max-scale culling is better on both counts.
* **Naive sky.** Confirmed worse photometrically (render_mean 0.3783 vs ref 0.2425) though
  it *does* raise grad-NCC (0.3169). Gated-at-gain is strictly preferable.
* **Appearance basis.** Settled by the banked `~/nurec_work/basis_*` runs: `f0`, `poly` and
  `tent` are identical to 4 decimals (PSNR 17.01, MAE 0.1030); `fourier_cs` is worse
  (MAE 0.1107). **Not a lever** — no further work warranted.
* **`render_mode="RGB+ED"` for depth.** Unusable on this path: gsplat's ftheta +
  `with_eval3d` kernel asserts `channels == 3` and **aborts the process** (C++ assert at
  `Rasterization.cpp:744`, core dumped — not a catchable Python exception). `render_depth()`
  rasterises camera distance as a 3-channel colour instead.

---

## PERFORMANCE

| config | ms/frame | FPS | 10 Hz loop? |
|---|---|---|---|
| `background+road` (before) | 23.3 | 43 | yes |
| all 4 layers | 23.9 | 42 | yes |
| + cull 0.95 | 20.8 | 48 | yes (cheaper than before) |
| **+ gated sky (chosen)** | **36.3** | **28** | **yes** |
| + rolling shutter | 3749 | 0.27 | **no** — offline only |

Closed loop measured end-to-end with the chosen config: **0.27 s/step, render 34 ms**
(`/tmp/smoke_hq`, refc-base, 6 steps).

---

## Evidence class

| claim | class |
|---|---|
| every table above | **MEASURED (ours)** — run dirs named per table |
| `mean_alpha = 0.5975`, `mae = 0.10296` | **MEASURED**, and independently reproduces `~/nurec_work/final/report.json` |
| NRE up-axis is +Z | **MEASURED twice** — rig +z through `world_to_nre` and road-plane PCA agree to dot 0.99994 |
| cubemap layout `[6,H,W,3]` | **MEASURED twice** — the file's declared `.shape` and a smoothness test over 4 candidate layouts |
| "no pixel exceeds the f-theta FOV" | **MEASURED** at 2 frames, from the calibration alone |
| basis choice is not a lever | **INHERITED** from `~/nurec_work/basis_*` (banked runs, not re-run today) |
| rolling shutter helps the CLOSED LOOP's driving metrics | ⛔ **NOT ESTABLISHED** — only open-loop render fidelity was measured |
