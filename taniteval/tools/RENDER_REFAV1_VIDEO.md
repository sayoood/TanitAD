# RENDER_REFAV1_VIDEO — the refav1 planner reel

**Instrument:** `taniteval/tools/render_refav1_video.py` ·
**imports:** `taniteval/tools/render_refcv3_video.py` (the camera model and the
draw helpers) · **helper:** `taniteval/tools/pai_extrinsics_table.py` ·
**verifier:** `taniteval/tools/verify_mp4.py` · **written:** 2026-09-04

> ⛔ **READ `taniteval/tools/REFAV1_ARM.md` §3a AND
> `taniteval/results/RESULT-refav1-21109-openloop.md` §0 FIRST.** They are the
> definition of what refav1's `cl` arm *is*. This file draws pixels; it invents
> no number, and **it runs no model** — every trajectory, every head argmax and
> every planner provenance field is read out of a dump written by
> `refav1_arm.py`, the tool that produced the scored eval record.

---

## 1. Why this reel exists, and why it is not the refcv3 reel

`render_refcv3_video.py` draws a **selection surface**: 128 anchors, a
reachability mask, a rank ramp. refav1 has none of those — it is an **iCEM
action-space planner**, and the fact the reel has to make visible is the
opposite of a rich selection: MEASURED at step 21,109 over the full 141-clip
v7.2 EVAL split, `n = 282` windows / 141 episode clusters,

| quantity | value |
|---|---|
| `cl` bit-identical to the trivial floor `ha0` (< 1e-9 m) | **270 / 282 = 0.9574** |
| planner curvature identically zero | **282 / 282 = 1.0000** |
| acceleration constant within the 2.0 s plan | **282 / 282 = 1.0000** |
| distinct plans emitted over the whole split | **2** — `a = 0` on 270, `a = −1.5` on 12 |

⭐ **So `ha0` is drawn on every frame, beside the model, and it is drawn WIDE and
UNDERNEATH.** Where the two coincide the white floor shows as a halo around the
orange plan, and the viewer reads *"these are the same path"* rather than *"one
of them is missing"*. That distinction is the entire finding, and a caption
cannot carry it.

---

## 1a. ⭐ THE DELIVERED REELS (2026-09-04)

Two reels, from the same checkpoint and the same planner config, differing ONLY
in the replan stride. Both were verified by **decoding them back** — container
metadata *and* a real full-packet decode — never by ffmpeg's exit code
(`taniteval/tools/verify_mp4.py`), and each ships as a pair because the delivery
channel refuses anything at or over **30 MiB**.

| reel | clips | replan | frames / duration | full | small (1280 px) | decode |
|---|---|---|---|---|---|---|
| **`refav1_reel_step21109_dense4`** | 4 | every **0.8 s** (stride 4) | 852 / **85.20 s** | 22.42 MiB | 3.20 MiB | 852 frames, clean, both |
| `refav1_reel_step21109_eval8` | 8 | every 8 s (stride 40 — the eval's own grid) | 444 / 44.40 s | 8.18 MiB | 1.53 MiB | 444 frames, clean, both |

**Why two.** The 141-clip EVAL dump is `--window-stride 40`: one plan every 8 s,
so a reel built on it shows ~2.2 s of each clip before the plan has nothing left
to draw (the renderer SKIPS and COUNTS those frames — 480 of them). It covers all
eight clips and is the companion. The dense reel re-rolls four of the same clips
at stride 4, where **no frame is ever skipped**, and is the one to watch.

⭐ **The dense reel is also an independent check of the headline at a different
stride and on a deliberately harder clip set.** Its four clips carry GT lateral
> 0.5 m on **48/84 = 57 %** of windows (30 of them > 2.0 m, max 7.26 m) — far more
lateral demand than the full split — and the planner still emits **κ identically
zero on 84/84 = 1.0000** of windows, constant acceleration on 84/84, and the same
**two** distinct accelerations (`a = 0` on 74, `a = −1.5` on 10). `cl` is
bit-identical to `ha0` on **74/84 = 0.8810**. Denser sampling on harder roads
does not find a single steering command.

---

## 2. The panels

| # | panel | what is drawn | source |
|---|---|---|---|
| **1** | **FRONT CAMERA** (top-left, 1280×512) | GT future (**green**), `ha0` (**white, wide, underneath**), `cl` (**orange, on top**), `ha` (**slate, thin**), projected into the image | `CylProjector(v2ep['frame'], per-clip sensor_extrinsics)` on `ep*.npz` `g/cl/ha0/ha` |
| **2** | **METRIC BEV** (right, 604 px) | the same four paths in metric ego coordinates, span adapted to the data, with the per-window verdict at the top | the same arrays; calibration-independent, so it carries the comparison even where the camera overlay is off |
| **3** | **TACTICAL** (left, below the camera) | the **factored** `lat` and `lon` head argmax under **three conditionings** (`nav_true` / `nav_shuffled` / `nav_zero`) against the v7.2 labels, plus the tactical decoder's **imagined goal** | `decisions/ep*.npz` |
| **4** | **STRATEGIC** (left, bottom) | the route head under the same three conditionings against the route label, and — in its **own slot, in amber** — the nav token that was **FED** | `decisions/ep*.npz` `route_pred_*`, `nav_cmd` |
| **5** | **HUD** (full width) | the emitted `(a, κ)`, the plan source, its cost and candidate count, the **plan age**, the per-window ADE of `cl` and `ha0`, and the bit-identity verdict | — |

Every frame is validated through **`tanitad.viz_standard.check_frame`** before it
is drawn, so a silently missing panel is structurally impossible. All five
`STANDARD_ELEMENTS` plus both `AUX_ELEMENTS` are declared on every frame, with
`kind` and `source`; an element that cannot be computed renders its **reason**.

**Cards.** `--cards <json>` prepends an intro and appends an outro full-screen
card. They are **text only** on purpose: a card states what was measured and on
what `n`, and never draws a figure a viewer could mistake for a result computed
on the frames around it.

---

## 3. Colour semantics — one meaning per colour, in every panel

| colour | meaning |
|---|---|
| **green** `(110, 231, 138)` | **GROUND TRUTH** — everywhere, always |
| **orange** `(255, 158, 61)` | **refav1's own plan** (`cl`) |
| **white** `(236, 240, 246)` | the **TRIVIAL FLOOR** `ha0`: `a = 0`, `κ = 0` at the measured `v0` |
| **slate** `(120, 140, 168)` | the hold-action control `ha` |
| **amber** `(240, 190, 90)` | a **GIVEN INPUT** — here only the nav token |
| **red** / **green** text | a head-vs-label **MISS** / **MATCH** |

---

## 4. ⛔ The five things this renderer refuses to get wrong

**(a) Frames are bridged LOCALLY (C79).** The overlay is drawn on the frame
decoded from the clip's own `*.v2ep.pt` at raw index `2t` — the last frame the
window at cache index `t` observes, and the same bytes the fp8 encoder consumed.
The `refav1_loader` reads the v2ep dict's **RAW** poses with **no `n_stack`
offset** (`refav1_loader.py` module docstring), so `2t` is the correct index and
copying the provider-view shift would be the `step_s` trap. The upscale is
**NEAREST**: no pixel is invented.

**(b) The codec is read from the `codec` FIELD, never from the buffer's NAME.**
The payload's buffer is called `jpeg_buf` and its `codec` says `png`. This tool
reads the field, **and asserts the magic bytes agree** (`0x89 0x50`), refusing
if they do not. Trusting the name is the E-DETECT-1 trap, where `decode_jpeg`
raised into a pre-allocated memmap and left a full-size array of zeros while the
job exited 0.

**(c) The projection is the clip's own, and it is CYLINDRICAL.** Read from each
`*.v2ep.pt`'s own `frame` dict (256×640, `f_ref` 305.5775), never assumed;
`CylProjector` refuses a projection it cannot express rather than approximating
it. Extrinsics are **per-clip and MEASURED** — on the eight clips of the
delivered reel the camera height spans **1.274–1.576 m** and the forward offset
**1.791–2.129 m**, so no constant would do. Without an entry the camera overlay
is **DISABLED** and says so on the frame; the BEV carries the comparison either
way. ⭐ **The frame carries its own falsifier:** the horizon row predicted from
the clip's extrinsics is drawn as a thin line — if it does not sit where the road
meets the sky, the projection is wrong and the viewer can see that without
trusting a caption.

**(d) Between replans the plan is HELD and TRANSFORMED, never re-invented.**
refav1 plans once per scored window. On the frames in between, the last plan is
carried into the current ego frame by a **rigid SE(2) transform of the RECORDED
poses** — exactly open-loop execution of an MPC plan — and the frame prints the
**age** of the plan it is drawing. Past the plan's own 2.0 s horizon there is no
action left, so the frame is **skipped and counted**, never drawn as a shrinking
stub that would read as a model which stopped planning.

**(e) Every arm is OPEN LOOP** (PI ruling 2026-09-02). The model controls
nothing; the ego data keeps arriving from the recording. `T0`/`T1` are the
doctrine's *conditioning* labels, not loop labels. The phrase "closed loop"
describes nothing in this reel.

---

## 5. ⚠️ Two unit facts that travel with every frame

* **The corpus's `v2ep actions[:, 0]` is a road-wheel STEER ANGLE**,
  `arctan(L·κ)` at L = 2.9 m (RETRACTION_LOG #15), and the dump behind the
  delivered reel was rolled under `--action-units kappa` — the legacy reading.
  So the `ha` and `ol` arms **over-rotate by ~2.9×**. `cl` and `ha0` are
  untouched: both are exactly zero in either unit, and they are the two arms this
  reel compares.
* ⛔ **The PLANNER's `controls` tensor uses the OPPOSITE order, and confusing the
  two is a scope error.** `refa_v1.py:2176-2177` computes
  `jerk = (controls[:, 1:, 0] − controls[:, :-1, 0]) / dt` and
  `W_KAPPA * controls[..., 1]²`, and `v_end = v0 + controls[..., 0].sum(-1)·dt`
  — so **channel 0 is ACCELERATION and channel 1 is curvature**, and `W_JERK`
  penalises **longitudinal** jerk, not lateral. (`refav1_loader._kin_actions`
  returns `stack([a, kap])` with its own comment *"(a, κ) — swapped"*.) The HUD
  prints `a` and `κ` under those names for exactly this reason.

---

## 6. Choosing the clips — MEASURED, and stated

⛔ **A hand-picked reel is not a result and must never be quoted as one.** The
eight clips of the delivered reel were ranked out of the same 141-clip EVAL
split by **measured** GT lateral extent and GT speed change inside the scored
windows, then chosen for spread:

| # | clip | manoeuvre (MEASURED from the dump's own GT) |
|---|---|---|
| 1 | `6ed4ef7a` | sustained high-speed bend — GT lateral 6.30 m at 15.6 m/s |
| 2 | `ed87040c` | **left turn** while accelerating — lateral 4.88 m, Δv +2.02 m/s |
| 3 | `d85682b8` | **right turn** at low speed — lateral 4.53 m at 5.7 m/s |
| 4 | `c8a39711` | **hard braking**, straight — Δv **−5.48 m/s** at 14.5 m/s |
| 5 | `f6e7827e` | **acceleration** on a bend — Δv +3.94 m/s, lateral 2.17 m |
| 6 | `24fee8a5` | **left turn into heavy braking** — Δv −3.88 m/s |
| 7 | `142a3a72` | motorway, the highest speed in the reel — `v0` 23.0 m/s |
| 8 | `1c3a2c7c` | **acceleration** from low speed — Δv +3.24 m/s |

Both lateral directions, both longitudinal signs, and a speed range of
5.7–23.8 m/s. ⭐ `d85682b8` and `6ed4ef7a` are **also in the refcv3 reel**, so
the two models can be watched on the same roads.

---

## 7. Invocation

```bash
# 1. the extrinsics (once per clip set)
python taniteval/tools/pai_extrinsics_table.py \
    --root <physicalai root> --clips <cid,cid,…> --out extrinsics8.json

# 2. the dump — the SAME tool and the SAME planner config as the eval
python taniteval/tools/refav1_arm.py \
    --ckpt   .../refav1-b1-v72-ep3-speed/ckpt.pt \
    --config .../config.json \
    --cache  <a dir of symlinks to just the chosen clips' fp8 .pt> \
    --episodes <the v2ep dir> --labels <v7.2 blob> --nav <v7.2 blob> \
    --no-lead-block --device cuda --window-stride 4 --wm-k 15 --no-navshuf \
    --dump-dir render_dump --out render.json --arm refav1-21109-render8

# 3. the reel
python taniteval/tools/render_refav1_video.py \
    --dump-dir render_dump --episodes <the v2ep dir> \
    --extrinsics extrinsics8.json --cards cards.json \
    --out refav1_reel_step21109.mp4 --fps 10 --expect-step 21109

# 4. ⛔ VERIFY BY DECODING BOTH FILES BACK — ffmpeg's exit code is not evidence
python taniteval/tools/verify_mp4.py refav1_reel_step21109.mp4 \
                                     refav1_reel_step21109_small.mp4
```

⚠️ **`--wm-k` must be ≥ 15.** `refa_v1.forward` refuses fewer operative steps
than the strategic level's first target (`refa_v1.py:1522`); `--wm-k 10` dies
with *"the level would train on an empty tensor"*. 15 is the minimum and it buys
the widest window grid.

⚠️ **`--expect-step` is a refusal, not a caption.** The tool exits rather than
label a reel with a step the dump is not.

⭐ **The reel always ships as a PAIR** — the full-quality render and a copy under
the **30 MiB** delivery ceiling MEASURED by a refusal on 2026-09-03. The small
copy is **re-encoded from the original PNG frames**, never transcoded from the
finished mp4, which would stack a second generation of loss and buy nothing.

---

## 8. Cost

The dump is 100 % of the cost: `plan()` is **~38 s/window** on the Jetson Thor at
the shipped `{samples 300, iters 30, elites 30}`. The renderer itself is
**~0.09 s/frame** on the dev-box CPU — 1,300 frames in about two minutes, no GPU.
⇒ **the replan stride, not the frame rate, sets the wall-clock.** At
`--window-stride 4` a clip is replanned every 0.8 s, which is both a realistic
MPC cadence and short enough that no frame is ever skipped for a stale plan.

`--max-frames N` renders a smoke reel end-to-end (frames + both encodes) in
seconds; use it before committing to a long run.
