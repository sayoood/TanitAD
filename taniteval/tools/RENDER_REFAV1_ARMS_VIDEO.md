# RENDER_REFAV1_ARMS_VIDEO — four cost geometries, one scene

**Instrument:** `taniteval/tools/render_refav1_arms_video.py` ·
**imports:** `render_refcv3_video.py` (the camera model and draw helpers),
`render_refav1_video.py` (the decode, the SE(2) carry, the cards),
`refav1_arm.py` (the integrator wrapper) ·
**tests:** `stack/tests/test_render_refav1_arms.py` ·
**verifier:** `taniteval/tools/verify_mp4.py` · **written:** 2026-09-06

> ⛔ **READ `taniteval/tools/RENDER_REFAV1_VIDEO.md` FIRST.** This file is its
> sibling and inherits every refusal in it. It draws pixels; it **invents no
> number and runs no model** — every trajectory, head argmax and planner
> provenance field is read out of dumps written by `refav1_arm.py`, the tool
> that produced the scored records.

---

## 1. Why a second reel, when one already exists

`render_refav1_video.py` draws **one** arm against the trivial floor, to make
visible that the shipped planner emits nothing. The 2026-09-05 cost-geometry
panel's finding is different in kind: it is a **trade-off between arms**, and a
trade-off cannot be seen one arm at a time. So this reel puts several arms on
**the same window**, in the same image.

⭐ **The one thing it exists to make visible** (MEASURED over the panel's own
40 windows / 8 episodes; source
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-cost-geometry/raw/arms/`):

| arm | cost geometry | ADE m | max\|kappa\| 1/m | outside the friction circle | straight plans |
|---|---|---|---|---|---|
| `ccos_argmax` | `ccos`, `W_KAPPA` 0 | 1.3272 | **0.2000** (= the clip) | **29.6 %**, peak **3.262 g** | 27.5 % |
| `combined` | + Kamm cap `mu` 0.7 + seed ladder | 1.0504 | 0.1505 | **0.0 %**, peak 0.618 g | 27.5 % |
| `wk15` | `ccos`, `W_KAPPA` 15.11 | 0.8934 | 0.0800 | 0.0 %, peak 0.332 g | **47.5 %** |
| `best` | `W_KAPPA` 15.11 + Kamm cap | **0.8838** | 0.0800 | 0.0 %, peak 0.332 g | 37.5 % |
| **`ha0_ext` FLOOR** | **do nothing** | **0.8772** | — | 18.5 %, peak 1.436 g | — |
| `g` **(CONTROL)** | the human | 0.0000 | 0.1701 | **0.0000** | — |

⇒ **the arm that TURNS drives WORST, and the do-nothing floor beats all four.**
Feasibility is on the **27 windows at `v0 >= 2 m/s`** — the exclusion
`raw/feas_audit.py` already uses on this rig, because below it
`kappa = a_lat / v^2` is ill-conditioned and even the ground truth reads
`max|kappa|` 31.4.

⚠️ **`n = 40` windows / 8 episode clusters is a PICTURE OF A BANKED PANEL, not a
new result.** No interval is computed here and none is claimed. The panel
carries only **2 GT-left and 2 GT-right** windows, so **no turn recall is
quotable from it**; the powered recall numbers live on the stratified
`--window-list` panel in that package's `RESULT.md` §7.5.

---

## 2. The panels

| # | panel | what is drawn | source |
|---|---|---|---|
| **1** | **FRONT CAMERA** (top-left, 1280×512) | GT (**green**), the `ha0_ext` floor (**white, wide, underneath**) and each arm's plan, projected cylindrically; a legend naming every line; the horizon **falsifier** | `CylProjector(v2ep['frame'], per-clip `sensor_extrinsics`)` on `ep*.npz` `g`/`ha0_ext`/`cl` |
| **2** | **METRIC BEV** (top-right, 604 px) | the same paths in metres, on a fixed per-window range that **starts behind the ego** | the same arrays, carried by a rigid SE(2) of the RECORDED poses |
| **3** | **ARMS TABLE** (mid-left) | per arm: its cost geometry, this window's ADE, `max\|kappa\|`, `a`, the plan source, the **peak lateral g** and the friction-circle verdict | ADE from the dump's arrays; the friction column from `tanitad.refs.feasible_decode.assert_feasible` — the **scorer's own** envelope |
| **4** | **WORLD MODEL** (mid-right) | `wm_mse_model` against the **persistence** and **zero** controls over 6.0 s, with the current step marked | `decisions/ep*.npz` — a **T0 diagnostic**, and the panel says so |
| **5** | **DECISIONS** (bottom, full width) | tactical lat/lon and strategic route argmax **by NAME** against the v7.2 labels, the nav token in its own slot, and the planner's decoded goal per arm | `decisions/ep*.npz` `*_pred_nav_true`, `*_label`, `nav_cmd`, `goal_{lat,lon}_cl` |

⛔ **THE CANVAS IS 1920x1122, THE SIBLING REEL'S, AND THAT IS NOT COSMETIC.**
`draw_card` is IMPORTED from `render_refav1_video.py` and paints at ITS size.
MEASURED by `verify_mp4.py` on the first complete render: with 1080-tall frames
and 1122-tall cards, ffmpeg's image demuxer took the **FIRST** frame (the intro
card) and rescaled every clip frame into it -- the whole reel stretched
vertically by **3.9 %**, and every exit code read 0. The tool now refuses a
canvas mismatch **before** the render and asserts the size of **every** frame it
writes. Pinned by two tests.

Every frame is validated through `tanitad.viz_standard.check_frame`, so all five
`STANDARD_ELEMENTS` plus `strategic_input` and `tactical_gt` are declared with
their `kind` and `source`. ⚠️ The world-model strip is deliberately **not**
declared: `world_model` is not in the standard's `KNOWN_ELEMENTS`, and this
renderer does not widen a shared contract to suit itself. It is titled on the
frame and recorded in the frame index instead.

⭐ **The world-model strip is the panel that shows what is WORKING.** MEASURED
over all 40 windows, the predictor **loses** to persistence by **0.0809** at
0.2 s and **wins** by **0.3168** at 6.0 s, winning **28 of its 30 steps**. The
defect this reel shows is in the planner's cost geometry, not in the model's
ability to predict the world — and a reel that showed only the planner would
leave the viewer with the wrong impression of the system.

---

## 3. Colour semantics — one meaning per colour

| colour | meaning |
|---|---|
| **green** `(110,231,138)` | **GROUND TRUTH**, everywhere, always |
| **white** `(236,240,246)` | the **FLOOR** `ha0_ext` — do nothing |
| **magenta** `(244,114,182)` | arm 1 (`ccos_argmax`) |
| **cyan** `(56,189,248)` | arm 2 (`combined`) |
| **orange** `(249,115,22)` | arm 3 (`wk15`) |
| **violet** `(167,139,250)` | arm 4 (`best`) |
| **amber** `(240,190,90)` | a **GIVEN INPUT** — here only the nav token |

⛔ **The orange is `(249,115,22)`, NOT the sibling reel's `(255,158,61)`.** That
one sits an L1 distance of **76** from the given-input amber, and a viewer
scanning one frame cannot be asked to hold a 76-unit distinction. The palette
test requires **> 90** from every reserved colour and **failed on the original**;
that is what the test is for.

⛔ **DRAW ORDER IS LOAD-BEARING, AND IT COST A RENDER TO GET RIGHT.** The floor
goes down **first and WIDE**, so an arm that coincides with it shows the white as
a halo and the viewer reads *"these are the same path"* rather than *"one is
missing"*. But the **ground truth goes ON TOP of the floor**: drawn underneath a
13 px white line it disappeared entirely on every straight window, and a reel
whose reference line is invisible is worse than no reel.

---

## 4. ⛔ The refusals

**(a) The arms are VERIFIED to share everything but the planner.** Before a frame
is drawn, the tool asserts that every rendered dump carries bit-identical `ws`,
`v0`, `g`, `ha0_ext`, `wm_mse_*` and decision-head argmaxes — **16 arrays** — and
**refuses otherwise**. That is what licenses drawing them on one image and saying
the difference is the cost geometry. ⭐ It also asserts the **converse**: at least
one arm must differ on `cl`, or the reel would draw one line under four names.

**(b) The codec is read from the `codec` FIELD, never the buffer's NAME**
(inherited: the buffer is `jpeg_buf` and the codec says `png`).

**(c) The projection is CYLINDRICAL and per-clip MEASURED**, and the frame
carries its own falsifier — the horizon row predicted from that clip's
extrinsics. On the eight clips here the camera height spans **1.213–1.662 m** and
the forward offset **1.780–2.144 m**, so no constant would do.

**(d) The integrator is NOT re-implemented** — every trajectory is read from the
dump, and the control (§5) exercises `refav1_arm.paths_from_controls` itself.

**(e) CONTENT IS ASSERTED ON EVERY FRAME.** A renderer that decoded nothing
writes a full-size, perfectly valid, **BLACK** video and exits 0. So each frame
must carry a plausible non-zero mean luminance and every overlaid path must be
finite and under 500 m, or the tool **refuses to encode**.

**(f) Every arm is OPEN LOOP** (PI ruling 2026-09-02).

---

## 5. ⭐⭐ The ground-truth control, and why it has two halves

Take the RECORDED actions out of each clip's own `*.v2ep.pt`, feed them to the
planner's integrator **as if they were a plan**, and require the result to land
on the dumped ground truth.

⛔ **A control with only the passing half cannot tell "the integrator is right"
from "the tolerance is loose."** `v2ep actions[:, 0]` is a road-wheel **STEER
ANGLE** `arctan(L·kappa)` at L = 2.9 m (RETRACTION_LOG #15). Read as `steer` it
must reproduce `g`; read as `kappa` — the legacy reading the dumps were rolled
under — it must **NOT**. So the criterion requires **both**.

MEASURED over all 40 windows, admissible block `v0 >= 2 m/s`, **n = 27**:

| reading | ADE mean | median | p90 | max |
|---|---|---|---|---|
| **STEER** | **0.1552 m** | 0.1512 | 0.2364 | 0.2981 |
| KAPPA | 0.8727 m | 0.2813 | — | — |

**separation 5.62×** (8.57× over the turning windows alone). The 13 excluded
windows are **counted and printed on the control frame**, never dropped.

⭐ **For scale:** the best planner arm scores ADE **0.8838 m**; the recorded
actions through the same integrator score **0.1552 m** — **5.7× better** — so the
geometry the reel draws is not what limits the arms. The control is written as a
**PNG beside the mp4**, so it is evidence rather than a log line, and a failing
control **aborts the render**.

---

## 6. Slow motion — a MEASURED criterion, said on the frame

A window is held `--slowmo`× longer when its v7.2 lateral label is a turn **OR**
its **ground-truth** lateral extent reaches `--slowmo-lat-m` (default 1.0 m).
⛔ Both halves are properties of the **ground truth alone**, fixed before any arm
is read — never *"the windows where the arms differ most"*, which would be
choosing the evidence by the answer. MEASURED on this panel the two halves select
**4** and **9** windows, **13 of 40** in union, with a clean gap in the GT extent
between **0.83 m** and **2.02 m**. Every slowed frame prints the factor and the
reason, so no viewer can mistake it for the vehicle's real speed.

`--max-plan-age` (default 1.4 s) stops each window once the remaining plan is a
stub under 0.6 s that lands under the camera's own nose; those frames are
**skipped and counted**, never drawn as a shrinking plan.

---

## 7. Usage

```bash
python taniteval/tools/pai_extrinsics_table.py \
    --root <physicalai root> --clips <cid,...> --out extrinsics.json

python taniteval/tools/render_refav1_arms_video.py \
    --arms-dir  <.../raw/arms>  --arms ccos_argmax,combined,wk15,best \
    --episodes  <dir of *.v2ep.pt>  --extrinsics extrinsics.json \
    --cards cards.json --notes notes.json \
    --out reel.mp4 --expect-step 21109 --gt-control \
    --fps 10 --base-hold 2 --slowmo 5 --slowmo-lat-m 1.0 --max-plan-age 1.4

python taniteval/tools/verify_mp4.py reel.mp4 reel_small.mp4
```

⚠️ **Always verify by DECODING BACK.** `ffmpeg` exiting 0 is not evidence, and
neither is the file size: a truncated mp4 keeps a header claiming the full
length. ⚠️ The delivery channel refuses anything at or over **30 MiB**, so every
reel ships as a pair; the small copy is **re-encoded from the original frames**,
never transcoded from the full one.
