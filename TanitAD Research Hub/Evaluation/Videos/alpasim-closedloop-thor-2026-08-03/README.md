# AlpaSim closed-loop videos — rendered on the Jetson Thor, 2026-08-03

**Four videos. Both arms × both traffic conditions.** 18.0 s each, 1800×850, 180 frames @ 10 fps.
Front camera + metric BEV inset + text overlay (decoded manoeuvre, route, ADE).

| file | arm | scene |
|---|---|---|
| `flagship-v1_with_objects.mp4` | flagship v1 | traffic visible |
| `flagship-v1_empty_road.mp4` | flagship v1 | empty road |
| `refc-base_with_objects.mp4` | REF-C base | traffic visible |
| `refc-base_empty_road.mp4` | REF-C base | empty road |

Each was verified by **decoding it back** and md5-matched against the Thor copy.

---

## ⭐ RE-DRAWN 2026-08-03 — labelled CLOSED-LOOP on-frame, with a legend

**The four files were re-drawn, NOT re-driven.** `rerender_closedloop_videos.sh` re-runs
`overlay_video.py --mode closed_loop` over the **already-banked** `~/cl_out_hq/long_*`
rollouts and frames, so every number in the HUD is the same measurement as before. Three
things changed, all in the drawing:

1. **A CLOSED-LOOP badge burned into the camera panel.** The programme now also has
   OPEN-loop videos (`../alpasim-openloop-thor-2026-08-03/`) and the two answer different
   questions. A filename is not a defence once a file has been copied somewhere else.
2. **A legend** naming GROUND TRUTH / MODEL PREDICTION / DRIVEN / annotated agent / LEAD /
   ego, in the dead space under the BEV. Two colours without a key is not "distinguishable".
3. **A thicker ground-truth polyline** (9 px camera / 5 px BEV, drawn first). At equal
   width the green vanished under the orange *exactly where the arm was right* — i.e.
   where the viewer most needs to see that the two coincide.

⛔ **Re-driving would NOT have been equivalent.** The renderer is a step function of pose:
a 0.1 px camera rotation has been measured to move the 2 s waypoint **6.65 m**. Re-using
the banked rollouts keeps these videos on the SAME numerical path as the panel below.

MEASURED after the re-draw: **180 frames / 18.0 s / 1800×850 each, 0 decode errors on all
four**, md5-matched against `thor:~/cl_videos_hq2`. Manifest:
`../alpasim-openloop-thor-2026-08-03/video_verification.json`.

---

## ⭐ RE-RENDERED 2026-08-03 EVENING — render-quality pass

**The four files above are the IMPROVED renders.** The morning versions rendered
`background + road` with no sky; these render **all four layers, cull the over-sized splats, and
composite the scene's own sky env map**. Full evidence, run dirs and the measured negatives:
`stack/experiments/alpasim-gsplat/RENDER_QUALITY.md`.

| | grad-NCC ↑ | neg-control margin ↑ | ms/frame |
|---|---|---|---|
| morning (`background+road`) | 0.2774 | +0.0873 | 23.3 |
| **these files** (4 layers + cull 0.95 + gated sky 0.3) | **0.3424 (+23.4 %)** | **+0.1020** | 36.3 |


> ⛔ **SUPERSEDED 2026-08-03 — the reference video is offset from the rig by a per-scene
> constant** (`+6` on `00040136`, `+5` on `7c72937c`; rule: `video_idx = rig_idx +
> (n_mp4_decodable − n_rig_frames)`, measured by the renderer, unanimous over 12 frames each).
> Re-baselined against the **aligned** reference the improvement is **roughly half the size and
> does not replicate**: `00040136` n=5 **+13.5 %** (was +23.4 %), n=12 **+8.0 %**, and
> `7c72937c` n=12 **+4.4 % — NOT SEPARATED** [−0.0097, +0.0521]. Absolutes move too:
> BEFORE 0.2774 → **0.4228**, AFTER 0.3424 → **0.4800**. The render is still better; the
> magnitude quoted here is not. Corrected table + estimator:
> `TanitAD Research Hub/Evaluation/Implementation/incoming/2026-08-03-render-rebaseline/`;
> `RETRACTION_LOG.md` R-2026-08-03-align. ⚠️ No closed-loop conclusion moves — `cl_metrics.py`
> never opens the reference video.


`run_dir = thor:~/rq_out/panel6_chosen`. Negative control passes 5/5 frames.
Closed loop measured at **0.15–0.24 s/step, render 26–29 ms** — inside the 10 Hz budget.

**What changed, and why:**
1. **All four layers.** `dynamic_rigids` (115,824 gaussians / 30 cuboids) and
   `dynamic_deformables` (1,039 / a `person` and a `rider`) were **absent from every previous
   frame**. Mapping is exact (35/35 and 2/2 at `best_cost_us == 0`); the wrong-time control
   separates by **+0.2358** grad-NCC. ⚠️ Their effect on the whole-frame metric is ~0
   (0.2774 → 0.2773) — they are in for correctness, not for the score.
2. **Scale cull at the 95th percentile** (153,506 splats above 1.4263 m). These were the long
   pink/cyan light **streaks** across the mid-frame — the most conspicuous defect. Removing them
   costs nothing (20.8 ms, *cheaper* than before).
3. **Gated sky at gain 0.3.** Fills the black upper band. That band is a **reconstruction hole,
   not an FOV clip** — zero pixels exceed the f-theta `max_angle` of 77.22°, and the reference
   shows 0.1439 mean brightness where we rendered 0.0176.

**Side-by-side:** `BEFORE_AFTER_f150.png` (render | render | reference | alpha | alpha).
**Artifact diagnosis:** `diagnose_f150.png` (render | reference | alpha | depth | FOV+magenta overlay).

⛔ **Rolling shutter is measured-better but OFF.** The rig declares `ROLLING_TOP_TO_BOTTOM` with a
30.559 ms readout over which the ego moves up to 0.63 m; rendering it lifts grad-NCC to **0.3747
(+35.1 %)** with the best margin of any arm (+0.1285) — at **3749 ms/frame, 161× the cost**. Enable
with `ROLL=1 run_quality_videos.sh` if wall clock is not a constraint (~45 min for these four).

✅ **The metrics below are now measured ON THESE FILES' render** (STREAM C, 2026-08-03 evening).
The morning panel has been superseded — it did not transfer, and it did not transfer by a very
large margin. `run_dir = thor:~/cl_out_hq`, report
`stack/experiments/alpasim-gsplat/results/closedloop-hq-render/STREAM_C_RENDER_AB.md`.

---

## These are REAL closed loop

The renderer generates each camera frame, the policy returns a trajectory, the controller executes
it, and the next frame is rendered **from where the car actually went**. Not a replay, not an
imagination proxy.

⭐ **Rendered on the edge device.** NVIDIA's NRE renderer is amd64-only, but `volume.nurec` is
gzip + MessagePack and **gsplat renders it natively on aarch64 including the f-theta camera model**
at 16–28 ms per 1920×1080 frame with the scene GPU-resident. Whole closed loop: **0.09–0.21 s/step
= 5–11 Hz on Thor.**

## What the run measured — MEASURED ON THIS RENDER (9 starts × 50 ticks)

`thor:~/cl_out_hq/panel_*` · scorer `cl_metrics.py` · paired episode-cluster bootstrap over the
**rollout starts** (disjoint segments of ONE clip, *not* 40 independent episodes) · 437 shared
windows · empty road. Positive Δ = **flagship worse**.

| family | metric | flagship v1 | REF-C base | paired Δ (F−C) | |
|---|---|---|---|---|---|
| **ADE** | ADE 0–2 s | 9.696 [8.229, 11.327] | 2.655 [1.756, 3.525] | **+7.164 [+5.265, +8.966]** | separated |
| **LONGITUDINAL** | target-speed err (abs) | 7.997 [6.801, 9.309] | 1.705 [1.055, 2.368] | **+6.397 [+5.000, +7.801]** | separated |
| **LONGITUDINAL** | target-speed err (signed) | **−7.997** | +1.211 | — | opposite signs |
| **LONGITUDINAL** | along-track ADE | 9.639 [8.162, 11.273] | 2.610 [1.698, 3.491] | **+7.153 [+5.240, +8.953]** | separated |
| **LONGITUDINAL** | headway to lead / time gap | 37.57 m / 3.99 s | 46.99 m / 2.93 s | — | not separated |
| **LATERAL** | heading err | 0.2037 [0.1649, 0.2450] | 0.0357 [0.0246, 0.0501] | **+0.1700 [+0.1303, +0.2119]** | separated |
| **LATERAL** | curvature err | 0.0179 [0.0136, 0.0230] | 0.0016 [0.0010, 0.0022] | **+0.0166 [+0.0125, +0.0213]** | separated |
| **LATERAL** | yaw-rate err | 0.1179 [0.0918, 0.1485] | 0.0132 [0.0103, 0.0164] | **+0.1066 [+0.0774, +0.1393]** | separated |
| **LATERAL** | cross-track abs | 5.162 [2.887, 7.861] | 0.630 [0.296, 1.021] | **+4.503 [+2.019, +7.372]** | separated |
| **LATERAL** | lateral ADE | 0.784 [0.626, 0.963] | 0.310 [0.234, 0.380] | **+0.475 [+0.320, +0.657]** | separated |
| **TACTICAL** | plan == logged manoeuvre | 0.240 [0.147, 0.333] | 0.300 [0.185, 0.417] | −0.062 [−0.217, +0.080] | **NOT separated** |
| **TACTICAL** | executed == planned | 0.552 [0.459, 0.641] | 0.878 [0.730, 0.985] | — | flagship still does not execute what it selects |
| **STRATEGIC** | route-corridor departure | 0.611 [0.540, 0.682] | 0.094 [0.000, 0.191] | **+0.506 [+0.382, +0.629]** | separated |
| **STRATEGIC** | route head == logged | 1.000 ⛔ | 0.180 ⛔ | ⛔ **not admissible** — see below | |

⚠️ `dist_to_gt_traj_m` is **not a separate line here**: it is literally `abs(cross_track)`
(`cl_metrics.py`, one dict literal). The morning table listed it under ADE *and* listed cross-track
under LATERAL, which showed **one** measurement as **two** separations.

⛔ **The STRATEGIC route-head row is a degenerate tie, not a flagship win.** flagship emits
`straight` on 100 % of windows and every *valid* logged label on this clip is `straight`, so a
constant predictor scores 1.0; REF-C's head actually spreads over three classes
(left 0.229 / straight 0.387 / right 0.384) and is punished for it. The estimator itself marks both
intervals `degenerate`. **A junction scene is still required before any strategic-accuracy claim.**

### The morning verdict: HALF SURVIVES, HALF IS REFUTED

| morning claim | on this render |
|---|---|
| REF-C beats flagship v1 closed-loop | ✅ **SURVIVES**, and the gap grows (cross-track Δ 1.171 → 4.503, ×3.85) |
| all four lateral separations hold | ✅ **SURVIVES** — every one, each wider |
| the separation is **ENTIRELY LATERAL** | ❌ **REFUTED** — ADE, both longitudinal metrics and strategic corridor departure all separate now |
| "an ADE-only table would have reported no difference" | ❌ **no longer true of this panel** — ADE separates at +7.164 [+5.265, +8.966] |

The four-family doctrine is *not* weakened by this: on the morning render ADE was blind and only the
families could see the gap, which is exactly the argument for reporting all five. What changed is
that **this particular panel is no longer the doctrine's example**, and it must stop being quoted as
one.

## ⭐ THE MECHANISM — and it is the biggest finding of the day

The gap did not widen because REF-C improved. **flagship v1 collapses when the render changes, and
REF-C barely notices.** Paired, same arm, same windows, HQ − MORNING:

| | flagship v1 | REF-C base |
|---|---|---|
| driven-path shift (mean / p50 / max) | **9.05 / 5.95 / 37.78 m** | 0.43 / 0.19 / 3.25 m |
| emitted-plan shift (mean / max) | **7.76 / 17.91 m** | 0.38 / 1.66 m |
| plan shift at `k=0` (zero accumulated drift) | 0.73 – 9.09 m | 0.02 – 0.72 m |
| ADE 0–2 s | **+6.052 [+4.129, +7.742]** separated | −0.079 [−0.286, +0.081] |
| target-speed err (abs) | **+5.038 [+3.637, +6.268]** separated | −0.055 [−0.203, +0.071] |
| plan == logged manoeuvre | **−0.138 [−0.227, −0.040]** separated | −0.018 [−0.089, +0.037] |
| mean commanded speed | **12.96 → 7.05 m/s** (log ≈ 15.0) | 16.08 → 15.95 m/s |

**flagship v1 brakes.** Given a better-looking frame it commands a mean 7.05 m/s where the log drives
15.0, and 8 m/s of that error is longitudinal. At `k=0` — the very first tick, before any divergence
can accumulate — its plan already moves up to 9.09 m from a render change alone. REF-C's moves 0.02 m.

⇒ **flagship v1's closed-loop behaviour on a neural reconstruction is not a stable quantity.** Its
morning closed-loop numbers were never quotable as *the* arm's numbers, and neither are these; what
*is* quotable is the **21× render-sensitivity ratio** between the arms.

### And it is the SCALE CULL, not the sky — a 2×2 says so

`empty` road has exactly two render changes, so they separate. Each cell paired against the morning
rollouts on identical windows (`results/closedloop-hq-render/ABLATE_*.json`):

| flagship v1, paired Δ vs morning | scale-cull 0.95 only | gated sky 0.3 only | both (= the videos) |
|---|---|---|---|
| ADE 0–2 s | **+4.489 [+3.146, +5.999]** sep | −0.457 [−0.999, +0.118] | **+6.052 [+4.129, +7.742]** sep |
| target-speed err (abs) | **+3.734 [+2.594, +4.949]** sep | −0.465 [−0.917, +0.055] | **+5.038 [+3.637, +6.268]** sep |
| along-track ADE | **+4.641 [+3.295, +6.180]** sep | −0.451 [−1.027, +0.170] | **+6.194 [+4.337, +7.825]** sep |

**Dropping the 153,506 over-sized splats — the change that removed the visible light streaks and
did most for grad-NCC — is what breaks flagship's speed control.** The gated sky is null or very
slightly *helpful* for it. The two are super-additive (4.489 − 0.457 = 4.03 < 6.05), so there is an
interaction too, but the dominant single factor is unambiguous. On REF-C every cell is ≤ 0.031.

⚠️ This is **not** an argument against the cull. The culled render is the one closer to the shipped
reference video. It is an argument that **flagship v1 has a brittle appearance dependence** that
nothing in the open-loop suite can see.

## ⛔ THE NEGATIVE CONTROL THAT LICENSES ALL OF THAT

The morning **config** was re-run today (`thor:~/cl_out_repro`) and paired against the morning
**rollouts**. It reproduced them **EXACTLY**: `0.0` m driven-path, `0.0` m plan, `0.0` on **all 19
paired metrics, both arms, 450/450 windows**. The loop is bit-deterministic, `closedloop_drive.py`'s
two changes today do not touch the `empty` path, and therefore **100 % of the deltas above are the
render**. Without this control none of them would have been attributable — the renderer is a step
function of pose, and a 0.1 px camera rotation has been measured to move the 2 s waypoint 6.65 m.

Every change was also tested as a **difference-in-differences** — `(flagship−REF-C)|HQ` minus
`(flagship−REF-C)|MORNING`, bootstrapped on the same windows — because comparing two CIs by eye is
not a test. 9 of 10 metrics separate; only `heading_err` does not, making heading the most
render-stable of the separations.

## ⚠️ Read these with the caveats

- **with-objects vs empty-road is NO LONGER NULL for flagship** on this render: **12 of 23** paired
  deltas separate, all in the direction *actors make flagship better* (ADE −2.340 [−3.786, −0.747],
  along-track −2.386, speed err −1.918, corridor departure −0.242). **REF-C stays null** (1 of 23,
  and that one is −0.024 m/s). ⚠️ **Do not read this as agent-aware reasoning yet.** flagship is the
  arm whose plan moves 9 m under *any* appearance change, so "116 k extra gaussians change the
  frame" and "the model reasons about the agents" predict the same sign. The discriminating control is a
  **wrong-time actor** rollout (draw the same actors at a shifted `t0_us`); it is pre-registered in
  `STREAM_C_RENDER_AB.md` and NOT yet run.
- ⛔ **NO `objects` morning-vs-HQ NUMBER IS ADMISSIBLE.** Its determinism control was run and it
  **fails**: re-running the morning `objects` config today moves flagship's driven path a mean
  **1.536 m (max 7.266)** and REF-C's **0.165 m (max 1.299)** with the render held fixed, because
  today's `closedloop_drive.py` sets `act["tracks"].t0_us` per rollout — a line that only executes
  with actors attached, which is why the exactly-zero `empty` control does not cover it. 18 of 19
  paired deltas move, 4 (flagship) / 7 (REF-C) separate, **from the code change alone**. The
  `empty` headline above is unaffected, and the HQ-internal objects-vs-empty contrast is still
  valid — but the morning and HQ `objects` panels are not comparable.
  Artifacts: `results/closedloop-hq-render/CONTROL_<arm>_objmorn_vs_morning.json`.
- ⛔ **Within-sim relative only.** REF-C's open-loop ADE is **1.5157 on these reconstructions vs
  0.4728 on real footage — 3.21× OOD**. Orderings survive; absolute rates do not. Never quote a sim
  rate as a real-world rate.
- **Scope:** this satisfies AlpaSim's renderer **wire contract** with a gsplat backend, driven by a
  TanitAD closed-loop harness. It is **not** `alpasim_runtime.simulate`, so there is **no AlpaSim
  collision/offroad/scene score** here.
- **The strategic ROUTE-HEAD metric is degenerate on this clip** — flagship's 1.0000 is a
  constant-predictor tie, the 20 s scene contains no junction, and every *valid* logged route label
  is `straight`. A junction scene is required before any strategic-accuracy claim.
  ✅ **What IS usable in this family is `route_corridor_departure_rate`** — flagship leaves the 2 m
  corridor on **0.611** of windows vs REF-C's **0.094**, separated at +0.506 [+0.382, +0.629].
  ⚠️ **The morning panel published REF-C as exposing no strategic route logits. That was FALSE** —
  `cl_metrics.py` read only `s_route_logits` and REF-C emits `route_logits`, so a key-name mismatch
  deleted a whole family for one arm. It is fixed, and every "morning" number quoted here is the
  morning **rollouts re-scored with the fixed scorer**, so the render comparison is not confounded
  with the scorer fix. MEASURED: all **10 of 10** paired deltas the morning file published are
  identical before and after the fix, and exactly **three** fields change —
  `arm_A/STRATEGIC/route_head_eq_logged`, `arm_B/STRATEGIC/route_head_eq_logged`,
  `arm_B/STRATEGIC/route_head`.
- ⚠️ **The renderer is a step function of pose** (discrete blend-order ties among 3.1 M gaussians):
  a 0.1 px camera rotation moves the 2 s waypoint **6.65 m**, and the gRPC float32 round-trip alone
  costs 4.59 m. **All production numbers must come from one numerical path.**

## Provenance

Code: `stack/experiments/alpasim-gsplat/` · render-quality run reports under
`results/render-quality/`.

**The driving numbers on this page** come from `results/closedloop-hq-render/` —
`HQ_flagship_vs_refc_{empty,objects}.json` (the four families), `RENDER_<arm>_hq_vs_morning.json`
(the render effect per arm), `CONTROL_<arm>_repro_vs_morning.json` (the determinism control),
`MORNRESC_*.json` (the morning rollouts re-scored today) and `RENDER_AB_empty.json` (divergence,
difference-in-differences, self-consistency). Narrative: `STREAM_C_RENDER_AB.md`, **generated from
those JSONs** so no number on this page was hand-copied. Rollout dirs on the device:
`thor:~/cl_out_hq`, `thor:~/cl_out` (morning), `thor:~/cl_out_repro` (control).
⚠️ The panel JSONs directly under `results/` (`metrics_empty.json`, `metrics_objects.json`,
`metrics_*_obj_vs_empty.json`) are the **MORNING** render and are **superseded** for anything about
these video files.
Render validated by **gradient-NCC** against the scene's shipped reference video. ⛔ PSNR and plain
NCC are **inadmissible on this clip** — a *wrong* reference frame outranks the correct one under
both, because every frame is a dark night street. ⚠️ **MEASURED 2026-08-03: this extends to MAE.**
Over 5 frames × 6 wrong references, grad-NCC identifies the correct frame **5/5 on every arm** while
MAE and PSNR manage only **1–4/5, and their reliability changes with the arm** (at sky-gain 0.5 MAE
is right 1 time in 5). A photometric difference between two arms is therefore not evidence about
quality; MAE appears in the tables as description and decides nothing.

⚠️ `*.mp4` is gitignored — these are committed with `git add -f`. Any new video needs the same or it
silently never lands.

*The older 10.4 s clips from the (now terminated) eval pod are in
`../alpasim-closedloop-archive-2026-07-22/`.*
