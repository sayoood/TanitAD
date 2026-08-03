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

## These are REAL closed loop

The renderer generates each camera frame, the policy returns a trajectory, the controller executes
it, and the next frame is rendered **from where the car actually went**. Not a replay, not an
imagination proxy.

⭐ **Rendered on the edge device.** NVIDIA's NRE renderer is amd64-only, but `volume.nurec` is
gzip + MessagePack and **gsplat renders it natively on aarch64 including the f-theta camera model**
at 16–28 ms per 1920×1080 frame with the scene GPU-resident. Whole closed loop: **0.09–0.21 s/step
= 5–11 Hz on Thor.**

## What the run measured (9 starts × 50 ticks, episode-cluster bootstrap)

**REF-C beats flagship v1 closed-loop, and the separation is ENTIRELY LATERAL.** Paired over the
437 shared windows:

| metric | paired Δ (flagship − REF-C) | |
|---|---|---|
| `dist_to_gt_traj` | +1.171 [0.030, 2.244] | **separated** |
| heading error | +0.084 [0.028, 0.175] | **separated** |
| curvature error | +0.0050 [0.0008, 0.0130] | **separated** |
| yaw-rate error | +0.038 [0.020, 0.057] | **separated** |
| **ADE** | +0.789 [−0.865, +2.728] | **NOT separated** |

⇒ **An ADE-only table would have reported "no difference."** This is the four-family doctrine's
strongest evidence, and it reproduces the 2026-07-23 native-1080 suite on different hardware, a
different renderer and a different scene.

Three defects only the families expose: the arms fail longitudinally in **opposite** directions
(flagship −2.04 m/s, REF-C +1.33 m/s — a pooled score cancels them); the flagship **executes what it
selects only 0.448** of the time vs REF-C's 0.874; and **REF-C's 5-way head never emits a
longitudinal class** (accelerate 0, brake_stop 0) while 42 % of logged windows are `brake_stop`.

## ⚠️ Read these with the caveats

- **with-objects vs empty-road is NULL for both arms** — 19 of 20 paired deltas have CIs containing
  zero; headway is statistically identical. But it only tests **distant** traffic: the agents cover
  0.02–0.4 % of frame at 40–45 m (~2.8 s gap). A **cut-in / close-following** scene is the
  discriminating follow-up, not a claim that vision is ignored.
- ⛔ **Within-sim relative only.** REF-C's open-loop ADE is **1.5157 on these reconstructions vs
  0.4728 on real footage — 3.21× OOD**. Orderings survive; absolute rates do not. Never quote a sim
  rate as a real-world rate.
- **Scope:** this satisfies AlpaSim's renderer **wire contract** with a gsplat backend, driven by a
  TanitAD closed-loop harness. It is **not** `alpasim_runtime.simulate`, so there is **no AlpaSim
  collision/offroad/scene score** here.
- **The strategic family is degenerate on this clip** — `route_head_eq_logged` 1.0000 is a
  constant-predictor tie, and the 20 s scene contains no junction. A junction scene is required
  before any strategic-accuracy claim.
- ⚠️ **The renderer is a step function of pose** (discrete blend-order ties among 3.1 M gaussians):
  a 0.1 px camera rotation moves the 2 s waypoint **6.65 m**, and the gRPC float32 round-trip alone
  costs 4.59 m. **All production numbers must come from one numerical path.**

## Provenance

Code: `stack/experiments/alpasim-gsplat/` · results JSONs in the same directory.
Render validated by **gradient-NCC** against the scene's shipped reference video (0.3664 correct vs
0.2052 best-wrong over the wire). ⛔ PSNR and plain NCC are **inadmissible on this clip** — a *wrong*
reference frame outranks the correct one under both, because every frame is a dark night street.

⚠️ `*.mp4` is gitignored — these are committed with `git add -f`. Any new video needs the same or it
silently never lands.

*The older 10.4 s clips from the (now terminated) eval pod are in
`../alpasim-closedloop-archive-2026-07-22/`.*
