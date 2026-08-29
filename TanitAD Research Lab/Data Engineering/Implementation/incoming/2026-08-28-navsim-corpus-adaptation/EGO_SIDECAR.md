# NavSim ego sidecar design (commission deliverable 3 of 4)

**Date** 2026-08-29 · **Owner** DataFlyWheel · **Status** DESIGNED, build with the corpus

## The mandate it implements

Two binding rules meet here: **labels/scoring may use ego; inference is
VISION-ONLY** (PI, 2026-08-03), and the commissioning brief's requirement that
ego state live in a **schema-distinct sidecar** so the inference stream is
frames + calib and nothing else. The sidecar therefore exists for the
**evaluator and label side only** — the adapter that feeds checkpoints never
opens it.

## Measured source schema (probed from the artifacts, 2026-08-29)

Two layers carry ego, both probed via the tolerant unpickler (nuplan classes
stubbed, numpy through):

**1. Synthetic eval scenes** (`synthetic_scene_pickles/*.pkl`, n=204) — per
frame dict `ego_status`, 4 history frames at 2 Hz (measured Δt ≈ 499.7–500.3 ms,
`num_future_frames = 0` — eval scenes end at the decision point):

| field | shape/type | measured example | frame |
|---|---|---|---|
| `ego_pose` | (3,) f64 | [5.877e5, 4.476e6, 0.9045] | **global** [x, y, ψ] (`in_global_frame=True`) |
| `ego_velocity` | (2,) f64 | [6.864, −0.147] | local [vx, vy] |
| `ego_acceleration` | (2,) f64 | [0.485, 0.229] | local [ax, ay] |
| `driving_command` | (4,) i64 one-hot | [0, 1, 0, 0] | route-level command |
| `in_global_frame` | bool | True | pose-frame declaration |

**2. Original-log metadata** (`openscene_meta_datas/*.pkl`, n=102, 5 frames
each): `ego2global_translation` (3,), `ego2global_rotation` (quaternion, 4,),
`ego_dynamic_state` (list[4] ≈ [vx, vy, ax, ay]), `driving_command` (4,),
`timestamp` (µs), plus `can_bus` (18,).

⚠️ `can_bus` is **deliberately NOT propagated** into the sidecar: its 18-slot
layout is undocumented in the artifact, and every field the evaluator needs
already exists under a named key. Copying an unverified layout forward is how
the `step_s`/`df` class of scope errors gets born.

## Sidecar schema

One parquet, one row per (scene_token, frame_idx) — columnar, joinable, and
structurally incapable of being confused with a frame tensor:

```
navsim_ego_sidecar.parquet
  scene_token       str   (synthetic scene id, joins attributes csv + pickles)
  log_name          str   (the GROUPING key — deliverable 1, Outcome A)
  frame_idx         int   (0..3 history; frame 3 = decision point)
  t_us              int64 (epoch µs, measured 2 Hz)
  ego_x, ego_y      f64   (global metres, as-measured; no re-projection)
  ego_heading       f64   (ψ, global frame)
  vx, vy            f64   (local)
  ax, ay            f64   (local)
  driving_command   int8  (argmax of the one-hot; raw 4-vector NOT stored —
                           a one-hot column invites accidental concatenation
                           into a feature block)
  source_pkl        str   (provenance: which pickle, which layer)
```

Derived quantities (speed, yaw-rate, curvature) are computed at USE time by the
evaluator from these raw fields — the sidecar stores measurements, not
derivations, so no consumer inherits a silent convention.

## Leak analysis (the part that makes it a design and not a dump)

- **`driving_command` is a route-level signal** — the NavSim analogue of
  `nav_command`. Under MANIFEST trainer condition 5 it is **oracle-input-only**:
  admissible as a *declared* goal input to an arm that trains with one, never
  as a hidden extra channel, and never derivable from the situation
  classifier's output (goal/situation disjointness, PI 2026-08-03). If an arm
  consumes it, the adapter passes it EXPLICITLY and the arm's registry row says
  so.
- **`ego_velocity`/`ego_acceleration` at inference = vision-only violation.**
  They exist here for the evaluator (PDM-style scoring rolls the GT kinematics)
  and for any label derivation — both offline uses, both sanctioned.
- **The inference stream stays: stitched cylindrical frames + `CanonicalFrame`
  tag.** Nothing in this sidecar is reachable from the adapter's model-input
  path; the two live in different files with different schemas, which is the
  point.

## Interaction with the other deliverables

- Grouping: `log_name` is IN the sidecar so split tooling needs no second join.
- Frame decision: unchanged; see the same-day distortion addendum in
  `FRAME_DECISION.md` (measured k1 = −0.356 — the stitch undistorts before the
  cylinder map).
- Adapter changes (deliverable 4): consumes frames+calib only; receives
  `driving_command` solely as an explicit oracle-goal argument.
