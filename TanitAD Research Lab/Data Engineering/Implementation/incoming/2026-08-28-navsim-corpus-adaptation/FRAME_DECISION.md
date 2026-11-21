# NavSim → TanitAD frame decision (commission deliverable 2 of 4)

**Date** 2026-08-29 · **Owner** DataFlyWheel · **Status** BUILT — see `RESULT.md`

> ⛔ **SELF-CORRECTION (same day, after the build measured it).** The coverage
> argument below is **HORIZONTAL ONLY**, and on that axis it held (measured
> camera share l0 24.1 % · f0 41.0 % · r0 24.2 %). **The vertical axis was never
> checked and does NOT hold:** the 256×640 frame needs **45.3° VFOV**, NavSim's
> cameras carry **38.5°** — a 6.8° deficit no stitch can close, leaving a
> scalloped unobserved band at top and bottom (**10.7 %** of the frame).
> The delivered corpus therefore defaults to **`PHYSICALAI_RIG_CLEAN_176x624`**,
> MEASURED **100.0000 %** observed over 204 scenes × 5 rigs.
> **Root-cause class: a coverage claim stated for "the frame" from an analysis
> of ONE of its two axes** — the same shape as quoting a rule outside its scope.
> The fix that generalises: *a coverage claim names every axis it checked.*

## The decision

**Target = the TRAINING frame: 256×640 CYLINDRICAL (the B1 epcache frame), via
`calib.py`'s `CanonicalFrame` object — never a re-declared literal.**
Not `CANONICAL_256` (256×256 pinhole, f_ref 266): that is the published-numbers
frame for banked comparisons, but an eval corpus must hand a checkpoint the input
distribution it was TRAINED on, or the number measures the train/eval frame gap —
precisely the domain-gap artifact this commission exists to avoid. The four B1
consumers (v7f, refcv3, refav1, refd) train on 256×640 cylindrical; NavSim eval
frames must match. (The exact `f_ref` is taken from the training cache's
`CanonicalFrame` tag at build time — read from the object, not quoted here, per
the v4-unreproducibility lesson.)

## The measured fact that forces a 3-camera stitch

From a warmup scene pickle (tolerant unpickle, nuplan classes stubbed, numpy
passed through — `K` read from the artifact itself, not docs):

```
cam_f0 K = [[1545, 0, 960], [0, 1545, 560], [0, 0, 1]]   (1920×1080 pinhole)
HFOV     = 2·atan(960/1545) = 63.7°
```

Our cylindrical frame spans **120°**. A single-camera reprojection therefore
**under-covers by 56°** — nearly half of every frame would be unobserved. Two
admissible designs, decided in favour of the first:

1. **Stitch `cam_l0 + cam_f0 + cam_r0` → 120° cylindrical** (CHOSEN). The scene
   pickles carry per-camera intrinsics AND extrinsics, so the reprojection is
   fully determined per scene; nuPlan's l0/r0 yaw offsets give comfortable
   overlap at the seams. Per-pixel source-camera map is static per rig — computed
   once, recorded in the corpus so seam provenance is inspectable.
2. Single-camera + declared valid-mask (REJECTED for training-frame parity: a
   56°-black frame is an input distribution none of the four checkpoints has
   ever seen, reintroducing the domain gap inside the frame itself).

⛔ Explicit REPROJECTION only (pinhole ray → cylinder), never a resize; verified
with the harness's own `lateral.assert_axis_convention`
(`taniteval/taniteval/lateral.py:170`).

**Addendum (same day, measured from a synthetic scene pickle):**

1. **The cameras are NOT ideal pinholes — every cam entry carries a 5-param
   `distortion` vector, measured `[-0.356, 0.173, -0.002, 0.000, …]`.** A
   k1 of −0.356 is strong barrel distortion; at the image edges (exactly where
   the stitch seams live) an undistortion-free mapping would misplace pixels by
   tens of px. ⇒ the cylinder→source-pixel map applies the distortion model
   (project ray with K, then distort), so the "static per rig" map is
   K+distortion+extrinsics per camera — still computed once, still recorded.
2. **Coverage confirmed from extrinsics, not assumed:** `cam_l0`/`cam_r0`
   `sensor2lidar_rotation` yields yaw offsets ≈ ±35°; with 63.7° HFOV each the
   l0+f0+r0 fan spans ≈ ±67° — the 120° cylindrical frame (±60°) is covered
   with seam overlap on both sides.
3. Synthetic frames carry `camera_dict` with all 8 cams (`cam_f0…cam_b0`), each
   holding `data_path`, `sensor2lidar_rotation/translation`, `cam_intrinsic`,
   `distortion` — so the build never touches the devkit for geometry.

## Provenance of the measurement

- archive: `navsim_v2.2_warmup_two_stage.tar.gz` (1.16 GB, content-verified by
  the commissioning brief), scene pickle read via a tolerant unpickler
  (`pathlib.PosixPath→WindowsPath` shim; missing `nuplan.*` classes stubbed;
  five intrinsics arrays found in the first scene walked).
- devkit pinned `autonomousvision/navsim@0a380a9`; intrinsics are PER-SCENE data
  (`camera_dict[name]["cam_intrinsic"]`, `navsim/common/dataclasses.py:85`), not
  devkit constants — so the build reads K per scene, never assumes this one.

## Interaction with the other deliverables

- **Grouping key** (deliverable 1, DONE): `log_name`, Outcome A, 100 %/0-leakage
  — `raw/grouping_key_warmup.json`.
- **Ego sidecar** (deliverable 3): unchanged by this decision; frames+calib in
  the inference stream, `EgoStatus` in a schema-distinct sidecar.
- **Adapter changes** (deliverable 4): the eval adapter consumes the SAME
  `CanonicalFrame` tag; no adapter-side geometry.
