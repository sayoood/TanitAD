# NavSim → TanitAD frame decision (commission deliverable 2 of 4)

**Date** 2026-08-29 · **Owner** DataFlyWheel · **Status** DECIDED, build pending B1

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
