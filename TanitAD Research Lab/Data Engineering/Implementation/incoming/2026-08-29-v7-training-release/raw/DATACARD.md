# tanitad-v7-training-corpus

**4,719 clips / 26.2 h** of front-camera driving with
hierarchical v7 labels (strategic goal+action · tactical multi-label goals ·
lateral/longitudinal actions · nav-command input) plus the FULL Alpamayo2-Super
augmentation (meta_action, CoT, auto-labeling, VQA, grounding boxes) and
provider egomotion. Corpus id `a48251e89c7a8603`. PRIVATE — research use within the
TanitAD programme; sources: nvidia/PhysicalAI-Autonomous-Vehicles (research
license) + Sayood/tanitad-alpamayo2-augmentation.

Released 2026-08-29 on PI authorization as the training baseline for
**v7f, refcv3, refav1, refd**. Ground-truth register row: `D-LABEL-GT`.

## Read this before training
Every consumer MUST apply the five `trainer_conditions_mandatory` in
`MANIFEST.json`. The labels carry their own epistemics per token —
`provenance`, `disputed`, `grounded`, `corroboration`, `time_basis`,
`t_nominal_s`, and per-clip `turn_suppression` — so a trainer chooses what to
trust explicitly rather than inheriting silent defaults.

## Camera — retrieval contract (drive it unattended)
Not shipped (~47 GB). Deterministic retrieval, per clip:

1. `chunk = index/clip_to_chunk.parquet.loc[clip_id].chunk` (also carries the
   train/val/test split).
2. `tools/pull_camera.py` opens
   `camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_{chunk:04d}.zip`
   on `nvidia/PhysicalAI-Autonomous-Vehicles` via **HTTP range reads** (the
   `HTTPRangeFile` class in `tools/pull_egomotion_range.py`): it fetches the
   zip's central directory from the final 256 KB, then ONLY the member
   `{clip_id}.camera_front_wide_120fov.mp4` (~10 MB vs the 2.05 GB chunk).
3. Every pulled file is verified by CONTENT (`ftyp` box in the first 64 bytes)
   before it is banked; failures print and skip — append them to
   `MANIFEST.json:exclusions.clips` with the reason.
4. Programmatic use: `import pull_camera; pull_camera.pull([clip_id, ...])`
   returns `{clip_id: path}`. Needs an HF token with read access.

## Cropping — per-clip cy is MANDATORY
`index/front_wide_cy.parquet` carries `clip_id, width, height, cx, cy, rig`.
TWO rigs exist (cy~543 rig A / cy~755 rig B); a geometric-center crop is
~215 px wrong for rig B. Crop around the CLIP'S cy, never the frame centre.

## Verify
Every file: sha256 in `MANIFEST.json`. Labels: 4,719 rows, 0 exclusion
violations, schema `s2-geom-v7`.
