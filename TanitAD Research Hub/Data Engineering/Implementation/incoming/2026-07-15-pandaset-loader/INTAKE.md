# INTAKE — PandaSet loader (CC-BY-4.0 owned-core real-urban adapter)

- **Package:** `Data Engineering/Implementation/incoming/2026-07-15-pandaset-loader/`
- **Author agent / date:** Data Engineering agent (Tuesday), 2026-07-15
- **Proposed target:** `stack/tanitad/data/pandaset.py` (sibling of `cosmos_drive.py`) + `stack/tests/test_pandaset.py`
- **Hypothesis / WP served:** H7 (data flywheel) / H4 arm-B real corpus / OWN_DATASET_PLAN §7 ingest #2

## What & why (≤10 lines)

A contract-clean loader for **PandaSet** (Hesai/Scale) — the cheapest **license-CLEAN real-urban** add in the
OWN_DATASET_PLAN (CC-BY-4.0, SF/El-Camino urban day). It reuses the Cosmos geometry: `poses.json`→signals via a
motion-heading 4×4 (offset-free — avoids the unknown camera→vehicle extrinsic) → the tested
`cosmos_drive.poses_to_signals`; D-015 9-ch stack; `CORPUS_META` byte-identical to comma2k19 (I7 → admissible in
the D-010 mix); SEQUENCE-level split (I3). Schema grounded from the pandaset-devkit source
(`poses.json`=`[{position:{x,y,z}, heading:{w,x,y,z}}]`, `intrinsics.json`=`{fx,fy,cx,cy}`), not guessed.
The pose/signal/contract path is **VALIDATED**. Geometry is **BLOCKED-by-design** (see risk) — the loader fails
loud rather than shipping mis-scaled frames. Note:
`Research/2026-07-15-worldmodel-pose-gate-and-pandaset-geometry.md`.

## Evidence & tests

- Tests: `tests/test_pandaset.py` — **16 passed (1.6 s)** on author machine (venv `C:/Users/Admin/venvs/tanitad`,
  torch 2.11). Zero real bytes / zero Pillow (decode + pose IO injected). Covers: analytic steer on a constant
  arc, motion-heading offset-invariance, straight-line/low-speed guards, `assert_contract(channels=9)`, I7
  fingerprint == comma2k19, I3 sequence split, devkit-schema parsing, discovery, mix admissibility, and the
  geometry blocker (below).
- **Measured (grounded on the REAL front-camera calibration, arXiv 2112.12610 / devkit: fx=1970.01, 1920×1080,
  k1=−0.589):** centered square-crop canonicalization is **height-bound** → ideal crop 1896 px but frame is
  1080 tall → clamps to 1080 → **achieved f_eff = 467 px vs canonical 266 px** (≈1.75× scale mismatch),
  `height_clamped=True`, `drop_in=False`. Closed form: on a 1080-tall frame any camera with `fx>1122 px` is
  height-bound. A wide pinhole (fx=1000) canonicalizes cleanly (f_eff=266.1, drop_in=True) — the guard is not
  over-broad.

## Risk & rollback

- **Geometry blocker (the reason this is not yet integrate-and-mix):** PandaSet's front camera cannot be
  square-crop-canonicalized to F_REF=266 on its 16:9 frame, and its barrel distortion (k1=−0.589) is ignored by
  the pinhole `focal_crop_resize` — the same silent-wrong-geometry class as the pre-D-016 fisheye bug and the
  in-flight two-rig `cy` fix. The loader **refuses** (`GeometryError`, `strict=True` default) so it cannot
  pollute the owned mix. **Requested D-016 R1 in `stack/tanitad/data/calib.py` (NOT edited here — boundary):**
  a pad/letterbox-aware crop (replicate-pad the below-frame overflow so the square may exceed frame height and
  reach 266) **+ undistort** using the shipped distortion coeffs. This is a **prerequisite for the whole owned
  real-urban tier** (ZOD fisheye, Udacity narrow-FOV all hit the same bound), not a PandaSet-only detail. Once it
  lands, PandaSet flips BLOCKED→drop-in by switching `strict`/using the pad-crop.
- Blast radius if integrated as-is: additive — one new `stack/tanitad/data/pandaset.py` (+ its test); imports
  existing `cosmos_drive`/`calib`/`comma2k19`/`_contract`; no change to any existing module. Because it fails
  loud on real front frames, integrating it does **not** risk silent bad data — worst case it is a
  ready-but-blocked loader until D-016 R1.
- Deferred real-bytes verification (`verify_real_clip`, documented, not in CI — the Cosmos precedent): confirm
  PandaSet's world-frame planarity/axis order and the camera↔pose timestamp alignment; and recover the constant
  camera-yaw offset (motion-heading vs `heading` quaternion) that would justify switching to quaternion+offset.
  Needs the ~44.5 GB HF mirror (`georghess/pandaset`, monolithic zip) — a pod/large-disk job, not this run.
- Rollback: delete the module + test; nothing in `stack/` depends on it.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate / integrate-with-changes / defer / reject
- **Date / by:**
- **Reason & notes:**
- **Integrated as:**
