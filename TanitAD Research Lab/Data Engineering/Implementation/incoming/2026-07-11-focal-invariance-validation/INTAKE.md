# INTAKE — Focal-invariance validation of D-016 + fail-loud intrinsics guard

- **Package:** `Data Engineering/Implementation/incoming/2026-07-11-focal-invariance-validation/`
- **Author agent / date:** Data Engineering agent (Tuesday), 2026-07-11
- **Proposed target:** `stack/tanitad/data/calib.py` (add `assert_effective_focal` + `achieved_focal`)
  · validation harness → `stack/scripts/focal_invariance_preview.py` · tests → `stack/tests/test_focal_invariance.py`
- **Hypothesis / WP served:** **H7** (heterogeneous-video flywheel / focal canonicalization, VLM3
  principle) · D-016 · D-010 real+sim mix consistency · Y-pilot-50 (YouTube dashcam, Y1 step)

## What & why (≤12 lines)

`calib.focal_crop_resize` (D-016) crops each camera so the effective focal at the model input is a
shared `F_REF` (266 px @ 256) — the fix for feeding the world model consistent action→pixel-motion
geometry across comma2k19 (~50°) and the 120° NVIDIA corpora. Until now this was **arithmetic-tested
only** (`test_calib.py`); no evidence it makes the **trained encoder** focal-invariant on real frames.
This package supplies (a) the first **measured validation** (G-H) and (b) a **fail-loud data-card
guard** the loaders can call at ingest. `focal_invariance.py` adds `virtual_focal_zoom` (re-image a
real scene at a longer focal, pinhole-exact), `assert_effective_focal` (raise if a camera's
focal/resolution can't reach `F_REF` — the silent-clamp failure comma-narrow cameras hit), and drift
metrics. `run_focal_invariance.py` runs the controlled correct-vs-wrong-intrinsics experiment on the
trained encoder. Research note:
`Data Engineering/Research/2026-07-11-focal-invariance-validation-and-sc13-sourcing.md`.

## Evidence & tests

- **Measured (G-H), RTX 4060, 7.9 s, $0** — `focal_invariance_result.json`. Base = Cosmos-Drive-Dreams
  front_wide_120fov (704×1280, f_eff canonicalizes to **265.7 ≈ F_REF 266** ✓), local `ckpt_full.pt`
  (partial ~step-6500 ckpt, P8), 48 scenes. Same scene at focal z·f0, encoded, **correct** vs **wrong**
  (=f0) intrinsics, relative latent drift:

  | zoom z | drift correct | drift wrong | reduction | cos correct | cos wrong |
  |---|---|---|---|---|---|
  | 1.25 | 0.041 | 0.402 | **9.8×** | 0.999 | 0.919 |
  | 1.50 | 0.049 | 0.711 | **14.5×** | 0.999 | 0.753 |
  | 2.00 | 0.063 | 0.934 | **14.9×** | 0.997 | 0.600 |

  Falsifier (`drift_correct ≥ drift_wrong`) **fails at every zoom** → **D-016 validated**: correct
  per-camera intrinsics hold the same scene's encoding near-identical (cos ≥ 0.997) across a 25–100 %
  focal change; ignoring intrinsics drives cos to 0.60 (≈ a different state). Both policies produce a
  *nominally* 266-px image — the wrong one just has the wrong physical scale, which is exactly the
  cross-corpus nuisance canonicalization removes.
- **Tests: `pytest tests/` → 8 passed / 1.7 s** on the venv (`C:\Users\Admin\venvs\tanitad`). Covers:
  `virtual_focal_zoom` magnification + z=1 identity + z<1 reject; `assert_effective_focal` passes for
  comma/120° corpora and **raises for a narrow (no-headroom) camera**; `achieved_focal` == calib
  side-channel; the **pixel-level correct-beats-wrong claim** (no encoder, deterministic); and the
  **G-D2 episode-contract test** (`assert_contract(channels=9)` through the focal + 3-frame-stack path).
- `test_calib.py` (the transform this validates) still **3 passed**. No stack file touched.

## Risk & rollback

- Blast radius: **additive only** — nothing imports the package into `stack/` yet; the proposed
  integration adds two pure functions to `calib.py` (no signature change to `focal_crop_resize`) plus a
  read-only script. 0 new dependencies (`av`/torch present).
- Known limits (P8): partial checkpoint (encoder still moving); **extrinsics (mount pitch/height) NOT
  normalized** — this validates the *intrinsic* axis only (horizon-homography is the D-016 R1 follow-up);
  `virtual_focal_zoom` cannot widen (z≥1), so the perturbation runs on a wide base corpus with headroom.
- Rollback: delete the package; nothing depends on it.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:**
- **Date / by:**
- **Reason & notes:**
- **Integrated as:**
