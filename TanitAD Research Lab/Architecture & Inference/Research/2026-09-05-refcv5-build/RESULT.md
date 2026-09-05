# refcv5 BUILD — Architecture & Inference FlyWheel

**STATUS: IN PROGRESS** · opened 2026-09-05 · agent generation 10 (nine predecessors died mid-work)
**Branch:** `agent/arch-inf-20260803` · **Bank:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refcv5-build/`

## The mandate (PI, 2026-09-05, verbatim)

> "I propose to prepare refcv5 for training, implement all missing pieces. Let's follow environment
> extraction from the front camera and train it based on GT data. If the whole refcv5 is driving and
> achieves good quality, we can extend it to multi cameras (first step 4) then to Lidar."

**This is a BUILD, not a refutation.** The deliverable is working code that makes refcv5 trainable.
Gates and controls apply — they stop us fooling ourselves — but a clean refutation is not a deliverable.

## Build order (each ships with a test that fails without it)

| # | component | status |
|---|---|---|
| 1 | Environment extraction from front camera, GT-supervised (`obstacle.offline` labels → mono-3D agent head → agent tokens) | PENDING |
| 2 | Score the fan we actually emit (four-family heads on the EMITTED fan) | PENDING |
| 3 | Diffusion in CONTROL space (anchored Gaussian, DDIM, x0-pred, per-layer AdaLN) | PENDING |
| 4 | Trainable end to end (`refc_v3_train.py` flags, seams stamp, tiny-rig smoke, launch argv) | PENDING |

⛔ **Do not launch the full training** — that is the Master Mind's call.
⛔ `refcv4b-b1-v72-40k` is LIVE on pod `tanitad-refcv3` — never touched.

## Binding constraints carried

- Frames are **256×640 CYLINDRICAL**, `f_ref` 305.577, **HFOV 120°** — the pinhole formula is WRONG
  here. Project only through `calib.py::CanonicalFrame`.
- Rig **z = 0 is the road plane** (MEASURED over 87,481 cuboids: ground-standing classes' bottom
  faces −0.05 to −0.13 m; `protruding_object` +1.68 m). Camera height 1.43–1.56 m.
- `obstacle.offline` is a **LABEL** (privileged); inference is **VISION-ONLY**.
- Canonical `ha0_ext` is the INTEGRATOR `refav1_arm.hold_ext_controls`, never the closed form
  (§M11 — they differ by 0.54 m at 2 s).
- Every seam **zero-init and gated**: a v5 build with all gates off must be **bit-identical to v4b**.

## Log

- `2026-09-05` STATUS header banked before first line of code.

## Deliverable manifest

(appended as components land)
