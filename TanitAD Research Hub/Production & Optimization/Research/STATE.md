# Production & Optimization — STATE

- **LAST_RUN:** 2026-07-09 (run #2, Saturday agent)
- **QUALITY:** full (one measured experiment with decision-grade numbers + one compliance review
  with a tested intake package; G-H, G-P1, G-P2 met. TRT-proper toolchain-blocked but the mandated
  measured experiment ran — precursor, not a skip.)
- **Phase:** 0

## Where this stream stands (one paragraph)

Run #2. **Precision policy is now measured:** on 64 real comma2k19 windows (step-6500, 4060),
**fp16 is decision-safe** (imagine-and-select agreement 95.3 %, encoder rel-err 7.8e-4, decoded-
waypoint shift 3.9 cm mean) but **bf16 is NOT** (67.2 % agreement — 1/3 maneuver picks flip —
rel-err 7.2e-3, 47.7 cm mean/3.58 m max shift), both finite. **Deploy TRT-fp16, never bf16**; the
G-P2 point made concrete (speed alone would have picked bf16). TensorRT-proper is **toolchain-
blocked** (`import tensorrt`→missing, ORT CPU-only) so the engine build is split to P1.4a (install
`tensorrt`+`onnxruntime-gpu` or build on an idle pod) with a **pre-registered fp16 acceptance bar**.
Compliance review #2 of `tanitad/models/` shipped a fail-fast intake (operative-predictor
`assert`-only guard → `-O`-proof `ValueError`s, `predictor.py:73`, 8 tests, export-safe). Review #1
(data cluster) was **integrated (integrate-with-changes)** by the orchestrator — no further action.

## Next actions (checkboxes)

- [ ] **Next run — P1.4a:** build the TRT-fp16 engine once the toolchain is in (dev-box
      `tensorrt`+`onnxruntime-gpu`, or on the pod when a trainer is idle); verify against the fp16
      bar (agreement ≥95 %, wp-shift ≤~4 cm). If GPU stays contended, do review #3 (`stack/scripts/`)
      instead and defer P1.4a to a clean-GPU window.
- [ ] **P1.4b:** clean latency re-measure on an exclusive, clock-pinned 4060 → publish fp16 absolute
      Hz + clean VRAM delta (this run's absolutes were CarlaUE4-contended).
- [ ] Carry logged review-#2 findings into a numerics package: `tactical_pred` fail-fast +
      `imagination_nll` `exp(-logvar)` clamp (backlog P1.7).

## Standing facts / gotchas (this stream)

- RTX 4060 is the declared Orin latency proxy (I8). Clean decision-tick baseline: **15.07 ms p50 /
  1.08 GB VRAM fp32** (2026-07-08 MVP loop, exclusive GPU). **Absolute latency needs an exclusive,
  clock-pinned GPU** — the 2026-07-09 run was CarlaUE4-contended (99 % util) and read 33.5 ms; only
  the accuracy deltas and speedup *ratios* survive contention, not absolute Hz or per-precision VRAM.
- **Precision policy: fp16 on the decision path, never bf16** (measured 2026-07-09). Keep the ViT
  tower ≥fp16. TRT-fp16 acceptance bar pre-registered (≥95 % agreement, ≤~4 cm wp-shift on 64 windows).
- **TensorRT not installed on the dev box** (`import tensorrt`→missing; ORT CPU EP only). The ONNX
  IR is exported + parity-clean, so only the toolchain/engine step remains.
- ONNX export deps (`onnx`, `onnxruntime`, `onnxscript`) installed in the venv — **export/dev only**,
  never in the inference runtime wheel.
- Windows dynamo ONNX export crashes on emoji progress under cp1252 → run with `PYTHONUTF8=1`.
- Boundary: NEVER write `stack/` directly. Experiments live in `Implementation/onnx_export/` (off-
  Drive for large `.onnx`); stack-changing fixes go through `Implementation/incoming/` intake.

## HANDOFF

None — run completed cleanly. All artifacts committed.
