# Production & Optimization — STATE

- **LAST_RUN:** 2026-07-08 (run #1, Saturday agent)
- **QUALITY:** full (one measured experiment with numbers + one compliance review with a tested
  intake package; both gates G-H and G-P1/G-P2 met)
- **Phase:** 0

## Where this stream stands (one paragraph)

First run of the Production & Optimization stream (D-020 §3). The operative-path **ONNX export is
proven**: encoder+readout and the multi-horizon predictor export clean at opset 17 (legacy) and
opset 18 (dynamo, torch-2.11's default) with PyTorch parity max|Δz| 8.8e-6 / 1.2e-5 (tol 1e-4) and
**zero unexportable ops** — the TensorRT-on-Orin path is unblocked. Compliance review #1 of
`tanitad/data/` found and fixed (as one tested intake package) a **cache-key collision** in
`epcache` that is the same silent-wrong-data class as the cosmos chunk-pairing bug, plus a
`save_episode` fail-fast guard. `PRODUCTION_READINESS.md` now records the data cluster as reviewed
and the ONNX export path as done.

## Next actions (checkboxes)

- [ ] **Next run (P1.4):** TensorRT fp16 engine from the two ONNX graphs on the 4060 — GPU latency
      vs PyTorch + accuracy Δ (max|Δz|) on 100 held-out real windows. ONNX parity already cleared.
- [ ] Read the orchestrator verdict on `2026-07-08-data-cluster-compliance` and adapt.
- [ ] Compliance review #2 target: `tanitad/models/` (numerics/determinism/malformed-window
      handling) — note the encoder/predictor are already ONNX-clean.
- [ ] Carry the two logged review-#1 findings (DONE-marker-unused; short-episode silent drop) into
      a later package.

## Standing facts / gotchas (this stream)

- RTX 4060 is the declared Orin latency proxy (I8). Decision-tick baseline: 15.07 ms p50 / 1.08 GB
  VRAM fp32 (2026-07-08 MVP loop).
- ONNX export deps (`onnx`, `onnxruntime`, `onnxscript`) installed in the venv — **export/dev only**,
  never in the inference runtime wheel.
- Windows dynamo ONNX export crashes on emoji progress under cp1252 → run with `PYTHONUTF8=1`.
- Boundary: NEVER write `stack/` directly. Experiments live in `Implementation/onnx_export/` (off-
  Drive for large `.onnx`); stack-changing fixes go through `Implementation/incoming/` intake.

## HANDOFF

None — run completed cleanly. All artifacts committed.
