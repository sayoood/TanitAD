# Production Readiness — status matrix

Maintained by the Saturday Production & Optimization agent (D-020 §3). One row per module;
review = full checklist pass with file:line findings; issues land as intake packages.

**Checklist per module:** typing coverage · error handling & failure modes · logging/observability
· determinism & seed discipline · resource cleanup (files/handles/GPU mem) · dependency hygiene ·
API stability · docstring accuracy vs behavior · batch-1/streaming compatibility · export-path
compatibility (ONNX/TRT).

## Module status

| Module | Reviewed | Open issues | Notes |
|---|---|---|---|
| `tanitad/data/` (epcache, mixing, contract, loaders) | **2026-07-08** | 2 fixed (intake), 2 logged | review #1 done → intake `2026-07-08-data-cluster-compliance` (cache-key collision + save fail-fast, 12 tests); DONE-marker-unused + short-episode-silent-drop logged for later |
| `tanitad/models/` (encoder, predictor, sigreg, imagination, fourbrain) | — | — | review #2. NOTE: encoder+predictor **ONNX-clean** at opset 17 & 18/dynamo (parity ≤1.2e-5) — no export-blocking ops |
| `tanitad/instruments/` | — | — | |
| `tanitad/eval/` (gates, spectral, metrics, scenarios) | — | — | |
| `stack/scripts/` + training loop | — | — | review #3; ops-fragility history F-5/F-6/F-7 |

## Deployment blockers (live list)

- ~~No batch-1 latency baseline~~ **MEASURED 2026-07-08** (`stack/scripts/latency_cnce_baseline.py`,
  step-6500 ckpt, 4060 fp32 strict-numerics batch-1): decision tick **15.07 ms p50** (encode
  9.38 + K9 select 5.69), p95 ≈ 17.2 ms, peak VRAM **1.08 GB** → ~66 Hz un-optimized. The
  operative-rate requirement (10–20 Hz) is met with 3–6× headroom BEFORE TensorRT/quantization.
- ~~No ONNX export path yet~~ **DONE 2026-07-08** (`Implementation/onnx_export/`): encoder+readout
  and predictor export **clean at opset 17** (legacy exporter) AND opset 18 (dynamo, torch's
  2.11 default); parity vs PyTorch max|Δz| **8.8e-6 / 1.2e-5** (tol 1e-4). No unexportable ops —
  MHA/FiLM/causal-triu all supported. `eval()` disables the grad-checkpoint (F-5) lever for export.
  ORT-CPU is 1.4–4.4× SLOWER than Torch-CPU (expected; ONNX value = TRT-on-Orin IR, not CPU speed).
- INT8 on ViT: native TensorRT is a known trap — OwLite/ModelOpt route confirmed (Phase 1). ModelOpt
  PTQ = in-place + calibration dataloader, QDQ nodes; keep the ViT tower FP16, quantize predictor/
  heads first, accuracy metric = probe-fit delta.
- Target hardware (Orin/Thor) not in-house; RTX 4060 is the declared latency proxy (I8).
- **Export ops gotcha (Windows dev machine):** the dynamo exporter prints emoji progress and crashes
  with `UnicodeEncodeError` under cp1252 — run exports with `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8`.
- Export-time deps (`onnx`, `onnxruntime`, `onnxscript`) are dev-only — must NOT enter the
  inference-only runtime wheel (backlog P2.10 dependency audit).

## Export-path status

| Stage | Status | Detail |
|---|---|---|
| ONNX (encoder+predictor) | **DONE 2026-07-08** | opset 17 (legacy) & 18 (dynamo); static [1,9,256,256] + [1,8,2048]/[1,8,2]; parity 8.8e-6 / 1.2e-5; no plugins needed |
| TensorRT fp16 | **unblocked (next run)** | build from the ONNX graphs on the 4060; report GPU latency + accuracy Δ on 100 held-out windows (backlog P1.4) |
| Quantization (OwLite/ModelOpt) | not started | Phase 1; ModelOpt PTQ (calib loader + QDQ); ViT tower FP16; accuracy metric = probe-fit delta |
