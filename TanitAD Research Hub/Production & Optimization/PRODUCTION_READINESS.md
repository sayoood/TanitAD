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
| `tanitad/data/` (epcache, mixing, contract, loaders) | — | — | review #1 scheduled (backlog P0.3) |
| `tanitad/models/` (encoder, predictor, sigreg, imagination, fourbrain) | — | — | review #2 |
| `tanitad/instruments/` | — | — | |
| `tanitad/eval/` (gates, spectral, metrics, scenarios) | — | — | |
| `stack/scripts/` + training loop | — | — | review #3; ops-fragility history F-5/F-6/F-7 |

## Deployment blockers (live list)

- No batch-1 latency baseline yet (backlog P0.1) — CNCE denominator missing.
- No ONNX export path yet (backlog P0.2); grad-checkpoint flags must be disabled for export.
- INT8 on ViT: native TensorRT is a known trap — OwLite/ModelOpt route decided (Phase 1).
- Target hardware (Orin/Thor) not in-house; RTX 4060 is the declared latency proxy (I8).

## Export-path status

| Stage | Status | Detail |
|---|---|---|
| ONNX (encoder+predictor) | not started | opset TBD; static [1,9,256,256] |
| TensorRT fp16 | not started | after ONNX parity |
| Quantization (OwLite/ModelOpt) | not started | Phase 1; accuracy metric = probe-fit delta |
