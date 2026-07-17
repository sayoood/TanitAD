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
| `tanitad/models/` (encoder, predictor, sigreg, imagination, fourbrain) | **2026-07-17** | 2 fixed (intakes), 1 logged | review #2 → intake `2026-07-09-models-predictor-failfast` (operative-predictor `assert`-only guard → `-O`-proof `ValueError`s, `predictor.py:89`, 8 tests, export-safe; **still unmerged** — the assert is live at `predictor.py:89`, same class covers `tactical_pred`). **review #3 (2026-07-17) → intake `2026-07-17-imagination-logvar-clamp`:** the logged `imagination_nll` unclamped `exp(-logvar)` overflow is now FIXED (clamp logvar to [-10,10] at head + in nll; `logvar=-100→inf` reproduced; 17 tests) — a live NaN-a-training-run mode in the flagship path (`train_worldmodel.py:338`) + NaN in the OKRI/LOPS export (`replay/arms.py:284`). Encoder+predictor **ONNX-clean** opset 17/18 (parity ≤1.2e-5). SigReg pins fp32; `eval()` disables F-5 grad-ckpt |
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
- **Precision policy (MEASURED 2026-07-09, 4060, step-6500, 64 real windows):**
  deploy **fp16** on the decision path, **never bf16**. fp16 → imagine-and-select
  agreement 95.3 %, encoder rel-err 7.8e-4, decoded-waypoint shift 3.9 cm mean;
  bf16 → agreement **67.2 %** (1/3 maneuver picks flip), rel-err 7.2e-3, shift
  **47.7 cm mean/3.58 m max**. Both finite (precision-limited, not range-limited).
  Keep the ViT tower ≥fp16. Pre-registered TRT-fp16 acceptance bar: match fp16
  (agreement ≥95 %, wp-shift ≤~4 cm) on these 64 windows. Source:
  `Implementation/half_precision/half_precision_step6500.json`.
- **TensorRT toolchain NOT installed on the dev box:** `import tensorrt` →
  ModuleNotFoundError; onnxruntime has **CPU EP only**. TRT-fp16 engine build
  needs `tensorrt` + `onnxruntime-gpu` (CUDA-12 EP) or an idle-pod build (backlog
  P1.4a). ONNX IR already exported + parity-clean, so the graph side is ready.
- ~~**Latency-measurement hygiene / P1.4b clean re-measure**~~ **DONE 2026-07-17
  (clean, exclusive 4060):** fp32 tick **14.79 ms / 67.6 Hz / 1.102 GB** (reproduces
  the 15.07 ms baseline → the 2026-07-09 33.5 ms WAS CarlaUE4 contention); fp16
  **10.67 ms / 93.7 Hz / 1.39×**, decision-safe (95.3 % agreement, 3.9 cm wp-shift);
  bf16 same 1.39× but unsafe (67.2 %). ~68 Hz fp32 / ~94 Hz fp16 = 3.4–4.7× headroom
  over the 10–20 Hz requirement before TRT/quant. **fp16's whole win is the ViT
  encoder** (8.98→4.69 ms ≈1.9×); predictor/select are batch-1 latency-floored →
  the latency lever is encoder precision, P1.6 quant is a VRAM/energy play not a
  batch-1-latency play. Clocks not admin-pinnable on this box (p50/p95 over 100 reps
  mitigates). Source: `half_precision_clean_20260717.json`.
- **Per-precision peak-VRAM still needs a one-process harness (P1.4c):** the fp16/bf16
  VRAM rows (1.65 GB) are co-resident-inflated (the accuracy harness keeps the fp32
  reference model alive; 261 M×2 B ≈ the 0.52 GB delta). Only fp32 standalone (1.10 GB)
  is clean. Measure each precision in its own process for a true fp16 footprint.
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
| TensorRT fp16 | **toolchain-blocked** (P1.4a) | `import tensorrt`→missing, ORT CPU-only. **fp16 precursor MEASURED 2026-07-09**: fp16 decision-safe (agreement 95.3 %, wp-shift 3.9 cm), bf16 NOT (67.2 %, 47.7 cm). Acceptance bar pre-registered. Install `tensorrt`+`onnxruntime-gpu` or build on idle pod |
| Quantization (OwLite/ModelOpt) | not started | Phase 1; ModelOpt PTQ (calib loader + QDQ); ViT tower FP16; accuracy metric = probe-fit delta |
