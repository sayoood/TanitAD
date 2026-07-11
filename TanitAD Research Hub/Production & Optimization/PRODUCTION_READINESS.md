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
| `tanitad/data/` (epcache, mixing, contract, loaders) | **2026-07-08 / -10** | 2 fixed + 1 fixed (intake), 1 logged | review #1 → intake `2026-07-08-data-cluster-compliance` (cache-key collision + save fail-fast, 12 tests). **review #3 (2026-07-10) → intake `2026-07-10-contract-windowing-failloud`**: silent short-episode drop across `_contract.py:120`/`toy_driving.py:131`/`comma2k19.py:278` → count+warn+`ValueError`-on-empty, parity preserved, 10 tests. Still logged: `epcache` DONE-marker written-never-read |
| `tanitad/models/` (encoder, predictor, sigreg, imagination, fourbrain) | **2026-07-09 / -11** | 2 fixed (intake), 1 logged | review #2 → intake `2026-07-09-models-predictor-failfast` (operative-predictor `assert`-only guard → `-O`-proof `ValueError`s, `predictor.py:73`, 8 tests) — also covers `tactical_pred` (same class, `fourbrain.py:49`). **review #3-cont (2026-07-11) → intake `2026-07-11-imagination-nll-logvar-clamp`**: `imagination_nll` unclamped `exp(-logvar)` overflow (`imagination.py:135`) → **silent NaN corruption** (measured; fp16 −11.09 / fp32 −88.72 boundary, reachable in 45 SGD steps) → clamp logvar to [−8,8], in-band parity 0.0, + seeded `d9_rows`, 10 tests. Still logged: unbounded `logvar_head`; hard-clamp edge-parking (softplus reparam deferred). Encoder+predictor **ONNX-clean** at opset 17 & 18/dynamo (parity ≤1.2e-5). SigReg correctly pins fp32; `eval()` disables F-5 grad-ckpt |
| `tanitad/instruments/` | — | — | |
| `tanitad/eval/` (gates, spectral, metrics, scenarios) | — | — | |
| `stack/scripts/` + training loop | — | — | review #4 (was #3); ops-fragility history F-5/F-6/F-7 |

## Deployment blockers (live list)

- **Silent-NaN in the imagination loss (FOUND + FIX SHIPPED 2026-07-11, intake pending):**
  `imagination_nll` (`imagination.py:135`) runs `exp(-logvar)` on an **unbounded** head → `+inf`
  at logvar < **−11.09 (fp16)** / **−88.72 (fp32)** (measured == `−ln(finfo.max)`), reachable by
  plain SGD in 45 steps (fp16). No nan/inf guard before `opt.step` (`train_worldmodel.py:330-358`)
  → one bad cell NaN-corrupts all weights and the atomic checkpoint. Fix = clamp logvar to [−8,8]
  (intake `2026-07-11-imagination-nll-logvar-clamp`, parity 0.0). **Still open (P1.8): the trainer
  itself has no `loss` finiteness guard** — add one as defence-in-depth before any long run.
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
- **Latency-measurement hygiene:** absolute I8/CNCE latency must be taken on an
  EXCLUSIVE, clock-pinned GPU. The 2026-07-09 run was contended (local CarlaUE4 +
  python, 99 % util) → fp32 tick read 33.5 ms vs the clean 15.07 ms; accuracy is
  contention-immune but absolute latency/Hz and per-precision peak-VRAM are not
  (VRAM double-counts the resident fp32 reference). Clean re-run = backlog P1.4b.
- **INT8 weight-quant policy (MEASURED 2026-07-10, decision space, 64 windows):** INT8-safe =
  **ViT encoder (98.4 % agreement, 1 flip) + predictor (95.3 %, 3 flips)**; INT8 red line = the
  **readout / state-projection** (heads INT8 → 48.4 %, 33 flips, 1.67 m wp-shift; `int8_all` ==
  `int8_heads`). Deploy INT8 encoder+predictor **weights** (−552 MB of 825 MB at 4×), readout ≥fp16.
  **Correction:** the "native-TRT ViT INT8 trap / keep the ViT FP16" rule is an **activation**-quant
  rule — ViT *weights* round to INT8 cleanly here. INT8 *latency* needs fused TRT int8 kernels
  (toolchain absent); today's INT8 delta = 4× weight-memory footprint. OwLite/ModelOpt PTQ route
  still confirmed for the real engine (calibration dataloader + QDQ). Source:
  `Implementation/int8_quant/int8_quant_step6500.json`.
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
| Quantization (OwLite/ModelOpt) | **weight-curve MEASURED 2026-07-10** | INT8 weight-only: **encoder 98.4 % / predictor 95.3 % SAFE; readout/heads 48.4 % UNSAFE** (localized). Plan = INT8 encoder+predictor weights + readout ≥fp16 + fp16 acts; joint bar ≥95 %/≤4 cm. Real engine (int8 kernels + latency) = P1.4a; ViT-weight-INT8 is fine (trap is activation-side) |
