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
| `tanitad/eval/` (gates, spectral, metrics, scenarios) | — | — | spectral exp/log/div sites audited safe (`clamp_min`) in the 2026-07-18 numerics sweep |
| numerics-safety class (cross-cutting) | **2026-07-18** | 0 open (class closed) | run #4 grep-sweep of all learned/data `exp`/`log`/`div` sites → every one guarded (clamp / count-gate / neg-exponent); shipped **11-test regression guard** intake `2026-07-18-numerics-safety-sweep` (test-only → `stack/tests/test_numerics_safety.py`, all green) |
| `stack/scripts/` + training loop | **2026-07-18** (part) | 1 fixed (intake), 3 logged | **review #3 (run #5):** resume-write path **atomic in every trainer** (`tmp→.replace`: `train_worldmodel.py:354`, `train_flagship4b.py:326`, `refc_train.py:136`, `refb_train.py:346`, `refa_train4b.py:303`). **LIVE BUG found + fixed:** the milestone archive is **non-atomic** in all 3 pod trainers (`train_flagship4b.py:337`, `refb_train.py:358`, `refa_train_plus.py:540`) — `shutil.copy2` guarded by `not arch.exists()` → a kill mid-copy leaves a truncated-but-existing `ckpt_step{m}.pt` the guard adopts forever → gate protocol loads a corrupt milestone. Intake `2026-07-18-atomic-milestone-archive` (`.partial`→`os.replace`, 4 tests, failing-then-passing). Logged for next run: log-hygiene (`/workspace` swallow-on-death vs `/tmp`), quota-preflight before copy (Errno122), `epcache` DONE-marker + short-episode-drop counter |

## ⭐ Thor deployment status (2026-08-02) — plan: `THOR_OPTIMISATION_PLAN.md`

- ⛔ **"Target hardware (Orin/Thor) not in-house" is RETRACTED.** A **Jetson Thor** (Blackwell
  sm_110, aarch64, L4T R38.4.0, torch 2.13.0+cu130, **TensorRT 10.13.3.9**, 122 GB unified) joined
  the fleet 2026-08-02 as `tanitad-thor`. It is now the TRT box; the dev-box `tensorrt` install
  (P1.4a) has left the critical path.
- **MEASURED on Thor** (Architecture & Inference, `…/incoming/2026-08-02-thor-deployment-profile/`):
  tick fp32 eager **272.56 ms** → bf16 encoder + CUDA-graph predictor **98.63 ms (2.76×)**;
  encoder 187.8 → **27.8 ms** bf16 (6.76×); predictor 4.23 → **1.168 ms** TRT-fp16 (3.62×);
  projected tick **≈51.2 ms (5.33×)**. TRT numerics gate PASS (fp16 rel-err 1.41e-3 @1 step,
  1.80e-3 after a 20-step roll — **no compounding**).
- ⚠️ **COVERAGE CAVEAT — the 51.2 ms is `encode_window` + ONE candidate roll.** The deployed
  `TacticalSelector.select` loops over **9 maneuvers** (`config.py:95`, `fourbrain.py:571`) and then
  decodes/scores each. Serialised through the batch-1 engine that is ~238 ms (2.4× over budget);
  batched it should be ~53 ms (on the 4060 a batch-9 `imagine` cost **0.93×** of batch-1). ⛔ **Do
  not quote 51.2 ms as "the decision tick" until B1 measures the complete path.** The shipped
  `predictor_fp16.plan` is **batch-1 static** and needs a batch-9/dynamic optimization profile.
- ⚠️ **PRECISION CONFLICT, unresolved.** Thor deploys **bf16** on the encoder on a **numerics**
  rel-err (0.0059) measured on **random weights**. Our standing policy — measured twice on the 4060
  in **decision space** — is **fp16, never bf16** (bf16: agreement 67.2 %, wp-shift 47.7 cm mean /
  3.58 m max). Different geometry (176×624 vs 256×256) and checkpoint, so neither transfers: the
  bf16 decision-agreement gate on Thor is P0 (plan §3 B2). If bf16 fails the 95.3 % bar, the 6.76×
  must be re-earned via a TRT-fp16 encoder engine (O2) and the 51.2 ms projection is void.
- 🔴 **ONNX parity is RETRACTED for torch 2.13 and UNVERIFIED for our own 2.11.** `nn.MultiheadAttention`
  fuses to `aten::_native_multi_head_attention`, which **opset 18 rejects loudly and opset 17 exports
  as a wrong graph with no error** (rel-err 0.726). Fix: `torch.backends.mha.set_fastpath_enabled(False)`
  before any export (costs 5.1e-7 in eager). ⇒ **ONNX parity must be re-verified per torch version,
  never inherited** — the 2026-07-08 "no unexportable ops" row below is admissible only for 2.11, and
  plan item A1 tests even that.
- ⛔ **Thor was OFFLINE at 2026-08-02 (this run)** — `ssh` timeout, ICMP fail, TCP 22/80 fail from the
  same subnet. Tier B of the plan is armed and waiting on a power-on.

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
- **DEPLOY TICK (MEASURED 2026-07-18 run #5, 4060, step-6500, 64 real windows, 200
  reps):** the operative tick deploys as **fp16 encoder + CUDA-graph predictor/select**
  = **17.75 → 11.16 ms, 56.3 → 89.6 Hz, 1.59×**, agreement **96.9 %** (2 flips/64),
  wp-shift **0.7 cm mean / 1.9 cm max**. The two levers COMPOSE (measured == run #4's
  additive projection to 0.4 %); the graph is zero-accuracy-cost (the 2 flips are the
  fp16 encoder's) and clock-robust (fixed ~4.4 ms replay). Absolute Hz is clock-dependent
  (non-pinnable box) but always 3–6× above the 10–20 Hz requirement. Deploy recipe:
  fp16 ViT tower + hand-rolled `torch.cuda.CUDAGraph` on the predictor/select path.
  Source: `Implementation/combined_tick/combined_tick_20260718.json`.
- ~~Per-precision peak-VRAM co-residency artifact (P1.4c)~~ **CLOSED 2026-07-18 run #5**
  (isolated one-process, exactly one model resident): fp32 **1.078 GB** (reproduces
  run #3's 1.10 GB → harness validated), fp16 **0.560 GB** (**1.93× smaller**). The
  run #3/#4 fp16 1.65 GB was fp32-reference co-residency — never quote it. Source:
  `Implementation/combined_tick/vram_{fp32,fp16}_20260718.json`.
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
- **Predictor half is LAUNCH-BOUND — CUDA-graph capture is a free win (MEASURED 2026-07-18,
  4060, fp32, 200 reps).** Manual `torch.cuda.CUDAGraph` on the operative predictor:
  predict_1pass **6.08→2.36 ms (2.57×)**, select_K9 **5.94→4.45 ms (1.33×)**; rel-err vs
  eager **2.8e-7**, imagine-and-select agreement **100 %**, waypoint shift **0.00 m** (same
  fp32 kernels — pure launch elimination). **Deploy note: on this Triton-less Windows box the
  graph route is MANUAL capture, NOT `torch.compile`** (inductor → `TritonMissing`;
  `backend="cudagraphs"` → 20× slower). Additive tick projection with fp16 encoder ≈ 9.1 ms /
  109 Hz (needs a combined harness to confirm). Source: `half_precision`-sibling
  `Implementation/predictor_latency/predictor_latency_20260718.json`.
- INT8 on ViT: native TensorRT is a known trap — OwLite/ModelOpt route confirmed (Phase 1). ModelOpt
  PTQ = in-place + calibration dataloader, QDQ nodes; keep the ViT tower FP16, quantize predictor/
  heads first, accuracy metric = probe-fit delta.
- ~~Target hardware (Orin/Thor) not in-house~~ **RETRACTED 2026-08-02 — a Jetson Thor is in-house**
  (see the Thor block above). The 4060 remains the *proxy* for quick local turns, but deployment
  numbers now come from the target itself. Orin-class remains un-owned (backlog O12).
- **Export ops gotcha (Windows dev machine):** the dynamo exporter prints emoji progress and crashes
  with `UnicodeEncodeError` under cp1252 — run exports with `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8`.
- Export-time deps (`onnx`, `onnxruntime`, `onnxscript`) are dev-only — must NOT enter the
  inference-only runtime wheel (backlog P2.10 dependency audit).

## Export-path status

| Stage | Status | Detail |
|---|---|---|
| ONNX (encoder+predictor) | **torch-2.11 only; RETRACTED for 2.13** | 2026-07-08 @torch 2.11: opset 17 (legacy) & 18 (dynamo); static [1,9,256,256] + [1,8,2048]/[1,8,2]; parity 8.8e-6 / 1.2e-5. ⛔ **@torch 2.13 the same export is silently wrong (rel-err 0.726) unless `torch.backends.mha.set_fastpath_enabled(False)`**; opset 18 fails loudly. **Per-torch-version re-verification is mandatory** — plan A1 tests whether 2.11 was a shape-dependent near-miss |
| TensorRT fp16 (predictor) | ⭐ **BUILT + numerics-gated on Thor 2026-08-02** | TRT 10.13.3.9, build 36 s / 174 MB; **1.168 ms (3.62× eager, 2.93× over the free CUDA graph)**; rel-err 1.41e-3 @1 step → 1.80e-3 @20 steps (no compounding). ⛔ **batch-1 static** — a 9-candidate fan needs a rebuild with a batch-9/dynamic profile (plan B1). ⛔ the **four-family accuracy gate on real windows has NOT run** (plan B3) |
| TensorRT fp16 (encoder) | **not started** (O2) | becomes P0 if the bf16 decision-agreement gate (plan B2) rejects bf16 |
| Quantization (OwLite/ModelOpt) | not started | Phase 1; ModelOpt PTQ (calib loader + QDQ); ViT tower FP16; accuracy metric = probe-fit delta |
