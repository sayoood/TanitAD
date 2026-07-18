# Production & Optimization — Knowledge Base

Deltas only, deduplicated, newest first. Each entry: fact + source (repo path or URL).

## 2026-07-18 (run #4)

- **The batch-1 predictor is launch-bound, and manual CUDA-graph capture is a FREE
  fix.** On the exclusive 4060 (fp32, tf32 off, step-6500, 64 real comma windows,
  200 reps), `torch.cuda.CUDAGraph` capture of the operative predictor pass:
  predict_1pass **6.08 → 2.36 ms (2.57×)**, select_K9 **5.94 → 4.45 ms (1.33×)**;
  accuracy vs eager max|Δ| 7.6e-6, **rel-err_max 2.8e-7**, cosine 1.0, imagine-and-
  select **agreement 100 %**, waypoint shift **0.00 m**. The graph replays the same
  fp32 kernels → the win is pure kernel-launch elimination. **Falsifier (>10 %)
  cleared 25×** → run #3's launch-bound diagnosis CONFIRMED. Source:
  `Implementation/predictor_latency/predictor_latency_20260718.json`.
- **The gain scales inversely with batch (launch-boundedness).** batch-1 predict 2.57×
  ≫ K=9 select 1.33× (at K=9 each kernel does 9× the work → launch overhead is a
  smaller fraction). **Two orthogonal deploy levers: encoder = compute-bound → fp16;
  predictor = launch-bound → CUDA graph.** Additive tick projection: fp16 encoder
  (4.69) + graph select (4.45) ≈ **9.1 ms ≈ 109 Hz** (from ~68 Hz fp32) — projection,
  needs a combined harness to confirm.
- **On this Windows box the deployable graph route is MANUAL `torch.cuda.CUDAGraph`,
  NOT `torch.compile`.** `torch.compile(mode="reduce-overhead")` → **`TritonMissing`**
  (Triton not installed; inductor needs it for GPU codegen). `torch.compile(backend=
  "cudagraphs")` runs Triton-free but is **~20× SLOWER** (117.8 ms) — per-call dynamo
  guard/re-trace overhead swamps the tiny predictor. Accuracy fine in both; only manual
  capture is viable. To enable inductor on the dev box, install a Windows Triton wheel
  (torch 2.11+cu128). Source: this run's JSON.
- **Numerics-safety class is CLOSED (P0 #4 grep-sweep of `stack/tanitad`).** Every
  unbounded `exp`/`log`/div on a learned/data output is guarded: imagination
  logvar `.clamp(-10,10)` (head+nll), OKRI/LOPS sigma `.clamp` (`arms.py:284`),
  `FeatureOOD.score` `count<2→zeros`+`var.clamp_min(eps)` (`refb.py:366`), spectral/
  fourbrain-erank `.clamp_min(1e-12)`, sigreg/epps/metrics bounded-by-construction
  (neg-exponent Gaussian kernels / clamped denom). No new unguarded site. Shipped as
  an **11-test executable regression guard** (intake `2026-07-18-numerics-safety-sweep`,
  test-only, all green) — witnesses: unclamped nll → **inf** at logvar=-100 while the
  guarded call is finite; head bounds logvar at 500× input scale; FeatureOOD finite
  before 2 samples / under zero variance.

## 2026-07-17 (run #3)

- **Clean-GPU absolutes (the 33.5 ms was contention, not a regression).** On the
  now-EXCLUSIVE 4060 (step-6500, 64 real windows), the fp32 decision tick is
  **14.79 ms / 67.6 Hz / 1.102 GB** — reproduces the 15.07 ms clean baseline
  (2026-07-08), confirming the 2026-07-09 33.5 ms was purely CarlaUE4 contention.
  fp16 **10.67 ms / 93.7 Hz / 1.39×**. Both far above the 10–20 Hz operative
  requirement before any TRT/quant (3.4–4.7× headroom). Source:
  `Implementation/half_precision/half_precision_clean_20260717.json`.
- **fp16's whole speedup is the ViT ENCODER** (encode 8.98 → 4.69 ms ≈ 1.9×); the
  predictor + K9-select passes barely move (5.81 → 5.99 ms) — at batch-1 they are
  launch/memory-bound, not compute-bound. **Production consequence: the batch-1
  latency lever is encoder precision.** INT8/FP8 on the predictor/heads (planned
  P1.6) buys ~0 batch-1 latency; the encoder holds the time and must stay ≥fp16
  (bf16 unsafe). Re-order P1.6: **quantize for VRAM/energy, not batch-1 latency.**
- **Precision policy reproduced to the digit on the clean run:** fp16 SAFE
  (agreement **95.3 %**, 3/64 flips, wp-shift 3.9 cm mean/19 cm max, enc rel-err
  7.8e-4); bf16 UNSAFE (**67.2 %**, 21/64 flips, 47.7 cm mean/**3.58 m max**,
  7.2e-3). Same numbers as 2026-07-09 across a different probe fit → the policy is
  reproducible. Deploy fp16, never bf16 (G-P2: bf16 is the *same* 1.39× but flips
  1 in 3 maneuvers).
- **Half-precision peak-VRAM is NOT lower in the current harness — it's a co-residency
  artifact.** fp16/bf16 read 1.65 GB vs fp32 1.10 GB because the accuracy harness keeps
  the fp32 reference model resident (261 M × 2 B ≈ the 0.52 GB delta). Only the fp32
  standalone (1.10 GB) is trustworthy; clean fp16 VRAM needs a one-process-per-precision
  harness (backlog P1.4c). Never quote the co-resident number as fp16's footprint.
- **H15 `imagination_nll` had an unclamped `exp(-logvar)` overflow** (`imagination.py:135`):
  `logvar=-100 → loss=inf → NaN grads`. LIVE in `train_worldmodel.py:338` (flagship),
  `train_flagship4b.py:164`, `finetune_traj.py:217`; the logvar also `.exp()`s in
  `replay/arms.py:284` (OKRI/LOPS/H2 export). Fix = clamp logvar to `[-10,10]` at the
  head output + defensively in the nll (behaviour-preserving in-range, allclose test).
  Source: `Implementation/incoming/2026-07-17-imagination-logvar-clamp/` (17 tests).

## 2026-07-09 (run #2)

- **FP16 is decision-safe on the operative path; BF16 is NOT.** Measured on 64
  real comma2k19 windows (step-6500), imagine-and-select over a K=9 fan decoded
  by one fixed fp64 probe: **fp16** → selection agreement **95.3 %** (3/64 flips),
  encoder rel-err 7.8e-4, decoded-waypoint shift **3.9 cm mean/19 cm max**;
  **bf16** → agreement **67.2 %** (21/64 flips), rel-err 7.2e-3, shift
  **47.7 cm mean/3.58 m max**. Both finite (no overflow). Mechanism: the deltas
  are **precision-limited, not range-limited**, so fp16's 10 mantissa bits vs
  bf16's 8 (~9× lower latent error) decide it; a tight argmin fan amplifies 9× →
  5 % vs 33 % flips. **Deploy TRT-fp16, never bf16, on the decision path.**
  Cosine stays ≈1.0 for both → cosine is too coarse; score precision in the
  DECISION space (agreement, waypoint metres), not latent space (G-P2).
  Source: `Implementation/half_precision/half_precision_step6500.json`.
- **TensorRT-proper is not installable on the dev box as-is:** `import tensorrt`
  → ModuleNotFoundError; **onnxruntime exposes only the CPU EP** (no CUDA/TRT
  provider). A TRT-fp16 engine build needs `tensorrt` + `onnxruntime-gpu` (CUDA
  12 EP) installed, or an idle-pod build. Non-paid → EXECUTE-class, sequence
  behind a clean-GPU window. Source: measured this run (`nvidia-smi`, ORT
  `get_available_providers()`).
- **Latency benchmarks need an EXCLUSIVE, clock-pinned GPU.** This run's absolute
  latency was contended — the 4060 was at 99 % util with a local `CarlaUE4` (~4 GB)
  + python resident, inflating the fp32 decision tick to 33.5 ms vs the clean
  15.07 ms (2026-07-08). Numerics/accuracy are contention-immune; **absolute
  latency/Hz and per-precision peak-VRAM are not** (VRAM also double-counts the
  resident fp32 reference). Pin clocks (`nvidia-smi -lgc`) + no other compute apps
  before quoting I8/CNCE absolutes. Source: measured this run.
- **Operative predictor had an `assert`-only input guard** (`predictor.py:73`) →
  stripped under `python -O`, a wrong (shorter) window then **re-aligns on every
  axis and runs silently** (silent-wrong-data class); wrong dims threw cryptic
  matmul `RuntimeError`s. Fixed via `validate_operative_inputs` (named-axis
  `ValueError`s, `-O`-proof, shape-int checks constant-fold on ONNX export).
  Source: `Implementation/incoming/2026-07-09-models-predictor-failfast/`.

## 2026-07-08 (run #1)

- **The TanitAD-4B operative path is ONNX-exportable with no op changes.** encoder+readout
  `[1,9,256,256]→[1,2048]` and predictor `states[1,8,2048],actions[1,8,2]→(z1,z2,z4)` export clean
  at **opset 17 (legacy exporter)** and **opset 18 (`dynamo=True`, torch-2.11 default)**. Parity vs
  PyTorch (fp32, strict numerics, ORT-CPU): max|Δz| 8.8e-6 (encoder) / 1.2e-5 (predictor), tol 1e-4.
  `nn.MultiheadAttention` (`need_weights=False`), FiLM, the causal `torch.triu` bool mask, and the
  `AvgPool2d` readout all have stable symbolics. Predictor dict output must be wrapped to a tuple.
  Source: `Implementation/onnx_export/{export_encoder_predictor.py,parity.json}`.
- **ONNXRuntime-CPU is 1.4–4.4× SLOWER than PyTorch-CPU** on our graphs (encoder 455 vs 103 ms;
  predictor 24.6 vs 17.1 ms, 1 thread each). ONNX's value is the portable IR for TensorRT-on-Orin,
  NOT a CPU speedup — never quote ORT-CPU latency as an optimization win (G-P2). Source: parity.json.
- **torch 2.11 removed the legacy-exporter fallback; `dynamo=True` is default since 2.9.** The
  literature warns of a fused `_native_multi_head_attention` `UnsupportedOperatorError` trap — but it
  does NOT fire for our `need_weights=False` blocks under `torch.export` (verified, parity 6.7e-6).
  Drop-in fallback if a future torch regresses: replace MHA forward with plain `Linear`/`bmm`/
  `softmax`. Source: `Research/2026-07-08-...md` §1/§3, https://docs.pytorch.org/docs/2.12/onnx.html
- **Windows ONNX-export gotcha:** the dynamo exporter prints emoji (`✅`) progress and crashes with
  `UnicodeEncodeError` under the default cp1252 console. Run exports with `PYTHONUTF8=1` /
  `PYTHONIOENCODING=utf-8`. Source: measured this run.
- **NVIDIA ModelOpt is the INT8/FP8/FP4 PTQ route** (in-place quantize + calibration dataloader →
  QDQ nodes TensorRT fuses to INT8/FP8 kernels). Multi-modal stacks keep the **vision tower FP16 by
  default** — reinforces "native-TRT ViT INT8 is a trap; quantize predictor/heads first, ViT stays
  higher precision, accuracy metric = probe-fit delta." Sources:
  https://www.spheron.network/blog/tensorrt-model-optimizer-modelopt-quantization-guide/ ,
  https://developer.nvidia.com/blog/model-quantization-turn-fp8-checkpoints-into-high-performance-inference-engines-with-nvidia-tensorrt/
- **`epcache` cache key was collision-prone** (basename-only path id; `None` for dict-without-
  `clip_id`) — same silent-wrong-data class as the cosmos chunk-pairing bug. Fixed via full-path /
  clip_id-or-raise identity. Source: `Implementation/incoming/2026-07-08-data-cluster-compliance/`.
- **`eval()` disables the F-5 grad-checkpoint lever** (`encoder.py:60-61` gates it on
  `self.training and t.requires_grad`) — so no export-specific flag is needed to turn checkpointing
  off; just export in eval mode. Source: `stack/tanitad/models/encoder.py`.
