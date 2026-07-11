# Production & Optimization — Knowledge Base

Deltas only, deduplicated, newest first. Each entry: fact + source (repo path or URL).

## 2026-07-11 (run #4)

- **`imagination_nll` has a live silent-NaN path via unclamped `exp(-logvar)`.**
  `imagination.py:135` computes `torch.exp(-logvar)` (logvar from an **unbounded**
  Linear head, imagination.py:110). It overflows to `+inf` at logvar < **−88.72**
  (fp32) / **−11.09** (fp16) — measured == theory `−ln(finfo.max)`. Wired into the
  live trainer (`train_worldmodel.py:338`) with **no nan/inf guard** before
  `backward`/`opt.step`, so one non-finite cell NaNs every gradient (clip_grad_norm
  can't recover), NaNs every weight, and the atomic save persists a corrupted resume
  point. Source: `Implementation/imagination_nll_overflow/logvar_overflow.json`,
  `Research/2026-07-11-...md` §1.
- **The overflow is REACHABLE, not theoretical.** The heteroscedastic NLL's own
  per-cell optimum is `logvar* = ln(err2)`; any cell with err2 < **1.53e-5** has its
  optimum past the fp16 boundary, so plain SGD toward it goes non-finite in **45 steps**
  (fp16, err2=1e-7) / 356 (fp32). Gradient explodes to **2.4e8@−20, 2.8e34@−80** even
  before overflow → clip_grad_norm collapses the step onto it. This is the textbook
  β-NLL instability (Seitzer et al., ICLR 2022: driving logvar→−∞ on easy cells).
  Source: same run.
- **Fix = clamp logvar to [−8, 8] before the exp** (default `logvar_clamp=8.0`): finite
  in fp32 AND fp16 across err2∈[0,20], and **identity in-band → parity vs the stack
  function = 0.0 exactly** (no retraining implied). Symmetric so the "infinite-uncertainty"
  direction is bounded too. Caveat: hard clamp has zero gradient outside the band (logvar
  can park at the edge; loss stays bounded) — softplus reparam deferred. Also seeded
  `d9_rows`' `randperm` (`generator=`) → reproducible D9 chance-floor. Source:
  `Implementation/incoming/2026-07-11-imagination-nll-logvar-clamp/` (10 tests).
- **General rule (adds to the review checklist):** any `exp`/`1/x`/`log` fed by an
  **unbounded** head is a deploy-time NaN risk; and the trainer needs a `loss` finiteness
  guard before `opt.step` (P1.8) as defence-in-depth. Source: this review.
- **`tactical_pred` fail-fast is subsumed, not a separate task:** `tactical_pred` is an
  `OperativePredictor` instance (`fourbrain.py:49`), same class as the operative predictor
  whose `assert`-guard (`predictor.py:73`) the pending review-#2 intake already fixes — one
  fix covers both. Source: `fourbrain.py:49`, grep this run.

## 2026-07-10 (run #3)

- **INT8 weight-quant sensitivity localizes to the READOUT, not the ViT tower.**
  Per-output-channel symmetric int8 **weight-only** fake-quant (activations fp32),
  scored in decision space on the same 64 windows (step-6500, 4060): `int8_encoder`
  (ViT only) **98.4 %** agreement / 1 flip / 1.6 cm wp-shift; `int8_predictor`
  **95.3 %** / 3 flips / 10.9 cm; `int8_heads` **48.4 %** / 33 flips / **1.67 m**;
  `int8_all` **48.4 %** (== heads). Since all-model == heads while encoder-alone and
  predictor-alone are safe, the collapse is entirely in **heads−predictor =
  {readout, inv_dyn, imagination}** — the tell is encoder-*state* rel-err 7e-4→1.67e-2,
  and the only state-producing module in "heads" is the **readout**. **Deploy rule:
  INT8 the ViT encoder + predictor weights (saves 228+324 = 552 MB of 825 MB at 4×),
  keep the readout ≥fp16.** Source: `Implementation/int8_quant/int8_quant_step6500.json`.
- **The "ViT INT8 trap" is an ACTIVATION-quant rule, not a weight-quant rule.** The
  KB/literature heuristic "keep the vision tower FP16, quantize heads first" is REFUTED
  for weight-only quant here — the ViT *weights* tolerate INT8 best (98.4 %) and the
  *heads* are sensitive. The ViT trap is about heavy-tailed activation outliers under
  INT8 activation-scaling; ViT **weights** round cleanly. Score per-module, don't assume
  the heuristic. Source: same run + `Research/2026-07-10-...md` §1.
- **Clean-GPU latency (P1.4b closed):** idle/exclusive 4060, batch 1 — fp32 decision
  tick **15.76 ms / 63.5 Hz / 1.10 GB**, fp16 **13.40 ms / 74.6 Hz / 1.18×**. Clean fp32
  ≈ the 15.07 ms 2026-07-08 baseline → confirms the 33.5 ms 2026-07-09 read was pure
  **contention** (falsifier fired as predicted). fp16 peak-VRAM (1.65 GB) still
  double-counts the resident fp32 reference — a clean per-precision VRAM delta needs a
  single-model-resident run. Source: `Implementation/int8_quant/int8_quant_step6500.json`.
- **INT8 latency is NOT claimable without fused int8 TRT kernels** (toolchain absent) —
  today's measurable INT8 efficiency delta is the 4× **weight-memory footprint** only.
  The eventual real engine must pass a pre-registered joint bar: fp16-act + int8-weight
  (encoder+predictor) ≥ 95 % agreement / ≤ ~4 cm wp-shift, readout ≥fp16 (the two ~5 %
  error terms may compound). Source: `Research/2026-07-10-...md` §1.
- **Silent short-episode drop in all three window datasets** (`_contract.py:120`,
  `toy_driving.py:131`, `comma2k19.py:278`): `range(t_max)` with `t_max ≤ 0` drops the
  episode with no counter/warn/log (`comma2k19` guards the negative-range but drops just
  as silently) → silent train-set shrink, or a `StopIteration` spin if all are too short.
  Fixed via fail-loud `build_window_index` (count + warn + `ValueError` on empty), parity
  preserved for valid episodes; 10 standalone tests. Source:
  `Implementation/incoming/2026-07-10-contract-windowing-failloud/`.

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
