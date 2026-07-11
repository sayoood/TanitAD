# Production & Optimization — Experiment Backlog

Prioritized roadmap (D-020 §3/§4). Each run: one module-cluster compliance review + ≥1 measured
optimization experiment. G-P2: accuracy delta next to every speed delta.

## P0 — first runs

1. ~~Batch-1 streaming latency baseline (I8)~~ **DONE 2026-07-08 (MVP loop):** decision tick
   15.07 ms p50 / 1.08 GB VRAM on 4060 fp32 (encode 9.38, predictor 6.14, K9 select 5.69 —
   batching amortizes the select almost entirely). ~66 Hz un-optimized. Script:
   `stack/scripts/latency_cnce_baseline.py`; JSON in `stack/experiments/p0-latency-baseline/`.
   Note: nvidia-smi power read N/A on this 4060 — wattage needs another probe (HWiNFO/GPU-Z or
   powercfg) for the CNCE watt variant.
2. ~~**ONNX export of encoder+predictor path**~~ **DONE 2026-07-08:** encoder+readout & predictor
   export clean at opset 17 (legacy) AND opset 18 (dynamo, torch-2.11 default); parity vs PyTorch
   max|Δz| **8.8e-6 / 1.2e-5** (tol 1e-4); **no unexportable ops** (MHA/FiLM/causal-triu all fine —
   falsifier did not fire). ORT-CPU 1.4–4.4× slower than Torch-CPU (ONNX value = TRT IR, not CPU
   speed). Script+parity: `Implementation/onnx_export/`. Gotcha: Windows dynamo export needs
   `PYTHONUTF8=1`.
3. ~~**Compliance review #1: `stack/tanitad/data/`**~~ **DONE 2026-07-08:** intake
   `2026-07-08-data-cluster-compliance` — cache-key collision (`epcache.py`, cosmos chunk-pairing
   class, proven live) + `save_episode` fail-fast; 12 tests. 2 lower-prio findings logged.

## P1

4. ~~**TensorRT fp16 engine from the ONNX path**~~ **PART-DONE / toolchain-blocked 2026-07-09.**
   `import tensorrt`→ModuleNotFoundError, ORT CPU-only → engine build not runnable on the dev box.
   Ran the **honest precursor** instead (fp16/bf16 casts the same weights a TRT-fp16 engine would):
   **fp16 decision-safe** (imagine-select agreement 95.3 %, encoder rel-err 7.8e-4, decoded-waypoint
   shift 3.9 cm mean), **bf16 NOT** (67.2 %, 7.2e-3, 47.7 cm mean/3.58 m max) on 64 real windows.
   Deploy fp16; acceptance bar for the eventual engine pre-registered. Split into 4a/4b:
   - **P1.4a — install the TRT toolchain** (`tensorrt` + `onnxruntime-gpu` CUDA-12 EP on the dev box,
     OR build the engine on the pod when a trainer is idle), then build TRT-fp16 from the exported
     ONNX and verify it matches the fp16 bar (agreement ≥95 %, wp-shift ≤~4 cm). Non-paid, EXECUTE-
     class; cross-check with Tools&DevEnv. Falsifier: a TRT-unsupported op ⇒ document, don't hack.
   - ~~**P1.4b — clean latency re-measure**~~ **DONE 2026-07-10** (idle/exclusive 4060): fp32
     **15.76 ms / 63.5 Hz / 1.10 GB**, fp16 **13.40 ms / 74.6 Hz / 1.18×**. Clean fp32 ≈ the
     15.07 ms baseline → the 33.5 ms 07-09 read was contention (confirmed). Remaining: **per-precision
     VRAM** is still not clean (fp16's 1.65 GB double-counts the resident fp32 reference) → one
     single-model-resident re-measure (small, folded into P1.4a).
   - **P1.4a is now the clear next-run top** and the acceptance bar is a **joint** one: INT8 weights
     on encoder+predictor + fp16 activations + readout ≥fp16 must hold ≥95 % agreement / ≤~4 cm
     wp-shift (the fp16-act and int8-weight ~5 % terms may compound). Fallback: INT8 encoder only
     (98.4 %, −228 MB).
5. ~~**Compliance review #2: `stack/tanitad/models/`**~~ **DONE 2026-07-09:** intake
   `2026-07-09-models-predictor-failfast` — operative-predictor `assert`-only guard (`predictor.py:73`,
   stripped under `-O` → silent wrong-window inference) → `-O`-proof `ValueError`s, export-safe, 8 tests.
   Logged: same guard on `tactical_pred`; `imagination_nll` unclamped `exp(-logvar)` overflow.
6. ~~**INT8/FP8 quantization curves**~~ **DONE 2026-07-10:** per-module int8 **weight-only** curve
   scored in decision space. **ViT-encoder 98.4 % / predictor 95.3 % = SAFE; heads/all 48.4 %
   (1.67 m) = UNSAFE → sensitivity localizes to the READOUT** (not the ViT — the "quantize heads
   first" heuristic is REFUTED for weight-quant). Deploy: INT8 encoder+predictor weights (−552 MB
   at 4×), readout ≥fp16. INT8 *latency* deferred to the TRT engine (P1.4a). Intake N/A (experiment,
   not a stack change). `Implementation/int8_quant/`.
7. ~~**`tactical_pred` fail-fast + `imagination_nll` logvar clamp**~~ **DONE 2026-07-11 (review
   #3-cont):** intake `2026-07-11-imagination-nll-logvar-clamp` — `imagination_nll` `exp(-logvar)`
   overflow is a **measured, reachable silent-NaN** (fp16 −11.09 / fp32 −88.72 boundary; 45 SGD
   steps; gradient 2.8e34@−80) → clamp logvar to [−8,8], in-band parity **0.0**, + seeded `d9_rows`,
   10 standalone tests. `tactical_pred` fail-fast **subsumed** (it is an `OperativePredictor`,
   `fourbrain.py:49`, already covered by the review-#2 intake). `Implementation/imagination_nll_overflow/`
   (experiment) + `Implementation/incoming/2026-07-11-imagination-nll-logvar-clamp/` (fix).
7b. **Trainer `loss` finiteness guard (P1.8, next-run top)** — even with the clamp, add a
   nan/inf check on `loss` before `backward`/`opt.step` in `train_worldmodel.py` (skip-step + loud
   log, never silently corrupt the atomic checkpoint). Small intake; falsifier = inject a NaN loss,
   assert the step is skipped and the ckpt stays finite. Generalise: grep the stack for
   `exp`/`1/x`/`log` fed by unbounded heads (deploy-time NaN class).
8. ~~**Window-index silent short-episode drop**~~ **DONE 2026-07-10 (review #3):** intake
   `2026-07-10-contract-windowing-failloud` — count + warn + `ValueError`-on-empty across
   `_contract.py:120` / `toy_driving.py:131` / `comma2k19.py:278`, parity preserved, 10 standalone
   tests. (Was the folded-in sub-finding of the scripts review.)
8b. **Compliance review #4: `stack/scripts/` + training loop** — resume paths, atomic writes,
   log hygiene, cgroup awareness; the ops-fragility class (F-5/F-6/F-7, duplicate-trainer). Fold in
   the `epcache` DONE-marker cleanup (written-never-read) from review #1's logged findings.

## P2

9. **DSSAD/ISMR logging scaffold** — map self-monitoring signals (imagination error, OOD rows)
   to the UN-regulation event-log format (with Benchmarks & Eval REGULATION_TRACE).
10. **Memory-envelope profile** — batch-1 peak RSS/VRAM over a 10-min stream; leak check
    (tracemalloc + torch allocator stats deltas).
11. **Dependency hygiene** — pin-audit of stack/ imports, minimal runtime set for an
    inference-only wheel; **keep onnx/onnxruntime/onnxscript OUT** (export-time only, added
    2026-07-08).

## Done / retired
- (2026-07-11 run #4) P1.7 imagination-NLL logvar clamp DONE (measured silent-NaN overflow +
  reachability + fix, 10 tests) — GPU was 100 %-contended so the measured experiment was CPU/$0.
  Next-run top: **P1.8** trainer `loss` finiteness guard (7b). `tactical_pred` fail-fast subsumed.
- (2026-07-10 run #3) P1.6 INT8 weight-quant curve DONE (localizes to the readout; ViT/predictor
  INT8-safe, heads unsafe) + P1.4b clean-GPU latency DONE (15.76/13.40 ms fp32/fp16; contention
  caveat confirmed) + review #3 windowing fail-loud intake (10 tests). Next-run top: **P1.4a** mixed
  INT8+fp16 engine against the joint bar. New: 8b scripts/training-loop review #4.
- (2026-07-09 run #2) P1.4 → precursor measured (fp16 safe / bf16 unsafe), engine build split to
  P1.4a (toolchain install) + P1.4b (clean-GPU latency); P1.5 models review #2 DONE (fail-fast intake,
  8 tests). Next-run top: P1.4a (idle-GPU/pod TRT build) or review #3 (scripts) if GPU stays contended.
- (2026-07-08 run #1) P0.2 ONNX export DONE (parity ≤1.2e-5, both exporters); P0.3 data-cluster
  compliance review DONE (intake pkg, 12 tests). P1.4 TensorRT fp16 promoted to next-run top.
- (2026-07-08) Stream created (D-020 §3): agent file + Saturday schedule + this backlog.
