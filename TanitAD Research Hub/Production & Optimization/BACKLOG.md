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

3b. **ZipDepth evaluation as independent safety-envelope channel (Sayed-delivered 2026-07-11,
   D-028 seam: ours; SECOND-STEP per Sayed — not Phase 0).** 6.1M params, 77 FPS TRT-fp16 on
   Orin NX, open weights, affine-invariant (no metric scale). When picked up: (a) run on 4060 as
   Orin proxy, measure latency + VRAM beside our 15 ms tick (do both fit an Orin budget
   together?); (b) qualitative artifact assessment on our val routes vs Depth-Anything-v2
   (Sayed flagged quality gaps — quantify before any safety-envelope claim); (c) metric-scale
   plan (camera height + ground plane vs small metric adapter). Also: pseudo-depth teacher
   candidate for Y-track (scale-free — no metric problem there).
   See `../2026-07-11-sayed-papers-screening.md`.

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
   - **P1.4b — clean latency re-measure** on an EXCLUSIVE, clock-pinned 4060 (`nvidia-smi -lgc`, no
     CarlaUE4/python resident) to publish the fp16 absolute Hz + a clean fp16-vs-fp32 VRAM delta
     (this run's absolutes were contended: fp32 tick 33.5 ms vs the clean 15.07 ms baseline).
5. ~~**Compliance review #2: `stack/tanitad/models/`**~~ **DONE 2026-07-09:** intake
   `2026-07-09-models-predictor-failfast` — operative-predictor `assert`-only guard (`predictor.py:73`,
   stripped under `-O` → silent wrong-window inference) → `-O`-proof `ValueError`s, export-safe, 8 tests.
   Logged: same guard on `tactical_pred`; `imagination_nll` unclamped `exp(-logvar)` overflow.
6. **INT8/FP8 quantization curves** — accuracy-vs-latency per module (encoder most sensitive);
   probe-fit delta / imagine-select agreement as the accuracy metric (the 2026-07-09 fp16/bf16
   result is the precedent: score precision in the DECISION space, not just latent cosine).
7. **`tactical_pred` fail-fast + `imagination_nll` logvar clamp** — the two findings logged in
   review #2 (same `assert`-only guard on the tactical predictor; unclamped `exp(-logvar)` overflow
   in `imagination.py:135` → NaN LOPS/H2 trigger). Small numerics-hardening package with a falsifier.
8. **Compliance review #3: `stack/scripts/` + training loop** — resume paths, atomic writes,
   log hygiene, cgroup awareness; the ops-fragility class (F-5/F-6/F-7, duplicate-trainer). Fold in
   the `epcache` DONE-marker cleanup (written-never-read) + `EpisodeWindowDataset` silent
   short-episode drop (add a dropped-window counter/log) from review #1's logged findings.

## P2

9. **DSSAD/ISMR logging scaffold** — map self-monitoring signals (imagination error, OOD rows)
   to the UN-regulation event-log format (with Benchmarks & Eval REGULATION_TRACE).
10. **Memory-envelope profile** — batch-1 peak RSS/VRAM over a 10-min stream; leak check
    (tracemalloc + torch allocator stats deltas).
11. **Dependency hygiene** — pin-audit of stack/ imports, minimal runtime set for an
    inference-only wheel; **keep onnx/onnxruntime/onnxscript OUT** (export-time only, added
    2026-07-08).

## Done / retired
- (2026-07-09 run #2) P1.4 → precursor measured (fp16 safe / bf16 unsafe), engine build split to
  P1.4a (toolchain install) + P1.4b (clean-GPU latency); P1.5 models review #2 DONE (fail-fast intake,
  8 tests). Next-run top: P1.4a (idle-GPU/pod TRT build) or review #3 (scripts) if GPU stays contended.
- (2026-07-08 run #1) P0.2 ONNX export DONE (parity ≤1.2e-5, both exporters); P0.3 data-cluster
  compliance review DONE (intake pkg, 12 tests). P1.4 TensorRT fp16 promoted to next-run top.
- (2026-07-08) Stream created (D-020 §3): agent file + Saturday schedule + this backlog.
