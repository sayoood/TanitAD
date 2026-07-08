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

4. **TensorRT fp16 engine from the ONNX path (NEXT RUN — unblocked)** — build on the 4060 from
   `encoder_readout.onnx` + `predictor.onnx`; latency vs PyTorch-GPU with accuracy delta (max|Δz|)
   on 100 held-out real windows. INT8 via ModelOpt later (native-TRT ViT INT8 is a known trap; keep
   ViT tower FP16). Falsifier: a TRT-unsupported op ⇒ document, don't hack.
5. **Compliance review #2: `stack/tanitad/models/`** — numerics discipline (fp32 islands,
   autocast boundaries), determinism/seeds, error handling on malformed windows.
6. **INT8/FP8 quantization curves** — accuracy-vs-latency per module (encoder most sensitive);
   probe-fit delta as the accuracy metric (cheap, decision-relevant).
7. **Compliance review #2: `stack/tanitad/models/`** — numerics discipline, determinism/seeds,
   error handling on malformed windows; carry the two logged data-review findings if not folded in.
   (was P1.5; renumbered — models is the natural review #2 given the export interest.)
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
- (2026-07-08 run #1) P0.2 ONNX export DONE (parity ≤1.2e-5, both exporters); P0.3 data-cluster
  compliance review DONE (intake pkg, 12 tests). P1.4 TensorRT fp16 promoted to next-run top.
- (2026-07-08) Stream created (D-020 §3): agent file + Saturday schedule + this backlog.
