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
2. **ONNX export of encoder+predictor path** — opset pin, shape `[1,9,256,256]` static; parity
   check vs PyTorch (max |Δz| under pinned numerics ≤ 1e-4 fp32). Deliverable: export script +
   parity numbers + list of unexportable ops (grad-checkpoint flags off for export).
   Falsifier: FiLM/attention op unsupported ⇒ document plugin need, do NOT hack the model.
3. **Compliance review #1: `stack/tanitad/data/`** (epcache, mixing, contract, loaders) — the
   most failure-prone cluster (F-3/F-4/F-6/F-7 history). Checklist per PRODUCTION_READINESS.md;
   findings as one intake package with failing-then-passing tests.

## P1

4. **TensorRT engine build from the ONNX path** — fp16 first (INT8 via OwLite/ModelOpt later —
   native-TRT ViT INT8 is a known trap); latency vs PyTorch on 4060 with accuracy delta on 100
   held-out windows.
5. **Compliance review #2: `stack/tanitad/models/`** — numerics discipline (fp32 islands,
   autocast boundaries), determinism/seeds, error handling on malformed windows.
6. **INT8/FP8 quantization curves** — accuracy-vs-latency per module (encoder most sensitive);
   probe-fit delta as the accuracy metric (cheap, decision-relevant).
7. **Compliance review #3: `stack/scripts/` + training loop** — resume paths, atomic writes,
   log hygiene, cgroup awareness; the ops-fragility class (F-5/F-6/F-7, duplicate-trainer).

## P2

8. **DSSAD/ISMR logging scaffold** — map self-monitoring signals (imagination error, OOD rows)
   to the UN-regulation event-log format (with Benchmarks & Eval REGULATION_TRACE).
9. **Memory-envelope profile** — batch-1 peak RSS/VRAM over a 10-min stream; leak check
   (tracemalloc + torch allocator stats deltas).
10. **Dependency hygiene** — pin-audit of stack/ imports, minimal runtime set for an
    inference-only wheel.

## Done / retired
- (2026-07-08) Stream created (D-020 §3): agent file + Saturday schedule + this backlog.
