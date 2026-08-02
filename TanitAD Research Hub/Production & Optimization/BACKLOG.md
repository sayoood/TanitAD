# Production & Optimization — Experiment Backlog

Prioritized roadmap (D-020 §3/§4). Each run: one module-cluster compliance review + ≥1 measured
optimization experiment. G-P2: accuracy delta next to every speed delta.

## ⭐ P0 — THOR DIRECTIVE 2026-08-02 (supersedes the 2026-07-17 ordering below)

Thor is deployed and the orchestrator handed this stream the O1–O14 backlog
(`THOR_DEPLOYMENT_RUNBOOK.md` §6). **The ordered, falsifiable execution plan is
`THOR_OPTIMISATION_PLAN.md`** — read it first; this list is its index.

| item | plan ref | state | one-line why |
|---|---|---|---|
| **A1** ONNX export-parity re-verify @torch 2.11 + permanent export guard | §3 Tier A | 🔵 **next, 0 deps** | the MHA fastpath bug retracts our own 2026-07-08 claim on 2.13; **unverified on 2.11**, where the claim was made. 4060, ~90 min, $0 |
| **A2** compliance review #4 — `taniteval/` + `tanitad/eval/` | §3 Tier A | 🔵 next | the only unreviewed cluster that **decides every deployment gate**, and the four families are now binding. C63 class |
| **A3** arm the Thor job cards | §3 Tier A | 🔵 next | power-on → results with no authoring latency |
| **B1** ⭐ **complete decision tick** (absorbs O3 + **O9**) | §3 Tier B | ⛔ Thor offline | ⭐⭐ the measured tick rolls **1** candidate; the deployed selector loops over **9** (`config.py:95`, `fourbrain.py:571`). Serialised ⇒ ~238 ms (2.4× over budget); batched ⇒ ~53 ms. **The engine is built at batch 1 and needs a batch-9/dynamic profile.** Largest open uncertainty in the deployment |
| **B0** val-window cache built ON Thor from HF | §3 Tier B | ⛔ Thor offline | O1 silently carries a pod dependency; this removes it (880 GB free, public HF weights). Parity-sha verified or refuse |
| **B2** ⭐ **bf16-vs-fp16 encoder DECISION-agreement** | §3 Tier B | ⛔ Thor offline | the 6.76× lever rests on rel-err 0.0059 on **random weights**, at a precision our own decision-space measurement **rejected** (bf16 → 67.2 % agreement, 47.7 cm). Fails ⇒ O2 becomes P0 and the 51.2 ms is void |
| **B3** = **O1** four-family gate on the TRT-fp16 engine | §3 Tier B | ⛔ needs B0 + B2's instrument check | paired episode-cluster bootstrap, per-family, never pooled |
| **O2** TRT encoder engine | §3 Tier C | P0-if-B2-fails, else P1 | the fallback that saves the encoder win |
| **O7** `nvpmodel` power modes | §3 Tier C | P1, parallel-safe | ~2 W / 61.9 °C — envelope unexplored, no accuracy dependency |
| **O5** INT8 PTQ | §3 Tier C | P1, gated on B3 | unreadable without a working four-family gate |
| **O4** K 20→10 | §3 Tier C | **DEMOTED** | ⛔ the runbook's "−23 ms at TRT speed" is **wrong ~2×**: the ksweep is *eager* (4.1–4.3 ms/step); at TRT-fp16 the cut saves **11.7 ms**. 23.36 ms is the whole K20 TRT roll |
| **O8** one engine for the whole roll | §3 Tier C | ⛔ **CLOSED** | measured **1.02×** vs per-step — its own falsifier (<1.1×) fires |
| **O6** NVFP4 · **O14** 2:4 sparsity | §3 Tier C | P2, re-price after B1 | our tensors sit on the wrong end of the size curve (FP8: 1.21× at 8×2048 vs 1.97× at 4096³) — but a batch-9 fan is a bigger GEMM |
| **O10–O13** resolution / multi-cam / Orin / DLA | §3 Tier C | P2 | downstream of a known tick + a working gate |
| **P1.4a** local TRT toolchain (below) | §3 Tier C | **DOWNGRADED** | Thor *is* the TRT box now; the dev-box install left the critical path |

## P0 — FLEET DIRECTIVE 2026-07-17 (Sayed; supersedes prior P0 ordering; resource-mandated G-I)

Context: `Project Steering/FLEET_REVIEW_2026-07-17.md`. Review verdict: your fp16 latency (A),
logvar-clamp (A — APPLIED 2026-07-17 to the live pod2 stack + repo mainline; the running process
resumes on the patched build at its next restart), parity test (A). Your fp16 diagnosis (the whole
win is the ViT; predictor is launch-bound) re-scopes this backlog.

0. ~~**Combined + full-tick CUDA-graph harness (A3)**~~ **DONE 2026-07-18 (run #5):** measured
   combined tick **fp16 encoder + CUDA-graph predictor = 17.75 → 11.16 ms, 89.6 Hz, 1.59×**,
   agreement **96.9 %** (2 flips/64, the fp16 encoder's — the graph adds 0), wp-shift 0.7 cm.
   Measured == run #4's additive projection to **0.4 %** → levers compose (falsifier <1.3× cleared:
   got 1.59×). Graph is also clock-robust (fixed ~4.4 ms replay). `Implementation/combined_tick/`.
1. **TRT-fp16 engine for flagship@30k — BOTH targets (JOB CARD in the 2026-07-18 run #4 note) — NOW
   THE TOP LATENCY ITEM:** eval pod A40 (throughput/server row) AND the 4060 (Orin-proxy deployment
   row). Report Hz + VRAM + decision-agreement vs fp32 (95.3% bar) AND vs the CUDA-graph baseline
   (11.16 ms — the engine must beat the free graph to be worth the toolchain). Toolchain-blocked on
   the dev box (`tensorrt` missing) → run when a pod is idle or `tensorrt`+`onnxruntime-gpu` land.
   ONNX IR already parity-clean.
2. ~~**P1.4c VRAM-isolation harness fix**~~ **DONE 2026-07-18 (run #5):** clean one-process standalone
   VRAM fp32 **1.078 GB** (reproduces run #3's 1.10 GB), fp16 **0.560 GB** (1.93× smaller). The
   run #3/#4 fp16 1.65 GB was fp32-reference co-residency — closed. `Implementation/combined_tick/
   vram_*.json`. P1.6 quant stays a VRAM/energy play (not batch-1 latency).
3. ~~**Predictor batch-1 latency attack**~~ **DONE 2026-07-18 (run #4):** manual `torch.cuda.CUDAGraph`
   = **2.57×** (predict-1) / **1.33×** (K9), rel-err **2.8e-7**, agreement **100 %**, wp-shift 0.00 m.
   Falsifier (>10%) cleared 25× → launch-bound CONFIRMED. `torch.compile` NOT viable here (Triton
   missing → inductor fails; dynamo-cudagraphs 20× slower) → deploy via manual capture. Both-GPU
   ask deferred: the A40 comparison rides item 1's engine build (single-stream = 4060 is the right
   proxy). `Implementation/predictor_latency/`.
4. ~~**Ops hardening — numerics grep-sweep**~~ **DONE 2026-07-18 (run #4):** the class is CLOSED —
   every learned/data `exp`/`log`/`div` in `stack/tanitad` is guarded (clamp / count-gate /
   neg-exponent); no new site. Shipped an **11-test executable regression guard** (intake
   `2026-07-18-numerics-safety-sweep`, test-only, all green) that keeps it closed on merges.

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
   - ~~**P1.4b — clean latency re-measure**~~ **DONE 2026-07-17** (exclusive 4060): fp32
     **14.79 ms/67.6 Hz/1.10 GB** (reproduces 15.07 ms → 33.5 ms was contention); fp16
     **10.67 ms/93.7 Hz/1.39×** (safe 95.3 %); bf16 same 1.39× unsafe. fp16's win is ALL
     encoder (8.98→4.69 ms); predictor/select batch-1 latency-floored. Clocks not
     admin-pinnable (p50/p95×100 mitigates). `half_precision_clean_20260717.json`.
   - **P1.4c (NEW) — one-process-per-precision VRAM harness.** The fp16/bf16 peak-VRAM in
     P1.4b is co-resident-inflated (accuracy harness keeps the fp32 reference alive;
     261 M×2 B ≈ the 0.52 GB gap). Measure each precision in a fresh process for a true
     fp16 standalone footprint. Local, $0, EXECUTE-class.
5. ~~**Compliance review #2: `stack/tanitad/models/`**~~ **DONE 2026-07-09:** intake
   `2026-07-09-models-predictor-failfast` — operative-predictor `assert`-only guard (`predictor.py:73`,
   stripped under `-O` → silent wrong-window inference) → `-O`-proof `ValueError`s, export-safe, 8 tests.
   Logged: same guard on `tactical_pred`; `imagination_nll` unclamped `exp(-logvar)` overflow.
6. **INT8/FP8 quantization curves** — accuracy-vs-latency per module. **Re-scoped 2026-07-17
   (P1.4b finding):** at batch-1 the predictor/heads are latency-floored — quantizing them buys
   ~0 tick; the encoder holds the time and must stay ≥fp16 (bf16 unsafe). So this is a **VRAM/
   energy** play, not a batch-1-latency play; measure VRAM + energy delta beside the accuracy
   delta (probe-fit / imagine-select agreement in DECISION space). ViT INT8 = the OwLite/ModelOpt
   trap; ViT tower FP16, quantize heads first.
7. **`tactical_pred` fail-fast** (+ merge the unmerged `2026-07-09-models-predictor-failfast`).
   The **`imagination_nll` logvar clamp is DONE** (2026-07-17, intake
   `2026-07-17-imagination-logvar-clamp`, 17 tests). Remaining: the operative/tactical predictor
   `assert w==window` guard is still live at `predictor.py:89` (the review-#2 intake never merged);
   re-flag or fold into the next models package. Small, no falsifier needed (fail-fast).
8. **Compliance review #3: `stack/scripts/` + training loop** — **PART-DONE 2026-07-18 (run #5):**
   found + fixed the LIVE **non-atomic milestone-archive** bug (`shutil.copy2` guarded by
   `not arch.exists()` in all 3 pod trainers → a kill mid-copy silently corrupts a gate milestone;
   intake `2026-07-18-atomic-milestone-archive`, 4 tests). **Resume-write path confirmed atomic in
   every trainer** (`tmp→.replace`). **Remaining (next run):** (a) log-hygiene — trainer logs to
   `/workspace` get swallowed on death vs `/tmp` (pod2 history); (b) a free-space/quota preflight
   before the archive copy (Errno122 class); (c) the `epcache` DONE-marker cleanup (written-never-read)
   + `EpisodeWindowDataset` silent short-episode drop (dropped-window counter) from review #1.

## P2

9. **DSSAD/ISMR logging scaffold** — map self-monitoring signals (imagination error, OOD rows)
   to the UN-regulation event-log format (with Benchmarks & Eval REGULATION_TRACE).
10. **Memory-envelope profile** — batch-1 peak RSS/VRAM over a 10-min stream; leak check
    (tracemalloc + torch allocator stats deltas).
11. **Dependency hygiene** — pin-audit of stack/ imports, minimal runtime set for an
    inference-only wheel; **keep onnx/onnxruntime/onnxscript OUT** (export-time only, added
    2026-07-08).

## Done / retired
- (2026-07-18 run #5) **Combined deploy-tick MEASURED** (fp16 enc + CUDA-graph pred = 11.16 ms/
  89.6 Hz/1.59×, agreement 96.9 %, = additive projection to 0.4 % → levers compose) + **P1.4c clean
  VRAM DONE** (fp32 1.08 GB / fp16 0.56 GB, co-residency artifact closed) + **review #3 PART-DONE**
  (live non-atomic milestone-archive bug fixed, intake `2026-07-18-atomic-milestone-archive`, 4 tests).
  **Next-run top:** P0 #1 TRT-fp16 engine (must beat the free 11.16 ms graph baseline) on an idle pod /
  TRT box; then review #3 continuation (log-hygiene / quota-preflight). $0, 4060.
- (2026-07-18 run #4) **Predictor launch-bound attack DONE** (manual CUDA-graph 2.57×/1.33×, free,
  rel-err 2.8e-7, 100% agreement; `torch.compile` non-viable on this Triton-less box) + **numerics
  grep-sweep DONE** (class closed, 11-test regression guard intake). New top: item 0 combined+
  full-tick graph harness (measured combined tick vs the additive ~9 ms projection). $0, 4060.
- (2026-07-17 run #3) **P1.4b clean-GPU latency DONE** (fp32 14.79 ms/67.6 Hz, fp16 1.39×/93.7 Hz,
  encoder-only win, precision policy reproduced); **review #3 imagination_nll logvar clamp DONE**
  (intake, 17 tests). New: P1.4c one-process VRAM harness; P1.6 re-scoped to VRAM/energy. **Next-run
  top:** P1.4a (TRT engine — install `tensorrt`+`onnxruntime-gpu` or idle-pod build) OR compliance
  review #3 `stack/scripts/`+training-loop (ops-fragility F-5/6/7; the pod-monitor stale-target +
  dead-trainer-relaunch history makes this timely) — pick review #3 if the GPU is contended.
- (2026-07-09 run #2) P1.4 → precursor measured (fp16 safe / bf16 unsafe), engine build split to
  P1.4a (toolchain install) + P1.4b (clean-GPU latency); P1.5 models review #2 DONE (fail-fast intake,
  8 tests). Next-run top: P1.4a (idle-GPU/pod TRT build) or review #3 (scripts) if GPU stays contended.
- (2026-07-08 run #1) P0.2 ONNX export DONE (parity ≤1.2e-5, both exporters); P0.3 data-cluster
  compliance review DONE (intake pkg, 12 tests). P1.4 TensorRT fp16 promoted to next-run top.
- (2026-07-08) Stream created (D-020 §3): agent file + Saturday schedule + this backlog.
