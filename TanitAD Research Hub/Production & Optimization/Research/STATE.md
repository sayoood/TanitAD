# Production & Optimization — STATE

- **LAST_RUN:** 2026-07-17 (run #3, Saturday agent) — branch `worktree-prod-opt-20260717`.
- **QUALITY:** full (one measured experiment with decision-grade clean-GPU numbers G-H/G-P2 +
  one compliance-review intake with 17 tests G-P1; readiness = validated). Loop 1 iter, $0, 0 searches.
- **Phase:** 0

## Where this stream stands (one paragraph)

Run #3. **P1.4b clean-GPU latency is DONE** (exclusive 4060, no CarlaUE4): fp32 decision tick
**14.79 ms / 67.6 Hz / 1.102 GB** — reproduces the 15.07 ms baseline, so the 2026-07-09 33.5 ms
**was CarlaUE4 contention, not a regression**; fp16 **10.67 ms / 93.7 Hz / 1.39×** and decision-safe
(95.3 % agreement, 3.9 cm wp-shift); bf16 the *same* 1.39× but unsafe (67.2 %). ~68/94 Hz = 3.4–4.7×
headroom over 10–20 Hz before any TRT/quant. Key structural finding: **fp16's whole speedup is the
ViT encoder** (8.98→4.69 ms ≈1.9×); the predictor/K9-select passes are batch-1 latency-floored →
**the latency lever is encoder precision**, and the planned INT8/FP8 (P1.6) is a VRAM/energy play
not a batch-1-latency play. **Compliance review #3** (`tanitad/models/imagination.py`) shipped intake
`2026-07-17-imagination-logvar-clamp` (17 tests): the H15 `imagination_nll` had an unclamped
`exp(-logvar)` that overflows fp32 to inf → NaN grads — LIVE in the flagship trainer
(`train_worldmodel.py:338`) + the OKRI/LOPS uncertainty export (`replay/arms.py:284`); fixed by
clamping logvar to [-10,10] at the head + in the nll (behaviour-preserving in-range). TensorRT-proper
still **toolchain-blocked** (`import tensorrt`→missing, ORT CPU EP only) — P1.4a stands.

## Next actions (checkboxes)

- [ ] **Next run — P1.4a:** build the TRT-fp16 engine once the toolchain is in (dev-box
      `tensorrt`+`onnxruntime-gpu`, or on the pod when a trainer is idle); verify against the fp16
      bar (agreement ≥95 %, wp-shift ≤~4 cm). If GPU stays contended, do review #3 (`stack/scripts/`)
      instead and defer P1.4a to a clean-GPU window.
- [ ] **Review #3 `stack/scripts/` + training loop** (ops-fragility F-5/6/7): resume/atomic-write/log
      hygiene/cgroup — timely given the pod-monitor stale-target + dead-trainer-relaunch history.
- [ ] **P1.4c:** one-process-per-precision VRAM harness (the fp16/bf16 VRAM in P1.4b is co-resident-
      inflated by the fp32 reference model → clean fp16 standalone footprint still owed).
- [ ] **`tactical_pred` fail-fast** + chase the unmerged `2026-07-09-models-predictor-failfast`
      (the `assert w==window` is still live at `predictor.py:89`). Imagination clamp = DONE this run.

## Standing facts / gotchas (this stream)

- RTX 4060 is the declared Orin latency proxy (I8). **Clean decision-tick (2026-07-17, exclusive):
  fp32 14.79 ms/67.6 Hz/1.10 GB, fp16 10.67 ms/93.7 Hz/1.39×** — reproduces the 15.07 ms 2026-07-08
  baseline. **Absolute latency needs an exclusive GPU** (the 2026-07-09 33.5 ms was CarlaUE4
  contention). Clocks are NOT admin-pinnable on this box → use p50/p95 over ≥100 reps. **fp16's
  speedup is ALL the ViT encoder** (≈1.9×); predictor/select are batch-1 latency-floored.
- **Precision policy: fp16 on the decision path, never bf16** (measured 2026-07-09, reproduced to the
  digit 2026-07-17). Keep the ViT tower ≥fp16. TRT-fp16 acceptance bar pre-registered (≥95 %
  agreement, ≤~4 cm wp-shift on 64 windows). bf16 = same 1.39× speed but flips 1/3 of maneuvers.
- **TensorRT not installed on the dev box** (`import tensorrt`→missing; ORT CPU EP only). The ONNX
  IR is exported + parity-clean, so only the toolchain/engine step remains.
- ONNX export deps (`onnx`, `onnxruntime`, `onnxscript`) installed in the venv — **export/dev only**,
  never in the inference runtime wheel.
- Windows dynamo ONNX export crashes on emoji progress under cp1252 → run with `PYTHONUTF8=1`.
- Boundary: NEVER write `stack/` directly. Experiments live in `Implementation/onnx_export/` (off-
  Drive for large `.onnx`); stack-changing fixes go through `Implementation/incoming/` intake.

## HANDOFF

- **To Benchmarks & Eval (efficiency ledger row):** `LEADERBOARD.md` is not on this worktree's base
  (origin/main; it lives on unmerged/local-main work — the D-026 stale-base debt). Add the clean-GPU
  efficiency row when merging: **fp32 14.79 ms/67.6 Hz/1.10 GB, fp16 10.67 ms/93.7 Hz/1.39× (safe),
  bf16 unsafe**, params 0.263 B, CNCE inputs = these + real log progress. Numbers are in
  `Research/2026-07-17-...md` and KB. (Run #2's LEADERBOARD write is the precedent.)
- **To MVP orchestrator:** two models intakes now pending — `2026-07-17-imagination-logvar-clamp`
  (this run, 17 tests, live flagship NaN mode) and the still-unmerged `2026-07-09-models-predictor-failfast`
  (the `assert w==window` is live at `predictor.py:89`). Both small + export-safe.
- Run completed cleanly; all artifacts committed on `worktree-prod-opt-20260717`.
