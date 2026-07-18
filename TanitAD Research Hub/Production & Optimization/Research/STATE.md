# Production & Optimization — STATE

- **LAST_RUN:** 2026-07-18 (run #4, Saturday agent) — branch `agent/prod-opt-20260718`
  (off HEAD 327a174; off-Drive worktree `C:/Users/Admin/wt-prod`). **RESOURCE (G-I):** local
  **RTX 4060** only, exclusive, ~25 min, **$0**. Why not the eval-pod A40: batch-1 single-stream
  latency = the Orin-proxy 4060 is the correct instrument (A40 measures throughput); the A40 TRT
  job stays toolchain-blocked (job card shipped). Pods all training → no idle-pod window.
- **QUALITY:** full (2 measured experiments G-H/G-P2 on real compute; one compliance intake with
  11-test regression guard G-P1; readiness = validated). Loop 1 iter, 0 web searches.
- **Phase:** 0

## Where this stream stands (one paragraph)

Run #4. **Predictor batch-1 latency ATTACKED (P0 #3) — manual CUDA-graph capture is a FREE win.**
On the exclusive 4060 (fp32, tf32 off, step-6500, 64 real comma windows, 200 reps), capturing the
operative predictor pass as a `torch.cuda.CUDAGraph`: predict_1pass **6.08 → 2.36 ms (2.57×)**,
select_K9 **5.94 → 4.45 ms (1.33×)**; accuracy vs eager rel-err_max **2.8e-7**, cosine 1.0,
imagine-and-select **agreement 100 %**, waypoint shift **0.00 m** → the graph replays the same fp32
kernels, so it is pure kernel-launch elimination. **Falsifier (>10 %) cleared 25×** → run #3's
launch-bound diagnosis CONFIRMED. Gain scales inversely with batch (predict-1 2.57× ≫ K9 1.33×) →
**two orthogonal levers: encoder=compute-bound→fp16, predictor=launch-bound→CUDA graph** (additive
tick projection ~9.1 ms / ~109 Hz). Deployment finding: on this Triton-less Windows box the graph
route is **manual `torch.cuda.CUDAGraph`, NOT `torch.compile`** — inductor → `TritonMissing`,
dynamo-cudagraphs → 20× SLOWER (117 ms). **Compliance review (P0 #4): numerics-safety class is
CLOSED** — the grep-sweep found every learned/data `exp`/`log`/div guarded (clamp / count-gate /
neg-exponent), no new site; shipped an **11-test executable regression guard** (intake
`2026-07-18-numerics-safety-sweep`, test-only, all green) with real failing-then-passing witnesses.
TRT-fp16 engine (P0 #1) still **toolchain-blocked** on the dev box → job card in the note.

## Next actions (checkboxes)

- [ ] **Next run — combined harness (A3):** full-tick single CUDA graph (encode+predict+select) +
      **fp16 encoder + graph predictor combined** latency/VRAM harness → replace the additive ~9 ms
      projection with a measured combined tick; fold in the P1.4c one-process VRAM row alongside.
- [ ] **P0 #1 TRT-fp16 engine (job card in the note):** run when a pod is idle or `tensorrt`+
      `onnxruntime-gpu` land on the dev box; verify vs the fp16 bar (agreement ≥95 %, wp-shift ≤~4 cm),
      report Hz+VRAM on A40 (server) AND 4060 (deploy).
- [ ] **Tools&DevEnv ask (A2):** install a Windows **Triton** wheel (torch 2.11+cu128) if we want
      `torch.compile` on the dev box; else the Orin runtime uses hand-rolled CUDA-graph capture.
- [ ] **Review #3 `stack/scripts/` + training loop** (ops-fragility F-5/6/7): resume/atomic-write/log
      hygiene/cgroup — still owed; timely given the pod-monitor stale-target history.
- [ ] **`tactical_pred` fail-fast** + chase the unmerged `2026-07-09-models-predictor-failfast`
      (the `assert w==window` is still live at `predictor.py:89`).

## Standing facts / gotchas (this stream)

- RTX 4060 is the declared Orin latency proxy (I8). **Clean decision-tick (2026-07-17, exclusive):
  fp32 14.79 ms/67.6 Hz/1.10 GB, fp16 10.67 ms/93.7 Hz/1.39×** — reproduces the 15.07 ms 2026-07-08
  baseline. **Absolute latency needs an exclusive GPU** (the 2026-07-09 33.5 ms was CarlaUE4
  contention). Clocks are NOT admin-pinnable on this box → use p50/p95 over ≥100 reps. **fp16's
  speedup is ALL the ViT encoder** (≈1.9×); predictor/select are batch-1 latency-floored.
- **Precision policy: fp16 on the decision path, never bf16** (measured 2026-07-09, reproduced to the
  digit 2026-07-17). Keep the ViT tower ≥fp16. TRT-fp16 acceptance bar pre-registered (≥95 %
  agreement, ≤~4 cm wp-shift on 64 windows). bf16 = same 1.39× speed but flips 1/3 of maneuvers.
- **Predictor half is launch-bound, not compute-bound (run #4).** fp16 barely moves it; manual
  `torch.cuda.CUDAGraph` capture gives 2.57× (predict-1) / 1.33× (K9) FREE (rel-err 2.8e-7, 100 %
  agreement). Two orthogonal levers: encoder→fp16, predictor→CUDA graph. **`torch.compile` is NOT
  the route on this box** (Triton missing → inductor fails; dynamo-cudagraphs 20× slower) — use
  hand-rolled capture. Static-IO pattern in `Implementation/predictor_latency/`.
- **TensorRT not installed on the dev box** (`import tensorrt`→missing; ORT CPU EP only). The ONNX
  IR is exported + parity-clean, so only the toolchain/engine step remains.
- ONNX export deps (`onnx`, `onnxruntime`, `onnxscript`) installed in the venv — **export/dev only**,
  never in the inference runtime wheel.
- Windows dynamo ONNX export crashes on emoji progress under cp1252 → run with `PYTHONUTF8=1`.
- Boundary: NEVER write `stack/` directly. Experiments live in `Implementation/onnx_export/` (off-
  Drive for large `.onnx`); stack-changing fixes go through `Implementation/incoming/` intake.

## HANDOFF

- **To Benchmarks & Eval (efficiency ledger / CNCE):** add the predictor-graph efficiency row —
  **operative predictor 6.08→2.36 ms (2.57×), select_K9 5.94→4.45 ms (1.33×), rel-err 2.8e-7,
  agreement 100 %, $0, 4060**; lower tick → higher CNCE at fixed params (0.263 B). Also still owed
  from run #3: **fp32 14.79 ms/67.6 Hz, fp16 10.67 ms/93.7 Hz/1.39× (safe), bf16 unsafe**. Numbers
  in `Research/2026-07-18-...md` + `2026-07-17-...md` + KB.
- **To MVP orchestrator (3 pending models/test intakes):** `2026-07-18-numerics-safety-sweep`
  (this run, 11-test regression guard, test-only → `stack/tests/`); `2026-07-17-imagination-logvar-clamp`
  (17 tests; the clamp is in mainline per e753a00 — confirm the intake verdict/close it);
  `2026-07-09-models-predictor-failfast` (still unmerged — the `assert w==window` is live at
  `predictor.py:89`). All small + export-safe.
- **To Tools&DevEnv:** install a Windows Triton wheel (torch 2.11+cu128) to unlock `torch.compile`
  on the dev box; and `tensorrt`+`onnxruntime-gpu` for the TRT engine build (P0 #1 job card).
- Run completed cleanly; all artifacts committed on `agent/prod-opt-20260718` (off-Drive worktree).
