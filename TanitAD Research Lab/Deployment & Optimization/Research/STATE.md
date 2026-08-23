# Production & Optimization — STATE

- **LAST_RUN:** 2026-07-18 (**run #5**, Saturday agent, follows run #4 09:17) — branch
  `agent/prod-opt-20260718` (off-Drive worktree `C:/Users/Admin/wt-prod`; **base 9c36c95 ← 327a174,
  BEHIND the shared tip fcbab02 which carries the milestone-archiving commits — flag for orchestrator
  merge, D-026**). **RESOURCE (G-I):** local **RTX 4060** only, exclusive, ~20 min, **$0**. Why not
  the eval-pod A40: batch-1 single-stream latency = the Orin-proxy 4060 is the correct instrument
  (A40 measures throughput); the A40 TRT job stays toolchain-blocked (run #4 job card). Pods all
  training → no idle-pod window.
- **QUALITY:** full (measured combined-tick + clean VRAM G-H/G-P2 on real compute; one compliance
  intake with a 4-test failing-then-passing guard G-P1; readiness = validated). Loop 1 iter, 0 searches.
- **Phase:** 0

## Where this stream stands (one paragraph)

Run #5. **The combined deploy tick is now MEASURED (A3), not projected.** fp16 encoder + CUDA-graph
predictor/select in one decision tick (4060, fp32 ref, step-6500, 64 real windows, 200 reps):
**17.75 → 11.16 ms, 56.3 → 89.6 Hz, 1.59×**, agreement **96.9 %** (2 flips/64), wp-shift **0.7 cm /
1.9 cm** — and the measured tick matches run #4's additive projection (11.21 ms) to **0.4 %** → the
two levers COMPOSE with no interference. Both are needed (fp16 alone 66.3 Hz; +graph 89.6 Hz); the
graph is **zero-accuracy-cost** (the 2 flips are the fp16 encoder's, identical without the graph) and
**clock-robust** (graphed select 1.92× here vs 1.33× run #4 — replay time is fixed ~4.4 ms while eager
scales with the non-pinnable clock). **P1.4c CLOSED:** clean one-process VRAM fp32 **1.078 GB** /
fp16 **0.560 GB** (1.93× smaller) — the run #3/#4 fp16 1.65 GB was co-residency pollution, never quote
it. **Compliance review #3 (`stack/scripts/` + training loop) found a LIVE bug:** the milestone
archive (`train_flagship4b.py:337`, `refb_train.py:358`, `refa_train_plus.py:540` — all 3 pod trainers)
copies non-atomically (`shutil.copy2` guarded by `not arch.exists()`); a kill mid-copy leaves a
truncated-but-existing `ckpt_step{m}.pt` that the guard adopts forever → the gate protocol loads a
corrupt milestone. Fix shipped: intake `2026-07-18-atomic-milestone-archive` (`.partial`→`os.replace`,
4 tests green, failing-then-passing). The atomic *resume* write is clean in every trainer; only the new
archive path was missed. TRT-fp16 engine (P0 #1) still toolchain-blocked → run #4 job card stands.

<details><summary>Run #4 (superseded header, kept for the record)</summary>

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

</details>

## Next actions (checkboxes)

- [x] **Combined harness (A3) — DONE run #5:** measured combined tick 11.16 ms/89.6 Hz (1.59×,
      agreement 96.9 %), matches the additive projection to 0.4 %; P1.4c clean VRAM fp16 0.56 GB.
- [x] **Review #3 `stack/scripts/` + training loop — DONE run #5:** live non-atomic milestone-archive
      bug → intake `2026-07-18-atomic-milestone-archive` (4 tests). Resume-write path confirmed clean.
- [ ] **P0 #1 TRT-fp16 engine (job card in run #4 note):** run when a pod is idle or `tensorrt`+
      `onnxruntime-gpu` land on the dev box; verify vs the fp16 bar (agreement ≥95 %, wp-shift ≤~4 cm),
      report Hz+VRAM on A40 (server) AND 4060 (deploy). Now the top remaining latency item.
- [ ] **Tools&DevEnv ask (A2):** install a Windows **Triton** wheel (torch 2.11+cu128) if we want
      `torch.compile` on the dev box; else the Orin runtime uses hand-rolled CUDA-graph capture.
- [ ] **Log-hygiene / cgroup pass (review #3 continuation):** the pod2 self-kill history (RAM/quota)
      warrants a sweep of the trainer log paths (`/workspace` vs `/tmp` swallow-on-death) + a
      free-space preflight before the archive copy — scope next run.
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
  hand-rolled capture. Static-IO pattern in `Implementation/predictor_latency/` + `combined_tick/`.
- **Deploy tick = fp16 encoder + CUDA-graph predictor, MEASURED (run #5): 11.16 ms / 89.6 Hz /
  1.59×**, agreement 96.9 %, wp-shift ≤1.9 cm. Levers compose (measured = additive projection to
  0.4 %); graph is zero-accuracy-cost + clock-robust (fixed ~4.4 ms replay). **Clean standalone VRAM
  (one process): fp32 1.078 GB, fp16 0.560 GB** — the 1.65 GB fp16 was co-residency pollution, never
  quote it. `Implementation/combined_tick/`.
- **Ops-fragility (review #3, run #5): the milestone-archive copy is non-atomic in all 3 pod
  trainers** (`train_flagship4b.py:337`, `refb_train.py:358`, `refa_train_plus.py:540`) → a kill
  mid-copy silently corrupts a gate milestone. Resume-write path is clean everywhere. Fix intake
  `2026-07-18-atomic-milestone-archive`.
- **TensorRT not installed on the dev box** (`import tensorrt`→missing; ORT CPU EP only). The ONNX
  IR is exported + parity-clean, so only the toolchain/engine step remains.
- ONNX export deps (`onnx`, `onnxruntime`, `onnxscript`) installed in the venv — **export/dev only**,
  never in the inference runtime wheel.
- Windows dynamo ONNX export crashes on emoji progress under cp1252 → run with `PYTHONUTF8=1`.
- Boundary: NEVER write `stack/` directly. Experiments live in `Implementation/onnx_export/` (off-
  Drive for large `.onnx`); stack-changing fixes go through `Implementation/incoming/` intake.

## HANDOFF

- **To Benchmarks & Eval (efficiency ledger / CNCE):** add the **combined deploy-tick row** —
  *fp16 encoder + CUDA-graph predictor 17.75 → 11.16 ms (1.59×), 89.6 Hz, agreement 96.9 %, wp-shift
  0.7 cm; fp16 standalone VRAM 0.56 GB (1.93× < fp32 1.08 GB); $0, 4060* (lower tick → higher CNCE at
  0.263 B params). Still owed from run #4: predictor-graph 6.08→2.36 ms (2.57×); from run #3:
  fp32 14.79 ms/67.6 Hz, fp16 10.67 ms/93.7 Hz/1.39×. Numbers in `Research/2026-07-18-combined-tick-...md`
  + KB + `Implementation/combined_tick/*.json`.
- **To MVP orchestrator (4 pending intakes):** **NEW** `2026-07-18-atomic-milestone-archive` (run #5,
  4 tests, LIVE bug in 3 pod trainers — the `shutil.copy2` archive at `train_flagship4b.py:337` /
  `refb_train.py:358` / `refa_train_plus.py:540` can silently corrupt a gate milestone; fix + drop-in
  helper); `2026-07-18-numerics-safety-sweep` (11-test regression guard → `stack/tests/`);
  `2026-07-17-imagination-logvar-clamp` (17 tests; clamp is in mainline per e753a00 — confirm/close);
  `2026-07-09-models-predictor-failfast` (still unmerged — `assert w==window` live at `predictor.py:89`).
  All small + export-safe. **NB: this branch base (9c36c95) is behind fcbab02 → orchestrator merge.**
- **To Tools&DevEnv:** install a Windows Triton wheel (torch 2.11+cu128) to unlock `torch.compile`
  on the dev box; and `tensorrt`+`onnxruntime-gpu` for the TRT engine build (P0 #1 job card).
- Run completed cleanly; all artifacts committed on `agent/prod-opt-20260718` (off-Drive worktree).
