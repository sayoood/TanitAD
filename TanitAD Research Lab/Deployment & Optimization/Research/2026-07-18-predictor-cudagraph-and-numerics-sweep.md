# Predictor batch-1 CUDA-graph latency attack + numerics-safety sweep

- **Agent / date:** Production & Optimization (Saturday), 2026-07-18 (run #4)
- **Phase:** 0 · **Backlog:** P0 #3 (predictor latency attack) + P0 #4 (numerics sweep)
- **Hardware / cost:** local **RTX 4060 8 GB** (declared Orin proxy, I8), exclusive; ~25 min
  wall-clock total; **$0**. GPU idle (1.06 GB baseline, 4 % util) before the run.
- **Resource declaration (G-I):** 4060 only. **Why not the eval-pod A40:** this is a **batch-1
  single-stream latency microbench** — the Orin target is single-stream, so the 4060 *is* the
  correct instrument (the A40 measures server/throughput, a different question). The A40 job is
  the TRT-fp16 *engine build* (P0 #1), which stays **toolchain-blocked** on the dev box
  (`import tensorrt` → ModuleNotFoundError; ORT exposes CPU/Azure EP only, no CUDA/TRT) and is
  carried as a job card, not run — the three pods are training (pod2 flagship no-touch; pod1/pod3
  arms) so no idle pod this window (M-3: blocked ≠ idle → job card + escalation below).

---

## 1. Experiment — predictor batch-1 latency attack (G-H / G-P2)

**Motivation (from run #3, 2026-07-17):** the fp16 win is *entirely* the ViT encoder
(8.98 → 4.69 ms, 1.9×); the predictor + K9-select passes barely moved under fp16 (5.81 → 5.99 ms)
→ they are **batch-1 launch-bound** (kernel-launch overhead dominates), not compute-bound.
Precision cannot help a launch-bound pass; **CUDA-graph capture** (record the whole launch
sequence once, replay in a single call) is the matching lever. Target = the operative predictor
forward (`world.imagine` == `world.predictor`, `OperativePredictor`, depth 12), exactly the pass
run #3 timed. fp32, tf32 **off** (honest reference), 200 reps p50/p95 after 20-rep warmup, ckpt
step-6500, 64 real comma2k19-val windows. Script:
`Implementation/predictor_latency/predictor_latency_attack.py`;
result: `Implementation/predictor_latency/predictor_latency_20260718.json`.

**G-P2 — speed AND accuracy, side by side:**

| variant | predict_1pass p50 (p95) | ×  | select_K9 p50 (p95) | ×  | worst rel-err vs eager | select agreement | wp-shift |
|---|---|---|---|---|---|---|---|
| eager fp32 (ref) | 6.08 ms (7.38) | 1.00 | 5.94 ms (7.30) | 1.00 | — | 1.00 | 0.00 m |
| **CUDA-graph (manual)** | **2.36 ms (2.79)** | **2.57×** | **4.45 ms (4.89)** | **1.33×** | **2.8e-7** | **1.00** | **0.00 m** |
| torch.compile(cudagraphs) | 117.8 ms | 0.05× | 74.9 ms | 0.08× | 2.7e-7 | 1.00 | 0.00 m |
| torch.compile(reduce-overhead) | — FAILED — | | | | | | |

**Findings**

1. **Manual CUDA-graph capture is a *free* latency win.** predict_1pass **6.08 → 2.36 ms (2.57×)**,
   select_K9 **5.94 → 4.45 ms (1.33×)**; accuracy vs eager: max|Δ| 7.6e-6, **rel-err_max 2.8e-7**,
   cosine 1.0 across all three horizons, imagine-and-select **agreement 100 %**, decoded-waypoint
   shift **0.00 m**. Expected — the graph replays the *same fp32 kernels*, so the only thing removed
   is per-kernel launch overhead. **Pre-registered falsifier (>10 % ⇒ launch-bound) cleared 25× on
   the predictor pass** → the launch-bound diagnosis from run #3 is **confirmed**.

2. **The gain scales with launch-boundedness (batch dilutes it).** predict_1pass (batch 1) 2.57× ≫
   select_K9 (batch 9) 1.33×: at K=9 each kernel does 9× the work, so launch overhead is a smaller
   fraction of the total → less to reclaim. The two deployment levers are **orthogonal**: the
   **encoder is compute-bound → precision (fp16)**; the **predictor is launch-bound → CUDA graph**.

3. **On this Windows dev box, the deployable graph route is MANUAL `torch.cuda.CUDAGraph`, not
   `torch.compile`.** `torch.compile(mode="reduce-overhead")` → **`TritonMissing`** (Triton is not
   installed; inductor needs it for GPU codegen). `torch.compile(backend="cudagraphs")` runs
   Triton-free but is **~20× SLOWER** here (117.8 ms) — per-call dynamo guard/re-trace overhead
   swamps the tiny predictor. Accuracy is fine in both; only the manual capture is viable.
   → **Deployment note:** target the Orin runtime with hand-rolled CUDA-graph capture of the
   operative tick; if we want `torch.compile` on the dev box, Tools&DevEnv must install a Windows
   Triton wheel (matches torch 2.11+cu128).

4. **Deployable-tick projection (additive, honest — not yet measured combined).** Run #3's fp32
   tick 14.79 ms = encode 8.98 + select 5.81. Swapping in **CUDA-graph select (4.45)** alone →
   ~13.4 ms ≈ **74 Hz**; **combined with the fp16 encoder (4.69) → ~9.1 ms ≈ 109 Hz** — from ~68 Hz
   fp32 today. This is a projection from separately-measured stages; a single-graph full-tick +
   fp16 combined harness is the confirming next step (backlog).

**Bounds (P8):** batch-1 microbench on the step-6500 ckpt; latency is architecture- not
weight-determined, so the numbers hold across training progress. Clocks are not admin-pinnable on
this box (run #3 finding) — mitigated by p50/p95 over 200 reps (tight: predict p95/p50 = 1.18).
`probe_imag_r2` is in-sample (2048-dim latent, 64 windows) → the probe is a **fixed reference
decoder** for attributing selection flips to the WM, **not** a capability claim.

## 2. Compliance review — numerics-safety sweep (G-P1), "close the class"

**Directive (P0 #4):** generalize the 2026-07-17 `imagination_nll` NaN fix — sweep `stack/tanitad`
for other unbounded `exp`/`log`/division on **learned/data** outputs. Result: **the class is
already closed.** Every site is guarded or bounded-by-construction:

| site | mechanism | verdict |
|---|---|---|
| `imagination.ImaginationField.logvar_head` | `.clamp(-10,10)` at head (`imagination.py:124`) | GUARDED |
| `imagination.imagination_nll` `exp(-logvar)` | `.clamp(-10,10)` in nll (`:136`) | GUARDED |
| `replay/arms.py:284` sigma export `(0.5·logvar).exp()` | `.clamp(-10,10)` | GUARDED |
| `refs/refb.py:366` `FeatureOOD.score` `sum/count` | `count<2 → zeros`; `var.clamp_min(eps)` | GUARDED |
| `models/sigreg` epps-pulley / `SigReg` `exp(-b²·…)` | negative exponent (∈(0,1]) | SAFE-BY-CONSTRUCTION |
| `eval/spectral` `effective_rank`/`tail`/`optimal_k` | `.clamp_min(1e-12)` | GUARDED |
| `models/fourbrain` erank `exp(-entropy)` (`:297`) | `.clamp_min(1e-12)` | GUARDED |
| `eval/metrics` `np.exp(-λ·col)`, `denom=max(.,1e-9)` | neg-exp / clamped denom | SAFE-BY-CONSTRUCTION |

No new unguarded site. **Contribution = the executable regression guard** that keeps it closed:
`Implementation/incoming/2026-07-18-numerics-safety-sweep/` — 11 tests, **all green (1.47 s)**,
test-only (proposed `stack/tests/test_numerics_safety.py`, no source change). Genuine
failing-then-passing witnesses: the pre-fix `imagination_nll` path overflows to **inf** at
logvar=-100 (asserted) while the guarded call stays finite + finite grads; the field head bounds
logvar to [-10,10] at 500× input scale (guards the OKRI/LOPS `.exp()`); `FeatureOOD.score` stays
finite before 2 samples and under zero variance. G-P1 satisfied (file:line + failing-then-passing).

## 3. Actionable recommendations

- **A1 (deploy):** capture the **operative predictor/select path as a CUDA graph** on the Orin
  runtime — free 1.33–2.57× on the launch-bound half, **zero accuracy cost**. Combine with the
  fp16 encoder (orthogonal levers). Projected tick ~9 ms / ~109 Hz.
- **A2 (Tools&DevEnv):** `torch.compile` is **not** the graph route on this box — install a Windows
  **Triton** wheel (torch 2.11+cu128) if we want inductor; else the runtime uses manual capture.
- **A3 (next run):** full-tick single-graph (encode+predict+select) + **fp16+graph combined
  latency/VRAM harness** to replace the additive projection with a measured combined tick; then the
  P1.4c one-process VRAM harness alongside it.
- **A4 (integrate):** ship the numerics regression guard to `stack/tests/`.
- **Handoff → Benchmarks & Eval (efficiency ledger / CNCE):** add the predictor-graph efficiency
  row — *operative predictor 6.08→2.36 ms (2.57×), select_K9 5.94→4.45 ms (1.33×), rel-err 2.8e-7,
  agreement 100 %, $0, 4060*; feeds the CNCE latency term (lower tick = higher CNCE at fixed
  params). Established handoff channel (run #2/#3 precedent).

## 4. Job card (M-3) — TRT-fp16 engine build (eval-pod A40 / idle pod)

Still owed (P0 #1), toolchain-blocked on the dev box. Runnable card for the next agent with an
idle pod or a TRT-equipped box:
- Env: `pip install tensorrt onnxruntime-gpu` (CUDA-12 EP). ONNX IR already exported + parity-clean
  (`Implementation/onnx_export/`, opset 17/18, max|Δz| ≤1.2e-5).
- Build TRT-fp16 from the exported encoder+predictor ONNX; verify against the fp16 bar
  (imagine-and-select **agreement ≥95 %**, wp-shift **≤~4 cm** on the same 64 windows).
- Report Hz + peak VRAM on **both** A40 (server row) and 4060 (Orin-proxy deploy row).
- Falsifier: any TRT-unsupported op ⇒ document, do not hack (run #1 verified MHA/FiLM/causal-triu
  all export clean, so none expected).

---

**Gate self-check:** G-A (every claim → JSON/repo path). G-B (A1–A4 actionable). G-C (KB updated).
G-D (no hypothesis *status* change; efficiency evidence handed to Benchmarks per precedent). G-E
(11-test guard, green). G-H (2 measured experiments, hardware/wall-clock/falsifier). G-P1 (file:line
+ failing-then-passing). G-P2 (accuracy delta beside every speed delta). G-I (resource declared;
real-compute run on 4060; TRT job card for the blocked A40 item). **QUALITY: full.**
