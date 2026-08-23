# 2026-07-09 — Half-precision latency+accuracy on the decision path; models review #2

**Agent:** Production & Optimization (Saturday run #2) · **Phase 0** · **QUALITY: full**
**Hardware:** local RTX 4060 (declared Orin proxy, I8) · **cost:** $0 (local) ·
**wall-clock:** ~1 experiment + 1 review, single session.

## TL;DR

- **TensorRT-proper is blocked on this dev box** (`import tensorrt` →
  ModuleNotFoundError; onnxruntime has **CPU EP only** — no CUDA/TRT provider).
  So P1.4 was run as the **honest TRT-fp16 precursor**: a TRT-fp16 engine casts
  the same weights, so its first question — *does half precision move the latent
  past tolerance or flip the driving decision?* — is answerable today.
- **FP16 is deployment-safe for the decision path; BF16 is NOT.** Measured on 64
  real comma2k19 val windows (step-6500 ckpt), decoding a K=9 sustained-steer
  imagine-and-select fan through **one fixed fp64 probe** (precision-invariant,
  so any flip is attributable to the world model, not the probe):

  | precision | speedup¹ | encoder rel-err | **imagine-select agreement** | decoded-waypoint shift (mean/max) |
  |---|---|---|---|---|
  | fp32 (ref) | 1.00× | — | 100 % | — |
  | **fp16** | ~1.6× | **7.8e-4** | **95.3 %** (3/64 flips) | **3.9 cm / 19 cm** |
  | bf16 | ~1.8× | 7.2e-3 | **67.2 %** (21/64 flips) | **47.7 cm / 3.58 m** |

  All outputs finite (no overflow) in both half formats. ¹ **speedup ratios are
  directional only** — see the honesty caveat below.
- **The G-P2 lesson, with a number:** bf16 is *faster* but changes the chosen
  maneuver in **1 of 3** windows and shifts waypoints ~0.5 m. Speed alone would
  have picked bf16 (the usual "safe training dtype"); the accuracy delta
  **rejects it** for the operative decision. **fp16 is the TRT target.**
- **Models review #2 finding:** the operative predictor's only input guard is an
  `assert` (stripped under `python -O` → silent wrong-window inference) + cryptic
  matmul errors on wrong dims. Fixed as intake `2026-07-09-models-predictor-failfast`
  (8 tests, export-safe). File:line = `stack/tanitad/models/predictor.py:73`.

## 1. Why fp16 wins and bf16 loses (mechanism, not just numbers)

fp16 carries **10 mantissa bits** (rel-precision ≈2⁻¹⁰ ≈ 9.8e-4); bf16 carries
**8** (≈2⁻⁸ ≈ 3.9e-3). The encoder/predictor deltas here are **precision-limited,
not range-limited** — both formats stay finite, so bf16's wider exponent buys
nothing, while its 2 fewer mantissa bits show up directly: encoder rel-err
7.2e-3 (bf16) vs 7.8e-4 (fp16), a ~9× gap. Amplified through the imagine-and-
select **argmin over a tight K=9 fan** (steer sweep ±0.12 rad, candidates only
centimetres apart at the 1.6 s horizon), that 9× latent-error gap is the
difference between **5 %** and **33 %** decision flips. The predictor heads show
the same ordering at every horizon (h1/h2/h4 rel-err ~7.8e-3 bf16 vs ~8.8e-4
fp16; max|Δ| 0.28 vs 0.048). Cosine stays ≈1.0 throughout — cosine is **too
coarse** to see this; the decision-space metric (selection agreement, waypoint
metres) is what catches it. That is the whole point of scoring precision changes
in the decision space, not the latent space (G-P2).

**Falsifier verdict.** Implicit hypothesis "half precision is safe on the
decision path": **TRUE for fp16, FALSE for bf16.** The folk default "bf16 is the
safe reduced precision" is **falsified for this inference decision path** — it is
a *training* heuristic (range for gradients), not an *inference* one.

## 2. Honesty caveat — absolute latency was measured under GPU contention (P8)

While the experiment ran, `nvidia-smi` reported the 4060 at **99 % utilization,
5.6 GB used**, with **`CarlaUE4-Win64-Shipping` (~4 GB) + a local python (~4 GB)
resident** (a local CARLA scenario job, not this agent's — left running). So:

- **Accuracy deltas are contention-immune** (model *numerics* do not change with
  GPU load) → the fp16-safe / bf16-unsafe result is **solid**.
- **Absolute latency / Hz are contended and NOT decision-grade** this run (fp32
  decision tick read 33.5 ms / 29.9 Hz here vs the clean **15.07 ms** baseline on
  2026-07-08). The **speedup *ratios*** (fp16 ~1.6×, bf16 ~1.8×) are directional
  (all three precisions ate the same contention) but should not be quoted as
  absolutes. **Peak-VRAM per precision is also confounded** — the fp32 reference
  model stays resident while the half model loads, so the reported 1.65 GB
  (fp16/bf16) double-counts and does **not** show fp16 using more memory than
  fp32 (real fp16 deployment VRAM is lower).
- **Production hygiene finding (→ backlog):** I8/CNCE latency claims must be taken
  on an **exclusive, clock-pinned GPU** (`nvidia-smi -lgc`, no other compute
  apps) or they are not comparable session-to-session. The 2026-07-08 15.07 ms
  number remains the reference until a clean re-run confirms the fp16 absolute.

## 3. Actionable recommendations

1. **Deploy fp16, not bf16, on the operative decision path.** TRT-fp16 is the
   target; keep the ViT tower ≥fp16 (fp16 finite/no-overflow here → the
   "vision-tower-stays-FP16, native-TRT-ViT-INT8-is-a-trap" KB guidance holds).
2. **Pre-registered TRT-fp16 acceptance bar** (ready for when the toolchain
   lands): TRT-fp16 must match this fp16 result — **selection agreement ≥ 95 %,
   decoded-waypoint shift ≤ ~4 cm mean** on the same 64 windows — else the engine
   has a fidelity regression beyond raw fp16 and is rejected (don't hack).
3. **Install path for TRT** (cross-agent, Tools&DevEnv / next run): `tensorrt` +
   `onnxruntime-gpu` (CUDA 12 EP) on the dev box, OR build the engine on the pod
   when a trainer is idle. Not a paid resource → EXECUTE-class (D-018), sequenced
   behind a clean-GPU window. Added to backlog P1.4a.
4. **Integrate the fail-fast guard** (intake below) — closes a silent-wrong-data
   hole on the hot path.
5. **Re-run latency on an idle/clock-pinned 4060** to publish the fp16 absolute
   Hz + a clean VRAM delta (backlog P1.4b).

## 4. Compliance review #2 — `tanitad/models/` (operative-predictor fail-fast)

Reviewed the models cluster (encoder, predictor, sigreg, imagination, readout,
fourbrain) against the PRODUCTION_READINESS checklist. Cluster is generally
clean and **export-verified** (2026-07-08): SigReg correctly pins fp32
(`sigreg.py:62-65`); `eval()` disables the F-5 grad-checkpoint lever
(`encoder.py:60-61`); FiLM/MHA/causal-triu all ONNX-clean. **One shipped
finding**, two logged:

- **SHIPPED — operative predictor input guard is `assert`-only**
  (`predictor.py:73`). Measured failure modes this run: wrong window →
  `AssertionError` (**stripped under `python -O`** → a W-1 window then re-aligns
  on every axis and runs **silently**); wrong state_dim/action_dim → cryptic
  `RuntimeError: mat1 and mat2 shapes cannot be multiplied`. → intake
  **`2026-07-09-models-predictor-failfast`**: `validate_operative_inputs` gives
  named-axis, `-O`-proof `ValueError`s; pure shape-int checks constant-fold on
  static-shape export (**`test_export_safe` proves ONNX opset-17 export still
  works**). **8 tests pass.** Small, mergeable, no valid-path behavior change.
- **LOGGED (P1 next)** — same `assert`-only guard on `tactical_pred` (same class;
  fold into the same fix). `SpatialGridReadout.__init__` asserts (`readout.py:25-26`)
  are construction-time only → lower priority (a stripped `-O` mis-config surfaces
  at first forward), keep as asserts or promote for symmetry.
- **LOGGED (models numerics)** — `imagination_nll` (`imagination.py:135`)
  `torch.exp(-logvar)` is unclamped; a very negative `logvar` (over-confident
  cell) overflows to `inf` → NaN loss / NaN LOPS trigger. Training-path today but
  the logvar is exported as the H2/LOPS/fallback trigger at inference → clamp
  `logvar` to a sane band. Deferred to a numerics-focused package (needs a
  training-side falsifier).

## 5. Artifacts

- Experiment: `Implementation/half_precision/half_precision_latency_accuracy.py`
  → `half_precision_step6500.json` (per-stage latency, per-horizon deltas,
  decision metrics, SUMMARY block).
- Intake: `Implementation/incoming/2026-07-09-models-predictor-failfast/`
  (`validate.py`, `tests/test_predictor_failfast.py` 8✓, `INTAKE.md`).
- Ledgers updated: `PRODUCTION_READINESS.md` (models row + fp16/bf16 export note),
  `BACKLOG.md` (P1.4a/P1.4b), `KNOWLEDGE_BASE.md`, `STATE.md`,
  `Benchmarks & Eval/LEADERBOARD.md` (efficiency block, contention-caveated).

## 6. Prior-package status

Review #1 intake `2026-07-08-data-cluster-compliance` → orchestrator verdict
**integrate-with-changes** (per PROJECT_STATE session log; cache-key collision +
save fail-fast now in `stack/tanitad/data/`, suite 178 green). No further action.
