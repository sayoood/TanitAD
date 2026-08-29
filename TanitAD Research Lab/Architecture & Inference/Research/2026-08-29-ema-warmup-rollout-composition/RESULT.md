# EMA-teacher warmup dynamics + composed-vs-trained multi-step rollout — literature pass

**Package** `Architecture & Inference/Research/2026-08-29-ema-warmup-rollout-composition` ·
**Author** Research Lab agent (daily run 003, 2026-08-29) · **Feeds** TONIGHT's MM-E1 stage-2
read (`Project Steering/PREREG_P0_EMA_BAKEOFF.md`, 30k pair launched 07:27 UTC) · literature
only, 0 GPU. All paper claims **PUBLISHED (banked, sha256 in `Library/library.json`)** unless
marked otherwise.

## Q1 — Is early-training damage from an EMA target documented? YES, with the fix.

1. **The mechanism is stated verbatim in two primaries.** Mean Teacher [1703.01780]: α=0.99
   during ramp-up, 0.999 later — *"the student improves quickly early in the training, and thus
   the teacher should forget the old, inaccurate, student weights quickly."* data2vec
   [2202.03555]: τ linearly annealed **0.999 → 0.9999** (30k updates speech / 100k NLP) — the
   schedule *"results in the teacher being updated more frequently at the beginning of training,
   when the model is random."* This is our warmup-transient hypothesis, published 2017/2022.
2. **The magnitude is measured (BYOL constant-τ ablation, 300ep ImageNet)** [2006.07733]:
   τ=0.99 → **72.5** top-1, τ=0.999 → 69.8, τ=0.9 → 68.4, τ=1 (frozen random) → 18.8,
   τ=0 (no EMA) → 0.3. Teacher speed is first-order in BOTH directions: a too-slow teacher
   costs 2.7 pts, and the optimum depends on run length.
3. **The JEPA-family standard is a ramp, never a constant.** I-JEPA [2301.08243]: momentum
   **0.996 linearly → 1.0 over pretraining**; BYOL: 0.996 cosine → 1.0. No published
   JEPA-family system trains with a fixed τ.
4. **Our arms are fixed τ=0.996, no ramp, no delayed start** — MEASURED,
   `stack/scripts/train_v6_staged.py:6401` (`--ema-decay` default 0.996; `_EmaCopy` applies it
   constant). Teacher memory horizon 1/(1−τ) = **250 steps**. Arithmetic on measured inputs:
   at 2k that horizon is **12.5 % of the run** — proportionally 10–100× above every published
   operating point (data2vec start ≈1 % of run; BYOL const-0.99 ≈0.1 %) — the teacher averages
   back into near-random weights for the whole run. At 30k the same τ is **0.83 % of the run**,
   inside the published early-phase band (though never annealing to 1.0 late).
5. **Documented alternatives beyond ramps.** (a) **Frozen teacher** — SALT [2509.24317]
   replaces the EMA teacher with a frozen pixel-pretrained encoder: ViT-g **76.2 SSv2 vs
   V-JEPA 2's 75.3**, better accuracy–FLOPs Pareto (SALT ViT-L at 1.2 EF beats V-JEPA 2 ViT-H
   at 3.5 EF), loss becomes downstream-predictive (R²=0.951), and *"high-performing students
   emerge even with small, sub-optimal teachers."* This is direct published support for our
   pre-committed EMA-OUT fallback (frozen-teacher feature target). (b) **Hard refresh +
   isolation gating** — CGTR [2606.03532]: continuous EMA = *"chronic teacher contamination"*;
   consolidation-gated refresh beats EMA 0.806 vs 0.713. ⚠️ Scope: LLM on-policy distillation
   (Qwen3-8B); transfer to vision SSL UNVERIFIED.

**Reading tonight's MM-E1 stage-2 with this:** the 2k cos deficit (−27.6/−46.9 %) is exactly
the deficit the family's ramp exists to prevent, at a τ-to-run-length ratio far off the
published map; the 30k arm runs inside the published band, so the transient account predicts
the cos gap **narrows materially at 30k**. ⚠️ Symmetric caveat, committed here before the
read: the 2k **drift gain** (−11.7/−19.6 %) may share the same transient origin (a 250-step-lag
teacher is closer to a mean-anchored target); if at 30k both the cos cost AND the drift gain
vanish, the 2k MIXED was warmup artifact end to end — that lands in the prereg's EMA-OUT row,
no new outcome class needed.

## Q2 — Composed one-step vs trained multi-step heads in world models

- **The four closest-relative SOTA systems all roll a ONE-STEP model; none has per-horizon
  heads.** DreamerV3 [2301.04104]: one-step RSSM, imagination horizon **H=15** by composition
  (hp table, banked PDF p.21). DINO-WM [2411.04983]: one-step teacher-forced predictor
  (frameskip 1–5), multi-step futures ONLY by autoregressive composition inside CEM/MPC.
  TD-MPC2 [2310.16828]: one-step latent model, MPPI plans at **H=3**. V-JEPA 2-AC
  [2506.09985]: autoregressive frame-level prediction, receding-horizon control.
- **What breaks at long horizons: compounding error, and it is quantified.** MBPO
  [1906.08253]: *"inaccuracies in learned models tend to make long rollouts unreliable"*;
  k=1 branched rollouts are *"a baseline that is surprisingly difficult to beat"*; 500-step
  rollouts "too inaccurate". V-JEPA 2-AC acknowledges verbatim that representation-space
  accuracy *"decreases with longer autoregressive rollouts."*
- **The SOTA fix lives in the TRAINING LOSS, not the architecture.** TD-MPC2 unrolls the
  one-step model **H=3** steps with discount λ=0.5 in training; V-JEPA 2-AC adds a rollout
  loss with **T=2, differentiating through exactly one recurrent step** (vanishing-gradient
  guard); DINO-WM trains pure teacher-forcing and still plans. Nobody trains separate h-step
  output heads.
- **Theory frame** [2603.23465]: composed one-step models are statistically preferable in
  **small-data regimes**; direct multi-step predictors need substantially more samples (harder
  identification per horizon). This matches our MEASURED F0 panel fact — h=1 real, h≥2 heads
  vestigial — as the expected small-data outcome, not an anomaly.

## What this changes for TanitAD (≤3)

1. **Tonight's stage-2 read gets the calibration above** — and whichever way it lands, the
   next step is already literature-backed: EMA-JOINS ⇒ do NOT adopt fixed-τ into V7_RECIPE
   §5.1 as-is; register a one-variable τ-ramp arm first (I-JEPA linear 0.996→1.0 — fixed-τ
   EMA is a mis-specification no published system ships). EMA-OUT ⇒ SALT licenses the
   pre-committed frozen-teacher fallback, and cheaply: the teacher may be small/sub-optimal.
2. **Cheap 2k discriminator available without waiting for 30k economics** (~35 min Thor /
   dev-box): rerun ema2k with a data2vec-style fast-early teacher (e.g. τ 0.99→0.9996 linear).
   If the cos deficit shrinks at 2k, the transient account is confirmed at the scale where it
   was observed; one variable, prereg-able as MM-E1 stage-2b.
3. **v7 rollout design: drop trained h≥2 heads, keep composed h=1 + a SHORT train-time unroll**
   (TD-MPC2 H=3/λ=0.5 or V-JEPA 2-AC T=2 one-recurrent-step gradient). Unanimous across the
   four closest-relative systems, theory-backed for small data, and consistent with F0's
   measured vestigial heads. Registerable on the tiny ladder (one variable: unroll loss on/off).

## Named empty searches

- **"Delayed EMA start"** (teacher literally frozen for the first N steps, then EMA begins) as
  a studied mechanism: NOT FOUND. Closest published relatives: data2vec's low-τ start;
  SALT's fully-frozen stage-1 teacher.
- **EMA-teacher behaviour at very short budgets (≤5k steps): NOT FOUND.** Every published
  schedule above is tuned for ≥30k–100k-update runs — our 2k read sits off the published map,
  which is itself a reason to distrust 2k EMA verdicts in either direction.
- **Latent world models with separate per-horizon prediction heads: NOT FOUND** in the WM SOTA
  (direct multi-horizon output exists in time-series forecasting and AD trajectory decoders,
  not in latent world models).

## Banked primaries (all sha256-verified via `kb_add.py`)

1703.01780 (Mean Teacher) · 2202.03555 (data2vec) · 2006.07733 (BYOL) · 2301.08243 (I-JEPA) ·
2509.24317 (SALT/frozen teachers) · 2606.03532 (CGTR/teacher refresh) · 2310.16828 (TD-MPC2) ·
1906.08253 (MBPO) · 2603.23465 (single- vs multi-step efficiency) · cited-by updated on
already-banked 2411.04983 (DINO-WM), 2506.09985 (V-JEPA 2), 2301.04104 (DreamerV3).
