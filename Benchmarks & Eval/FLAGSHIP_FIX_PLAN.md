# Flagship Fix Plan — making TanitAD-4B-M drive (2026-07-12)

**Grounded in the diagnostic (`DRIVING_DIAGNOSTIC_FRAMEWORK.md §Results`), not guesses.** Every step
has a pre-registered prediction + falsifier + resource. Cheapest-decisive first.

## The proven diagnosis
- Model ADE@1s: held-out MLP **3.89 m**, oracle in-dist ceiling **1.65 m**, vs constant-velocity
  **0.28 m** — 10–15× worse than trivial EVERYWHERE, including straight highway (2.75 vs 0.18).
- **Two separable deficits:** (a) **readout/generalization** = held-out 3.89 vs oracle 1.65 (2.4×
  route-overfit); (b) **representation floor** = oracle 1.65 ≫ 0.28 (even perfect decode of the
  frozen latent can't recover metric trajectory).
- **Root cause (mechanistic):** the model optimizes JEPA latent-future prediction + SigReg
  anti-collapse + imagination-ranking (that is why **D2 PASSES**), but has **no loss that forces the
  latent to encode metric ego-trajectory** — the waypoint readout is a POST-HOC ridge probe, never a
  trained head. SigReg isotropization plausibly spreads the position signal across dims, hurting
  linear decodability further (the step-21k "regression" was this).

## The core hypothesis to fix
> The representation is under-grounded in metric ego-motion because nothing in the objective demands
> it. Making the trajectory an **explicitly trained extraction head (H13) with gradients into the
> encoder+predictor** — instead of a frozen post-hoc probe — will (a) reshape the latent to be
> ego-grounded (push the 1.65 m ceiling down) and (b) generalize across routes (close 3.89→ceiling),
> WITHOUT sacrificing the SSL world-model core (JEPA + imagination stay; trajectory is auxiliary).

This is not abandoning the 1000×-SSL thesis — it corrects an under-specified readout. H13 always
planned extraction heads; the error was leaving them post-hoc.

## Phase A — cheap diagnostics (hours–1 day, mostly no retrain)
- **A1 step-curve** (8.5k/27k held-out ADE): is it undertrained or representation-bound? If ADE still
  descending steeply at 27k → more steps help the readout; if flat → objective-bound. *(14k ckpt
  unavailable — 2-point curve.)* Resource: pod1, hours. Falsifier: flat curve ⇒ steps won't fix it.
- **A2 REF-B D1 = the trained-trajectory-head reference** (already training, pod2). REF-B is
  DIRECTLY supervised on trajectories — its held-out ADE@1s is the number a trained head achieves at
  this scale/data. **If REF-B ≪ 3.89 m → confirms the fix (objective, not architecture).** If REF-B
  also ~4 m → the deficit is data/representation-deeper, escalate. THE key external check.
- **A3 REF-A-full-mix D1** (training, pod3): encoder-axis evidence (frozen DINO vs from-scratch).
- **A4 SigReg-weight probe:** decodability of the 27k latent vs an ablation with SigReg zeroed on a
  short fine-tune — does isotropization cost linear position? Resource: pod1, ~1 h.

## Phase B — THE FIX (the decisive experiment, days)
- **B1 (LAUNCHING NOW): jointly-trained trajectory head + fine-tune.** Add a multi-horizon waypoint
  head (ego-frame, {0.5,1,1.5,2}s) on the encoder state + predictor outputs, trained with a tuned
  auxiliary weight, and **fine-tune the frozen 27k checkpoint** ~8k steps (fast; bounded episodes +
  OOM guard). Also upweight inverse-dynamics + add a forward-dynamics (state,action→Δpose) aux.
  **Pre-registered targets:** oracle ceiling ADE@1s **< 1.0 m**, held-out **< 1.5 m**, **beat
  constant-velocity on the straight stratum**, AND **D2 direction-acc stays ≥ 0.80** (do not break
  what works). **Falsifier:** if a jointly-trained head can't beat the post-hoc 3.89 m after 8k
  fine-tune steps → the encoder's information content is the bottleneck, not the objective → escalate
  to encoder capacity/architecture/data (Phase B2). Resource: pod1, ~half a day.
- **B2 (conditional on B1):** if representation-bound — SigReg geometry fix (exempt the position
  subspace from isotropization) + curve/maneuver oversampling + resolution ablation (secondary) +
  encoder-capacity ablation. Each pre-registered separately.

## Phase C — the real Phase-0 exit (validate DRIVING, not just ADE)
- **C1 closed-loop** MetaDrive/CARLA with imagine-and-select — does D2's ranking translate to route
  completion? (open-loop ⊥ closed-loop, 2605.00066 — ADE is necessary-not-sufficient.)
- **C2 full diagnostic re-run** on the fixed model: beat CV on straight AND curve strata; held-out
  within a factor of the ceiling. Only then → multi-cam / sensors / H-stack.

## Sequencing & the one decision for Sayed
Phase A runs in parallel with the already-training reference arms; **B1 launches now on pod1**
(the free pod). **Decision (default = proceed):** B1 adds explicit trajectory supervision as an
auxiliary head — a training-objective change. Recommendation: **proceed** — it is the highest-leverage,
diagnostic-supported fix and keeps the SSL core intact. Override only if you want the pure-SSL readout
preserved (then we pursue A4/SigReg-geometry first). Config default unchanged until B1 proves out.

## Results (append as experiments land)
- **B1 fine-tune:** _launching on pod1 — trajectory-head + fine-tune 27k; ceiling/held-out/D2 shift._
