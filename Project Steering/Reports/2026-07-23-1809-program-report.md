# TanitAD Program Report — 2026-07-23 18:09 Berlin

*3×/day filed program report (D-025). Evidence class on every number: MEASURED (ours, artifact) ·
PUBLISHED (cited) · HYPOTHESIS. Decision-grade numbers cite the registry / raw eval JSON.*

## Headline
The **flagship restart chain reaches its decision point within the hour**: v4.2b (floor-0.15 fix) hits
Phase B (~step 2000) in ~15 min, and its canary trend over steps 2500–4000 either ends the chain or fires
the pre-staged **from-scratch v4** fallback. **Branch B** (own encoder) and the **Gate-1** prep both
advanced with positive/decisive results. Two bets in flight, both with committed exits.

## 1. Flagship (v4 line) — at the decision point
The architecture is fixed and correct (v1 WM + strategic + tactical & operative anchored-diffusion planners,
**joint, nothing frozen** — verified: encoder+predictor 149/149 require grad, `gnorm_encoder`>0 live, eff-batch
64 = v1). The whole saga is **training-schedule tuning of the planner↔WM coupling**, all MEASURED:
- **v4** lr_trunk 3e-4 → WM ran away (canary 1.3+). ✗
- **v4.1** halve-to-zero controller → planner starved → **10k gate FAIL, held-out `ade_0_2s` 0.852**. ✗
- **v4.2** cap-and-hold floor 0.25 → **floor-too-high, interim eval @4000 `ade_0_2s` 0.987 / canary 0.722**,
  WORSE than v4.1 at <½ the steps; canary climbed 0.62→0.72 in Phase B, controller `hard_breach`. ✗
- **v4.2b** floor 0.15 (fresh from v1) — **LIVE, step 1850, Phase A** (canary 0.520 = v4.2's Phase A, identical
  until λ_plan ramps @2000; `lam_mult` at the 0.15 floor, ready). **Pre-registered decision @ steps 2500-3000:**
  canary ≤0.55 & <v4.2 & `gnorm_pred`↑ → PASS, continue to 10k gate; ≥0.65 (~v4.2) → FAIL, fire from-scratch;
  0.55-0.65 → floor 0.10 or pivot per planner trend.

**Why v4.2b has a real prior (not a blind restart):** it's the first restart that's a *targeted interpolation
between two characterized failures* (v4.1-starve ≈ floor 0, v4.2-degrade = floor 0.25). **⭐ Root-cause insight
(from Sayed's v1 question):** v4's WM degradation is a **warm-start artifact** — v1 succeeded because it trained
its WM + its (simple `tactical_policy`, 22.7M) planner **jointly FROM SCRATCH** (co-evolution, canary 0.42, no
pre-converged WM to degrade). v4.x warm-starts v1's prediction-converged WM and the new planner's gradient yanks
it off-manifold.
- **FROM-SCRATCH v4 fallback: READY** (`…/incoming/2026-07-23-v4-fromscratch/`) — `--from-scratch` = skip
  warm-start, one flag from v4.2b (max attributability). Smoke MEASURED-confirms the premise: from random init
  canary 1.52→1.165 (WM co-evolves, no degradation). Fires if v4.2b's canary runs away, on Sayed's go. ~53h/30k.

## 2. Own encoder (Branch B) — healthy, mechanism working
From-scratch camera-conditioned video-SSL, 2466-clip multi-rig (Branch A cheap warm-start FAILED the ablation).
**LIVE ~11k+/40k, 100% util.** Loss 10.2→~1.0, IDM 5.8→0.3-0.8. ⭐ **Camera-conditioning working (MEASURED):**
all 12 blocks learned from zero-init, rig-A-vs-B token-delta 2.7-7.5/block (vs Branch A's +0.1). Decisive test =
**held-out-rig transfer eval @ step 40k (~10h)** vs the −2.1 cross-rig ablation.

## 3. Closed-loop research / Gate-1 — resolved + prepped
Cited verdict `Architecture & Inference/Research/2026-07-23-closed-loop-wm-training-verdict.md` (3 angles, ~40
sources + Gate 0/0b + rung-3): **RL is the LAST lever**; the free inference floor (guidance + safety filter +
WM-MPC) is RULED OUT for junction off-road. **Gate-1 = closed-loop-aware training** is the fix.
- **Gate-1 prep DONE (MEASURED):** 15 junction scenes, 675 on-policy steps, **recovery labels collected**. ⭐
  Refined mechanism (corrects "execution failure"): the ego tracks its plan tightly (0.49m); the **PLAN degrades
  on-policy** (plan_xte 0.57→12.98m) = **planner covariate-shift**. ⚠️ NuRec ~3.2× OOD → **sufficient to
  prototype, not train robust** until a lower-OOD closed-loop source exists.
- Ship the gradient-nudge floor as a **free safety override** (intersection at-fault collisions 0.71→0.43).

## 4. Benchmarks & closed-loop (within-sim, ~3.2× OOD — relative only)
- REF-C base **beats** flagship-v1's tactical head closed-loop, confirmed at 854 + native res (n=12 paired;
  RETRACTION C7 — the n=1 "flagship wins" was a lucky scene).
- **Scenario-stratified (balanced 38-scene):** flagship **TIES** on roundabout+highway, both **collapse at
  intersections** (the joint target). ΔScore −0.123.
- AlpaSim on A40: ~0.8-1.0× real-time @854 / 0.29× native, renderer-bound.

## 5. Deployment (Orin/Thor) — DONE
**FP16 is the deployment precision. INT8 not worth it** (MEASURED: 2.1% faster enc / 2.1% *slower* pred = QDQ
overhead; readout-head activation collapse; compounding rollout). Tick clears 10 Hz in FP16.

## 6. Fleet
| pod | stream |
|---|---|
| `tanitad-pod2` | **v4.2b** (floor 0.15), step 1850, Phase B in ~15 min |
| `tanitad-pod3` | **Branch B** own-encoder (~11k+/40k, 100%) |
| `tanitad-eval` | FREE — v4.2b-eval host (Phase-B evals ~soon); Gate-1 prep DONE |
| `tanitad-pod` (pod1) | FREE — spare |
IDLE: eval + pod1 hold for v4.2b's Phase-B evals; the only *new* levers (Gate-1 fine-tune) are Sayed-gated +
NuRec-OOD-confounded, so nothing clean to launch.

## 7. Decisions for Sayed
1. **v4.2b verdict — imminent (~1-2.5h).** I apply the pre-registered rule mechanically: continue to 10k, or
   fire the ready **from-scratch** run (your go on the ~53h commitment).
2. **Gate-1** — green-light closed-loop-aware training? Prep done; needs your go **and** a lower-OOD closed-loop
   source for a clean read (NuRec confounds a fine-tune now).
3. **Branch B** continues to its 40k transfer eval (auto).

## 8. Retractions & process this period (root-cause classes in `RETRACTION_LOG.md`)
- **C7** — n=1 "flagship beats REF-C" reversed by n=12. **C2** — my "ablation crashed" false alarm.
- **Data-integrity invariant** — val split `physicalai-val-f1b378f295ae` leaks 78% into train; never eval on it.
- **Process (mine, corrected):** (a) over-requested a classifier-blocked HF push — won't route around it;
  (b) overstated the agent-monitor issue as universal — it's a delivery-gap for *some* agents; (c) idled a
  decisive result waiting for a rubber-stamp — now act per the looping rules (never idle, reversible + veto);
  (d) the 3×/day report cron had expired ~07-15 (masked by the drumbeat) — recreated (`076c9e21`).
