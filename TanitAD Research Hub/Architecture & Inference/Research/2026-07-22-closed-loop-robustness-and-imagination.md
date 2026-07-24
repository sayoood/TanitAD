# Closed-loop robustness needs imagination over a HEALTHY world model — the run's synthesis

**2026-07-22, orchestrator (11-h autonomous run).** Ties two independent MEASURED findings from today
into one architecture direction + a now-cheap proof. Evidence class marked throughout.

---

## 1. The finding (MEASURED, AlpaSim closed-loop, this run)

REF-C — our anchored-diffusion planner, trained **open-loop** (single-shot trajectory ADE vs GT) — was
evaluated in genuine closed loop for the first time (AlpaSim, correct f-theta canonicalization, NuRec
reconstructions):

- **REF-C fails ~half the scenes closed-loop:** at-fault collision **33 % (4/12)**, passes 6/12 (base).
  [n=12 suite; the n=1 highway scene that read "always collides" was the worst case — C5 refinement logged.]
- **2× XL capacity gives NO closed-loop advantage:** base ≥ XL (mean score **0.345 vs 0.246**, 6 vs 5
  passes). Consistent with the registry's "anchor width is the lever, not encoder scale."
- **Open-loop ADE does NOT predict closed-loop.** REF-C base's open-loop ADE@2s is a strong 0.47 m, yet
  it drives into the lead actor half the time. (The eval JSONs already label open-loop as *"weak"*
  evidence, arXiv:2605.00066 — this is the demonstration.)

## 2. The mechanism (HYPOTHESIS, but well-motivated)

Open-loop training optimizes a single-shot map (frames → trajectory) against ground truth; the planner
**never sees the consequences of its own errors compounding**. In closed loop, a small heading/speed
error at t feeds the next observation, and errors accumulate into a lead-actor collision. **Scale can't
fix this** because the bottleneck is not fan quality (already saturated — small ≈ base per anchor) but the
**absence of closed-loop consequence-awareness.** More encoder capacity proposes marginally different
open-loop trajectories; none of them was selected by anticipating the collision.

## 3. The architecture lever — imagination-in-the-loop (= v4's thesis)

A planner that **rolls its candidate actions through a world model and sees the imagined consequences**
(the imagined lead-actor gap closing, the imagined off-road) can *reject the colliding plan before
committing it*. This is exactly the flagship v4 thesis: hierarchical planners (strategic/tactical/
operative) that **predict via imagination over a shared world-model trunk**. Today's REF-C result is the
**strongest evidence yet FOR that thesis** — the pure open-loop diffusion planner is precisely the
ablation (imagination removed) that v4 is meant to beat.

## 4. The catch — imagination needs a HEALTHY world model (the v4 canary finding)

The imagination benefit is only real if the WM's forward prediction is faithful. Today's **v4 canary
finding** (MEASURED) is the warning: v4's WM-integrity canary degraded 0.42 → ~1.1–1.3 and the WM loss is
*rising* **even with the planner→trunk gradient fully clamped** (controller `lam_mult=0`) — so the
degradation is the **trunk LP-FT itself (lr_trunk 3e-4)**, not the planner. A v4 that ships a degraded WM
would get **no imagination advantage** — its imagined rollouts would be as wrong as REF-C's open-loop
guesses, and it would collide too.

**⇒ The two findings compose into one prescription:** closed-loop robustness = imagination **over a WM
that stayed healthy through training.** v4's design is right; v4's current **lr_trunk is too hot** and is
eating the very asset (the WM) that's supposed to deliver the closed-loop win. **v4.1 = lower lr_trunk /
longer trunk freeze in phase A** is not a tuning nicety — it is the prerequisite for the thesis to be
testable at all.

## 5. The proof — now CHEAP, because the harness exists

The AlpaSim closed-loop suite + the REF-C baseline (33 % at-fault, n=12) are **built and staged this run**.
So the discriminating experiment is now cheap and pre-registerable:

> **Once v4 (or v4.1) reaches a checkpoint with a HEALTHY canary (~0.45, near the v1 reference), run it
> through the SAME `public_2601` AlpaSim suite and compare its closed-loop at-fault rate to REF-C's 33 %.**
> Pre-registered: if the imagination-based hierarchical planner has a **materially lower closed-loop
> failure rate than REF-C at matched or better open-loop ADE**, the imagination thesis is validated in
> closed loop — the first hard, closed-loop evidence that v4's hierarchy earns its complexity. If it
> **ties REF-C**, imagination-as-implemented is not buying closed-loop robustness and the lever moves to
> §6.

This is the closed-loop analogue of the open-loop gate — and it is the number that actually matters for a
driving planner, which open-loop ADE demonstrably does not predict.

## 6. Fallback levers (if imagination doesn't close the gap)

- **Closed-loop-aware training** — DAgger / rollout-in-the-loop fine-tuning: expose the planner to its own
  compounding errors during training (the exact distribution shift open-loop training omits). Our
  imagination-in-the-loop closed-loop v1 harness (task #21, no-renderer, kinematic PDM) is a cheap
  proving ground before AlpaSim.
- **Collision-/consequence-aware objective** — add an imagined-collision penalty to the planner loss, so
  selection is not purely GT-ADE.

## 7. Actions

1. **Bank this as the case for v4.1** (lower lr_trunk) — the closed-loop payoff depends on a healthy WM.
2. When a healthy-canary flagship checkpoint exists, run the **§5 pre-registered AlpaSim comparison** vs
   REF-C's 33 %. The harness is staged (`…/incoming/2026-07-22-alpasim-closedloop-evalpod/`).
3. Keep REF-C's **33 % at-fault / base≥XL** as a **within-sim relative** baseline — ⚠️ **it is
   reconstruction-OOD-confounded** (REF-C's open-loop ADE *on the AlpaSim reconstructions* = 1.52, **3.21×**
   its real-footage 0.47; RETRACTION_LOG C6), so it is NOT a clean absolute model measure. Open-loop ADE is
   not a substitute either.

## 8. RESULTS (2026-07-22 late) — the thesis has early, consistent support

Two independent MEASURED results, both directional, both pointing the same way:

- **No-renderer proof (CI-separated, clean, verified).** Flagship v1 WM + the kinematic closed-loop harness
  (n=265 windows / 12 held-out val eps): imagination-in-the-loop (re-plan every 0.1 s on the imagined
  latent) vs single-shot open-loop → **paired Δ ADE@2s = −0.213 m [−0.341, −0.053]** (paired
  episode-cluster bootstrap, separated, P(B>A)=0.0045); **divergence-rate >5 m nearly halves 39 %→22 %.**
  IMAGINATION_HELPS. ⚠️ Weak form (re-plan, NOT CEM imagine-and-select); modest (~11 %); FDE@2s endpoint
  NOT separated; self-referential WM. (`…/incoming/2026-07-22-imagination-closedloop-proof/`)
- **AlpaSim (n=1, OOD-confounded but fair — both models see the same reconstruction input).** On the scene
  where all three REF-C variants collide at-fault, **flagship v1 (WM + tactical policy) drives
  collision-free** (PASS, score 0.699). (`…/2026-07-22-alpasim-closedloop-evalpod/`, rollout `71f9740c`)

**Read:** the WM/imagination planner beats open-loop diffusion in closed loop on two different harnesses —
the case FOR v4. The **v4.1 lr_trunk fix** (canary stays healthy at 0.45) is precisely what lets the full
CEM imagine-and-select version be tested at scale. **The decision-grade test remains §5** (a healthy-canary
v4.1 vs REF-C on the AlpaSim suite, read with the OOD caveat).
