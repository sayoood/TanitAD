# Counterfactual Action Augmentation — teaching the world model to drive by its own consequences (Sayed directive, 2026-07-12)

## The core idea (Sayed)
A real drive gives ONE trajectory per scene — the path the human took. The model never sees *"what
if I had steered left / braked here?"*, so it can only imitate the single observed path; it cannot
learn the **causal action→consequence map** a world model needs to plan. Fix: take real (scene,
action, ego-motion) tuples, **vary the actions**, and **regenerate the corresponding camera data +
ego-motion** with an AI reconstruction engine (NuRec). The variation is an **action** (steer, accel),
NOT a label — the model learns *"in THIS scene, action a' produces observation o' and motion m'."*
This is the training signal for imagination/imagine-and-select (H15) and for metric action-grounding
(the B1 fix), made concrete and self-supervised.

## Why it is high-leverage
- **Isolates action from scene.** Same 3D scene, many actions → the model must attribute the different
  outcomes to its OWN actions, not to scene change. That is exactly the invariance a planner needs.
- **Counterfactual coverage of the rare & dangerous.** Real logs almost never contain "drove toward
  the obstacle then recovered." Reconstruct a safe scene, then generate the risky action branch and
  its consequence — safely, in reconstruction. Directly feeds the Opponent-scenario excellence goals.
- **Multiplies our data assets** (Sayed): each real drive → a fan of counterfactual drives; our
  comma+PhysicalAI corpus becomes a *generative* asset, not a fixed set. Strengthens the
  data-efficiency edge (H7) — more action-diverse supervision per real mile.
- **SSL-faithful.** The regenerated ego-motion is a *simulated proprioceptive consequence*, not a
  human annotation — consistent with the "ground on actions, not labels" principle (B1).

## The pipeline
```
real drive (frames F, actions A, poses M, from comma/PhysicalAI)
  → 1. RECONSTRUCT scene  : NuRec / 3DGS neural reconstruction of the drive volume
  → 2. VARY actions       : smart action-fan A' = A ⊕ {steer±, accel/brake, lane-offset}
  → 3. RENDER             : new camera frames F'(A') from the reconstruction along A'
  → 4. DERIVE consequence : new ego-motion M'(A') from the kinematic rollout (bicycle model + scene)
  → 5. CONTRACT           : (F', A', M') into the episode contract → training tuples
```
The model then trains: given F' and A', predict M' and future latents — the action-conditioned
metric-dynamics objective (B1) with **counterfactual** action coverage.

## Golden sample — the smart variation to build first
Pick ONE clean comma highway scene + ONE PhysicalAI urban scene (both with good geometry). Generate
a **pre-registered action fan** per scene, chosen to teach the consequence structure, not random noise:
1. **Nominal** (A) — the human path, as the anchor/consistency check (F' ≈ F must hold — reconstruction fidelity gate).
2. **Lateral fan** — steer offsets {−0.15, −0.05, +0.05, +0.15} rad held over 2 s → lane-change / drift
   consequences; teaches the steering→lateral-motion map.
3. **Longitudinal fan** — accel {−3, −1.5, +1.5} m/s² → following-distance / stopping consequences.
4. **One safety-critical branch** — steer toward a reconstructed hazard then the recovery action —
   the counterfactual real logs never contain.
Each branch renders ~2 s of frames + ego-motion. **Fidelity gate (falsifier):** the nominal branch's
regenerated frames must match the real frames (SSIM / feature-match above a bar) AND its regenerated
ego-motion must match the real odometry (ADE < ~0.3 m) — if the reconstruction can't reproduce the
REAL action, its counterfactual branches are not trustworthy. Report both.

## Feasibility, cost, risk (honest)
- **NuRec availability:** part of the NVIDIA PhysicalAI / Cosmos family; assess access + license (the
  PhysicalAI firewall applies — internal use, no public redistribution of derived frames). Feasibility
  probe is the first Data-Eng step (does NuRec reconstruct a comma clip acceptably?).
- **Compute:** neural reconstruction is heavy (per-scene optimization). Golden-sample = 2 scenes ×
  ~10 branches — pilot-scale, feasible on a burst pod. Fleet-scale is Phase-1 infrastructure.
- **Domain-gap risk:** reconstruction artifacts could teach the encoder wrong statistics — mitigated
  by the fidelity gate and by keeping counterfactual data a MINORITY augmentation over real data.
- **Alternative/complement:** CARLA/Cosmos give action-varied synthetic scenes cheaply but with a
  larger visual gap; NuRec's value is *real* geometry + counterfactual actions (real+synthetic hybrid,
  matches the scenario-data doctrine's "NuRec neural reconstructions perturbed with scenario elements").

## Sequencing & ownership
- **Now:** this design; Data-Eng NuRec feasibility probe (access + one comma-clip reconstruction
  fidelity number) → backlog P0.
- **After** the B1 action-grounding validates (the objective that CONSUMES this data must work first):
  build the golden sample, train a pilot, measure counterfactual-generalization lift (held-out
  action-consequence accuracy vs the non-augmented flagship).
- Ties to: B1 (the objective), H15 (imagination target), Opponent scenario DB (safety-critical
  branches), scenario-data doctrine (NuRec lane), HYPOTHESIS_LEDGER (propose as a new H — "counterfactual
  action augmentation lifts action-consequence generalization").
