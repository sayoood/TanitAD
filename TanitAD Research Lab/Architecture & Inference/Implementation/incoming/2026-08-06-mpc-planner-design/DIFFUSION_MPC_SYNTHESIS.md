# Diffusion-MPC — combining v5f's diffusion planner with the MPC mechanism

**2026-08-07, PI question: "Can we combine/extend the diffusion planner of v5f with the
MPC mechanism?"** Answer: yes — four integration levels, each independently testable,
composing into one architecture. Extends `MPC_WM_DESIGN.md`; consistent with the corrected
`V5F_ARCHITECTURE_REVIEW.md`.

## Why the fit is exact

- MPPI/CEM's weak point is its **proposal distribution** (Gaussian around a warm start —
  poor coverage of discrete choices like branch/gap selection). v5f's anchor+denoise fan
  is a **learned, scene- and goal-conditioned proposal** that MEASURED reaches oracle
  0.21 m — better than any hand-crafted sampling prior we could write.
- v5f's weak point is **choosing** (sel_gap ≈ 2×, static over 17k steps) and its
  imagination conditioning has **no per-candidate axis** (one shared consequence summary
  for the whole fan — documented in `imagine_probes`). MPC is precisely a per-candidate
  consequence check with an explicit, auditable cost.

## The four integration levels

**L1 — MPC as re-ranker (cheapest, first).** Keep everything frozen. Selector prunes the
fan 256→top-8; for each survivor, convert the trajectory to a unicycle action sequence
(inverse kinematics; see prerequisite P2), roll the predictor per candidate (batched — 8
rolls ≈ the §1.12 pipeline's cost), score each on ITS OWN imagined future with the
MPC_WM_DESIGN cost terms (comfort barriers, distance-keeping vs predicted lead, goal
progress, ensemble/latent-plausibility). Emit the argmin.
*Pre-registered gate X0: MPC re-rank closes ≥ 50 % of sel_gap on the same fan (plan
0.42-0.48 → ≤ 0.33 at oracle 0.21), evaluated at T1.* This alone would recover most of the
2× the selector leaves on the table.

**L2 — cost-guided denoising.** The denoiser is differentiable and most cost terms are
differentiable: inject cost gradients into the (only 2) truncated denoise steps,
classifier-guidance style: `x ← denoise(x) − η·∇C(x)`. The fan itself bends toward
low-cost regions before selection — guidance shapes, MPC still chooses. η swept with a
zero ablation; watch fan DIVERSITY (guidance can collapse modes — oracle_ade is the canary:
if oracle worsens while plan improves, guidance is eating the multimodality that makes
the fan valuable).

**L3 — receding-horizon warm start.** Next cycle's denoise starts from the previous
selected candidate time-shifted (noise seeded around it) instead of the static anchor
vocabulary — the diffusion equivalent of MPC's warm start. Directly targets the
replan-jump/temporal-stability metrics.

**L4 — amortisation (fast/slow closed).** Distill L1's MPC choices back into the learned
selector (DAgger-style: MPC is the teacher on frames where it ran). Deployed shape:
selector always (fast); MPC verification triggered by low score-margin, high ensemble
disagreement, or high-stakes context flags (slow). The PI's fast/slow-thinking pattern at
the operative level, with the slow path defined by explicit costs rather than a second
network.

## Prerequisites (all already in the backlog)

- **P1 — action-controllability of the v5f trunk** (stage-A probes / E3): if the predictor
  under-weights actions (v1arch's measured longitudinal pathology), per-candidate rolls
  are fiction. Must be measured on the 30k trunk BEFORE trusting L1 scores; stage-A
  post-training is the remedy if it fails.
- **P2 — feasible action conversion:** inverse-unicycle of free-waypoint candidates only
  works if candidates are near-feasible (REF-C-XL fan: 72 % were not). Either project
  candidates onto the unicycle manifold before rolling (cheap, lossy) or retrofit
  unicycle-parameterised anchors (the clean fix; review §4). The fan-geometry dump at 30k
  decides which.
- **P3 — cost-term feasibility** (MPC E0 probe): lead-gap/curvature decodability from
  ẑ decides the safety/geometry terms.

## Experiment ladder (pre-registration sketches; all post-30k, frozen-trunk)

| id | test | cost | gate |
|---|---|---|---|
| X0 | L1 re-rank vs learned selector vs oracle, T1 | ~2 GPU-h | ≥50 % sel_gap closed |
| X1 | L2 guidance η sweep | ~2 GPU-h | plan ↓ with oracle NOT ↓ (diversity guard) |
| X2 | L4 distilled selector vs original | ~3 GPU-h | distilled ≥ 80 % of MPC's gain at zero runtime cost |
| X3 | L1+L2+L3 full stack, T1 + temporal stability | ~4 GPU-h | pre-registered composite |

## Relation to the hierarchy (v1.8+)

This synthesis lives entirely at the **operative** level and composes with the layered
design: g_tac (E5) conditions the denoise (upgrading cond_route's 3-way class to a
geometric goal), the tactical layer's own fan-and-select repeats the same pattern one
level up, and the imagination field keeps feeding spatial belief into the cost (occlusion
-aware distance-keeping). Nothing here competes with the hierarchy — it is the hierarchy's
bottom layer done right.
