# The left-turn defect is in the GOAL FIELD, and the decomposition is exact

`raw/LEFT_TURN_GOAL_FIELD.md` — MEASURED 2026-09-06, T1, **ZERO GPU**, from the banked decisions
sidecars (`raw/turn_asymmetry.txt`). This sharpens `raw/TURN_ASYMMETRY_EXPLAINED.md` from *"the
goal term will not pay for a left turn"* to a statement about **which term** and **by how much**.

## 1. ⭐ The decomposition is exact, and a control proves it

The constant-velocity candidate has **`kappa` ≡ 0 by construction**. The curvature penalty is
`w · kappa²`, so **it contributes EXACTLY ZERO to `finecost_cv`, whatever `W_KAPPA` is.**

⛔ **That is not an argument, it is checked.** `med finecost_cv` per decoded goal class is
**bit-identical across three arms whose curvature weight differs by every mechanism available** —
scalar 7, scalar 15.11245, and the by-goal map:

| decoded goal | n | `wk7` (W=7) | `wk15` (W=15.11) | `gkappa` (by-goal) |
|---|---|---|---|---|
| `LANE_KEEP` | 18 | **0.9242** | **0.9242** | **0.9242** |
| `TURN_L` | 9 | **1.7030** | **1.7030** | **1.7030** |
| `TURN_R` | 13 | **0.7685** | **0.7685** | **0.7685** |

⇒ **`finecost_cv` is the goal term plus the terminal-speed term and NOTHING else.** The
`TURN_L` / `TURN_R` ratio of **2.22×** is therefore located entirely outside the curvature penalty,
and no setting of `W_KAPPA` can move it.

## 2. ⛔ What the magnitudes say — stated as a hypothesis, with its caveat

Under `ccos` the goal term is `1 − cos(z − z_ref, g − z_ref)`, which lives on **[0, 2]**: **0** =
the candidate points *at* the goal, **1** = orthogonal, **2** = points *away* from it.

| decoded goal | `finecost_cv` | `finecost_plan` (wk7) |
|---|---|---|
| `LANE_KEEP` | 0.9242 — near-orthogonal | — |
| ⛔ **`TURN_L`** | **1.7030** | ⛔ **1.9964** |
| ⭐ `TURN_R` | 0.7685 | ⭐ **0.0526** |

⇒ **HYPOTHESIS `H-CCOS-TURNL-FIELD`: on `TURN_L` windows the left-turn goal field points somewhere
no candidate reaches, and the best the planner finds is close to its opposite** — `finecost_plan`
**1.9964** sits at the top of the `ccos` range, while on `TURN_R` the same planner reaches
**0.0526**, i.e. near-perfect alignment. A 38× difference in achieved cost between two mirror-image
manoeuvre classes is not a tuning gap; it has the shape of a **sign or frame error in the
left-turn goal**.

⚠️ **THE CAVEAT THAT KEEPS THIS A HYPOTHESIS AND NOT A RESULT.** `finecost` is the **composed**
cost, so it carries `W_VEND = 64.2972` on the terminal-speed error as well as the goal term. A
`finecost` near 2.0 is **consistent with** anti-alignment but is **not proof of it**: a large
speed error could produce the same number. ⛔ **The decomposition in §1 is exact and the
attribution in §2 is not**, and they are reported separately for exactly that reason.

**The discriminating experiment, pre-registered here before it is run:** emit the goal term and the
`W_VEND` term as separate columns in the decisions sidecar (they are already computed separately
inside `_cost_chunk`), then re-read this table. If the goal term alone carries the 2.22×, the
hypothesis is supported; if `W_VEND` carries it, the defect is longitudinal and the left-turn
field is fine. **Both outcomes are committed, and neither is a claim yet.**

## 3. Why this matters more than the frontier point itself

`W_KAPPA = 7` is the best weight available, and §1 proves the weight is not where the left turn is
lost. The programme's lateral ceiling on this rig is therefore set by two things a weight cannot
touch:

1. **the goal field** for `TURN_L` (this file), and
2. **the goal VOCABULARY** — `max|kappa|` is pinned at `GOAL_KAPPA_TURN = 0.08` on seven arms while
   the recorded human reaches **0.1797**, because the decoded token's canonical profile is the
   planner's only curvature-carrying candidate.

## Scope

40 windows / 8 episodes, ckpt 21,109, T1 (self-action open loop). Medians over 9 / 13 / 18 windows,
quoted with their `n` and **without** intervals they do not have. ⚠️ `finecost` is a re-scoring;
selection uses the base cost and `baseline_won_frac` is 0.25 on these arms.
