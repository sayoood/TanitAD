# Stage 1 — the goal input, read out of their released code (not out of the paper)

**Date:** 2026-09-20 · **Evidence class: MEASURED (ours)** — every statement below is read from
`C:/Users/Admin/dz/DriveZero/DriveRL/src/driverl/…` at the released commit, with file:line.
Where the paper and the code disagree, the code is quoted and the disagreement is named.

This matters because the PI's REFe idea rests on it: *use the published checkpoint for training
signals **with goal augmentation**.* That only produces diverse supervision if the teacher is
genuinely goal-conditioned. It is, and here is exactly how.

## 1 · The teacher consumes TWO goal points, and that is a training fact we did not have

`release/configs/driverl_teacher.yaml:12` — `goal_count_probs: [0.5, 0.5]`, and
`src/driverl/env/config.py:27` defines `num_goal_positions = len(goal_count_probs)`.

⇒ **the released teacher's goal input is 2 points, and during PPO the *effective* count was drawn
50 / 50 between one and two** (slots beyond the sampled count repeat the last goal —
`goal_position_utils.py:44`). A single-goal reproduction is therefore **not** the released recipe,
and `apply_goal_pair_mode` (`:50`) exists precisely to collapse the pair on purpose
(`duplicate_earlier` / `duplicate_later`), which is a ready-made ablation handle.

## 2 · The inference goal is a deterministic function of the route — so augmenting it is cheap

`route_goal_positions` (`goal_position_utils.py:512`), which our Stage-0 runs already exercise via
`DRIVERL_EVAL_ROUTE_GOAL_HORIZON_S=12.0`, `…MIN_SPEED_MPS=5.0`, `…PAIR_MODE=legacy`:

1. project ego onto the route polyline → `current_progress` (arc length);
2. `final = current_progress + max(‖v‖, min_speed_mps) · horizon_s`, clamped to route length;
3. place `num_goal_positions` goals **evenly spaced in progress** between current and `final`, with
   the last goal exactly at `final`.

⇒ at 10 m/s the far goal sits **120 m** ahead; **stationary it is still 60 m ahead**, because the
`min_speed_mps = 5.0` floor is a floor on the *lookahead*, not on the ego. That floor is the reason
a stopped car still has a meaningful goal, and it is a knob we get for free.

**Augmentation levers, in increasing cost:**

| lever | how | cost |
|---|---|---|
| goal **distance** | vary `horizon_s` / `min_speed_mps` | free — env vars, no map work |
| goal **pairing** | `goal_pair_mode ∈ {legacy, duplicate_earlier, duplicate_later}` | free |
| **route alternative** | swap the route polyline (nuPlan roadblock / lane-connector graph; their feature builder already consumes `route_roadblock_ids`) | map work, the real lever |
| **lane snapping** | `goal_snap_keep_prob < 1`, `goal_snap_k`, `goal_snap_max_distance` | released, see §3 |

## 3 · ⛔ Their docstring contradicts their implementation — and the docstring is the wrong one

`sample_valid_positions` (`goal_position_utils.py:190`) documents the snap draw as
*"weight proportional to `1 / distance` (closer lanes win)"*. **The implementation does not do
that.** The code computes

```
closeness = clamp_min(1 - lane_distances / goal_snap_max_distance, 0)
inv_dist  = closeness * closeness          # (1 - d/max)^2, zero outside the radius
```

and its own inline comment says this shape was chosen **to avoid the `1/dist` blow-up** that
concentrates probability on the nearest lane. The variable is still *named* `inv_dist` — a leftover
from the version the docstring describes.

⚠️ **Consequence for us:** our `REFE_PLAN.md` quoted Table A3's `[max(1 − d/20, 0)]²`. **The paper
and the code agree; the docstring is the stale artifact.** Anyone implementing from the docstring —
the most natural thing to read — would build a materially different sampler: at max = 20 m the
implemented weights for lanes at 0.1 / 3 / 8 / 10 m normalise to ≈ 0.42 / 0.31 / 0.15 / 0.11,
whereas `1/d` would put ≈ 0.94 on the 0.1 m lane and effectively delete the other three.
⭐ Same family as the programme's own `df` / `step_s` / cgroup traps: **prose beside code, quoted
instead of the code.** Read the implementation, then check the docstring against it — not the
reverse.

## 4 · Snapping is released but NOT enabled in the released checkpoint's runtime config

`grep -rn goal_snap release/configs/ src/driverl/nuplan/config.py` → **no hits**, so
`goal_snap_keep_prob` defaults to `None` and the branch at `goal_position_utils.py` is skipped
entirely. The released teacher therefore takes its goals from `route_goal_positions` alone.

⚠️ **State the layer.** This is the **inference runtime** config. It is **not** evidence that PPO
training ran without snapping — Table A3 describes snapping as a *training-time* augmentation, and
the training config is not in the release. The honest claim is: *the mechanism is released and
usable; the shipped inference config does not enable it; whether the teacher was trained with it is
undetermined by the artifact we hold.* ⛔ Do not upgrade this to "they did not use snapping".

## 5 · What Stage 1 now has to build, reduced by this reading

The plan's deliverable `augment_routes(scene) → [route_k]` **does not need a new sampler** — theirs
is released and correct. What is genuinely ours:

1. a **route-alternative generator** over the nuPlan roadblock graph (the one lever that is not a
   free knob), K ≈ 3–5 per frame;
2. a **teacher rollout per `(frame, route_k)`**, reusing the Stage-0 harness unchanged;
3. the **controls the plan already commits to**: `route_0` (the logged route) must reproduce the
   un-augmented rollout **exactly** — a discriminating control that must read a known value, not a
   plausible one — and the six released calculators score the **logged human** trajectory at a
   banked value before anything else is scored.

⚠️ The exactness of control (1) is testable and cheap, and it is the check that would catch a
silently-different goal pipeline. Run it before trusting a single augmented target.

---

# IMPLEMENTED AND VALIDATED — 2026-09-20 (`code/augment_routes.py`)

⭐ **Why this stage outranks the rest, in their own numbers** (Table 7 supervision ablation):
**human 93.92 · DriveRL teacher 93.61 · teacher + goal augmentation 94.41.** The teacher **alone
loses to plain imitation**; augmentation is worth **+0.80** and is the only reason to use an RL
teacher at all. It is not one element of REFe — it is the element that justifies the architecture.

## What was built, and why it is small

Reading their source rather than re-deriving geometry collapsed the work. Their route is built by
`_sample_route_points_from_map` (nuplan-devkit `driverl_runtime_map_features.py`), which per roadblock
picks **one** interior edge via `_choose_route_edge` — nearest lane at the first roadblock, then
connectivity + `_route_edge_continuity_score`. ⇒ **the augmentation seam is that single choice.**
Taking the **k-th nearest** lane at the first roadblock and letting *their* continuity rule carry it
forward yields a lateral alternative with **no new geometry code**, and is exactly Table A3's lane
semantics.

## Controls — both read their known value EXACTLY

| control | requirement | MEASURED |
|---|---|---|
| our rank-0 route == **their** `_sample_route_points_from_map` | identical | **max abs difference 0.000e+00**, on **8 of 8** scenarios |
| our rank-0 goals == the goals the policy **actually consumed at runtime** | identical | logged `(+20.1, −17.9) / (+35.6, −43.6)`; reconstructed `(+20.1, −17.95) / (+35.6, −43.64)` |

⭐ The second control is the stronger one and was not in the plan: it checks the whole chain
(roadblock ids → lane choice → stitch → anchor frame → `route_goal_positions`) against the
`goal_points` persisted in the simulation log, i.e. against what the **running policy** received. A
pipeline can match their polyline function and still feed the goal head something different; this
rules that out.

## Coverage over the 8 mini scenarios (`raw/augment_routes_census.txt`)

| scenario type | routes | lateral offsets (m) |
|---|---|---|
| following_lane_with_lead | 4 | 3.42, 3.81, 7.28 |
| near_multiple_vehicles | 4 | 3.54, 3.66, 7.01 |
| starting_left_turn | 4 | 7.42, 3.84, 7.28 |
| accelerating_at_traffic_light_without_lead | 3 | 3.85, 7.32 |
| changing_lane_to_left | 2 | 4.44 |
| starting_unprotected_cross_turn | 2 | 3.85 |
| starting_protected_noncross_turn | 1 | no parallel lane |
| stopping_at_stop_sign_with_lead | 1 | no parallel lane |

**mean 2.62 routes/scenario; 6 of 8 have at least one alternative; offsets 3.4–7.4 m** — one and two
lane widths against a ~3.5 m nuPlan lane, so these are genuine lane changes and not jitter.

⚠️ **A scenario with one route is a MAP FACT, not a failure** — there is no parallel lane at the
ego's roadblock. ⛔ **Branch-level alternatives** (a different roadblock *successor*) are a separate
family and are **NOT** implemented: measured on one route, 4 of the first 12 roadblocks have more
than one successor, so the family exists and is reachable. It is the obvious next increment.

## The link that makes it augmentation rather than geometry

Feeding each alternative through **their** `route_goal_positions` (horizon 12.0 s, min speed 5.0 m/s,
`num_goal_positions=2` from the released config) moves the goal by **4.51 m near / 4.43 m far**. An
alternative that moved the polyline but left the goal unchanged would be useless, because the goal is
what the teacher is conditioned on. This is asserted, not assumed.

## What remains before this produces training signal

1. **teacher rollout per `(frame, route_k)`** — the Stage-0 harness runs unchanged; the only new
   input is the route override.
2. ⛔ **the un-augmented rollout must reproduce exactly under route_0** — the same control one level
   up, at the *rollout* rather than the *route*. Not yet run.
3. the six released calculators scoring the **logged human** trajectory at a banked value first.

---

# ROLLOUT LOOP CLOSED — 2026-09-20 14:45 (`code/run_stage1_rollouts.sh`)

The route-level control passed at the geometry. This is the same control **at the rollout**, plus the
first augmented arm. Injection is via `code/route_lane_rank_patch.py` + a venv `sitecustomize.py`,
because their harness spawns its own python and exposes no route hook. **At rank 0 the wrapper is
not installed at all**, so the control arm is untouched code.

| control | requirement | MEASURED |
|---|---|---|
| rank 0 rollout == Stage 0 | identical | **0.971925 vs 0.971925**, all **8/8** per-scenario scores equal |
| run-to-run noise floor | — | **0.000e+00** (two launches 2.5 h apart, bit-identical) |

⇒ **every difference below is caused by the route override, with no inference-variance term to
subtract.**

## rank 0 vs rank 1 (route starts in the neighbouring lane)

| scenario type | rank 0 | rank 1 | delta | progress 0 → 1 |
|---|---|---|---|---|
| **changing_lane_to_left** | 87.46 | **97.67** | **+10.21** | **0.613 → 1.000** |
| starting_protected_noncross_turn | 99.23 | 99.74 | +0.51 | 1.000 → 1.000 |
| starting_left_turn | 96.61 | 97.06 | +0.45 | 0.956 → 0.964 |
| starting_unprotected_cross_turn | 94.61 | 95.00 | +0.39 | 0.934 → 0.955 |
| near_multiple_vehicles | 99.92 | 100.00 | +0.08 | 1.000 → 1.000 |
| stopping_at_stop_sign_with_lead | 100.00 | 100.00 | +0.00 | 1.000 → 1.000 |
| accelerating_at_traffic_light_without_lead | 99.74 | **98.66** | **−1.07** | 1.000 → 0.957 |
| following_lane_with_lead | 99.96 | **98.82** | **−1.14** | 1.000 → 1.000 |
| **suite** | 97.1925 | **98.3704** | **+1.1779** | |

**7 of 8 scenarios changed.** Every safety term stays perfect under the alternative route
(drivable area, at-fault collisions, comfort, driving direction all 1.0000), so the alternative
routes are legal, not shortcuts.

## ⛔ What this is NOT

⛔ **It is NOT "goal augmentation improves the teacher by +1.18".** The published **+0.80** is the
gain from **training a STUDENT on augmented targets** (Table 7). This measures something else
entirely: *the teacher's own closed-loop score when it is routed differently*. Conflating the two
would be the same category error as quoting a suite mean for a per-scenario property.

⭐ **What it IS, and it is the thing Stage 1 had to prove:** the chain
**alternative lane → different goal → materially different teacher rollout** is real and works
end to end. **Two scenarios got WORSE**, which is not a defect but the point — augmentation must
produce *diverse* `(goal, trajectory)` supervision, not uniformly better trajectories. A route
change that only ever improved things would be suspicious.

## ⭐ An independent confirmation of today's retraction

`changing_lane_to_left` is the scenario whose "stall" was retracted as a **non-reactive replay
artifact**. Routing the ego into the neighbouring lane takes its progress from **0.613 to 1.000**
— i.e. the trap is avoided by changing *lane*, exactly as the reactive-protocol control implied
(log-replay traffic ignores the ego and boxes in whoever occupies that lane). **Two independent
routes to the same mechanism**, one by changing the background-agent policy, one by changing the
ego's lane. Neither is a policy defect.

## Status

| Stage-1 element | state |
|---|---|
| route alternatives over the lane graph | ✅ built, control exact 8/8 |
| route → goal (their `route_goal_positions`) | ✅ goals move 4.4–4.5 m |
| goal → teacher rollout (injection) | ✅ closed, control exact, noise floor 0 |
| branch-level alternatives (different roadblock successor) | ⛔ not implemented — next increment |
| K>2 per scenario as a target bank | ⏳ mechanism ready; needs a bulk run |
