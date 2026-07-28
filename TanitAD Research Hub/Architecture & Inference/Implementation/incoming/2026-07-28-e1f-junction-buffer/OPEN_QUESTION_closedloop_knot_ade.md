# 🔵 OPEN INSTRUMENT QUESTION — the closed-loop knot-ADE may be confounded by ego-state divergence

**Raised 2026-07-28.** **This is NOT a finding. It is a question with numbers attached and a named
resolution path.** No conclusion is drawn and nothing in the E1 chain's verdicts changes.

## 1. What was measured (MEASURED, from the four committed frontier artifacts)

`M1_lateral_split.closedloop_K185.ego.paired_delta_ft_minus_base`, step 4000 of each arm.
**Estimator: paired episode-cluster bootstrap** (`taniteval/ci.py`, B=2000), 43 clusters at K=185.

| arm | `ade_over_knots` | `cross_abs@2s` | `along_abs@2s` |
|---|---|---|---|
| **E1c** λ=1 full | **+0.1406** [+0.042, +0.274] ✅sep | **+0.2154** [+0.004, +0.482] ✅sep | +0.0643 [−0.088, +0.199] – |
| **E1e-A** λ=3 full | +0.0836 [+0.004, +0.177] ✅sep | +0.1181 [−0.049, +0.317] – | +0.0168 [−0.100, +0.136] – |
| **E1e-B** λ=8 full | +0.0217 [−0.029, +0.073] – | +0.0608 [−0.091, +0.224] – | −0.0114 [−0.097, +0.072] – |
| **E1f** λ=3 junction | +0.0510 [+0.018, +0.089] ✅sep | +0.0893 [+0.014, +0.175] ✅sep | +0.0259 [−0.056, +0.109] – |

**The surface reading:** inside the K=185 rollout, the fine-tuned arms' 2 s waypoint predictions
deviate from the expert's **MORE** than base does — separated for E1c and E1f — and the excess is
**lateral**, not longitudinal, in every arm. Meanwhile those same arms depart the corridor far **less**
(E1c: −0.4407).

Taken at face value this would say the CL-SFT arms are not becoming *more expert-like*; they are
becoming *corridor-conservative* — buying departure reduction with worse trajectory tracking.
**That would materially reframe the closed-loop programme, which is exactly why it is not asserted.**

## 2. 🔴 Why it cannot be asserted — a confound I could not rule out

**In a closed-loop rollout, an arm that departs the corridor less is BY DEFINITION at a different ego
state than base at the same step.** The paired comparison therefore scores the two arms at **different
positions along the episode**, and a larger knot-ADE may reflect *where the arm is* rather than *how
well it plans*.

What I did establish (MEASURED, reading `/workspace/e1b/e1a_horizon.py:218-299`):
- `gt2 = gt_ego_waypoints(poses, last, wp_steps=WP_STEPS)` — ground truth comes from the **recorded
  expert poses**, not from a re-simulation.
- `_cap_pred` stores `ego_ego[:, WP_IDX]`, the model's waypoints transformed by the rollout's own
  `oyaw`; `_cap_gt` stores `gt2`. Both are `[N,4,2]`, described as ego frame.

What I could **not** establish cheaply: whether those two ego frames remain mutually consistent once
the rollout has drifted from the recorded pose at index `last`. If they do not, part of the reported
excess is drift geometry rather than planning error.

⚠️ **Note on locating the source:** `gt2s` is defined in **`/workspace/e1b/e1a_horizon.py`** — the
capture copy bound via `e1a = EB.e1a` — **not** in `/workspace/e1a_e2a/e1a_horizon.py`, where a first
probe looked and found nothing. Absence at one path was not absence.

## 3. Resolution path (no GPU required)

1. Read `gt_ego_waypoints` and the rollout loop end-to-end and determine whether `gt2` is expressed in
   the **rollout ego frame** or the **recorded-pose frame** at index `last`.
2. If the frames diverge under drift, the metric is **not** a clean tracking measure in closed loop and
   should be marked NON-DECIDING for cross-arm comparison — the same treatment K=20 already carries.
3. If the frames are consistent, the surface reading in §1 stands and **deserves its own write-up**,
   since it changes what "closed-loop gain" has meant across four arms.

**Until (1) is done, no claim from §1 may be quoted in either direction.**

## 4. What is unaffected

- **The departure-rate results are untouched.** P1/P2 are computed from corridor departure, not from
  knot-ADE, and every verdict in E1c/E1d/E1e/E1f rests on those.
- **The open-loop lateral/longitudinal split is unaffected** — measured on static windows with no
  rollout, hence no divergence confound. See `OPENLOOP_LATERAL_LONGITUDINAL_SPLIT.md`.
