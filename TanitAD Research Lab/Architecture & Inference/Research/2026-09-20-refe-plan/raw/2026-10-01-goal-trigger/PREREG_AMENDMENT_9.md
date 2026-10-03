# SPEC_NAVTEST Amendment 9 — the off-route nav goal, fixed by the privileged planner's route (pre-registration)

> **Registered (from `date`): see the `REGISTERED` line at the bottom.** It is written BEFORE any model output exists on
> the confirmation tokens below. The token sets were fixed by `goal_fix_census.py` (CPU, the planner's own `_goal_for`,
> no network). The baseline arm's seam (`refe_navtest_final.npz`) was written on 2026-10-02 before Amendment 9 existed;
> it was scored on 2026-10-04, and its numbers are not read per token before the confirmation arms are scored.

## 1. What is broken (MEASURED, exploratory; found on all navtest tokens, label-free)

- **475 of 12,146 navtest tokens** have a scenario route (`route_roadblock_ids`) that does not contain the ego's
  roadblock (`goal_census_navtest_full.json`: ego-to-route > 20 m).
  - REFe's goal path (`planner._goal_for` → `_route_with_lane_rank` → `route_goal_positions`) projects the ego onto
    that route at any distance, so the nav goal lands 100–480 m away. Example: token `134a3394c9b756d9` sits
    171 m from its route (`offroute_134a3394.png`).
- **NAVSIM's own privileged planner does not see this defect.** PDM-Closed corrects the route at iteration 0
  (`navsim/.../pdm_closed_planner.py:73` → `route_utils.py:96 route_roadblock_correction`).
  - When the ego's roadblock is off-route, a roadblock-graph BFS connects it to the route.
  - Its corrected `centerline` is what the metric cache stores, and what the scorer measures EP and DDC against.
  - REFe is therefore scored against a route it was never given.
- **CPU probe** (`pdm_route_goal_probe.json`). Each goal is read as a path [0, p1, p2], taken at the human's travelled
  arc length and compared with the human's own 4 s endpoint (m, mean / median / p90):

  | goal | 475 off-route tokens | 300 untriggered control tokens |
  |---|---|---|
  | current (raw route) | 8.65 / 5.60 / 20.8 | 1.81 / 0.56 / 5.08 |
  | **PDM-corrected route** | **1.62 / 0.77 / 4.28** | 1.70 / 0.61 / 5.13 |
  | straight along the heading | 2.57 / 1.07 / 8.57 | 2.88 / 1.11 / 8.56 |
  | constant-curvature arc | 2.95 / 0.83 / 9.18 | 3.15 / 0.91 / 9.39 |

  - On the off-route tokens the ego is a median 0.35 m from PDM's corrected centerline (max 6.7 m).
  - On the control tokens the corrected and the current goals agree (median 0.11 m apart).

## 2. The candidates (implemented in `refe/planner.py`, `goal_fix`, default None = bit-identical)

| arm | what changes | inputs | admissibility under the PI's 2026-10-01 ruling |
|---|---|---|---|
| **pdm_route** (PRIMARY) | Route ids are corrected by NAVSIM's `route_roadblock_correction` (vendored verbatim, `refe/pdm_route_correction.py`) before the unchanged goal derivation. | route ids, map roadblock topology, ego pose | Only the nav goal moves; no new input reaches the model. **The PI suggested "the output of the privileged planner"; deployment still needs the PI's explicit OK**, because the correction reads the roadblock graph. |
| **navgoal_straight** (MAP-FREE ALTERNATE) | A trigger that reads only the nav goal and v0: \|p2\| > 1.2 · max(\|v0\|, 5) · 12 s. When it fires, the goal is re-placed along the heading at the same arc lengths. | nav goal, v0 | Map-free: admissible by the ruling as stated. |
| navgoal_arc (reported) | Same trigger. The goal lies on the arc with curvature ω / max(\|v0\|, 1). | nav goal, v0, yaw rate | Map-free. |
| a8_lane (reference) | Amendment 8's lane-graph fallback. | lane graph + command | **Inadmissible** (lane geometry); reported as the known reference only. |

κ = 1.2 was chosen on the 1,123 selection tokens only (`goal_trigger_analysis.json`) and frozen. Its held-out
agreement with the route-distance trigger is F1 0.914 (TP 368, FP 15, FN 54).

## 3. Design

- **Snapshot:** `model_final.pt`, md5 `b54773f8c302b2ec8ff4c5971d1e1370`. Amendment 7's repair is ON and the rule is
  navsim_v1, in every arm.
- **Baseline (OFF):** the full-navtest seam `refe_navtest_final.npz`, which is the same snapshot and settings.
- **Confirmation tokens, per arm:** the tokens whose goal changes under that arm (census), MINUS the 1,123 selection
  tokens, MINUS the 36 tokens that fail the frame/time control (`frame_control_navtest_full.json`; a DB-vs-OpenScene
  pose disagreement in 4 logs). Counts are in §5.
- **Controls inside each GPU run:** 24 tokens whose goal does NOT change, drawn by seed 20260927 from outside the
  selection tokens. Their poses must equal the OFF seam rows exactly. That shows nothing but the goal differs between
  arms.
- **Estimator:** the paired log-cluster bootstrap (10,000 resamples, percentile 95 %, seed 20260927) of
  per-token PDMS(arm) − PDMS(OFF) over the confirmation tokens. It answers "another draw of episodes" only.
  - **PRIMARY read:** all confirmation tokens.
  - **FRESH-LOG read:** only the tokens in logs that hold no selection token.
- **Full-navtest effect (reported):** the same per-token deltas, summed and divided by 12,146. Every other token's
  delta is 0 by construction, and the controls test that.

## 4. Decision — both outcomes committed now, per gating arm (pdm_route; navgoal_straight)

- **ADOPT** iff BOTH reads' CI lower bounds are > 0 AND no longitudinal or lateral family component separates
  adversely.
  - For pdm_route, ADOPT makes it the candidate default. It is deployed only after the PI confirms admissibility.
  - If the PI rules pdm_route out, navgoal_straight is deployed if it reads ADOPT.
- **REFUTED** iff the PRIMARY CI upper bound is < 0.
- Otherwise **NOT PROVEN**.
- Validity gates, all before the statistic is read:
  - (a) every confirmation token's goal changes in the run's own recorded inputs;
  - (b) the regression control (`goal_fix=None` reproduces the pre-patch goals on all 12,146 tokens, max |Δ| = 0);
  - (c) the 24 controls reproduce the OFF seam bit for bit;
  - (d) every harness run PASSes with every token valid;
  - (e) the cross-check: pdm_route goals agree with NAVSIM's metric-cache centerline goals (reported quantiles).
- The four families are reported for each gating arm; strategic is UNAVAILABLE in NAVSIM by design.

## 5. Token sets (filled from `goal_fix_census.json` before any GPU forward)

| arm | goal changes (census) | of which selection | of which frame-fail | **confirmation** | logs | FRESH-LOG tokens / logs | L / S / R |
|---|---|---|---|---|---|---|---|
| pdm_route | 3386 | 316 | 25 | **3045** | 82 | 578 / 37 | 997 / 1493 / 555 |
| navgoal_straight | 433 | 50 | 0 | **383** | 26 | 244 / 20 | 117 / 236 / 30 |
| navgoal_arc | 433 | 50 | 0 | **383** | 26 | 244 / 20 | 117 / 236 / 30 |
| a8_lane | 475 | 53 | 0 | **422** | 32 | 279 / 26 | 112 / 266 / 44 |

- Census (`goal_fix_census.json`, all 12,146 tokens, 0 errors after a retry of one log that hit a transient sqlite disk I/O error):
  - regression `goal_fix=None` vs the pre-patch goals: n 12146, max |Δ| 0.0;
  - pdm_route changes the route ids on 3455 tokens and the goal on 3386 (958 by > 1 m).
    NAVSIM's correction also relinks unlinked route roadblocks (Fix 2) and cuts loops (Fix 3), so it is NOT confined to the 475 off-route tokens.
    The > 1 m stratum is reported beside the PRIMARY read (not gating).
- **Correction BEFORE any confirmation output (2026-10-04 ~01:25):**
  - §3's FRESH-LOG read ('tokens in logs that hold no selection token') is EMPTY by construction, because the 1,123 selection tokens touch all 136 navtest logs.
  - It is replaced by Amendment 8's own definition: the logs that hold no selection token **the arm changes** (where the defect was never seen during exploration).
  - Check: for the a8_lane arm this reproduces Amendment 8's registered FRESH-LOG set exactly (279 tokens / 26 logs).
- **Disclosed:** the OFF arm's AGGREGATE PDMS on the 475 off-route tokens (55.45) was read before registration (navtest readout). No arm's per-token score had been read.


REGISTERED 2026-10-04 01:25:16 -- before any Amendment 9 GPU forward exists.
