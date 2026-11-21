# P6 D7: the PDM-side lane-rank fix WORKS, and it is deliberately NOT applied, because the twin bank's trajectories do not follow their rank-K lane

**Master Mind, 2026-10-16.** The fix was built by a Sonnet agent and verified on the pod through the real `cmd_build`. This package is the code plus the evidence. ⛔ **The code changes are here as `p6_d7_pdm_route_rank.patch` and `code/`. They are NOT applied to `refe/`.** Read why before applying them.

## What the fix does, and what it proves (MEASURED; `results/verify_final_k*.json`, `results/control_rank0.json`)
- `pdm_route_rank.py` wraps NAVSIM's `AbstractPDMPlanner._get_starting_lane`. With it, PDM-Closed starts from DriveRL's rank-K lane of the starting roadblock, using the same `_route_edge_anchor_score` ordering and rear-axle anchor.
- A `Pool(initializer=…)` installs the wrapper in EVERY spawn worker, and each job re-asserts it. Rank 0 installs nothing.

Results on 74 twins (51 / 14 / 9 at K = 1 / 2 / 3), built through the real build:
- The twin centreline equals PDM's own search from DriveRL's rank-K lane on **74/74**.
- The route digest differs from the logged route on **71/74**. In the other 3, PDM's own lane IS DriveRL's rank-1 lane.
- The clamp rate is 0.
- **Rank-0 control:** 12/12 identical modulo nuPlan `track_id`, with `score_plans` equal.
- **Mutations:**
  - with the initializer removed, the build fails loud;
  - with the initializer and the guard both removed (P6's original defect), checks (a) and (b) read 0/6.

## ⛔ Why it is NOT applied: the pre-registered semantic check (c) FAILED
Among twins whose banked target trajectory diverges ≥ 0.5 m, the trajectory is closer to the **twin** centreline than to the logged one on only **0.029 / 0.0 / 0.0** (K = 1 / 2 / 3; bar 0.75).
- ⇒ The lane-rank twins in P6's bank are perturbations of the LOGGED route, not alternative-route intents.
- ⇒ Scoring them on the rank-K route, which is what this fix enables, would pair a target on one route with PDM labels on another: contradictory supervision.
- Applied to P6's shipped tree, the fix would make D7's distinctness check PASS and put exactly those labels into training.

**P6 therefore runs SPEC_P6 §2 D7's declared fallback**: `P6_D7_FALLBACK=1` in `/workspace/ops/p6_start.env`, so twins train with the trajectory loss only. P6's shipped tree stays `c1dc5d3`.

The cause is upstream of the cache build. It is likely the teacher's per-step route rebuild (`_choose_route_edge` is wrapped only for `prev_edge is None`), but that is **UNVERIFIED**.

## When to apply this
Apply it only together with a twin bank whose trajectories pass check (c), i.e. a re-rolled twin bank with a route held fixed over the rollout. That is a PI decision; see `PI_DECISION_QUEUE.md`.

**Other residual risks the agent measured:**
- the twin centreline is under half the logged length on 15/51, 3/14 and 6/9 twins (dead-end lanes);
- `lane-distinct`'s 0.9 bar on 40 keys falsely refuses a correct build about 8.4 % of the time (it suggests 0.8);
- the cost is 3.1 s per twin per worker.

The agent's harness refused a report `.md`. This README and `HANDOFF.md` are the report.
