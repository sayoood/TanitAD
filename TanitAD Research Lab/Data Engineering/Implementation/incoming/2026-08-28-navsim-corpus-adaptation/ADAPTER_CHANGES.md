# NavSim eval adapter — required changes (commission deliverable 4 of 4)

**Date** 2026-08-29 · **Owner** DataFlyWheel (spec) → TanitEval owner (impl)
**Status** SPEC. The corpus is built and validated; none of this blocks it.

The corpus was deliberately shaped so the adapter carries **no geometry**. Every
change below is plumbing or bookkeeping — if an adapter change needs a camera
formula, the corpus build was wrong, not the adapter.

## 1. A dataset source that yields corpus frames (small)

`taniteval` reads PhysicalAI episode caches. NavSim needs one more source
yielding the same contract: `frames uint8 [T,H,W,3]` + `CanonicalFrame` + mask.
**Use `navsim_loader.load_scene` as-is** — it is the single place the frame
choice is made, it defaults to `rig_clean`, and it *asserts* the 100 %-observed
guarantee per scene rather than trusting it.

⛔ Do not add a resize. A checkpoint whose frame tag differs from the corpus's
must be REFUSED with the two tags printed, not silently interpolated.

## 2. T = 4, and it is HISTORY only

NavSim eval scenes carry `num_future_frames = 0` — 4 history frames at 2 Hz
(measured Δt 499.7–500.3 ms), ending at the decision point. Consequences:

* **Δt is 0.5 s, not the PhysicalAI cadence.** Any rollout horizon expressed in
  *steps* silently changes meaning here. Express horizons in **seconds** and
  derive steps from the measured Δt per corpus (this is the `HORIZON` derived-
  constant trap in a new costume — a constant that changes meaning when its
  input changes).
* There is **no future in the corpus** to teacher-force against, so a **T0
  number is not computable from this corpus alone**. NavSim scoring is
  closed-loop against its own simulator, i.e. the **T1** family. Any NavSim
  row in the registry carries a T1 stamp or it is incomplete.

## 3. Ego is evaluator-side; the model input stays vision-only

Load the sidecar with `navsim_loader.load_ego`. The adapter must **not** route
`vx/vy/ax/ay/ego_*` into a model input path. `driving_command` may be passed
**only** as an explicit, declared oracle-goal argument for an arm that trains
with one — never a default, never a hidden channel, and it must be recorded in
that arm's registry row.

## 4. Splits key on `log_name` — the column is already in the sidecar

No second join. Report intervals as an episode-cluster bootstrap over log
clusters and **print n = 7** beside them (see RESULT.md caveat).

## 5. The four metric families still bind

A NavSim eval reports LONGITUDINAL, LATERAL, TACTICAL and STRATEGIC alongside
ADE. What this corpus can and cannot feed, stated per family so no family is
silently dropped:

| family | source in this corpus | note |
|---|---|---|
| longitudinal | sidecar `vx/ax`; lead-agent headway from scene `annotations.boxes/velocity_3d` | annotations are present per frame |
| lateral | sidecar heading + `vy/ay`; curvature/yaw-rate derived at use time | derive from raw fields, never store derived |
| tactical | manoeuvre vs `driving_command` (oracle) and vs executed motion | command skew {685/86/45} — report n per class |
| strategic | `roadblock_ids` + `map_location`/`map_name` per frame; nuPlan maps are on the box | ⭐ the only corpus we hold with a real **lane graph** — PhysicalAI has none |

⭐ **Item 5's last row is the strategic opportunity.** PhysicalAI carries no map,
lane graph, junction or route signal (settled, five probes). NavSim scenes carry
`roadblock_ids` and a nuPlan map name, and `nuplan-maps-v1.1` is already on the
dev box. This is the first corpus in the programme where the **strategic layer
can be measured against ground truth** rather than argued about.

## 6. Two guards worth adding at the same time

* **Preflight import + corpus probe at startup.** An analysis-time import that
  fails after the rollout destroys a paid run (measured: `t1_eval.py` rolled
  both arms, 40 episodes, then died in `analyze()`).
* **Assert corpus content, not presence** — a frame bank that exists at the
  right size can still be zeros. `load_scene` already checks the mask; the
  adapter should also require a non-trivial pixel mean per scene.
