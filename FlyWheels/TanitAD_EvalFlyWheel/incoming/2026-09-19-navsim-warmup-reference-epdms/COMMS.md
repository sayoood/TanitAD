# COMMS — E1

## Decisions received
* **2026-09-19 ~11:30, orchestrator (PI authorised the navhard download in chat):** after warmup +
  `CACHE_DONE.json`, price navhard_two_stage from the warmup per-scene timings; if ≤ ~6 h CPU at the
  worker cap, build `C:/Users/Admin/navsim/exp/metric_cache_navhard_two_stage` (+ `CACHE_DONE.json`,
  same schema) and score `constant_velocity_agent` + `human_agent` on the OFFICIAL protocol
  `EPDMS_v2_navhard_two_stage`; compare with banked published navhard reference numbers. Do not read
  `…/openscene/navhard_two_stage/` before its `EXTRACT_DONE.json` exists.
* **~11:50, orchestrator:** the navhard CV cross-check value is **11.4** ([N2] 2506.04218 **v3**
  Tab. 2, post-#151 fix; v2 pre-fix gave 10.9) — INHERITED via the 2026-09-13/-10/-15/-17 packages;
  re-read from the banked primary before relying on it. State which side of #151 our SHA sits on; a
  large miss is a harness defect to diagnose first. Budget: keep the turn lean (weekly usage 87 %).

## Decisions made by E1 (recorded, reversible)
* Runtime moved to a byte-verified C: mirror (PLAN.md, Deviation 1). The metric cache still lives at
  the canonical path; its entries record `map_root = C:/Users/Admin/navsim-crun/data/maps` (a
  sha256-verified copy), which the IDM policy re-opens at scoring time. Override:
  `traffic_agents_policy.reactive.map_root_override=<root>`.
* RAM guard policy changed after it aborted on another process's transient spike (PLAN.md).
* `ego_status_mlp_agent` skipped by rule — no checkpoint at two probed locations.

## Integration asks (escalated in the report headline, not left here)
* see RESULT.md §Integration asks
