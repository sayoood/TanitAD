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

## 2026-09-20 — navhard resumption (after the overnight agent-free chain)
Received: `mirror` and `cache` OK (5,187 s, 1 worker, C:-only at
`C:/Users/Admin/navsim-crun/exp/metric_cache_navhard_two_stage`); `N1` FAILED rc 1 after 4,144 s.
Decisions made by E1, with their evidence:
1. **The rollout was NOT recoverable**: `compute_final_scores` never ran, so the dump hook never
   fired and no CSV exists (checked, absent). ⇒ re-run — and the wrapper now banks the per-token
   frame *before* the aggregation (`create_scene_aggregators` hook), so this can never cost the
   compute again.
2. **Root cause MEASURED** (`raw/navhard/diag/idm_assert_diag.json`): an IDM agent whose baseline is
   fully consumed has a **zero-length** `path_to_go`; its flat-cap buffer is an **EMPTY** polygon
   under the pinned shapely 2.0.7, so the agent cannot intersect its own path and
   `navsim_idm_agent_manager.py:84` asserts. 166/5,462 stage-2 tokens (3.04 %), 0/450 stage-1. One
   such NaN row then kills the whole run in the aggregator. It is NOT the warmup human mechanism
   (0 future frames) — only the downstream fragility is shared.
3. **Patch applied to the C: COPY only** (`code/patches/apply_idm_degenerate_path_patch.py`,
   diff + blobs in `raw/navhard/idm_degenerate_path_patch.*`): the ASSERT CONDITION alone, letting a
   degenerate-path agent fall through to the devkit's own "free road" branch. Verified both ways
   (`raw/navhard/diag/patchcheck_COMPARE.json`): 3/3 previously-failing tokens now score, 4/4 healthy
   tokens **bit-identical**, zero drift. ⛔ It is part of the column and is quoted with every navhard number.
4. **Cache content-verified by E1** (the chain had not run the verifier): 5,912 = 450 + 5,462, token
   sets equal the yaml, 0 scene-type mismatches ⇒ `CACHE_DONE.json` written (schema `e1-cache-done/1`).
5. Worker: `sequential` (one worker, as instructed) — identical chunking to a 1-worker pool, and it
   keeps the per-token hooks recording in the parent.

## Integration asks (escalated in the report headline, not left here)
* see RESULT.md §Integration asks; W1 (promoting E1 code into `taniteval/taniteval/bench/**`) should
  absorb: `navsim_win.py` (loader patch + RAM guard + pre-aggregation dump), `verify_cache.py`,
  `verify_controls.py`, `build_artifacts.py::navsim_gates` + `gate_mutations`, and the two patch
  files. E1 does not edit that tree.
  ⚠️ **2026-09-20 — that handoff is now one-way.** My restructure of `build_artifacts.py` deleted
  `short_row`, `navsim_gates` and `gate_mutations`, and with nothing committed there is no history to
  recover them from: **W1's promoted copies at `taniteval/taniteval/bench/navsim/artifacts.py:56,89,124`
  are the only surviving ones.** I am NOT restoring them — W1 marked them `ORPHANED_PROMOTIONS` with a
  control that goes RED on restoration, and two of the three were deleted precisely because they had
  gone stale (`navsim_gates` would now FAIL a correct artifact; `short_row` is superseded by the
  adapter's `stage=`). The deletion was right; the irreversibility was not, and it has the same
  single cause as ask 9 — staged, never committed.
