# P4 implementation choices -- fixed BEFORE any mask, gated pick or G-arm score exists

Written 2026-10-04T21:13Z (UTC) by the P4 executor. SPEC: `SPEC_P4_DRIVABLE_GATE.md` sha256 `f613a15088062726a3340a8dd92fd59b3cd9c08a3c8f99754d078e2f88291a5c`
(re-hashed at start: match). Nothing below changes a threshold, check point, horizon or bar; each item fixes a point the SPEC
leaves to the executor ("the executor documents the exact tensor and the frame transform with file:line"). Its sha256 is
recorded in `raw/p4_impl_choices.sha256` before the GPU export is queued.

## 1. The tensor
* Head: refcv7's ONLY map head, the 10 cm `map_hires` branch (`MapHiresBranch.forward`), output
  `out["perception"]["map_hires_logits"]`, shape `[1, 8, 1000, 600]`, classes
  `("nocls", "drivable", "lane", "crosswalk", "arrow", "edge", "hatched", "sidewalk")` -> **drivable = channel 1**.
  The 0.5 m map head does not exist in refcv7 (removed by A6; config.json `map_hires.the_map`); the 0.25 m tensor
  (`map_hires_bev`) is an encoder FEATURE map, not a class output, so it has no drivable class.
  - scoring tree (C:/Users/Admin/ev7nav = git archive 0c444082, the tree the official 50,400 scoring imported):
    `stack/tanitad/models/map_head_hires.py:610` (class), `:702` (return of the logits), `stack/tanitad/refs/refc_v3.py:1895`
    (logits ride out on the perception dict), `stack/tanitad/refs/refc.py:4884` (`out["perception"]`).
  - lander tip (agent/arch-inf-20260803 @ 7efc49f2): `map_head_hires.py:779` / `:871`, `refc_v3.py:2149`, `refc.py:5151`.
    The forward block `class BEVLiftProjectFirst` .. `def build_map_hires_branch` is byte-identical in both trees (CR stripped).
* **Probability = the calibrated posterior** `p_hat = softmax(z - ln w)[:, 1]`, `w` = the run's frozen class weights
  (`model._map_hires_class_weight`; config.json `map_hires.class_weights.weights`, sha256 `d70dec80...`).
  Why this and not the raw softmax: the head is trained with a class-weighted CE, so its raw softmax learns
  `q_c ∝ w_c P(c|x)` (`MapHiresConfig.decision_rule` docstring); the module's own calibrated posterior is
  `softmax(z - log w)` (`_decision_logw` / `decide_masks`, tip `map_head_hires.py:188-247`), and this checkpoint's
  declared decision rule is `prior_corrected` = argmax of the same quantity (config.json argv `--map-hires-decision-rule
  prior_corrected`). The raw-softmax drivable mask is exported beside it as a DIAGNOSTIC ONLY; it enters no arm and no bar.
* Threshold: a cell is drivable for the gate iff `p_hat_drivable >= 0.5`, evaluated in float32 on the GPU from the
  float32-cast logits and stored as an exact bitmask (plus a uint8 `round(255 p_hat)` copy for diagnostics).
* "The drivable class" is used literally: the single softmax channel 1. Cells the head assigns to a painted class
  (lane / crosswalk / arrow / hatched) carry little drivable mass by construction; the gate is NOT widened to a union of
  classes (that would be an adaptation). The share of failing check points whose prior-corrected argmax is a painted class
  is reported as a diagnostic.

## 2. Window and frame
* Extent (config.json `map_hires.extent`): `x in [0, 100) m` ahead, `y in [-30, 30) m`, cell 0.1 m.
  Row `i` <-> `x = (i + 0.5) 0.1`, column `j` <-> `y = -30 + (j + 0.5) 0.1` (`bev_raster._cell_centers`, ev7nav
  `stack/tanitad/data/bev_raster.py:115-120`; `semantic_map_gt_fine.MapExtent.grid`, `:219-226`).
* Frame: refcv6/7 RIG frame, +x forward, +y LEFT, origin on the rear axle (rig6.py:12-18 of the refcv6 suite); NavSim's ego
  frame has the same x/y axes and the same rear-axle origin (only z differs). The bridge applies the IDENTITY in x/y to
  the plan (`knots_to_navsim`, a time spline through the knots); the mask is put in the NavSim frame by the same identity.
  Lookup: `i = floor(x / 0.1)`, `j = floor((y + 30) / 0.1)`; a point is INSIDE the window iff `0 <= i < 1000` and
  `0 <= j < 600`; outside -> unchecked (passes), counted.
* Frame control (K-FRAME, pre-committed): agreement of the predicted drivable bit with the metric-cache GT drivable bit
  at the candidates' check points must be higher under this transform than under the y-mirrored transform.

## 3. Check points
* For every candidate: its 8 NavSim poses (`cands_poses` of the banked P1' fan = rear-axle `(x, y, heading)` in the t0
  ego frame at t = 0.5, 1.0, ..., 4.0 s), and per pose the footprint FRONT_LEFT, REAR_LEFT, REAR_RIGHT, FRONT_RIGHT and
  CENTER computed with the devkit's own `state_array_to_coords_array` and `get_pacifica_parameters()` (width 2.297,
  front 4.049, rear 1.127; the scorer's vehicle). 8 x 5 = 40 points per candidate.
* The PLAN poses are checked, not the devkit's LQR-tracked states (the gate is an inference-time filter; the tracker is the
  scorer's).

## 4. Re-ranking (the deployed rule)
* Deployed rule = `argmax over reach_keep of sel_score_v3` (`refcv7_bridge.derived_selection`, refc_v3.py:2285-2298 as cited
  there), first index on ties. Asserted to reproduce `sel_idx` on every scene before use.
* G1: `argmax over {reach_keep AND gate_pass} of sel_score_v3`; if that set is empty -> the deployed pick (`sel_idx`).
  (Gate-passing candidates the deployed rule masks out are NOT eligible: the deployed rule never selects them.)
* G0's plan = the official 50,400 seam poses (`step50400/bridge_navhard/seam_R7_A1.npz`); a changed pick's plan =
  `cands_poses[new_idx]` of the banked P1' fan (`raw/p1x_fan/fan_R7_A1.jsonl`), the plans P1' scored.

## 5. G1-DER and G-ORC
* G1-DER: the 5,912 tokens sorted lexicographically; `rng = numpy.random.default_rng(20261004)`; `perm = rng.permutation(5912)`,
  re-drawn from the SAME rng until `perm[k] != k` for every k; scene k uses the mask of scene `perm[k]`.
* G-ORC: GT drivable = the union of the metric cache's `drivable_area_map` polygons of the four layers the devkit's DAC
  uses (ROADBLOCK, INTERSECTION, DRIVABLE_AREA, CARPARK_AREA; `pdm_scorer._calculate_ego_area`), points moved to the
  global frame with the t0 rear-axle pose. Same check points and the SAME window rule as G1 (only the mask source
  differs); the all-points variant is reported as a sensitivity.

## 6. Scoring and aggregation
* Exact devkit per token (`navsim.evaluate.pdm_score.pdm_score` with the objects of `d6_rescore.build_objects`, the ones
  that reproduced 2,678 banked rows exactly), for EVERY G0 token (the full G0 frame is rebuilt, not copied) and every changed
  pick of G1 / G1-DER / G-ORC. Unchanged tokens of an arm reuse that arm's G0 row. P1' tier-C rows are used as a
  cross-check where the candidate coincides (must be equal).
* Aggregation: the devkit's `create_scene_aggregators` -> `compute_final_scores` -> `calculate_individual_mapping_scores`
  with the export's `reactive_all_mapping`, rows in the banked final frame's order. Official EPDMS = the combined score.
* Rates: DAC-zero / NC-zero = share of the 5,912 tokens with that sub-score exactly 0 (unweighted, both stages pooled);
  EP = mean `ego_progress` over the 5,912 tokens (the devkit's stage aggregates reported beside it).
* Estimator: the P2 estimator (`d6_p2_analyze.paired_boot`): paired log-cluster bootstrap, B 2000, seed 0, 2.5/97.5
  percentiles; EPDMS deltas on per-mapping-key contributions clustered by the key's orig-token log; rates on per-token
  deltas clustered by token log. `taniteval/adapters/navsim_ci.paired_log_cluster_bootstrap` reported beside it.
* Seed floor: `|EPDMS(R7_A1) - EPDMS(R7_A1_s1)|` and `|DAC0(A1) - DAC0(A1_s1)|` from the banked 50,400 frames, same tokens.
* Strata: `raw/spec_tokens_p1x_S1.txt`, `_S1b.txt`, `_S2.txt` (P1' rule sets), rest = all others.
