# INTEGRATION — the v9 labels, nav and route checkpoint as WP-B (Architecture & Inference) receives them

*WP-A (Data FlyWheel) · 2026-10-04. Binding for WP-B and the EvalFlyWheel NavSim bridge. Definitions: `SPEC.md` +
addenda; field list: `V9_SCHEMA.md`; validation: `RESULT.md`. ⛔ WP-A does not edit `refc_v3_train.py` or model code;
this document is the contract they wire against.*

## 1. Release files

| split | file | rows / clips | md5 | where |
|---|---|---|---|---|
| train | `v9_labels_train.npz` + `.manifest.json` | 869,278 / 4,369 | see `LANDING_READY_WPA.txt ## WPA-S3` | Thor `/home/nvidia/refcv8_v9labels/release/`; dev box `D:/Projects/TanitAD-artifacts/v9labels/` |
| eval139 | `v9_labels_eval139.npz` + `.manifest.json` | 27,664 / 139 | idem | idem |

One row per cached camera frame (a superset of every dataset window). Not in git (size); the manifest carries every
source md5, the builder md5s and base commit `50efa52`, every literal and the policies.

## 2. Loading and the join (exact API: `stack/tanitad/data/v9_labels.py`)

```python
from tanitad.data import v9_labels as V9
rel = V9.load_v9_release(npz_path, expect_md5=<md5 from the landing record>)      # immutable, picklable
r   = V9.row_for_now(rel, sid=int(ep.episode_id), k=t + w - 1 + n_stack - 1,      # = t + 9 for w 8, n_stack 3
                     now_s=self._now_s(ep, t))                                     # refuses |Δ| > 1e-6 s
T   = V9.window_targets(rel, r, lat_variant="a", ids="v7")                         # D-WPA-1 / D-WPA-2 defaults
I   = V9.window_inputs(rel, r, rc_variant="A50")                                   # tentative (§5)
```
* ⛔ **No module-level mutable state** in the reader; hold the `V9Release` on the dataset object. A worker receives it
  by pickle with its arrays read-only (pinned: `stack/tests/test_v9_labels.py`).
* The join is exact on all trainer windows: 746,946 / 746,946 train and 23,772 / 23,772 eval139, `now_s` identical
  to the trainer clock to 7e-15 s (V11).

## 3. Tensors per sample (WP-B adds the batch dim B)

### 3.1 Targets
| name | shape · dtype | content | mask / loss |
|---|---|---|---|
| `v9_lat` | [] int64 | lateral action, frozen v7 ids (variant a by default; b for the pre-registered arm) | −100 ⇒ use `v9_lat_allowed` |
| `v9_lat_allowed` | [8] bool | partial-label mask over v7 ids | loss `−log Σ_{c ∈ allowed} softmax(z)_c`; all-false ⇒ no loss |
| `v9_lon` · `v9_lon_allowed` | [] int64 · [8] bool | longitudinal action (STOP and DECELERATE → v7 `BRAKE_TO`; HOLD, CREEP, FOLLOW, ACCELERATE, KEEP → HOLD, CREEP, FOLLOW, ACCELERATE, CRUISE) | as lateral; FOLLOW undetermined ⇒ {class, FOLLOW} |
| `v9_goal_y` · `v9_goal_w` | [22] f32 · [22] f32 | the v7 goal tokens per frame | BCE × w; w = 0 ⇒ ignore; `SPEED_BAND` always w = 0 |
| `v9_lat_c` · mask | [12] f32 · [12] bool | `V9.LAT_CONSTRAINTS`: Θ °, turn t_start/t_end s, d_start/len m, Δψ °, R_arc m, exit x/y m, exit ψ °, LC t/d | mask = finite; supervise the turn fields only when the class is a TURN (or the turn is the band's dominant segment) |
| `v9_lon_c` · mask | [10] f32 · [10] bool | `V9.LON_CONSTRAINTS`: v_target m/s, t_reach s, d_reach m, t_start s, stop x/y m, lead gap m, time gap s, min gap m, min time gap s | mask = finite |
| `v9_speed_goal` · mask | [4] f32 · [4] bool | v(NOW+2), v(NOW+8), min, max over the band, m/s | mask = finite |

Recommended scales (WP-B decides): distances / 50 m, times / 8 s, angles / 90°, speeds / 15 m/s.

### 3.2 Inputs (admissible at inference; SPEC §8)
| name | shape · dtype | content | train-time treatment |
|---|---|---|---|
| `nav_cmd` | [] int64 | 0 FOLLOW · 1 TURN_L · 2 TURN_R (spatial rule, `d_next ≤ max(30 m, 6 s · v_now)` or in progress) | the same 3-way index the trainer's nav one-hot already reads |
| `nav_args` | [5] f32 | d_next m, d_end m, Δψ_next °, side_next (±1), lookahead m (≤ 1000) — zeros when `nav_args_valid` = 0, never "0 m" as a value | **nav-args dropout (D-WPA-7 ii): with p = 0.5 set `nav_args_known` = 0 and zero the args, KEEPING the token** (the NavSim command form) |
| `nav_args_known` | [] bool | `nav_args_valid` AND not dropped | — |
| `route_cp` | [3] f32 | x m, y m, ψ ° of the chosen RC variant, NOW frame | **RC dropout ≥ 0.3 (D-WPA-7 i): `route_cp_valid` = 0 and zeros**; **training noise** on the kept rows: along-track N(0, 2.0 m), lateral N(0, 0.75 m) in the route-tangent frame |
| `route_cp_valid` | [] bool | RC valid and not dropped | — |
| `v_limit_n2` (optional) | [] int64 | D4 N2 in km/h on {20, …, 130}; or N3 {0,1,2} | with an "unknown" row on ~0.45 of windows (X3) |

⛔ Never inputs: `nav_t_next_s`, `nav_token_ttime` (they encode future speed — measured leak +0.089 linear / +0.267
trees vs +0.039 / +0.152 for the spatial nav, RESULT §S2.2), `rcH_*` (the leak reference), every TARGET above, any
VLM token (situation outputs, PI 2026-08-03).

## 4. The route-checkpoint variant (status)
Every variant ships (`rc_A30/A50/A80/B`). **RC-A50 with the training noise above is CONFIRMED** by the registered
E2′/E3′ (`SPEC_ADDENDUM_S2A2.md`; RESULT §S2.7): under a certified nonlinear instrument the noised RC-A50 carries
LESS future-speed information than the road-level route (Δ −0.035, bar ≤ 0.01) and passes both lateral-leak bars
(straight-support p90 0.32 / 0.34 m; lane-change-scale median 0.14 / 0.16 m, train / eval). RC-B passes too (fallback).
⛔ **The noise is load-bearing:** the CLEAN RC-A50 carries +0.021 more than the road-level route — feed it WITH the
noise at training time, and never the clean point. ⚠️ This rests on the Master Mind's provisional ruling (road-geometry
speed information is admissible; PI confirmation pending). Optimism stamp on every result: the checkpoint is the ego's own future path (smoothed)
— **optimistic on PhysicalAI by construction**, exactly like the nav.

## 5. NavSim — what a legal agent receives, the mapping, the two reporting rows (Master Mind, 2026-10-04)
* MEASURED (S1, local devkit): NAVSIM's `AgentInput` is `ego_statuses, cameras, lidars` only
  (`navsim/common/dataclasses.py:150-155`); the route centreline exists only in the EVAL-side metric cache
  (`planning/metric_caching/metric_cache.py:43`, `centerline: PDMPath`). ⇒ a leaderboard-legal agent receives
  **neither the route checkpoint nor a turn distance — only NavSim's bare `driving_command`**.
* **Mapping** (NavSim `driving_command` is a 4-dim one-hot `(left, forward, right, unknown)` derived from the route
  point 20 m ahead, left iff y ≥ +2 m — INHERITED, `taniteval/adapters/navsim.py:62-63, 1077`):

  | NavSim | our `nav_cmd` | `nav_args_known` | `route_cp_valid` |
  |---|---|---|---|
  | left | 1 TURN_L | 0 | 0 |
  | forward | 0 FOLLOW | 0 | 0 |
  | right | 2 TURN_R | 0 | 0 |
  | unknown | 0 FOLLOW | 0 | 0 |

  ⚠️ **A DELIBERATE MISMATCH, stated on every NavSim number:** NavSim's command also fires on CURVES (any |y| ≥ 2 m at
  20 m) and carries no distance; ours is announced JUNCTION turns only, with distance, time-localised. A model trained
  on ours meets "left" on a bend out of distribution — D6 measured premature turns on ~59 % of 724
  command-but-straight navhard scenes (INHERITED, Master Mind 2026-10-04, D6). The nav-args dropout trains the
  token-only regime; the curve-firing difference remains.
* **Two reporting rows for every NavSim number:**
  1. **PRIVILEGED** (diagnostic only, ⛔ never submitted to the leaderboard): `nav_cmd` re-derived as OUR announced
     token from the scene route's lane connectors (a junction connector whose heading change meets `is_turn`'s 15°),
     `nav_args` from the route geometry (distance along the route centreline), `route_cp` = the centreline point at
     the same arc L (or the end of the next announced connector, clamped 20–80 m), from the metric-cache route.
  2. **LEGAL** (the leaderboard form): the bare command via the table above; `nav_args_known` = 0; `route_cp_valid` = 0.
  The gap between the rows is the price of the privileged route; a model that collapses on row 2 has not learned the
  dropout regimes.

## 6. The v7_labels fix (lands with WP-A Stage 4; `code/fix/stack/tanitad/data/v7_labels.py`, base blob `86f0c46e` → `abb1f64c`)
* `V7Label.goal_geometry_tokens` / `V7Label.goal_cot_tokens` — `frozenset` of the blob the label was loaded from (set by
  `load_v7_labels`; `None` only for hand-built labels, which use the legacy mirror).
* `LabelManifest.goal_geometry_tokens` / `goal_cot_tokens` — sorted tuples; `to_dict()` now emits both (the run's
  `config.json` states the policy the workers actually used).
* `tactical_goal_targets(label, …)` reads the LABEL's policy (private `_geometry_tokens_for`); `goal_supervision_census`,
  `TacGoalEmitter.provenance()` (new attribute `TacGoalEmitter.geometry_tokens`) and `cot_backed_tokens(labels)` read
  the labels' policy and REFUSE labels mixing two policies. The module globals remain a last-load legacy mirror only.
* **WP-B owns the trainer-side test:** after `train()` loads the TRAIN and then the EVAL label blob, assert
  `train_emitter.geometry_tokens == frozenset(train_manifest.goal_geometry_tokens)` and that the per-window `(y, w)` the
  DataLoader worker produces for a train clip equals the train policy's — through a pickled `V3Dataset` (the worker
  path). WP-A's own pins: `stack/tests/test_v9_labels_v7_policy_isolation.py` (6 tests; RED on the unfixed module).

## 7. The corpus manifest (`raw/refcv8_corpus_manifest.json`)
* **MASK (apply):** `mask_boxes` = (sha12, raw frame k, track_id) of every agent / box3d box whose centre lies in the ego
  footprint (x ∈ [−1, 4] m, |y| < 1 m): 693 boxes in 18 TRAIN clips (reproduces D3), + 375 in clips outside both
  corpora. Remove those boxes from the agent / box3d targets at that frame (k = the trainer's join key).
* **DROP (PI decision 8; Master Mind default = KEEP all clips, mask only):** `drop_list` = 8 train clips (6 recording
  partners of eval clips + 1 of each duplicate-video pair); `drop_list_tierA_only` = 5. eval139 is unchanged.

## 8. What the labels can and cannot carry (from the validation — RESULT.md §S3)
* LANE_CHANGE classes are **never positive** in this release (`lc_measurable` = 0; V5 failed): lateral is effectively
  3-way + partial {LANE_KEEP, LANE_CHANGE_x} where a ≥ 2 m excursion makes a lane change possible.
* `reversing` windows (≥ 0.3 s reversing within the horizon) carry no action / geometry-goal label.
* Absence claims (LANE_KEEP, KEEP, HOLD, CREEP, no-STOP, FOLLOW_LANE) need the full 8-s band (V7).
* The nav announces EARLY on decelerating approaches (spatial rule); 22.2 % (train) / 30.4 % (eval) of turn-commanded
  windows have the turn starting > 6 s later — by design (D-WPA-4).
