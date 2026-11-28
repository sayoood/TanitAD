# ROLLOUT_SCHEMA -- `rollout.npz` and the human reference, schema `refe-rollout-v1`

For the sim session (TanitScenarioFlyWheel). Scorer: `behaviour_score.py` (see `AMBIGUITIES.md` for open definitions). The reference emitter is `synth_rollouts.make()`; a worked example is `example/rollout.npz` + `example/human.npz` + `example/score.json`.

**File format.** One `.npz` per scenario run, written with `np.savez` (compressed is fine), loaded with `allow_pickle=False`: **no object arrays**. Strings are 0-d `<U` arrays, scalars are 0-d arrays. All floats `float64` (float32 is accepted and upcast). `T` = number of 20 Hz samples, `A` = number of agents (may be 0), `H` = number of signal heads (may be 0), `M` = number of model calls.

**Frames and units.** One right-handed **world frame** (the sim's local metric frame, any origin), SI units, angles in radians, yaw CCW from +x (may wrap; the loader unwraps). Rollout and human reference **must share the same world frame and the same clock origin**. `ego_x, ego_y` and `agent_x, agent_y` are **bounding-box centres**. If the sim pose is the rear axle, convert: `x_c = x_ra + d*cos(yaw)`, `d` = rear axle to box centre. `ego_v` and `ego_a` are along the ego heading (not world-axis components). Speeds `>= 0`.

**Clock.** `t[0]` is the start of the scenario window (`t0 - 2 s` in SCORECARD section 3.1), recommended `0.0`. `t` is strictly uniform with step `0.05 s +- 1e-6` (the loader refuses anything else). The 2.0 s warm-up is measured from `t[0]`. The scorer subsamples/interpolates to the 10 Hz metric grid (SCORECARD section 1).

## Required keys (rollout)
| key | dtype | shape | unit / meaning |
|---|---|---|---|
| `schema_version` | str | () | `"refe-rollout-v1"` |
| `scenario_id` | str | () | unique scenario id |
| `category` | str | () | one of `cruise, steady_lead, lead_braking, stop_go, merge_cutin, junction, traffic_light` |
| `synthetic` | bool | () | `True` for the junction / traffic-light slots ("synthetic until map lifting exists") |
| `t` | float64 | (T,) | s, 20 Hz |
| `ego_x`, `ego_y` | float64 | (T,) | m, box centre, world frame |
| `ego_yaw` | float64 | (T,) | rad |
| `ego_v` | float64 | (T,) | m/s, speed along heading, realised |
| `ego_a` | float64 | (T,) | m/s^2, longitudinal acceleration along heading, **realised** (what the vehicle did, not what was commanded) |
| `ego_steer` | float64 | (T,) | rad, front-wheel steer angle, + left (stored in the NAVSIM state; not scored) |

## Identity keys (G-ID, SCORECARD 3.2) -- strongly requested; `identity.complete` in the output is `False` without them
`checkpoint_sha256` str(64 hex) (), `dtype` str () (`"fp32"`/`"bf16"`: must be identical across compared arms), `seed` int64 (), `suite_tokens_sha256` str (64 hex) (), optional `log_id` str () (bootstrap cluster; default = `scenario_id`).

## Ego geometry (optional; defaults are the nuPlan Pacifica)
`ego_length` float64 () m, default 5.176; `ego_width` float64 () m, default 2.297. `ego_rear_axle_to_center` float64 () m, **optional**, default 1.461 (nuPlan Pacifica; A-10: NAVSIM's `is_agent_behind` is evaluated at the rear axle).

## Agents (padded array + mask)
Agents are the **logged tracks, replayed** (identical in the rollout and the human reference). Arrays are padded to `A` agents and `T` samples; **`agent_present` is the authoritative mask** and values where it is `False` are ignored (write NaN). Row order is irrelevant (the scorer sorts by id).
| key | dtype | shape | unit / meaning |
|---|---|---|---|
| `agent_id` | int64 | (A,) | unique, >= 0 |
| `agent_present` | bool | (A,T) | track exists at this sample |
| `agent_x`, `agent_y` | float64 | (A,T) | m, box centre |
| `agent_yaw` | float64 | (A,T) | rad |
| `agent_v` | float64 | (A,T) | m/s, scalar speed along the agent's heading |
| `agent_l`, `agent_w`, `agent_h` | float64 | (A,) | m, length (along yaw), width, height; constant per agent |
| `agent_cls` | int8 | (A,) | **optional** (A-2): `0` vehicle (default, only these can be a lead), `1` pedestrian, `2` cyclist, `3` static object. All classes count for collisions. If the key is absent the output carries `agent_cls_defaulted = true` (AMENDMENT_1 A-2) |

With `A = 0` write shape `(0,T)` / `(0,)` arrays (or omit the block).

## Scenario fields for the junction / traffic-light slots (omit or leave empty elsewhere)
| key | dtype | shape | meaning |
|---|---|---|---|
| `signal_state` | int8 | (T,H) | per head, per 20 Hz sample: `0` none/unknown, `1` red, `2` yellow, `3` green (an arrow head reports its own colour) |
| `signal_governs_ego` | bool | (H,) | heads that govern the ego's movement. Governing state per tick = most restrictive of those heads (red > yellow > green > none) |
| `stop_line_s` | float64 | () | m, **station of the stop line along the human path**: cumulative Euclidean length of the human positions at 20 Hz samples 0, 2, 4, ... (the 10 Hz grid) from sample 0. NaN = none. A station beyond the human's last position lies on the straight extension (A-6) |
| `conflict_zone_xy` | float64 | (K,2) | world-frame polygon vertices (K >= 3) of the conflict zone (protected-left or merge), else shape `(0,2)` |
| `conflict_agent_ids` | int64 | (C,) | agents that conflict with the ego in the zone; empty = every agent counts (A-13) |

## Model calls (about 2 Hz; same clock as `t`)
`call_t` (M,) s; `call_ax_fed` (M,) m/s^2 (the acceleration fed to the model); `call_pick` (M,) int32 (selected proposal index); `call_plan_v4` (M,) m/s (planned speed at +4 s); `call_plan_v0` (M,) m/s (planned speed at +0 s); `call_agg_score` (M,) float64. Only `call_plan_v4 / call_plan_v0` is used (held zero-order to 10 Hz, ratio defined where `v0 > 0.1`) for the closed-loop planned slow-down share beside rule a2; the rest are carried for G-REPLAY and analysis.

## Optional
`epdms` float64 () (scenario EPDMS on 0-1, passed through to the guard g); `epdms_terms_names` str (K,), `epdms_terms` float64 (K,) (NC, DAC, DDC, TLC, EP, TTC, LK, HC).

## NOT in the file (derived by the metric code)
lead id, gap, TTC, time gap, a_lead, free-road flag, arc length, cross-track, heading and curvature error, comfort share. Any sim-computed copy of these is ignored.

## Human reference (`human.npz`)
The logged ego as the same arrays on the same clock: **required** `t, ego_x, ego_y, ego_yaw, ego_v, ego_a, ego_steer`. **Optional** (fall back to the rollout's): the agent block, `signal_state`/`signal_governs_ego`, `stop_line_s`, `conflict_zone_xy`, `conflict_agent_ids`, `ego_length`, `ego_width`. May be shorter or longer than the rollout. It defines the reference path (corridor), `v_h(s)`, `kappa_h(s)`, heading reference, the human share for d1 and the human launch time for c4. The human log is scored by the same code (Gate H): every output rule carries `human_status`.

## Loader checks (a failure raises `SchemaError`, nothing is silently repaired)
all required keys present; `t` 1-D, >= 3 samples, uniform 0.05 s; ego arrays shape `(T,)` and finite; `category` in the list; agent arrays shaped `(A,T)` / `(A,)`, ids unique.

## Output (`behaviour_score.py score`)
One JSON per scenario (`sort_keys`, deterministic; no timestamps): `scenario_id, log_id, category, synthetic, tier_stamp (= "T2 (re-perception closed-loop sim; MM ruling 2026-11-07)"), agent_cls_defaulted, identity, epdms, rules{a1..f2: status, raw_status, human_status, value, threshold, n, human_value, details}, families, derived, params, inputs{sha256 of both files}`. `status` is `PASS | FAIL | INFORMATIONAL | NA`. `families.STRATEGIC` is the PAUSED statement; TACTICAL carries c1-c4 only (decision table: A-14). `--dump-ticks T.npz` additionally writes the per-tick derived series (`lead_id, gap, ttc, time_gap, free, cross_track, ...`) for audit.

```
python behaviour_score.py score --rollout rollout.npz --human human.npz --out scenario.score.json [--dump-ticks ticks.npz]
python behaviour_suite.py aggregate --dir <dir of *.score.json> --out suite.json [--md suite.md]
python behaviour_suite.py compare --cand <dir> --base <dir> --base-replicate <dir> --tier 1|2 [--pdms-cand c.npz --pdms-base b.npz] --out adoption.json
```
`pdms_*.npz` (condition 3): `token` str (N,), `log_id` str (N,), `pdms` float64 (N,) on 0-1, optional `weight` float64 (N,).

## AMENDMENT_3 / AMENDMENT_5 additions (2026-11-08; AMENDMENT_3.md sha256 886e4465...)
* Optional scalar `slot` str `()`, one of `J-UL, J-R, J-S, TL-G, TL-S`. Validated when present, copied to the score JSON as `slot` only when present (absent => every pre-A3 output is byte-identical). `behaviour_suite.py aggregate` adds a DESCRIPTIVE `per_slot` row per block when any scenario carries a slot; `category` pooling is unchanged and slots are never an adoption count.
* Optional scalar `human_ref_valid_until_s` float64 `()` **in `rollout.npz`** (not `human.npz`; the scorer reads it from the rollout and applies it to the human run too). Seconds from the first tick, computed by the sim on the 20 Hz grid: the first tick at which the human is more than 1.75 m from the route-lane centre. **Absent if the human never leaves the route lane.** Rules a1, a2, c4, e1, e2 are evaluated only on ticks with `t - t0 <= human_ref_valid_until_s`; later ticks are NA for them. Their `details` then carry `human_ref_valid_until_s` and `n_ticks_na_after_human_ref` (10 Hz ticks). Rules b, c1-c3, d, f and EPDMS keep the whole window. With the key absent, results are bit-identical to before. The score JSON echoes the key only when present.
* `stop_line_s`, `signal_state`, `signal_governs_ego`, `conflict_zone_xy`, `conflict_agent_ids` are computed per AMENDMENT_3 section 5 (selector `MAP_FIELDS_A3.json`).
* Scoring changes (CH-1..CH-3): free-road intervals exclude HELD ticks (red, before the line, within 40 m; latched to the first `v > 1.0` at or after a green); c1 is NA when the ego is stopped at the first clear tick (`v <= 1.0` inside a run of >= 1.0 s); c2 counts crossing EVENTS on red.
