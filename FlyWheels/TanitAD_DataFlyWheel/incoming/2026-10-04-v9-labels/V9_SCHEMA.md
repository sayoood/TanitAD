# v9 label release — schema (`tanitad.v9_labels/1`)

Files per split: `v9_labels_<split>.npz` (numpy, compressed) + `v9_labels_<split>.manifest.json` (sources + md5s,
builder md5s + base commit, every literal, vocabularies, policies, `npz_md5`). Builder:
`stack/scripts/build_v9_labels.py`; reader: `stack/tanitad/data/v9_labels.py`. Definitions: `SPEC.md` (+ addenda).

**Roles:** KEY · INPUT (fed to the model at train AND inference) · TARGET (a loss target) · DIAG (diagnostic; ⛔ never
an input, never a target unless a SPEC says so). Units: s = seconds relative to NOW unless "raw"; m; m/s; ° (left +).
NaN = undefined for this row. Class ids: v9 (`LAT9`, `LON9`) and the frozen v7 ids (`LAT7`, `LON7`) in the manifest.
Bit masks: bit i set ⇔ class / token i (`GOAL22` order for goals). -100 = IGNORE (or a PARTIAL label when the
matching `*_allowed` mask is non-zero).

## Clip table (`clip__*`, one entry per clip)
| field | dtype | role | definition |
|---|---|---|---|
| `sid` | int64 | KEY | `stable_episode_id(clip_id)` — the trainer's `ep.episode_id` |
| `sha12` | str | KEY | `sha256(clip_id)[:12]` (no UUID is stored) |
| `row0`, `n_rows`, `k0` | int64 | KEY | the clip's rows are `[row0, row0 + n_rows)`; row j holds raw row `k = k0 + j` (k0 = 2) |
| `grid_start_s`, `dt_s`, `clock_src` | f64, f64, int | KEY | the trainer's clock: `now_s = grid_start_s + k·dt_s`; src 0 sidecar / 1 pose_dt / 2 nominal |
| `has_record`, `has_lead`, `has_lc` | int | DIAG | v8 record (VLM), agent join, SAM3 lane offsets present |
| `n_turns`, `n_announced`, `suppressed` | int | DIAG | whole-recording turn segments, announced ones, clip-level suppression applied |

## Per-row fields (`row__*`)
| field | dtype | unit | role | definition (SPEC §) |
|---|---|---|---|---|
| `k` | int16 | — | KEY | raw row; a window with start t has NOW at k = t + 9 |
| `now_s` | f64 | s raw | KEY | the trainer's NOW for this row |
| `h_obs_s` | f64 | s | DIAG/mask | observed horizon (§2.4): min(8, first gap > 1 s, log end) − NOW |
| `max_gap8_s` | f64 | s | DIAG | largest log gap in [NOW, NOW+8] |
| `reversing` | int8 | — | mask | ≥ 0.3 s of reversing in [NOW, NOW+min(h,8)] ⇒ every action / geometry-goal cell IGNORE (post-validation fix) |
| `v_now_ms` | f64 | m/s | DIAG | speed at NOW (the trainer already has v0) |
| `lat_cls_a`, `lat_cls_b` | int16 | v9 id | TARGET | lateral action over [NOW+2, NOW+8]: variant a (junction, DEFAULT), b (heading) (§3.2) |
| `lat_allowed_a`, `lat_allowed_b` | uint8 | bits/LAT9 | TARGET | partial-label mask (§3.6); 0 = IGNORE |
| `lat_v7id_a`, `lat_v7id_b`, `lat_allowed_v7_a`, `lat_allowed_v7_b` | int16, uint8 | v7 | TARGET | the same in the frozen v7 vocabulary (D-WPA-2: WP-B uses these) |
| `lat_theta_deg` | f64 | ° | TARGET (constraint) | largest heading excursion in the observed band, from NOW+2 |
| `curve` | int8 | — | DIAG | \|Θ\| ≥ 30° but not a junction turn (variant a calls it LANE_KEEP) |
| `lat_seg_found`, `lat_gap_affected` | int8 | — | mask | a dominant turn segment exists / its timing is interpolated over a gap |
| `turn_t_start_s`, `turn_t_end_s` | f64 | s | TARGET (constraint) | dominant segment start / end relative to NOW (may be < 2 or > 8) |
| `turn_in_progress`, `turn_is_turn` | int8 | — | DIAG | start before NOW+2 / builder `is_turn` |
| `turn_d_start_m`, `turn_len_m` | f64 | m | TARGET (constraint) | arc NOW → segment start (0 if in progress); segment arc length |
| `turn_dyaw_deg`, `turn_r_arc_m`, `turn_vmin_ms` | f64 | °, m, m/s | TARGET (constraint) | segment Δψ (left +), arc radius, min speed |
| `turn_exit_x_m`, `turn_exit_y_m`, `turn_exit_psi_deg` | f64 | m, m, ° | TARGET (goal pose) | ego pose at the segment end, NOW frame (x fwd, y left) |
| `lc_measurable` | int8 | — | mask | 1 = LC usable as a label; **0 everywhere in this release** (V5 failed, `--lc-labels off`) |
| `lc_measurable_raw`, `lc_frac` | int8, f64 | — | DIAG | the SAM3 measurability before the V5 switch; share of band frames with a valid lane position |
| `lc_t_cross_s`, `lc_d_cross_m`, `lc_side` | f64, f64, int8 | s, m, ±1 | DIAG | first SAM3 lane-line crossing in the band (time, arc, side L +1 / R −1) |
| `exc_lat_m` | f64 | m | DIAG | D2-style curvature-detrended lateral excursion in the band (signed, left +); opens LC in the partial label at ≥ 2 m |
| `lon_cls`, `lon_allowed`, `lon_v7id`, `lon_allowed_v7` | int16, uint8 | ids/bits | TARGET | longitudinal action (§3.5) + partial mask (FOLLOW undetermined ⇒ {class, FOLLOW}) |
| `lon_cls_computed` | int16 | v9 id | DIAG | the class before the FOLLOW partial step |
| `lon_v_target_ms`, `lon_t_reach_s`, `lon_d_reach_m`, `lon_t_start_s`, `lon_truncated` | f64 / int8 | m/s, s, m, s | TARGET (constraint) | target speed, when / where it is reached, when the change starts; extremum at the band edge |
| `stop_x_m`, `stop_y_m`, `stop_dur_s` | f64 | m, m, s | TARGET (goal) | STOP point (NOW frame); time stopped (NaN past the log) |
| `lead_valid`, `lead_gap_m`, `lead_tg_s`, `lead_gap_min_m`, `lead_tg_min_s`, `lead_cov_s`, `lead_follow_frac` | int8 / f64 | m, s | TARGET (distance keeping) / DIAG | in-path lead (1.75 m corridor along the ego path, ≤ 100 m): bumper gap and time gap at the first band frame with data, minima, agent coverage, following share |
| `v_a_ms`, `v_end_ms`, `v_lo_ms`, `v_hi_ms` | f64 | m/s | TARGET (SPEED goal) | v(NOW+2), v(NOW+8), min / max over the band |
| `goal_y`, `goal_w` | uint32 | bits/GOAL22 | TARGET | 22-token goal set per row (§4); w = 0 ⇒ IGNORE; `SPEED_BAND` always w = 0 |
| `goal_lc_prov` | int8 | — | DIAG | LANE_CHANGE goal provenance: 0 none, 1 sam3, 2 vlm (sam3 never fires while `lc_measurable` = 0) |
| `vlm_frame`, `vlm_has_cot`, `vlm_tl_probe_negative`, `vlm_lat_evidence` | int8 | — | DIAG | NOW within ±2 s of NOW* = 3.1 s; a CoT exists; TL probe "no light visible"; D2 excursion ≥ 0.3 m in the VLM window |
| `red_propagated` | int8 | — | DIAG | RED positive by the stop-episode propagation (§4.3) |
| `nav_token` | f64→int | {0,1,2} | **INPUT** | FOLLOW / TURN_L / TURN_R — spatial rule `d_next ≤ max(30 m, 6 s·v_now)` or in progress (§5.2) |
| `nav_side_next`, `nav_d_next_m`, `nav_d_end_m`, `nav_dyaw_next_deg`, `nav_args_valid`, `nav_lookahead_m` | f64 | ±1, m, m, °, —, m | **INPUT** | next announced turn's side, distance to its start / end, angle; args valid; recorded arc ahead |
| `nav_in_progress`, `nav_gap_affected`, `nav_d_ann_m` | f64 | —, —, m | DIAG | turn under way; timing interpolated; the announcement distance used |
| `nav_t_next_s`, `nav_token_ttime` | f64 | s / id | ⛔ DIAG | time to the turn and A5's time-rule token — they encode the ego's FUTURE speed; never inputs |
| `s_now` | f64 | m | DIAG | arc position of NOW on the whole-recording path |
| `rc_{A30,A50,A80,B}_{x,y,psi,valid}` | f64 | m, m, °, — | **INPUT** (one variant) | route checkpoint (σ = 8 m path) in the NOW frame (§6); the training input adds noise (INTEGRATION.md) |
| `rc_b_L_m`, `rc_b_kind` | f64, int8 | m, code | DIAG | RC-B arc distance; 0 turn_end / 1 clamped_max / 2 clamped_min / 3 no_turn |
| `rcH_{A30,A50,A80,B}_{x,y,psi,valid}` | f64 | m, m, °, — | ⛔ DIAG | the σ = 25 m road-level route — the leak REFERENCE (S2-A2); never an input |
| `speed_n2_kmh`, `speed_n3` | int16, int8 | km/h, code | INPUT (optional, D-WPA-5) | D4 N2 past-20-s max snapped up (urban floor 50); N3 0 urban / 1 rural / 2 motorway |
