# WP-A SPEC — the v9 label release: per-frame tactical actions + goals with constraints, per-frame nav, route checkpoint

*TanitAD Data FlyWheel · 2026-10-04 · Stage 1 deliverable (definitions BEFORE any build). Brief: `BRIEF.md` in this
folder. Requirements: PI R8-1, R8-2, R8-3 (`Project Steering/PLAN_REFCV8.md` §0). Status: **DRAFT for the Master
Mind's review — nothing below has been built.** Every threshold is a literal with its physical justification; every
bar is committed here, before the number exists, with both outcomes.*

## 0. Stamps

* **Evidence classes.** `MEASURED (S1)` = measured in this stage, artifact in `raw/` (code in `code/`).
  `MEASURED (D1…D4, A5–A7)` = the refcv8 data-audit streams and the route package, cited with their file.
  `PUBLISHED` = a banked primary (`TanitAD Research Lab/Library/`). `ESTIMATED` = reasoning, no artifact.
  `INHERITED` = another package, not re-run.
* **Code tree.** Tip `50efa52` extracted to `C:/Users/Admin/r8_wpa/{stack,taniteval,tools}`; nothing on `G:` was read.
* **Data.** Train = the 4,369 clips / 746,946 windows refcv7 read (manifest md5 `3c9f8bc8…`); EVAL = eval139 (139 clips /
  23,772 windows, manifest md5 `34433232…`); clock sidecar md5 `78466f99…`; labels v8 train `b45377a1…` (4,572
  records) and eval `eefc38d1…` (147 records).

### 0.1 What I measured to write this SPEC (S1, CPU, dev box, nothing built)

| quantity | train | eval139 | artifact |
|---|---|---|---|
| clips with a per-clip egomotion log (`egomotion_alpamayo/<clip>.parquet`) | **4,369 / 4,369** | **139 / 139** | `raw/s1_clip_log_coverage.json` |
| log span (s): p5 / median / max | 30.93 / 139.70 / 141.25 | 39.79 / 140.26 / 140.39 | same |
| log rate: **100 Hz on raw [0, ~20] s**, then irregular; median rate after 20.5 s | 8.5 Hz (p5 2.7, p95 24.6) | 8.1 Hz | same + `code/s1_probe_*` |
| trainer WINDOWS whose log reaches NOW+8 s | 738,067 / 746,946 = **98.81 %** | 23,508 / 23,772 = **98.89 %** | `raw/s1_window_band_coverage.json` |
| … and largest log gap inside [NOW, NOW+8] ≤ 1.0 s | 710,887 = **95.17 %** | 22,577 = **94.97 %** | same |
| … ≤ 0.5 s | 703,188 = 94.14 % | 22,236 = 93.54 % | same |
| windows whose whole [NOW, NOW+8] lies in the 100 Hz part (NOW+8 ≤ 20 s) | 478,534 = 64.07 % | 15,238 = 64.10 % | same |

Two consequences shape the design: (i) **every frame has its 8-s future in the log on ~99 % of windows**, but ~36 % of
bands reach into the sparse part, so validity must be judged on log GAPS, not on presence (§2.3); (ii) a strict
"full 8-s band at ≤ 1.0 s gaps" rule sits **at** the 95 % coverage bar (95.17 / 94.97 %), so absence claims get a
measured truncation rule (§2.4) instead of a silently lowered bar.

Also MEASURED (S1, read from the local NAVSIM devkit `C:/Users/Admin/navsim-crun/devkit/navsim`): `AgentInput` carries
only `ego_statuses, cameras, lidars` (`common/dataclasses.py:150-155`); the route centreline exists only in the
EVAL-side metric cache (`planning/metric_caching/metric_cache.py:43`, `centerline: PDMPath`). This decides §6.4.

## 1. The PI's words → where each is met

| PI clause (2026-10-04, `BRIEF.md`) | section |
|---|---|
| "each camera frame should have corresponding tactical goals and actions" | §2.1 one row per cached 10 Hz frame, every clip, train + eval139 |
| "in the future time frame (2 to 8 seconds from the current reference frame)" | §2.2 band **B = [NOW+2 s, NOW+8 s]**, used by every action and goal |
| "with corresponding constraints like distance and time" | §3.3, §3.5, §4 — every action/goal carries start/end time, arc distance, and its own physical constraint |
| "not only one per clip" | §2.1 — the v8 one-record-per-clip anchor is not used for any geometry field; VLM fields are re-timed (§4.3) |
| "each frame should have a corresponding nav command as already defined" | §5 — the builder's own `is_turn` turn definition, over the whole recording, per frame, with distance to the turn |
| "a route checkpoint extracted from the ego trajectory … the next reference route point in the vehicle coordinate system" | §6 — RC-A / RC-B in the NOW vehicle frame (x forward, y left) |
| "this is not a label … training and inference way point in addition to the nav command" | §6, §8 — an INPUT channel with validity, never a loss target |
| "simulating the nav system of the car" | §6.4 the deployment analogue and the NavSim bridge contract |

## 2. Frames, clock, sources, validity

### 2.1 What a row is
* One row per **cached camera frame**: every provider row `p = 0 … T−1` of every clip in the train (4,369) and eval139
  (139) caches, i.e. ~869 k train + ~27.7 k eval rows (a superset of every training window's NOW).
* **Key: `(sid, k)`** — `sid = stable_episode_id(clip_id)` (blake2b-8 ≫ 1, `v2_dataset.py:69`; the trainer's
  `ep.episode_id`), `k = p + 2` = the RAW row (`_raw_offset = n_stack − 1 = 2`). A dataset window with start index `t`
  has NOW at raw row **`k = t + w − 1 + 2 = t + 9`** (w = 8).
* **NOW clock = the trainer's own:** `now_s = grid_start_s + k · dt_s` from the clip-clock sidecar; for the 22 train +
  3 eval clips the sidecar does not cover, exactly `refc_v3_train.py::_clock_for` (`clip_clock.pose_dt`, else the
  nominal dt, with `grid_start_s = 0.0`), and the row carries `clock_src ∈ {sidecar, pose_dt, nominal_dt}`. D1
  reproduced this clock bit-exactly against the run's own log (D1 COVERAGE §2 control C).
* Clip ids appear only as `sid` and `sha12 = sha256(clip_id)[:12]`. No UUID is written anywhere.

### 2.2 The band
`B(NOW) = [NOW+2.0 s, NOW+8.0 s]` (PI). The operative band [NOW, NOW+2) is NOT labelled by v9 (refcv7's kin3 core aux
already supervises 0–2 s densely, D4 USAGE T4). Quantities that need the state at the band start use
`v_a = v(NOW+2)`, `ψ_a = ψ(NOW+2)`, `s_a = s(NOW+2)`.

### 2.3 Ego source and sampling
* **Source:** the provider's per-clip egomotion log (`C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo/
  <clip>.parquet`; Thor holds the same store), read exactly as `tanitad/data/egomotion_source.py` reads it: raw time
  `= (timestamp − timestamp[0]) / 1e6` (the clip sits at offset 0 of the recording, verified there and by D2's clock
  check), speed `= ‖(vx, vy, vz)‖`, heading = quaternion yaw (ZYX), **unwrapped before interpolation**, position
  `(x, y)`. 100 Hz on [0, ~20] s, irregular 2.7–25 Hz to the recording end (S1).
* **Grid:** `τ_j = j · 0.1 s`, `j = −20 … +120` (NOW−2 s … NOW+12 s; margins for smoothing and look-ahead), linear
  interpolation of the native samples.
* **Arc length** `s(t)` = cumulative chord length of the native `(x, y)` samples (the builder's `_arc_to` rule).
* ⚠️ Poses lead the image by 0–34 ms (D3 X10, mean 0.19 m); the trainer's clock is the contract, so labels sit on the
  grid times, not on the camera timestamps. Stated, not corrected (a trainer-side item, X10).

### 2.4 Validity — judged on GAPS, and absence claims need an observed band
| literal | value | justification |
|---|---|---|
| `G_MAX_S` — largest admissible gap between consecutive log samples inside an interval used by a time-domain field | **1.0 s** | linear interpolation over ≤ 1 s keeps an event's timing within ±0.5 s (the timing tolerance in §9) and a chord's sagitta on an R = 15 m turn at 10 m/s ≤ 0.83 m; MEASURED coverage at 1.0 s: 95.17 % / 94.97 % (§0.1) vs 94.14 / 93.54 % at 0.5 s |
| `h_obs(NOW)` — the observed horizon | `min(8.0, t_gap − NOW, t_end − NOW)`, `t_gap` = start of the first gap > `G_MAX_S` after NOW, `t_end` = last log sample | a field never looks past what the log shows |
| `H_ABS_MIN` — the shortest observed horizon on which an ABSENCE claim (LANE_KEEP, KEEP, HOLD, "no stop", FOLLOW_LANE, nav FOLLOW) is admitted | **6.0 s** (≥ 4 s = 2/3 of the 6-s band observed) | D4 admitted ≥ 4 s of future; pre-registered TRUNCATION TEST (§9 V7): on full-band frames, truncating to 6.0 s must change ≤ 5 % of lateral and of longitudinal labels, else `H_ABS_MIN = 8.0` |
| a POSITIVE event (TURN, LC, STOP, DECEL, ACCEL) | admitted when the event lies at τ ≤ `h_obs` | an observed event does not depend on what follows it |
| a row with `h_obs < 3.0 s` | every action/goal cell IGNORE | less than 1 s of band observed |

Every row carries `h_obs_s` and `max_gap_s` so a consumer can tighten the rule without a rebuild.

## 3. Per-frame tactical ACTIONS over B (R8-1)

Actions describe **what the ego DOES** in B. Two families, each a single class per frame (for a CE head) PLUS an
`allowed` bit-mask for partial labels (§3.6), PLUS constraint fields. Literal thresholds below; the v7 builder literals
are reused where the PI already accepted them, and say so.

### 3.1 Turn segments (one primitive shared by actions §3.2 and nav §5)
Detected on the WHOLE-RECORDING 10 Hz track with the builder's own literals (`s2_geom_emit_v7.py:59-62, 87-89`),
because the PI asked for nav "as already defined":

| literal | value | justification |
|---|---|---|
| yaw-rate smoothing | 0.5 s centred moving average | builder `manoeuvre_sequence` |
| sustained-yaw threshold | \|ω̄\| ≥ **6 °/s** for ≥ **0.8 s** | builder; a lane-keeping vehicle stays well below 6 °/s, a junction turn runs 10–30 °/s |
| segment net heading | \|Δψ\| ≥ **20°** | builder `MIN_TURN_DEG` |
| radius | **R_arc = arc length / \|Δψ\|** over the segment — NOT 1/κ_peak | builder's MEASURED lesson (`43bbcbf9`: peak curvature read 40.1 m on a 144 m S-bend) |
| `is_turn` (a JUNCTION turn) | \|Δψ\| ≥ **15°** ∧ R_arc ≤ **140 m** ∧ v_min ≤ **8.0 m/s** | builder, calibrated against Alpamayo's own turn text (precision 70.6 %, recall 61.3 %, INHERITED from the builder docstring); physics: v ≤ 8 m/s at a_lat ≤ 2.1 m/s² ⇔ R ≤ 30 m, i.e. junction/roundabout geometry, not a through-road bend |
| `gap_affected` | any log gap > `G_MAX_S` inside [t_start − 0.5 s, t_end + 0.5 s] | the segment exists (heading did change) but its timing is interpolated |

Segment record: `t_start, t_end` (raw s), `s_start, s_end` (arc m), `Δψ` (signed °, left +), `L_arc`, `R_arc`, `v_min`,
`is_turn`, `gap_affected`.

### 3.2 Lateral action class
Let `Θ = ψ(τ*) − ψ(NOW+2)`, `τ* = argmax_{τ ∈ [2, h_obs]} |ψ(τ) − ψ(NOW+2)|` — the largest heading excursion inside the
observed band, measured from the band start (left +).

| class | rule (precedence top-down) | literal / justification |
|---|---|---|
| `TURN_L` / `TURN_R` — variant **a (junction, DEFAULT)** | \|Θ\| ≥ **30°** AND the segment holding τ* `is_turn`; side = sign Θ | 30° = the route package's GT-turn bar (A&I SPEC §3) and D1/D2's turn definition; `is_turn` as §3.1 |
| `TURN_L` / `TURN_R` — variant **b (heading)** | \|Θ\| ≥ 30°; side = sign Θ | D4's variant b; covers bends (98.2 % of variant-a misses, D4 TRAIN) |
| `LANE_CHANGE_L` / `_R` | a SAM3 lane-line crossing in B (§3.4), only where measurable | — |
| `LANE_KEEP` | otherwise, admitted only when `h_obs ≥ H_ABS_MIN` | absence claim |

Both variants ship (`lat_cls_a`, `lat_cls_b`); a bend that variant a calls LANE_KEEP carries `curve = 1` and its Θ and
R_arc, so the planner still sees the geometry. NUDGE and ABORT_LC are **not emitted** (D2: NUDGE is a curve label,
10.3 / 12.1 % corroborated; ABORT_LC has zero windows, D1).

### 3.3 Lateral constraints (always emitted when the field is defined; NaN + flag otherwise)
| field | unit | definition |
|---|---|---|
| `lat_theta_deg` | ° | Θ above (also for LANE_KEEP) |
| `turn_t_start_s`, `turn_t_end_s` | s from NOW | the segment holding τ* (may be < 2 or > 8; `turn_in_progress = 1` if t_start < 2 s) |
| `turn_d_start_m` | m | arc from the NOW position to the segment start (0 if in progress) |
| `turn_len_m`, `turn_dyaw_deg`, `turn_r_arc_m`, `turn_vmin_ms` | m, °, m, m/s | the segment's own |
| `turn_exit_x_m`, `turn_exit_y_m`, `turn_exit_psi_deg` | m, m, ° | the ego pose at `turn_t_end` in the NOW frame (x forward, y left) — the turn's goal pose |
| `lc_t_cross_s`, `lc_d_cross_m` | s, m | §3.4 crossing time and arc distance from NOW |
| `lat_seg_found`, `lat_gap_affected` | bool | no segment holds τ* (slow bend) / timing interpolated |

### 3.4 Lane change — only where SAM3 lane lines make it measurable
D4 measured that ego motion alone cannot separate a lane change from a nudge (two rules failed the Alpamayo
lane-change TEXT referee, `D4 raw/d4_lanechange.json`), so the lateral position is measured against the **SAM3 map GT
lane lines** (`sam3_gt_v3`, schema `tanitad.sam3_map_gt/3`, fine grid 0.1 m, code 2 = "lane / road line", rig frame,
one map per v2ep frame; train GT on Thor only, eval GT on the dev box).

| literal | value | justification |
|---|---|---|
| cross-section | path-normal samples at arc **s ∈ {5, 6, 7, 8} m** ahead of frame f's rear axle along the ego's own driven path, \|y\| ≤ **4.5 m**, 0.1 m steps; `d_L(f)`, `d_R(f)` = distance to the nearest code-2 cell left / right, median over the 4 stations | the near field the front camera sees (x < ~4 m is hood); following the path keeps the cross-section perpendicular on bends; 4.5 m reaches the adjacent line of a 3.75 m lane with the ego up to 0.75 m off-centre |
| a lane | `d_L + d_R ∈ [2.5, 4.5] m` | EU/US lane widths 2.75–3.75 m, ± marking width and 0.1 m quantisation |
| lane position | `u(f) = d_R / (d_L + d_R)` ∈ [0, 1] | 0 = on the right line, 1 = on the left line |
| crossing | left: `u ≥ 0.7` → `u ≤ 0.3` across ≤ **1.0 s** of invalid frames; right: the mirror | a crossing makes the nearest-line identity wrap |
| not a turn | \|ψ(t_c + 3 s) − ψ(t_c − 3 s)\| ≤ **10°** | a lane change restores the heading; a junction turn that crosses markings does not |
| persistence | no reverse crossing within **3.0 s** | rejects line-straddling wobble |
| measurable | ≥ **50 %** of the band frames with a map (within `h_obs` and the clip's map frames) carry a valid `u` | otherwise LC is undetermined (§3.6) |

Validation bar (pre-registered, §9 V5): against the Alpamayo lane-change TEXT (`EXECUTED`, 72 L / 29 R clips, D4),
dominant side correct ≥ **0.80**, firing on ≤ **0.10** of no-text clips, and a mirrored-y map + path mutation must
flip the side. **If the bar FAILS, LC ships as never-positive** (`lc_measurable = 0` everywhere, the partial label of
§3.6 applies), the failure is reported, and the next lever is named.

### 3.5 Longitudinal action class and constraints
Speed `v(τ)` on the grid; `v̄` = 0.5 s centred moving average (extremum logic only). Event times inside B:
`t_acc` = first τ with v ≥ v_a + 1.5; `t_dec` = first τ with v ≤ v_a − 1.5; `t_stp` = first τ with v ≤ 0.5 (only if
v_a > 0.5).

| class | rule (precedence top-down) | literal / justification |
|---|---|---|
| `HOLD` | max_B v ≤ **0.5 m/s** (needs `h_obs ≥ H_ABS_MIN`) | 0.5 m/s = 1.8 km/h, below any deliberate motion; the v7 `V_STOP_MS` |
| `CREEP` | max_B v ≤ **2.0 m/s** (needs `H_ABS_MIN`) | walking pace (7.2 km/h); D2's independent crawl bar |
| `STOP` | t_stp exists and (no t_acc or t_stp < t_acc) | the ego comes to rest inside B |
| `FOLLOW` | a lead governs the speed (below) | — |
| `DECELERATE` / `ACCELERATE` | the earlier of t_dec / t_acc | **±1.5 m/s** = the documented v7 bar (`DV_ACCEL_MS`, `DV_BRAKE_MS`), 5.4 km/h, above human speed-keeping drift; D2 found the builder silently used +1.0 (fixed here by construction) |
| `KEEP` | otherwise (needs `H_ABS_MIN`) | absence claim |

**FOLLOW** (agents, `join3d` per raw frame, boxes in that frame's rig frame): at each band frame f with agent data, the
lead = the nearest vehicle-class agent whose centre lies within **1.75 m** of the ego's own future path (as seen from
f) at along-path distance 0 < s ≤ **100 m**, excluding any box whose centre falls inside the ego footprint
(−1.5 < x < 4.5 m, \|y\| < 1.2 m — the D3 ego-box defect); `gap_m` = along-path centre distance − ½ lead length − 3.6 m
(rear axle → front bumper, ESTIMATED, stated); `time_gap_s = gap / max(v_ego(f), 1 m/s)`. Following holds at f iff
`time_gap ≤ 3.0 s` and `v_ego(f) ≥ 2.0 m/s`. **FOLLOW** iff following holds on ≥ **50 %** of the band frames that
have agent data AND those frames cover ≥ **3.0 s** of B. Justification: 1.75 m = half a 3.5 m lane (the lead is in the
ego's lane, measured along the ego's own path so it works on curves); 3.0 s ≥ the 2-s rule + reaction margin — inside
it the lead's speed constrains the ego's, beyond it the ego's speed is free (ESTIMATED convention, stated as such);
100 m = the agent store's useful range at highway headway (D4: the 61 m store scope excluded 84 % of joined agents).
If agent data covers < 3.0 s of B, FOLLOW is undetermined (§3.6).

| constraint field | unit | DECEL / ACCEL | STOP | HOLD | CREEP | FOLLOW | KEEP |
|---|---|---|---|---|---|---|---|
| `lon_v_target_ms` | m/s | v̄ at the first local extremum after the bar crossing (plateau tol 0.1 m/s) | 0 | 0 | max_B v | mean_B v | mean_B v |
| `lon_t_reach_s` | s from NOW | time of that extremum | `t_stp` | — | — | — | — |
| `lon_d_reach_m` | m | arc NOW → there | `d_stop` = arc NOW → stop point | — | — | — | — |
| `lon_t_start_s` | s from NOW | last opposite extremum of v̄ before the crossing, in [0, crossing] | same, before t_stp | — | — | — | — |
| `stop_x_m`, `stop_y_m` | m (NOW frame) | — | the stop point | — | — | — | — |
| `stop_dur_s` | s | — | time until v > 0.5 again (NaN if beyond the log) | launch time if seen | — | — | — |
| `lead_gap_m`, `lead_tg_s` (at the first band frame with data), `lead_gap_min_m`, `lead_tg_min_s` | m, s | emitted whenever a lead exists, **whatever the class** (distance keeping, X7) |||||||
| `v_a_ms`, `v_end_ms`, `v_lo_ms`, `v_hi_ms` | m/s | v(NOW+2), v(NOW+8), min_B, max_B — always (the SPEED goal, §4.1) |||||||
| `lon_truncated` | bool | the extremum hit the edge of the observed band |||||||

### 3.6 Partial labels instead of guessed ones
Each family ships `*_cls` (−100 when not single-valued) and `*_allowed` (a bit-mask over the family's classes): the
loss is `−log Σ_{c ∈ allowed} p_c` (a standard partial-label CE; = plain CE when one bit is set). Cases:
* LC not measurable, no turn: `lat_allowed = {LANE_KEEP} ∪ {LANE_CHANGE_x : the D2-style curvature-detrended lateral
  excursion toward x within the observed band ≥ 2.0 m}` (2.0 m = the 2.75 m minimum lane width minus 0.75 m of
  off-centre start/end; below it no lane change is possible).
* FOLLOW undetermined: `lon_allowed = {computed class, FOLLOW}` when the computed class ∈ {DECELERATE, ACCELERATE,
  KEEP}; HOLD / CREEP / STOP stay single-valued (they take precedence over FOLLOW).

### 3.7 Mapping onto the frozen v7 vocabulary (no new token is needed)
The v7 vocabulary is frozen (PI 2026-08-23). v9 classes are a subset/merge of it, so WP-B can keep the 8-wide heads with
`effective_mask`:

| v9 lateral | v7 id | | v9 longitudinal | v7 id |
|---|---|---|---|---|
| LANE_KEEP | LANE_KEEP | | HOLD | HOLD |
| TURN_L / TURN_R | TURN_L / TURN_R | | CREEP | CREEP |
| LANE_CHANGE_L / _R | LANE_CHANGE_L / _R | | STOP, DECELERATE | BRAKE_TO (`v_target` 0 vs > 0 carries the difference) |
| — (masked) | NUDGE_L/R, ABORT_LC | | FOLLOW / ACCELERATE / KEEP | FOLLOW / ACCELERATE / CRUISE |
| | | | — (masked) | ADAPT_SPEED_FOR_CURVE (curve info lives in lateral), YIELD_MERGE (0 windows, D1) |

Both encodings ship (`lat_cls_*`, `lon_cls` in v9 ids and `*_v7id`). Choosing between them is **D-WPA-2**.

## 4. Per-frame tactical GOALS over B (R8-1)

Goals describe **what the ego must reach / respect** in B. A 22-token multi-label set in the frozen v7 vocabulary
(`y`, `w` bits per frame; `w = 0` is IGNORE), plus continuous goal geometry. Every token carries a provenance and a
validity rule; **negatives are written into the file per frame**, so no sidecar and no module state decides them
(the D1 F2 defect cannot recur).

### 4.1 Geometry goals (dense)
| token | positive | negative | IGNORE | constraint (goal geometry) |
|---|---|---|---|---|
| `TURN_L` / `TURN_R` | lateral variant-a class = TURN_x (variant-b set shipped too) | the other classes, with `h_obs ≥ H_ABS_MIN` | `h_obs` too short | exit pose `(x, y, ψ)`, `turn_d_start_m`, `turn_t_end_s`, `turn_r_arc_m` |
| `FOLLOW_LANE` | lateral class LANE_KEEP | TURN or LC present | lateral partial label | — |
| `STOP_POINT` | longitudinal STOP | v_a > 0.5 and no stop in the observed band (`H_ABS_MIN`) | v_a ≤ 0.5 (already stopped / launching — the builder's exclusion) | `stop_x/y`, `d_stop`, `t_stop` |
| `YIELD_FOR_TURN_L/_R` | TURN_x AND a stop (v ≤ 0.5 for ≥ 0.5 s) between NOW and the turn start (the builder's kinematic definition) | TURN_x without that stop; any non-turn (entailed) | as TURN | the stop's `t`, `d` |
| `SPEED` (continuous, replaces `SPEED_BAND` in the BCE) | — | — | `h_obs` < 8 s for `v_end` | `v_a, v_end, v_lo, v_hi` (m/s) — a regression target. `SPEED_BAND` is dropped from the BCE: positive on 4,572 / 4,572 records it is a band, not a decision (D4 §5.3) |

### 4.2 Lane-change goals (SAM3, falling back to VLM)
`LANE_CHANGE_L/_R`: provenance `sam3` where `lc_measurable` (positive iff the §3.4 event in B, negative otherwise);
elsewhere provenance `vlm` under §4.3 (VLM frames only); elsewhere IGNORE.

### 4.3 VLM goals — re-timed to the VLM's own window, negatives by the PI's 2026-09-16 ruling
Tokens: `YIELD, EVADE_IN_CORRIDOR, CORRIDOR_OFFSET, OVERTAKE_VEHICLE, MERGE, GAP_TARGET, REACT_ON_ONCOMING,
TAKE_EXIT_L/_R, TRAFFIC_LIGHT_REACT, _RED, _YELLOW, _GREEN` (+ `LANE_CHANGE_x` where SAM3 cannot measure).

* **Source:** the record's CoT through the builder's own `cot_tokens_v7.goals_from_cot` — with the builder's NUDGE
  "lateral-evidence" gate (`s2_geom_emit_v7.py:329-335`) **REMOVED** (D4: those tokens inherited NUDGE's defect), and
  the builder's two other gates kept: geometry wins on a side conflict (against the v9 turn side), and a CoT that names
  both exit sides emits neither. Each cell records `vlm_lat_evidence` (a D2-style detrended lateral excursion ≥ 0.3 m in
  the VLM window) so WP-B can re-gate without a rebuild.
* **Timing (the VLM's own window, not our anchor):** Alpamayo describes `W_vlm = [t0 + anchor_offset_s,
  t0 + anchor_offset_s + 6]` = raw **[5.1, 11.1] s** (`alpamayo.anchor_offset_s = −2.9`; D2 measured its side peaks at a
  window start of 5.1–6.1 s). Our band equals W_vlm exactly at `NOW* = 5.1 − 2 = 3.1 s`; a VLM token is supervised on
  frames with **|NOW − 3.1| ≤ 2.0 s** (NOW ∈ [1.1, 5.1] raw), where B overlaps W_vlm by ≥ 4 of 6 s. Elsewhere IGNORE.
  ⚠️ This differs from D4's [3.1, 7.1] because D4 re-timed for the v7 [2, 6] band; for the PI's [2, 8] band the
  overlap-maximising centre is 3.1.
* **Negatives:** on the VLM frames, a token the CoT does not assert is a **negative** (the PI's ruling, verbatim in
  `v7_labels.COT_ABSENCE_RULING` and `…_WIDENED`; policy id `cot-absence-negative/2026-09-16` stamped in the manifest),
  for every clip that HAS a CoT; a clip without one is IGNORE. Additionally the **TL probe negatives**: a clip asked the
  traffic-light question with `traffic_light_visible = false` is a negative for all four TL tokens on its VLM frames.
  The measured cost travels with the ruling: 692 / 867 = 79.8 % of visible-light clips become TL negatives (INHERITED,
  `2026-09-16-flywheel-negatives/RESULT.md`).
* **Propagation (D4's rule, the only one shipped):** `TRAFFIC_LIGHT_REACT_RED` is also positive beyond the VLM frames
  when its stop episode is identifiable: E = the stop episode (v ≤ 0.5 m/s for ≥ 1.0 s) overlapping W_vlm, `t_a` its
  start, `t_b` the launch, `t_app` = last time before `t_a` with v ≥ 5 m/s (≤ 8 s back, else `t_a − 8`). RED is positive
  on frames with `max(t_app, t_a − 8) ≤ NOW ≤ t_b − 3.0` (from when the stop enters the 8-s band until the band holds
  < 1 s of red standstill). Clips without such an E stay VLM-frames-only. D4 measured the signature on 223 / 353 RED
  clips. GREEN / YELLOW / YIELD propagation stay proposals (not shipped).
* ⛔ TL and YIELD are **situation outputs: targets only, never an input** (PI 2026-08-03).

## 5. Per-frame NAV (R8-2)

### 5.1 The whole-recording announced-turn table (the three builder defects fixed)
Per clip, §3.1's segments over the **whole log** (raw 0 → the recording end, median 139.7 s, S1). A segment is an
**announced turn** iff `is_turn` and not suppressed:
* defect 1 (`build_v8_nav30s.py:83` read `suppressed`, the emitter writes `applied`): v9 reads
  `turn_suppression.applied` — the emitter's own key;
* defect 2 (30 s cap vs `nav_command`'s 35 s): **no cap** — the table runs to the end of the recording;
* defect 3 (`nav_command` looks at `seq[0]` only; a turn behind a leading curve was hidden, 70 records A6): every
  `is_turn` segment is announced, as A7 ruled ("a navigation system announces the turn, not the curve");
* `NAV_FOLLOW_ROAD` pseudo-entries with null `dyaw_deg` (D1) do not exist in v9: the table holds turns only.
* **Suppression scope:** A7 fixed it **clip-level, exactly as the builder applies it**. The builder's view was the
  35 s after the anchor, so v9 suppresses the announced turns starting in raw **[8.0, 43.0] s** of a suppressed clip,
  and no turn outside that window (no CoT ever spoke about it). The per-contested-turn alternative is **D-WPA-3**.

### 5.2 Per-frame nav fields
With `s_now = s(NOW)`: the **next announced turn** = the first with `s_end > s_now`; `in_progress` iff
`s_start ≤ s_now < s_end`. Everything is SPATIAL (arc positions), so an input never encodes how fast the ego will go.

| field | unit | definition | role |
|---|---|---|---|
| `nav_token` | {FOLLOW, TURN_L, TURN_R} | `TURN_side` iff in_progress OR `d_next ≤ D_ann(v_now) = max(30 m, 6.0 s · v_now)`; else FOLLOW | **INPUT** |
| `nav_side_next` | {none, L, R} | side of the next announced turn | INPUT |
| `nav_d_next_m` | m | `max(0, s_start − s_now)` | INPUT |
| `nav_d_end_m` | m | `s_end − s_now` | INPUT |
| `nav_dyaw_next_deg` | ° | the turn's Δψ (a nav map knows the turn angle) | INPUT |
| `nav_args_valid` | bool | an announced turn exists ahead within the recording; ⛔ **never encoded as d = 0** (D4: the existing reader validates by key presence) | INPUT mask |
| `nav_lookahead_m` | m | recorded arc ahead of NOW (how far "no turn" is known) | INPUT / mask |
| `nav_t_next_s` | s | `t_start − NOW` | ⛔ **LABEL / diagnostic only** — it encodes the ego's FUTURE speed (e.g. how long it waits at a red light) |
| `nav_token_ttime` | as `nav_token` | A5/A6's rule: TURN iff in progress or `t_next ≤ 6.0 s` | ⛔ diagnostic only (continuity with A5/A6), same leak |
| `nav_gap_affected` | bool | the next turn's timing is interpolated over a log gap | mask |

**The horizon, and why.** `H = 6.0 s at the CURRENT speed, floored at 30 m`. 6.0 s = the planner's trajectory horizon
(the plan must contain the turn once it is announced); A5 MEASURED that H = 8 s adds no GT-turn window and 17 straight
ones. The **30 m floor** makes a queued or stopped ego that is within 30 m of its turn start hear the turn (a
time-only rule would stay silent until 6 s before the car starts MOVING, i.e. it would encode the light changing —
ESTIMATED geometry: signalised stop line → turn start ≤ 30 m). Distance equivalents: 30 m at v ≤ 5 m/s, 60 m at
10 m/s, 83 m at 13.9 m/s (50 km/h), 167 m at 27.8 m/s. Because `v_now` is the CURRENT speed, `D_ann` is computable by
a real car from its route and speedometer.

⚠️ **How R8-2's bar reads on this token** (D-WPA-4): the bar is written in true time ("nav says turn but no turn starts
within 6 s ≤ 5 %"). An admissible token announces a decelerating approach earlier than 6 s of true time (ESTIMATED:
12 → 5 m/s over the approach puts the first announced window ~8.5 s out), so the bar will be reported (i) as written,
(ii) restricted to windows that are NOT inside the announcement distance, and (iii) on `nav_token_ttime`. Choosing the
default token is the MM's; my recommendation is the admissible spatial token (the time-based one leaks departure
timing at stops).

## 6. ROUTE CHECKPOINT (R8-3) — an INPUT, not a label

### 6.1 The path it is read from
The ego's driven path from the log (native samples, whole recording), resampled at **Δs = 0.5 m** in arc length and
smoothed with a Gaussian in arc length: **σ = 8 m (`S_8`, the input)** and **σ = 25 m (`S_25`, the heavy reference,
diagnostic only)**, kernel truncated at ±3σ and renormalised. σ = 8 m removes within-lane wiggle (wavelengths ≲ 30 m)
while keeping junction-turn geometry (a 90° turn at R 15 m has a 24 m arc); it is the same order as a navigation
polyline's vertex spacing and CARLA's target-point spacing (~30 m, up to 50 m — PUBLISHED, Jaeger et al. 2023 §3,
banked `2306.07957`). The checkpoint is **spatial only — no speed enters it** (§9 E2 tests that).

Path fidelity: `rc_valid = 0` unless every native chord inside `[s_now − 3σ, s_now + L + 3σ]` has sagitta
`c · |Δψ_chord| / 8 ≤ 0.25 m` (the interpolation error of a chord on an arc), and the recording covers `s_now + L + 3σ`.

### 6.2 Variants (all shipped; §9 E decides which is the input)
| variant | point P | extra fields |
|---|---|---|
| **RC-A30 / RC-A50 / RC-A80** | `S_8(s_now + L)`, L ∈ {30, 50, 80} m | route heading at P (tangent of `S_8`, relative to the NOW heading) |
| **RC-B** (next decision point) | `S_8(s_now + L_B)`, `L_B = clip(s_end(next announced turn) − s_now, 20 m, 80 m)`; no announced turn ahead → 80 m | `L_B`, `rc_b_kind ∈ {turn_end, clamped_max, clamped_min, no_turn}` |
| `RC-H*` (σ = 25 m, diagnostic) | `S_25(s_now + L)` | ⛔ never an input; the lateral-leak reference |

### 6.3 Frame
NOW vehicle frame: origin = the ego's RAW log position at NOW (the rig/rear-axle origin the trainer's trajectory targets
use), x along the quaternion heading ψ(NOW), y left; metres and degrees. `rc_valid` per variant.

### 6.4 Deployment analogue and the NavSim bridge contract (for EvalFlyWheel)
* **A car:** the navigation system's route polyline, map-matched; P = the polyline point at arc L ahead (RC-A) or the
  end of the next announced manoeuvre clamped to [20, 80] m (RC-B), in the vehicle frame from the car's localisation.
* **NavSim:** the bridge computes the route centreline from the scene's route (roadblock / lane ids → lane centrelines,
  the object NAVSIM's metric cache stores as `centerline: PDMPath`, `metric_cache.py:43`), map-matches the ego pose,
  takes the same arc L (or the end of the next route lane connector whose heading change satisfies `is_turn`'s 15°,
  clamped), and expresses P in the ego frame. ⚠️ **The route is NOT part of NAVSIM's `AgentInput`** (MEASURED S1,
  `dataclasses.py:150-155`): an RC-fed NavSim run is a privileged-route arm, exactly as CARLA's target point is. ⇒
  EvalFlyWheel must report RC-ON and RC-OFF (`rc_valid = 0`) arms, and **training must carry an RC-absent row**
  (dropout; the same lesson as the never-trained "unknown" max-speed row, X3). Rate and noise are WP-B's (D-WPA-7);
  the study (§9 E) measures a noised arm (σ_along 2.0 m, σ_lat 0.75 m — ESTIMATED lane-level map + localisation error).
* The literature risk is the **target-point shortcut** (PUBLISHED, Jaeger et al., ICCV 2023, banked `2306.07957`): a
  TP-conditioned TransFuser completes 84 ± 7 % of routes vs 56 ± 12 % with a discrete command (Table 1) because it
  extrapolates its waypoints toward the nearest TP — which resets lateral error but cuts corners when the TP lies
  behind a turn (§3.1, Fig. 3); attention pooling (RC 93 ± 3, Table 2) and shift/rotation augmentation mitigate it.
  Their §3.3 also samples the expert path at fixed DISTANCES to keep speed out of the path representation — the reason
  RC-A is a fixed arc length.

## 7. (Optional, D-WPA-5) the speed-limit input proxy, carried for X3
D4's non-oracle default **N2** (past-20 s realised max snapped UP to the 8-step road-law ladder {20, 30, 50, 70, 80,
100, 120, 130} km/h with a 50 km/h urban floor; leak 3.5 %) and **N3** (urban / rural / motorway), per frame from the
same log, past samples only. Costs nothing extra in the build; shipped only if the MM confirms it belongs to WP-A.

## 8. Admissibility, per field (PI 2026-08-03: labels may use ego / future / agents / maps; inference = vision + the supplied nav, route checkpoint and speed limit; a goal input must not carry the situation classifier's output)

| field group | role | computed from | admissible? | optimism stamp |
|---|---|---|---|---|
| lateral / longitudinal actions + constraints, geometry goals, SPEED | TARGET | ego future (log), agents (FOLLOW), SAM3 map (LC) | yes (labels) | — |
| VLM goals | TARGET | Alpamayo CoT + ego kinematics (RED propagation) | yes (labels); ⛔ never an input (situation output) | — |
| `nav_token`, `nav_side_next`, `nav_d_next_m`, `nav_d_end_m`, `nav_dyaw_next_deg`, masks | **INPUT** | ego future PATH (arc, heading change), current speed | yes as a supplied route; contains no situation-classifier output | ⚠️ **optimistic on PhysicalAI by construction** — the route is the ego's own future path (`oracle = ego-future`, `allow_oracle_nav` stamped) |
| `nav_t_next_s`, `nav_token_ttime` | diagnostic | ego future TIME | ⛔ not as input (future-speed leak) | — |
| RC-A / RC-B (+ heading), `rc_valid` | **INPUT** | ego future PATH, smoothed, spatial only | yes as a supplied route; no situation output | ⚠️ optimistic on PhysicalAI by construction (it is where the ego actually went, incl. its lane); privileged on NavSim (§6.4) |
| RC-H (σ 25 m) | diagnostic | ego future path | ⛔ never an input | — |
| N2 / N3 (if shipped) | INPUT | ego PAST speed + road-law floor | yes (ego history); ⚠️ at deployment the MAP limit replaces it (D4) | — |

## 9. Pre-registered experiments and bars (both outcomes committed now)

### 9.E — The ECHO / LEAK study for the route checkpoint and nav (brief item 2, Stage 2; CPU, before any training)
**Population.** EVAL = eval139, every trainer window (23,772, D1 truth table `label_truth_eval139.npz`, row = `ds`
index) with `rc_valid` for the arm, plus the route package's EVAL-DIAG grid (1,112 windows, digest `92e36a1a…`) for
the comparison with refcv7's banked picks (`D:/refcv7_route_bin/2026-10-04/eval_s0g.npz`). Replicate population: a
seeded 600-clip TRAIN sample (seed 20261004), for the fits that need power. Slots: the bank's 8 slots
(0.5, 1, 1.5, 2, 3, 4, 5, 6 s, `route_metrics.py:3`); GT = the log at the slot times (cross-checked against the bank's
`gt` on EVAL-DIAG). Window classes: GT-turn / straight / gentle exactly as `route_metrics.gt_class` (re-typed, not
imported) and, on all windows, D1's columns.

**E1 — trivial planner floor (the "echo").** TP(v) = the constant-curvature arc tangent to the NOW heading through P
(`κ = 2·y_P / (x_P² + y_P²)`), driven at **constant v0** for 6 s, sampled at the 8 slots. Arms: TP per RC variant
(A30, A50, A80, B), TP on the noised RC (σ_along 2.0 m, σ_lat 0.75 m, seeded), TP on a **deranged** RC (a seeded clip
derangement, A5's T2c construction — the control), CV-straight (κ = 0, v0 — no route), and **TP with the GT speed
profile** (isolates the lateral content). Metrics: ADE (8 slots), heading-within-15° of the terminal (slot 5→6 s)
segment, turn-direction-correct on GT-turn windows (`route_metrics.dir_class`, τ 10.35°), cross-track at 6 s. Paired
episode-cluster bootstrap (B = 2000) vs refcv7's E9 pick on EVAL-DIAG.
* **Bars:** controls — the deranged-RC TP must be **worse** than CV-straight on GT-turn heading-within-15° (else the
  instrument is broken and E stops); TP's numbers are **the floor refcv8 must beat** (R8-3), quoted beside refcv7
  (0.841 turn direction, 0.514 heading-15, turn ADE 3.19 m, A&I RESULT §1.1). Expected (ESTIMATED): TP-RC turn
  direction ≥ 0.9 and well above refcv7 — that is the route information working, not a failure; the risk read here is
  a floor so high that a model can reach it by copying P (the TP shortcut).

**E2 — speed leak.** Targets: v(NOW+τ), τ ∈ {1, …, 6} s, from the log. Feature sets: `F0 = {v0}` (primary),
`F0' = {v0, a0}` (a0 = past 0.5 s mean acceleration; admissible history), `F0 + RC_v` (x_P, y_P, ψ_P), `F0 + NAV`
(token, d_next, d_end, Δψ_next, valid), `F0 + NAV + RC_v`. Models: OLS and a gradient-boosted tree regressor (a linear
negative is not a negative about learnability — CLAUDE.md), 5-fold clip-grouped OOF; hyper-parameters fixed now
(trees: 200 iterations, depth 6, lr 0.05). Metric: **recovered fraction ρ = (R²(F) − R²(F0)) / (1 − R²(F0))**, mean
over τ (D4's leak metric). Controls that must read known values: RC shuffled → \|ρ\| ≤ 0.01; the target itself → 1.0;
D4's per-window future-max oracle O1 → ρ ≥ 0.40 (D4 measured 57.0 %); n and d printed.
* **Bar (≈ equal):** `ρ(F0 + RC_v) ≤ ρ(F0 + NAV) + 0.03` AND `ρ(F0 + NAV + RC_v) − ρ(F0 + NAV) ≤ 0.03`, under BOTH
  model classes — the checkpoint may carry the speed information the route geometry itself implies (a sharp turn ahead
  ⇒ slower), which a real navigation system supplies too, and not more. Fail ⇒ the variant is not an admissible input
  as built; the next lever is a larger σ or dropping its heading field, re-measured in the same run.

**E3 — lateral leak.** δ_v = signed perpendicular offset of RC_v's P from `S_25` at the same arc length. Report
median / p90 \|δ\| per variant, overall and split by windows with vs without a lateral manoeuvre in [NOW, NOW+8]
(SAM3 LC, or a D2-style detrended excursion ≥ 1.5 m). Plus OOF R² of the ego's lateral offset at NOW+6 s (relative to
`S_25` at the same arc) from RC_v vs from `RC-H` — how much within-lane information the input adds over the
heavy route.
* **Bar:** p90 \|δ\| ≤ **1.0 m** on all valid windows (below lane-change scale ≥ 2.5 m, and below a lane-level
  map + GNSS error), and the lane-manoeuvre windows' median \|δ\| reported (no bar; it is the leak's size where it
  matters).

**E4 — per variant, and the decision rule (fixed now).** One table: A30 / A50 / A80 / B (+ noised) × E1–E3. The
**recommended RC variant** = among variants that pass E2 and E3, the one with the highest TP heading-within-15° on
GT-turn windows; ties → the larger L (fewer corner-cutting chances, TP-shortcut paper). If **no** variant passes:
report FAILED with the measured leak, and run, in the same stage, the next levers in this order: σ = 15 m; drop the
heading field; noised input as the training default — each re-measured; if all fail, the RC goes to the MM as "not
admissible as built", with the numbers.

### 9.V — Validation of the release (brief item 4, Stage 3; bars from PLAN_REFCV8 §0 R8-1/R8-2)
All geometry cross-checks are re-derived by an **INDEPENDENTLY written** module (`code/v9_independent.py`) that imports
nothing from the builder and uses different channels where they exist: heading from the PATH TANGENT (positions) not
the quaternion, arc length from **∫ speed dt** not chords, radius from the provider's own `curvature` channel.

| id | check | bar |
|---|---|---|
| V1 | coverage: windows with a single-valued OR partial lateral and longitudinal label | ≥ **95 %** of train (746,946) and eval139 (23,772) windows; **every** independent turn window (\|Θ\| ≥ 30° on B, observed) labelled |
| V2 | agreement with the independent derivation: TURN side (variant b vs independent heading-only), STOP | ≥ **0.95** each (n, split, Wilson CI printed); variant a's agreement on independent junction turns printed |
| V3 | constraint error vs the independent 100 Hz derivation | turn t_start / t_end within ±0.5 s on ≥ 90 %; Δψ within ±3°; R_arc within ±10 % of the curvature-channel radius; `t_stop` within ±0.3 s and `d_stop` within ±1.0 m on ≥ 90 %; nav `d_next` within ±2 m |
| V4 | analytic controls (synthetic logs through the builder) | circle R 15 m, 90°, 5 m/s → TURN, R_arc 15 ± 0.3 m, Δψ 90 ± 1°, t_start ± 0.1 s; constant −2 m/s² from 10 m/s at NOW+2.5 s → STOP at t 7.5 ± 0.1 s, d_stop 50.0 ± 0.2 m; straight → LANE_KEEP / KEEP; +0.5 m/s² ramp → ACCELERATE; synthetic 3.5 m lane change on a synthetic map → LC on the right side; synthetic lead at 2.0 s gap → FOLLOW with gap ± 0.2 m; RC on an analytic circle (σ → 0) → the exact chord point; nav on a known turn → exact `d_next` |
| V5 | lane change vs the Alpamayo LC text (§3.4) | side ≥ 0.80, firing ≤ 0.10 on no-text clips, mirror flips |
| V6 | a MUTATION per field that must go RED | yaw sign flip → turn side flips; band shifted +6 s → turn agreement collapses; speed ×1.5 → d_stop moves; map mirrored → LC side flips; lead removed → FOLLOW disappears; nav arc origin at the clip start (D4's mutation) → d_next wrong; RC y-flip → TP direction collapses; ⛔ every validator also run on its own mutated input must read RED (a check that cannot fail is not a check) |
| V7 | the truncation rule (§2.4) | on full-band frames, truncating to 6.0 s changes ≤ 5 % of lat and of lon labels, else `H_ABS_MIN = 8.0` |
| V8 | nav bars (R8-2) | turn-commanded windows with no announced turn starting within 6 s: ≤ 5 % (reported three ways, §5.2); realised announced turns with the matching token within the announcement horizon ≥ 90 %; un-announced realised turns (curves, suppressed) reported separately with their n |
| V9 | VLM coverage | windows per token (positive / negative / ignore), RED-propagated windows and clips, vs v8's in-band counts |
| V10 | class support per split | per class, per split (train / eval139), windows and clips |
| V11 | join control | a v9 row joins every trainer window: `(sid, t + 9)` exists for 746,946 / 746,946 and 23,772 / 23,772; `now_s` equals D1's `t_now_s` to ≤ 1e-6 s |

If a bar fails: the release ships with that field's mask OFF and the failure reported, the cause is diagnosed, the
next lever is named and — if cheap — run in the same stage (Rule Zero).

### Hypotheses (IDs for the register; the MM owns `GOALS_AND_CLAIMS.md`)
* `H-V9-COV` dense v9 actions cover ≥ 95 % of windows and every observed turn window (V1).
* `H-V9-GEOM` v9 TURN side and STOP agree ≥ 0.95 with an independent derivation (V2) within the V3 tolerances.
* `H-V9-NAV` the per-frame announced nav meets R8-2's bars (V8).
* `H-V9-LC` SAM3 lane lines make lane changes measurable at the V5 bar.
* `H-RC-ECHO` / `H-RC-SPEEDLEAK` / `H-RC-LATLEAK` the route checkpoint's floor, speed leak and lateral leak (E1–E3).

## 10. Release format (the schema doc and the reader come in Stage 4)
* Per split: `v9_labels_{train,eval139}.npz` — clip table (`sid`, `sha12`, `row0`, `n_rows`, `k0`, `grid_start_s`,
  `dt_s`, `clock_src`) + flat per-row arrays named `<group>__<field>` (groups: `key, valid, lat, lon, lead, goal, nav,
  rc, speed`); goal bits as `uint32` y / w masks over the frozen 22-token order + per-token provenance codes.
* `v9_labels_{split}.manifest.json`: every source with md5 (logs: per-clip md5 list digest; labels; join; SAM3; clock
  sidecar; manifests), the builder commit + file md5s, every literal of this SPEC, the policy ids
  (`cot-absence-negative/2026-09-16`, `allow_oracle_nav`), counts per field.
* Reader `stack/tanitad/data/v9_labels.py`: pure functions over an immutable loaded object, lookup by `(sid, k)` or
  `(sid, now_s)` with a ≤ 1e-6 s match or a refusal; ⛔ no module-level mutable state (D1 F2), pinned by a test that
  loads train then eval and asserts the first object is unchanged.
* Expected size (ESTIMATED): ~869 k train rows × ~90 fields × 4 B ≈ 0.3 GB raw, less compressed → NOT for git (Thor +
  dev box, md5 in the manifest, which IS in the package).

## 11. Corpus manifest rules (brief item 6, Stage 4)
From D3 `raw/c1_*` and `raw/c3b_agents_detail.json`: DROP the train recording-partners of eval clips (image-confirmed
tier first, the looser tier listed separately) and one clip of each duplicate-video pair; MASK the 693 ego-agent-box
frames (18 clips) for the agent/box3d targets AND exclude them as FOLLOW leads (§3.5 already rejects ego-footprint
boxes). eval139 unchanged.

## 12. Compute plan
* **Dev box (CPU, < 2 GB resident, `OMP_NUM_THREADS=4`):** everything log-derived (all 4,508 logs are local, 1.9 GB):
  segments, actions, nav, RC, speed proxy, the echo/leak study; FOLLOW from the local `join3d` (409 MB `.xz`,
  streamed per clip); eval139 LC from the local eval SAM3 GT.
* **Thor (one detached job under `flock /home/nvidia/refcv8_audit/cpu.lock`, outputs < 5 GB under
  `/home/nvidia/refcv8_v9labels/`):** train LC from `/home/nvidia/data/sam3_gt_v3` (train SAM3 GT exists only there).
  Thor ships back a per-row LC table (~10 MB) that the dev-box build merges.
* Order: eval139 end to end (+ V1–V11 on eval139) → train → train validation.

## 13. Decisions requested from the Master Mind (none blocks Stage 2)
| id | question | my default |
|---|---|---|
| D-WPA-1 | default lateral TURN variant for training | **a** (junction, `is_turn`); b shipped for the v7-tiny arm D4 pre-registered |
| D-WPA-2 | v9 class ids vs the frozen v7 ids (8-wide heads + masks) | ship both; WP-B uses the **v7-id** encoding (no vocabulary change, PI freeze respected) |
| D-WPA-3 | suppression scope over the whole recording | A7 clip-level, restricted to the builder's own 35-s view (raw [8, 43] s); per-turn is the alternative |
| D-WPA-4 | nav token rule; how R8-2's 6-s bar is read for it | **spatial / admissible** token (`D_ann = max(30 m, 6 s · v_now)`); the true-time token as diagnostic; the bar reported three ways |
| D-WPA-5 | should WP-A ship the N2/N3 speed-limit proxy | yes (cheap, same log) |
| D-WPA-6 | VLM lateral tokens without the NUDGE gate | yes, with `vlm_lat_evidence` recorded per cell |
| D-WPA-7 | RC dropout rate and training noise (WP-B) | dropout ≥ 0.3; noise per E1's noised arm — WP-B decides |
| D-WPA-8 | absence claims on ≥ 6 s observed horizons | yes, gated by the V7 truncation test |

## 14. What this SPEC changes relative to D4's design, and why
* Band **[NOW+2, NOW+8]** (the PI's words) instead of D4's [NOW, NOW+6] for every action and goal.
* Variant a uses the builder's calibrated `is_turn` with ARC radius instead of D4's `1/κ_max ≤ 40 m` (the builder's
  MEASURED S-bend lesson: peak curvature mis-reads bends as junctions).
* Lane change is measured against SAM3 lane lines (D4's proposal made concrete), with a partial label where it cannot be.
* FOLLOW is measured along the ego's own path (curve-safe) out to 100 m, with partial labels where agents are missing.
* The nav token is SPATIAL (current speed only); `t_next` and A5's time rule are kept as diagnostics because they encode
  the ego's future speed (e.g. a red light's duration).
* VLM tokens re-timed to the 6-s band's own overlap centre (NOW* = 3.1 s, not D4's 5.1 s centre for the v7 band).
* Negatives live inside the release per frame — no sidecar binding, no module state.
