# D2 / SEMANTICS — field-by-field definition audit of the v8 labels refcv7 consumed

Stream D2, refcv8 data audit, 2026-10-04.  Label file: `s2_labels_v8_train.jsonl.gz` md5 `b45377a1f25263b5c0f3d318c126b1ac`
(4,572 records, one per clip) and `s2_labels_v8_eval.jsonl.gz` md5 `eefc38d1453bd1c73802d44d45affced` (147 records).
Evidence class on every row: **MEASURED** = recomputed by D2 from the cache poses / label file (artifact in `raw/`),
**READ** = quoted from source, **UNVERIFIED** = not tested.

**Which code produced the label?**  The v8 `a_tac`, `g_tac`, `nav_command`, `manoeuvre_sequence`, `bands`, `turn_suppression`
come from the **v7 emitter** `stack/scripts/s2_geom_emit_v7.py` (git blob `cc1cee146ebf…`, identical in the repo and in the
launch-tree extract `C:/Users/Admin/ev7/stack`, MEASURED by `git hash-object`) + `stack/tanitad/data/ego_manoeuvre.py`
(blob `f6c30be5…`, identical in both).  v7.1 / v7.2 / v8 are **patches** on top (`TanitAD-artifacts/_s2build-copy-20260919/release/build_v71.py`,
`build_v8_nav30s.py`, `fix_v8_nav30s_arc.py`, `build_v8_speedmax.py`, `build_v8_tacsit.py`, …), each asserting that every
earlier trainer-visible field stayed byte-identical.  All line numbers below are for those blobs.

**What does the trainer actually read?**  `v7_labels.load_v7_labels` (`v7_labels.py:376-417`, launch blob `86f0c46d…`) keeps ONLY:
`a_tac.lat`, `a_tac.lon`, `a_str.token`, `g_str.token`, `g_tac.anchor`, `g_tac.goals` (names + meta), `bands`, `t0_s`, `horizon`,
an `audit` dict (never read by training code), and, behind the oracle gate, `nav_command` and `speed_max_input`.
Everything else in the 10 KB record (`lat_args`, `lon_args`, `nav_30s`, `alpamayo`, `scene`, `tac_SIT`, `cot_tokens`, `semantics`,
`strata`, `lane_change_text`, `manoeuvre_sequence`, `turn_suppression` content) is **dropped at load** (READ + grep of
`refc_v3_train.py`, `v7_labels.py`, `tanitad/refs/*`, `tanitad/models/*`: no consumer of `lat_args`/`lon_args`/`lat_peak_m`).
The window rule is `v7_labels.window_in_band` (`:678-698`): a window with NOW=`t` is labelled iff `|t − t0_s| ≤ (hi−lo)/2` =
**2.0 s**, i.e. NOW ∈ [6.0, 10.0] raw.  The NOW clock is `refc_v3_train.py:3753` (`grid_start_s + (row + 2)·dt_s`, launch tree
fec3a0d; trainer md5 `0a6fb0d8…` equals `git show fec3a0d:…`, MEASURED).

---------------------------------------------------------------------------------------------------------------------
## 0. Verdict table — fields whose NAME, DOCSTRING and COMPUTATION disagree

| # | field | name / docstring says | computation does | consequence | n (train 4,369 scored) |
|---|---|---|---|---|---|
| 1 | `a_tac.lat = NUDGE_L/R` | a deliberate lateral nudge (obstacle pass / lane wobble), `ego_manoeuvre.py:103-112` calls 1.0 m "the deciding evidence for a NUDGE" | key-frame lateral peak ≥ 1.0 m **over [0,6] s from the anchor, no curvature removal, no upper bound, no "returns" test**, plus `5° ≤ |peak yaw| < 25°` (`ego_manoeuvre.py:326-337`, `s2_geom_emit_v7.py:451-453`) = **a gentle road curve** | MEASURED: 89.7 % of NUDGE records (1,028) have a heading that does NOT come back; median net yaw 9.3°, median key-frame lateral peak 5.9 m; only 10.3 % (NUDGE_L) / 12.1 % (NUDGE_R) are corroborated by an independent nudge/shift test; an analytic 10.7° R=300 m bend reads NUDGE_L and a true 1.0 m out-and-back at 14 m/s reads LANE_KEEP (`raw/controls_result.json`) | **1,028 labels = 23.5 % of the tactical lateral mass** |
| 2 | `a_tac.lat_args.lat_peak_m` | "signed peak lateral offset that decides a NUDGE (`abs ≥ 1.0 m`)" (`build_v71.py:20-22`, `ego_manoeuvre.py:103-112`, `V71_MANIFEST`) | peak signed lateral displacement over the **first 20 s of the clip, in the heading frame of the clip's first sample** (`build_v71.py:68-86,128-134`) — a road-geometry quantity | see §3: MEASURED reproduces at corr 0.9995; **not a defect in the data, a defect in the name/docstring**; already withdrawn in `V72_MANIFEST.field_semantics` (`patch_v72_manifest_latpeak.py`) but not in the code docstring | 0 training exposure |
| 3 | band `[2,6]` vs computed span | `bands.tactical_s = [2,6]`; the PI's rule "0–2 s belongs to the operative layer" (`s2_geom_emit_v7.py:45-57`) | `a_tac.lat` NUDGE and `a_tac.lon` dv/stops/ADAPT-gate are computed on **[0,6]** from the anchor (`:424,428,455,463`); only TURN and HOLD/CREEP use [2,6] (`:441,472-476`) | MEASURED: independent longitudinal agreement is 80.3 % on [2,6] but 89.5 % on [0,6] (train; eval139 78.6 → 91.5 %) | all |
| 4 | `DV_ACCEL_MS = 1.5` | ACCELERATE at Δv ≥ +1.5 m/s (`:93`, branch `:493`) | the cascade's last `else` returns ACCELERATE for any `dv_end ≥ +1.0` (`:497-500`) | MEASURED: 18.0 % of 974 ACCELERATE records have Δv[0,6] ∈ [1.0, 1.5) | 175 records |
| 5 | `a_tac.lon_args.v_target_ms` | a target speed | `round(m.v_min, 2)` = **minimum speed over [0,6]** (`:505-506`, `ego_manoeuvre.py:383`) | MEASURED: equals the window minimum (corr 1.0000, median err 0.004 m/s); on ACCELERATE the "target" is the START speed, the end speed is +3.2 m/s higher (median, n=974) | not consumed |
| 6 | `nav_30s.entries` suppression | "a contested turn commands NO turn" | `build_v8_nav30s.py:83` reads key `"suppressed"`; the emitter writes `"applied"` (`s2_geom_emit_v7.py:893`) → suppression never read | MEASURED: 66 of 66 suppressed clips list the turn in `nav_30s` while `nav_command` says FOLLOW | 66 records; refcv7 did not read nav_30s |
| 7 | `nav_command` vs `nav_30s` horizon | both "next 30 s" | `nav_command` has no horizon cap (look-ahead is 35 s past the anchor, `emit_one:864`, `max_s = RAW_T0_S + 30 + 5`); `nav_30s` caps at ≤ 30.0 s (`build_v8_nav30s.py:46,87`) | MEASURED: 68 TURN tokens with `time_s` 30.1–35 s have no `nav_30s` entry | 68 records |
| 8 | `nav_command` = FOLLOW | "no turn ahead" | token = side of `seq[0]` only (`:699-716`); if the FIRST sustained segment is a curve the later junction turn is hidden | MEASURED: 76 clips whose `seq[0]` is a non-turn curve while a later entry `is_turn` get FOLLOW (66 more are the suppressed ones in #6) | ≥ 76 records |
| 9 | `alpamayo.lateral.agree` / `.longitudinal.agree` | "do the two sources agree" | compare the VLM's claim on **its own window [5.1, 11.1]** against geometry on a **different** window: lateral vs our label over [8,14] (`:773-782,805-810`); longitudinal on the VLM's own window only | MEASURED: 231 of 1,519 lateral "conflicts" are timing — both sources are right on their own windows (see RESULT §3) | all |
| 10 | `speed_max_input` / `SPEED_BAND.v_hi_ms` as a "ceiling / limit" | a posted limit the driver may not exceed | the ego's **realised** max speed over [anchor+2, anchor+6] (`:309-315`), declared in-record (`training_value_is`) and oracle-gated | MEASURED stale outside ±2 s of the anchor, but applied to **every** window (§7) | all windows |

Not in dispute (name = computation, MEASURED against cache poses): `a_tac.lat_args.within_m` (corr 1.0000, median err 0.025 m),
`SPEED_BAND.v_lo_ms/v_hi_ms` (median err 0.004/0.005 m/s, 100 % within 0.3 m/s, n=4,369+139), `bands`, `t0_s`.

---------------------------------------------------------------------------------------------------------------------
## 1. `t0_s` and `bands`

* **Built:** `t0_s = ES.RAW_T0_S = 8.0` (`egomotion_source.py:56`, `emit_one:920`); `bands = {operative [0,2], tactical [2,6], strategic [8,30]}` (`s2_geom_emit_v7.py:50-52`, `emit_one:927-934`).
* **MEASURED (label file, D2 + MM):** `t0_s = 8.0` on 4,572/4,572; `bands` identical on 4,572/4,572; `schema_version=s2-geom-v7, vocab=v7, release=v8` on 4,572/4,572; `a_tac.truncated=false` on 4,572/4,572 (`raw/census_nav_train.json`).
* **Clock:** the NOW clock is exact where the sidecar covers the clip — the 10 Hz cache speed is reproduced from the 100 Hz egomotion log with `grid_start_s + (row+2)·dt_s` at median mean-|err| **0.0017 m/s** (4,347 sidecar clips) / **0.0022** (136 eval clips), against 0.027 at ±0.05 s and 0.052 at ±0.10 s shifts (MEASURED, `raw/clock_check_train.json`, `clock_check_eval139.json`).  The 22 train / 3 eval clips the sidecar does not cover run on the fallback `grid_start_s = 0` and are visibly off (median 0.03–0.045 m/s ≈ 0.1 s).
* **Trainer assumption** (`v7_labels.py:678-698`): the record describes windows with NOW ∈ [6,10]; each such window's own forward band [NOW+2, NOW+6] overlaps the record's [10,14] raw by ≥ half.  The record is computed once, at NOW=8.0, and applied unchanged to NOW=6…10 — see RESULT §2 for how accuracy behaves over that range (flat 69–73 % lateral; longitudinal decays 80 % → 64 %).

## 2. `a_tac.lat`  (frozen vocabulary LANE_KEEP, NUDGE_L/R, TURN_L/R in the data; LANE_CHANGE_*, ABORT_LC never emitted)

* **Built:** `tactical_actions` `s2_geom_emit_v7.py:423-453`.
  * `seq = manoeuvre_sequence(poses, key)` (`:100-153`): sustained yaw-rate segments after the anchor — `|rate| ≥ 6°/s` (`:60`) for ≥ 0.8 s (`:61`), net `|dyaw| ≥ 20°` (`:62`), radius = arc/|dyaw|, v_min.
  * `is_turn` (`:156-176`): `|dyaw| ≥ 15°` ∧ `R_arc ≤ 140 m` ∧ `v_min ≤ 8.0 m/s` (`:87-89`).
  * In-plan = each segment **clipped to [2,6] s** (`split_by_band:509-599`), `|dyaw_clipped| ≥ 20°` (`:590`).
  * TURN_L/R = sign of the largest in-plan `is_turn` segment (`:441-449`); if the turn is **suppressed** (`turn_suppression`) it becomes NUDGE of the same sign (`:442-446`).
  * else NUDGE_x iff `EM.analyse(poses[:key+61], key).lateral_class == NUDGE_x` (`:428,451-453`): `abs(lat_peak) ≥ 1.0 m` ∧ `abs(peak yaw) ≥ 5°`, with `|peak yaw| < 25°` (≥ 25° is ROAD_BEND ⇒ LANE_KEEP; `ego_manoeuvre.py:326-337`), where `lat_peak` is the key-frame lateral offset over the whole analysed span **[key, key+6 s]** (`:322-324`); **no upper bound** (`:110` "NUDGE has NO UPPER BOUND (D-NUDGE-ABSORB)"), no curvature removal, no return test.
  * else LANE_KEEP.
* **Trainer meaning:** class of the window's forward tactical band `[NOW+2, NOW+6]` (cross-entropy on windows NOW∈[6,10], `refc_v3_train.py:3846`).
* **MEASURED vs independent geometry** (`raw/agree_train.json`, `agree_eval139.json`; literal thresholds in `d2_lib.py`, built from road geometry, not from the builder; analytic controls in `raw/controls_result.json`): see RESULT §2.  Per class (train, band [2,6]): LANE_KEEP 91.6 % (2,587/2,824), TURN_L 87.0 % (235/270), TURN_R 85.4 % (211/247), NUDGE_L 10.3 % (48/464), NUDGE_R 12.1 % (68/564).  A turn's side was never inverted where both sources name a side (TURN_L→TURN_R 0, TURN_R→TURN_L 0, train band [2,6]).
* **Verdict:** TURN and LANE_KEEP mean what the trainer assumes.  NUDGE does not — see row 1 above.

## 3. `a_tac.lat_args.lat_peak_m`  — the settled question

* **Writer:** NOT the emitter.  Added by the v7.1 patch `build_v71.py:68-86` (`c, s = cos(-yaw[0]), sin(-yaw[0]); track = s·(x−x0) + c·(y−y0)`; window `0 ≤ t ≤ 20` from the egomotion tar; value = `track[argmax|track|]`) and stored at `:128-134`.  `yaw[0]` is the heading of the **first sample of the clip (raw t≈0)**, not the anchor's.
* **Meaning (MEASURED):** the signed peak lateral displacement over the clip's first 20 s, in the clip-start heading frame; **left = +**; units **metres**.  Recomputing it from the cache poses of 4,369 train + 139 eval clips reproduces the stored value at correlation **0.99953 / 0.99959**, sign equal on 98.9 % / 100 %, median |error| 0.59 / 0.53 m (the cache starts ≈ 0.3 s after the log's origin and is sampled at the camera clock, which explains the residual).  Rejected alternatives: anchor-frame over [0,6] s (corr 0.59 train / 0.67 eval), band-start frame over [2,6] s (0.44 / 0.51).  (`raw/agree_train.json → recompute_checks.lat_peak_m`.)
* **Why |x| is large:** any bend or junction displaces the vehicle tens of metres laterally in 20 s: median |x| 19.6 m, p90 109.6 m, max 305.3 m, 70.2 % > 5 m (train, MEASURED) — exactly CONTEXT.md's 19.7 / 110.8 / 305.3.
* **Is it the NUDGE quantity?  No.**  85.0 % of builder LANE_KEEP records exceed 1.0 m on it and 0.3 % of NUDGE records are below 1.0 m (train n=4,369) — the 1.0 m rule cannot be applied to it.  `ego_manoeuvre.Manoeuvre.lat_peak_m` (`ego_manoeuvre.py:112,322-324,379`) is a *different* quantity under the same name (key-frame, over the analysed span, never written to the label file).
* **Verdict: neither broken nor a lateral offset of a nudge — a correctly computed 20-s road-geometry quantity carrying a name and a code docstring that describe another quantity.**  The data was documented correctly in `V72_MANIFEST.field_semantics` (`patch_v72_manifest_latpeak.py`, "work_item_NOT_done_here": ship the true deciding quantity); that work item is still open, and the docstrings at `ego_manoeuvre.py:103-112` and `build_v71.py:20-22` still carry the withdrawn claim.  **refcv7 never read the field.**

## 4. `a_tac.lon`  (CRUISE, ACCELERATE, ADAPT_SPEED_FOR_CURVE, BRAKE_TO, FOLLOW, CREEP, HOLD in the data)

* **Built:** `tactical_actions` `:455-500`, a first-match cascade over `m = EM.analyse(poses[:key+61], key)`:
  1. `band_max` (max v over **[2,6]**, `:472-475`) `≤ 0.5` → HOLD, `≤ 2.0` → CREEP (`:475-476`);
  2. `v0 ≤ 0.5 ∧ v_end ≤ 0.5` → HOLD (`:477`); a stop episode in **[0,6]** with `v0 > 0.5` → BRAKE_TO (`:463,479-482`); slow crawl `0.5 < v0 ≤ 2.5 ∧ v_end ≤ 3.5` → CREEP (`:483-488`);
  3. `turning` = TURN lat **or** (`turn radius ≤ 80 m ∧ |peak yaw| ≥ 20°`) (`:456-457`) → **ADAPT_SPEED_FOR_CURVE** (`:489-492`) — this *pre-empts* the speed class;
  4. `dv_end = v(+6) − v(0) ≥ +1.5` → ACCELERATE (`:493`); `dv_min ≤ −1.5` → BRAKE_TO if `dv_end ≤ −0.75` else FOLLOW (`:495-496`); `|dv_end| < 1.0` → CRUISE (`:497`); else FOLLOW if `dv_end < 0` else ACCELERATE (`:499-500`).
* **Windows used:** HOLD/CREEP on [2,6]; everything else on **[0,6]** from the anchor.  The record's band says [2,6].
* **MEASURED (independent: stop = v<0.5, crawl ≤ 2 m/s, brake Δv ≤ −1.5, accelerate ≥ +1.5, over the stated window):**

  | | band [2,6] (record's own) | span [0,6] (builder's actual) |
  |---|---|---|
  | eval139, 117 speed-defined records | 92/117 = 78.6 % | 107/117 = 91.5 % |
  | train, 3,419 speed-defined records | 2,744/3,419 = 80.3 % (CI 78.9–81.6) | 3,060/3,419 = 89.5 % (88.4–90.5) |

  Per class on [0,6] (train): CRUISE 100 %, BRAKE_TO 96.2 %, FOLLOW 100 % (BRAKE∪CRUISE), ACCELERATE 81.6 % (the +1.0 vs +1.5 effect, row 4), HOLD 44.5 % and CREEP 23.5 % (they are defined on [2,6]; on [2,6] they are 98.0 % and 68.1 %).  ADAPT_SPEED_FOR_CURVE (950): every record has a ≥ 20° heading change over [0,6] (99.2 %), median 55° — it is a valid curve flag, but it **hides the speed class** (underneath: 272 cruise, 486 accelerate, 192 brake on [0,6]).
* **Verdict:** longitudinal means what the trainer assumes **if the trainer's "forward [NOW+2, NOW+6]" is read as "[NOW, NOW+6]"**; the documented band and the computed span disagree and the labels follow the computation.

## 5. `a_tac.lat_args` / `a_tac.lon_args` / `serves_goals`

`within_m` = arc length over [0,6] from the anchor (`_arc_to`, `:179-185,504`): MEASURED exact.  `v_target_ms` = window **minimum** speed (row 5).  `serves_goals` (`vocab_v7.action_serves_goals`) is bookkeeping; not read.  `lat_peak_m`: §3.

## 6. `nav_command` and `nav_30s`

* **`nav_command` built:** `nav_command()` `s2_geom_emit_v7.py:692-716`: NAV_TURN_L/R iff `seq[0]` (the FIRST sustained segment after the anchor, up to 35 s ahead) satisfies `is_turn`; FOLLOW otherwise or when the turn is suppressed.  `args.distance_m` = anchor-relative arc, `args.time_s` = anchor-relative start (correct).  Provenance `ego-future`, oracle.
* **Trainer meaning** (`refc_v3_train.py:3880-3912`, `--nav-from-v7`): the **same** token on **every** window of the clip (train 2,752 follow / 779 left / 838 right clips).
* **MEASURED time structure** (train, 1,675 turn tokens): the turn starts a median **7.3 s** after the anchor (p25 1.0 s, p75 17.5 s, p90 26.2 s); 52.7 % start > 6 s and 42.0 % > 10 s after the anchor; the anchor is itself at raw 8.0 s, so for windows at NOW=2 s the turn is ≈ 13 s away.
* **MEASURED per-window precision** (independent turn = ≥ 30° inside any 3 s, within the window's own next 6 s; `raw/census_nav_train.json`; 576,553 windows whose next 6 s lie inside the cache): of 213,371 windows carrying a TURN token only **24.7 %** (52,728) see a same-side turn begin in their own next 6 s; conversely 64.9 % of the 81,248 windows that do have a turn in the next 6 s carry the right-side token, and 6.1 % of FOLLOW windows (22,315/363,182) have a turn starting.  In the supervised range |NOW−t0| ≤ 2 s: L-token 35.6 % (11,015/30,916), R-token 33.2 % (11,050/33,268) same-side hits.  So the bare token is **a clip-level route label, not a per-window command**; `nav_30s` carries the time/distance and refcv7 did not read it.
* **`nav_30s` built:** `build_v8_nav30s.py` — rows 8, 6, 7 of the verdict table; distance arithmetic corrected 2026-09-09 (`fix_v8_nav30s_arc.py`, anchor-relative arc).

## 7. `SPEED_BAND` → `speed_max_input` → the fed 4-way ceiling

* **Built:** `SPEED_BAND` `s2_geom_emit_v7.py:309-315`: `v_lo = min v`, `v_hi = max v` of the 10 Hz ego speed over **[anchor+2, anchor+6]** (the log-derived `egomotion_source.load` track, resampled to 10 Hz), unconditional, `held` descriptive.  `speed_max_input.v_max_ms` copies `v_hi_ms` (`build_v8_speedmax.py`).  The trainer's sidecar `refcv6_speed_max_v8_*.jsonl` quantises `v_hi_ms` to the containing window of the 4-way ladder **{30,50,100,120} km/h** (`tanitad/refs/refcv6_max_speed.py:91`, config `refcv6_max_speed._derivation`): (0,30]→30, (30,50]→50, (50,100]→100, (100,120]→120, >120 clamps.  Note the record's own `speed_max_input` block carries a *different* 8-step ladder (20…130) that refcv7 did not use.
* **Trainer meaning:** one value per clip fed on **every** window (`window_ceiling_frac 1.0`, config).
* **MEASURED:** the sidecar bin equals an independently written ladder on 4,369/4,369 + 139/139 clips; `v_hi` equals the cache-pose recompute at median 0.005 m/s.  Validity of the *fed* value per window: RESULT §4.

## 8. `g_tac.goals` — 22 tokens, provenance and positivity  (train n=4,572; `raw/census_nav_train.json`)

Positive when (the `goals` dict carries the token); absence is a supervised **negative only for geometry-provenance tokens** (`negatives="measured"`, `v7_labels.py:917-1000`; refcv7 config `tac_goal_stats`: 17 of 22 trainable).

| token | n | provenance (as stored) | positive when / evidence | independent check |
|---|---|---|---|---|
| FOLLOW_LANE | 3,629 | geometry | no other lateral goal fits the exclusion matrix (`:411-417`) | default; not independently testable |
| TURN_L / TURN_R | 275 / 259 | geometry (no `provenance` key; `corroboration` confirmed 211+187, uncorroborated 64+72) | largest in-plan (clipped to [2,6]) `is_turn` segment with start < 6.0 s, not suppressed (`:225-243`) | MEASURED: 235/270 (87.0 %) and 211/247 (85.4 %) have an independent same-side ≥ 30° turn in [2,6]; goal ABSENT but independent turn present: 78/4,099 and 85/4,122 |
| YIELD_FOR_TURN_L / _R | 21 / 20 | geometry | a turn goal plus a ≥ 0.5 s stop in [0, turn start) (`:239-242`) | not separately tested |
| STOP_POINT | 327 | geometry | stop episode (v ≤ 0.5) in [2,6], no TURN/YIELD goal, not "already stopped and launching" (`:279-289`) | MEASURED: 291/291 scored positives have an independent v<0.5 sample in [2,6]; 28 stops without a token (the launch exclusion) |
| SPEED_BAND | 4,572 | geometry | unconditional | §7; masked from training (no supervised negative) |
| CORRIDOR_OFFSET | 860 | vlm-cot | CoT mentions an offset; `time_basis` untimed 859/860; never groundable (`not_checkable`) | masked from training |
| EVADE_IN_CORRIDOR | 240 | vlm-cot 238 + alpamayo-structured 2 | CoT/segment says evade **and** `lat ∈ {NUDGE_L,NUDGE_R}` (`:329-335`) — inherits NUDGE's defect (row 1) | untested |
| YIELD | 609 | vlm-cot, all `disputed` | CoT text | masked |
| GAP_TARGET 368 · MERGE 79 · OVERTAKE_VEHICLE 20 · REACT_ON_ONCOMING 333 · TAKE_EXIT_L 21 · TAKE_EXIT_R 128 · LANE_CHANGE_L 23 · LANE_CHANGE_R 15 | — | vlm-cot (TAKE_EXIT_L: 20 cot + 1 structured) | CoT keyword; `time_basis` untimed on 77–100 %; box- or component-grounded on 0–70 %, `disputed: true` on 30–100 % | VLM text, **no geometry check possible**; lateral-evidence rule (`:329-335`) makes LANE_CHANGE_* require a NUDGE |
| TRAFFIC_LIGHT_REACT 18 · _RED 376 · _GREEN 363 · _YELLOW 22 | — | vlm-cot | CoT names the colour; `untimed` 344/363 (GREEN) … ; box-grounded on 17–32 % | VLM text; PI ruling 2026-09-16 on absence-as-negative not applied in refcv7 (`cot_absence_negative: null`) |

No token carries a time guarantee for the VLM class: `t_nominal_s = 4.0` is the band midpoint (an assumption, flagged `time_basis: untimed`); the trainer applies them to NOW∈[6,10] by the same window rule as geometry.

## 9. `turn_suppression`

Emitted `s2_geom_emit_v7.py:885-898` (PI 2026-08-29: geometry turn + nudge/pass-parked text ⇒ not a turn).  MEASURED 66 records (30 left, 36 right); all 66 carry `a_tac.lat = NUDGE_L/R` and `nav_command = NAV_FOLLOW_ROAD`; keys: `applied, dyaw_deg, evidence, radius_m, rule, side, t_start_s` (note: `applied`, not `suppressed` — row 6).  These 66 turn manoeuvres are, by the PI rule, labelled NUDGE — i.e. 6.4 % of the 1,028 NUDGE labels are by construction real ≥ 30° geometric turns relabelled by a text rule (so they sit in the "independent turn" column of my matrices: 55 of the 1,028 NUDGE records have an independent TURN over [0,6]).

## 10. `alpamayo.*`, `cot_source`, `scene`, `tac_SIT`, `semantics`, `strata`, `lane_change_text`

Not read by the trainer (dropped at `v7_labels.py:376-417`).  `alpamayo.lateral.alpamayo_side` ← `meta_action.lateral` ("left"/"right"/"straight" by substring, `alpamayo_records.py:181-190`), anchored at **5.1 s** (`:149`) not 8.0 s; `agree` rows 9 above.  `strata.road_class/daynight_clock` are used by D2 only for stratification and were not audited (UNVERIFIED).
