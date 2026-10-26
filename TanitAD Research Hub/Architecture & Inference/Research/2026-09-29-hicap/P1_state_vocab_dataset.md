# P1 — newest programme state: the vocabulary, the corpus, refcv7, audio

**Stream P1 · 2026-09-29 · read-only research, nothing trained, nothing pushed.**
Written incrementally (banked section by section).

## 0. Read this first — where the newest state lives, and evidence classes

* The GitHub `origin/handoff` branch (tip `901ccac`, 2026-08-15) is **NOT the newest state**. The PI's Google Drive
  holds the programme through 2026-09-29 (the Drive copy of `MODEL_REGISTRY.md` was modified 2026-09-15; the
  vocabulary/corpus/REF-C documents below are dated 2026-08-23 … 2026-09-10). Nothing after `handoff` on GitHub touches
  REF-C (branches `fix-b1-traj-head`, `validate/axis6-precheck` are 2026-07-12; `claude/google-drive-domains-5ml2j9`
  is the 2026-09-24 camera-calibration work, unrelated). Checked: `git ls-remote` (all heads), open PRs #2-#6
  (`mcp github list_pull_requests`), `search_code "refcv7" repo:sayoood/TanitAD` → 0 hits.
* Evidence classes used below. **Every Drive-sourced fact is INHERITED** from the named Drive file: I read the
  file's text; I did **not** open the label blobs, the HF repos or the checkpoints (no network to HF, no pod).
  `MEASURED (ours)` is used only where I computed something in this repo today.
* Drive file ids are given so a later reader can re-open the exact file (`read_file_content`).

| short name | Drive title | Drive id | modified |
|---|---|---|---|
| VOCAB_PY | `vocab_v7.py` (35,110 B) | `1KPzkoQZU3lMZol8bbFzpW62ZAlgAmVJC` | 2026-09-09 |
| FILL | `VOCABULARY_AND_FILLING.md` (9,872 B) | `1_fJaA-5PFIk4swyNlt89L5ItDWtB8sDc` | 2026-09-06 |
| CARD81 | `DATACARD_v8.1_shipped.md` (12,276 B) | `1CniQv_wLR6Hht42jBm8rhB-dR8wIZ_8B` | 2026-09-09 |
| CARD7 | `DATACARD.md` (v7 original, 2,447 B) | `1dZ3KjddnYA6dUUg_dBGQ1EqFWysq7kRI` | 2026-08-29 |
| V8MAN | `V8_MANIFEST.json` (13,761 B) | `1djECqAyIT4mUyPx4wycnkxlqyaic6o6D` | 2026-09-09 |
| CONS | `SPEC_V7_LABELS_CONSUMER.md` | `1i3gsS4YATRK_KztsflOzvYEALHNbdcFr` | 2026-08-30 |
| WIRE | `SPEC_V7_LABEL_TRAINER_WIRING.md` | `1ooVyEu-J9VaBv73MRpjCHHC9C-6nNXYK` | 2026-08-30 |
| B1PREP | `B1_TRAINING_PREP.md` | `1VXIe7RcaXPZCa3BBJMQMXqLQN3X0zAX7` | 2026-08-29 |
| GLOSS | `CORPUS_TOKEN_GLOSSARY.md` | `1Uh6Pw5murJrGbmCIvhL00HslNKDwDsv8` | 2026-09-02 |
| RECIPE | `V7_RECIPE_AND_SCALEUP.md` | `1N9mt-YfOuTWJ7H1sgG9ZJqX1f1XRxJwE` | 2026-08-30 |
| PREREG6 | `PREREG_REFCV6.md` (37,703 B) | `1DcskOQqlJHEe9NXXFeVkCv0tvrl2m3xW` | 2026-09-10 |
| CENSUS | `v7_vocab_reach_census.py` | `1hlwBM95NPeBagvjLP68Z4Sl1XyiCW20V` | 2026-09-07 |
| V3GV | `V3_GOAL_VOCABULARY_V1.md` (frozen 2026-07-19) | `1mFkP3SCxqJ2qaNW_ENcMeIgpem4YVrfY` | 2026-09-06 |
| HIERV | `HIERARCHY_VOCABULARY.md` (ancestor, 2026-08-11) | git `origin/handoff` (read; also Drive `1X-CIvnUYjcQ5MG_Nx8Guv4-BX22W-fWD`) | — |

---

## 1. THE VOCABULARY — which one is "the last"

### 1.1 Lineage (oldest → newest) — three vocabularies, only the last is what the newest trainers consume

| version | where | size | status |
|---|---|---|---|
| **v3 goal vocabulary v1** (frozen 2026-07-19) | V3GV | strategic 34 tokens / 7 slots + tactical 80 tokens / 11 slots = **114 tokens / 18 slots** (`MISSION ROUTE LANEOBJ SPEEDPOLICY STYLE RISK ODD` / `VTARGET VSOURCE LONMODE LATMANEUVER HEADWAY DYN RULECTX SIGNAL INTERACT TACPOINT LIGHTSTATE`) | superseded; pinned in `stack/tanitad/lake/vocab.py` (on `origin/handoff`) |
| **HIERARCHY_VOCABULARY** (2026-08-11, PI redesign) | HIERV (`origin/handoff`, 161 lines, read completely) | `g_str` 11 tokens (`KEEP_CORRIDOR … NONE_ABSTAIN`), `a_str` 6, `g_tac` 9 (`ANCHOR_GOAL … TRAFFIC_LIGHT_REACT`), `a_tac` LAT 6 / LON 6 (counts match `models/v6.py:132-160` on the branch) | ancestor. Introduced factored LAT×LON, the 6 s / 60-step trajectory contract (§4b), the typed slots `within_m / by_time_s / at_arc_m / hold_for_s`. **Not what the labels carry.** |
| **v7 vocabulary — FROZEN 2026-08-23** (`vocab_v7.py`, "PI redesign") | VOCAB_PY | **56 declared token slots**: `g_str` 8 + `a_str` 7 + `g_tac` 22 + `a_tac.lat` 8 + `a_tac.lon` 8 + `nav_command` 3 | **THE CURRENT VOCABULARY.** Frozen by PI ("stabilize now the vocabulary and keep it constant"); `assert_frozen()` raises on any emitted token outside it; adding a token = a new `*_V8` tuple behind a version switch. `schema_version` = `s2-geom-v7`, `vocab` = `v7`. |
| label releases on the v7 vocabulary | CARD81, V8MAN | v7 → v7.1 → v7.2 → **v8 / v8.1 (current)** | see §1.6. The **vocabulary is identical across all four**; releases add *fields*, not tokens. |

Token ids = **tuple index in `vocab_v7.py`** (the consumer builds `HEADS[...]` from the tuples; head width = full
vocabulary, loss mask applied to the loss, never to the head width — CONS §3). Ids are listed below in that order.

### 1.2 Bands (the temporal contract every token lives in) — VOCAB_PY header item 1; CONS §5

| band | seconds (anchor-relative; anchor = `RAW_T0_S` = 8.0 s on the raw recording timeline) | layer |
|---|---|---|
| operative | [0, 2] | continuous (a, κ) — no tokens |
| tactical | [2, 6] | `g_tac`, `a_tac.lat`, `a_tac.lon` |
| *unassigned* | (6, 8) — "nothing owns 6-8 s"; manoeuvres landing only there are reported in `bands.unassigned_manoeuvres`, not absorbed | — |
| strategic | [8, 30] | `g_str`, `a_str` |

`bands`, `t0_s`, `horizon` are **per record** — read them per clip (CONS §5). Record `bands.tactical_s != bands.strategic_s`
on 301/301 sampled v7 records (WIRE §3.1) ⇒ band test must be evaluated **per family**.

### 1.3 STRATEGIC — `g_str` (goal, 8 tokens) and `a_str` (action, 7 tokens)

Args on every strategic token: `STRATEGIC_ARG_SLOTS = (within_m, by_time_s)` (a named dict, not a vector — WIRE §3.2).
Suffix `_FOLLOW_ROUTE` = "the overnext manoeuvre, in order to follow the route".
Counts = **per FIELD**, v8 census, n = 4,719 clips (train 4,572 + eval 147), source FILL §4-5. (FILL abbreviates the
`a_str` names by dropping the `_FOLLOW_ROUTE` suffix; the code names are used here.)

| id | `g_str` token | n | status | definition (VOCAB_PY `DEFINITIONS`) |
|---|---|---|---|---|
| 0 | `FOLLOW_ROUTE` | 3,041 (64.44 %) | HEALTHY | no overnext manoeuvre in the band; stay on the route. (Code comment says "~84 % of clips"; the v8 census says 64.44 %. Census is the later measurement.) |
| 1 | `TURN_LEFT_FOLLOW_ROUTE` | 585 | HEALTHY | overnext manoeuvre is a left turn, then the route continues |
| 2 | `TURN_RIGHT_FOLLOW_ROUTE` | 624 | HEALTHY | as above, right |
| 3 | `STOP_AT_FOLLOW_ROUTE` | 469 | HEALTHY | overnext event is a stop (light, sign, queue) |
| 4 | `EXIT_LEFT_FOLLOW_ROUTE` | **0** | NOT_EXTRACTABLE | leave the road to the left (ramp/split) — CoT states exits without a time; geometry cannot tell an exit from a bend; no map |
| 5 | `EXIT_RIGHT_FOLLOW_ROUTE` | **0** | NOT_EXTRACTABLE | as above |
| 6 | `LANE_CHANGE_L_FOLLOW_ROUTE` | **0** | NOT_EXTRACTABLE | route-required lane change left — no timestamp in CoT |
| 7 | `LANE_CHANGE_R_FOLLOW_ROUTE` | **0** | NOT_EXTRACTABLE | as above |

| id | `a_str` token | n | status | definition |
|---|---|---|---|---|
| 0 | `HOLD_MAIN_ROAD` | 2,468 (52.30 %) | HEALTHY | nothing to prepare for (replaced `HOLD_CORRIDOR`: a corridor is lane-level, this layer is route-level) |
| 1 | `PREPARE_TURN_L_FOLLOW_ROUTE` | 585 | HEALTHY | begin setting up for a left turn **not** in the current 6 s plan; args carry `within_m`, `by_time_s` |
| 2 | `PREPARE_TURN_R_FOLLOW_ROUTE` | 624 | HEALTHY | as above, right |
| 3 | `PREPARE_STOP_FOLLOW_ROUTE` | 469 | HEALTHY | set up for a stop beyond the tactical horizon |
| 4 | `PREPARE_EXIT_FOLLOW_ROUTE` | **0** | NOT_EXTRACTABLE | set up to leave the road |
| 5 | `PREPARE_LANE_CHANGE_FOLLOW_ROUTE` | **0** | NOT_EXTRACTABLE | position for a route-required lane change |
| 6 | `RESUME_CRUISE_FOLLOW_ROUTE` | 573 | HEALTHY | the constraint forcing a slowdown ends inside the band |

⛔ `REDUCE_TO_FOLLOW_ROUTE` is **not** in the vocabulary (removed by PI 2026-08-23, `s2_geom_emit_v7.py:613-619`;
MEASURED 0 emissions in v8 — FILL §4). Any document listing it reads a different artifact.
**Derivation / provenance (strategic):** from the **ego's own future path** ("geometry", `provenance` per record) placed in the
[8, 30] s band; the strategic layer is the weakest: 4 of 8 goals and 2 of 7 actions are empty, one root cause "no time"
(`chain_of_causation` carries a time expression on 6/4,729 clips = 0.13 %) — FILL §5. **No VLM is used for the strategic tokens
that are populated**; they are ego-future geometry ⇒ labels-may-use-ego (PI rule) but **oracle-like if fed at inference**.

### 1.4 TACTICAL ACTIONS — `a_tac.lat` (8) and `a_tac.lon` (8): factored, **never flatten** (CONS §2)

Args: `LAT_ACTION_ARG_SLOTS = (within_m,)`, `LON_ACTION_ARG_SLOTS = (v_target_ms, within_m)`. `serves_goals` carries
`lat_serves` / `lon_serves` (GOAL_ADMISSIBLE_LAT/LON maps). `truncated` = false on all 4,719.

| id | `a_tac.lat` | n | status | definition |
|---|---|---|---|---|
| 0 | `LANE_KEEP` | 3,057 (64.78 %) | HEALTHY | hold the lane centre |
| 1 | `LANE_CHANGE_L` | **0** | ⛔ UNREACHABLE (emitter has no branch) | — |
| 2 | `LANE_CHANGE_R` | **0** | ⛔ UNREACHABLE | — |
| 3 | `ABORT_LC` | **0** | ⛔ UNREACHABLE | abandon a begun lane change |
| 4 | `NUDGE_L` | 502 | HEALTHY | small lateral displacement left, in lane |
| 5 | `NUDGE_R` | 613 | HEALTHY | small lateral displacement right, in lane |
| 6 | `TURN_L` | 280 | HEALTHY | (junction turn, geometry-gated) |
| 7 | `TURN_R` | 267 | HEALTHY | |

| id | `a_tac.lon` | n | status | definition |
|---|---|---|---|---|
| 0 | `FOLLOW` | 175 (3.71 %) | HEALTHY | track the lead vehicle's speed holding a gap |
| 1 | `CRUISE` | 1,287 (27.27 %) | HEALTHY | hold speed, \|dv\| < 1.0 m/s across the plan |
| 2 | `YIELD_MERGE` | **0** | NOT_EXTRACTABLE | slow to let a merging agent in (needs `MERGE` claim ∧ measured decel; text route gives 13 clips) |
| 3 | `BRAKE_TO` | 927 (19.64 %) | HEALTHY | decelerate to a target speed or a stop point |
| 4 | `CREEP` | 141 | HEALTHY | move slowly forward |
| 5 | `HOLD` | 133 | HEALTHY | stationary, brakes applied |
| 6 | `ADAPT_SPEED_FOR_CURVE` | 1,018 (21.57 %) | HEALTHY | speed change for curvature (junction turn or bend); set automatically on a detected turn |
| 7 | `ACCELERATE` | 1,038 (22.00 %) | HEALTHY | dv > +1.5 m/s across the plan |

**Derivation (tactical actions):** *purely geometric* — thresholds `DV_BRAKE_MS`, `DV_CRUISE_MS`, `DV_ACCEL_MS`, `V_STOP_MS`
over the 2–6 s ego speed profile (LON); largest in-plan manoeuvre decides LAT: gated turn → `TURN_*`, contested turn (obstacle
pass, PI 2026-08-29) → `NUDGE_*`, else `EM.analyse().lateral_class` → `NUDGE_*`/`LANE_KEEP` (FILL §1-2). The emitter has
exactly three assignments to `lat` (`s2_geom_emit_v7.py:446, 449, 452`) ⇒ lane changes are unreachable by construction.
⇒ **ego-derived labels**: legitimate as labels (PI rule "labels may use ego"), inadmissible as inference inputs.
Source-of-truth check: ⚠️ FILL is a doc about the census; the emitter file itself was not read by me.

### 1.5 TACTICAL GOALS — `g_tac.goals` (22 tokens, **multi-label SET**, never empty, no ABSTAIN)

Uniform arg slots: `ANCHOR_ARG_SLOTS = (goal_x_m, goal_y_m, t_reach_s, band_s)` — the **anchor** (the point the ego reaches
at the end of the band) is the *argument frame*, **not a goal token** (PI). Per-goal metadata on every emitted goal (one nesting
level deeper — `g_tac.goals.<TOKEN>`): `provenance`, `disputed`, `grounded`, `corroboration`, `time_basis`, `t_nominal_s`
(CONS §0: 2,343 records = 49.6 % carry ≥1 DISPUTED goal). Two floors: `GOAL_MIN_N_FOR_TRAINING = 30`, `GOAL_MIN_N_FOR_METRIC = 200`.
n = clips carrying the goal (v8 census, FILL §3; ⚠️ per-field, multi-label):

| id | token | n | class | derivation | definition (short) |
|---|---|---|---|---|---|
| 0 | `FOLLOW_LANE` | 3,748 (79.42 %) | HEALTHY | geometry | stay in lane; fallback when nothing else applies |
| 1 | `TURN_L` | 280 | HEALTHY | geometry | left turn within the 6 s plan (a turn inside the plan is TACTICAL) |
| 2 | `TURN_R` | 267 | HEALTHY | geometry | |
| 3 | `YIELD_FOR_TURN_L` | 23 | **UNDERPOWERED (<30)** | CoT | left turn that must first give way; one goal not two |
| 4 | `YIELD_FOR_TURN_R` | 20 | **UNDERPOWERED** | CoT | |
| 5 | `YIELD` | 626 | HEALTHY | CoT in the blob (609 CoT-sourced in v7.2 train) although **absent** from the declared `TACTICAL_GOAL_NEEDS_PERCEPTION` set | give way without a turn (sign, merge, hazard) |
| 6 | `STOP_POINT` | 336 | HEALTHY | geometry | come to rest at a place; args `within_m`, `hold_for_s` |
| 7 | `SPEED_BAND` | 4,719 (100 %) | HEALTHY | geometry | **always present**: target speed interval over 2–6 s, args `v_lo_ms / v_hi_ms / band_s`; a stop = band 0.0–0.0 |
| 8 | `CORRIDOR_OFFSET` | 885 | HEALTHY | **CoT terms** per `vocab_v7.py` (PI's 2026-08-28 CoT-term route, 906 clips, 89 % side agreement; "perception-only, forever" — arc-removed geometry cannot see a held in-lane offset). ⚠️ FILL §3 lists it under "geometric … from poses" — the two Drive docs disagree; the code comment and `CORRIDOR_OFFSET.disputed` on 885 records (disputed = CoT provenance) support CoT | hold a lateral offset inside the lane |
| 9 | `EVADE_IN_CORRIDOR` | 246 | HEALTHY | **fused** (CoT claim ∧ lateral evidence; 74 % of emissions once had no lateral motion) | lateral adjustment for a STATIC obstacle or VRU; stays in corridor |
| 10 | `OVERTAKE_VEHICLE` | 21 | **UNDERPOWERED** | CoT | pass a SLOWER MOVING vehicle ahead (moving ⇒ OVERTAKE; static/VRU ⇒ EVADE) |
| 11 | `MERGE` | 84 | HEALTHY (<200 metric floor) | CoT | join a stream of traffic |
| 12 | `GAP_TARGET` | 382 | HEALTHY | CoT | aim for a gap in an adjacent stream |
| 13 | `REACT_ON_ONCOMING` | 344 | HEALTHY | CoT | respond to oncoming traffic (renamed from `WAIT_FOR_ONCOMING`; dominant phrasing is lateral clearance) |
| 14 | `TAKE_EXIT_L` | 21 | **UNDERPOWERED** | CoT terms, not geometry (PI) | leave the road left |
| 15 | `TAKE_EXIT_R` | 133 | HEALTHY (<200 metric floor) | CoT terms | leave the road right |
| 16 | `TRAFFIC_LIGHT_REACT` | 18 | **UNDERPOWERED** | CoT (colour not extracted) | |
| 17 | `TRAFFIC_LIGHT_REACT_RED` | 384 | HEALTHY | CoT (**Alpamayo teacher signal**) | react to a light observed red |
| 18 | `TRAFFIC_LIGHT_REACT_YELLOW` | 25 | **UNDERPOWERED** | CoT (teacher) | yellow/amber |
| 19 | `TRAFFIC_LIGHT_REACT_GREEN` | 378 | HEALTHY | CoT (teacher) | a reaction, not the absence of one |
| 20 | `LANE_CHANGE_L` | 23 | **UNDERPOWERED** (populated as a *goal* although unreachable as a *lat action*) | CoT text | move to the left lane within the plan |
| 21 | `LANE_CHANGE_R` | 16 | **UNDERPOWERED** | CoT text | |

⚠️ **Declaration vs data disagree on four tokens** (`v7_labels.py`, measured on the v7.2 train blob): `YIELD` (609), `EVADE_IN_CORRIDOR` (238), `LANE_CHANGE_L` (23), `LANE_CHANGE_R` (15) are CoT-sourced in the blob while absent from the frozen perception set — so the `negatives="measured"` policy reads provenance from the blob, not from the declaration.

`TACTICAL_GOAL_NEEDS_PERCEPTION` (may NOT be emitted from ego geometry): `CORRIDOR_OFFSET, GAP_TARGET, REACT_ON_ONCOMING,
OVERTAKE_VEHICLE, MERGE, TAKE_EXIT_L/R, TRAFFIC_LIGHT_REACT{,_RED,_YELLOW,_GREEN}`. Per FILL §3 the perception goals come **only from
the Alpamayo `chain_of_causation` text, never from geometry**; `provenance` values seen: `geometry`, `vlm-cot` (CENSUS
`provenance_divergence`). **Populated: 22/22** (the only fully populated family) but 8 have n<30 (`TRAFFIC_LIGHT_REACT_YELLOW` 25,
`YIELD_FOR_TURN_L` 23, `LANE_CHANGE_L` 23, `OVERTAKE_VEHICLE` 21, `TAKE_EXIT_L` 21, `YIELD_FOR_TURN_R` 20, `TRAFFIC_LIGHT_REACT`
18, `LANE_CHANGE_R` 16).

**`TACTICAL_GOAL_EXCLUSIVE`** (**22 pairs** — counted by me from `vocab_v7.py`; pairs that may never co-occur; a multi-label head is *checked* against this): `TURN_L|TURN_R`,
`TURN_L|FOLLOW_LANE`, `TURN_R|FOLLOW_LANE`, `LANE_CHANGE_L|LANE_CHANGE_R`, `LANE_CHANGE_L|TURN_L`, `LANE_CHANGE_R|TURN_R`,
`YIELD_FOR_TURN_L|YIELD_FOR_TURN_R`, `YIELD_FOR_TURN_L|TURN_R`, `YIELD_FOR_TURN_R|TURN_L`, `STOP_POINT|FOLLOW_LANE`, the
six traffic-light pairs (`RED|GREEN`, `RED|YELLOW`, `GREEN|YELLOW`, and plain `TRAFFIC_LIGHT_REACT` ✕ each of the three colours),
`OVERTAKE_VEHICLE|FOLLOW_LANE`, `OVERTAKE_VEHICLE|EVADE_IN_CORRIDOR`, `TAKE_EXIT_L|TAKE_EXIT_R`, `TAKE_EXIT_L|TURN_R`,
`TAKE_EXIT_R|TURN_L`, `MERGE|FOLLOW_LANE`. (`SPEED_BAND|STOP_POINT` was removed 2026-08-27.)
**`GOAL_ADMISSIBLE_LAT / GOAL_ADMISSIBLE_LON`**: per-goal maps of which tactical actions may serve it (e.g. `TRAFFIC_LIGHT_REACT_RED`
→ lat `{LANE_KEEP}`, lon `{BRAKE_TO, HOLD, CREEP}`); `action_serves_goals(lat, lon, goals)` returns the per-axis verdict —
the *goal→action* link the PI asked for ("documentation where goals are linked to possible actions").

### 1.6 NAV COMMAND — a model **input**, not a label (3 tokens)

`NAV_COMMAND_TOKENS = (NAV_FOLLOW_ROAD, NAV_TURN_L, NAV_TURN_R)`; args `(distance_m, time_s)`; `NAV_PROVENANCE = (nav-system, ego-future)`.
Counts (scalar `nav_command`, v8): `NAV_FOLLOW_ROAD` 2,993 (63.42 %) · `NAV_TURN_R` 902 · `NAV_TURN_L` 824 (FILL §6).
⛔ **Provenance is `ego-future` on 4,719/4,719** (key on `provenance`, not on the `oracle` boolean, which is absent on 529 records — CONS §1).
The Drive glossary records a PI ruling of 2026-09-04 that reverses the earlier "oracle" stigma: *nav command = GROUND TRUTH and a
FIRST-CLASS INPUT, a route signal like a map router supplies; `os` (nav fed) is the deployment-relevant arm and `os_navzero` a
robustness ablation* (`VOCABULARY.md` glossary, id `1w6m5Wipxvp3h3scKod7EqxzMxM3PG3Or`, "nav command (v7.2)" row) — with the nuance
that ours is noiseless and perfectly timed. Note the conflict with `CLAUDE.md`'s 2026-08-03 vision-only rule (a *supplied route is
optimistic on PhysicalAI*) — it is a PI-ruled exception for the route channel only; v8's `nav_30s` and `speed_max_input` keep the
`ego-future`/`oracle` stamps.

### 1.7 The v8 / v8.1 additions — **fields, not vocabulary** (CARD81, V8MAN)

| field | coverage | what it is | how to use it |
|---|---|---|---|
| `tac_SIT` | 1,585 of 4,719 clips cite ≥1 term (33.6 %) | 13 situation terms from `alpamayo.chain_of_causation`, three-valued **CITED / NOT_CITED — never present/absent**; each citation ships its evidence sentence; `yield` is emitted under `demand`, not `cited`. Top: intersection 444, parked_car 350, pedestrian 255, oncoming_vehicle 212, crosswalk 193, pedestrian_in_crosswalk 179, roundabout 163, cyclist 85, roadwork 50. `emergency_vehicle`, `highway`, `blocking_obstacles` = insufficient support. **FN rate UNMEASURED; precision UNMEASURED.** | evaluation strata / auxiliary; not a ground-truth presence label |
| `nav_30s` | 100 %; 5,418 entries (2,926 FOLLOW_ROAD / 1,200 TURN_L / 1,292 TURN_R) | every nav manoeuvre in [0, 30] s, **same 3-token alphabet**; times anchor-relative, distance = **arc length** from the anchor; `distance_end_m` null on 535 | `ego-future` on every entry ⇒ **ORACLE input**; must be recorded as `oracle_nav_payload: nav_30s` + `n_entries`. Additive; does NOT replace scalar `nav_command` (a live input `nav_conditioning.py` refuses a batch without). |
| `lane_change_text` | EXECUTED 105 (L 74 / R 31); ANTICIPATED 57; NEGATED 15; MULTI 1; NO_SIDE 1; NOT_CITED 4,540 | text-only from `chain_of_causation`; `geometry_used = false` on every record (PI 2026-09-07 after three geometry detectors scored 53.2/62.1/61.0 % side agreement vs 50 % chance) | **no timestamp by construction**; precision UNMEASURED |
| `speed_max_input` (+`v_max_bucket`) | 100 % | `SPEED_BAND.v_hi_ms` in m/s (median 10.07, mean 11.46); bucket snaps UP to {20,30,50,70,80,100,120,130} km/h (histogram 832/958/1,645/645/170/238/144/87) | **INPUT channel** (user/nav-supplied at inference); training value = ego's *realised* max speed over [anchor+2 s, +6 s] ⇒ train/deploy mismatch, stamped `ego-future` + `oracle: true`; required controls `hold` and `shuffled`. 75 % of intersection clips snap to ≤30 km/h (stopped ego). |

**Traffic lights** (`g_tac.goals`): 805 emissions with colour — RED 384, GREEN 378, YELLOW 25, colourless 18. ⛔ **Alpamayo-derived
teacher signals, never "GT traffic light"** (PI 2026-09-07 "no vlm, we stick to the alpamayo labels as teacher signals"); 182/805
carry an independent 2D box (0 contradicted), 601 never checked; grounding boxes carry **0 colour attributes**. Consequence: an
**ego-only control is mandatory** on any traffic-light head (CARD81).

### 1.8 `trainer_conditions_mandatory` — text of the five conditions

Both cards say "Every consumer MUST apply the five `trainer_conditions_mandatory` in `MANIFEST.json`" (CARD7, CARD81). **The five
sentences themselves are in `MANIFEST.json`, which is a Drive `.json`/HF file I could not open as text; UNVERIFIED verbatim.**
Their content is reconstructible from CONS/B1PREP §2 (the "D-LABEL-GT conditions", implemented once in
`stack/tanitad/data/v7_labels.py`): (1) mask the `NOT_YET_EXTRACTABLE` classes on the loss (head width stays full-vocab, per-head
arithmetic — `LANE_CHANGE_L/R` tac-lat were "two dead logits" until masked empirically); (2) strategic-skew weighting computed
from the loaded split, not hard-coded; (3) `disputed` policy exposed as a filter (default = include with flag) — audit only, never
a training input; (4) `time_basis`/`t_nominal_s` band placement (untimed → band midpoint per PI); (5) `turn_suppression` passed
through as audit. Plus the `allow_oracle_nav=True` stamp gate for nav. Treat as INHERITED-reconstructed.

---

## 2. THE CORPUS the newest REF-C arms train on — `tanitad-v7-training-corpus`

Sources: CARD81 (shipped card), CARD7, V72MAN/V8MAN, WIRE, B1PREP, GLOSS, `HF_CARD_tanitad-refc-v3.md` (Drive id
`1m8D-6nZ58bFU_pxZ3G8qEh9v7o2XT-_A`), PREREG6. All INHERITED (Drive files read verbatim; blobs not opened).

### 2.1 Identity and size

| item | value |
|---|---|
| HF repo | **`Sayood/tanitad-v7-training-corpus`** — **PRIVATE** dataset; research use inside TanitAD; sources `nvidia/PhysicalAI-Autonomous-Vehicles` (research licence) + `Sayood/tanitad-alpamayo2-augmentation` |
| corpus id / hash | **`a48251e89c7a8603`** = sha256 over the sorted clip ids (B1PREP §0); label-blob lineage `121a8d93` at directive time |
| size | **4,719 clips / 26.2 h** of front-camera driving (`alpamayo ∩ egomotion`); ~20 s per clip |
| released | 2026-08-29 (PI authorisation) as the training baseline for **v7f, refcv3, refav1, refd** (i.e. the REF-C line is a consumer). Ground-truth register row `D-LABEL-GT`. |
| parity | ⛔ **NEW comparability domain.** Does NOT re-select `physicalai-train-e438721ae894` (2,376 eps, skip-hash `f09e44db`), which stays untouched. MEASURED in GLOSS: parity ∩ v7.2 = 190/2,400 (7.9 %); B1 ∩ v7.2 = 4,572/4,713 (97.0 %); **parity is not a subset of B1**. Arms trained here are never same-data-compared to parity arms (B1PREP §0). refcv3's own `config.json` records `v2_parity.parity false` (HF card §4.1). |
| train / eval | **4,572 train + 147 eval = 4,719** (v7.2 split-by-file; eval = 3.12 % of the corpus). Eval composition: urban 64.6 / intersection 18.4 / highway 17.0 %; day 55.8 / night 44.2 %. Train: urban 64.9 / intersection 18.8 / highway 16.3 %; day 53.6 / night 46.4 % (V72MAN). |
| episode caches | **`physicalai-b1-w120-256x640cyl`: 4,713 episodes / 177,998,547,213 bytes** (= 4,572 train + 141 eval; `_geometry.json` sha `f0d34b84`), plus **`physicalai-b1-EVAL6-w120-256x640cyl`: 6 episodes, 208 MB** (the 6 v7.2-eval clips that are also in the deployed val40, dropped by the parity gate). **Eval set = union of both roots = 147** (a loader pointed at one root is silently short — WIRE §6). Train cache asserted val40-clean by digest (0 of 4,713). |
| OOD split | **none declared for this corpus** (UNVERIFIED — no doc I read names an OOD split of v7). The older OOD-val cache `epcache_oodval_290/` (291 files, 34.02 GB) lives in `Sayood/tanitad-ph0-aug120` (handoff line). |

### 2.2 Repo layout (as named in the cards)

`labels/` (`s2_labels_v7.jsonl.gz` 4,719 rec · `s2_labels_v7.1.jsonl.gz` · `s2_labels_v7.2_{train,eval}.jsonl.gz` · `s2_labels_v8_{train,eval}.jsonl.gz`)
· `index/` (`clip_to_chunk.parquet` [also carries the raw train/val/test split], `front_wide_cy.parquet` [`clip_id, width, height, cx, cy, rig`],
`clip_index_v7.2_{train,eval}.json`) · `splits/` · `camera/<clip_id>.mp4` (**4,719 files**, feature `camera_front_wide_120fov`, the member the pull tools range-read from the upstream zips; per-file sha256
`camera/camera_sha256.json`; SHIPPED since the PI directed it — the 2026-08-29 card said "not shipped" and was corrected 2026-09-09) ·
`tools/` (`pull_camera.py`, `pull_egomotion_range.py` — HTTP-range readers of the upstream 2.05 GB chunk zips, retained for provenance) ·
the Alpamayo2-Super records and a provider **egomotion store** · `MANIFEST.json`, `V71_MANIFEST.json`, `V72_MANIFEST.json`, `V8_MANIFEST.json`, `DATACARD.md`.
Label md5s (pin by md5, never by name — six copies of the v7 blob existed under three roots with differing md5s):

| release | file | n | md5 |
|---|---|---:|---|
| v7.1 | `s2_labels_v7.1.jsonl.gz` | 4,719 | `3bd19c5ab9aee5e166e0c0d37f2f7854` (V72MAN `based_on`) |
| **v7.2** train / eval | `s2_labels_v7.2_{train,eval}.jsonl.gz` | 4,572 / 147 | `0ff902130ce76886b8a925eceed9e3a5` / `aa12c948f062181c3297265b51526ec5` (**what refcv3, refcv4b, refcv5-v2 and PREREG6's refcv6 BASE read**) |
| **v8.1** train / eval (current) | `s2_labels_v8_{train,eval}.jsonl.gz` | 4,572 / 147 | `b45377a1f25263b5c0f3d318c126b1ac` / `eefc38d1453bd1c73802d44d45affced` (superseded v8.0: `bb54dfa0…` / `920a9fcb…` — nav_30s arc-length bug, RAW_T0_S offset) |

⛔ **No arm I could find has trained on v8.1.** `--max-speed-input` is *UNRUNNABLE on v7.2* (`speed_max_input` on 0/4,572 train, 0/147 eval; the trainer
refuses the flag) — it needs "the v8 label blob reaching the trainer's corpus, not a flag flip" (`PI_DECISION_QUEUE.md` item 10, Drive id
`1Johkx0PzK2uy_75zTvMKY3-eKQzFgYss`).

### 2.3 Frame geometry and the episode contract (the pixel side)

* **Camera: ONE camera only — `camera_front_wide_120fov`.** Upstream PhysicalAI-AV has **7** cameras (`pai_features.csv`: cross_left_120, cross_right_120,
  front_tele_30, **front_wide_120**, rear_left_70, rear_right_70, rear_tele_30); the v7 corpus, B1 epcache and EVAL6 cache carry **only the front-wide one**.
  Source mp4: 1080p, 30 fps (`pai_card.md:402`), decoded to **10 Hz**.
* **Projection: cylindrical `256×640` at `f_ref = 305.577` (f_eff 305.5775), column linear in azimuth; `az_max = (W/2)/f_ref` = true 120°**
  (`requested_hfov 120.0 / achieved 120.0 / observed_frac 1.0`). ⛔ The pinhole formula `2·atan((W/2)/f)` gives 92.6° and is wrong here (CARD81 §Projection).
  `w120` in a cache name = the 120° field of view, **not** a frame count (GLOSS).
* **Per-clip crop is MANDATORY:** `index/front_wide_cy.parquet` — two rigs, `cy ≈ 543` (rig A) and `cy ≈ 755` (rig B); **rig B is the majority, 2,723 of 4,719**; a
  centre crop is ~215 px wrong for B.
* **Model input:** 3 RGB frames at 100 ms spacing, channel-stacked → **9 channels**, 256 × 640 (HF card refc-v3 §1; WIRE §4 `channels == 3 × n_stack`, 9 ch → 3 frames → offset 2).
* **Episode file `<clip_id>.v2ep.pt`** (PNG-coded): `frames_u8`, `actions`, `poses`, `episode_id` (+ `maneuvers` in older builds). ~199 frames per episode @ 10 Hz ≈ 19.9 s.
  `poses[:, 3]` = speed (`v0 = pose_last[:, 3]`, `refc_v3_train.py:445`). ⛔ **`actions` stores `(κ, a)` — curvature FIRST**, the reverse of `RefAV1Config`'s `(a, κ)` (GLOSS).
  Older 256-px-square cache record (from the handoff-line doc `REFC_CORPUS_AND_LABELS.md`): `frames_u8 [199, 9, 256, 256] uint8`, `actions [199, 2]`, `poses [199, 4]`.
  ⚠️ The 9×256×640 shape is my inference from that + the geometry lines; not read from a `.v2ep.pt`.
* **Windows / stride:** the label record is **one per clip, anchored at `t0_s = 8.0`**; a window at `t_now` is described by the record iff
  `|t_now − t0| ≤ (tactical_hi − tactical_lo)/2` = **±2.0 s** (`window_in_band`, derived from the record's own bands) — so only windows near the anchor are
  token-supervised; other windows get `IGNORE_ID = −100` / weight 0 (`v7_labels.py`). REF-C v3's eval grid: `2s`, dt 0.5 s, 4 horizon steps, **n = 4,823 windows over 141 episodes**.
  Training windows per episode: UNVERIFIED (not stated in any doc I read).

### 2.4 The label record — `schema_version = "s2-geom-v7"` (`vocab = "v7"`), one JSON line per clip

Top-level keys on **4,719/4,719** (CONS §0, measured by the consumer at load): `_provenance, a_str, a_tac, alpamayo, bands, clip_id, cot_source, cot_tokens, g_str,
g_tac, horizon, manoeuvre_sequence, nav_command, scene, schema_version, semantics, t0_s, turn_suppression, vocab`. v8 adds `tac_SIT, nav_30s, lane_change_text,
speed_max_input` (+`v_max_bucket`). Nested keys read from `v7_labels.py` (Drive id `19nyI3tC9iIsGa3HIj-3JSttlqNOARu_t`):

| block | fields |
|---|---|
| `a_str`, `g_str` | `{token, args{within_m, by_time_s}, provenance, reason}` — args are a **named dict**, not a vector |
| `a_tac` | `{lat, lon, lat_args{within_m, lat_peak_m}, lon_args{v_target_ms, within_m}, truncated, serves_goals{lat_serves, lon_serves}}` (`lat_peak_m` = signed 20 s peak lateral displacement, dominated by road geometry — **not** the NUDGE decider) |
| `g_tac` | `{anchor{goal_x_m, goal_y_m, t_reach_s, band_s}, goals{<TOKEN>: {provenance, disputed, grounded/grounding, held, time_basis, t_nominal_s, state, corroboration, args…}}, violations}` |
| `nav_command` | `{token, args{distance_m, time_s}, provenance="ego-future", oracle (absent on 529), reason}` |
| `bands`, `t0_s`, `horizon` | `bands{operative_s [0,2], tactical_s [2,6], strategic_s [8,30], unassigned_manoeuvres[…]}`; `t0_s` = 8.0; `horizon{available_s, recording_span_s}` |
| `alpamayo` | the Alpamayo2-Super teacher block: `chain_of_causation` (CoC text), `meta_action`, `lateral{agree,…}`, `longitudinal{agree,…}` (agree = teacher vs measured; lateral 2575 T / 1841 F / 303 null, longitudinal 2995 T / 1718 F / 6 null); the augmentation also carries VQA, grounding (2D boxes; **no colour attribute**) and auto-labeling |
| `manoeuvre_sequence`, `scene`, `semantics`, `cot_tokens`, `cot_source`, `turn_suppression` | audit / VLM-derived; `turn_suppression` non-null on 68 records; **never training input** (CONS §6) |

Audit-only rule (`v7_labels.py:117`): `audit` (per-goal `disputed`, `alpamayo_agree`, `turn_suppression`, `g_tac_violations`, `a_tac_args`) is never a training input;
`_oracle` (nav_command, speed_max_input) is reachable only through `oracle_nav()` / `oracle_max_speed()` under `allow_oracle_nav=True`, which stamps the run manifest.

### 2.5 Clip counts per label layer (train 4,572 + eval 147 = 4,719 unless stated)

| layer | source | coverage | note |
|---|---|---|---|
| ego-derived tokens (`a_tac.lat/lon`, `g_tac` geometric goals, `g_str`, `a_str`, `nav_command`, `SPEED_BAND`) | ego poses / egomotion, hindsight geometry | 4,719 (100 %) | label-side privileged; strategic band [8,30] s extends beyond the ~20 s clip for many clips — `distance_end_m` null on 535 nav_30s entries |
| Alpamayo CoC-text goals (`YIELD`, `CORRIDOR_OFFSET`, traffic-light colours, `MERGE`, `GAP_TARGET`, `REACT_ON_ONCOMING`, `OVERTAKE_VEHICLE`, `TAKE_EXIT_*`, `LANE_CHANGE_*` goals, `EVADE_IN_CORRIDOR` [fused]) | `alpamayo.chain_of_causation` (teacher, unverified) | goal set 2–7 tokens/record, mean **2.751** (v7.2 train); 49.6 % of records carry ≥1 DISPUTED goal (2,343) | v7.2-train provenance census: `geometry` 9,106 annotations / 9 tokens; `vlm-cot` 3,472 annotations / 15 tokens (`v7_labels.py` docstring) |
| Alpamayo teacher text fields | Alpamayo2-Super (meta_action, CoT, VQA, grounding) | all 4,719 (the augmentation set is 4,729 clips × 5 tasks = 23,644 rows, `Sayood/tanitad-alpamayo2-augmentation`, MEASURED by direct read 2026-08-14 per the handover) | CoC carries a time expression on 6/4,729 = 0.13 % |
| 3D agent join (`obstacle.offline`) | PhysicalAI `obstacle.offline` cuboids, `--agent-join` | **4,427 / 4,572 train = 96.83 %** (`D-B1TRAIN-JOIN-1`); eval-side coverage not stated | only the REF-C `--agents head/oracle` levers read it |
| traffic-light grounding question asked | Alpamayo VQA | only **998 of 4,572** train clips; 692 clips carry a visible TL box and no TL token | absence of a TL token ≠ no light (`v7_labels.py`) |
| `tac_SIT` cited ≥1 term | CoC | 1,585 (33.6 %) | CITED/NOT_CITED |
| `lane_change_text` EXECUTED | CoC | 105 (2.23 %) | L 74 / R 31 |
| **VLM meta-action / SAM3 pixel classes** | the handoff-line PH0 pipeline (Qwen3.5-9B + SAM3) | **not part of the v7 corpus**: PI 2026-09-07 "no vlm, we stick to the alpamayo labels as teacher signals" (relayed; item 1 of the decision queue reads *RELAYED, PENDING*) | PH0/SAM3 labels exist only for 600 w120-val + 201 aug120 clips on `Sayood/tanitad-ph0-aug120` |

**Trajectory-level label.** The operative layer has **no tokens by design** — "supervised continuously by trajectory regression (`ade_dense_m` / `fde_last_m`, scored
at T1)" (V72MAN `band_coverage`). The per-clip future ego trajectory is the episode cache's `poses` (10 Hz); provider egomotion is ~111 Hz raw (`pai_features.csv` row 1:
`timestamp, qx…qw, x,y,z, vx,vy,vz, ax,ay,az, curvature`, ~2,224 rows/clip). REF-C v3 predicts the **whole 6 s trajectory in one pass** from a **128-anchor × 8-slot** vocabulary,
slots `(5, 10, 15, 20, 30, 40, 50, 60)` steps @ 10 Hz = 0.5…6.0 s, plus geometric goals `(x, y, heading, speed)` at 2/4/6 s (`goal_tau_steps (20, 40, 60)`); refcv6 BASE uses `--n-anchors 117`.
A record-level goal-point label also exists (`g_tac.anchor` = `goal_x_m, goal_y_m, t_reach_s, band_s`) — the "predicted geometric goal point" the PI's 2026-08-03 ruling admits.
The 6 s / `o5_k = 60` contract from HIERV §4b is thus already the REF-C trajectory horizon.

### 2.6 The other HF repos named in the docs (all INHERITED; none probed — HF not reachable from this session)

| repo | kind / gating | contents (per source) | source |
|---|---|---|---|
| `Sayood/tanitad-v7-training-corpus` | dataset, **private** | §2.2 | CARD81 |
| `Sayood/tanitad-alpamayo2-augmentation` | dataset | `records.parquet` **23,644 rows = 4,729 clips × 5 tasks** (`trajectory / meta_action / auto_labeling / vqa / grounding_via_vqa`; columns include `clip_id`, `task`, `raw_json`, `quantisation` [records the quantized-run arms]), `selection_manifest.json` (4,800), `vqa_bank_500.json` | handover 2026-08-15; `ph1_fuse.py` reads `clip_id/task/raw_json` |
| `Sayood/tanitad-physicalai-w120-256x640cyl` | dataset | the **parity-line** w120 corpus: train 2,403 files / 85 GB, val 603 / 21.2 GB, `epcache-256px` 3,053 files / 349.5 GB (square 256, the older cache) — not the v7 corpus | handover; `LOOP_STATE` 2026-08-03 |
| `Sayood/tanitad-ph0-aug120` | dataset, public + gated-manual | `batch_*` (201 Alpamayo clips VLM+SAM3-labelled at 120°), `fused_w120val/` (600 PH1-fused records), `bridged_w120train_2400/`, `epcache_oodval_290/`, `w120val_600/`, `g1_evidence/` | handover |
| `Sayood/tanitad-v6` | model, public + gated-manual | `v6F-SW-30k/` step-6250 checkpoint + config + `pbattery_*` | handover |
| `Sayood/tanitad-refc-v3` (**PUBLIC since 2026-09-03**), `Sayood/tanitad-refc-v4b` (private), `Sayood/tanitad-refc-base`, `-xl` (gated manual) | models | REF-C v3 ckpt `ckpt_40284_FINAL.pt` md5 `fc304b62686ddb9e685d14bdab482404`; card `HF_CARD_tanitad-refc-v3.md` | HF card; `PI_DECISION_QUEUE.md` item 4 (account storage 991.415 GiB across 46 repos, no numeric ceiling exposed) |
| `facebook/sam3`, `nvidia/PhysicalAI-Autonomous-Vehicles` | upstream, gated | — | handover; CARD81 |

⚠️ Correction to the brief: the four repos listed in the assignment are the **handoff-branch** map. The **newest training data** is the private v7 corpus above, which is *not* in that list.

### 2.7 The handoff-branch (older, v6-line) label and record schemas — for contrast

* **Vocabulary in code on `origin/handoff` is v6.1**, in `stack/tanitad/models/v6.py:132-160` (MEASURED — read from the branch): `STRATEGIC_GOAL_TOKENS` (11): `KEEP_CORRIDOR LANE_TARGET EXIT_RIGHT EXIT_LEFT TURN_LEFT TURN_RIGHT STRAIGHT_THROUGH ROUTE_TO STOP_AT FOLLOW_MAIN_ROAD NONE_ABSTAIN`;
  `STRATEGIC_ACTION_TOKENS` (6): `PREPARE_LANE_CHANGE HOLD_CORRIDOR REDUCE_TO PREPARE_EXIT PREPARE_STOP RESUME_CRUISE`; `TACTICAL_GOAL_TOKENS` (9): `ANCHOR_GOAL CORRIDOR_OFFSET GAP_TARGET SPEED_BAND YIELD_AT STOP_POINT WAIT_FOR_ONCOMING EVADE_IN_CORRIDOR TRAFFIC_LIGHT_REACT`;
  `TACTICAL_LAT_ACTIONS` (6): `LANE_KEEP LANE_CHANGE_L LANE_CHANGE_R ABORT_LC NUDGE_L NUDGE_R`; `TACTICAL_LON_ACTIONS` (6): `FOLLOW CRUISE YIELD_MERGE BRAKE_TO CREEP HOLD`. **`ph1_fuse.py` imports exactly these** (`from tanitad.models.v6 import STRATEGIC_GOAL_TOKENS, TACTICAL_LAT_ACTIONS, TACTICAL_LON_ACTIONS`; `SCHEMA = "ph1-fused-v1"`).
  ⇒ any design built on `origin/handoff`'s vocabulary is built on the **superseded** tuples: `v7` renamed/re-cut (`_FOLLOW_ROUTE` suffix, `HOLD_MAIN_ROAD`, `ANCHOR_GOAL` demoted to an argument frame, `YIELD_AT→YIELD`, `WAIT_FOR_ONCOMING→REACT_ON_ONCOMING`, `+ACCELERATE`, `+ADAPT_SPEED_FOR_CURVE`, tactical goals 9→22 incl. traffic-light colours, lat 6→8 with `TURN_L/R`, lon 6→8, `NAV_*` inputs). `vocab_v7.py`'s "TENSOR CONTRACT: every tuple is a NEW name; no v6 tuple is edited".
* **PH1 fused record (`ph1-fused-v1`, one JSON per clip):** `schema_version, clip_id, geometry{frame_wh, note}, ego{ego_state, route, speed_profile, speed_events, lane_change_events, situations}, perception{tracks[], per_concept_hits, src="sam3"}, semantics{scene, signs, symbols, src="vlm", sign_text_status="pending_g1_gate"}, alpamayo{<task>: raw_json}, corroboration{}, vocab{g_str{token,src,corroborated_by_route}, g_tac_lat{token,voters,votes}, g_tac_lon{…}}, scenario_description, _conflicts, inference_admissible=["perception","semantics"], _provenance{ego:"privileged-labels-only", sam3:"vision", vlm:"vision", alpamayo:"external-labels-only"}`. First run over 600 w120-val clips: 175 corroborations, 41 conflicts, 56 with the Alpamayo layer (handover, MEASURED at the time).
* **PH0 `ph0-v2.2` schema** (PH0_PIPELINE_VALIDATION §2): `scene{illumination, weather, road_type, domain, lanes_visible, lane_ego, conf}`, `signs{n_signs, signs[{kind,text,state,applies_to_ego}]}`, `grounding` (⛔ diagnostic-only, 2/49 agree with SAM3), `symbols{goal_kind, goal_evidence_sign, actions[{verb,direction}], conf}`; SAM3 detections 1,383 / 50 pilot clips (car 703, sign 292, light 167, pedestrian 130, truck 57, bus 22, cyclist 12).

---

## 3. "refcv7" — what it is, which vocabulary release refcv6 consumes, and what is quotable about token prediction

### 3.1 refcv7: NO document, file, table row or log line names it (one cheap probe set, per the orchestrator's re-scope)

| probe | result |
|---|---|
| GitHub: `git ls-remote` all 47 heads, open PRs #2-#6, `search_code "refcv7" repo:sayoood/TanitAD`, `git ls-tree` of `origin/handoff` + 3 fetched branches, `grep` of the working tree | 0 hits. `origin/handoff` file names contain `v7` only in `adv7_matched_denominator_and_precision.py`. |
| Drive `fullText contains 'refcv7'` (all mime types) | 1 hit, false: `v7_vocab_reach_census.py` (matches on the separate tokens "refcv5 … v7", literal string `refcv7` does not occur in it — I read the whole file). |
| Drive titles `refcv7 / REFCV7 / refc_v7 / REFC_V7` | only `…refcv3-training-readiness_raw_preflight_PRE-FIX_v7vocab_default_txt.ajson`, `test_refc_v3_nav_from_v7.py(c)` — both "refcv3" + "v7". |
| Drive `LAB_BACKLOG.md` (106 KB, 2026-09-15), read in full and regex-searched `refcv7|refc.?v7|refc-v7` | 0 hits. |
| Drive wiki `log.md` (2026-09-29) | an Obsidian LLM-wiki maintenance log; unrelated to the programme. |

**Most plausible referents (HYPOTHESIS — evidence is the naming, not a statement by the PI):**
1. **"REF-C trained on the v7 corpus / v7 vocabulary / v7.2 labels"** — every REF-C arm since refcv3 (`refc_v3_train.py --v7-labels s2_labels_v7.2_train.jsonl.gz --nav-from-v7`) is exactly that, and the corpus itself is named `tanitad-v7-training-corpus`. Under this reading "refcv7" = the refcv3 → refcv5-v2 → refcv6 line and the **newest arm that has a result is refcv5-v2** (40,284 steps; see P2), the newest *designed* arm is **refcv6** (staged 2026-09-10, never launched).
2. "the REF-C after refcv6" — no design exists for it.
3. Nothing named v7 is a REF-C model: the other v7 things are the **v7f world-model line** (`PREREG_V7F.md`, Drive id `1viBvy7SFWldGThyBpz9dPBIjAGQ7XSto`), the **v7-tiny 29-min rig**, `PREREG_V7_SEED_POS.md`, and `V7_RECIPE_AND_SCALEUP.md` (the v7 WM recipe: S-W, `--o5-form l1 --w-o5 1.0 --w-o6 0.1 --o5-k 8 …`). Note `HIERV`'s v6 config E is superseded as "the newest model" only in the sense of the WM line; REF-C is a separate supervised anchor arm.

The PI has since said to treat **refcv6 as the nearest** (orchestrator, latest). P2 covers refcv6's architecture. Facts relevant to P1:

### 3.2 What refcv6 (PREREG6, 2026-09-10) consumes

* **Labels: v7.2** — BASE argv (65 tokens, verified `MATCH` vs refcv5-v2's banked `config.json['argv']`): `--v7-labels …/s2_labels_v7.2_train.jsonl.gz --eval-labels …/s2_labels_v7.2_eval.jsonl.gz --nav-from-v7 --goal-str --tac-goal-tok-head` + `--agent-join <join>` (B1 TRAIN agent join, 4,427/4,572 = 96.83 %). **NOT v8.1.** Image geometry `--image-hw 256 640`, `--n-anchors 117`, `--anchor-v0-conditioned`, `--sampler ddim`, batch 20, lr 1e-4, warmup 2000, `--steps 40,284` (full) or 12,000 (cut).
* **Which tokens it can learn:** `nav_command` as an (oracle) INPUT; factored `a_tac.lat`/`a_tac.lon` (`tac_lat_ce`, `tac_lon_ce`); the 22-token `g_tac` set only via `--tac-goal-tok-head` **AND** `--w-tac-goal > 0` (arm C; the weight is a PI decision, `arms.py` raises `PIDecisionRequired` rather than invent one). **`--goal-str` is NOT the 8-token `g_str`**: it is a 3-unit *geometric* bearing/distance head supervised from LAN (`str_ce_refc: "refc has no strategic TOKEN head"`, CENSUS). The 15-token strategic vocabulary (`refc_strategic.py`, a *different* tuple from v7's 8) "stays out" (`PI_DECISION_QUEUE.md` item 3: P4 support bar failed — 6 of 15 tokens populated, 11.43 % of the horizon supervisable).
* **Reach census (CENSUS `SURFACES` table, refc line):** loss surfaces `tac_lat_ce` (v7 `HEADS["tac_lat"]`), `tac_lon_ce`, `tac_goal_bce`; input surface `nav_input`; audit surface `goal_audit`. The **v6-line trainer** (`train_v6_staged.py`) has strategic CEs (`str_goal_ce_v6`, `str_action_ce_v6`) gated on `w_s2_goal` (default 0.0 ⇒ `p.grad is None`) and `tac_action_ce_v6` = "landed tensors, CE is a pre-registered follow-up". ⇒ **the strategic TOKEN heads (g_str 8 / a_str 7) are trained by no REF-C argv and by no default v6 argv.**
* **Trainer conditions that travel with the labels:** `allow_oracle_nav=True` stamp; per-token mask `effective_mask()` (vocab ∪ empirically-absent); `class_weights` computed from the loaded split; `goal_pos_weight(cap=50)`; `negatives="measured"` (negatives supervised only for tokens the blob emits from geometry; CoT-only tokens get ignore-weight unless entailed-false by `TACTICAL_GOAL_EXCLUSIVE`).

### 3.3 Every quotable number on how well tactical/strategic tokens are predicted (with source; all INHERITED from Drive text unless stated)

| quantity | value | source |
|---|---|---|
| refcv3 (step 40,284) tactical **lateral decision**, trajectory-derived 3-class (`factor_from_kinematics`, NOT v7 tokens) | accuracy 0.9540 [0.9396, 0.9664], κ 0.8113 [0.7541, 0.8578], n = 4,823 windows / 141 eps; recalls lane_keep 0.9715, turn_left 0.8048, turn_right 0.8636 | `HF_CARD_tanitad-refc-v3.md` §4.2 (cites `eval/refcv3-40284-openloop.json`) |
| refcv3 **longitudinal decision** | accuracy 0.7477 [0.7170, 0.7790], κ 0.3078 [0.2484, 0.3659]; recalls brake_stop 0.3835, steady 0.8700, accelerate 0.3439 (misses 62 % of braking, 66 % of accelerations) | same |
| refcv3 tactical goal-setting | goal-point error 0.9288 m [0.8611, 0.9947], bearing MAE 1.5851°, range ratio 1.0087 | same |
| refcv3 STRATEGIC family | **UNAVAILABLE, n = 0**; its 3-class route head probe: accuracy 0.7667 [0.7097, 0.8224] vs majority 0.6742, κ 0.4604, n 3,622 / 128 eps — reads identically under true/shuffled/zero nav (nav-insensitive) | same (`_strategic_source: "refcv3 route head (sidecar)"`) |
| refcv5-v2 (40,284 steps) route accuracy | 0.7708 [0.7146, 0.8254], κ 0.4614, n 3,622, chance 0.3333, from a 771-parameter head; tactical lateral κ 0.8193 vs refcv4b 0.7374; lost speed MAE (0.2919 vs 0.2540) and along-track (0.2655 vs 0.2348); `os − ha0_ext` = +0.0205 [+0.0043, +0.0390] ⇒ FAIL | PREREG6 fact 10 — **INHERITED from `REFCV6_ARCHITECTURE_REVIEW.md`, not re-verified against raw JSON; PREREG6 itself says to re-read the raw JSON before scoring** |
| the 22-token tactical goal head in refcv5-v2 | **`tac_goal_tok_head` 11,286 params, `grad_abs_sum` exactly 0 for all 40,284 steps** (2/2 grads `None`) ⇒ **no result may be attributed to the 22-token vocabulary** (`PREREG_REFCV5_V2_LANDING.ERRATUM-1.md`). `tac_goal_head` (factored) had `grad_abs_sum` 14.368 — trained. | PI_DECISION_QUEUE item 10 (`raw/gradreach_live.json`); PREREG6 fact 11 |
| accuracy / κ / AUC of the **v7 8-way `a_tac`, 22-way `g_tac`, 8-way `g_str`, 7-way `a_str`** heads against v7 labels | **NOT FOUND in any document I read** — no measured per-token result exists (UNVERIFIED-negative; the census measures *label* supply and *loss reach*, not prediction quality) | — |

⛔ Every tactical/strategic number above is a **T0/open-loop fidelity** number on an oracle-nav arm (`os`); `os_navzero` (nav withheld) is the deployment-honest row (refcv3: ADE 0.4659 vs `os` 0.4419 vs the hold-action control `ha` 0.2996 — the trivial control won).

---

## 4. Integration facts and risks for a NEW arm on the newest state

**Where the label/vocabulary code lives.** `stack/tanitad/models/vocab_v7.py` (35 KB), `stack/tanitad/data/v7_labels.py` (47 KB, "the ONE label consumer", Drive id `19nyI3tC9iIsGa3HIj-3JSttlqNOARu_t`), `stack/tanitad/models/nav_conditioning.py`, `stack/tanitad/refs/{refc_v3.py, tac_goal_head.py, refc_wp_index.py, cold_start.py}`, `stack/scripts/{refc_v3_train.py, s2_geom_emit_v7.py, v7_vocab_reach_census.py}` — all on the PI's branch `agent/arch-inf-20260803` (staged, **not pushed**: the remote head `0fcb1c1` predates them; WIRE §5.1 "the artifact is not repo-resident"). **None of it is on `main`, on `origin/handoff` or on this working branch.** The label blobs live only on the private HF repo.

| # | risk / difference | consequence for REF-F (`REFF_CONTRASTIVE_SELECTOR_PLAN.md` §7, written against `main` 467ce8a) |
|---|---|---|
| 1 | **Vocabulary**: main/handoff = v3-goal (114 tokens) / v6.1 tuples; newest = v7 (56 slots / **52 unique** strings — MEASURED by me from `vocab_v7.py`: 8+7+22+8+8+3 = 56; `TURN_L/R`, `LANE_CHANGE_L/R` occur in two fields) | REF-F's maneuver labels ("from the factorised vocabulary") must be the v7 `HEADS["tac_lat"]`/`HEADS["tac_lon"]` names (8 + 8), not `refc_tactical`'s 3 + 3 nor v6's 6 + 6 |
| 2 | **Model I/O**: plan §7.1 assumes `RefCModel.forward(frames[B,8,9,256,256], nav_cmd, v0, …)` with horizons 5/10/15/20 (2 s) and 256 px square. Newest REF-C: `forward(frames, nav_cmd=None, v0=None, steps=0, lan=None, nav_known=None)` (`refc_v3.py:480`), **3 frames → 9 ch, 256×640 cylindrical**, horizons `(5,10,15,20,30,40,50,60)` = 0.5–6 s, 128 anchors × 8 slots (`(128, 8, 2)` buffer; refcv6: 117), goal taus 2/4/6 s | candidate fans must be 8-slot / 6 s, not 4-slot / 2 s; window contract is 3 frames not 8; parity `assert_parity_corpus(require=True)` **must be dropped** — the v7 corpus is NON-parity by design (refcv3 `config.json`: `parity false`) |
| 3 | **Corpus**: main = parity 2,376 eps square-256; newest = B1 4,713 eps (4,572 train + 141 eval) + EVAL6 (6) cylindrical 256×640; eval = 147 = union of two cache roots; per-clip `cy` crop mandatory | REF-F's `reff_embed_cache.py` must read the `.v2ep.pt` w120 caches and both eval roots; cache identity = md5 of label blob + `_geometry.json` sha |
| 4 | **Inputs**: `nav_command` is `ego-future` on 4,719/4,719 (oracle; PI ruling 2026-09-04 makes the nav-fed `os` arm the deployment-relevant one with a "noiseless, perfectly timed" nuance); `nav_30s` and `speed_max_input` are stamped ORACLE; `allow_oracle_nav=True` is binary — record `oracle_nav_payload` | a contrastive selector conditioned on nav/speed inherits the oracle stamp; the plan's "no `actions` tensor reaches the model" allowlist must also allowlist-audit `nav_*`/`speed_max_input` |
| 5 | **Actions layout**: `actions` stores `(κ, a)` curvature first (GLOSS) | plan §7.1 says REF-F never reads `actions` (good); anything that does must not assume `(a, κ)` |
| 6 | **Eval harness**: newest = `taniteval/tools/openloop_suite.py` over `refcv3_arm.py` dumps, arms `os / os_navzero / os_navshuf / ha / ha0 / oracle_sel`, grid `2s` dt 0.5 s, 4 horizon steps, **n = 4,823 windows / 141 episodes**, `t1_eval.py` for T1, criteria registry 2.5.0, `verdict_refcv6.py` (P1–P3, L1–L4, S1, G1, T1 clauses, `MISSING_DATA` ≠ `FAIL` ≠ `SUCCESS`) | plan §7.1's `refc_eval.collect` contract (881 windows / 40 episodes) is the **old** harness; a REF-F arm must emit `os`/`ha`/`ha0` rows on the 4,823-window grid and be checked by a `verdict_*` file that refuses `overlapping_holdout_se` and non-T1 stamps |
| 7 | **Hold-action baselines dominate**: `ha` 0.2996 beats refcv3 `os` 0.4419 (paired +0.1423 [+0.1187, +0.1658]); `ha0_ext` is the floor refcv6 must beat separated by ≥ 0.10 relative | REF-F's selector bar = beat `ha0_ext` AND `ha`, replicate-floor ×3 (PREREG6 §5B) — not "beat REF-C's own top-1" alone |
| 8 | **Labels vs the deployable arm**: PI rule "labels may use ego; inference is vision-only" + 2026-09-04 nav ruling + v8 `speed_max_input` are three separate exceptions at different stamp levels | state per input at inference what it is computed from (goal/situation disjointness); `tac_SIT`, `disputed`, `alpamayo.*` are audit-only and must never be REF-F inputs |
| 9 | **Dead/unsupplied classes**: `LANE_CHANGE_L/R`, `ABORT_LC` (lat actions) and `YIELD_MERGE` (lon) have n = 0; 4 `g_str` and 2 `a_str` tokens n = 0; 8 goal tokens n < 30 | any candidate-set or PMI-style label built from these will contain empty classes; mask per head with `effective_mask()`; report per-token n |
| 10 | **Speed/step mismatch**: `step_s` in trainer logs is accumulated ÷ `--log-every`; refcv6 rate NOT MEASURED (refcv5-v2's A40 receipt 4.0 s/step; Thor 4.21 s/step) | do not import a wall-clock estimate |
| 11 | **GitHub vs Drive**: the `handoff` branch's `ckpt_compat`, `v6_probe_trunk`, `ph1_fuse.py` belong to the **v6 WM line** and its v6.1 vocabulary; the REF-C line's code is elsewhere (Drive/PI branch) | reuse of `ph1_fuse` votes needs a re-import from `vocab_v7`; its `GOAL_TO_GSTR` map is for the old strategic set |

---

## 5. AUDIO — does ANY corpus the programme uses carry audio?

**ABSENT** in every corpus the programme trains or evaluates on; **UNVERIFIED** only for the audio *stream inside the mp4 containers* (never probed).

| corpus | verdict | evidence (file:line) |
|---|---|---|
| **PhysicalAI-Autonomous-Vehicles** (source of the v7 corpus, the parity corpus, the w120 caches) | **ABSENT.** Exactly **36** features = 7 camera + 6 calibration + 3 label + 1 lidar + 19 radar; **no audio / microphone row**. `grep -ci audio` over `pai_features.csv`, `pai_card.md`, `PHYSICALAI_FEATURE_PROBE.md`, `pai_tree_l1/l2/l3_sample.json`, `pai_label_schemas.json`, `pai_metadata_summary.json`, `pai_sizes_and_revs.json`, `pai_repo_info.json`, `our_corpus_profile.json` = **0 in each** (MEASURED by me 2026-09-29). | `TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-26-physicalai-feature-probe/PHYSICALAI_FEATURE_PROBE.md:20` ("exactly 36 rows. 7 camera + 6 calibration + 3 label + 1 lidar + 19 radar"), `:40-116` (the 36-feature table), `pai_features.csv` (36 rows, listed); dataset card `pai_card.md:365` — *"We store the data separately for each sensor (camera, LiDAR and radar). Besides these sensors we also provide ego motion"*; `pai_card.md:402` — camera mp4s are 1080p at 30 fps. The card lists no audio. |
| the **v7 training corpus** / B1 / EVAL6 epcaches | **ABSENT** as a label or feature: CARD81 lists camera mp4 + egomotion + Alpamayo text + labels only. Whether the shipped `camera/<clip_id>.mp4` (the pull tools fetch the upstream `{clip_id}.camera_front_wide_120fov.mp4` member; whether it is a byte-copy is itself UNVERIFIED) carries an audio track: **UNVERIFIED — no doc shows it `ffprobe`d**; the epcache decode is video-only. | CARD81 §Camera |
| **Alpamayo2-Super augmentation** | **ABSENT** — text (meta_action, CoT, VQA, grounding, auto-labeling) only | CARD81 |
| comma2k19, nuScenes-mini, Cosmos-Drive-Dreams, l2d, argoverse2 | not audio-carrying as used; no doc says otherwise (UNVERIFIED for their raw releases) | `DataEng/DATA_STRATEGY.md` corpus table |
| YouTube POV pilot (`harvest.py`) | the harvest downloads **≤ 480 p, no audio** (explicit) — some source videos have audio (e.g. "Binaural Audio" titles in `db_retry_evidence*.json`) but it is dropped | `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-24-youtube-idm-pilot/harvest.py:6` |
| emergency-vehicle scenarios | **siren audio explicitly out of scope, "Phase 0 visual-only proxy"**: `limits: {"audio": "out-of-scope-phase0", "cues": "visual-only"}`; test asserts it | `TanitAD Research Hub/Opponent Analyzer/Implementation/incoming/2026-08-07-emergency-scene-scenario/emergency_scene.py:55, :208-209`; `…/tests/test_emergency_scene.py:91`; `TanitAD Research Hub/Data Engineering/Research/2026-07-09-physicalai-r1-selection-and-worldmodel-scenarios-license.md:94` ("visual light-pattern proxy, audio out of scope P0") |

Consequence: a hierarchical selector cannot use an audio cue (siren, horn, tyre noise) on this data; emergency-vehicle behaviour is reachable only through vision — and `emergency_vehicle` has "insufficient support" even in the CoC `tac_SIT` field.

---

## 6. Implications for a contrastive hierarchical selector (≤ 12 bullets; facts only, no design)

1. **A frozen, versioned, tokenised hierarchy already exists as labels**: 8 `g_str` + 7 `a_str` (strategic, band [8, 30] s), 22 multi-label `g_tac` + 8 `a_tac.lat` × 8 `a_tac.lon` (tactical, band [2, 6] s), 3 `NAV_*` inputs; `assert_frozen()` blocks extension — a selector's token side must import `vocab_v7`, never re-derive.
2. **Ready-made positive/negative structure for a contrastive objective**: `TACTICAL_GOAL_EXCLUSIVE` (22 forbidden pairs) gives **entailed negatives** for free (`entailed_false`), and `GOAL_ADMISSIBLE_LAT/LON` gives a goal→action compatibility map (`action_serves_goals`) — exactly a "which candidate serves which goal" relation. Negatives outside these are only sound for the 9 geometry-emitted tokens (`negatives="measured"`).
3. **Candidates are a 128- (v3) / 117- (v6) anchor × 8-slot × 6 s fan** with a per-clip goal point `g_tac.anchor(goal_x_m, goal_y_m, t_reach_s)` and 2/4/6 s geometric goals; an "oracle-in-fan" gap and `oracle_sel` (ADE 0.3668 vs `os` 0.4419 on refcv3) already price selection headroom (0.0751 m).
4. **Supply is thin exactly where a hierarchy needs it**: strategic populated 4/8 goals, 5/7 actions; `LANE_CHANGE`, `ABORT_LC`, `YIELD_MERGE`, `EXIT_*` empty; 8 tactical goals n < 30 (a contrastive loss on those has < 30 positives in 4,719 clips).
5. **Strategic labels are ego-future geometry (oracle-like), not vision-derivable ground truth**; `nav_command`/`nav_30s`/`speed_max_input` are `ego-future`-stamped — usable as labels and as stamped inputs, ⛔ never as an unstamped selector input.
6. **Perception-style tactical tokens (traffic-light colours, `GAP_TARGET`, `REACT_ON_ONCOMING`, …) are Alpamayo teacher text, 49.6 % of records carry a DISPUTED goal, traffic-light colour is unverified** ⇒ any selector head on them needs the mandatory ego-only control.
7. **The labels are one record per clip anchored at t0 = 8 s**; window supervision is confined to ±2 s of the anchor (`window_in_band`) — a selector trained on all windows gets IGNORE on most of them.
8. **The operative layer has no tokens by design** (continuous trajectory regression); the 2–6 s tactical band and [8,30] s strategic band are per-record — the selector's "levels" map onto `bands`, with (6, 8) s owned by nothing.
9. **No corpus carries audio, a map, lane graph, or traffic-light channel**; only the front-wide 120° camera exists in the v7 corpus (1 of 7 upstream cameras); 3D agent cuboids join 96.83 % of train clips.
10. **The bar is a hold-action control, not the incumbent**: `ha` (0.2996) beats refcv3's own `os` (0.4419); refcv6's pre-registered SUCCESS needs separated wins over `ha0_ext` and `ha` with margin ≥ 0.10 and effect ≥ 3× the replicate floor, plus non-regression on lateral and strategic families.
11. **Nothing about token-level prediction quality has been measured** (§3.3 last row): the 22-token head never received gradient in refcv5-v2, `g_str`/`a_str` token heads are trained by no REF-C argv — a selector cannot lean on a measured tactical/strategic token predictor; it would be the first.
12. **There is no refcv7**; the reference arm is refcv6 (staged, never run) on v7.2 labels; v8.1 fields (`tac_SIT`, `nav_30s`, `lane_change_text`, `speed_max_input`) exist in the corpus but no trainer has consumed them.

---

## 7. Deliverable manifest

| artifact | where it lives | single-copy? |
|---|---|---|
| this report | `repo:TanitAD Research Hub/Architecture & Inference/Research/2026-09-29-hicap/P1_state_vocab_dataset.md` (staged) | in-repo, staged only (not committed) |
| raw material extracted from `origin/handoff` (registry, handoff docs, `ph1_fuse.py`, `models/v6.py`, `vocab.py`, …) | scratchpad `/tmp/claude-0/-home-user-TanitAD/47a1cfa2-478d-5914-8f27-fb9c6388aa94/scratchpad/h/` | **scratchpad only — ephemeral, reproducible via `git show origin/handoff:"<path>"`** |
| Drive text read (not copied into the repo beyond quotes) | Drive ids in §0 table | Drive is the primary copy |
| fetched blobless remote refs (`origin/fix-b1-traj-head`, `origin/validate/axis6-precheck`, `origin/claude/google-drive-domains-5ml2j9`) | local `.git` remote-tracking refs only | local refs, no working-tree change |
| **NOT delivered** (could not, and why) | the v7 label blobs, `MANIFEST.json` (its five `trainer_conditions_mandatory` sentences), `V71_MANIFEST.json`, `s2_geom_emit_v7.py`, `refc_v3.py`, `PREREG_V7F.md`, `REFCV6_ARCHITECTURE_REVIEW.md` (P2's), any HF repo listing | Drive `.json`/blobs not opened as text; HF not reachable — see UNVERIFIED tags |
