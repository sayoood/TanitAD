# PH0 extraction coverage — what we actually extract vs what v6 needs

**PI question, 2026-08-12:** *"Did you include all the information we want to extract like scenario etc"*

**Answer: NO. Strategic is complete; tactical is largely empty; and the SCENARIO CLASS the
question names is genuinely absent.** This is the gap table, derived by walking
`HIERARCHY_VOCABULARY.md` §3–§4 and `V6_TRAINING_MEASURES.md` §1–§5 against the shipped
schemas in `stack/scripts/ph0_v2.py` (`S_B1`–`S_B4`) and `ph0_pilot.py`
(`engine_a_summary`). Evidence class: **MEASURED** (read from source, file:line below), not
inherited from a summary.

---

## 1. What IS extracted today (`ph0-v2.2`)

| engine | call | fields |
|---|---|---|
| B (VLM) | **B1 SCENE** | `illumination` · `weather` · `road_type` · `domain` · `lanes_visible` · `lane_ego` · `conf` |
| B | **B2 SIGNS** | `n_signs` + ≤6 × {`kind` ∈ light/speed/nav/stop/yield/other · `state` ∈ red/amber/green/none · `text` (verbatim OCR) · `applies_to_ego`} |
| B | **B3 GROUNDING** | per sign: `visible` · `frame_idx` · `bbox` (Qwen-native normalised 0–1000 → px) |
| B | **B4 SYMBOLS** | `goal_kind` (11) · `goal_evidence_sign` · ≤3 × {`verb` (6) · `direction`} · `conf` |
| A (geometry) | hindsight | `route.{token,token_valid,dist_m,arc_m,maneuver_dyaw_rad,graded_route}` · `lane_change_events[]` · `speed_events[]` · `speed_profile.{v_t0,v_min_future,v_max_future,net_dv,stops}` · `peak_kappa_per_m` |
| A (v2.2 NEW) | pre-decision ego | `v_now/mean/min/max` · `accel_1s/3s` · `yaw_rate` · `net_dyaw` · `dist_travelled` · `motion` · `turning` |
| C (SAM3) | pixels | box-prompted masklets + `frac_mask_in_box` / `frac_box_covered`; **text-prompted concept detection added 2026-08-12** |

---

## 2. STRATEGIC layer — ✅ COMPLETE

`GOAL_KINDS` and `ACTION_VERBS` in `ph0_v2.py:34-38` are a **term-for-term match** to
`HIERARCHY_VOCABULARY.md` §3's `g_str` and `a_str` tables, including `FOLLOW_MAIN_ROAD` as
the no-route default and `NONE_ABSTAIN` kept for genuinely ambiguous geometry. Args
(`target_arc_m`, `distance_m`, `within_m`, …) are supplied by Engine A by design — the
organising principle is *the VLM chooses SYMBOLS, the algorithm supplies NUMBERS*.

**Nothing to add here.** The one measured weakness is behavioural, not structural:
`follow_main_road` was selected 5/8 on the smoke, i.e. the head defaults rather than
discriminates. That is a PH1-sample-size question, not a missing field.

---

## 3. TACTICAL layer — ⛔ THE BIG HOLE

`HIERARCHY_VOCABULARY.md` §4 defines nine `g_tac` tokens. **PH0 emits none of them.** B4
emits only strategic goals; there is no tactical call at all.

| `g_tac` token | args needed | extractable today? |
|---|---|---|
| `ANCHOR_GOAL` | anchor_id, t_reach_s | ⚠️ derivable from Engine A's polyline — **not assembled** |
| `CORRIDOR_OFFSET` | lat_offset_m, arc_m | ⚠️ derivable from Engine A — **not assembled** |
| `SPEED_BAND` | v_lo, v_hi | ⚠️ B2 gives the speed-sign `text`, Engine A gives corridor speed stats — **never combined** |
| `GAP_TARGET` | **agent_slot_id**, time_gap_s | ⛔ **NO** — no agents extracted |
| `YIELD_AT` | position_arc_m, **gap_slot** | ⛔ **NO** — B2 has a `yield` sign kind but no position, no gap |
| `STOP_POINT` | position_arc_m, **reason** ∈ {sign,light,queue,hazard} | ⛔ **NO** — reason not extracted |
| `WAIT_FOR_ONCOMING` | narrow_arc_m, **oncoming_slot** | ⛔ **NO** — nothing at all |
| `EVADE_IN_CORRIDOR` | lat_offset_m, **obstacle_slot**, past_arc_m | ⛔ **NO** — nothing at all |
| `TRAFFIC_LIGHT_REACT` | **light_slot_id**, state, **stopline_arc_m** | 🟡 PARTIAL — B2 gives `state`; no slot id, no stopline distance |

`a_tac` LAT (`LANE_KEEP`/`LANE_CHANGE_L/R`/`NUDGE_L/R`) maps onto Engine A's
`lane_change_events`; **`ABORT_LC` has no detector.** `a_tac` LON `FOLLOW(time_gap_s)`
needs a lead agent — and **LF0 RC1 (the zero-parameter geometric lead read) was REFUTED**,
so distance-keeping still has no instrument.

### ⭐ The single root cause: **PH0 extracts NO AGENTS.**
Five of the nine tactical goals need an agent/obstacle slot. This is also why the
binding four-family rule's **LONGITUDINAL** family (headway / time-gap / TTC to the lead
agent) cannot be computed — 88.7 % of the oracle gap is longitudinal.

**Two independent supplies exist and neither is wired:**
1. **SAM3 text-prompted detection** — added to `ph0_sam3.py` 2026-08-12 (`--mode text`,
   `AGENT_CONCEPTS`), gives per-frame masklets propagated into TRACKS. Running now.
2. **`obstacle.offline` in PhysicalAI-AV** — 3D agent cuboids on **97.44 %** of the corpus,
   **87,481 cuboids over 10 dynamic classes**. ⛔ Our ingest reads **4 of 36 features**
   (`physicalai_r0.py`; `physicalai.py:153-154` adds intrinsics/extrinsics). *This is
   already-paid-for ground truth we are not reading* — the cheapest fix in the table.

---

## 4. SCENARIO / SITUATION — ⛔ ABSENT, and this is what the PI asked about

The programme's **scenario classes are `lane_change` / `intersection` / `roundabout`**,
defined once in `stack/tanitad/data/situations.py` (thresholds FROZEN by the
2026-07-26 pre-registration; roundabout deliberately unpowered at n=26).

**PH0 does not emit a situation class.** B1's `road_type` / `domain` are *static scene
descriptors* ("urban", "junction"), not the *dynamic situation* the classifier defines. A
clip driving straight through a junction and a clip turning left at one get the same B1
`domain=intersection` and are different situations.

Two facts that constrain how this gap is closed — both binding:

- **Labels may use ego; inference is vision-only** (PI 2026-08-03). Situation labels are a
  pure deterministic function of the ego pose track (`emit_situation_labels.py:54-62`), so
  the *label* side is already solved algorithmically — Engine A can emit it for free. The
  missing piece is the **vision-only read**, which is what a VLM call would add.
- **The goal input must NOT carry the situation classifier's output** (PI 2026-08-03).
  ⇒ If a situation call is added it must be **information-disjoint from B4**: B4 must not
  receive it, or goal-vs-situation attribution dies exactly as it did in the `--v2`
  conflation and the C6 confound.

---

## 5. Other gaps worth naming rather than discovering later

- **No distance/range to anything.** B3 gives a 2D box; there is no arc position for a sign,
  a stopline or a light. `camera_intrinsics` + `sensor_extrinsics` *are* ingested since
  D-016 R1, so a ground-plane range estimate is available — unbuilt.
- **No lane/corridor geometry, and there never will be from this corpus.** Settled at five
  probes: PhysicalAI-AV has **no map, lane graph, junction annotation, roundabout label,
  traffic-light feature or route/goal signal**; the card says verbatim *"we do not include
  open maps data"*. `lanes_visible` / `lane_ego` are the VLM's *visual* count and carry no
  ground truth to score against. Strategic topology must come from AlpaSim or elsewhere.
- **`alpamayo_rows = 0` on every clip** — engine D contributed nothing. Undiagnosed.
- **Dedup may over-collapse.** 13 sign and 13 action duplicates were removed on 8 clips;
  genuinely repeated signage (a speed limit restated 200 m later) is indistinguishable from
  a padding artifact under the current key.

---

## 6. Priority order to close it (cheapest-first, each independently valuable)

1. **Wire `obstacle.offline`** → agent slots → unblocks 5 tactical goals *and* the
   LONGITUDINAL metric family. Already-available data; no GPU; no PI decision.
2. **Engine-A situation labels** — call `situations.py` in `engine_a_summary`; free, exact,
   and the label side of the scenario gap. Keep information-disjoint from B4.
3. **B5 tactical call + B6 vision-only situation call** in `ph0_v2.py` — the VLM half.
4. **SAM3 `--mode text` tracks** → independent agent slots, cross-checkable against (1).
5. **Range from intrinsics** → `stopline_arc_m`, `position_arc_m`, sign distance.
6. **`ABORT_LC` detector** in `refb_labels` — the one missing `a_tac` LAT token.

*Items 1–2 need no PI decision and no GPU. Item 4 is running.*
