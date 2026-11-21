# AMENDMENT 3 (DRAFT, for Master Mind freeze) to BEHAVIOUR_SCORECARD.md (sha256 fac6c7f1...): nuPlan-seeded junction and traffic-light slots

**Status:** DRAFT. Stream BEHAV-A3, 2026-11-08. Nothing here is in force until the Master Mind freezes it. No checkpoint has been scored. `BEHAVIOUR_SCORECARD.md` is not edited.
**Scope:** adds token predicates (§3.1 style), the map-scene fields for rules c1-c4, and two Master Mind rulings on a1/a2 and c1 (§7). No threshold, window or adoption bar changes.
**Evidence class of the pool counts:** MEASURED, `amendment3/feasibility_report.json`, produced by `amendment3/feasibility_a3.py` over all 136 navtest logs. Descriptive only: no token is selected here and no ranked list was looked at.

## 1. What the map and the logs actually hold (verified by opening the files)

Opened `F:/Archive/devbox-C/navsim/data/maps/<city>/<version>/map.gpkg` for all four cities with sqlite (`gpkg_contents`, `pragma table_info`).

| fact | source |
|---|---|
| Layer `lane_connectors`, columns `fid, geom, exit_lane_fid, entry_lane_fid, lane_group_connector_fid, intersection_fid, turn_type_fid, traffic_light_stop_line_fids, bulb_fids, ...` | gpkg (Las Vegas table_info; values probed in all four) |
| Layer `gen_lane_connectors_scaled_width_polygons`, join column `lane_connector_fid` | gpkg; devkit `nuplan_map.py:81` (`_LANE_CONNECTOR_POLYGON_LAYER`) |
| Layer `intersections`, column `intersection_type_fid`. Devkit maps `SemanticMapLayer.INTERSECTION` to it | gpkg; `nuplan_map.py:65` |
| Layer `stop_polygons`, column `stop_polygon_type_fid`. Devkit maps `STOP_LINE` to it | gpkg; `nuplan_map.py:66`; `stop_line.py:23` |
| Intersection types: `DEFAULT 0, TRAFFIC_LIGHT 1, STOP_SIGN 2, LANE_BRANCH 3, LANE_MERGE 4, PASS_THROUGH 5`. All six occur in every city. Every connector has an `intersection_fid`, so **the type, not the join, is what marks a junction** | `maps_datatypes.py:85-95`; gpkg counts |
| Stop-line types: `PED_CROSSING 0, STOP_SIGN 1, TRAFFIC_LIGHT 2, TURN_STOP 3, YIELD 4, UNKNOWN 5` | `maps_datatypes.py:61-71` |
| Connector turn type: `STRAIGHT 0, LEFT 1, RIGHT 2, UTURN 3, UNKNOWN 4`. The devkit's `NuPlanLaneConnector.turn_type` raises `NotImplementedError`, so the column is read directly | `maps_datatypes.py:49-58`; `lane_connector.py:194-196` |
| Connector graph: `exit_lane_fid` is the **incoming** lane, `entry_lane_fid` the **outgoing** lane. The names read backwards | `lane_connector.py:63, 81` |
| Stop lines of a connector: fids in `traffic_light_stop_line_fids` (comma-separated; empty when unsignalised). Devkit keeps those whose polygon intersects the baseline, else the nearest to the baseline start | `lane_connector.py:156, 159-192` |
| Signals: OpenScene `traffic_lights` = list of (`lane_connector_id`, `is_red`). `is_red` True maps to RED, False to GREEN. No yellow channel | NAVSIM v1.1 `navsim_scenario.py:282-283` (the `navsim-3e8291b...` tree, `setup.py:15` version 1.1.0). The `devkit/` tree is 2.0.0 and its line numbers differ |
| DB `traffic_light_status(token, lidar_pc_token, lane_connector_id, status)`; the devkit query reads green/yellow/red | DB schema; `nuplan_scenario_queries.py:612-616` |
| OpenScene equals the DB on the signal: 330 of 330 connector entries over 101 frames of one log agree (red/True 231, green/False 99), 0 OpenScene-only entries, and that log's whole table holds only `green` and `red` | MEASURED, `2021.09.09.17.18.51_veh-48_01248_01450`. **One log only: UNVERIFIED that no other log holds a yellow.** UNVERIFIED how a DB yellow, if any exists, is converted by OpenScene |
| All 1,236 signalised connectors (four cities) have at least one type-2 polygon. Stop point source: 959 on the baseline, 277 nearest polygon, 0 connector entry | `feasibility_report.json` `map_stats` |

Map turn type and connector geometry agree on 96.9 % of geometric L/R and 94.5 % of geometric S junction connectors (`map_stats.geo_vs_turntype`); the `bend` and `U` classes belong to no slot. The predicates require both to agree.

## 2. Common definitions

Population and rules are unchanged from §3.1: navtest tokens, window `[t0 - 2 s, t0 + 15 s]`, human safe, order by `sha256(token + "|refe-behaviour-v1")` ascending, one token per log, 5 candidates per slot (3 where the pool is thin, shortfall reported), QC and fill as written.

**Human safe** is the frozen selector's own flag: `ok and safe` from `select_suite_tokens.py` (`tsf_bundle2_20261107/run2/per_token_results.pkl`, DB ego track and `lidar_box` at 10 Hz). It is not recomputed here. 2 Hz annotated frames are used for every other predicate, so the selector and the sim read the same inputs.

Let `j` be the token's frame index in its OpenScene log (located by list index) and `W` the 35 frames `j-4 .. j+30`. The script's window check excludes tokens with a dt gap (154) or too close to the log end (536), leaving 11,456 of 12,146.

- **Ego:** rear axle = `ego2global` translation; heading from its rotation; `v = hypot(ego_dynamic_state[0], [1])`; front bumper = rear axle + 4.049 m.
- **Human path:** the rear-axle points of `W`, resampled every 0.5 m; for stations of points beyond its end, extended straight along the final heading by 300 m (A-6).
- **Extended human path (only for the governing-connector search):** `W` continued for up to 60 further frames (30 s) while the 2 Hz log stays continuous. Selection may look past the window; the scenario does not.
- **Traversed connector:** baseline sampled every 1 m lies within 1.5 m of the path end to end; projected entry station < exit station. Ordered by entry station, ties by mean distance to the path.
- **Junction connector:** `intersections.intersection_type_fid` of the connector's `intersection_fid` in {0, 1, 2}.
- **Turn class of a connector:** `dpsi = wrap(h1 - h0)` where `h0` and `h1` are baseline headings over the first and last 2 m. **L** `dpsi >= +45 deg`; **R** `dpsi <= -45 deg`; **S** `|dpsi| <= 20 deg`. Anything else (bends, U-turns `|dpsi| >= 150`) belongs to no slot. **The class must equal the map's `turn_type_fid`** (S 0, L 1, R 2).
- **Legal entry:** at the first frame of `W` where the front bumper reaches the connector's entry station, the connector is not listed red.
- **Connector polygon:** the row of `gen_lane_connectors_scaled_width_polygons` with `lane_connector_fid` = the connector.
- **Conflict set `X(c)` of connector `c`:** connectors whose polygon intersects `c`'s with overlap area >= 0.5 m^2, excluding `c`, its siblings (same `exit_lane_fid`) and merges (same `entry_lane_fid`). `rel` = absolute difference of the two connectors' initial headings.
- **Moving agent hit:** at a frame of `W`, an agent of class `vehicle` or `bicycle` with speed >= 1.0 m/s whose box footprint intersects the overlap region and whose heading is within 45 deg of the conflicting connector's tangent at its projection. This drops a lead vehicle travelling along the ego's own connector.
- **Left-hand traffic.** `sg-one-north` drives on the left, so the **far-side turn across oncoming traffic** is geometric R there and the near-side turn is geometric L. J-UL and J-R are named by function. UNVERIFIED that the Master Mind wants Singapore in J-UL (see §8, Q2).

## 3. Junction slots

The ego connector is the first traversed connector in entry-station order that passes all clauses. These slots carry no `tokens_cruise.json` clause.

- **J-UL (unprotected far-side turn).** Junction connector, turn class far-side (L, or R in left-hand traffic), legal entry, and an oncoming set `X_o(c)` = connectors of `X(c)` with `rel >= 135 deg`, not merges. Holds when **at least one moving agent hit occurs on `X_o(c)`** in some frame of `W`, and **at least one connector of `X_o(c)` is not listed red** at the entry frame (an unlisted oncoming connector counts as not red, so stop-controlled junctions qualify).
- **J-R (near-side turn).** Junction connector, turn class near-side (R, or L in left-hand traffic), legal entry. No agent requirement (the predicate is the turn). If a moving agent hits the overlap of any `X(c)` connector the zone is emitted (§5); otherwise the zone stays empty and c3 is NA.
- **J-S (straight through a junction with crossing traffic).** Junction connector, class S, legal entry, and a crossing set `X_p(c)` = connectors of `X(c)` with `60 deg <= rel <= 120 deg`, not merges, with at least one moving agent hit on `X_p(c)`.

## 4. Traffic-light slots

**Governing connector:** the first connector traversed on the extended human path that appears in `traffic_lights` in at least one frame of `W`. Connectors entered within 3 m of the first such one are a tie, resolved by smaller mean distance to the path, then smaller id. If none is traversed, the token has no governing connector and is not eligible for the primary slot.

Let `state[k]` be 1 (red) if the governing connector is listed red at frame `k`, 3 (green) if listed not red, 0 if unlisted. `before[k]` = front bumper station <= `stop_line_s` (§5, rear-axle station of the stop point on the extended path for this feasibility count). `V[k]` = human speed.

Common to both: let `k1` be the first frame with `V <= 0.1 m/s` (c2's `STOPPED_V`), `before[k1]`, `state[k1] == 1`, and gap = `stop_line_s - front_bumper(k1)` with `0 <= gap <= 6.0 m` (the c2 cap, so the human passes c2 at Gate H).

- **TL-G (red, then green, human proceeds).** Additionally: there is a first frame `k2 > k1` with `state[k2] == 3`; `V <= 1.0` (c4's `C4_STOP_V`) at `k2 - 1` and for the 2 frames (1.0 s) ending there (c4's `STOP_MIN_S`); the front bumper crosses the line in `[k2, window end]` with `max V[k2:] > 1.0`; and **no frame has `state == 1` with the bumper past the line** (the human passes c2 as it is coded today; see CH-3).
- **TL-S (stop at red, still red at the end).** Additionally: there is no green frame after `k1`; `state[k1:]` is all red; the bumper stays before the line through the window end; a run of `V <= 0.1` of at least **5.0 s** (2 Hz frames) inside `[k1, end]`; and a **rolling approach**, `max V[:k1] >= 3.0 m/s`, so the ego is actually asked to brake to a stop. The 5.0 s was chosen for the stop to outlast warm-up transients, and the pool holds at 3 s and 8 s (§6), so the choice does not rest on the data.

The predicates use the 2 Hz annotation frames; `stop_line_s`, the signal timeline and the agent set are then computed once by the fields below, and the sim replays them.

## 5. Map-scene fields (computed from gpkg + OpenScene meta, identical for selector and sim)

- **`stop_line_xy(c)`** (rule c1, c2): fids in `lane_connectors.traffic_light_stop_line_fids` whose `stop_polygons.stop_polygon_type_fid == 2`. (a) Among those intersecting `c`'s baseline, the baseline point of the **smallest baseline station** of any intersection (the upstream edge). (b) Else, the point of the polygon nearest the baseline start (the devkit's fallback, `lane_connector.py:181-192`). (c) If the column is empty or lists no type-2 polygon, the connector's baseline start. Junction slots use the ego connector; TL slots use the governing connector. **Layer used: `stop_polygons`** (connector entry point is the fallback only, used by 0 of 1,236 signalised connectors).
- **`stop_line_s`** (schema key): the station of `stop_line_xy` on the human reference path in the schema's frame, i.e. `HumanPath.project` (`behaviour_score.py:306`) over the **box-centre** positions at the 10 Hz samples of `human.npz`, using the scorer's straight extension (A-6). A point beyond the human's last position lands on the extension. Emitted for every J and TL scenario. The rear-axle version in §4 is the selector's approximation for the pool count only.
- **`signal_state`, `signal_governs_ego`:** heads = connector ids appearing in `traffic_lights` over `W`, sorted ascending (`H` = their number). Per 20 Hz sample, zero-order hold of the latest 2 Hz frame: `1` red if listed `is_red`, `3` green if listed not red, `0` if unlisted at that frame. `signal_governs_ego` is True only for the governing connector (TL slots) or the ego connector (J slots, when listed); all False when the J-slot connector is unlisted. No yellow is produced.
- **`conflict_zone_xy`:** with `S = X_o(c)` for J-UL, `X_p(c)` for J-S, all of `X(c)` for J-R: the **convex hull** of `union over o in S of (polygon(c) intersect polygon(o))`, exterior ring, CCW, no repeated closing vertex, as `(K,2)` float64. The hull exists because the schema takes one polygon and the scorer tests one ring (`_box_in_zone`, `behaviour_score.py:364`); the union is multipart for 158 of 506 J-S tokens (up to 6 parts, `feasibility_report.json`). Emit shape `(0,2)` when `S` is empty or no agent hit exists.
- **`conflict_agent_ids`:** the track tokens with at least one moving agent hit on a connector of `S` in `W`, at the 2 Hz annotation frames, mapped by the sim to its `agent_id`. Empty means every agent counts (A-13), so **whenever this list is empty the zone is emitted empty** (`(0,2)`) and c3 reads NA. The list is fixed at selection time and is identical for every checkpoint.

## 6. Feasibility over navtest (MEASURED, 136 logs, 11,456 windows, 9,835 human-safe)

Counts are after human safe. Pools below 5 logs are thin under §3.1 item 2.

| slot | tokens | logs | logs by city | status |
|---|---|---|---|---|
| J-UL | 20 | **4** | sg-one-north 1, pittsburgh 3 | **SHORTFALL of 1** (thin clause allows 4 candidates) |
| J-R | 833 | 56 | sg 9, boston 19, vegas 15, pittsburgh 13 | ok |
| J-S | 506 | 67 | sg 6, boston 28, vegas 27, pittsburgh 6 | ok |
| TL-G | 53 | 10 | boston 6, vegas 4 | ok; two cities |
| TL-S (5 s, rolling) | 45 | 8 | boston 7, pittsburgh 1 | ok; Boston-heavy |

Sensitivity, same counting: TL-S 3 s rolling 53 tokens / 8 logs; 8 s rolling 25 / 5; 5 s without the rolling clause 50 / 9. TL-G without the c2-as-coded clause 60 / 10; and, also without that clause, with strictly no unlisted frames between stop and green 46 / 9.

**J-UL shortfall. The strict pool fails the 5-log requirement because 48 of 68 human-unfiltered tokens are not human-safe under the selector, and few left turns actually meet an oncoming vehicle in the zone inside 17 s.** Explicit ALTERNATIVES, not applied:
- **ALT-B (minimal):** oncoming threshold `rel >= 120 deg`: 21 tokens, **5 logs** (adds one Boston log).
- **ALT-A (wider, changes the meaning):** any non-merge crossing connector in place of the oncoming set: 127 tokens, 26 logs (cross-street traffic also qualifies).
- Relaxing human safe is **not** offered; §3.1 requires it in every slot. Before the safety filter the strict J-UL pool is 68 tokens.

Instrument checks: a deliberate-regression arm (inverting `is_red`) took TL-G from 60 to 0 and TL-S 5 s from 50 to 8 (the arm is off in the banked script). Two tokens drawn by `sha256(token + "|handcheck-a3")`, one TL-G and one TL-S, read correctly against the **DB directly** (signal letters and ego `vx, vy`). J-UL, J-R and J-S were not hand-checked this way: UNVERIFIED beyond the pipeline and its controls.

## 7. Rule mapping (no code edited; every item below is a request)

| slot | c1 | c2 | c3 | c4 |
|---|---|---|---|---|
| J-UL | if signalised or zone present (rolling approaches only, ruling 2) | if signalised and a red occurs | **primary** | if the ego stops and a green or lead pull-away occurs |
| J-R | same | same | only when the zone is emitted | same |
| J-S | same | same | **primary** | same |
| TL-G | NA at the stop approach (ruling 2) | **primary** | NA (no zone) | **primary** |
| TL-S | NA (no green before the end) | **primary** | NA (no zone) | NA (no launch) |

`category` is `junction` or `traffic_light`, `synthetic = False` for these. The scorecard's sentence that c1-c4 are scored only on the synthetic block is superseded for these slots; the `nuplan_seeded` block in `behaviour_suite.py` already separates them.

### Master Mind rulings (Gate H finding on a synthetic stop-at-red-then-go-on-green case; no threshold changes)

**Ruling 1, a1 and a2 free-road intervals: a held tick is not free road.** A tick is *held* if it lies inside a hold, and:
- A **hold starts** at a tick where the governing signal is red (`signal_governs_ego` true, `_gov_state == SIGNAL_RED`) and the ego's front bumper is before the stop line (`s_front <= stop_s + EPS`).
- A **hold ends** at the first tick at or after a governing green at which `v > P["C4_LAUNCH_V"]` (1.0 m/s). The launch ticks between the green and that tick stay held, so a1 does not punish the launch that c4 already judges.
- A held tick is excluded from `free`. Warm-up, the 2.0 s minimum interval and everything else are unchanged. Applied separately to the rollout and the human reference, each on its own stations.

**Ruling 2, c1 applies to rolling approaches only.** If the ego is stopped at the first clear tick of the approach, c1 is NA for that approach and reports the reason. "Stopped" is c4's definition: `v <= P["C4_STOP_V"]` at that tick, within a run of such ticks of at least `P["STOP_MIN_S"]` (1.0 s). Start-from-stop is judged by c4, relative to the human.

### Requested changes to `code/behaviour_score.py`

| id | change | where | new tests |
|---|---|---|---|
| CH-1 (Ruling 1) | add `_signal_hold(R, D)` (latch from the first red-and-before-line tick to the first green-then-`v > C4_LAUNCH_V` tick); replace `free = ~(near \| closing)` with `free = ~(near \| closing) & ~hold` | `behaviour_score.py:453`, helper beside `_gov_state` (`:535`); free intervals and a2 pick it up through `free_intervals` (`:455`, `:488`) | (a) a stationary ego at a red before the line: no tick free, a1 PASS or NA, a2 NA; (b) launch ticks between green and `v > 1.0` not free, ticks after are free; (c) control: past the line on red stays free (a1 still scored); (d) control: head with `signal_governs_ego` False does not hold; (e) control: a scenario with no signal is bit-identical to today. Add a mutation site on the `& ~hold` term |
| CH-2 (Ruling 2) | in `rule_c1`, after `i0` is chosen: if `Rc["v"][i0] <= P["C4_STOP_V"]` and `i0` lies in a `_runs(v <= C4_STOP_V)` run of `>= nt(P["STOP_MIN_S"])` ticks, return `_na("stopped at the first clear tick; start-from-stop is judged by c4")` | `rule_c1`, `behaviour_score.py:583` | (a) start-from-stop at green: c1 NA with the reason; (b) control: rolling approach that then stalls below 1.0 m/s while clear: FAIL; (c) control: rolling approach holding above 1.0 m/s: PASS. Add a mutation site |
| CH-3 (needs a ruling, see Q3) | c2 counts a crossing as a **crossing event**, not any red tick past the line: `crossed = red[1:] & (sf[:-1] <= s0 + EPS) & (sf[1:] > s0 + EPS)` | `rule_c2`, `behaviour_score.py:606-607` | (a) crossing on green, light turns red later: 0 crossings, PASS (today it FAILs); (b) control: crossing on red: 1, FAIL |
| CH-4 (optional) | optional `slot` string key (J-UL ...) and a DESCRIPTIVE per-slot row in `behaviour_suite.py`, because `category` pools J-UL, J-R and J-S | loader and `behaviour_suite.py:85` | loader accepts both forms |

**Gate H case that must now read correctly (added to `test_behaviour_score.py`, using `synth_rollouts`):** the synthetic human that stops at a red then launches on green. Required: `a1` human status PASS or NA (no longer FAIL at share 0.75), `c1` NA with the CH-2 reason, `c2` PASS, `c4` PASS, and `a2` not FAIL. A Gate H read of the first real nuPlan TL-G scenario must also read these. `CODE_PIN.json`, `MUTATION_RESULT.txt` and `TEST_RESULT.txt` must be regenerated after CH-1 to CH-3; the sim's pinned copy needs the new sha.

## 8. HANDOFF notes

**Q1 (MM, n per slot).** The scorecard's table has junction 1 / 4 and traffic-light 0 / 4 (tier 1 / tier 2). Proposal: tier 1 keeps one junction slot (J-UL); tier 2 becomes J-UL 2, J-R 1, J-S 1, TL-G 2, TL-S 2. Whether the nuPlan-seeded slots replace the synthetic 720 and TanitSim signal scenarios, or sit beside them, is the MM's call; §3 forbids pooling them either way.
**Q2 (MM, J-UL).** Choose the strict pool with 4 candidates (thin clause), or ALT-B (5 logs) or ALT-A. Confirm the left-hand-traffic reading (Singapore far-side turn = geometric R).
**Q3 (MM, c2).** CH-3 corrects a defect that the synthetic cases could not show: a human or ego passing on green and seeing the light turn red before the window ends is counted as crossing on red. Measured on real data: 7 of 60 TL-G tokens are affected today. TL-G's primary predicate already excludes them, but the J slots and TL-G rollouts are exposed until CH-3 lands.
**Q4 (MM, Ruling 1 scope).** The hold has no distance bound, as ruled. A model that stops far short of a red has a non-free-road hold; c2's 6 m cap still reads it. Say if a 40 m bound (c1's approach radius) is wanted.
**Q5 (MM, Ruling 2 reading).** "c4's stop definition" is read as speed `<= 1.0` plus the 1.0 s run. The alternative is the speed test alone.

**UNVERIFIED / open:**
- Stop-polygon choice (upstream edge of a type-2 polygon): the human's stop gap on the TL-G pool is 0.46 / 5.58 / 5.65 m at p5/50/95 (69 tokens before safety; 48 of the 60 human-safe TL-G tokens before the c2-as-coded clause are in Boston), inside the 6 m cap but close to it. Worth the sim session's visual check on 2-3 rendered scenarios before the freeze.
- No yellow in OpenScene; the DB check is one log.
- The selector's human-safe flag covers ego vs `lidar_box` at 10 Hz; that is the §3.1 meaning, but it was computed by another agent's run (`run2`).
- TL slots concentrate in Boston and Las Vegas; pool composition is reported, not corrected.

**What the first attempt got wrong (fixed in `feasibility_a3.py`):** it had only been run with `--limit-logs 2`; used a TTC proxy instead of the selector's human-safe; used a stop speed of 0.3 m/s where c2's is 0.1; and a stop point ahead of the path could project just short of its end, which made every "human stays stopped before the line" read as a red-light crossing (TL-S pool was 0 for this reason). The station now uses the scorer's straight extension.

**Exact next step for the MM:** answer Q1-Q5, freeze this file (with `AMENDMENT_3.sha256`), then hand CH-1 to CH-3 and the tests to the metric-code stream and §5 to the sim session. The selector's implementation of the slots belongs to the sim session, built from `feasibility_a3.py`.

**Manifest:** this draft; `amendment3/feasibility_a3.py` (v2; the first attempt's script is kept as `feasibility_a3.v1_superseded.py`), `feasibility_report.json`, `feasibility_rows.csv.gz`, `feasibility_run.log`. Each lives only on E:.
