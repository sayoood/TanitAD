# PREREG — NAVSIM v2 EPDMS, ONE-STAGE, on `navtest` (`navsim_v2 --split navtest_single_stage`)

**W8, EvalFlyWheel, 2026-09-26.** Written and hash-banked (`raw/PREREG_HASH.txt`) **before any
navtest v2 score existed**: at writing time the only v2 navtest artifacts on this box are the
metric caches (a 200-token smoke cache, verified, and the 12,146-token build in progress). No
scorer has been run on navtest with the v2 devkit. Every expectation below is derived from
**source** (two devkits, file:line) and from **independently banked v1.1 artifacts** (W3), never
from a v2 number.

## 0. Why the acceptance is INTERNAL

`published_results.json` has 18 `EPDMS_v2_navtest_single_stage` rows: 8 `fix151: pre`, 6
`unverified`, 4 `post` — and the 4 post rows are all competitor MODELS. The one reference agent,
`Human (logged)` 90.3, is `fix151: pre`, basis INFERRED. NAVSIM #151 (2025-09-29) changed the human
filter; our devkit (`0a380a9`) is post-fix. ⇒ **there is no post-#151 external floor or human
reference for this column.** 90.3 is reported beside our HUMAN number, labelled
**pre-fix / INFERRED / NOT COMPARABLE**, and is never a pass/fail target.

The reference is **W3's NAVSIM v1.1 navtest run** (`FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/{CV,STOP,HUMAN}_navtest/*.csv`,
12,146/12,146 tokens each, reproduced against the NAVSIM paper; sha256 CV `21257aa4…`, STOP
`b60e305e…`, HUMAN `3a2090fb…`). It is an independently produced artifact (different devkit tree,
different metric cache, different run) — which is what lets it test the v2 wiring rather than
re-measure the v2 wiring's own determinism.

## 1. The protocol being wired

| item | value | source |
|---|---|---|
| runner | `navsim.planning.script.run_pdm_score_one_stage` (via the suite's `devkit_side/navsim_win.py --script pdm_score_one_stage`) | v2 `run_pdm_score_one_stage.py` |
| split | devkit `train_test_split=navtest`: **12,146 tokens, 136 logs**, token set and order IDENTICAL to the v1.1 filter (only diff: trailing whitespace on the `tokens:` line) | both `scene_filter/navtest.yaml` (MEASURED 2026-09-26) |
| background traffic | **NON-REACTIVE log replay**, `traffic_agents=non_reactive` passed EXPLICITLY | v2 `default_common.yaml:22` (the runner's default), `default_evaluation.yaml:2`, `traffic_agents_policies/log_replay_traffic_agents.py:30-60`; docs/traffic_agents.md: "Identical to NAVSIM v1" |
| logs | `D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test` (147 files ⊇ the 136) — NOT the C: dir (76, navhard's): `dataloader.py:34-36` silently skips absent logs | MEASURED |
| human filter | ON (`pdm_scorer.yaml:27 human_penalty_filter: True`), post-#151 form | v2 `evaluate/pdm_score.py:171-219` |
| summary row | ONE row, token `average_all_frames` = `pdm_score_df[score_cols].mean(skipna=True)` | v2 `run_pdm_score_one_stage.py:292-299` |
| EC | two-frame extended comfort between ADJACENT tokens of the same log (≤ 0.55 s) — ⚠ depends on WHICH tokens are scored together, so a SUBSET run's per-token `score` is not the full-split one | `run_pdm_score_one_stage.py:129-156, 200-227` |
| tier / loop | T1-family; **OPEN loop** (PI ruling 2026-09-02: one query on a real logged frame, plan then fixed; ego never re-queried) | — |

⚠ **The traffic policy of the PUBLISHED rows is UNVERIFIED.** No banked primary (2605.09701,
2512.07745, 2604.03581, 2506.04218 — searched 2026-09-26) states it. We use the runner's default.
E1's navhard N2b used `traffic_agents=reactive` — a different setting; not mixed here.

## 2. v1.1 vs v2 — which sub-metrics kept their definition (MEASURED from source)

Trees: v1.1 = `D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896` (unpacked,
`3e8291b`); v2 = `D:/Archive/devbox-C/navsim/devkit` (git, `0a380a9`). Scorer = `…/pdm_planner/scoring/pdm_scorer.py`.

**Shared path, verified text-identical (formatting / import order only):** `batch_lqr.py`,
`batch_kinematic_bicycle.py`, `pdm_simulator.py` (the LQR + bicycle ego propagation),
`pdm_scorer_utils.py` (`get_collision_type`), `pdm_occupancy_map.py`, `pdm_object_manager.py`,
`pdm_closed_planner.py`, `abstract_pdm_planner.py`, `constant_velocity_agent.py`, `human_agent.py`,
`evaluate/pdm_score.py::transform_trajectory / get_trajectory_as_array` (v1 L24-80 = v2 L26-82), the
initial ego state (v1 `navsim_scenario.py:163-180` = v2 `:170-187` + `navsim_scenario_utils.py:34-46`),
the PDM-Closed map inputs (`map_radius 100`, v1 `metric_cache_processor.py:46,60` = v2 `:54,68`), and
the scored proposal pair `[PDM-Closed, agent]` (v1 `pdm_score.py:107` = v2 `:144`).

| sub-metric | v1.1 | v2 | definition | what differs in the INPUTS | pre-registered relation |
|---|---|---|---|---|---|
| **DAC** | scorer L351-358 | L422-429 | **IDENTICAL** (off-road = any corner outside every drivable polygon, `_calculate_ego_area` v1 L240-291 = v2 L310-361) | none: states + drivable map only, no observation | **EXACT per token** |
| **NC** | L293-349 | L363-420 | **IDENTICAL** text | v2 log replay **removes every agent whose box intersects the ego at t0** from all frames (`log_replay_traffic_agents.py:35-58`); v1's cache observation keeps them with `collided_track_ids = []` (v1 `pdm_observation.py:236-260`, L258). Object set v2 ⊆ v1, and NC is monotone in the object set | **EXACT per token outside E-T0; v2 ≥ v1 on E-T0** |
| TTC | L414-498 | L492-580 | loop domain TRUNCATED: v2 evaluates `time_idx` 0..31 (L536-538, `num_poses − 9`) vs v1 0..40 (L456); same objects (t ≤ 4 s) minus E-T0 | + human filter only raises | **one-sided: v2 ≥ v1 per token** |
| DDC | L360-396 (weighted, **weight 0**, L42) | L431-474 (MULTIPLIER) | oncoming progress **zeroed inside intersections** in v2 (L445-451) ⇒ less oncoming progress | + human filter only raises | **one-sided: v2 ≥ v1 per token** |
| EP | raw L398-412 (identical text); normalisation L163-173 | raw L476-490; normalisation L229-238 | CHANGED: v1 masks by NC·DAC and divides masked by max-masked; v2 masks the NORMALISER by NC·DAC·DDC·TLC and divides the UNMASKED raw, clipped to [0, 1] | — | none (descriptive agreement only) |
| comfort | `comfort` L500-509 (4 s horizon) | `history_comfort` L656-700 (padded with 1.5 s of logged past) | DIFFERENT metric | — | none |
| TLC, LK, EC | — | L582-613, L615-654, scene_aggregator | NEW in v2 | — | none |
| score | PDMS `NC·DAC·(5EP+5TTC+2C)/12` | EPDMS `NC·DAC·DDC·TLC·(5EP+5TTC+2LK+2HC+2EC)/16` (EC NaN ⇒ /14) | DIFFERENT | — | none (never compared) |

**The v2 human filter** (`pdm_score.py:171-219`): on ORIGINAL frames the logged human trajectory is
scored ALONE; every column (except `multiplicative_metrics_prod`, `weighted_metrics`,
`weighted_metrics_array`, `pdm_score`, L194) where the human reads **exactly 0** is set to **1** for
the agent. ⭐ **MEASURED from v1 (before any v2 score): on navtest the v1.1 HUMAN reads NC = DAC =
TTC = 1.0 on 12,146/12,146 tokens** (the split was filtered to scenes the human passes) — so, on NC
and DAC, the filter is predicted to fire on **0** tokens and the relation above needs **no**
substitution. (DDC: the v1 human reads 0 on 82 and 0.5 on 121 tokens — covered by the one-sided rule.)

Two source facts the filter adds, both exercised by §3 C-HUMF: the human-alone scoring gets NO past
trajectory (`pdm_score.py:184-192` omits it ⇒ `history_comfort` = 1 for the human alone, L663) and its
EP is never 0 — so HC and EP are **not** filtered in effect, and HUMAN may read HC = 0 or EP = 0.

## 3. Pre-registered checks (all per token unless stated; tolerance **0 violations** — the values are discrete and computed by identical code on identical inputs; a violation is a FINDING, never absorbed)

Arms: **CV** (devkit `constant_velocity_agent`), **STOP** (all-zero plan through the suite's seam
agent), **HUMAN** (devkit `human_agent`, privileged, reference only). v1 reference = W3's CSV of the
same arm.

| id | check | PASS iff |
|---|---|---|
| **C-COUNT** | runner log `successful == N`, `failed == 0`; CSV = N valid token rows + EXACTLY one summary row `average_all_frames`; token set == the requested set | all three arms |
| **C-FORMULA** | per token `score == NC·DAC·DDC·TLC·(5EP+5TTC+2LK+2HC+2EC)/16`, EC NaN ⇒ /14 (E1's `c4`, `summarize.py`) | max \|Δ\| ≤ 1e-9 |
| **C-AVG** | the `average_all_frames` `score` == the skipna mean of the N token scores | \|Δ\| ≤ 1e-12 |
| **C-DAC** (primary) | `v2_DAC(t) == v1_DAC(t)` | 0 mismatches, each of CV / STOP / HUMAN |
| **C-NC** (primary) | `v2_NC(t) == v1_NC(t)` for t ∉ E-T0, `v2_NC(t) ≥ v1_NC(t)` for t ∈ E-T0 | 0 violations, each arm |
| **C-TTC** | `v2_TTC(t) ≥ v1_TTC(t)` | 0 violations, each arm |
| **C-DDC** | `v2_DDC(t) ≥ v1_DDC(t)` | 0 violations, each arm |
| **C-HUMF** (structural, the post-#151 filter is live) | v2 HUMAN reads **1.0** on NC, DAC, TTC, TLC, LK on every token and DDC ∈ {0.5, 1} (no 0) | 0 exceptions |
| **H-HIGH** | v2 HUMAN EPDMS (full split) | ≥ 0.90 — reported beside 90.3 labelled pre-fix / INFERRED / NOT comparable |
| **K-NEG** (the check can fail) | C-DAC re-run pairing v2 **CV** with v1 **STOP** | ≥ 1,000 mismatches (v1 CV and STOP DAC differ on 4,699 tokens) |
| **K-MUT** (mutation) | flip ONE DAC cell of the v2 CV CSV in memory, re-run C-DAC | exactly 1 mismatch |

**E-T0** (the only exception set) = tokens whose metric cache holds an agent box intersecting the
ego footprint at t0 — computed from the CACHE CONTENT by a separate script with the SAME expression
as `log_replay_traffic_agents.py:35-46` (`observation.detections_tracks[0]` vs
`ego_state.car_footprint.oriented_box.geometry`), never from a scorer output. Its size is reported;
it is not tuned.

**Floors (reported, not bars):** STOP and CV on every run. Directional prediction: **STOP > CV** on
EPDMS (as on every NAVSIM split measured so far). Every row is read against STOP.

**Descriptive only (no bar):** per-token agreement rate of EP between v1 and v2; the EPDMS headline
of each arm; per-log means; the log-cluster interval (`navsim_log_cluster_bootstrap` over the 136
`log_name` clusters, `taniteval/adapters/navsim_ci.py`, B = 2000); paired deltas vs STOP and CV.

## 4. Stages and decision rule

1. **SMOKE** (`raw/smoke_tokens_sub200.json` = W3's banked sub200, 200 tokens / 93 logs, chosen by
   W3 on 2026-09-20): all §3 checks except H-HIGH, on the 200 tokens, against the verified smoke cache
   `C:/Users/Admin/navsim-crun/exp/metric_cache_navtest_v2_smoke200`. Per-token checks C-DAC / C-NC /
   C-TTC / C-DDC / C-HUMF do not depend on the subset; EC and `score` do, so no smoke `score` is ever
   compared with a full-split one. **From the smoke, the full-split scoring wall-clock is PROJECTED per
   arm and reported BEFORE full scoring starts.**
2. **FULL SPLIT** (12,146): all §3 checks. Scheduled AFTER the refcv6 FINAL battery (brief), or
   strictly RAM-gated (≥ 9 GB available, sustained).
3. **Verdict:** the wiring is **VALIDATED** iff every §3 check passes on the full split. A failed check
   is reported as **FAILED as pre-registered**, then diagnosed and the next lever run (RULE ZERO); the
   bar is never moved and no check is dropped after the data.

## 5. What this pre-registration does NOT claim

* It does not make our number comparable to the 8 `pre` / 6 `unverified` published rows — only to the
  4 `post` rows, and even those only if their traffic policy was non-reactive (UNVERIFIED).
* NavSim's `driving_command` is a ROUTE-LEVEL ORACLE (W2); no route skill is claimable from a
  command-conditioned row. The floors here consume no command.
* One-stage EPDMS is the protocol the benchmark authors DISCOURAGE ([N2] §4.2); it is wired because
  it is on the leaderboard, and because navtest is the only NavSim split where every token has a
  logged human future (the four families are computable on all of it).

---

## AMENDMENT A1 — 2026-09-26 15:53 (Berlin), written BEFORE any smoke score was read

**What happened (MEASURED, a harness finding, not a result):** the sub200 smoke CRASHED in the devkit's
one-stage AGGREGATION on its first arm (CV): all 200 tokens were scored (`pdm_calls=200`), then
`create_scene_aggregators` raised `ValueError: No objects to concatenate`
(`run_pdm_score_one_stage.py:222`, `pd.concat(all_updates)` over an EMPTY list). Mechanism, from source:
EC pairs a token with the ADJACENT earlier token of the same log (≤ 0.55 s, `infer_start_adjacent_mapping`,
L129-156); sub200 is W3's sorted-token-order sample (~2 tokens per log over 93 logs), so it contains **no
adjacent pair at all**, the mapping is empty, and the devkit's aggregation cannot handle an empty mapping.
The full split cannot hit this (consecutive navtest frames are 0.5 s apart within every log). No score of
this run was read before this amendment; its banked outputs are kept as the evidence of the failure.

**Change (the smoke SUBSET only; no check, bar or tolerance changes):** the smoke becomes **every navtest
token of the first K logs of `scene_filter/navtest.yaml` in yaml order, K = the smallest prefix with ≥ 200
tokens** ⇒ K = 3, **218 tokens** (43 + 19 + 156). A whole-log subset has two properties the random one lacks:
(i) the aggregation is defined; (ii) EC adjacency is WITHIN a log (L142 groups by `log_name`), so every
smoke token's `score` is the SAME as it will be in the full split — the smoke's per-token rows become a
direct preview of the full-split rows, not a different quantity.

**What is kept from sub200:** its per-token PRE-AGGREGATION rows (NC, DAC, DDC, TLC, EP, TTC, LK, HC —
the aggregation only adds EC and `score`) are valid per-token metric values and are checked against §3's
per-token rules (C-DAC, C-NC, C-TTC, C-DDC, C-HUMF) as a SECOND smoke sample, reported separately.

---

## AMENDMENT A2 — 2026-09-26 (Berlin), written AFTER the whole-log smoke and BEFORE any full-split score exists

**The smoke verdict stands as recorded.** On `smoke218` (the A1 subset, run
`_scratch/navsim_v2/navtest_single_stage/20260926T135821Z-navsim_v2-none-ce7ac0`) the pre-registered checks
read **23 of 24 PASS**; **C-NC[CV] FAILED as pre-registered: 1 violation / 218** (token `937ca624cc2658a6`,
v2 NC 1.0 vs v1 NC 0.5; `raw/cross_protocol_smoke218.json`). It is not relabelled.

**Diagnosis (MEASURED; three independent probes, `raw/diag_937ca624/`, `raw/counterfactual_smoke218_*.json`):**
1. §2 missed an INPUT difference. The metric cache's observation is interpolated over **5.0 s in v1.1**
   (v1 `metric_cache_processor.py:101`, `time_horizon = 5.0`) and **4.0 s in v2** (v2 `:109`), and an object
   observed EXACTLY ONCE in the window is placed at EVERY step (`StateInterpolator` start == end →
   `initial_detection_track` appended; v1 `:165-166`, v2 `:177-178`) — a GHOST. The token's v1 cache holds
   5 objects at all 41 steps that v2 holds at none (1 VEHICLE, 4 GENERIC_OBJECT); the LOG shows each of the 5
   observed ONLY at +5.0 s (`raw/diag_937ca624/ghost_objects_log_check.json`); CV's straight path meets a
   GENERIC_OBJECT ghost → NC 0.5 (static) in v1.
2. The difference is an INPUT, not a FUNCTION: re-scoring each arm's own plan with the **v2 function on the
   v1.1 cache's inputs** reproduces W3's v1.1 NC and DAC **exactly on 218/218 tokens for CV, STOP and HUMAN**
   (TTC never below v1), while the **v2 function on the v2 inputs** reproduces our run's CSV exactly (the
   control: the re-scoring path IS the scorer's). On the failing token: v2 inputs → NC 1.0, v1 inputs → NC 0.5.
3. Ghosts go BOTH ways (an object seen once inside 0–4 s but again at +4.5/+5.0 s is a v2 ghost and a v1
   interpolated track), so C-NC and C-TTC are two-sided wherever the observations differ: per-token
   classification of both caches (`code/e_obs_classify.py`, cross-tree read control 418/418 identical):
   smoke218 IDENTICAL 9 / V1_SUPERSET 28 / V2_EXTRA 181; sub200 23 / 30 / 147.

**Pre-registered NOW for the FULL split (12,146 tokens) — in ADDITION to §3, which is still evaluated and
reported with its own verdicts (nothing dropped):**

| id | check | PASS iff |
|---|---|---|
| **C-NC-FN** (function identity) | v2 function on v1.1 inputs vs W3's v1 CSV, on the DISCREPANCY set (every token where v2 NC or DAC ≠ v1 for any arm) ∪ a fixed sample (every 25th token in sorted order), each arm | NC and DAC mismatches = **0**; TTC below v1 = **0**; and the CONTROL (v2 function on v2 inputs vs our CSV) mismatches = **0** |
| **C-NC-CLS** | per `e_obs_classify.py` class: IDENTICAL → NC exact; V1_SUPERSET → v2 NC ≥ v1 NC | 0 violations, each arm |
| **C-TTC-CLS** | on IDENTICAL ∪ V1_SUPERSET: v2 TTC ≥ v1 TTC | 0 violations, each arm |
| C-DAC, C-DDC, C-HUMF, C-FORMULA, C-AVG, C-COUNT, H-HIGH, K-NEG, K-MUT | unchanged (§3) | as §3 |

**Verdict rule for the full split:** the wiring is VALIDATED iff every row above AND every §3 row other than
C-NC / C-TTC passes; the original C-NC / C-TTC are reported beside it with their counts, and each of their
violations must fall in a token the classification marks non-IDENTICAL (a violation on an IDENTICAL token
FAILS the wiring outright).
