# Traffic-Light Scenario + TLC Metric — SC-14 intake (2026-07-24)

**Owner:** Benchmarks & Eval agent. **Closes:** the two honest gaps Sayed named in beyond-ADE eval —
(1) the **missing** traffic-light / signalized-intersection scenario + a traffic-light-handling
**metric**, and (2) converting the beyond-ADE suite from synthetic-fixture-validated toward **real**
numbers. Scenario is **SC-14** in `Opponent Analyzer/SCENARIO_DATABASE.md` (red-light running, Dallas;
W-03 rule-compliance family), flagged there as the cheapest next spec (reuses the SC-04 stop-line
barrier machinery).

**Evidence-class legend:** MEASURED (ours + artifact) · PUBLISHED · INHERITED · ESTIMATED · HYPOTHESIS.

---

## P1 — Traffic-light scenario + TLC metric  [DONE — synthetic-MEASURED]

### What was built
| artifact | repo path | what it is |
|---|---|---|
| `traffic_light.py` | `stack/tanitad/eval/scenarios/traffic_light.py` | SC-14 scenario: signalized approach, per-step signal phase (G→Y→R or stale green), + design-oracle `simulate_policy` (`rule_barrier` vs `soft_prior`) + `red_run_rate` reducer |
| TLC metric (additive) | `stack/tanitad/eval/metrics.py` | `compute_tlc` / `tlc_report` + `SIGNAL_*` encoding + `traffic_light_metrics` assembly (base-5 suite + TLC). No existing function modified. |
| metric tests | `stack/tests/test_tlc_metric.py` | 8 analytic fixtures, TLC hand-derived in each comment |
| scenario tests | `stack/tests/test_traffic_light_scenario.py` | 16 tests: contract, dilemma-zone guard, discriminative structure through the *real* metric, barrier invariance |
| design-oracle results | `…/incoming/2026-07-24-traffic-light-scenario-metric/traffic_light_oracle_results.json` | the numbers below, regenerable |

### The TLC metric (Traffic-Light Compliance)
Formula (in `stack/tanitad/eval/metrics.py`, full docstring on `tlc_report`):

```
TLC = red_entry_gate * stop_quality * green_flow          in [0,1], higher better; 0 = ran a red
```
- **red_entry_gate ∈ {0,1}** — the hard legal barrier: **0** iff the ego crosses the stop line while
  the signal is RED above a creep speed (running the red — a single violation zeroes the whole score).
- **stop_quality ∈ [0,1]** (only when a RED is faced) = `margin_factor * smooth_factor`:
  `margin_factor` = 1 for a halt within `TLC_MARGIN_COMFORT_M` (5 m) before the line, decaying for
  over-cautious far-back stops (a flow cost); `smooth_factor = 1/(1 + TLC_DECEL_K·max(0, peak_decel −
  TLC_COMFORT_DECEL))`, penalizing an emergency slam. N/A → 1.
- **green_flow ∈ [0,1]** (only on a genuine proceed-on-green) penalizes phantom braking: a sustained
  speed drop below the free-cruise reference drives it toward 0. Holding speed → 1. N/A → 1.

All thresholds are named constants (`TLC_CREEP_SPEED`, `TLC_MARGIN_COMFORT_M`, `TLC_MARGIN_SCALE_M`,
`TLC_COMFORT_DECEL`, `TLC_DECEL_K`, `TLC_REF_FRAC`, `TLC_PHANTOM_DEADBAND`, `TLC_PHANTOM_DROP_FRAC`) —
no magic numbers in the body. Direction-of-goodness is carried in the return dict so a report cannot
invert it. TLC covers all four behaviors Sayed named: **stops before the line on red, no entry on red,
smoothness of the stop, no phantom-braking on green.**

### Synthetic-MEASURED design-oracle result (P8: NOT our model)
Archetypal-policy telemetry (a *design oracle* encoding what the scenario is for) scored through the
**real** `traffic_light_metrics`. `rule_barrier` = TanitAD H9 hard barrier; `soft_prior` = the
documented failure that treats the signal as a soft cost.

| plan | policy | TLC | entered_on_red | OKRI (lower safer) | phantom_brake |
|---|---|---|---|---|---|
| red | rule_barrier | **1.000** | False | 21.0 | False |
| red | soft_prior | **0.000** | **True** | 57.1 | False |
| green | rule_barrier | **1.000** | False | — | False |
| green | soft_prior | **0.421** | False | — | **True** |

- **Red-run violation rate:** `rule_barrier` **0.0** / `soft_prior` **1.0** over the cross-clearance
  sweep {0…12} m (bar for a rule barrier is exactly 0).
- **Barrier invariant to the temptation; soft prior is not:** soft-prior line-crossing speed grows
  **2.4 → 9.0 m/s** as the apparent cross-clearance opens 0 → 12 m, while the barrier never enters on
  red at either extreme — the mechanistic barrier-vs-soft-prior signature (same as SC-04).

Evidence class: **MEASURED (synthetic fixtures)** — artifact `traffic_light_oracle_results.json`
+ `stack/tests/`. This is the metric math proven on analytically-known inputs. It is **not** a claim
about a TanitAD checkpoint (that is P2, below).

`pytest -q` for the two new files: **24 passed**. Full-suite status: see final report.

---

## P2 — Beyond-ADE suite on real telemetry  [PARTIAL — real TMS/CNCE MEASURED; TLC/LAL/OKRI/LOPS renderer-gated]

### Renderer probe (absence confirmed at THREE locations, per the operating standard)
The closed-loop occluder / work-zone / traffic-light **scenario source** needs a rendered
occlusion/signal geometry (MetaDrive/CARLA). **MetaDrive is genuinely absent on the dev box:**
not in `venvs/tanitad`, not in `venvs/carla312`, not in pip metadata; `metadrive_frontcam.py`
lazily imports it and its live-rollout test skips when absent. So **LAL / OKRI / LOPS / TLC on a
real closed-loop rollout are renderer-gated here** — no fabricated number.

### What DID run on real telemetry we already have  [MEASURED]
Real path: the local **comma2k19 val cache** `C:/Users/Admin/tanitad-data/eval/comma2k19-val-61c46fca8f7f`
(90 episodes, real ego kinematics — speed 27–30 m/s highway, real CAN accel/steer) + the **real
base250cam WorldModel architecture** (262.84 M params, 0.2628 B) timed on the dev-box **RTX 4060**
(NOT a training pod; `tanitad-eval`/pod2/pod3 untouched). Script + JSON in this folder
(`real_telemetry_tms_cnce.py`, `real_tms_cnce.json`), reusing the accepted
`stack/scripts/latency_cnce_baseline.py` method; no trained checkpoint needed (latency + params are
weight-independent).

| number | value | evidence class |
|---|---|---|
| decision-tick latency p50 (encode + K9 select) | **14.33 ms** (9.27 + 5.06) on RTX 4060 | MEASURED |
| active params | **0.2628 B** (262.84 M) | MEASURED |
| **TMS** — expert-log smoothness (reference band, 30 eps) | median **0.0435** (min 0.024 / max 0.131) | MEASURED |
| **CNCE** — architecture efficacy (30 eps) | median **210,551** (min 768 / max 257,125) | MEASURED |

Honesty: **TMS scores the human EXPERT LOG**, a reference band for later closed-loop comparison — it
is **not** a claim about our policy (P8). **CNCE** is a real *architecture* efficiency number
(D_progress from the real log, measured decision-tick latency + real param count, collisions=0 by
log-replay construction; latency/params are weight-value-independent, so identical for trained or
random init). These are the **first real beyond-ADE numbers** for the suite outside the 2026-07-08
scripted-archetype CARLA work-zone build.

### Precisely what is still gated, and the cheapest unblock
- **TLC (the new traffic-light metric) on real data** needs per-step **signal-state** telemetry on a
  real intersection approach. Cheapest unblock, in order:
  1. `pip install metadrive` into the dev-box venv + a **signalized-junction** build (MetaDrive has
     native traffic-light phase control) → run `traffic_light.simulate_policy`'s real replacement
     (a checkpoint-driven ego) closed-loop and log the signal phase into this contract. This is the
     same CARLA-on-pod plan the SCENARIO_DATABASE already schedules (W31-32) but MetaDrive is the
     lighter dev-box path.
  2. Label a set of **comma2k19 intersection segments** with signal state (INFER-quality, per SC-14's
     data-sources note) → real ego kinematics + a labeled `signal_state` array gives a real-ish TLC
     without a renderer. (DataEng handoff.)
- **LAL / OKRI / LOPS** need rendered occlusion geometry (blind-spot distance, hidden-agent GT) —
  same renderer unblock as SC-01/SC-02.

---

## P3 — Wire + document  [DONE]

- **Scenario registry** `stack/tanitad/eval/scenarios/registry.py` — the single seam a runner
  iterates. Registers `work_zone_phantom`, `traffic_light_red`, `traffic_light_green`, each with its
  scorer (`run_scenario_suite` vs `traffic_light_metrics`). `run_registered_suite()` scores every
  registered scenario's archetypal policies. Adding a scenario = adding one `ScenarioEntry`.
- **Runner** `stack/scripts/scenario_suite_dryrun.py` now picks up the traffic light via the registry
  (additive; the work-zone flow and the `telemetry_from_oracle` export are unchanged). `all_pass`
  now includes the three TLC checks.
- **Wiring guard** `stack/tests/test_scenario_suite_wiring.py` extended: the registry lists all three
  scenarios, `run_registered_suite` scores each end-to-end, and TLC discriminates through the
  registry (red barrier 1.0 / soft 0.0; green barrier > soft; OKRI barrier < soft on red).

### `pytest -q` (full stack suite)
**836 passed, 2 skipped** (the 2 skips are pre-existing `@slow`/renderer-gated tests, not mine),
71 s. The 24 new tests (8 TLC-metric + 16 scenario) + 5 wiring tests are all green.

### Integration to escalate (do NOT let this sit in a README)
1. **SC-14 status** in `Opponent Analyzer/SCENARIO_DATABASE.md` should advance
   **catalogued → oracle-tested** (that file is Opponent-Analyzer-owned; not edited here). The
   Benchmarks-&-Eval handoff it names ("add a `violation_rate` reducer") is **delivered**: TLC +
   `red_run_rate` (bar 0.0) live in the stack suite.
2. **LEADERBOARD.md** SC-14 excellence row: gated on a real TLC number (renderer unblock above).

---

## DELIVERABLE MANIFEST (all staged in the repo working tree; nothing pod-only)

| # | artifact | repo path | evidence |
|---|---|---|---|
| 1 | Traffic-light scenario (SC-14) | `stack/tanitad/eval/scenarios/traffic_light.py` | new |
| 2 | TLC metric + `SIGNAL_*` + assembly (additive) | `stack/tanitad/eval/metrics.py` | modified, additive only |
| 3 | Scenario registry | `stack/tanitad/eval/scenarios/registry.py` | new |
| 4 | TLC metric tests (8 analytic) | `stack/tests/test_tlc_metric.py` | new |
| 5 | Scenario tests (16) | `stack/tests/test_traffic_light_scenario.py` | new |
| 6 | Wiring guard (extended) | `stack/tests/test_scenario_suite_wiring.py` | modified |
| 7 | Runner picks up traffic light | `stack/scripts/scenario_suite_dryrun.py` | modified, additive |
| 8 | Design-oracle results (synthetic) | `…/incoming/2026-07-24-traffic-light-scenario-metric/traffic_light_oracle_results.json` | MEASURED (synthetic) |
| 9 | Real TMS/CNCE script | `…/incoming/2026-07-24-traffic-light-scenario-metric/real_telemetry_tms_cnce.py` | new |
| 10 | Real TMS/CNCE result | `…/incoming/2026-07-24-traffic-light-scenario-metric/real_tms_cnce.json` | MEASURED (real) |
| 11 | This note | `…/incoming/2026-07-24-traffic-light-scenario-metric/NOTE.md` | — |

Staged, never committed/pushed (agent operating standard). Full suite green.
