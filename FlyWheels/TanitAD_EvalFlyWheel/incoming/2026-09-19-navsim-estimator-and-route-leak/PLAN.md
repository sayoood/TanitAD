# PLAN — E3: settle `navsim.estimator_unit` and `navsim.route_leak_check`

**Stream** E3 of the EvalFlyWheel session 2026-09-19 · **Brief** `BRIEF.md` (this folder) ·
**Schema** `Project Steering/TANITAD_PROGRAMME.md` §3.
**Constraints** CPU only (GPU = live training, PIDs 16996/21724, never touched) · no dataset download ·
RAM ≤ ~5 GB and system free RAM ≥ ~3 GB (siblings E1/E2 share the box) · stage exact paths, never
commit/push · evidence class on every claim.

## Priority order (a PRIORITY ORDER, not a dependency chain)

| # | step | output | blocks |
|---|---|---|---|
| A1 | **Census** of every NavSim split from the pinned devkit YAMLs + the local OpenScene test metadata: tokens per stage, `log_name`s, nuPlan drives, cities, scenes/log (min/median/max), stage-2 → stage-1 → log mapping. Every count paired with a READ CONTROL. Scene lists reconstructed TWICE (the devkit's own `filter_scenes` and an independent re-derivation) and required to agree. | `code/split_census.py`, `raw/split_census.json`, `raw/split_census_stdout.txt` | A2 |
| A2 | **Overlap measurement**: frames shared by scenes within a log; EC two-frame adjacency; cross-log (same-drive segments) frame/time disjointness; do mapping keys / synthetic scenes ever span two logs. | inside `raw/split_census.json` (`overlap` blocks) | A3 |
| A3 | **SPEC.md** — the pre-registered estimator, written from A1/A2 **before any interval exists**: unit, floor (RG-14), paired form, B = 2000, aggregation reproduced verbatim (two-stage mapping-key mean / single-stage token mean), NaN semantics, refusal conditions, the question the interval answers (and the two it does not). | `SPEC.md` | A4 |
| A4 | **Implement** `taniteval/adapters/navsim_ci.py` (reusing `taniteval/ci.py`), wire an OPTIONAL interval path into `taniteval/adapters/navsim.py` (every existing signature kept). Cross-check the two-stage aggregation against the **devkit's own** `calculate_individual_mapping_scores` run in the NavSim venv (an independently authored reference, banked as literals). | code + `code/devkit_aggregation_crosscheck.py` + `raw/devkit_aggregation_reference.json` | A5 |
| A5 | **Tests**: literal analytic targets (zero-width; two-cluster atoms/mean; hand-computed per-key contribution), the devkit-reference literals, the scene-token MUTATION (must go RED, narrower interval), the floor refusal, the paired form, backward-compat of the adapter API. | `taniteval/tests/test_navsim_ci.py` (+ updates to `taniteval/tests/test_navsim_adapter.py` where it pinned the old "unsettled" wording) | A6 |
| A6 | **Registry + checker**: gate UNRESOLVED → settled (version bump + changelog); `tools/criteria_check.py` evaluates the NavSim benchmark block (it evaluated NONE of it before — measured); log-cluster interval with n ≥ floor PASSES; scene-token / episode-cluster / n < floor FAIL; honest refusal = work item. Deliberate-regression arms in `tools/tests/test_criteria_check.py`. | registry, checker, tests | A7 |
| A7 | Run `pytest -q tools/tests` and the taniteval suites touched; read every skip reason. | `raw/pytest_*.txt` | — |
| B1 | **Leak, source probe (i)** — devkit @0a380a9: where `driving_command` is read/created. | RESULT §B | — |
| B2 | **Leak, source probe (ii)** — OpenScene conversion code at a pinned SHA (who computes the command, from what) + nuPlan route provenance (nuplan-devkit schema + scenario code). | `raw/source_probes/` (fetched raw files + sha256) | — |
| B3 | **Leak, empirical probe (iii)** — on the local test metadata + maps: (a) re-compute the stored command from (pose, route, map) with OpenScene's own function; (b) route vs the roadblocks the logged ego actually traverses; (c) the agreement matrix command × future ego path at several horizons, with n; (d) the counterfactual: command recomputed from an ego-FUTURE-derived route. | `code/route_leak_probe.py`, `raw/route_leak_*.json` | B4 |
| B4 | Verdict (LEAK / NO LEAK / PARTIAL), machine-readable, consumed by `navsim.py::route_leak_check` (API kept). | `raw/route_leak_verdict.json`, adapter | — |
| C | RESULT.md (findings first, proposed register rows verbatim), COMMS.md, stage exact paths, verify blobs at end of turn. | — | — |

## Resource plan

Census: one NavSim-venv process, pickles streamed one log at a time (1.1 GB of metadata on disk).
Leak probe: one process, ONE CITY MAP loaded at a time and released before the next; free RAM
checked before each city; abort a city rather than push system free RAM under ~3 GB.
