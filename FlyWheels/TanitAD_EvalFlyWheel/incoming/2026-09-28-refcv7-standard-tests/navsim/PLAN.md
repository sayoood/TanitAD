# PLAN — refcv7 NavSim suite (priority order, not a dependency chain)

Status 2026-09-28 ~07:50 Berlin (EvalFlyWheel NavSim stream; the brief's priority order).

| # | step | artifact | status |
|---|---|---|---|
| 1 | SPEC pre-registration BEFORE any refcv7 NavSim score, sha256 recorded; amendment A1 before any score | `SPEC.md`, `raw/SPEC_PREREG_HASH.txt` | ✅ 05:19 Berlin; A1 05:46 Berlin |
| 2 | harness known values through THIS package's drivers | `raw/HARNESS_REPRO.json`, `raw/controls/KH_ECHO_warmup.json`, `raw/controls/KH_navtest_STOP.json`, `raw/floors/navhard_two_stage/FLOORS_PROVENANCE.json` | warmup CV/STOP/ECHO ✅ \|Δ\| 0.0; navtest STOP ✅ \|Δ\| 0.0; navhard CV ⏳ running since 07:38 (≈ 3.2 h; `code/harness7.py navhard`) |
| 3 | `refcv7_bridge.py` + tests: literal/analytic, KT7 (trainer feed), mutation arms RED | `code/refcv7_bridge.py`, `tests/` | ✅ 43 input + 13 model-seam + 6 mutation tests green; 4/4 mutations RED |
| 4 | pipeline validation on the step-1,500 checkpoint, warmup, VALIDATION ONLY | `raw/validation/step1500/` (CPU, scored), `raw/validation/step1500_cuda/` (CUDA bridges + navtest/navhard smokes) | ✅ |
| 5 | detached event-driven waiter for step 5,000 | `code/milestone_waiter7.py`, `raw/milestones/waiter_5000.json` | ✅ ARMED 06:46 Berlin, Windows PID 33704 |
| 6 | `LANDING_READY.txt` | `LANDING_READY.txt` | ✅ written at hand-over (`code/landing7.py`); the navhard KH and the step-5,000 outputs land in a later batch |
| next | waiters for 15k / 20k / 30k; a FINAL waiter with a done-marker rule | — | not armed (15k lands ≈ 27 h after 5k) |
