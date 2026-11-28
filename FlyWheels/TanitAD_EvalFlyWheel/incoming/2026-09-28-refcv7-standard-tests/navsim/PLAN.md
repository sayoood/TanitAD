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

## Status 2026-10-04 ~01:30 Berlin (EvalFlyWheel NavSim operator)

| # | step | artifact | status |
|---|---|---|---|
| 7 | step 5,000 milestone | `raw/milestones/step5000/` | ✅ DONE 2026-09-28 20:53 Berlin — BAR-R7-N1 **PASS**, NW1 FAILED, NH1 FAILED (RESULT §5) |
| 8 | step 30,000 milestone | `raw/milestones/step30000/` | ⏳ runner killed 2026-10-02 (USB resets 12:46–12:50 killed the navtest bridge; a host restart 16:55 killed the runner — RESULT §6.1). Completion RUNNING: `code/complete_milestone7.py`, log `complete.log`. warmup final (NW1 NOT PROVEN); navhard needs PRIOR_ha0p (scoring); navtest needs a ~10–14 h CPU bridge (1,072 + 12,146 rows) then 4 scorers |
| 9 | step 50,400 FINAL | `raw/milestones/step50400/` | ⏳ RUNNING (`run_navsim_refcv7.py`, `--gpu-wait-s 43200`); warmup on CPU fp32 (my cwd error, COMMS 6), navtest + navhard on CUDA through the lock |
| 10 | RESULT §6 navtest/navhard numbers + 5k→30k comparisons; RESULT §7 numbers | — | gated on 8 / 9 (the driver writes `BARS.json` and `compare_vs_step5000_*.json` itself) |
| 11 | runner hardening: log each bridge's exit code; mirror `runner.log` to local disk; K0/KD read independent of pytest's rootdir | — | not started (the rootdir half is covered by `pytest.ini`) |
