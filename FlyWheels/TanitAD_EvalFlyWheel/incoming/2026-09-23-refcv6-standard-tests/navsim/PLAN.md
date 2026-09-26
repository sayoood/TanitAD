# PLAN — refcv6 NavSim suite (priority order, not a dependency chain)

Status 2026-09-26 ~08:35Z (resumed after the weekly-limit reset; Master Mind's priority order).

| # | step | artifact | status |
|---|---|---|---|
| 1 | Harness reads its known values, every split, through this package's drivers | `raw/HARNESS_REPRO.json`, `raw/controls/KH_ECHO_warmup.json`, `raw/floors/navhard_two_stage/FLOORS_PROVENANCE.json`, `raw/controls/KH_navtest_STOP.json` | ✅ warmup CV/STOP/ECHO, navhard CV, navtest STOP — all max \|Δ\| 0.0 |
| 2 | refcv6's inputs as tested functions (+ SPEC §12 oracle) | `code/`, `tests/` | ✅ 68 tests + KL |
| 3 | SPEC + amendments A1–A4 staged before the scores they touch | `SPEC.md`, `raw/SPEC_PREREG_HASH.txt` | ✅ |
| 4 | step 1,000 pipeline validation: warmup (6 arms), navhard (5,912), navtest sub200 + diagnostics | `raw/*_s1000*`, `raw/navtest_s1000_sub200/` | ✅ reviewed |
| 5 | **step 5,000 RESULT**: navhard (2 arms, CUDA), navtest (full, CPU) + diagnostics with seed replicate | `raw/milestones/step5000*/` | ✅ reviewed + banked (RESULT §3.3); BAR-R6-H1 and BAR-R6-T1 FAILED |
| 5b | warmup @ 5,000 (left empty by the unattended run) | `raw/milestones/step5000/` | ⏳ re-running (CPU, `code/campaign_0926.sh`) |
| 6 | **step 30,000** — warmup → navtest (+diag) → navhard | `raw/milestones/step30000*/` | ⏳ queued after 5b in the results lane |
| 7 | **FINAL** checkpoint (run ends ≈ 2026-09-27 17:30Z) | `raw/milestones/step<final>*/` | ⏳ ARMED: `milestone_waiter.sh final` |
| 8 | precision floor KP (registered) + KP-navtest (diagnostic) | `raw/controls/KP_step1000/`, `raw/controls/KP_navtest_step5000/` | ⏳ waits for the GPU gate (`code/kp_lane.sh`) |
| 9 | Landing: copy to D: + `LANDING_READY.txt` + message the Master Mind per batch (no git) | `LANDING_READY.txt` | ongoing |
