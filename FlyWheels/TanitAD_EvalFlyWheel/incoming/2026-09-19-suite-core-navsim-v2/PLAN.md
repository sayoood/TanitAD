# PLAN — W1 (suite core + NAVSIM v2), priority order, budget, deviations

Priority order (a killed agent still leaves value at every cut):

| # | step | state |
|---|---|---|
| 1 | SPEC pre-registered (blob `fa510ef4…`, `raw/SPEC_PREREG_HASH.txt`) before any product code | ✅ |
| 2 | **the schema** (`bench/schema/*.schema.json` + `schema_check.py`) — W3/W4/W5/W6 build against it; sent to the orchestrator at 15:5x | ✅ |
| 3 | the legacy-collision shim (`bench/_legacy.py`) — the package must not break `taniteval.bench`'s importers | ✅ 81 = 81 |
| 4 | the contract core (`contract.py`, `cli.py`, plugin slots for W3/W6) | ✅ |
| 5 | NAVSIM v2 end to end by PROMOTING E1/E2 (`bench/navsim/**`, `devkit_side/**`) | ✅ |
| 6 | STOP + CV floors by rule, ECHO with a ckpt; HUMAN refused on two-stage | ✅ |
| 7 | the GPU-gap launcher (mock-tested; UNVERIFIED on hardware by design) | ✅ |
| 8 | the submission refusal | ✅ |
| 9 | `internal_t1` wrapping `refcv3_arm.py` | ✅ |
| 10 | tests: literals + a RED mutation per binding rule; the OFFLINE end-to-end twin on E1/E2's banked CSVs | ✅ |
| 11 | **the live acceptance** (`code/run_acceptance.sh`, RAM-gated, clean shell) | ⏳ |
| 12 | RESULT.md + manifest + staging | ⏳ |

## Compute

CPU only. Unit tests + the offline twin run anytime (seconds). The acceptance is HEAVY (E2 measured
CV 533 s + STOP 373 s) and was gated on **A8's done-marker** (present since ~21:0x on 2026-09-19)
**and** a RAM window ≥ 4,500 MB — this box also runs other agents' suites and a CPU refcv7 smoke,
and the promoted wrapper aborts below a 3,000 MB floor. ONE worker (`worker=sequential`).
NavSim outputs go to `C:/Users/Admin/navsim-crun/exp/tanitad_bench/` (C:), never to the D: junction.

## Deviations from the brief, and why

1. **`taniteval/taniteval/bench.py` already existed.** The brief's `taniteval/taniteval/bench/**`
   shadows it (CPython prefers a package). Rather than renaming another owner's module, the package
   delegates every legacy attribute to `bench/_legacy.py`, which executes that file unchanged, and
   `__main__` routes `--model/--all` to its CLI. Evidence: the five pre-existing importer test files
   read **81 passed** before and after (`raw/legacy_importers_{BEFORE,AFTER}.txt`).
2. **Report hook target renamed** to `taniteval.benchreport` (orchestrator ruling 2026-09-19 (b)) —
   W5's package no longer collides with the legacy `report.py`.
3. **Plugin slots** `bench/plugins/{navsim_v1,nuscenes_ol}.py` (orchestrator ruling (a)). W1 shipped
   docstring-only stubs stating the `ctx` contract; **W3 and W6 have since replaced both**.
4. **One promoted file carries a W1 ADDITION**: `devkit_side/navsim_win.py` gains
   `--dump-preaggregation` (the `worker_map` seam). Reason, MEASURED 2026-09-19: the navhard CV run
   scored all 5,912 scenarios in 68 min, then the aggregation raised and the runner wrote NO CSV —
   and because that run used a PROCESS POOL, the wrapper's in-process `pdm_score` hook recorded
   `pdm_calls=0`, so nothing survived. `worker_map` returns in the PARENT for every worker type.
   Stripping the marked block reproduces E1's file byte-for-byte (pinned).
5. **navhard** is wired (profile + the waiter's cache at `navsim-crun/exp/metric_cache_navhard_two_stage`)
   but NOT run here: E1 owns the aggregation-crash diagnosis. The suite's behaviour for that class is
   implemented and tested (`AGGREGATION_FAILED`, pre-aggregation dump, `reaggregate` subcommand).
