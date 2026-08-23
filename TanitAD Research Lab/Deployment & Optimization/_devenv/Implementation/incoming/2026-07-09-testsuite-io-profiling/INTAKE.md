# INTAKE — test-suite I/O profiler + cold-cost regression guard

- **Package:** `Tools&DevEnv/Implementation/incoming/2026-07-09-testsuite-io-profiling/`
- **Author agent / date:** tools-devenv-agent, 2026-07-09
- **Proposed target:** `stack/scripts/profile_testsuite.py` (dev tooling; sibling of `ci.ps1` when that lands)
- **Hypothesis / WP served:** none directly — serves **G-E cost** (every agent's pytest gate) and CI (backlog #3)

## What & why (≤10 lines)

Measures and guards the cost every scheduled agent pays to satisfy gate G-E (`pytest`).
On the dev machine the venv is off-Drive but the repo/tests/fixtures are on Google
Drive File Stream, so a **cold** first run of the day is **40.6 s** vs **10.7 s warm**
(same 181-passed suite; pytest-reported test time is only 9.2 s). The ~30 s delta is
Drive *hydration latency* (0.44 MB of source, not byte volume; not compute). The tool
(a) `profile` — cold/warm decomposition + slowest-test ranking, JSON out; (b) `check`
— a CI/agent regression guard that exits nonzero if warm overhead or any single test
exceeds a budget (catches a newly-added slow fixture at commit). Stdlib only, 0 deps.
Full measurement + the Drive "Available offline" mitigation are in the research note
`2026-07-09-carla-render-blocker-and-testsuite-io-cost.md`.

## Evidence & tests

- Tests included: `tests/test_profile_testsuite.py` — **9 passed in 0.30 s** (parsers +
  budget logic on canned pytest text; no pytest-in-pytest).
- End-to-end: `python profile_testsuite.py check --stack-dir stack` →
  `TESTSUITE CHECK OK: 181 passed, overhead 1.38s, wall 11.47s`, exit 0.
- Measured decomposition (dev machine, RTX-4060 host, 2026-07-09): cold 40.6 s /
  warm 10.7 s / reported-test 9.2 s / collect-only 4.9 s / import torch 1.9 s /
  warm full-read of stack src 0.13 s (87 files, 0.44 MB). Slowest tests:
  `test_smoke_training` 3.02 s, `test_base250_parameter_budget` 1.09 s.

## Risk & rollback

- Blast radius if integrated: additive dev script; no runtime/stack imports touched.
  `check` is opt-in (only fails when wired into CI with a budget).
- Rollback: delete the script; nothing depends on it.
- G-T1 verdict: **GO** — setup cost 0 min (stdlib), pays for itself as CI's timing
  guard; the actionable payoff (pin `stack/` to Drive "Available offline") is a
  one-click Drive UI change, no code.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** **integrate-with-changes**
- **Date / by:** 2026-08-03 · orchestrator daily sweep (age at adjudication: **25 days**, oldest
  in the queue)
- **Reason & notes:**
  - **Re-verified here, not inherited:** `tests/` re-run on this box → **9 passed / 0.10 s**
    (`venvs/tanitad`). Stdlib only, 0 deps, additive dev script — the risk read holds.
  - The measurement it rests on is still live: the repo is on Google Drive File Stream and the
    venv is off-Drive, so the cold/warm split it profiles is a cost **every scheduled agent**
    pays on its first `pytest` of the day. With six research agents now running **daily** (the
    2026-08-03 rotation change), that cost is paid 6× a day, not once a week — the tool got
    *more* valuable while it sat unadjudicated, not less.
  - **The one change:** the test's `sys.path.insert(..., parents[1])` resolved to the package
    root; under `stack/tests/` that is `stack/`, and the script lives in `stack/scripts/`
    (`ModuleNotFoundError: profile_testsuite` on collection). Repointed to
    `parents[1] / "scripts"`, which is the convention every other stack test uses
    (`test_accum_effective_batch.py:44`, `test_anchor_tactical.py:35`, …).
  - **Not wired into CI in this sweep.** `check` stays opt-in exactly as the package ships it.
    Turning on a budget gate is a separate decision with a separate blast radius (a red CI on a
    slow Drive morning is a false alarm that costs an agent-hour); the tool has to be *in* the
    repo before that can even be argued.
- **Integrated as:**
  - `stack/scripts/profile_testsuite.py`
  - `stack/tests/test_profile_testsuite.py` (9 tests, in the suite that must stay green)
