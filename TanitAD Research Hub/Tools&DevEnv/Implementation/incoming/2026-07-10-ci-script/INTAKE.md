# INTAKE — stack CI gate `ci.ps1` (I2 tripwire + full suite + timing budget)

- **Package:** `Tools&DevEnv/Implementation/incoming/2026-07-10-ci-script/`
- **Author agent / date:** tools-devenv-agent, 2026-07-10
- **Proposed target:** `stack/scripts/ci.ps1` (dev tooling; sibling of `profile_testsuite.py`)
- **Hypothesis / WP served:** none directly — CI infra (agent-file duty #3, backlog P0.1); enforces
  the I2 instrument doctrine (D-004) and the G-E timing budget on every commit.

## What & why (≤10 lines)

One command a scheduled agent or a git pre-commit hook runs before pushing. It (1) runs the **I2
collapse tripwire** first as a ~1.5 s fail-fast — `test_i2_batch_consistency_of_encoder`, the
BatchNorm-in-inference canary that guards the single most expensive silent-bug class (D-004) — then
(2) runs the **full pytest suite + timing-budget guard** by delegating to `profile_testsuite.py check`
(warm-overhead ≤ `-MaxWarmOverhead`, no single `call` > `-MaxTest`). If the profiler is not on disk it
**falls back** to an inline `pytest --durations` run enforcing the same budgets, so `ci.ps1` is
self-sufficient. Distinct exit codes let a hook branch: `0` green / `2` I2 tripwire / `1` suite or
budget breach. Python is resolved from `$env:VIRTUAL_ENV` → dev venv → PATH; `StackDir` defaults to
the script's parent (i.e. `stack/`). Pure-ASCII (PS 5.1 reads BOM-less `.ps1` as ANSI).

## Evidence & tests (measured 2026-07-10, dev machine, venv off-Drive / stack on Drive)

- **Green path:** `ci.ps1 -StackDir stack` → I2 `1 passed in 1.57s`; full suite
  `189 passed, overhead 1.112s, wall 8.712s`; **total 11.2 s warm, exit 0** (< 15 s goal).
- **Falsifier (backlog requirement):** injected `tests/test_ci_falsifier_slow` with `time.sleep(7.0)`
  → gate printed `slow test ... 7.0s > 6.0s` and **exited 1** (total 18.1 s). Temp test removed; tree
  clean. This proves a newly-added slow fixture cannot pass CI unnoticed.
- **I2 exit-2 path:** documented (not demonstrated — would require breaking the encoder); the tripwire
  step returns exit `2` distinctly from suite failures so a hook can special-case a collapse regression.
- Dependency: co-locate with `profile_testsuite.py` (`2026-07-09-testsuite-io-profiling/`, still pending
  triage). If integrated together, `ci.ps1` finds it at `stack/scripts/profile_testsuite.py`; if the
  profiler is deferred, `ci.ps1`'s inline fallback still enforces the same budgets (no hard dependency).

## Risk & rollback

- Blast radius: additive dev script; imports nothing from the stack; changes no runtime code. `ci.ps1`
  is opt-in (invoked by an agent/hook, not by `pytest` itself).
- Rollback: delete the script; nothing depends on it.
- **G-T1 verdict: GO** — setup cost 0 min (no deps beyond the already-required pytest); measured warm
  11.2 s < 15 s; falsifier verified. Recommend wiring into a repo `.git/hooks/pre-commit` (or the
  agents' session-end ritual) as `pwsh stack/scripts/ci.ps1` once integrated.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate / integrate-with-changes / defer / reject
- **Date / by:**
- **Reason & notes:**
- **Integrated as:**
