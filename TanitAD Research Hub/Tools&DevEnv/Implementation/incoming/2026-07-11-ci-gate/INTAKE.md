# INTAKE — commit gate (ci.ps1): I2 tripwire + pytest + timing guard

- **Package:** `Tools&DevEnv/Implementation/incoming/2026-07-11-ci-gate/`
- **Author agent / date:** tools-devenv-agent, 2026-07-11
- **Proposed target:** `stack/scripts/ci.ps1` + `stack/scripts/ci_i2_tripwire.py`
  (dev tooling; sibling of `profile_testsuite.py` from the 2026-07-09 intake)
- **Hypothesis / WP served:** none directly — serves **G-E** (every agent's pytest
  gate) + **D-004/I2** (batch-1 deployment invariant) + backlog #3 (CI script)

## What & why (≤10 lines)

One command an agent or pre-commit hook runs before committing to `stack/`. Two
fail-fast gates: **(1) I2 tripwire** (`ci_i2_tripwire.py`) — builds the real
`WorldModel(smoke_config())` and asserts `encode(batch=1) == encode(batch=B)` to
1e-4; deployment is batch-1 streaming on Orin, so a batch-statistic layer is silent
in training but breaks the exported TensorRT engine. Runs FIRST (~2 s) so the whole
BatchNorm bug class fails before the slow suite. **(2) Suite + timing guard** — the
full pytest suite (G-E) run *through* `profile_testsuite.py check`, which fails the
commit on any test failure, warm import/collection overhead > budget, or any single
`call` test > budget (catches a newly-added slow fixture). Nonzero exit if either
gate fails; total wall reported. Research note:
`2026-07-11-ci-gate-and-tensorrt-orin-qdq-trap.md` §1.

## Evidence & tests

- Package tests: `tests/test_ci_i2_tripwire.py` — **3 passed in 2.07 s** (venv
  `tanitad`, py3.13). Adversarial: a synthetic batch-mean-subtracting encoder MUST
  fail the tripwire (falsifier), a per-frame linear encoder MUST pass, and the real
  WorldModel encoder passes. Standalone via a `conftest.py` that finds `stack/`.
- End-to-end (RTX-4060 host, 2026-07-11, dev machine): `ci.ps1` →
  `CI gate PASS - total 17.2s (I2 2.4s + suite 14.8s)`, exit 0; I2 dev **1.74e-07**;
  suite **189 passed**, warm overhead **1.429 s**, wall 14.689 s.
- **Gate falsifier (measured):** injected a 7.0 s test into `stack/tests/` →
  `ci.ps1` → `CI gate FAIL`, exit **1**, timing guard flagged
  `test_deliberately_slow_fixture 7.0s > 6.0s`. Temp test removed. The stated
  falsifier ("a newly-added slow fixture must make ci.ps1 exit nonzero") holds.

## Risk & rollback

- Blast radius: additive dev scripts; no runtime/stack imports touched. `ci.ps1`
  auto-detects `stack/`, prefers the off-Drive `tanitad` venv, and finds
  `profile_testsuite.py` at `stack/scripts/` (once integrated) or the 2026-07-09
  intake folder, else falls back to plain `pytest -q` with a warning.
- **Dependency:** the timing guard needs `profile_testsuite.py` (pending 2026-07-09
  intake). If that package is rejected, `ci.ps1` still runs the I2 tripwire + plain
  pytest (timing guard skipped) — no hard failure.
- Encoding note: `ci.ps1` is pure ASCII — Windows PowerShell 5.1 reads no-BOM `.ps1`
  as the system codepage, so non-ASCII punctuation breaks parsing. Keep it ASCII.
- Rollback: delete the two scripts; nothing imports them.
- **G-T1 verdict: GO** — setup cost 0 min (0 new deps; pwsh/powershell already
  present on the dev box + pods), measured warm wall 17.2 s, catches two real bug
  classes (batch-statistic inference + slow-fixture creep) at commit time.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate / integrate-with-changes / defer / reject
- **Date / by:**
- **Reason & notes:**
- **Integrated as:**
