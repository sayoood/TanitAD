# Tools&DevEnv — CI commit gate + per-test latency budget (backlog P0 #1)

**Date:** 2026-07-15 (Tools&DevEnv weekly run; STATE LAST_RUN was 2026-07-09 W3)
**Author:** Tools&DevEnv agent (autonomous scheduled run)
**Worktree:** `worktree-tools-devenv-20260715` (D-026 isolation)
**Hardware:** local dev box (RTX 4060, CPU-only tests) · wall-clock ≈ 6 min · cost **$0**
**QUALITY:** full (G-A…G-F + G-H + G-T1 met)

## TL;DR

Shipped the top backlog item: **`ci.ps1` — a one-command commit gate** every agent, the
loop, and pre-commit hooks can run. It is three gates, most-load-bearing first: (1) the **I2
tripwire** (encoder batch-1 consistency, D-004) run first + alone + fail-fast; (2) a **per-test
latency budget** (no single `call` phase > 6 s) that stops the suite silently rotting into a slow
mess; (3) **suite-green + optional warm-wall budget**. Logic lives in a unit-tested Python core
(`ci_check.py`, **12 tests, 0.20 s**); `ci.ps1` is a thin venv-resolving wrapper. Measured warm
wall: **quick pre-commit gate 8.3 s**, **full gate 24.4 s** (363 tests). The registered falsifier
fires: a 7 s test makes the full gate **exit 1** with a precise message. Zero new deps.

## Why this, why now

- It is **backlog P0 #1** and the last un-shipped item of my classic remit (CARLA harness + Colab
  burst already shipped by the loop).
- **Leverage:** all six weekly agents + the loop commit many times a day. A single go/no-go command
  that names *why* it blocked beats "run pytest and eyeball it."
- The old target ("full suite < 15 s warm") is **stale**: the suite grew 181 → 351 tests since my
  2026-07-09 run, and warm wall is now ~21 s — driven by *breadth* (351 small tests), not a few slow
  ones (slowest single `call` = 1.97 s). So the design splits into a **fast curated pre-commit gate**
  (the safety subset, 8.3 s) and a **full gate** (24.4 s), rather than chasing an unmeetable single
  number. This is the honest reframe (P8): the metric moved, so the tool adapts.

## What shipped (committed to `stack/`, tooling exception — see "Boundary note")

- `stack/scripts/ci_check.py` — the gate logic. Pure helpers (`parse_durations`,
  `max_call_duration`, `suite_passed`, `reported_wall`, `evaluate`, `QUICK_SUITE`, `I2_NODE`) +
  a thin subprocess `main`. stdlib only (`argparse/re/subprocess/time/dataclasses`).
- `stack/scripts/ci.ps1` — thin PowerShell wrapper: resolves the interpreter (explicit override →
  active `$VIRTUAL_ENV` → known tanitad venv → PATH), `Push-Location stack/`, forwards `-Quick`,
  `-SlowTestS`, `-WarmBudgetS`, propagates the exit code.
- `stack/tests/test_ci.py` — **12 unit tests** driving the pure logic on synthetic pytest output
  (no nested pytest process → fast, deterministic), plus two drift guards (every `QUICK_SUITE`
  file exists; `I2_NODE` names a real test).

### The three gates

| Gate | Mechanism | Falsifier |
|---|---|---|
| **1. I2 tripwire** | run `test_i2_batch_consistency_of_encoder` first + alone; red → exit 2, suite skipped | a BatchNorm/batch-stat layer in the inference path → I2 red → CI blocks in <2 s |
| **2. latency budget** | parse `--durations=0`, take slowest `call`; > `--slow-test-s` (6 s) → exit 1 | **demonstrated below**: a 7 s test blocks the full gate |
| **3. suite + wall** | pass counts + no failures; if `--warm-budget-s`>0, warm wall must be under it | a failing test, or (opt-in) warm wall creeping over a set ceiling |

## Measured results (G-H)

All on the local box, venv `C:\Users\Admin\venvs\tanitad`, warm caches.

| Run | Command | Wall (warm) | Detail | Exit |
|---|---|---|---|---|
| unit tests | `pytest tests/test_ci.py` | **0.20 s** | 12 passed | 0 |
| **quick gate** | `ci.ps1 -Quick` | **8.3 s** | I2 1.46 s + 38 tests 5.10 s; slowest `test_smoke_training` 2.06 s | 0 |
| **full gate** | `ci.ps1` | **24.4 s** | I2 1.55 s + 363 passed/2 skipped 20.88 s; slowest `test_pipeline_end_to_end_cpu` 1.97 s | 0 |
| **falsifier** | full gate + injected `time.sleep(7)` test | — | `BLOCKED: slow test: …::test_deliberately_slow took 7.00s call (budget 6.0s)` | **1** |

- **Falsifier verdict: CONFIRMED.** The per-test budget catches a newly-added slow test and blocks
  the commit with an actionable message. (Note: the *quick* gate collects only the curated file list,
  so it does **not** see an arbitrary new test — the *full* gate is the catch-all. Recorded honestly:
  pre-commit quick is a fast safety screen, the full gate is the completeness screen.)
- **I2 fail-fast:** covered by unit test (`evaluate(i2_passed=False)` → BLOCKED); not exercised
  live because it would require breaking the encoder (model code, out of scope for this run).
- The quick **safety subset** (`QUICK_SUITE`) = instruments (I1–I4), gate runner (BLOCKED≠FAIL +
  I2-missing gating), SigReg anti-collapse, the end-to-end smoke train, and the data-contract
  integrity tests (`mixing`, `epcache_key`, `comma2k19_contract`) — the "did this commit break a
  safety invariant" set, all `call` < 2.2 s.

### G-T1 verdict

**GO.** Setup cost **0 min / 0 new deps** (stdlib + existing venv). Pre-commit quick gate 8.3 s is
well within a "run before every commit" envelope; full gate 24.4 s is a reasonable pre-push / CI
screen. Fits the P5 resource envelope trivially (CPU, $0).

## Boundary note (why `stack/`, not intake)

Backlog P0 #1 originally said "deliver as intake," but that predates **D-029**, which made dev
tooling (TanitResim, replay/scena apps) a **push-to-`stack/`-directly exception** to the intake
rule. `ci.ps1`/`ci_check.py` are structurally identical to the already-in-`stack/` tooling
(`resim_app.py`, `replay_app.py`, `scena_app.py` — all committed directly under `resim:`/`replay:`/
`scena:` prefixes, not via intake). So this follows the established convention: committed into
`stack/` on the isolated worktree branch, suite kept green, pushed for the orchestrator to
fast-forward to `main`. No model code is touched.

## Tooling sweep (currency since 2026-07-09)

- **NVIDIA Alpamayo family** (newsroom, 2026-06-01): AlpaSim (open E2E sim on GitHub) + **AlpaGym**
  (high-throughput **closed-loop RL** framework) + Alpamayo-1 (10 B) / Alpamayo-2 Super (32 B VLA,
  weights "this summer"). **Verdict unchanged (P5): Phase-1 cloud, not Phase-0** — 40–60 GB VRAM,
  Docker/HF-gated. AlpaGym is the reference for our own Phase-1 closed-loop-RL harness over a <100 M
  driver; our edge stays efficiency/labels, not scale (C2). [newsroom](https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development)
- **Bench2Drive** remains the definitive closed-loop CARLA harness: **220 routes = 5 × 44 scenarios ×
  weather × location, ~150 m each**, task-disentangled to cut seed variance. This is the **template
  our CARLA-on-pod closed-loop runner (backlog P1.3) should mirror** — scenario-disentangled short
  routes, not one long drive. [OpenReview](https://openreview.net/forum?id=y09S5rdaWY)
- **Seam hand-off (D-028):** two July-2026 closed-loop releases surfaced that are **Benchmarks&Eval /
  Opponent** territory, not Tools&DevEnv — flagged, not deep-dived: **CLEAR** (Closed-Loop RL at
  Scale, arXiv 2607.02841) and **Fail2Drive** (closed-loop generalization benchmark, arXiv
  2604.08535). Routed in STATE HANDOFF.

## Actionable recommendations

1. **Adopt `ci.ps1 -Quick` as the pre-commit / pre-report screen** for all agents (8.3 s, names the
   failing invariant). Wire into the loop's commit path and the D-025 program-report script.
2. **CARLA-on-pod closed-loop runner should mirror Bench2Drive's 220-route disentangled structure**
   (backlog P1.3) — short scenario-attributed routes, not a monolithic drive. (H1/H11, opponent seam.)
3. Verify the still-pending **Drive "Available offline"** pin (backlog P1.5) — the ~30 s cold-I/O tax
   compounds directly on top of every `ci.ps1` run.

## Falsifiers left open / next step

- **I2-red live path** not exercised (would need to break the encoder). Covered by unit test only —
  acceptable, but a future run could add a `--self-test` mode to `ci_check.py` that monkeypatches a
  batch-stat layer in and asserts CI exits 2, closing this without touching real model code.
- The **warm-wall budget is opt-in (0 = off)** by default. Once the suite crosses ~30 s we should
  turn it on in the loop's invocation (`-WarmBudgetS 35`) so the "suite is getting slow" signal is
  loud, not discovered.
