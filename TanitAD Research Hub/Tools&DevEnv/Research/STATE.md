# STATE — Tools&DevEnv

LAST_RUN: 2026-07-15 (weekly run; worktree branch `worktree-tools-devenv-20260715`)
QUALITY: full (G-A…G-F + G-H + G-T1 met; measured experiment = CI commit-gate wall-clock + falsifier)

## HANDOFF
Shipped the top backlog item (**CI commit gate**, P0 #1). No half-done work; suite green (363/2s).

**Ready for orchestrator merge (tooling → `main`, not intake):** commit on
`worktree-tools-devenv-20260715` adds `stack/scripts/ci_check.py` + `stack/scripts/ci.ps1` +
`stack/tests/test_ci.py`. Per D-029 dev-tooling exception (same convention as resim/replay/scena),
committed into `stack/` directly, not via intake — fast-forward to `main` is safe (3 new files, no
model code touched, suite green).

**Hand-offs to other agents:**
- → **Benchmarks & Eval / Opponent (D-028 seam):** two July-2026 closed-loop releases are your
  territory, flagged not deep-dived — **CLEAR** (Closed-Loop RL at Scale, arXiv 2607.02841) and
  **Fail2Drive** (closed-loop generalization benchmark, arXiv 2604.08535).
- → **Whoever owns the loop / D-025 program-report:** adopt `pwsh stack/scripts/ci.ps1 -Quick`
  (8.3 s) as the pre-commit / pre-report screen — it names the failing invariant instead of a raw
  pytest dump.
- → **CARLA closed-loop (my own P1.3):** mirror Bench2Drive's 220-route disentangled structure
  (5 × 44 scenarios × weather × location, ~150 m), not one monolithic drive.

**Still pending Sayed (carried from W3, ~1 click each):**
1. Pin `stack/` to Drive **"Available offline"** → removes the ~30 s cold-I/O tax that now compounds
   on top of every `ci.ps1` run (backlog P1.5 verifies once pinned).
2. Graphics-capable pod recreation → unblocks CARLA camera pixels (P1.3). NOT urgent.

## Done this run
- **`ci.ps1` CI commit gate (backlog P0 #1, DONE):** I2 tripwire (fail-fast) → per-test latency
  budget (>6 s `call` → block) → suite-green + opt-in warm-wall budget. Unit-tested core
  `ci_check.py` (12 tests, 0.20 s). Measured: **quick gate 8.3 s** / **full gate 24.4 s** (363
  tests). Falsifier CONFIRMED (7 s test → full gate exit 1). `ci.ps1` verified end-to-end on Windows
  PowerShell 5.1 (auto-resolves venv). 0 new deps → G-T1 GO.
- Reframed the stale "<15 s full suite" target: suite grew 181→351 tests (breadth, not slow tests);
  split into fast pre-commit quick gate + full gate. Honest metric-moved note (P8).
- Research note + KB delta (3 findings) + tooling sweep (AlpaGym/Bench2Drive currency, verdicts
  unchanged) + BACKLOG re-prioritized + created **RESIM_ROADMAP.md** (was a mission-named but missing
  artifact).

## Open threads / proposals to raise
- **`ci_check.py --self-test` mode** (next-run candidate): monkeypatch a batch-stat layer into a
  throwaway module and assert CI exits 2 — closes the I2-red-live falsifier without touching real
  model code. Currently I2-red is unit-tested only.
- Turn on the warm-wall budget in the loop's invocation (`-WarmBudgetS 35`) once the full suite
  crosses ~30 s, so "the suite is getting slow" is loud, not discovered.
- AlpaGym closed-loop RL post-training with our own <100 M driver — A100-gated Phase-1 proposal
  (draft to `Project Steering/Proposals/` once D1–D3 pass). Verdict unchanged: efficiency/labels
  edge, not scale (P5/C2).
- TanitResim P1 / TanitScena P2 continuous products: not touched this run (chose the highest-leverage
  cross-agent item, the CI gate). Next run: pick a measured real gap from `RESIM_ROADMAP.md`
  (3-arm view now REF-B has landed, or checkpoint A/B diff).
