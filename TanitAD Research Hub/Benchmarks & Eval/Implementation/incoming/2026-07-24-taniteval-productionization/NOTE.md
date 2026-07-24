# TanitEval productionization — working note (2026-07-24)

Agent: benchmarks-eval. Territory: `taniteval/` only. Staged, never pushed.
Evidence class tags: MEASURED (ours+path) · INHERITED · HYPOTHESIS.

---

## P1 — DE-STRAND: already done by a prior sweep. NO re-strand needed. (MEASURED)

The brief said 9 modules (`bench, closedloop, runner, report, registry, hierarchy,
plan_fan, planner_p2, refc_rerank`) are stranded in worktree
`.claude/worktrees/dazzling-villani-bb4728/` and absent from main. **That is no
longer true** — the main working tree already contains all of them, plus 4 the
worktree lacks (`ci, driving, efficiency, plan_fan_clips`). Verified:

- **Worktree branch is fully contained in mine.** `git log HEAD..claude/dazzling-villani-bb4728`
  is EMPTY; merge-base == worktree HEAD (`08c48af`). My branch
  (`agent/benchmarks-eval-20260721`, `8194807`) is 4 commits ahead.
- **Content diff (CRLF-normalized, the raw line diff was 100% CRLF noise):**
  20/28 shared modules byte-identical; the 8 that differ are all *main-ahead*
  (WT-only lines ≤13, MAIN-only up to 163). Every WT-only line is a SUPERSEDED old
  version — the mislabeled `"8-split episode-disjoint jackknife"` strings that main
  correctly relabeled to `overlapping_holdout_se … DEPRECATED, not a jackknife`
  (the CLAUDE.md CI-estimator correction). Every WT-only *symbol*
  (`tms_openloop, run, draw_bev, episode_planfan, traj_capable, open_grnd, …`)
  still exists in main. Nothing was lost.
- **`tests/` in main is a strict superset** of the worktree's (12 vs 5 files; the
  5 shared are identical-or-main-ahead).
- **Absence-at-one-location guard paid off:** the brief pointed at the *stalest*
  copy. Two OTHER worktrees (`opponent-20260721`, `data-engineering-20260721`)
  have newer-*mtime* taniteval, but their CONTENT diff vs main is byte-for-byte
  the SAME stale snapshot (identical WT/MAIN-only counts) — newer mtime is a
  checkout artifact, not newer content. **Main is the newest+most-complete copy
  everywhere.**

**Decision: do NOT copy the worktree modules into main.** Doing so was the naive
reading of the brief and would REGRESS main — reintroducing the mislabeled
"jackknife" strings CLAUDE.md forbids quoting, and older function signatures.
The de-strand is complete; the worktree is safe to prune.

Main taniteval modules are currently STAGED (index shows `M`/`A` vs HEAD) — i.e.
"in the repo, staged, with provenance" = done per the operating standard. Not my
commit to make.

## P3a — reproducible tests: FIXED. (MEASURED)

`python -m pytest taniteval/tests/` was RED off-pod (collection error): `bench.py`
imports `driving_diagnostic` from `stack/scripts`, and only *some* test files added
that path — `test_bench_diagnostic.py` had pod-only `/root/...` inserts, so one bad
import aborted the whole collection. Added `taniteval/conftest.py` (machine-relative
path bootstrap, derives repo root from its own location → works on dev box AND pod).

Result: **147 passed in 15.32s** on the dev box venv
(`C:/Users/Admin/venvs/tanitad`, py3.13 + torch 2.11). Was: 1 collection error, 0 run.

---

## P3b — leaky-split refusal: DONE. (MEASURED)

`data.list_val_episodes(val_dir, n, allow_leaky=False)` now RAISES on the leaky
`physicalai-val-f1b378f295ae` split (points the caller at `data.CLEAN_VAL`).
It's the single chokepoint all 10 decision-grade callers route through, so the
refusal covers runner/closedloop/hierarchy/pathspeed/efficiency/refc_rerank/
planning/planner_p2/bench/generalization at once. Guard triggers ONLY on that
hash → other corpora (comma/cosmos/OOD) and the clean split pass. `label_overlay`
does NOT route through it (own glob; leaky path only a default for the VLM-label
AUDIT tool) → intentionally unaffected, left as-is. Added `data.CLEAN_VAL`/
`LEAKY_VAL` constants; `runner.VAL` now derives from `CLEAN_VAL` (one source).

## P2 — ONE canonical entrypoint: CONFIRMED + closed-loop wired. (MEASURED)

`runner.py` was already the canonical CLI (run/run-all/ab/driving/hierarchy/
generalize/pathspeed/efficiency/imagination/regression/report) and already pins
the clean split + has the per-model leak guard. GAP: closed-loop existed only as
its own `python -m taniteval.closedloop`. Wired `closedloop` / `closedloop-all` /
`closedloop-report` as runner subcommands (thin dispatch to the existing
`closedloop.run_and_save` / `closedloop_report.main` — no reimplementation).
Verified via `runner --help`: 20 subcommands, closed-loop present, standard axes
intact. Wrote `taniteval/README.md` (install · the one command · outputs table ·
the 3 production invariants).

## P3 (surfaced, not changed) — already correct, documented in README:
- **CI estimator**: `ci.episode_cluster_bootstrap` / `paired_…` IS the decision-
  grade estimator and the `driving` panel already emits it; deprecated
  `overlapping_holdout_se` kept for reproducibility only. (INHERITED from ci.py
  rationale + MEASURED: driving.py:84 imports ci, lines 453/479/551 emit it.)
- **Viz standard**: `corpus_overlay.py` is already THE STANDARD default (camera +
  BEV inset + decoded maneuver + route/goal HUD + ADE). No change needed.

## Tests
`taniteval/conftest.py` (machine-relative path bootstrap) + new
`tests/test_productionization.py` (6 tests: leaky refused / clean passes /
allow_leaky / other-corpora / constants / runner-exposes-closedloop).
**Full suite: 153 passed, ~17 s** on the dev box venv. Was un-runnable off-pod.

## Production-ready vs still research-grade (honest)
- READY: the leaky-split refusal, the pinned clean split + leak guard, the CI
  estimator, the conftest-backed reproducible test suite, the documented one-CLI.
- STILL RESEARCH-GRADE: fresh model rollouts need a GPU + `/root/valdata` cache
  (dev box can only import / unit-test / offline-recompute `driving`); the many
  loose top-level probe scripts (`v2_tier0_probe.py`, `postmortem_*`, `levers_*`,
  `recompute_ci.py`, …) remain one-off analyses, NOT folded into the CLI — that
  was out of scope ("wire what exists, don't re-implement") and they are not
  claimed as canonical.

## DELIVERABLE MANIFEST (all `repo:`, staged, never pushed/committed)
| artifact | path | state |
|---|---|---|
| pytest path bootstrap (NEW) | `taniteval/conftest.py` | staged |
| leaky-split refusal + constants | `taniteval/taniteval/data.py` | staged (my working-tree edit) |
| closed-loop wired into the one CLI | `taniteval/taniteval/runner.py` | staged (my edit atop prior sweep) |
| canonical entrypoint docs (NEW) | `taniteval/README.md` | staged |
| guard + wiring tests (NEW) | `taniteval/tests/test_productionization.py` | staged |
| this note | `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-24-taniteval-productionization/NOTE.md` | staged |

Nothing lives in only one place. No file outside `taniteval/` (+ the incoming
NOTE) was touched — sibling territory (`stack/tanitad/eval/scenarios/`,
`stack/tanitad/resim/`, `stack/scripts/resim_app.py`) untouched.

## ESCALATION (integration)
Nothing blocks. The main taniteval modules (bench/ci/closedloop/driving/… from
the prior sweep) are STAGED but NOT committed on `agent/benchmarks-eval-20260721`
— the orchestrator/Sayed commits. The `dazzling-villani-bb4728` worktree (and the
identical stale snapshots in `opponent-20260721` / `data-engineering-20260721`)
are fully superseded and safe to prune.
