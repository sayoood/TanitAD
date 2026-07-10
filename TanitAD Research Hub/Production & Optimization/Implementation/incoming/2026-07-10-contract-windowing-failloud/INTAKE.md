# INTAKE — Fail-loud episode windowing (silent short-episode drop → loud error)

- **Package:** `Production & Optimization/Implementation/incoming/2026-07-10-contract-windowing-failloud/`
- **Author agent / date:** Production & Optimization (Saturday), 2026-07-10
- **Proposed target:** shared helper `stack/tanitad/data/_windowing.py` (new), used by
  `stack/tanitad/data/_contract.py::EpisodeWindowDataset` (`:120-121`),
  `stack/tanitad/data/toy_driving.py::ToyDrivingDataset` (`:131-132`), and
  `stack/tanitad/data/comma2k19.py` (`:278-279`). Minimal alternative: inline the
  `build_window_index` guard at each of the three sites.
- **Hypothesis / WP served:** production hardening (P3, ops-fragility class F-5/F-6/F-7);
  protects every training run's data-integrity invariant. Compliance review #3.

## What & why (≤10 lines)

All three window datasets build their index with `t_max = T - window - max_horizon;
index.extend((e_i, t) for t in range(t_max))`. When an episode has `T < window +
max_horizon + 1`, `t_max ≤ 0`, `range` is empty, and the episode is **silently dropped**
— no counter, no warning, no log (`comma2k19.py` even wraps `max(0, t_max)`, guarding the
negative-range but keeping the drop just as silent). Consequences, both in this program's
ops history: (1) a data/config change (bigger `window`/`max_horizon`, or a corpus of
shorter clips) silently shrinks and long-episode-biases the train set with **zero signal**;
(2) if *every* episode is too short the dataset is `len == 0` and the trainer spins on
`StopIteration` (or `drop_last` yields nothing) — a no-progress hang whose message hides the
real cause. Fix: count drops, **warn** when any episode is dropped, and **raise a clear
`ValueError`** if the index would be empty — naming `window`, `max_horizon`, the required
minimum length, and the lengths actually seen. Non-dropped behaviour is byte-for-byte
identical (the conservative `range(T-window-max_horizon)` convention is preserved for parity
across all three datasets). See research note `Research/2026-07-10-int8-quant-curve-and-windowing-failloud.md` §2.

## Evidence & tests

- Tests included: `tests/test_windowing_failloud.py` — **10 passed, 2 warnings** on the author
  machine (venv `tanitad`, torch 2.11, standalone — no `tanitad` install, no CUDA, no real data).
  The 2 warnings are the intended fail-loud `UserWarning`s exercised by the drop-path tests.
- Coverage: window-count parity vs the current convention; boundary `T == window+max_horizon+1`
  kept (exactly one window); one-below-boundary dropped-and-warned (with a surviving episode);
  all-too-short → `ValueError("EMPTY")` naming the numbers; happy path warning-free; `__getitem__`
  returns the byte-identical contract dict (shapes/keys); uint8→float[0,1]; invalid `window` guard.
- Truthfulness (G-P1): the defect names real file:lines — `_contract.py:120`, `toy_driving.py:131`,
  `comma2k19.py:278` (grep-confirmed this run). This is a fail-loud/observability fix, not a
  numeric change; there is no accuracy delta to report (not an optimization claim).

## Risk & rollback

- Blast radius if integrated: dataset construction only. For valid data (every episode ≥
  `window+max_horizon+1`) behaviour is unchanged — same index, same window dicts. The only
  new behaviour is on **degenerate/too-short** inputs (warn or raise instead of silent drop),
  which is the intended change. A run that *relied* on the silent drop of a few short episodes
  will now emit a `UserWarning` (still runs); a run whose data is entirely too short now fails
  fast at build instead of hanging — strictly better.
- Rollback: delete `_windowing.py` and revert the three call sites to the inline
  `range(t_max)` / `range(max(0,t_max))` — no schema, no checkpoint, no data-format impact.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate / integrate-with-changes / defer / reject
- **Date / by:** <...>
- **Reason & notes:** <...>
- **Integrated as:** <commit hash / stack path> (if applicable)
