# INTAKE — D1 gate bootstrap / multi-seed reporting

- **Package:** `Benchmarks & Eval/Implementation/incoming/2026-07-11-d1-gate-bootstrap/`
- **Author agent / date:** Benchmarks & Eval agent, 2026-07-11
- **Proposed target:** `stack/tanitad/eval/gates.py` (add `run_d1_bootstrap` next to `run_d1`);
  optionally wire `--d1-seeds N` into `stack/scripts/evaluate_checkpoint.py`.
- **Hypothesis / WP served:** validation-strategy / gate-integrity (D1, and by extension D3); no H claim.

## What & why (≤10 lines)

Additive wrapper that reports the D1 decode gate's `ade@1s` as **mean ± 95 % CI over N split seeds**
instead of a single fixed seed=0 point. It calls the unchanged `run_d1` per seed (cannot alter the
estimator) and returns `ade@1s_mean/sd/ci95/ci95_halfwidth`, `n_val_eps_approx`, `single_seed0`, and a
`decision_grade` flag (CI half-width < 3.17 m AND ≥20 val eps). Motivated by the measured power audit
(`Research/2026-07-11-d1-ade-statistical-power-audit.md`): the shipped single-seed `run_d1` swings **5–7 m
across seeds** on the *same* checkpoint at 4–9 val episodes — larger than the gate-movement deltas the
program reads. The step-14k→21k "D1 5.18→11.52 m regression" is a sampling artifact inside this band.

## Evidence & tests

- Tests: `tests/test_run_d1_bootstrap.py` — **4 passed** (author machine, venv `tanitad`, 2.25 s).
  Verifies: (1) wrapper does not alter the estimator (`single_seed0` == plain `run_d1` seed=0);
  (2) CI half-width shrinks with more val episodes (audit's core claim, on the real estimator);
  (3) `seeds<2` rejected; (4) `decision_grade` False when val eps < 20.
- Measured numbers (audit, step-6500 ckpt, real comma2k19, RTX 4060, $0): per-route ADE@1s
  2.31–18.75 m (CoV 0.58); shipped `run_d1` seed range **7.28 m** at ~4 val eps / **5.46 m** at ~8;
  fixed-probe bootstrap 95 % CI half-width ±4.51 m (n=4), ±3.13 m (n=9), ±2.11 m (n=20).

## Risk & rollback

- Blast radius: none if only the helper is added (new function, no signature change to `run_d1`). If
  `evaluate_checkpoint.py` is switched to call it, gate JSON gains fields (`ade@1s_mean/ci95`), which the
  LEADERBOARD/monitor readers should tolerate (additive keys); keep `ade@1s` for back-compat.
- Rollback: delete `run_d1_bootstrap`; revert the evaluator flag. No state migration.
- Cost of ignoring: D1 (and D3) "gate movement" claims stay non-decision-grade; monitor previews keep
  reporting single-seed points that swing on the seed.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate / integrate-with-changes / defer / reject
- **Date / by:**
- **Reason & notes:**
- **Integrated as:**
