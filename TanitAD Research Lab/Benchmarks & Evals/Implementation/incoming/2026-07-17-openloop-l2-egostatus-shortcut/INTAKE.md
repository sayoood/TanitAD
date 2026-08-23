# INTAKE — Open-loop L2 protocol + ego-status shortcut ceiling

- **Package:** `Benchmarks & Eval/Implementation/incoming/2026-07-17-openloop-l2-egostatus-shortcut/`
- **Author agent / date:** Benchmarks & Eval agent, 2026-07-17
- **Proposed target:** `stack/tanitad/eval/openloop_l2.py` (beside `metrics.py`, the gate runner, and
  the floor predictors) — the community-standard open-loop leg of the eval suite.
- **Hypothesis / WP served:** WP6 / gate D1 / G1 (honest driving-capability denominator) — the
  validation-strategy mission (own the leaderboard, honest placement).

## What & why (≤10 lines)
Adds the **nuScenes-style open-loop L2** protocol (metric-BEV ego frame, metres, both UniAD `pointwise`
and ST-P3/VAD `cumulative` averaging conventions, 1/2/3 s), a `collision_rate` proxy, and the **no-vision
ego-status shortcut** (`RidgeTrajectoryHead`, AD-MLP repro, arXiv 2312.03031). Motivation: our D1 is
camera-frame vs a kinematic floor — neither is comparable to the number the field ranks driving on, and
open-loop L2 is dominated by an ego-status shortcut that must be reported *beside* every model number.
Measured on our real val corpora, the shortcut ceiling is **avg L2 0.66 m (comma-hwy)**, and comma is
**73.9 % straight** — the same shortcut pathology as nuScenes. `skill_score = model_L2 ÷ shortcut` is now
defined in leaderboard-comparable units. Research note:
`Research/2026-07-17-openloop-l2-egostatus-shortcut.md`.

## Evidence & tests
- Tests: `tests/test_openloop_l2.py` — **8 passed** (0.14 s) on author machine (venv `tanitad`,
  numpy-only). All analytic ground truth (G-B2): convention arithmetic, kinematic baselines exact on
  CV motion, ridge exact-linear recovery, shortcut learns CV, collision hit/miss, skill_score.
- Measured numbers: `results_openloop_l2.json` — comma-hwy 7 920 val anchors: shortcut avg L2 **0.658 m**
  (0.144/0.552/1.256 @1/2/3s), best-of-3 floor 0.571 m, stop-null 49.9 m; 73.9 % straight population.
  cosmos-urban 318 anchors: shortcut 1.191 m (beats kinematic floor 1.335 m). Clip-level held-out split.
- `baseline_predictors.py` is **vendored** from `../2026-07-15-baseline-floor/` for a standalone
  `pytest`; source of truth stays that package / the integration target.

## Risk & rollback
- Blast radius: additive, self-contained eval module + tests; no change to training, contract, or the
  running gate runner. If integrated at `stack/tanitad/eval/openloop_l2.py`, de-duplicate the vendored
  `baseline_predictors` against the floor-package integration (single source in `stack/tanitad/eval/`).
- Rollback: delete the module + test; nothing depends on it yet.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate / integrate-with-changes / defer / reject
- **Date / by:**
- **Reason & notes:**
- **Integrated as:**
