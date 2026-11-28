# W-A7 result (PREREG_P5C_A2 section 6)

**Outcome row: SMOKE (not an outcome)**

- Evidence class: MEASURED. Tier: navtrain held-out tracked-label check (NAVSIM v1.1 PDMS, LQR tracker, non-reactive); not a driving number, not navtest.
- Rows 20, logs 9. Units: PDMS x 100 (pp).
- Rule: REBUILD iff CI hi < 0 AND point <= -0.10 PDMS x100 (= -0.001 on the 0..1 scale).
- Estimator: paired log-cluster bootstrap over 9 logs, 10000 draws, seed 20260927, ratio mean, 95% percentile (TF-1/E2 estimator verbatim); answers 'another draw of LOGS' only.

## Controls (any failure = INVALID)

```
{
 "coverage": {
  "n_population": 20,
  "n_scored": 20,
  "n_error_rows": 0,
  "pass": true
 },
 "i_reproduction_tf1_tc20s_tracked": {
  "n_compared": 20,
  "max_abs_diff": 0.0,
  "n_over_1e-6": 0,
  "tol": 1e-06,
  "pass": true
 },
 "ii_red_arm": {
  "n_rows_changed": 16,
  "need": 1,
  "pass": true
 },
 "iii_identity": {
  "plans_bit_equal_rows": 20,
  "scores_bit_equal_rows": 20,
  "n": 20,
  "pass": true
 },
 "iv_determinism": {
  "n_rerun_rows": 4,
  "n_not_bit_identical": 0,
  "pass": true
 }
}
```

All controls pass: True

- Raw-fallback label rows: {'p5r_raw_fallback_rows': 0, 'tc20s_raw_fallback_rows': 0}
- Info: max |dual Tc20s - banked P5-S Tc20s| = 1.021405182655144e-14

## Instrument blobs

- dual_labeller: `50b9f0dbaa22563239d6c711e9f7dab846e1cc69` (E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c/a1/p5c_dual_labels.py)
- p5r_labels: `16d2aaf05eea5c00bd081628e07b02b7dc7f480f` (E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5r/p5r_labels.py)
- p5r_run: `d0a4084b7187005df3e966a1ba0132a42a61e7d2` (E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5r/p5r_run.py)
- p5_target_labels: `294644df61ca479097971ebc92b033e24f3b7019` (E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval/p5_target_labels.py)
- planner: `6fb6598a2ecf230153a4d7adfb790afc30728070` (E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/planner.py)
- transp_score: `3cbdd118fa82790caf7c38bdd8ce6f92320c575b` (E:\Projects\TanitAD-artifacts\scorer_gym_20261107\p5c_launchkit\w_a7/instr/transp_score.py)
- navsim_dac: `1884fee71508e8828d4642e81c665c9bb2a8c846` (E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/navsim_dac.py)
- navsim_pdm_targets: `f778e56e7220754cddf15176c356914d6753b339` (E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/navsim_pdm_targets.py)
- navtrain_scenarios: `848fe684e6927a4a21d43831954b72a1c8191690` (E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/navtrain_scenarios.py)

## Inputs

- population_npz: E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-11-p5s-smoothed/p5s_cubic.npz
- population_npz_sha256: 0a0866fe631a5b52ca9d42c88f46cc8944c1acd6d13bd0fcb0345c908a006072
- labels_sha256: 940b4e2198b646357e1d269a0fffba527c5b0e31fc966add9cf3f2203705dc33
- scores_sha256: ef86637a6eae4f22fcdf07cccc6f5a6eb7e8808a78c5400c643b0a825f8e3c38
- tf1_rows: ['E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/tf1/rows/tf1_s0.jsonl', 'E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/tf1/rows/tf1_s1.jsonl', 'E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/tf1/rows/tf1_s2.jsonl', 'E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/tf1/rows/tf1_s3.jsonl', 'E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/tf1/rows/tf1_s4r.jsonl']
