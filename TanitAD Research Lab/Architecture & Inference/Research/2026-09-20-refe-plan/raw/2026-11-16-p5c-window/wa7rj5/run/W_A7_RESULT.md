# W-A7-RJ5 result (PREREG_P5C_A2 section 6, A3 section 5)

**Outcome row: W-A7-PASS**

- Evidence class: MEASURED. Tier: navtrain held-out tracked-label check (NAVSIM v1.1 PDMS, LQR tracker, non-reactive); not a driving number, not navtest; labels built from the RJ5 teacher trajectories (A3).
- Rows 1018, logs 11. Units: PDMS x 100 (pp).
- Rule: REBUILD iff CI hi < 0 AND point <= -0.10 PDMS x100 (= -0.001 on the 0..1 scale).
- Estimator: paired log-cluster bootstrap over 11 logs, 10000 draws, seed 20260927, ratio mean, 95% percentile (TF-1/E2 estimator verbatim); answers 'another draw of LOGS' only.

| label | PDMS no A7 | PDMS with A7 | delta [95% CI] (pp) | rows delta!=0 |
|---|---|---|---|---|
| P5-R (decides) | 95.8372 | 95.8239 | -0.01337 [-0.09283, +0.12245] | 513 |
| Tc20s (reported only) | 95.0799 | 94.3527 | -0.72715 [-1.88232, -0.05672] | 515 |

## Controls (any failure = INVALID)

```
{
 "coverage": {
  "n_population": 1018,
  "n_scored": 1018,
  "n_error_rows": 0,
  "pass": true
 },
 "i_prime_c1_table_reproduction": {
  "pass": true,
  "n_population": 1018,
  "n_control_rows": 1018,
  "n_common_rows": 1018,
  "n_rows_without_c1_record": 0,
  "n_table_label_not_its_record": 0,
  "n_p5r_error_rows_in_this_run": 0,
  "failed_criteria": [],
  "per_rank": {
   "0": {
    "n_common_rows": 1018,
    "n_control_rows": 1018,
    "n_sim8_identical": 1018,
    "frac_sim8_identical": 1.0,
    "rowmax_median": 8.999021202826807e-06,
    "rowmax_p99": 0.0015279254672605942,
    "rowmax_max": 0.0581071385370745,
    "p99_estimator": "numpy.percentile linear, over per-row max abs label difference (x, y, yaw)",
    "R-TOL-1": true,
    "R-TOL-2": true,
    "tb6_ratio_acc": 1.3464917571987696,
    "tb6_ratio_hr": 1.469888447404654,
    "tb6_ref_source": "recomputed with learn() from the stored control labels and the bank trajectories",
    "tb6_ref_acc": 1.3464916112496055,
    "tb6_ref_hr": 1.4698886341817203,
    "tb6_diff_acc": 1.4594916408761094e-07,
    "tb6_diff_hr": -1.8677706625780388e-07,
    "sim8_fail_rate_pod": 0.005893909626719057,
    "sim8_fail_rate_dev_eval": 0.005893909626719057,
    "sim8_fail_rate_diff_pp": 0.0,
    "R-TOL-3": true
   },
   "1": {
    "n_common_rows": 0,
    "n_control_rows": 0,
    "note": "no EVAL control row of this rank in the table"
   }
  },
  "criteria": {
   "R-TOL-1": "sim8 identical >= 99.5% of common rows",
   "R-TOL-2": "per-row max abs diff: median <= 0.001 and p99 <= 0.005",
   "R-TOL-3": "TB-6 acc/hr ratio within +-0.01 of the dev-box ratio on the SAME common rows (row-matched; verdict.json is context only) and sim8-fail rate within +-0.25 pp of the dev-box EVAL value"
  },
  "gates_module_blob": "5c2f49cf485241df4d7e7949ed629a04fd324d5c",
  "c1_table_sha256": "3b198314721fc7e75e6bd0ca9b569b5b0a56ba25178e4fb63a94b215995637a2",
  "c1_records_sha256": "90318b56cb88e55ad15dccbebf3b0ec843fdd17773f1811437650d88184b5c5e",
  "rule": "A3 section 5 (i'): R-TOL-1/2/3 of A1/A2, thresholds unchanged, by the pinned gates' _p5r_vs_control"
 },
 "i_bank_run_instrument_validity": {
  "pass": true,
  "path": "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_launchkit/w_a7/run/w_a7_result.json",
  "sha256": "eed753f7e7c272a28955ab44ec5e4f6e93c8be7cee1541e82a53db82b7f2d971",
  "outcome_row": "W-A7-PASS",
  "control_i": {
   "n_compared": 1018,
   "max_abs_diff": 0.0,
   "n_over_1e-6": 0,
   "tol": 1e-06,
   "pass": true
  },
  "context_delta_P5R_pp_point_lo_hi": [
   -0.03034,
   -0.09785,
   0.05495
  ],
  "role": "validates the instrument (control i on bank trajs); its delta is context, it informs no rule"
 },
 "ii_red_arm": {
  "n_rows_changed": 755,
  "need": 1,
  "pass": true
 },
 "iii_identity": {
  "plans_bit_equal_rows": 1018,
  "scores_bit_equal_rows": 1018,
  "n": 1018,
  "pass": true
 },
 "iv_determinism": {
  "n_rerun_rows": 102,
  "n_not_bit_identical": 0,
  "pass": true
 }
}
```

All controls pass: True

- Raw-fallback label rows: {'p5r_raw_fallback_rows': 0, 'tc20s_raw_fallback_rows': 0}
- Info: max |dual Tc20s - banked P5-S Tc20s| = 6.551157665140263

## Instrument blobs

- dual_labeller: `50b9f0dbaa22563239d6c711e9f7dab846e1cc69` (E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c/a1/p5c_dual_labels.py)
- p5r_labels: `16d2aaf05eea5c00bd081628e07b02b7dc7f480f` (E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5r/p5r_labels.py)
- p5r_run: `d0a4084b7187005df3e966a1ba0132a42a61e7d2` (E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5r/p5r_run.py)
- p5_target_labels: `294644df61ca479097971ebc92b033e24f3b7019` (E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval/p5_target_labels.py)
- planner: `6fb6598a2ecf230153a4d7adfb790afc30728070` (E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/planner.py)
- transp_score: `3cbdd118fa82790caf7c38bdd8ce6f92320c575b` (E:\Projects\TanitAD-artifacts\scorer_gym_20261107\p5c_wa7rj5/instr/transp_score.py)
- navsim_dac: `1884fee71508e8828d4642e81c665c9bb2a8c846` (E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/navsim_dac.py)
- navsim_pdm_targets: `f778e56e7220754cddf15176c356914d6753b339` (E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/navsim_pdm_targets.py)
- navtrain_scenarios: `848fe684e6927a4a21d43831954b72a1c8191690` (E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/navtrain_scenarios.py)

## Inputs

- population_npz: E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-11-p5s-smoothed/p5s_cubic.npz
- population_npz_sha256: 0a0866fe631a5b52ca9d42c88f46cc8944c1acd6d13bd0fcb0345c908a006072
- labels_sha256: 40f9da37cb308357da3c6e66cf284b430e6304c5e5e6cede9c14b8f2fc652f28
- scores_sha256: 852bb833efcb01e143afd61474fb6fece712b49b83d50eee7066964d1bbaf0d3
- tf1_rows: ['E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/tf1/rows/tf1_s0.jsonl', 'E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/tf1/rows/tf1_s1.jsonl', 'E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/tf1/rows/tf1_s2.jsonl', 'E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/tf1/rows/tf1_s3.jsonl', 'E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/tf1/rows/tf1_s4r.jsonl']
