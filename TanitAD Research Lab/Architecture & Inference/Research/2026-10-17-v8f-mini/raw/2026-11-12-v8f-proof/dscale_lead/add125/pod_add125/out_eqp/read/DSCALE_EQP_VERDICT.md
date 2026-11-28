# DSCALE_EQP_VERDICT: card PREREG_V8F_PROOF_ADD125 (DSCALE-LEAD-EQP)

TIER: T0 (a probe on cached tokens of logged frames; it says what the tokens allow, not what a model does with them); never driving performance

- LONGITUDINAL (distance keeping): not measured by this read (it is the family this lever serves)
- LATERAL: not measured by this read
- TACTICAL: not measured by this read
- STRATEGIC: NOT COMPUTED (n = 0; a token probe reads no strategic quantity)

Controls e0 to e6: all passed before this file was written (DSCALE_EQP_CONTROLS.json). B 2000; two-level bootstrap (seed indices, then episodes; rng 0).

## Name: **LEAD-READ-EQP-FLAT**

- e_above_0_with_n_pos_at_least_3: False
- e_below_0_with_n_neg_at_least_3: False
- gap_error_bar_met: False
- gap error ratio E to C 1.043 (bar 0.90)
- upper end of the interval of E on gap R2 (the gain the read leaves open): +0.0734

## E, T

- E gap R2 (P16E minus P16C, per seed, last steps): mean -0.0189 [-0.1267, +0.0734]; seeds above 0: 2 of 4; per seed -0.0685, -0.0767, +0.0479, +0.0218
- E gap error (metres; negative = E better): mean +0.1712 [-0.3860, +0.7453]
- T gap R2 (P16E minus P16Q: 4.0 against 2.5 passes): mean +0.0097 [-0.1237, +0.1486]; seeds above 0: 2 of 4; per seed -0.0758, -0.0491, +0.0224, +0.1413
- T gap error (metres): mean -0.0114 [-0.7069, +0.7635]

## Beside (names nothing)

- P16Q minus P16U gap R2 (2.5 against 1.25 passes, recomputed): mean +0.0466 [-0.1181, +0.2197]; seeds above 0: 3 of 4; per seed +0.0897, +0.0136, +0.1874, -0.1045
- known value: row 192 of card ADD123: P16Q minus P16U (2.5 against 1.25 passes, end states): +0.0466 [-0.1181, +0.2197]
- S of card ADD123 recomputed (P16Q minus P16C) gap R2: mean -0.0286 [-0.1608, +0.0726]; seeds above 0: 2 of 4; per seed +0.0072, -0.0276, +0.0255, -0.1195
- S of card ADD123 recomputed gap error (metres): mean +0.1825 [-0.5018, +0.9072]

## Per probe (M2 gap R2 / gap error m / M4 AUROC)

- P16C_s0: R2 +0.712; gap error 4.12 m; AUROC 0.870
- P16C_s1: R2 +0.734; gap error 4.05 m; AUROC 0.883
- P16C_s2: R2 +0.698; gap error 3.83 m; AUROC 0.860
- P16C_s3: R2 +0.694; gap error 3.99 m; AUROC 0.875
- P16U_s0: R2 +0.629; gap error 4.86 m; AUROC 0.903
- P16U_s1: R2 +0.693; gap error 4.18 m; AUROC 0.899
- P16U_s2: R2 +0.536; gap error 5.20 m; AUROC 0.910
- P16U_s3: R2 +0.679; gap error 4.14 m; AUROC 0.905
- P16CSH_s0: R2 -0.227; gap error 9.58 m; AUROC 0.869
- P16USH_s0: R2 +0.018; gap error 8.44 m; AUROC 0.896
- P16Q_s0: R2 +0.719; gap error 3.95 m; AUROC 0.917
- P16Q_s1: R2 +0.707; gap error 4.16 m; AUROC 0.918
- P16Q_s2: R2 +0.724; gap error 3.83 m; AUROC 0.919
- P16Q_s3: R2 +0.575; gap error 4.77 m; AUROC 0.914
- P16E_s0: R2 +0.643; gap error 4.39 m; AUROC 0.916
- P16E_s1: R2 +0.658; gap error 4.43 m; AUROC 0.914
- P16E_s2: R2 +0.746; gap error 3.66 m; AUROC 0.901
- P16E_s3: R2 +0.716; gap error 4.20 m; AUROC 0.923

## Limits

One frozen encoder; pooled tokens; the evaluation windows come from the corpus only; four seeds, and the two-level interval treats them as the draw of seeds; the corpus probes are banked runs of 2026-11-20 on the same pod and tool. An interval above or below 0 on four seeds is necessary for a name and not sufficient for a claim about training runs at large. A probe is not a model.

