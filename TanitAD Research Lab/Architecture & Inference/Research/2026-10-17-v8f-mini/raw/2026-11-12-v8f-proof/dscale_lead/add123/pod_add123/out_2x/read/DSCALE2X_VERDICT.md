# DSCALE2X_VERDICT: card PREREG_V8F_PROOF_ADD123 (DSCALE-LEAD-2X)

TIER: T0 (a probe on cached tokens of logged frames; it says what the tokens allow, not what a model does with them); never driving performance

- LONGITUDINAL (distance keeping): not measured by this read (it is the family this lever serves)
- LATERAL: not measured by this read
- TACTICAL: not measured by this read
- STRATEGIC: NOT COMPUTED (n = 0; a token probe reads no strategic quantity)

Controls q0 to q6: all passed before this file was written (DSCALE2X_CONTROLS.json). B 2000; two-level bootstrap (seed indices, then episodes; rng 0).

## Name: **LEAD-READ-2X-SHORT-OF-TRAINING**

- s_above_0_with_n_pos_at_least_3: False
- gap_error_clause_met: False
- mean_s_larger_than_f: False
- dose_clause_holds: False
- worse_clause_met: False
- gap error ratio Q to C 1.046 (bar 0.90); upper end of L +0.4835 (dose bar +0.020)

## S, L, F

- S gap R2 (P16Q last step minus P16C, per seed): mean -0.0286 [-0.1608, +0.0726]; seeds above 0: 2 of 4; per seed +0.0072, -0.0276, +0.0255, -0.1195
- S gap error (metres; negative = Q better): mean +0.1825 [-0.5018, +0.9072]
- L gap R2 (38,328 minus 28,746): mean +0.2350 [+0.0448, +0.4835]; per seed +0.3712, +0.0842, +0.3988, +0.0861
- F |P16Q_s0 minus P16U2_s0| gap R2: 0.0000; paired episode interval of the signed difference [+0.0000, +0.0000]

## Beside (names nothing)

- gain_19164_to_28746: mean +0.1541 [-0.1637, +0.5816]; per seed +0.4451, +0.0371, -0.2302, +0.3645
- Q_minus_banked_U_M2: mean +0.0466 [-0.1181, +0.2197]; per seed +0.0897, +0.0136, +0.1874, -0.1045
- Q_minus_banked_U_MAE: mean -0.4152 [-1.4118, +0.6153]; per seed -0.9023, -0.0204, -1.3711, +0.6331

## Per probe (M2 gap R2 / gap error m / M4 AUROC)

- P16C_s0: R2 +0.712; gap error 4.12 m; AUROC 0.870
- P16U_s0: R2 +0.629; gap error 4.86 m; AUROC 0.903
- P16C_s1: R2 +0.734; gap error 4.05 m; AUROC 0.883
- P16U_s1: R2 +0.693; gap error 4.18 m; AUROC 0.899
- P16C_s2: R2 +0.698; gap error 3.83 m; AUROC 0.860
- P16U_s2: R2 +0.536; gap error 5.20 m; AUROC 0.910
- P16C_s3: R2 +0.694; gap error 3.99 m; AUROC 0.875
- P16U_s3: R2 +0.679; gap error 4.14 m; AUROC 0.905
- P16U2_s0: R2 +0.719; gap error 3.95 m; AUROC 0.917
- P16CSH_s0: R2 -0.227; gap error 9.58 m; AUROC 0.869
- P16USH_s0: R2 +0.018; gap error 8.44 m; AUROC 0.896
- P16Q_s0@19164: R2 -0.097; gap error 8.12 m; AUROC 0.860
- P16Q_s0@28746: R2 +0.348; gap error 6.59 m; AUROC 0.920
- P16Q_s0@38328: R2 +0.719; gap error 3.95 m; AUROC 0.917
- P16Q_s1@19164: R2 +0.585; gap error 5.07 m; AUROC 0.876
- P16Q_s1@28746: R2 +0.623; gap error 4.95 m; AUROC 0.870
- P16Q_s1@38328: R2 +0.707; gap error 4.16 m; AUROC 0.918
- P16Q_s2@19164: R2 +0.555; gap error 5.13 m; AUROC 0.824
- P16Q_s2@28746: R2 +0.325; gap error 6.85 m; AUROC 0.885
- P16Q_s2@38328: R2 +0.724; gap error 3.83 m; AUROC 0.919
- P16Q_s3@19164: R2 +0.124; gap error 7.41 m; AUROC 0.875
- P16Q_s3@28746: R2 +0.489; gap error 5.71 m; AUROC 0.900
- P16Q_s3@38328: R2 +0.575; gap error 4.77 m; AUROC 0.914

## Limits

One frozen encoder; pooled tokens; evaluation windows from the corpus only; four seeds, and the two-level interval treats them as the draw of seeds; the learning rate falls on a cosine, so a small late gain is partly the schedule; a probe is not a model.

