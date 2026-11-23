# DSCALE_PRES_VERDICT: card PREREG_V8F_PROOF_ADD126 (DSCALE-PRES-T0)

TIER: T0 (a probe on cached tokens of logged frames; it says what the tokens allow, not what a model does with them); never driving performance

- LONGITUDINAL (distance keeping): not measured by this read (it is the family this lever serves)
- LATERAL: not measured by this read
- TACTICAL: not measured by this read
- STRATEGIC: NOT COMPUTED (n = 0; a token probe reads no strategic quantity)

Controls p0 to p5: all passed before this file was written (DSCALE_PRES_CONTROLS.json). B 2000; two-level bootstrap (seed indices, then episodes; rng 0); AUROC computed again on every episode draw.

## Name: **LEAD-PRES-EQP-SCALES**

- interval_above_0: True
- interval_below_0: False
- seeds_above_0: 4
- seeds_below_0: 0
- mean of A +0.0415 (bar 0.020)

## A (presence AUROC of P16E minus P16C, per seed, same windows)

- A: mean +0.0415 [+0.0142, +0.0717]; seeds above 0: 4 of 4; per seed +0.0459, +0.0303, +0.0414, +0.0482

## Beside (names nothing)

- P16Q minus P16C presence AUROC: mean +0.0453 [+0.0128, +0.0821]; seeds above 0: 4 of 4; per seed +0.0472, +0.0350, +0.0595, +0.0393
- P16U minus P16C presence AUROC: mean +0.0322 [-0.0014, +0.0692]; seeds above 0: 4 of 4; per seed +0.0329, +0.0159, +0.0500, +0.0301
- windows: 1844, with a lead 534, without 1310, in 139 episodes

## Per probe (AUROC; share of lead windows scored below 0.5 in all and by label gap; share of no-lead windows scored at 0.5 or more)

- P16C_s0: AUROC 0.870; below 0.5: all 0.109, 0-20 m 0.061 (n 294), 20-40 m 0.167 (n 240), 40+ m n/a (n 0); no-lead at or above 0.5: 0.344
- P16C_s1: AUROC 0.883; below 0.5: all 0.124, 0-20 m 0.082 (n 294), 20-40 m 0.175 (n 240), 40+ m n/a (n 0); no-lead at or above 0.5: 0.305
- P16C_s2: AUROC 0.860; below 0.5: all 0.148, 0-20 m 0.092 (n 294), 20-40 m 0.217 (n 240), 40+ m n/a (n 0); no-lead at or above 0.5: 0.276
- P16C_s3: AUROC 0.875; below 0.5: all 0.103, 0-20 m 0.065 (n 294), 20-40 m 0.150 (n 240), 40+ m n/a (n 0); no-lead at or above 0.5: 0.283
- P16U_s0: AUROC 0.903; below 0.5: all 0.051, 0-20 m 0.041 (n 294), 20-40 m 0.062 (n 240), 40+ m n/a (n 0); no-lead at or above 0.5: 0.369
- P16U_s1: AUROC 0.899; below 0.5: all 0.049, 0-20 m 0.048 (n 294), 20-40 m 0.050 (n 240), 40+ m n/a (n 0); no-lead at or above 0.5: 0.367
- P16U_s2: AUROC 0.910; below 0.5: all 0.041, 0-20 m 0.027 (n 294), 20-40 m 0.058 (n 240), 40+ m n/a (n 0); no-lead at or above 0.5: 0.376
- P16U_s3: AUROC 0.905; below 0.5: all 0.051, 0-20 m 0.051 (n 294), 20-40 m 0.050 (n 240), 40+ m n/a (n 0); no-lead at or above 0.5: 0.394
- P16CSH_s0: AUROC 0.869; below 0.5: all 0.127, 0-20 m 0.085 (n 294), 20-40 m 0.179 (n 240), 40+ m n/a (n 0); no-lead at or above 0.5: 0.287
- P16USH_s0: AUROC 0.896; below 0.5: all 0.049, 0-20 m 0.044 (n 294), 20-40 m 0.054 (n 240), 40+ m n/a (n 0); no-lead at or above 0.5: 0.402
- P16Q_s0: AUROC 0.917; below 0.5: all 0.054, 0-20 m 0.024 (n 294), 20-40 m 0.092 (n 240), 40+ m n/a (n 0); no-lead at or above 0.5: 0.398
- P16Q_s1: AUROC 0.918; below 0.5: all 0.052, 0-20 m 0.024 (n 294), 20-40 m 0.087 (n 240), 40+ m n/a (n 0); no-lead at or above 0.5: 0.374
- P16Q_s2: AUROC 0.919; below 0.5: all 0.069, 0-20 m 0.034 (n 294), 20-40 m 0.113 (n 240), 40+ m n/a (n 0); no-lead at or above 0.5: 0.337
- P16Q_s3: AUROC 0.914; below 0.5: all 0.073, 0-20 m 0.051 (n 294), 20-40 m 0.100 (n 240), 40+ m n/a (n 0); no-lead at or above 0.5: 0.368
- P16E_s0: AUROC 0.916; below 0.5: all 0.081, 0-20 m 0.061 (n 294), 20-40 m 0.104 (n 240), 40+ m n/a (n 0); no-lead at or above 0.5: 0.291
- P16E_s1: AUROC 0.914; below 0.5: all 0.097, 0-20 m 0.061 (n 294), 20-40 m 0.142 (n 240), 40+ m n/a (n 0); no-lead at or above 0.5: 0.291
- P16E_s2: AUROC 0.901; below 0.5: all 0.124, 0-20 m 0.088 (n 294), 20-40 m 0.167 (n 240), 40+ m n/a (n 0); no-lead at or above 0.5: 0.252
- P16E_s3: AUROC 0.923; below 0.5: all 0.062, 0-20 m 0.041 (n 294), 20-40 m 0.087 (n 240), 40+ m n/a (n 0); no-lead at or above 0.5: 0.292

## Limits

One frozen encoder; pooled tokens; the evaluation windows come from the corpus only; four seeds, and the two-level interval treats them as the draw of seeds. The point values were seen before the card (card section 2). An interval above 0 on four seeds is necessary for a name and not sufficient for a claim about training runs at large. A probe is not a model.

