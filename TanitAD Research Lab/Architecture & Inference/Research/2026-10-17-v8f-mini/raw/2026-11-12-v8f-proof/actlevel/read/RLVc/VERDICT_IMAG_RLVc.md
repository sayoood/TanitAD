# VERDICT_IMAG_RLVc (card PREREG_V8F_PROOF_ADD36)

Tier: T0 (a world-model diagnostic on logged frames, never driving performance). STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read.

## Controls (first)

* c1: PASS
* c2: PASS
* c4: PASS
* c3: PASS
* c5: PASS

## Names

{"H1": "IMAG-HEAD-FLAT", "H2": "IMAG-RESPONDS", "H2_overshoot_seeds": []}

## H1 (head on imagined states; label = lead gap at t + 2.0 s; seeds [0, 1])

* H1: per seed ['+0.0107', '+0.0956'], pooled +0.0531, two-level 95% interval [-0.0163, +0.1592]
* H1e: per seed ['-0.0020', '+0.0581'], pooled +0.0281, two-level 95% interval [-0.0332, +0.1183]
* H1c: per seed ['+0.1510', '+0.1282'], pooled +0.1396, two-level 95% interval [+0.0674, +0.2202]
* H1ec: per seed ['+0.1411', '+0.1133'], pooled +0.1272, two-level 95% interval [+0.0571, +0.2071]
* mean absolute error [m]: rollout (i) 6.137, present (ii) 6.515, true future (iii) 4.998 (windows [441, 441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356, 356])

* b T2.0 (named): per seed ['0.751', '0.753'], pooled 0.752, two-level interval [0.701, 0.803]
* b T1.0: per seed ['0.990', '0.898'], pooled 0.944, two-level interval [0.817, 1.052]
* b T2.0_lt20: per seed ['0.650', '0.640'], pooled 0.645, two-level interval [0.571, 0.726]
* b T1.0_lt20: per seed ['0.883', '0.898'], pooled 0.890, two-level interval [0.793, 0.993]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
