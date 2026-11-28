# VERDICT_IMAG_RLVa (card PREREG_V8F_PROOF_ADD36)

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

* H1: per seed ['+0.0137', '+0.0601'], pooled +0.0369, two-level 95% interval [-0.0208, +0.1174]
* H1e: per seed ['+0.0009', '+0.0254'], pooled +0.0131, two-level 95% interval [-0.0414, +0.0830]
* H1c: per seed ['+0.1493', '+0.1277'], pooled +0.1385, two-level 95% interval [+0.0669, +0.2197]
* H1ec: per seed ['+0.1396', '+0.1129'], pooled +0.1262, two-level 95% interval [+0.0563, +0.2056]
* mean absolute error [m]: rollout (i) 6.317, present (ii) 6.498, true future (iii) 4.975 (windows [441, 441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356, 356])

* b T2.0 (named): per seed ['0.879', '0.928'], pooled 0.903, two-level interval [0.855, 0.953]
* b T1.0: per seed ['1.015', '0.946'], pooled 0.980, two-level interval [0.878, 1.068]
* b T2.0_lt20: per seed ['0.813', '0.892'], pooled 0.852, two-level interval [0.772, 0.933]
* b T1.0_lt20: per seed ['0.994', '1.142'], pooled 1.068, two-level interval [0.952, 1.190]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
