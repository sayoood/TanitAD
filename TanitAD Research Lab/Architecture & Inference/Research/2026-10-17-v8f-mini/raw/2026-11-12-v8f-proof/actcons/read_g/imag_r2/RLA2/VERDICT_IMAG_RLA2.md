# VERDICT_IMAG_RLA2 (card PREREG_V8F_PROOF_ADD36)

Tier: T0 (a world-model diagnostic on logged frames, never driving performance). STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read.

## Controls (first)

* c1: PASS
* c2: PASS
* c4: PASS
* c3: PASS
* c5: PASS

## Names

{"H1": "IMAG-HEAD-FLAT", "H2": "IMAG-RESPONDS", "H2_overshoot_seeds": []}

## H1 (head on imagined states; label = lead gap at t + 2.0 s; seeds [2, 3])

* H1: per seed ['+0.0037', '-0.0077'], pooled -0.0020, two-level 95% interval [-0.0231, +0.0232]
* H1e: per seed ['+0.0043', '-0.0130'], pooled -0.0044, two-level 95% interval [-0.0307, +0.0222]
* H1c: per seed ['+0.1528', '+0.1413'], pooled +0.1470, two-level 95% interval [+0.0815, +0.2197]
* H1ec: per seed ['+0.1412', '+0.1294'], pooled +0.1353, two-level 95% interval [+0.0666, +0.2113]
* mean absolute error [m]: rollout (i) 10.595, present (ii) 6.350, true future (iii) 4.686 (windows [441, 441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356, 356])

* b T2.0 (named): per seed ['0.685', '0.886'], pooled 0.785, two-level interval [0.666, 0.911]
* b T1.0: per seed ['1.365', '1.151'], pooled 1.258, two-level interval [1.096, 1.425]
* b T2.0_lt20: per seed ['0.619', '0.929'], pooled 0.774, two-level interval [0.605, 0.974]
* b T1.0_lt20: per seed ['1.126', '1.266'], pooled 1.196, two-level interval [1.067, 1.386]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
