# VERDICT_IMAG_RLVd (card PREREG_V8F_PROOF_ADD36)

Tier: T0 (a world-model diagnostic on logged frames, never driving performance). STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read.

## Controls (first)

* c1: PASS
* c2: PASS
* c4: PASS
* c3: PASS
* c5: PASS

## Names

{"H1": "IMAG-HEAD-FLAT", "H2": "IMAG-RESPONDS-WEAKLY", "H2_overshoot_seeds": []}

## H1 (head on imagined states; label = lead gap at t + 2.0 s; seeds [0, 1])

* H1: per seed ['+0.0108', '+0.0911'], pooled +0.0509, two-level 95% interval [-0.0167, +0.1569]
* H1e: per seed ['+0.0014', '+0.0537'], pooled +0.0275, two-level 95% interval [-0.0335, +0.1140]
* H1c: per seed ['+0.1476', '+0.1282'], pooled +0.1379, two-level 95% interval [+0.0667, +0.2183]
* H1ec: per seed ['+0.1379', '+0.1133'], pooled +0.1256, two-level 95% interval [+0.0557, +0.2045]
* mean absolute error [m]: rollout (i) 5.946, present (ii) 6.493, true future (iii) 4.972 (windows [441, 441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356, 356])

* b T2.0 (named): per seed ['0.395', '0.296'], pooled 0.345, two-level interval [0.233, 0.435]
* b T1.0: per seed ['0.687', '0.546'], pooled 0.616, two-level interval [0.410, 0.791]
* b T2.0_lt20: per seed ['0.253', '0.061'], pooled 0.157, two-level interval [-0.001, 0.313]
* b T1.0_lt20: per seed ['0.345', '0.125'], pooled 0.235, two-level interval [-0.013, 0.453]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
