# VERDICT_IMAG_RLAh (card PREREG_V8F_PROOF_ADD36)

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

* H1: per seed ['-0.0156', '+0.0511'], pooled +0.0178, two-level 95% interval [-0.0312, +0.0797]
* H1e: per seed ['-0.0201', '+0.0371'], pooled +0.0085, two-level 95% interval [-0.0352, +0.0613]
* H1c: per seed ['+0.1497', '+0.1291'], pooled +0.1394, two-level 95% interval [+0.0672, +0.2204]
* H1ec: per seed ['+0.1396', '+0.1140'], pooled +0.1268, two-level 95% interval [+0.0571, +0.2059]
* mean absolute error [m]: rollout (i) 6.872, present (ii) 6.477, true future (iii) 4.961 (windows [441, 441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356, 356])

* b T2.0 (named): per seed ['0.912', '0.928'], pooled 0.920, two-level interval [0.882, 0.960]
* b T1.0: per seed ['1.147', '1.110'], pooled 1.128, two-level interval [1.053, 1.194]
* b T2.0_lt20: per seed ['0.864', '0.859'], pooled 0.861, two-level interval [0.796, 0.927]
* b T1.0_lt20: per seed ['1.167', '1.124'], pooled 1.146, two-level interval [1.045, 1.249]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
