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

## H1 (head on imagined states; label = lead gap at t + 2.0 s; seeds [0, 1])

* H1: per seed ['+0.0161', '-0.0126'], pooled +0.0018, two-level 95% interval [-0.0219, +0.0285]
* H1e: per seed ['+0.0157', '-0.0138'], pooled +0.0010, two-level 95% interval [-0.0201, +0.0288]
* H1c: per seed ['+0.1480', '+0.1321'], pooled +0.1401, two-level 95% interval [+0.0687, +0.2213]
* H1ec: per seed ['+0.1383', '+0.1157'], pooled +0.1270, two-level 95% interval [+0.0572, +0.2060]
* mean absolute error [m]: rollout (i) 11.817, present (ii) 6.504, true future (iii) 4.969 (windows [441, 441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356, 356])

* b T2.0 (named): per seed ['0.872', '0.852'], pooled 0.862, two-level interval [0.816, 0.906]
* b T1.0: per seed ['1.316', '1.088'], pooled 1.202, two-level interval [1.019, 1.377]
* b T2.0_lt20: per seed ['0.794', '0.751'], pooled 0.772, two-level interval [0.711, 0.839]
* b T1.0_lt20: per seed ['1.168', '1.063'], pooled 1.116, two-level interval [0.947, 1.287]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
