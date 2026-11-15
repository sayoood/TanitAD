# VERDICT_IMAG_LA01 (card PREREG_V8F_PROOF_ADD36)

Tier: T0 (a world-model diagnostic on logged frames, never driving performance). STRATEGIC: PAUSED (PI item 25). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read.

## Controls (first)

* c1: PASS
* c2: PASS
* c4: PASS
* c3: PASS
* c5: PASS

## Names

{"H1": "IMAG-HEAD-FLAT", "H2": "IMAG-RESPONDS-WEAKLY", "H2_overshoot_seeds": []}

## H1 (head on imagined states; label = lead gap at t + 2.0 s; seeds [0, 1])

* H1: per seed ['-0.0075', '-0.0263'], pooled -0.0169, two-level 95% interval [-0.0457, -0.0025]
* H1e: per seed ['-0.0177', '-0.0373'], pooled -0.0275, two-level 95% interval [-0.0661, -0.0080]
* H1c: per seed ['+0.1290', '+0.1176'], pooled +0.1233, two-level 95% interval [+0.0614, +0.1936]
* H1ec: per seed ['+0.1189', '+0.1069'], pooled +0.1129, two-level 95% interval [+0.0503, +0.1821]
* mean absolute error [m]: rollout (i) 143.418, present (ii) 6.574, true future (iii) 5.127 (windows [441, 441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356, 356])

* b T2.0 (named): per seed ['0.809', '0.496'], pooled 0.652, two-level interval [0.457, 0.844]
* b T1.0: per seed ['1.082', '1.059'], pooled 1.071, two-level interval [0.657, 1.459]
* b T2.0_lt20: per seed ['0.797', '0.461'], pooled 0.629, two-level interval [0.425, 0.848]
* b T1.0_lt20: per seed ['1.065', '0.995'], pooled 1.030, two-level interval [0.259, 1.819]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
