# VERDICT_IMAG_RL (card PREREG_V8F_PROOF_ADD36)

Tier: T0 (a world-model diagnostic on logged frames, never driving performance). STRATEGIC: PAUSED (PI item 25). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read.

## Controls (first)

* c1: PASS
* c2: PASS
* c4: PASS
* c3: PASS
* c5: PASS

## Names

{"H1": "IMAG-HEAD-ADDS + IMAG-HEAD-ADDS-EGO-ONLY", "H2": "IMAG-DEAF", "H2_overshoot_seeds": []}

## H1 (head on imagined states; label = lead gap at t + 2.0 s; seeds [0, 1])

* H1: per seed ['+0.0431', '+0.0721'], pooled +0.0576, two-level 95% interval [+0.0088, +0.1097]
* H1e: per seed ['+0.0223', '+0.0517'], pooled +0.0370, two-level 95% interval [-0.0125, +0.0865]
* H1c: per seed ['+0.1107', '+0.1378'], pooled +0.1242, two-level 95% interval [+0.0551, +0.2075]
* H1ec: per seed ['+0.0938', '+0.1188'], pooled +0.1063, two-level 95% interval [+0.0420, +0.1836]
* mean absolute error [m]: rollout (i) 5.773, present (ii) 6.823, true future (iii) 5.619 (windows [441, 441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356, 356])

* b T2.0 (named): per seed ['-0.875', '-0.901'], pooled -0.888, two-level interval [-0.958, -0.822]
* b T1.0: per seed ['-1.552', '-1.602'], pooled -1.577, two-level interval [-1.697, -1.454]
* b T2.0_lt20: per seed ['-0.808', '-0.824'], pooled -0.816, two-level interval [-0.914, -0.723]
* b T1.0_lt20: per seed ['-1.432', '-1.461'], pooled -1.447, two-level interval [-1.620, -1.273]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
