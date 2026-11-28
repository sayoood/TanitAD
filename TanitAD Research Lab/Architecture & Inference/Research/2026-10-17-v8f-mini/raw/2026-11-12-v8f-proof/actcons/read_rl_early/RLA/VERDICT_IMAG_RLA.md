# VERDICT_IMAG_RLA (card PREREG_V8F_PROOF_ADD36)

Tier: T0 (a world-model diagnostic on logged frames, never driving performance). STRATEGIC: PAUSED (PI item 25). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read.

## Controls (first)

* c1: PASS
* c2: PASS
* c4: PASS
* c3: PASS
* c5: PASS

## Names

{"H1": "IMAG-HEAD-FLAT", "H2": "IMAG-DEAF", "H2_overshoot_seeds": []}

## H1 (head on imagined states; label = lead gap at t + 2.0 s; seeds [0])

* H1: per seed ['-0.0031'], pooled -0.0031, two-level 95% interval [-0.0223, +0.0170]
* H1e: per seed ['-0.0072'], pooled -0.0072, two-level 95% interval [-0.0209, +0.0054]
* H1c: per seed ['+0.0640'], pooled +0.0640, two-level 95% interval [-0.0011, +0.1377]
* H1ec: per seed ['+0.0626'], pooled +0.0626, two-level 95% interval [+0.0061, +0.1295]
* mean absolute error [m]: rollout (i) 8093.860, present (ii) 6.701, true future (iii) 6.206 (windows [441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356])

* b T2.0 (named): per seed ['-6.821'], pooled -6.821, two-level interval [-10.594, -3.834]
* b T1.0: per seed ['-5.023'], pooled -5.023, two-level interval [-10.181, -1.342]
* b T2.0_lt20: per seed ['-5.222'], pooled -5.222, two-level interval [-7.790, -3.175]
* b T1.0_lt20: per seed ['-3.022'], pooled -3.022, two-level interval [-5.925, -0.584]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
