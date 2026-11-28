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

## H1 (head on imagined states; label = lead gap at t + 2.0 s; seeds [0, 1])

* H1: per seed ['-0.0031', '-0.0164'], pooled -0.0097, two-level 95% interval [-0.0275, +0.0061]
* H1e: per seed ['-0.0072', '-0.0172'], pooled -0.0122, two-level 95% interval [-0.0259, -0.0011]
* H1c: per seed ['+0.0640', '+0.0500'], pooled +0.0570, two-level 95% interval [+0.0097, +0.1097]
* H1ec: per seed ['+0.0626', '+0.0146'], pooled +0.0386, two-level 95% interval [-0.0092, +0.0972]
* mean absolute error [m]: rollout (i) 27597.490, present (ii) 8.164, true future (iii) 7.566 (windows [441, 441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356, 356])

* b T2.0 (named): per seed ['-6.821', '-13.280'], pooled -10.050, two-level interval [-17.301, -5.145]
* b T1.0: per seed ['-5.023', '3.284'], pooled -0.869, two-level interval [-8.440, 9.789]
* b T2.0_lt20: per seed ['-5.222', '-23.581'], pooled -14.401, two-level interval [-30.944, -4.293]
* b T1.0_lt20: per seed ['-3.022', '-13.241'], pooled -8.132, two-level interval [-25.335, 1.974]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
