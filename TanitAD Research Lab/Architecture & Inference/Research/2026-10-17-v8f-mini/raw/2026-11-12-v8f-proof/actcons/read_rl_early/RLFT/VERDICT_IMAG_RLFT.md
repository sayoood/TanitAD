# VERDICT_IMAG_RLFT (card PREREG_V8F_PROOF_ADD36)

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

* H1: per seed ['+0.0462'], pooled +0.0462, two-level 95% interval [-0.0097, +0.1061]
* H1e: per seed ['+0.0377'], pooled +0.0377, two-level 95% interval [-0.0164, +0.0938]
* H1c: per seed ['+0.1503'], pooled +0.1503, two-level 95% interval [+0.0535, +0.2516]
* H1ec: per seed ['+0.1405'], pooled +0.1405, two-level 95% interval [+0.0475, +0.2400]
* mean absolute error [m]: rollout (i) 5.475, present (ii) 6.284, true future (iii) 4.746 (windows [441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356])

* b T2.0 (named): per seed ['-0.911'], pooled -0.911, two-level interval [-1.018, -0.805]
* b T1.0: per seed ['-1.611'], pooled -1.611, two-level interval [-1.796, -1.424]
* b T2.0_lt20: per seed ['-0.796'], pooled -0.796, two-level interval [-0.924, -0.668]
* b T1.0_lt20: per seed ['-1.444'], pooled -1.444, two-level interval [-1.709, -1.198]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
