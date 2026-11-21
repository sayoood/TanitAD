# VERDICT_IMAG_LA (card PREREG_V8F_PROOF_ADD36)

Tier: T0 (a world-model diagnostic on logged frames, never driving performance). STRATEGIC: PAUSED (PI item 25). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read.

## Controls (first)

* c1: PASS
* c2: PASS
* c4: PASS
* c3: PASS
* c5: PASS

## Names

{"H1": "IMAG-HEAD-FLAT", "H2": "IMAG-RESPONDS", "H2_overshoot_seeds": []}

## H1 (head on imagined states; label = lead gap at t + 2.0 s; seeds [0, 1])

* H1: per seed ['-0.0003', '-0.0350'], pooled -0.0177, two-level 95% interval [-0.0596, +0.0000]
* H1e: per seed ['-0.0007', '-0.0441'], pooled -0.0224, two-level 95% interval [-0.0733, +0.0004]
* H1c: per seed ['+0.1143', '+0.0937'], pooled +0.1040, two-level 95% interval [+0.0445, +0.1709]
* H1ec: per seed ['+0.1035', '+0.0870'], pooled +0.0952, two-level 95% interval [+0.0373, +0.1634]
* mean absolute error [m]: rollout (i) 7371.981, present (ii) 6.661, true future (iii) 5.443 (windows [441, 441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356, 356])

* b T2.0 (named): per seed ['0.912', '0.799'], pooled 0.855, two-level interval [0.694, 0.958]
* b T1.0: per seed ['1.613', '2.195'], pooled 1.904, two-level interval [1.178, 3.293]
* b T2.0_lt20: per seed ['0.842', '0.882'], pooled 0.862, two-level interval [0.798, 0.909]
* b T1.0_lt20: per seed ['1.212', '3.565'], pooled 2.389, two-level interval [1.109, 6.395]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
