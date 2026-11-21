# VERDICT_IMAG_LAm (card PREREG_V8F_PROOF_ADD36)

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

* H1: per seed ['-0.0032', '-0.0141'], pooled -0.0087, two-level 95% interval [-0.0234, +0.0001]
* H1e: per seed ['-0.0105', '-0.0143'], pooled -0.0124, two-level 95% interval [-0.0258, +0.0001]
* H1c: per seed ['+0.1312', '+0.1055'], pooled +0.1184, two-level 95% interval [+0.0556, +0.1890]
* H1ec: per seed ['+0.1185', '+0.0984'], pooled +0.1085, two-level 95% interval [+0.0446, +0.1796]
* mean absolute error [m]: rollout (i) 101.878, present (ii) 6.383, true future (iii) 4.984 (windows [441, 441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356, 356])

* b T2.0 (named): per seed ['0.859', '1.066'], pooled 0.962, two-level interval [0.808, 1.128]
* b T1.0: per seed ['0.862', '1.216'], pooled 1.039, two-level interval [0.795, 1.269]
* b T2.0_lt20: per seed ['0.820', '1.077'], pooled 0.948, two-level interval [0.735, 1.190]
* b T1.0_lt20: per seed ['0.770', '1.179'], pooled 0.974, two-level interval [0.649, 1.249]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
