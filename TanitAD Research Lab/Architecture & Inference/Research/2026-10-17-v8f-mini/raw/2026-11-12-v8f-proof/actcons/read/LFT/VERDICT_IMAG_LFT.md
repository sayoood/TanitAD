# VERDICT_IMAG_LFT (card PREREG_V8F_PROOF_ADD36)

Tier: T0 (a world-model diagnostic on logged frames, never driving performance). STRATEGIC: PAUSED (PI item 25). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read.

## Controls (first)

* c1: PASS
* c2: PASS
* c4: PASS
* c3: PASS
* c5: PASS

## Names

{"H1": "IMAG-HEAD-ADDS + IMAG-HEAD-ADDS-LEAD", "H2": "IMAG-DEAF", "H2_overshoot_seeds": []}

## H1 (head on imagined states; label = lead gap at t + 2.0 s; seeds [0, 1])

* H1: per seed ['+0.0705', '+0.0767'], pooled +0.0736, two-level 95% interval [+0.0334, +0.1190]
* H1e: per seed ['+0.0544', '+0.0704'], pooled +0.0624, two-level 95% interval [+0.0173, +0.1076]
* H1c: per seed ['+0.1297', '+0.1058'], pooled +0.1178, two-level 95% interval [+0.0542, +0.1882]
* H1ec: per seed ['+0.1169', '+0.0986'], pooled +0.1077, two-level 95% interval [+0.0434, +0.1795]
* mean absolute error [m]: rollout (i) 5.826, present (ii) 6.412, true future (iii) 4.998 (windows [441, 441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356, 356])

* b T2.0 (named): per seed ['-0.406', '-0.497'], pooled -0.452, two-level interval [-0.528, -0.367]
* b T1.0: per seed ['-0.579', '-0.837'], pooled -0.708, two-level interval [-0.879, -0.517]
* b T2.0_lt20: per seed ['-0.419', '-0.456'], pooled -0.437, two-level interval [-0.498, -0.360]
* b T1.0_lt20: per seed ['-0.685', '-0.780'], pooled -0.732, two-level interval [-0.843, -0.592]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
