# VERDICT_IMAG_RLVb (card PREREG_V8F_PROOF_ADD36)

Tier: T0 (a world-model diagnostic on logged frames, never driving performance). STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read.

## Controls (first)

* c1: PASS
* c2: PASS
* c4: PASS
* c3: PASS
* c5: PASS

## Names

{"H1": "IMAG-HEAD-FLAT", "H2": "IMAG-DEAF", "H2_overshoot_seeds": []}

## H1 (head on imagined states; label = lead gap at t + 2.0 s; seeds [0, 1])

* H1: per seed ['+0.0127', '+0.1053'], pooled +0.0590, two-level 95% interval [-0.0087, +0.1653]
* H1e: per seed ['+0.0064', '+0.0627'], pooled +0.0346, two-level 95% interval [-0.0218, +0.1148]
* H1c: per seed ['+0.1475', '+0.1287'], pooled +0.1381, two-level 95% interval [+0.0671, +0.2177]
* H1ec: per seed ['+0.1380', '+0.1140'], pooled +0.1260, two-level 95% interval [+0.0563, +0.2044]
* mean absolute error [m]: rollout (i) 5.736, present (ii) 6.484, true future (iii) 4.971 (windows [441, 441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356, 356])

* b T2.0 (named): per seed ['-0.358', '-0.448'], pooled -0.403, two-level interval [-0.549, -0.268]
* b T1.0: per seed ['-0.719', '-0.707'], pooled -0.713, two-level interval [-0.970, -0.397]
* b T2.0_lt20: per seed ['-0.441', '-0.673'], pooled -0.557, two-level interval [-0.778, -0.335]
* b T1.0_lt20: per seed ['-1.007', '-1.330'], pooled -1.168, two-level interval [-1.555, -0.760]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
