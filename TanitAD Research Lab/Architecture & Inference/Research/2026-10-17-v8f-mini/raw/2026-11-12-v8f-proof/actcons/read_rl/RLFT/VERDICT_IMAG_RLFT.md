# VERDICT_IMAG_RLFT (card PREREG_V8F_PROOF_ADD36)

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

* H1: per seed ['+0.0462', '+0.0920'], pooled +0.0691, two-level 95% interval [+0.0166, +0.1335]
* H1e: per seed ['+0.0377', '+0.0677'], pooled +0.0527, two-level 95% interval [+0.0059, +0.1077]
* H1c: per seed ['+0.1503', '+0.1296'], pooled +0.1399, two-level 95% interval [+0.0688, +0.2208]
* H1ec: per seed ['+0.1405', '+0.1147'], pooled +0.1276, two-level 95% interval [+0.0577, +0.2070]
* mean absolute error [m]: rollout (i) 5.483, present (ii) 6.488, true future (iii) 4.960 (windows [441, 441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356, 356])

* b T2.0 (named): per seed ['-0.911', '-0.878'], pooled -0.895, two-level interval [-0.977, -0.817]
* b T1.0: per seed ['-1.611', '-1.558'], pooled -1.584, two-level interval [-1.727, -1.446]
* b T2.0_lt20: per seed ['-0.796', '-0.799'], pooled -0.797, two-level interval [-0.886, -0.705]
* b T1.0_lt20: per seed ['-1.444', '-1.412'], pooled -1.428, two-level interval [-1.612, -1.245]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
