# VERDICT_IMAG_RLFTb (card PREREG_V8F_PROOF_ADD36)

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

* H1: per seed ['+0.0444', '+0.0909'], pooled +0.0676, two-level 95% interval [+0.0154, +0.1318]
* H1e: per seed ['+0.0363', '+0.0670'], pooled +0.0517, two-level 95% interval [+0.0058, +0.1065]
* H1c: per seed ['+0.1482', '+0.1296'], pooled +0.1389, two-level 95% interval [+0.0678, +0.2196]
* H1ec: per seed ['+0.1384', '+0.1151'], pooled +0.1268, two-level 95% interval [+0.0574, +0.2055]
* mean absolute error [m]: rollout (i) 5.510, present (ii) 6.497, true future (iii) 4.989 (windows [441, 441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356, 356])

* b T2.0 (named): per seed ['-0.911', '-0.871'], pooled -0.891, two-level interval [-0.974, -0.811]
* b T1.0: per seed ['-1.611', '-1.545'], pooled -1.578, two-level interval [-1.724, -1.436]
* b T2.0_lt20: per seed ['-0.796', '-0.794'], pooled -0.795, two-level interval [-0.884, -0.703]
* b T1.0_lt20: per seed ['-1.452', '-1.399'], pooled -1.426, two-level interval [-1.618, -1.237]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
