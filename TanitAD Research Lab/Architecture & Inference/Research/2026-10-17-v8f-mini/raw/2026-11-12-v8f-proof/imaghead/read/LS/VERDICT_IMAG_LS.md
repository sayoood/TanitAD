# VERDICT_IMAG_LS (card PREREG_V8F_PROOF_ADD36)

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

* H1: per seed ['+0.0702', '+0.0447'], pooled +0.0575, two-level 95% interval [+0.0167, +0.1076]
* H1e: per seed ['+0.0467', '+0.0323'], pooled +0.0395, two-level 95% interval [+0.0003, +0.0852]
* H1c: per seed ['+0.1519', '+0.1065'], pooled +0.1292, two-level 95% interval [+0.0500, +0.2140]
* H1ec: per seed ['+0.1379', '+0.0953'], pooled +0.1166, two-level 95% interval [+0.0384, +0.1985]
* mean absolute error [m]: rollout (i) 6.141, present (ii) 6.826, true future (iii) 5.228 (windows [441, 441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356, 356])

* b T2.0 (named): per seed ['-0.444', '-0.493'], pooled -0.468, two-level interval [-0.530, -0.415]
* b T1.0: per seed ['-0.704', '-0.823'], pooled -0.763, two-level interval [-0.877, -0.659]
* b T2.0_lt20: per seed ['-0.378', '-0.446'], pooled -0.412, two-level interval [-0.495, -0.339]
* b T1.0_lt20: per seed ['-0.619', '-0.786'], pooled -0.702, two-level interval [-0.873, -0.559]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
