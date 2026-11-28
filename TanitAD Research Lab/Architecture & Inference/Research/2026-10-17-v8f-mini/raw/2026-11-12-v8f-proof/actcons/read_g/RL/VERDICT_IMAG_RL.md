# VERDICT_IMAG_RL (card PREREG_V8F_PROOF_ADD36)

Tier: T0 (a world-model diagnostic on logged frames, never driving performance). STRATEGIC: PAUSED (PI item 25). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read.

## Controls (first)

* c1: PASS
* c2: PASS
* c4: PASS
* c3: PASS
* c5: PASS

## Names

{"H1": "IMAG-HEAD-ADDS + IMAG-HEAD-ADDS-LEAD", "H2": "IMAG-DEAF", "H2_overshoot_seeds": []}

## H1 (head on imagined states; label = lead gap at t + 2.0 s; seeds [2, 3])

* H1: per seed ['+0.0824', '+0.0626'], pooled +0.0725, two-level 95% interval [+0.0298, +0.1243]
* H1e: per seed ['+0.0694', '+0.0501'], pooled +0.0597, two-level 95% interval [+0.0195, +0.1077]
* H1c: per seed ['+0.1488', '+0.1578'], pooled +0.1533, two-level 95% interval [+0.0920, +0.2230]
* H1ec: per seed ['+0.1344', '+0.1496'], pooled +0.1420, two-level 95% interval [+0.0812, +0.2151]
* mean absolute error [m]: rollout (i) 5.853, present (ii) 6.987, true future (iii) 5.138 (windows [441, 441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356, 356])

* b T2.0 (named): per seed ['-0.854', '-0.649'], pooled -0.751, two-level interval [-0.895, -0.622]
* b T1.0: per seed ['-1.451', '-1.187'], pooled -1.319, two-level interval [-1.519, -1.137]
* b T2.0_lt20: per seed ['-0.789', '-0.564'], pooled -0.677, two-level interval [-0.849, -0.532]
* b T1.0_lt20: per seed ['-1.405', '-1.049'], pooled -1.227, two-level interval [-1.505, -0.993]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
