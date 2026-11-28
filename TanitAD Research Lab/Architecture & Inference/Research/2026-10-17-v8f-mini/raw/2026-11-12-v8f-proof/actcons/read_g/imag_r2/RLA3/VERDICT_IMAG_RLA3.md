# VERDICT_IMAG_RLA3 (card PREREG_V8F_PROOF_ADD36)

Tier: T0 (a world-model diagnostic on logged frames, never driving performance). STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read.

## Controls (first)

* c1: PASS
* c2: PASS
* c4: PASS
* c3: PASS
* c5: PASS

## Names

{"H1": "IMAG-HEAD-FLAT", "H2": "IMAG-RESPONDS", "H2_overshoot_seeds": []}

## H1 (head on imagined states; label = lead gap at t + 2.0 s; seeds [2, 3])

* H1: per seed ['-0.0465', '+0.0098'], pooled -0.0184, two-level 95% interval [-0.1006, +0.0509]
* H1e: per seed ['-0.0419', '+0.0088'], pooled -0.0165, two-level 95% interval [-0.0951, +0.0496]
* H1c: per seed ['+0.1520', '+0.1380'], pooled +0.1450, two-level 95% interval [+0.0798, +0.2176]
* H1ec: per seed ['+0.1390', '+0.1264'], pooled +0.1327, two-level 95% interval [+0.0632, +0.2091]
* mean absolute error [m]: rollout (i) 9.007, present (ii) 6.307, true future (iii) 4.656 (windows [441, 441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356, 356])

* b T2.0 (named): per seed ['0.900', '0.876'], pooled 0.888, two-level interval [0.835, 0.950]
* b T1.0: per seed ['1.104', '1.195'], pooled 1.150, two-level interval [1.014, 1.260]
* b T2.0_lt20: per seed ['0.902', '0.791'], pooled 0.847, two-level interval [0.750, 0.983]
* b T1.0_lt20: per seed ['1.067', '1.198'], pooled 1.133, two-level interval [0.947, 1.253]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
