# VERDICT_IMAG_RLAm (card PREREG_V8F_PROOF_ADD36)

Tier: T0 (a world-model diagnostic on logged frames, never driving performance). STRATEGIC: PAUSED (PI item 25). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read.

## Controls (first)

* c1: PASS
* c2: PASS
* c4: PASS
* c3: PASS
* c5: PASS

## Names

{"H1": "IMAG-HEAD-ADDS + IMAG-HEAD-ADDS-EGO-ONLY", "H2": "IMAG-RESPONDS", "H2_overshoot_seeds": []}

## H1 (head on imagined states; label = lead gap at t + 2.0 s; seeds [2, 3])

* H1: per seed ['+0.0943', '+0.0333'], pooled +0.0638, two-level 95% interval [+0.0141, +0.1345]
* H1e: per seed ['+0.0812', '+0.0193'], pooled +0.0502, two-level 95% interval [-0.0004, +0.1155]
* H1c: per seed ['+0.1522', '+0.1374'], pooled +0.1448, two-level 95% interval [+0.0787, +0.2174]
* H1ec: per seed ['+0.1394', '+0.1259'], pooled +0.1327, two-level 95% interval [+0.0640, +0.2088]
* mean absolute error [m]: rollout (i) 5.807, present (ii) 6.331, true future (iii) 4.670 (windows [441, 441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356, 356])

* b T2.0 (named): per seed ['0.835', '0.805'], pooled 0.820, two-level interval [0.766, 0.878]
* b T1.0: per seed ['1.266', '1.245'], pooled 1.255, two-level interval [1.173, 1.340]
* b T2.0_lt20: per seed ['0.694', '0.664'], pooled 0.679, two-level interval [0.605, 0.756]
* b T1.0_lt20: per seed ['1.137', '1.108'], pooled 1.122, two-level interval [1.029, 1.221]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
