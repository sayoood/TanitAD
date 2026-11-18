# VERDICT_IMAG_RLAm (card PREREG_V8F_PROOF_ADD36)

Tier: T0 (a world-model diagnostic on logged frames, never driving performance). STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read.

## Controls (first)

* c1: PASS
* c2: PASS
* c4: PASS
* c3: PASS
* c5: PASS

## Names

{"H1": "IMAG-HEAD-ADDS + IMAG-HEAD-ADDS-LEAD", "H2": "IMAG-RESPONDS", "H2_overshoot_seeds": []}

## H1 (head on imagined states; label = lead gap at t + 2.0 s; seeds [0, 1])

* H1: per seed ['+0.0255', '+0.0748'], pooled +0.0502, two-level 95% interval [+0.0060, +0.1118]
* H1e: per seed ['+0.0239', '+0.0530'], pooled +0.0385, two-level 95% interval [+0.0010, +0.0865]
* H1c: per seed ['+0.1481', '+0.1280'], pooled +0.1380, two-level 95% interval [+0.0665, +0.2188]
* H1ec: per seed ['+0.1382', '+0.1133'], pooled +0.1257, two-level 95% interval [+0.0560, +0.2045]
* mean absolute error [m]: rollout (i) 5.953, present (ii) 6.506, true future (iii) 4.987 (windows [441, 441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356, 356])

* b T2.0 (named): per seed ['0.805', '0.817'], pooled 0.811, two-level interval [0.759, 0.866]
* b T1.0: per seed ['1.187', '1.200'], pooled 1.194, two-level interval [1.100, 1.286]
* b T2.0_lt20: per seed ['0.723', '0.679'], pooled 0.701, two-level interval [0.612, 0.780]
* b T1.0_lt20: per seed ['1.084', '1.044'], pooled 1.064, two-level interval [0.928, 1.194]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
