# VERDICT_IMAG_RLAm (card PREREG_V8F_PROOF_ADD36)

Tier: T0 (a world-model diagnostic on logged frames, never driving performance). STRATEGIC: PAUSED (PI item 25). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read.

## Controls (first)

* c1: PASS
* c2: PASS
* c4: PASS
* c3: PASS
* c5: PASS

## Names

{"H1": "IMAG-HEAD-FLAT", "H2": "IMAG-RESPONDS", "H2_overshoot_seeds": []}

## H1 (head on imagined states; label = lead gap at t + 2.0 s; seeds [0])

* H1: per seed ['+0.0255'], pooled +0.0255, two-level 95% interval [-0.0071, +0.0635]
* H1e: per seed ['+0.0239'], pooled +0.0239, two-level 95% interval [-0.0079, +0.0635]
* H1c: per seed ['+0.1481'], pooled +0.1481, two-level 95% interval [+0.0525, +0.2486]
* H1ec: per seed ['+0.1382'], pooled +0.1382, two-level 95% interval [+0.0454, +0.2374]
* mean absolute error [m]: rollout (i) 5.899, present (ii) 6.272, true future (iii) 4.740 (windows [441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356])

* b T2.0 (named): per seed ['0.805'], pooled 0.805, two-level interval [0.748, 0.862]
* b T1.0: per seed ['1.187'], pooled 1.187, two-level interval [1.090, 1.293]
* b T2.0_lt20: per seed ['0.723'], pooled 0.723, two-level interval [0.638, 0.805]
* b T1.0_lt20: per seed ['1.084'], pooled 1.084, two-level interval [0.949, 1.244]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
