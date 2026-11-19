# VERDICT_IMAG_RLFT (card PREREG_V8F_PROOF_ADD36)

Tier: T0 (a world-model diagnostic on logged frames, never driving performance). STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read.

## Controls (first)

* c1: PASS
* c2: PASS
* c4: PASS
* c3: PASS
* c5: PASS

## Names

{"H1": "IMAG-HEAD-ADDS + IMAG-HEAD-ADDS-LEAD", "H2": "IMAG-DEAF", "H2_overshoot_seeds": []}

## H1 (head on imagined states; label = lead gap at t + 2.0 s; seeds [2, 3])

* H1: per seed ['+0.1152', '+0.0421'], pooled +0.0787, two-level 95% interval [+0.0214, +0.1537]
* H1e: per seed ['+0.1032', '+0.0317'], pooled +0.0674, two-level 95% interval [+0.0112, +0.1398]
* H1c: per seed ['+0.1500', '+0.1373'], pooled +0.1437, two-level 95% interval [+0.0773, +0.2168]
* H1ec: per seed ['+0.1372', '+0.1255'], pooled +0.1314, two-level 95% interval [+0.0603, +0.2080]
* mean absolute error [m]: rollout (i) 5.286, present (ii) 6.338, true future (iii) 4.684 (windows [441, 441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356, 356])

* b T2.0 (named): per seed ['-0.774', '-0.718'], pooled -0.746, two-level interval [-0.823, -0.676]
* b T1.0: per seed ['-1.268', '-1.315'], pooled -1.291, two-level interval [-1.401, -1.177]
* b T2.0_lt20: per seed ['-0.725', '-0.629'], pooled -0.677, two-level interval [-0.799, -0.588]
* b T1.0_lt20: per seed ['-1.225', '-1.157'], pooled -1.191, two-level interval [-1.366, -1.056]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
