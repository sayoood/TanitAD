# VERDICT_IMAG_RLA3 (card PREREG_V8F_PROOF_ADD36)

Tier: T0 (a world-model diagnostic on logged frames, never driving performance). STRATEGIC: PAUSED (PI item 25). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read.

## Controls (first)

* c1: PASS
* c2: PASS
* c4: PASS
* c3: PASS
* c5: PASS

## Names

{"H1": "IMAG-HEAD-FLAT", "H2": "IMAG-RESPONDS", "H2_overshoot_seeds": []}

## H1 (head on imagined states; label = lead gap at t + 2.0 s; seeds [0, 1])

* H1: per seed ['-0.0009', '-0.0134'], pooled -0.0072, two-level 95% interval [-0.0347, +0.0155]
* H1e: per seed ['-0.0160', '-0.0161'], pooled -0.0161, two-level 95% interval [-0.0400, +0.0033]
* H1c: per seed ['+0.1480', '+0.1297'], pooled +0.1389, two-level 95% interval [+0.0673, +0.2200]
* H1ec: per seed ['+0.1386', '+0.1150'], pooled +0.1268, two-level 95% interval [+0.0570, +0.2060]
* mean absolute error [m]: rollout (i) 8.118, present (ii) 6.484, true future (iii) 4.961 (windows [441, 441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356, 356])

* b T2.0 (named): per seed ['1.007', '0.936'], pooled 0.971, two-level interval [0.907, 1.047]
* b T1.0: per seed ['1.187', '1.252'], pooled 1.220, two-level interval [1.134, 1.304]
* b T2.0_lt20: per seed ['0.985', '0.845'], pooled 0.915, two-level interval [0.803, 1.063]
* b T1.0_lt20: per seed ['1.163', '1.231'], pooled 1.197, two-level interval [1.058, 1.310]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
