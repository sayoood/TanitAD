# VERDICT_IMAG_LFTb (card PREREG_V8F_PROOF_ADD36)

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

* H1: per seed ['+0.0704', '+0.0721'], pooled +0.0713, two-level 95% interval [+0.0320, +0.1152]
* H1e: per seed ['+0.0545', '+0.0660'], pooled +0.0602, two-level 95% interval [+0.0169, +0.1038]
* H1c: per seed ['+0.1296', '+0.1087'], pooled +0.1192, two-level 95% interval [+0.0561, +0.1880]
* H1ec: per seed ['+0.1169', '+0.1011'], pooled +0.1090, two-level 95% interval [+0.0449, +0.1803]
* mean absolute error [m]: rollout (i) 5.807, present (ii) 6.384, true future (iii) 4.968 (windows [441, 441])

## H2 (response of the imagined gap to the ego's offset acceleration; windows [356, 356])

* b T2.0 (named): per seed ['-0.415', '-0.491'], pooled -0.453, two-level interval [-0.523, -0.377]
* b T1.0: per seed ['-0.596', '-0.827'], pooled -0.712, two-level interval [-0.871, -0.532]
* b T2.0_lt20: per seed ['-0.424', '-0.452'], pooled -0.438, two-level interval [-0.498, -0.362]
* b T1.0_lt20: per seed ['-0.697', '-0.772'], pooled -0.734, two-level interval [-0.842, -0.599]

Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.
