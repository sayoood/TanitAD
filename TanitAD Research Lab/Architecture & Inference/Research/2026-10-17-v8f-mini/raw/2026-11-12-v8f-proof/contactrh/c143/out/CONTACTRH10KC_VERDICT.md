# CONTACTRH10KC verdict (card PREREG_V8F_PROOF_ADD143)

Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions after a pre-roll of 30 steps, logged camera frames; the lead does not react; not driving performance).

Controls: c0 PASS, c1 PASS, c2 PASS, c3 PASS, c4 PASS, c5 PASS

Name: **CONTACTRH10KC-UNRESOLVED** (no name condition met; X 23, S 89)

Counts: |X| = 23, |S| = 89, |Y| = 5; per seed {"0": {"X": 7, "S": 29, "Y": 1}, "1": {"X": 11, "S": 31, "Y": 1}, "2": {"X": 5, "S": 29, "Y": 3}}

* Q1 misread (mean signed gap error X(RH) minus S(RH), m, positive = reader sees the lead farther than it is): +1.9522 [-0.6747, +5.5142] (n_a=21, n_b=89)
* Q2 presence misses (share X(RH) minus S(RH)): +0.0870 [+0.0000, +0.1463] (n_a=23, n_b=89)
* Q3 of record, own rolled speed at the aged tick minus the rule lead speed (v0 - v_L), S, RH minus UE, paired by window (m/s, positive = RH closes faster): +0.0480 [-0.3579, +0.7652] (n_a=89, n_b=None)
* beside, Q1 where UE makes contact and RH does not (UE reader, Y minus S): +4.9154 [-0.8258, +7.5751] (n_a=5, n_b=89)
* beside, -c_src (identical across the arms by construction, control c5): +0.0000 [+0.0000, +0.0000] (n_a=89, n_b=None)

Estimator: episode-cluster bootstrap, 2000 draws, seed 143, 95 percent percentile interval; it answers what another draw of episodes would say.

Four-family line:
* LONGITUDINAL: read here (where the contacts of RH under readout K are made, reader form, T1-kin/hb0.2)
* LATERAL: not read by this card; n = 0
* TACTICAL: not read by this card; n = 0
* STRATEGIC: not read by this card; n = 0

Limits: AGED states: every window starts after a pre-roll of 30 steps (6 s of the model's own driving); a result here does not transfer by itself to the fresh reader-form cost. T1-kin only: the lead moves as logged, the reader reads logged frames. Contact windows of three model seeds over one evaluation set, so X is small and the intervals are wide. The interval answers what another draw of evaluation episodes would say, not what another training run or another inference run would say. A gap error is defined on windows where the reader reports a lead; a missed lead is counted in Q2 and not in Q1.
