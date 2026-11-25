# CONTACTRH10KB verdict (card PREREG_V8F_PROOF_ADD142)

Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions, logged camera frames; the lead does not react; not driving performance).

Controls: c0 PASS, c1 PASS, c2 PASS, c3 PASS, c4 PASS, c5 PASS

Name: **CONTACTRH10KB-UNRESOLVED** (no name condition met; X 16, S 53)

Counts: |X| = 16, |S| = 53, |Y| = 4; per seed {"0": {"X": 4, "S": 21, "Y": 1}, "1": {"X": 6, "S": 14, "Y": 1}, "2": {"X": 6, "S": 18, "Y": 2}}

* Q1 misread (mean signed gap error X(RH) minus S(RH), m, positive = reader sees the lead farther than it is): +0.0821 [-1.2671, +2.4118] (n_a=16, n_b=53)
* Q2 presence misses (share X(RH) minus S(RH)): +0.0000 [+0.0000, +0.0000] (n_a=16, n_b=53)
* Q3 of record, own rolled speed minus the rule lead speed (v0 - v_L), S, RH minus UE, paired by window (m/s, positive = RH closes faster): +0.5940 [-0.2318, +0.8559] (n_a=53, n_b=None)
* beside, Q1 where UE makes contact and RH does not (UE reader, Y minus S): -1.0193 [-3.4688, +1.5004] (n_a=4, n_b=53)
* beside, -c_src (identical across the arms by construction, control c5): +0.0000 [+0.0000, +0.0000] (n_a=53, n_b=None)

Estimator: episode-cluster bootstrap, 2000 draws, seed 142, 95 percent percentile interval; it answers what another draw of episodes would say.

Four-family line:
* LONGITUDINAL: read here (where the contacts of RH under readout K are made, reader form, T1-kin/hb0.2)
* LATERAL: not read by this card; n = 0
* TACTICAL: not read by this card; n = 0
* STRATEGIC: not read by this card; n = 0

Limits: T1-kin only: the lead moves as logged, the reader reads logged frames. Contact windows of three model seeds over one evaluation set, so X is small and the intervals are wide. The interval answers what another draw of evaluation episodes would say, not what another training run or another inference run would say. A gap error is defined on windows where the reader reports a lead; a missed lead is counted in Q2 and not in Q1.
