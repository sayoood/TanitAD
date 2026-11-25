# CONTACTRH10K verdict (card PREREG_V8F_PROOF_ADD141)

Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions, logged camera frames; the lead does not react; not driving performance).

Controls: c1 PASS, c2 PASS, c3 PASS, c4 PASS

Name: **CONTACTRH10K-UNRESOLVED** (no name condition met; |X| = 19)

Counts: |X| = 19, |S| = 51, |Y| = 5; per seed {"0": {"X": 6, "S": 18, "Y": 2}, "1": {"X": 5, "S": 17, "Y": 1}, "2": {"X": 8, "S": 16, "Y": 2}}

* Q1 misread (mean signed gap error X(RH) minus S(RH), m, positive = reader sees the lead farther than it is): +1.1896 [-0.2609, +4.0802] (n_a=19, n_b=51)
* Q2 presence misses (share X(RH) minus S(RH)): +0.0000 [+0.0000, +0.0000] (n_a=19, n_b=51)
* Q3 closing rate S, RH minus UE, paired by window (m/s, positive = RH closes faster; -c_src): +0.0000 [+0.0000, +0.0000] (n_a=51, n_b=None)
* beside, Q1 where UE makes contact and RH does not (UE reader, Y minus S): -1.1966 [-2.4311, +0.3925] (n_a=5, n_b=51)
* beside, Q3 with ego speed minus the rule lead speed: +0.5911 [+0.1979, +0.8306] (n_a=51, n_b=None)

Estimator: episode-cluster bootstrap, 2000 draws, seed 141, 95 percent percentile interval; it answers what another draw of episodes would say.

Four-family line:
* LONGITUDINAL: read here (where the contacts of RH under readout K are made, reader form, T1-kin/hb0.2)
* LATERAL: not read by this card; n = 0
* TACTICAL: not read by this card; n = 0
* STRATEGIC: not read by this card; n = 0

Limits: T1-kin only: the lead moves as logged, the reader reads logged frames. Contact windows of three model seeds over one evaluation set, so X is small and the intervals are wide. The interval answers what another draw of evaluation episodes would say, not what another training run or another inference run would say. A gap error is defined on windows where the reader reports a lead; a missed lead is counted in Q2 and not in Q1.
