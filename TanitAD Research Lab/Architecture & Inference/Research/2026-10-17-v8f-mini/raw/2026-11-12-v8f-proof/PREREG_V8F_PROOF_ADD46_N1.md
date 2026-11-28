# Note N1 to card PREREG_V8F_PROOF_ADD46 (INTEG10K-PRE): the banked value of control k5, and how STRATEGIC is reported

Note of the v8F proof study, written by the coordinator fork. Frozen before any tool of the card has run on a checkpoint and before any run of the card (time of the freeze: the line of this note in `TIMES_NOTE.md`; hash in the sidecar file). Tier as in the card.

## 1. Control k5 (the card asks that the banked value and its file are cited before `R0` is read)

INHERITED from the other session (its evaluator output on the dev box, not re-verified here):

* `UE_s0`, stage A: file `F:/TanitAD-scratch/datascale/read3/t1/t1v8f_DS-UE_s0-A_m10_causal_heads.json`, sha256 2755705b23070769a0a476eee9b3ca9b65f2ca2e3b97917e5034be006d2e5ca3; block `intervals.metrics` (full-set pooled mean over 2,399 windows of 141 episodes): `fde_last_m` 9.0714 m, `ade_dense_m` 2.7348 m.
* Reading of k5, fixed here: the JSON that the pinned evaluation tool writes for `R0` of `UE_s0` on this fork's pod reads `fde_last_m` within 0.02 m of 9.0714 and `ade_dense_m` within 0.02 m of 2.7348, both from `intervals.metrics`. The legacy row of the file (a mean of episode means) is not used.
* `UE_s1` has no banked T1 read yet (it is read by the other session after its running arm). k5 for `UE_s1` is evaluated when that file exists, against the same two cells with the same tolerance, on this fork's `R0` dump as it then is (no rerun). Until then no name of the card is given; the differences of `UE_s0` alone are not printed either.

## 2. STRATEGIC (replaces the STRATEGIC sentence of section 3 of the card)

The STRATEGIC family of the other session's reads is a T0 read at the first tick of each window (its route passes on the checkpoint and the evaluation cache); it does not read T1 dumps. A rule that acts at inference on the ego's actions cannot change a first-tick read. The candidate is therefore reported as follows:

* STRATEGIC, T0 at the first tick: the banked read of `UE_s0` and `UE_s1` of the other session, cited with its file, stated as unchanged by construction for an inference-time rule, and backed by one identity control that the other session runs: its strategic reader on the wrapper with the rule on and with the rule off gives equal outputs bit for bit. Without that control the family is reported as the banked read of the model without the rule and nothing is said about the candidate.
* STRATEGIC along the rule-driven rollout (T1): NOT COMPUTED (n = 0; no instrument exists for it and none is pre-registered). No strategic effect of the rule is claimed in either direction.

## 3. What does not change

Everything else of the card: parts, runs, reads E1 and E2, names, the other controls, what follows.
