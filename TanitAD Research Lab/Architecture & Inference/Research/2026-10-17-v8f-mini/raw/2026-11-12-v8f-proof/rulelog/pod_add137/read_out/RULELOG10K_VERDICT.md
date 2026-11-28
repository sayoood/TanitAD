# VERDICT RULELOG10K (card PREREG_V8F_PROOF_ADD137)

Tier: T1-kin/hb0.2 (kinematic rollout on the model's own actions, executor of record v8f_t1_eval_hb.py with the heading bound 0.2 1/m; not driving performance). Not driving performance. Controls w0 to w5 passed before this file was written (controls file sha256 9d56d21f59a24f1a).

## Names

* Split: **RULELOG10K-K0-MIXED**
* Rule state: **RULELOG10K-K0-RULE-ON**
* Braking limit: **RULELOG10K-K0-BRAKING-LIMIT-BINDS**

## Measures on the unsafe windows of RLc (pooled over the four model seeds; T1-kin/hb0.2)

| measure | value | 95 % interval | n |
|---|---|---|---|
| R1 share with the rule on at the tick (unsafe windows) | 1.0000 | [1.0000, 1.0000] | 83 |
| R2 share with sat_exec above 0 at the tick (unsafe windows) | 1.0000 | [1.0000, 1.0000] | 83 |
| R3 pooled share of the loss carried by the forecast (rule-on windows) | nan | [nan, nan] | 83 |
| R4 share of rule-on windows with e_L larger than e_X | 0.8072 | [0.6800, 0.8824] | 83 |

## Beside (no name)

* Medians over the rule-on windows: S -0.129 m, e_L 0.206 m, e_X -0.288 m, i 3.0 steps; over all unsafe windows: S -0.129 m, i 3.0 steps.
* Share of rule-on windows in which the lead's mean speed over the interval lies more than 0.5 m/s below v_L: 0.4096; share with a negative e_X: 0.9880; share with the tick at step 0: 0.0120.
* Unsafe windows per model seed: [22, 19, 22, 20]; rule-on windows: 83 of 83.
* R1 to R4 per model seed (values): {'R1': ['1.0000', '1.0000', '1.0000', '1.0000'], 'R2': ['1.0000', '1.0000', '1.0000', '1.0000'], 'R3': ['nan', 'nan', 'nan', '3.0941'], 'R4': ['0.8636', '0.8421', '0.6364', '0.9000']}.
* GEc (g0 is the camera gap, P is read with the label gap): R1 1.0000 [1.0000, 1.0000] n 108; R2 0.8611 [0.6753, 0.9697] n 108; R3 0.6172 [0.2344, 1.1538] n 108; R4 0.7685 [0.6374, 0.9643] n 108
* GEc third part (g0 minus the label gap at the tick): median 0.780 m, share positive 0.8333; medians over rule-on windows: S 0.964, e_L 0.389, e_X -0.163.

## Families

* LONGITUDINAL: read here (distance keeping: the gap lost against the rule's target split into the lead's forecast and the rollout's execution), T1-kin/hb0.2, not driving performance
* LATERAL: not read by this card (the family reads of these banked runs are in rows 193 and 202); n = 0 here
* TACTICAL: not read by this card (the family reads of these banked runs are in rows 193 and 202); n = 0 here
* STRATEGIC: not read by this card (the family reads of these banked runs are in rows 193 and 202); n = 0 here

## Limits

T1-kin/hb0.2 only, no re-perception, the lead moves as logged; windows of one line over four model seeds (about 83, 31 distinct), so the intervals are wide; the cap is written with the realised speed of the step before, the rule itself uses the planned one; the split reads the last tick before the first unsafe step alone, a loss built up over earlier ticks sits in g_m(k_t); the label form is a reference with label values and not a candidate. The interval answers what another draw of episodes would say, not what another training run would say.
