# VERDICT LABELUNSAFE10K (card PREREG_V8F_PROOF_ADD135)

Tier: T1-kin/hb0.2 (kinematic rollout on the model's own actions, heading bound 0.2 1/m; not driving performance). Not driving performance. Controls u0 to u4 passed before this file was written (controls file sha256 16955bd05e63724b).

## Names

* Mechanism: **LABELUNSAFE10K-K0-RULE-FOLLOWS-CLOSER-THAN-LOGGED**
* Repeat: **LABELUNSAFE10K-K0-SAME-WINDOWS-ACROSS-SEEDS**

## Measures on the unsafe windows of RLc (pooled over the four model seeds; T1-kin/hb0.2)

| measure | value | 95 % interval | n |
|---|---|---|---|
| M1 unsafe by the time gap alone (no step with gap below 2.0 m) | 0.9518 | [0.8909, 1.0000] | 83 |
| M2 lowest time gap at least 0.9 s | 0.2892 | [0.2087, 0.4615] | 83 |
| M3 logged driver's lowest time gap below 1.2 s | 0.1084 | [0.0000, 0.3865] | 83 |
| M4 lowest model time gap more than 0.2 s below the logged one | 0.9518 | [0.8084, 1.0000] | 83 |
| M5 of distinct windows unsafe at one seed or more, share unsafe at three or four | 0.6129 | [0.3750, 0.7778] | 31 |

## Beside (no name)

* Median of the lowest model time gap: 0.8303 s. M3 over all scored windows: 0.0712 [0.0163, 0.1480], n 2248.
* Unsafe windows per model seed: [22, 19, 22, 20]; scored windows per model seed: [562, 562, 562, 562]; distinct windows unsafe at one seed or more: 31; by number of seeds: {'1': 7, '2': 5, '3': 10, '4': 9}.
* Step index of the first unsafe step: median 24.0, quartiles 19.0 to 27.0, range 5 to 30.
* Label gap 0 to 10 m at the first unsafe step: n 51, median label gap 6.42 m, median model speed 5.41 m/s.
* Label gap 10 to 20 m at the first unsafe step: n 32, median label gap 12.69 m, median model speed 9.46 m/s.
* Label gap 20 to 40 m at the first unsafe step: n 0, median label gap nan m, median model speed nan m/s.
* Label gap 40 to 60 m at the first unsafe step: n 0, median label gap nan m, median model speed nan m/s.
* GEc: M1 0.8796 [0.7464, 0.9608] n 108; M2 0.2407 [0.1281, 0.3889] n 108; M3 0.1574 [0.0000, 0.4429] n 108; M4 0.9722 [0.9167, 1.0000] n 108; M5 0.7714 [0.6154, 0.8889] n 35
* A90: M1 0.9390 [0.8947, 0.9796] n 82; M2 0.2683 [0.1200, 0.3565] n 82; M3 0.1829 [0.0000, 0.5444] n 82; M4 0.9756 [0.9200, 1.0000] n 82; M5 0.6429 [0.5000, 0.8235] n 28
* Unsafe windows of GEc that are also unsafe windows of RLc at the same model seed: pooled 0.6667 (n 108), per seed ['0.6552', '0.5926', '0.7308', '0.6923'].

## Families

* LONGITUDINAL: read here (distance keeping: time gap and gap against the logged driver), T1-kin/hb0.2, not driving performance
* LATERAL: not read by this card (the family reads of these banked runs are in rows 193 and 202); n = 0 here
* TACTICAL: not read by this card (the family reads of these banked runs are in rows 193 and 202); n = 0 here
* STRATEGIC: not read by this card (the family reads of these banked runs are in rows 193 and 202); n = 0 here

## Limits

T1-kin/hb0.2 only, no re-perception, the lead moves as logged; about 83 unsafe windows over four model seeds, so the intervals of the shares are wide; the lines of 0.9 s, 1.2 s and 0.2 s are choices of the card; four model seeds of one line; the label form is a reference with label values and not a candidate. The interval answers what another draw of episodes would say, not what another training run would say.
