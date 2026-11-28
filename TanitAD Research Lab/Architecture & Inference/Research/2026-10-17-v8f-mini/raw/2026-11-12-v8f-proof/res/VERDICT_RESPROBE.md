# VERDICT_RESPROBE (PREREG_V8F_PROOF_ADD1, H-V8F-PROOF-4, theorem T5)

Tier T0: a frozen-feature probe on logged frames. This is a perception diagnostic, not driving performance.
STRATEGIC family: PAUSED (PI decision queue item 25, 2026-09-19); not a pass, not omitted.

MEASURED, RESULT_RESPROBE.json (F:/TanitAD-scratch/v8proof/res/), tier T0: n frames 9321, episodes 139, folds 5, B 2000.

## Card outcome label (headline probe class in brackets)
* [P1] **RES-REFUTED** (P4a-1 False, P4a-2 False, GAP-ABSENT False, REFUTED-bar True)
* [P2] **RES-REFUTED** (P4a-1 False, P4a-2 False, GAP-ABSENT True, REFUTED-bar True)

The P1 label is the primary one (the card lists P1 first); the ego-only control E uses the better of P1/P2 per target.

## Controls
* k1 (shuffled targets) all within bar: False; offenders: {"LEADP|E|P1": 0.4377176960709638, "LEADP|E|P2": 0.4599384121508525, "LEADP|V40|P1": 0.43567027855785745, "LEADP|V160|P1": 0.44161515607100815, "LEADP|V40+E|P1": 0.44512641724738483, "LEADP|V160+E|P1": 0.43464273758401184}
* k2 (synthetic fixed linear function of V40 = random combination of its top-8 global kernel PCs, P1) R2 0.9981, bar 0.95: True
* A control missed its bar: the read above is NOT admissible until the control is explained.
* k1 per-fold-mean variant (secondary, same predictions): worst R2 -0.002, AUROC range [0.485, 0.513] (a pooled metric over folds with different base rates sits below chance by construction when targets are shuffled within folds)

## GAP far-stratum (>= 12.5 m) vision gain over E, P1 (REFUTED bar: V40+E gain >= 0.20)
* V40+E: R2 -0.139, E -0.846, gain +0.707 CI [0.35323623809442756, 1.1782440838047832]
* V160+E: R2 -0.136, E -0.846, gain +0.710 CI [0.3559783018471803, 1.1869234278197411]

## GAP true-gap distribution (lead present within 40 m)
* [0.0,6.3) m: n 274, share 0.110
* [6.3,12.5) m: n 449, share 0.181
* [12.5,25.0) m: n 985, share 0.396
* [25.0,1000000000.0) m: n 777, share 0.313

## Metrics (pooled out-of-fold; R2, AUROC for LEADP)
* GAP: E/P1 0.016; E/P2 -0.111; V40/P1 0.299; V40/P2 0.017; V160/P1 0.316; V160/P2 0.134; V40+E/P1 0.309; V40+E/P2 0.098; V160+E/P1 0.320; V160+E/P2 0.074
* NEAR: E/P1 0.135; E/P2 0.134; V40/P1 0.249; V40/P2 -0.014; V160/P1 0.252; V160/P2 0.113; V40+E/P1 0.284; V40+E/P2 0.076; V160+E/P1 0.286; V160+E/P2 0.121
* COUNT: E/P1 0.102; E/P2 0.071; V40/P1 0.284; V40/P2 0.157; V160/P1 0.267; V160/P2 0.170; V40+E/P1 0.265; V40+E/P2 0.220; V160+E/P1 0.268; V160+E/P2 0.230
* LEADP: E/P1 0.590; E/P2 0.551; V40/P1 0.769; V40/P2 0.657; V160/P1 0.787; V160/P2 0.737; V40+E/P1 0.770; V40+E/P2 0.680; V160+E/P1 0.788; V160+E/P2 0.723

## Paired episode-cluster bootstrap differences (headline P1)
* GAP V40+E minus E: +0.2927 CI [+0.1082, +0.5512]
* GAP V160+E minus E: +0.3038 CI [+0.1166, +0.5642]
* GAP V160+E minus V40+E: +0.0111 CI [-0.0233, +0.0479]
* NEAR V40+E minus E: +0.1491 CI [+0.0220, +0.2661]
* NEAR V160+E minus E: +0.1510 CI [+0.0259, +0.2674]
* NEAR V160+E minus V40+E: +0.0019 CI [-0.0085, +0.0129]
* COUNT V40+E minus E: +0.1637 CI [+0.0320, +0.3689]
* COUNT V160+E minus E: +0.1662 CI [+0.0435, +0.3817]
* COUNT V160+E minus V40+E: +0.0026 CI [-0.0066, +0.0262]
* LEADP V40+E minus E: +0.1801 CI [+0.0899, +0.2673]
* LEADP V160+E minus E: +0.1980 CI [+0.1053, +0.2852]
* LEADP V160+E minus V40+E: +0.0178 CI [+0.0005, +0.0355]

## GAP gain by stratum (P1)
* V40+E minus E: near<12.5 -0.186 [-2.0874622405241516, 2.6055630794106723]; far>=12.5 +0.707 [0.35323623809442756, 1.1782440838047832]; far minus near +0.893 CI [-1.914257006484173, 2.86990946187884]
* V160+E minus E: near<12.5 +0.118 [-1.9397200718939662, 3.4036544075309556]; far>=12.5 +0.710 [0.3559783018471803, 1.1869234278197411]; far minus near +0.591 CI [-2.634275347616354, 2.7111249179334918]

Next lever, committed for every outcome: un-pooled 16x40 tokens and a native-resolution centre crop on the pod (P4a full, P4b, P4c).

Implementation choices the card leaves open (kernel PCA rank 256, ridge alpha by inner split, MLP 256 units, E scaling, P4a-3 'well above' read as CI above 0 and gain >= 0.05, REFUTED needs the far-stratum R2 gain of V40+E over E >= 0.20) are in the docstring of resprobe.py.
