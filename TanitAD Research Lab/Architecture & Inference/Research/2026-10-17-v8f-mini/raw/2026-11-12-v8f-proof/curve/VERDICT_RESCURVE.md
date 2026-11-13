# VERDICT_RESCURVE (PREREG_V8F_PROOF_ADD7)

Tier T0: a frozen-feature probe on logged frames, a perception diagnostic, not driving performance. STRATEGIC family: PAUSED (PI item 25), not applicable.
MEASURED, RESULT_RESCURVE.json, n frames 9321, episodes 139, folds 5, B 2000, smoke 0. Estimator: pooled out-of-fold R2 (AUROC for LEADP); episode-cluster bootstrap over the held-out episodes, paired draws; closed-form probes, no training seed.

## Controls
* c1 (banked RESPROBE P1 reproduced to 1e-6 at rank 256): PASS, max abs diff 0.00e+00
* c2 (shuffled targets, rank 256): PASS; offenders {}
* c2 full kernel (shuffled targets, V+E): PASS; offenders {}
* c3 (random draw through the SELECT path inside the 20-draw range): PASS; max identity diff 0.0e+00; extra independent draw inside range in 15/16 cells (informational, chance outside is 2/21 per cell)

## A. RANK: **RANK-HIDES** (bar: V160+E minus V40+E on GAP >= +0.03 R2 with the 95 % interval excluding 0, at rank 1,024 or the full kernel)
* GAP r1024: +0.1590 CI [+0.0004, +0.3693]: PASS
* GAP full: +0.0755 CI [+0.0012, +0.1704]: PASS

Metric by rank (V40+E | V160+E | difference with 95 % CI); R2, AUROC for LEADP
* GAP (E floor +0.016), per-target label RANK-HIDES:
    * r16: 0.112 | 0.138 | +0.0267 [-0.0097, +0.0597]
    * r64: 0.227 | 0.218 | -0.0084 [-0.0443, +0.0286]
    * r256: 0.309 | 0.320 | +0.0111 [-0.0233, +0.0479]
    * r1024: 0.172 | 0.331 | +0.1590 [+0.0004, +0.3693]
    * full: 0.276 | 0.351 | +0.0755 [+0.0012, +0.1704]
* NEAR (E floor +0.135), per-target label RANK-FLAT:
    * r16: 0.273 | 0.274 | +0.0012 [-0.0103, +0.0140]
    * r64: 0.256 | 0.251 | -0.0056 [-0.0260, +0.0094]
    * r256: 0.284 | 0.286 | +0.0019 [-0.0085, +0.0129]
    * r1024: 0.271 | 0.272 | +0.0012 [-0.0071, +0.0101]
    * full: 0.273 | 0.276 | +0.0029 [-0.0104, +0.0175]
* COUNT (E floor +0.102), per-target label RANK-FLAT:
    * r16: 0.232 | 0.222 | -0.0100 [-0.0248, +0.0038]
    * r64: 0.234 | 0.257 | +0.0229 [+0.0010, +0.0791]
    * r256: 0.265 | 0.268 | +0.0026 [-0.0066, +0.0262]
    * r1024: 0.286 | 0.293 | +0.0068 [-0.0032, +0.0293]
    * full: 0.279 | 0.283 | +0.0044 [-0.0061, +0.0312]
* LEADP (E floor +0.590), per-target label RANK-HIDES:
    * r16: 0.741 | 0.726 | -0.0158 [-0.0329, +0.0007]
    * r64: 0.778 | 0.792 | +0.0141 [+0.0018, +0.0286]
    * r256: 0.770 | 0.788 | +0.0178 [+0.0005, +0.0355]
    * r1024: 0.784 | 0.821 | +0.0364 [+0.0100, +0.0663]
    * full: 0.795 | 0.838 | +0.0429 [+0.0221, +0.0660]

Rank the inner split prefers (smallest mean normalised inner-validation MSE over the 5 folds): V40|GAP r64; V40|NEAR r1024; V40|COUNT r64; V40|LEADP full; V160|GAP r64; V160|NEAR r1024; V160|COUNT r256; V160|LEADP full

## B. DATA (bar: last doubling 1/2 -> 1 gains >= 0.03 with the interval excluding 0 = DATA-LIMITED; upper end below 0.03 = SATURATED; else UNDECIDED)
* GAP V40+E r256: gain +0.0672 CI [-0.0325, +0.1638] **UNDECIDED** (bars: >= 0.03 and lo > 0: FAIL; hi < 0.03: FAIL)
* GAP V40+E pref=r64: gain +0.0416 CI [-0.0820, +0.1570] **UNDECIDED** (bars: >= 0.03 and lo > 0: FAIL; hi < 0.03: FAIL)
* GAP E floor (with the V40 folds): gain -0.0785 CI [-0.2714, +0.0754] UNDECIDED (informational)
* GAP V160+E r256: gain +0.1265 CI [-0.0039, +0.2729] **UNDECIDED** (bars: >= 0.03 and lo > 0: FAIL; hi < 0.03: FAIL)
* GAP V160+E pref=r64: gain +0.0658 CI [-0.0593, +0.1913] **UNDECIDED** (bars: >= 0.03 and lo > 0: FAIL; hi < 0.03: FAIL)
* GAP E floor (with the V160 folds): gain -0.0785 CI [-0.2714, +0.0754] UNDECIDED (informational)
* NEAR V40+E r256: gain +0.0356 CI [-0.0036, +0.0673] **UNDECIDED** (bars: >= 0.03 and lo > 0: FAIL; hi < 0.03: FAIL)
* NEAR V40+E pref=r1024: gain +0.0279 CI [-0.0050, +0.0670] **UNDECIDED** (bars: >= 0.03 and lo > 0: FAIL; hi < 0.03: FAIL)
* NEAR E floor (with the V40 folds): gain +0.0511 CI [+0.0156, +0.0893] DATA-LIMITED (informational)
* NEAR V160+E r256: gain +0.0413 CI [+0.0042, +0.0758] **DATA-LIMITED** (bars: >= 0.03 and lo > 0: PASS; hi < 0.03: FAIL)
* NEAR V160+E pref=r1024: gain +0.0275 CI [-0.0059, +0.0636] **UNDECIDED** (bars: >= 0.03 and lo > 0: FAIL; hi < 0.03: FAIL)
* NEAR E floor (with the V160 folds): gain +0.0511 CI [+0.0156, +0.0893] DATA-LIMITED (informational)
* COUNT V40+E r256: gain +0.0416 CI [-0.0256, +0.0945] **UNDECIDED** (bars: >= 0.03 and lo > 0: FAIL; hi < 0.03: FAIL)
* COUNT V40+E pref=r64: gain +0.0263 CI [-0.0472, +0.0761] **UNDECIDED** (bars: >= 0.03 and lo > 0: FAIL; hi < 0.03: FAIL)
* COUNT E floor (with the V40 folds): gain +0.0837 CI [+0.0498, +0.1374] DATA-LIMITED (informational)
* COUNT V160+E r256: gain +0.0494 CI [-0.0045, +0.1113] **UNDECIDED** (bars: >= 0.03 and lo > 0: FAIL; hi < 0.03: FAIL)
* COUNT V160+E pref=r256: gain +0.0494 CI [-0.0045, +0.1113] **UNDECIDED** (bars: >= 0.03 and lo > 0: FAIL; hi < 0.03: FAIL)
* COUNT E floor (with the V160 folds): gain +0.0837 CI [+0.0498, +0.1374] DATA-LIMITED (informational)
* LEADP V40+E r256: gain +0.0217 CI [-0.0012, +0.0455] **UNDECIDED** (bars: >= 0.03 and lo > 0: FAIL; hi < 0.03: FAIL)
* LEADP V40+E pref=full: gain +0.0425 CI [+0.0112, +0.0710] **DATA-LIMITED** (bars: >= 0.03 and lo > 0: PASS; hi < 0.03: FAIL)
* LEADP E floor (with the V40 folds): gain +0.0068 CI [-0.0254, +0.0400] UNDECIDED (informational)
* LEADP V160+E r256: gain +0.0215 CI [-0.0052, +0.0487] **UNDECIDED** (bars: >= 0.03 and lo > 0: FAIL; hi < 0.03: FAIL)
* LEADP V160+E pref=full: gain +0.0574 CI [+0.0302, +0.0846] **DATA-LIMITED** (bars: >= 0.03 and lo > 0: PASS; hi < 0.03: FAIL)
* LEADP E floor (with the V160 folds): gain +0.0068 CI [-0.0254, +0.0400] UNDECIDED (informational)

Learning curve: metric (mean over draws, sd) against training episodes (mean over folds; in brackets those with a valid target)
* GAP V40+E r256: 14(6) ep: -0.273+-0.286 | 28(11) ep: 0.070+-0.054 | 56(22) ep: 0.242+-0.114 | 111(45) ep: 0.309+-0.000
* GAP V40+E pref=r64: 14(6) ep: -0.417+-0.421 | 28(11) ep: -0.040+-0.128 | 56(22) ep: 0.185+-0.085 | 111(45) ep: 0.227+-0.000
* GAP V40+E E: 14(6) ep: -16.263+-19.440 | 28(11) ep: -0.728+-1.459 | 56(22) ep: 0.095+-0.085 | 111(45) ep: 0.016+-0.000
* GAP V160+E r256: 14(6) ep: -0.281+-0.276 | 28(11) ep: 0.029+-0.062 | 56(22) ep: 0.193+-0.180 | 111(45) ep: 0.320+-0.000
* GAP V160+E pref=r64: 14(6) ep: -1.461+-2.512 | 28(11) ep: -0.249+-0.267 | 56(22) ep: 0.152+-0.139 | 111(45) ep: 0.218+-0.000
* GAP V160+E E: 14(6) ep: -16.263+-19.440 | 28(11) ep: -0.728+-1.459 | 56(22) ep: 0.095+-0.085 | 111(45) ep: 0.016+-0.000
* NEAR V40+E r256: 14(14) ep: 0.022+-0.102 | 28(28) ep: 0.166+-0.053 | 56(56) ep: 0.249+-0.045 | 111(111) ep: 0.284+-0.000
* NEAR V40+E pref=r1024: 14(14) ep: -0.026+-0.071 | 28(28) ep: 0.160+-0.050 | 56(56) ep: 0.243+-0.057 | 111(111) ep: 0.271+-0.000
* NEAR V40+E E: 14(14) ep: -0.836+-1.457 | 28(28) ep: -0.028+-0.095 | 56(56) ep: 0.084+-0.045 | 111(111) ep: 0.135+-0.000
* NEAR V160+E r256: 14(14) ep: 0.000+-0.069 | 28(28) ep: 0.183+-0.039 | 56(56) ep: 0.245+-0.034 | 111(111) ep: 0.286+-0.000
* NEAR V160+E pref=r1024: 14(14) ep: -3.030+-6.048 | 28(28) ep: 0.171+-0.036 | 56(56) ep: 0.244+-0.053 | 111(111) ep: 0.272+-0.000
* NEAR V160+E E: 14(14) ep: -0.836+-1.457 | 28(28) ep: -0.028+-0.095 | 56(56) ep: 0.084+-0.045 | 111(111) ep: 0.135+-0.000
* COUNT V40+E r256: 14(14) ep: 0.091+-0.109 | 28(28) ep: 0.115+-0.096 | 56(56) ep: 0.224+-0.017 | 111(111) ep: 0.265+-0.000
* COUNT V40+E pref=r64: 14(14) ep: 0.068+-0.085 | 28(28) ep: 0.123+-0.088 | 56(56) ep: 0.208+-0.035 | 111(111) ep: 0.234+-0.000
* COUNT V40+E E: 14(14) ep: -0.071+-0.082 | 28(28) ep: -0.014+-0.047 | 56(56) ep: 0.018+-0.038 | 111(111) ep: 0.102+-0.000
* COUNT V160+E r256: 14(14) ep: 0.087+-0.107 | 28(28) ep: 0.120+-0.097 | 56(56) ep: 0.218+-0.021 | 111(111) ep: 0.268+-0.000
* COUNT V160+E pref=r256: 14(14) ep: 0.087+-0.107 | 28(28) ep: 0.120+-0.097 | 56(56) ep: 0.218+-0.021 | 111(111) ep: 0.268+-0.000
* COUNT V160+E E: 14(14) ep: -0.071+-0.082 | 28(28) ep: -0.014+-0.047 | 56(56) ep: 0.018+-0.038 | 111(111) ep: 0.102+-0.000
* LEADP V40+E r256: 14(14) ep: 0.689+-0.018 | 28(28) ep: 0.702+-0.028 | 56(56) ep: 0.748+-0.020 | 111(111) ep: 0.770+-0.000
* LEADP V40+E pref=full: 14(14) ep: 0.689+-0.029 | 28(28) ep: 0.709+-0.017 | 56(56) ep: 0.753+-0.020 | 111(111) ep: 0.795+-0.000
* LEADP V40+E E: 14(14) ep: 0.554+-0.038 | 28(28) ep: 0.573+-0.024 | 56(56) ep: 0.583+-0.033 | 111(111) ep: 0.590+-0.000
* LEADP V160+E r256: 14(14) ep: 0.693+-0.022 | 28(28) ep: 0.713+-0.028 | 56(56) ep: 0.766+-0.017 | 111(111) ep: 0.788+-0.000
* LEADP V160+E pref=full: 14(14) ep: 0.695+-0.027 | 28(28) ep: 0.729+-0.020 | 56(56) ep: 0.781+-0.017 | 111(111) ep: 0.838+-0.000
* LEADP V160+E E: 14(14) ep: 0.554+-0.038 | 28(28) ep: 0.573+-0.024 | 56(56) ep: 0.583+-0.033 | 111(111) ep: 0.590+-0.000

## C. SELECT (T8): rules above the best / below the worst of the random draws at BOTH budgets
* GAP: **SELECT-FLAT**
    * V40+E budget 0.25: kcentre -0.159, leverage 0.079, random min/mean/max -1.351/-0.091/0.256
    * V40+E budget 0.5: kcentre 0.340, leverage 0.294, random min/mean/max -0.547/0.142/0.357
    * V40+E: SELECT-FLAT (kcentre helps False, hurts False; leverage helps False, hurts False; mixed False)
    * V160+E budget 0.25: kcentre 0.037, leverage 0.092, random min/mean/max -1.153/-0.091/0.277
    * V160+E budget 0.5: kcentre 0.275, leverage 0.325, random min/mean/max -0.220/0.162/0.361
    * V160+E: SELECT-FLAT (kcentre helps False, hurts False; leverage helps False, hurts False; mixed False)
* NEAR: **SELECT-FLAT**
    * V40+E budget 0.25: kcentre 0.177, leverage 0.056, random min/mean/max -0.001/0.153/0.270
    * V40+E budget 0.5: kcentre 0.197, leverage 0.072, random min/mean/max 0.067/0.213/0.272
    * V40+E: SELECT-FLAT (kcentre helps False, hurts False; leverage helps False, hurts False; mixed False)
    * V160+E budget 0.25: kcentre 0.200, leverage 0.064, random min/mean/max 0.038/0.151/0.248
    * V160+E budget 0.5: kcentre 0.249, leverage 0.155, random min/mean/max 0.071/0.213/0.288
    * V160+E: SELECT-FLAT (kcentre helps False, hurts False; leverage helps False, hurts False; mixed False)
* COUNT: **SELECT-FLAT**
    * V40+E budget 0.25: kcentre 0.158, leverage -0.209, random min/mean/max -0.074/0.178/0.304
    * V40+E budget 0.5: kcentre 0.145, leverage 0.196, random min/mean/max 0.169/0.259/0.365
    * V40+E: SELECT-FLAT (kcentre helps False, hurts False; leverage helps False, hurts False; mixed False)
    * V160+E budget 0.25: kcentre 0.022, leverage -0.195, random min/mean/max -0.021/0.178/0.280
    * V160+E budget 0.5: kcentre 0.251, leverage 0.239, random min/mean/max 0.174/0.252/0.344
    * V160+E: SELECT-FLAT (kcentre helps False, hurts False; leverage helps False, hurts False; mixed False)
* LEADP: **SELECT-FLAT**
    * V40+E budget 0.25: kcentre 0.736, leverage 0.630, random min/mean/max 0.628/0.681/0.755
    * V40+E budget 0.5: kcentre 0.775, leverage 0.723, random min/mean/max 0.687/0.732/0.787
    * V40+E: SELECT-FLAT (kcentre helps False, hurts False; leverage helps False, hurts False; mixed False)
    * V160+E budget 0.25: kcentre 0.725, leverage 0.636, random min/mean/max 0.635/0.688/0.757
    * V160+E budget 0.5: kcentre 0.774, leverage 0.728, random min/mean/max 0.707/0.746/0.802
    * V160+E: SELECT-FLAT (kcentre helps False, hurts False; leverage helps False, hurts False; mixed False)

## What each outcome commits (card)
* RANK-HIDES: the RESPROBE grid comparison was rank-limited; the finer grid stays a live lever. RANK-FLAT with DATA-LIMITED: more labelled clips before a finer grid. RANK-FLAT with SATURATED: the content is not linearly present at either pooled grid; un-pooled tokens and the native-resolution crop on the pod are the remaining test.
* SELECT-HELPS: carried to the choice of training clips (own card, trained arm). SELECT-FLAT: no selection claim from T8 for this representation.

## Limits (card)
Probe-level statements about a frozen representation, 139 episodes. A null from these probes is not a null about what a trained reader can learn. The interval answers another draw of episodes; closed-form probes have no training seed. Deviations R1 onward are in CURVE_DEFS.md and are part of this read.
