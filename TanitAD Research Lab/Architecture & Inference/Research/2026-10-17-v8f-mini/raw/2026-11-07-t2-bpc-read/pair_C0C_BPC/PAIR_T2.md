# PAIR_T2: C0C - BPC (READ)

Card PREREG_T2_ATTRIB (sha256 f7d2cfa1...); script sha256 cbd0fdea284f1747eb3828bb0c6fbf74275c2eae297c149c5751adffd2126874; pinned dominance read sha256 7fd65ffd... (now 7fd65ffd...).
A = C0C (stage 4), B = BPC (stage 5). argv: `--root2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/pull_box2_run2/pull2/rollouts --root3 C:/Users/Admin/t2v8f_pulls/stage3/pull_box2/pull3/rollouts --root4 C:/Users/Admin/t2v8f_pulls/stage4/pull_box2/pull4/rollouts --sources2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/SCENE_SOURCES.json --sources3 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage3/SCENE_SOURCES3.json --g1 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/C0C_ID_RESULT.json --result1 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage1/pull_box2/pull/STAGE1_RESULT.json --manifest1 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage1/t2v8f_stage1.tar::pkg/MANIFEST.json --result2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/pull_box2_run2/pull2/STAGE2_RESULT.json --result3 C:/Users/Admin/t2v8f_pulls/stage3/pull_box2/pull3/STAGE3_RESULT.json --result4 C:/Users/Admin/t2v8f_pulls/stage4/pull_box2/pull4/STAGE4_RESULT.json --exclusions2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/EXCLUSIONS2.json --exclusions3 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage3/EXCLUSIONS3.json --exclusions4 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage4/EXCLUSIONS4.json --out-dir E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage5/BPC/pair_C0C_BPC --a-arm C0C --b-arm BPC --b-stage 5 --b-root C:/Users/Admin/t2v8f_pulls/stage5_BPC/pull5_BPC/rollouts --b-exclusions E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage5/BPC/EXCLUSIONS5.json --b-overlay E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage5/BPC/t2v8f_stage5_overlay_BPC.tar::pkg/MANIFEST5.json --b-gate-log E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/logs/pytest_full_Q.log`
Tier: T2 re-perception (agents replayed at log time; not a reactive-traffic result). Estimator: paired scene-cluster bootstrap, B 2000, seed 0 (answers another draw of SCENES only; the seed floor is the training-run term, read beside).

## Gates

* ATTRIB: PASS 
* G1: PASS 
* G2[BPC@5]: PASS 
* G2[C0C@4]: PASS 
* G3[BPC@5]: PASS 
* G3[C0C@4]: PASS 
* G4: PASS 

## Excluded scenes (rule R-SCN, the union over every stage read)

* Scenes read on: 48 of 62 (14 excluded). Starts read on: 93 of 121 (lost 28).

| scene | idx | excluded in (stage: reason) | starts lost |
|---|---|---|---|
| dffd4c04 | 9 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:BPC: ACTOR_REFUSED | 2 |
| 00169207 | 13 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:BPC: ACTOR_REFUSED | 2 |
| 1aa48b8a | 17 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:BPC: ACTOR_REFUSED | 2 |
| c67bdfef | 18 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:BPC: ACTOR_REFUSED | 2 |
| 90a3d349 | 27 | 2: NO_DYNAMIC_LAYER, 4: NO_DYNAMIC_LAYER, 5:BPC: NO_DYNAMIC_LAYER | 2 |
| aa094af8 | 28 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:BPC: ACTOR_REFUSED | 2 |
| 42a44314 | 33 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:BPC: ACTOR_REFUSED | 2 |
| 829c55af | 42 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:BPC: ACTOR_REFUSED | 2 |
| 964e7935 | 43 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:BPC: ACTOR_REFUSED | 2 |
| b0ca8b13 | 44 | 3: NO_DYNAMIC_LAYER, 4: NO_DYNAMIC_LAYER, 5:BPC: NO_DYNAMIC_LAYER | 2 |
| aace43c1 | 45 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:BPC: ACTOR_REFUSED | 2 |
| 359e435d | 46 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:BPC: ACTOR_REFUSED | 2 |
| ca2cb16e | 54 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:BPC: ACTOR_REFUSED | 2 |
| fbbadc60 | 61 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:BPC: ACTOR_REFUSED | 2 |

## C0C - BPC, seed 0 (paired over the same starts, per row)

| family | row | A - B | lo | hi | separated | n windows | n scenes | seed floor A | seed floor B |
|---|---|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -0.7337 | -1.1010 | -0.3591 | separated | 93 | 48 | 0.1494 | 0.1132 |
| LONGITUDINAL | speed MAE | -0.6445 | -0.8560 | -0.4367 | separated | 93 | 48 | 0.1170 | 0.0931 |
| LONGITUDINAL | along-track MAE | -0.7914 | -1.1469 | -0.4204 | separated | 93 | 48 | 0.1202 | 0.1181 |
| LONGITUDINAL | accel MAE | -0.9121 | -1.2656 | -0.5023 | separated | 93 | 48 | 0.1958 | 0.2463 |
| LONGITUDINAL | DK-VIOL | +0.0000 | +0.0000 | +0.0000 | not separated | 31 | 25 | 0.0000 | 0.0000 |
| LONGITUDINAL | DK-HW | -0.3817 | -1.1736 | +0.1732 | not separated | 40 | 26 | 0.1439 | 0.1030 |
| LONGITUDINAL | DK-TG | -0.1666 | -0.4699 | +0.0147 | not separated | 33 | 25 | 0.0051 | 0.0884 |
| LATERAL | heading MAE | +7.4130 | -0.7808 | +17.7174 | not separated | 93 | 48 | 5.1153 | 10.1081 |
| LATERAL | yaw-rate MAE | -1.1696 | -3.8349 | +1.6724 | not separated | 93 | 48 | 2.9756 | 2.6252 |
| LATERAL | curvature MAE | -0.0268 | -0.0468 | -0.0072 | separated | 93 | 48 | 0.0084 | 0.0198 |
| LATERAL | cross-track MAE | +0.0570 | -0.1056 | +0.2358 | not separated | 93 | 48 | 0.0963 | 0.1823 |
| LATERAL | L1 lateral-accel MAE | -0.0997 | -0.4196 | +0.2667 | not separated | 93 | 48 | 0.2805 | 0.3937 |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0367 | -0.0650 | -0.0118 | separated | 93 | 48 | 0.0138 | 0.0224 |
| TACTICAL | TAC-LAT | +0.0108 | -0.0957 | +0.1170 | not separated | 93 | 48 | 0.0215 | 0.1828 |
| TACTICAL | TAC-LON | +0.1183 | +0.0326 | +0.2105 | separated | 93 | 48 | 0.0323 | 0.1075 |
| STRATEGIC | S-POS@30 | +0.2461 | -1.2260 | +1.8452 | not separated | 90 | 48 | 1.9151 | 0.9780 |
| STRATEGIC | S-POS@50 | -0.8805 | -3.4008 | +1.7011 | not separated | 73 | 48 | 3.5407 | 0.5270 |
| STRATEGIC | S-POS@80 | -1.9770 | -7.7038 | +3.9536 | not separated | 44 | 48 | 3.4955 | 2.4074 |
| crawl | M1 crawl share | +0.0332 | -0.0200 | +0.0822 | not separated | 93 | 48 | 0.0032 | 0.0312 |

## C0C - BPC, seed 1 (paired over the same starts, per row)

| family | row | A - B | lo | hi | separated | n windows | n scenes | seed floor A | seed floor B |
|---|---|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -0.7698 | -1.1783 | -0.3965 | separated | 93 | 48 | 0.1494 | 0.1132 |
| LONGITUDINAL | speed MAE | -0.4344 | -0.8438 | -0.0840 | separated | 93 | 48 | 0.1170 | 0.0931 |
| LONGITUDINAL | along-track MAE | -0.7934 | -1.2323 | -0.4072 | separated | 93 | 48 | 0.1202 | 0.1181 |
| LONGITUDINAL | accel MAE | -1.3542 | -1.7727 | -0.9608 | separated | 93 | 48 | 0.1958 | 0.2463 |
| LONGITUDINAL | DK-VIOL | +0.0000 | +0.0000 | +0.0000 | not separated | 31 | 25 | 0.0000 | 0.0000 |
| LONGITUDINAL | DK-HW | -0.3359 | -0.8184 | -0.0266 | separated | 37 | 26 | 0.1439 | 0.1030 |
| LONGITUDINAL | DK-TG | -0.0731 | -0.2027 | +0.0151 | not separated | 33 | 25 | 0.0051 | 0.0884 |
| LATERAL | heading MAE | -7.8104 | -13.9153 | -1.8847 | separated | 93 | 48 | 5.1153 | 10.1081 |
| LATERAL | yaw-rate MAE | -6.7705 | -9.7320 | -4.1281 | separated | 93 | 48 | 2.9756 | 2.6252 |
| LATERAL | curvature MAE | -0.0154 | -0.0284 | -0.0029 | separated | 93 | 48 | 0.0084 | 0.0198 |
| LATERAL | cross-track MAE | -0.2216 | -0.4272 | -0.0286 | separated | 93 | 48 | 0.0963 | 0.1823 |
| LATERAL | L1 lateral-accel MAE | -0.7738 | -1.1415 | -0.4479 | separated | 93 | 48 | 0.2805 | 0.3937 |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0280 | -0.0442 | -0.0128 | separated | 93 | 48 | 0.0138 | 0.0224 |
| TACTICAL | TAC-LAT | +0.1720 | +0.0652 | +0.2796 | separated | 93 | 48 | 0.0215 | 0.1828 |
| TACTICAL | TAC-LON | +0.0430 | -0.0645 | +0.1522 | not separated | 93 | 48 | 0.0323 | 0.1075 |
| STRATEGIC | S-POS@30 | +1.1833 | -0.7595 | +3.2217 | not separated | 90 | 48 | 1.9151 | 0.9780 |
| STRATEGIC | S-POS@50 | +2.2474 | -1.8546 | +6.5341 | not separated | 74 | 48 | 3.5407 | 0.5270 |
| STRATEGIC | S-POS@80 | -0.8889 | -9.1475 | +7.7265 | not separated | 44 | 48 | 3.4955 | 2.4074 |
| crawl | M1 crawl share | +0.0611 | +0.0105 | +0.1105 | separated | 93 | 48 | 0.0032 | 0.0312 |

Seed floor = the dominance card's seed gap of that arm against P10 (|delta_s1 - delta_s0|); `none` = the floor P10 has no seeds. The interval answers another draw of SCENES only; a separated row is necessary, not sufficient, for a lever claim (H-ESTIM-SEED-1).

Disclosure: 62 scenes selected on the human's logs only, launch scenes over-represented (42 of 62); a result here is a result on these scenes at T2. A T2 pair is not a reactive-traffic result.
