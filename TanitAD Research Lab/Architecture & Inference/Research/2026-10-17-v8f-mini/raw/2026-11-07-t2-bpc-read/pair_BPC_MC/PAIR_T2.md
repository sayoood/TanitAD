# PAIR_T2: BPC - MC (READ)

Card PREREG_T2_ATTRIB (sha256 f7d2cfa1...); script sha256 cbd0fdea284f1747eb3828bb0c6fbf74275c2eae297c149c5751adffd2126874; pinned dominance read sha256 7fd65ffd... (now 7fd65ffd...).
A = BPC (stage 5), B = MC (stage 5). argv: `--root2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/pull_box2_run2/pull2/rollouts --root3 C:/Users/Admin/t2v8f_pulls/stage3/pull_box2/pull3/rollouts --root4 C:/Users/Admin/t2v8f_pulls/stage4/pull_box2/pull4/rollouts --sources2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/SCENE_SOURCES.json --sources3 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage3/SCENE_SOURCES3.json --g1 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/C0C_ID_RESULT.json --result1 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage1/pull_box2/pull/STAGE1_RESULT.json --manifest1 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage1/t2v8f_stage1.tar::pkg/MANIFEST.json --result2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/pull_box2_run2/pull2/STAGE2_RESULT.json --result3 C:/Users/Admin/t2v8f_pulls/stage3/pull_box2/pull3/STAGE3_RESULT.json --result4 C:/Users/Admin/t2v8f_pulls/stage4/pull_box2/pull4/STAGE4_RESULT.json --exclusions2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/EXCLUSIONS2.json --exclusions3 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage3/EXCLUSIONS3.json --exclusions4 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage4/EXCLUSIONS4.json --out-dir E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage5/BPC/pair_BPC_MC --a-arm BPC --a-stage 5 --a-root C:/Users/Admin/t2v8f_pulls/stage5_BPC/pull5_BPC/rollouts --a-exclusions E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage5/BPC/EXCLUSIONS5.json --a-overlay E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage5/BPC/t2v8f_stage5_overlay_BPC.tar::pkg/MANIFEST5.json --a-gate-log E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/logs/pytest_full_Q.log --b-arm MC --b-stage 5 --b-root C:/Users/Admin/t2v8f_pulls/stage5_MC/pull5_MC/rollouts --b-exclusions E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage5/MC/EXCLUSIONS5.json --b-overlay E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage5/MC/t2v8f_stage5_overlay_MC.tar::pkg/MANIFEST5.json --b-gate-log E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/logs/pytest_full_Q.log`
Tier: T2 re-perception (agents replayed at log time; not a reactive-traffic result). Estimator: paired scene-cluster bootstrap, B 2000, seed 0 (answers another draw of SCENES only; the seed floor is the training-run term, read beside).

## Gates

* ATTRIB: PASS 
* G1: PASS 
* G2[BPC@5]: PASS 
* G2[MC@5]: PASS 
* G3[BPC@5]: PASS 
* G3[MC@5]: PASS 
* G4: PASS 

## Excluded scenes (rule R-SCN, the union over every stage read)

* Scenes read on: 48 of 62 (14 excluded). Starts read on: 93 of 121 (lost 28).

| scene | idx | excluded in (stage: reason) | starts lost |
|---|---|---|---|
| dffd4c04 | 9 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:BPC: ACTOR_REFUSED, 5:MC: ACTOR_REFUSED | 2 |
| 00169207 | 13 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:BPC: ACTOR_REFUSED, 5:MC: ACTOR_REFUSED | 2 |
| 1aa48b8a | 17 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:BPC: ACTOR_REFUSED, 5:MC: ACTOR_REFUSED | 2 |
| c67bdfef | 18 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:BPC: ACTOR_REFUSED, 5:MC: ACTOR_REFUSED | 2 |
| 90a3d349 | 27 | 2: NO_DYNAMIC_LAYER, 4: NO_DYNAMIC_LAYER, 5:BPC: NO_DYNAMIC_LAYER, 5:MC: NO_DYNAMIC_LAYER | 2 |
| aa094af8 | 28 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:BPC: ACTOR_REFUSED, 5:MC: ACTOR_REFUSED | 2 |
| 42a44314 | 33 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:BPC: ACTOR_REFUSED, 5:MC: ACTOR_REFUSED | 2 |
| 829c55af | 42 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:BPC: ACTOR_REFUSED, 5:MC: ACTOR_REFUSED | 2 |
| 964e7935 | 43 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:BPC: ACTOR_REFUSED, 5:MC: ACTOR_REFUSED | 2 |
| b0ca8b13 | 44 | 3: NO_DYNAMIC_LAYER, 4: NO_DYNAMIC_LAYER, 5:BPC: NO_DYNAMIC_LAYER, 5:MC: NO_DYNAMIC_LAYER | 2 |
| aace43c1 | 45 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:BPC: ACTOR_REFUSED, 5:MC: ACTOR_REFUSED | 2 |
| 359e435d | 46 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:BPC: ACTOR_REFUSED, 5:MC: ACTOR_REFUSED | 2 |
| ca2cb16e | 54 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:BPC: ACTOR_REFUSED, 5:MC: ACTOR_REFUSED | 2 |
| fbbadc60 | 61 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:BPC: ACTOR_REFUSED, 5:MC: ACTOR_REFUSED | 2 |

## BPC - MC, seed 0 (paired over the same starts, per row)

| family | row | A - B | lo | hi | separated | n windows | n scenes | seed floor A | seed floor B |
|---|---|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | +0.0678 | -0.2602 | +0.4207 | not separated | 93 | 48 | 0.1132 | 0.5089 |
| LONGITUDINAL | speed MAE | +0.2195 | +0.0634 | +0.3846 | separated | 93 | 48 | 0.0931 | 0.1471 |
| LONGITUDINAL | along-track MAE | +0.0804 | -0.2512 | +0.4179 | not separated | 93 | 48 | 0.1181 | 0.4390 |
| LONGITUDINAL | accel MAE | -0.0987 | -0.4022 | +0.1915 | not separated | 93 | 48 | 0.2463 | 0.0882 |
| LONGITUDINAL | DK-VIOL | +0.0000 | +0.0000 | +0.0000 | not separated | 31 | 25 | 0.0000 | 0.0000 |
| LONGITUDINAL | DK-HW | -0.0390 | -0.5084 | +0.4470 | not separated | 40 | 27 | 0.1030 | 0.3057 |
| LONGITUDINAL | DK-TG | +0.0607 | -0.0193 | +0.1696 | not separated | 34 | 26 | 0.0884 | 0.0439 |
| LATERAL | heading MAE | -12.6207 | -24.2279 | -2.9575 | separated | 93 | 48 | 10.1081 | 5.0588 |
| LATERAL | yaw-rate MAE | -1.7083 | -5.4787 | +1.7432 | not separated | 93 | 48 | 2.6252 | 0.5229 |
| LATERAL | curvature MAE | +0.0035 | -0.0140 | +0.0200 | not separated | 93 | 48 | 0.0198 | 0.0271 |
| LATERAL | cross-track MAE | +0.0065 | -0.1781 | +0.1893 | not separated | 93 | 48 | 0.1823 | 0.0282 |
| LATERAL | L1 lateral-accel MAE | -0.2236 | -0.6065 | +0.1297 | not separated | 93 | 48 | 0.3937 | 0.2235 |
| LATERAL | L2 curvature MAE (GT>=5.2) | +0.0111 | -0.0143 | +0.0355 | not separated | 93 | 48 | 0.0224 | 0.0301 |
| TACTICAL | TAC-LAT | +0.0323 | -0.0851 | +0.1489 | not separated | 93 | 48 | 0.1828 | 0.0323 |
| TACTICAL | TAC-LON | -0.0645 | -0.1685 | +0.0326 | not separated | 93 | 48 | 0.1075 | 0.1398 |
| STRATEGIC | S-POS@30 | +0.6190 | -0.5092 | +1.7688 | not separated | 90 | 48 | 0.9780 | 2.4326 |
| STRATEGIC | S-POS@50 | +1.1347 | -1.0031 | +3.3714 | not separated | 73 | 48 | 0.5270 | 3.4137 |
| STRATEGIC | S-POS@80 | -0.0633 | -5.1585 | +4.9897 | not separated | 44 | 48 | 2.4074 | 3.8021 |
| crawl | M1 crawl share | +0.0248 | -0.0267 | +0.0786 | not separated | 93 | 48 | 0.0312 | 0.0574 |

## BPC - MC, seed 1 (paired over the same starts, per row)

| family | row | A - B | lo | hi | separated | n windows | n scenes | seed floor A | seed floor B |
|---|---|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | +0.4634 | +0.1522 | +0.7949 | separated | 93 | 48 | 0.1132 | 0.5089 |
| LONGITUDINAL | speed MAE | +0.2735 | +0.1108 | +0.4513 | separated | 93 | 48 | 0.0931 | 0.1471 |
| LONGITUDINAL | along-track MAE | +0.4014 | +0.0954 | +0.7449 | separated | 93 | 48 | 0.1181 | 0.4390 |
| LONGITUDINAL | accel MAE | +0.0594 | -0.2075 | +0.3362 | not separated | 93 | 48 | 0.2463 | 0.0882 |
| LONGITUDINAL | DK-VIOL | +0.0000 | +0.0000 | +0.0000 | not separated | 31 | 25 | 0.0000 | 0.0000 |
| LONGITUDINAL | DK-HW | +0.1347 | -0.2345 | +0.6748 | not separated | 37 | 26 | 0.1030 | 0.3057 |
| LONGITUDINAL | DK-TG | +0.0179 | -0.1213 | +0.1660 | not separated | 33 | 25 | 0.0884 | 0.0439 |
| LATERAL | heading MAE | +2.5462 | -1.6309 | +7.6735 | not separated | 93 | 48 | 10.1081 | 5.0588 |
| LATERAL | yaw-rate MAE | +1.4398 | -0.6096 | +3.5895 | not separated | 93 | 48 | 2.6252 | 0.5229 |
| LATERAL | curvature MAE | +0.0107 | +0.0006 | +0.0215 | separated | 93 | 48 | 0.0198 | 0.0271 |
| LATERAL | cross-track MAE | +0.2170 | +0.0280 | +0.4313 | separated | 93 | 48 | 0.1823 | 0.0282 |
| LATERAL | L1 lateral-accel MAE | -0.0534 | -0.3256 | +0.2164 | not separated | 93 | 48 | 0.3937 | 0.2235 |
| LATERAL | L2 curvature MAE (GT>=5.2) | +0.0189 | +0.0046 | +0.0334 | separated | 93 | 48 | 0.0224 | 0.0301 |
| TACTICAL | TAC-LAT | -0.1183 | -0.2151 | -0.0213 | separated | 93 | 48 | 0.1828 | 0.0323 |
| TACTICAL | TAC-LON | -0.0968 | -0.2043 | +0.0000 | not separated | 93 | 48 | 0.1075 | 0.1398 |
| STRATEGIC | S-POS@30 | -0.8356 | -2.2109 | +0.5099 | not separated | 90 | 48 | 0.9780 | 2.4326 |
| STRATEGIC | S-POS@50 | -1.9875 | -4.6592 | +0.6757 | not separated | 73 | 48 | 0.5270 | 3.4137 |
| STRATEGIC | S-POS@80 | -1.4579 | -8.6317 | +5.1553 | not separated | 44 | 48 | 2.4074 | 3.8021 |
| crawl | M1 crawl share | +0.0510 | +0.0095 | +0.0960 | separated | 93 | 48 | 0.0312 | 0.0574 |

Seed floor = the dominance card's seed gap of that arm against P10 (|delta_s1 - delta_s0|); `none` = the floor P10 has no seeds. The interval answers another draw of SCENES only; a separated row is necessary, not sufficient, for a lever claim (H-ESTIM-SEED-1).

Disclosure: 62 scenes selected on the human's logs only, launch scenes over-represented (42 of 62); a result here is a result on these scenes at T2. A T2 pair is not a reactive-traffic result.
