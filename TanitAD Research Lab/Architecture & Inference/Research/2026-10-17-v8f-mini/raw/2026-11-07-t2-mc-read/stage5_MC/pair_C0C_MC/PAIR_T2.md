# PAIR_T2: C0C - MC (READ)

Card PREREG_T2_ATTRIB (sha256 f7d2cfa1...); script sha256 cbd0fdea284f1747eb3828bb0c6fbf74275c2eae297c149c5751adffd2126874; pinned dominance read sha256 7fd65ffd... (now 7fd65ffd...).
A = C0C (stage 4), B = MC (stage 5). argv: `--root2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/pull_box2_run2/pull2/rollouts --root3 C:/Users/Admin/t2v8f_pulls/stage3/pull_box2/pull3/rollouts --root4 C:/Users/Admin/t2v8f_pulls/stage4/pull_box2/pull4/rollouts --sources2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/SCENE_SOURCES.json --sources3 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage3/SCENE_SOURCES3.json --g1 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/C0C_ID_RESULT.json --result1 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage1/pull_box2/pull/STAGE1_RESULT.json --manifest1 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage1/t2v8f_stage1.tar::pkg/MANIFEST.json --result2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/pull_box2_run2/pull2/STAGE2_RESULT.json --result3 C:/Users/Admin/t2v8f_pulls/stage3/pull_box2/pull3/STAGE3_RESULT.json --result4 C:/Users/Admin/t2v8f_pulls/stage4/pull_box2/pull4/STAGE4_RESULT.json --exclusions2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/EXCLUSIONS2.json --exclusions3 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage3/EXCLUSIONS3.json --exclusions4 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage4/EXCLUSIONS4.json --out-dir E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage5/MC/pair_C0C_MC --a-arm C0C --b-arm MC --b-stage 5 --b-root C:/Users/Admin/t2v8f_pulls/stage5_MC/pull5_MC/rollouts --b-exclusions E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage5/MC/EXCLUSIONS5.json --b-overlay E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage5/MC/t2v8f_stage5_overlay_MC.tar::pkg/MANIFEST5.json --b-gate-log E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/logs/pytest_full_Q.log`
Tier: T2 re-perception (agents replayed at log time; not a reactive-traffic result). Estimator: paired scene-cluster bootstrap, B 2000, seed 0 (answers another draw of SCENES only; the seed floor is the training-run term, read beside).

## Gates

* ATTRIB: PASS 
* G1: PASS 
* G2[C0C@4]: PASS 
* G2[MC@5]: PASS 
* G3[C0C@4]: PASS 
* G3[MC@5]: PASS 
* G4: PASS 

## Excluded scenes (rule R-SCN, the union over every stage read)

* Scenes read on: 48 of 62 (14 excluded). Starts read on: 93 of 121 (lost 28).

| scene | idx | excluded in (stage: reason) | starts lost |
|---|---|---|---|
| dffd4c04 | 9 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:MC: ACTOR_REFUSED | 2 |
| 00169207 | 13 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:MC: ACTOR_REFUSED | 2 |
| 1aa48b8a | 17 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:MC: ACTOR_REFUSED | 2 |
| c67bdfef | 18 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:MC: ACTOR_REFUSED | 2 |
| 90a3d349 | 27 | 2: NO_DYNAMIC_LAYER, 4: NO_DYNAMIC_LAYER, 5:MC: NO_DYNAMIC_LAYER | 2 |
| aa094af8 | 28 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:MC: ACTOR_REFUSED | 2 |
| 42a44314 | 33 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:MC: ACTOR_REFUSED | 2 |
| 829c55af | 42 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:MC: ACTOR_REFUSED | 2 |
| 964e7935 | 43 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:MC: ACTOR_REFUSED | 2 |
| b0ca8b13 | 44 | 3: NO_DYNAMIC_LAYER, 4: NO_DYNAMIC_LAYER, 5:MC: NO_DYNAMIC_LAYER | 2 |
| aace43c1 | 45 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:MC: ACTOR_REFUSED | 2 |
| 359e435d | 46 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:MC: ACTOR_REFUSED | 2 |
| ca2cb16e | 54 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:MC: ACTOR_REFUSED | 2 |
| fbbadc60 | 61 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:MC: ACTOR_REFUSED | 2 |

## C0C - MC, seed 0 (paired over the same starts, per row)

| family | row | A - B | lo | hi | separated | n windows | n scenes | seed floor A | seed floor B |
|---|---|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -0.6659 | -1.0067 | -0.3377 | separated | 93 | 48 | 0.1494 | 0.5089 |
| LONGITUDINAL | speed MAE | -0.4250 | -0.6385 | -0.2166 | separated | 93 | 48 | 0.1170 | 0.1471 |
| LONGITUDINAL | along-track MAE | -0.7109 | -1.0536 | -0.3847 | separated | 93 | 48 | 0.1202 | 0.4390 |
| LONGITUDINAL | accel MAE | -1.0108 | -1.4115 | -0.5960 | separated | 93 | 48 | 0.1958 | 0.0882 |
| LONGITUDINAL | DK-VIOL | +0.0000 | +0.0000 | +0.0000 | not separated | 31 | 25 | 0.0000 | 0.0000 |
| LONGITUDINAL | DK-HW | -0.4326 | -0.9523 | -0.0205 | separated | 39 | 26 | 0.1439 | 0.3057 |
| LONGITUDINAL | DK-TG | -0.1042 | -0.3122 | +0.0035 | not separated | 33 | 25 | 0.0051 | 0.0439 |
| LATERAL | heading MAE | -5.2078 | -12.1177 | +0.9495 | not separated | 93 | 48 | 5.1153 | 5.0588 |
| LATERAL | yaw-rate MAE | -2.8779 | -5.9151 | +0.2184 | not separated | 93 | 48 | 2.9756 | 0.5229 |
| LATERAL | curvature MAE | -0.0234 | -0.0405 | -0.0077 | separated | 93 | 48 | 0.0084 | 0.0271 |
| LATERAL | cross-track MAE | +0.0635 | -0.0764 | +0.2006 | not separated | 93 | 48 | 0.0963 | 0.0282 |
| LATERAL | L1 lateral-accel MAE | -0.3233 | -0.6941 | +0.0565 | not separated | 93 | 48 | 0.2805 | 0.2235 |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0255 | -0.0499 | -0.0042 | separated | 93 | 48 | 0.0138 | 0.0301 |
| TACTICAL | TAC-LAT | +0.0430 | -0.0652 | +0.1398 | not separated | 93 | 48 | 0.0215 | 0.0323 |
| TACTICAL | TAC-LON | +0.0538 | -0.0109 | +0.1277 | not separated | 93 | 48 | 0.0323 | 0.1398 |
| STRATEGIC | S-POS@30 | +0.8651 | -0.2652 | +2.0761 | not separated | 90 | 48 | 1.9151 | 2.4326 |
| STRATEGIC | S-POS@50 | +0.2542 | -1.9391 | +2.5548 | not separated | 73 | 48 | 3.5407 | 3.4137 |
| STRATEGIC | S-POS@80 | -2.0402 | -7.3242 | +3.4067 | not separated | 44 | 48 | 3.4955 | 3.8021 |
| crawl | M1 crawl share | +0.0580 | +0.0011 | +0.1175 | separated | 93 | 48 | 0.0032 | 0.0574 |

## C0C - MC, seed 1 (paired over the same starts, per row)

| family | row | A - B | lo | hi | separated | n windows | n scenes | seed floor A | seed floor B |
|---|---|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -0.3064 | -0.5839 | -0.0346 | separated | 93 | 48 | 0.1494 | 0.5089 |
| LONGITUDINAL | speed MAE | -0.1609 | -0.4404 | +0.0733 | not separated | 93 | 48 | 0.1170 | 0.1471 |
| LONGITUDINAL | along-track MAE | -0.3920 | -0.6525 | -0.1424 | separated | 93 | 48 | 0.1202 | 0.4390 |
| LONGITUDINAL | accel MAE | -1.2948 | -1.6987 | -0.9196 | separated | 93 | 48 | 0.1958 | 0.0882 |
| LONGITUDINAL | DK-VIOL | +0.0000 | +0.0000 | +0.0000 | not separated | 31 | 25 | 0.0000 | 0.0000 |
| LONGITUDINAL | DK-HW | -0.3071 | -0.6500 | -0.0180 | separated | 43 | 26 | 0.1439 | 0.3057 |
| LONGITUDINAL | DK-TG | -0.0553 | -0.1464 | +0.0057 | not separated | 33 | 25 | 0.0051 | 0.0439 |
| LATERAL | heading MAE | -5.2643 | -12.0088 | +1.1768 | not separated | 93 | 48 | 5.1153 | 5.0588 |
| LATERAL | yaw-rate MAE | -5.3307 | -8.4432 | -2.3210 | separated | 93 | 48 | 2.9756 | 0.5229 |
| LATERAL | curvature MAE | -0.0047 | -0.0144 | +0.0051 | not separated | 93 | 48 | 0.0084 | 0.0271 |
| LATERAL | cross-track MAE | -0.0046 | -0.1448 | +0.1490 | not separated | 93 | 48 | 0.0963 | 0.0282 |
| LATERAL | L1 lateral-accel MAE | -0.8273 | -1.2474 | -0.4152 | separated | 93 | 48 | 0.2805 | 0.2235 |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0091 | -0.0197 | +0.0021 | not separated | 93 | 48 | 0.0138 | 0.0301 |
| TACTICAL | TAC-LAT | +0.0538 | -0.0339 | +0.1522 | not separated | 93 | 48 | 0.0215 | 0.0323 |
| TACTICAL | TAC-LON | -0.0538 | -0.1170 | +0.0108 | not separated | 93 | 48 | 0.0323 | 0.1398 |
| STRATEGIC | S-POS@30 | +0.3477 | -1.1266 | +1.8326 | not separated | 90 | 48 | 1.9151 | 2.4326 |
| STRATEGIC | S-POS@50 | +0.2370 | -3.2302 | +3.7999 | not separated | 73 | 48 | 3.5407 | 3.4137 |
| STRATEGIC | S-POS@80 | -2.3469 | -10.6472 | +7.0272 | not separated | 44 | 48 | 3.4955 | 3.8021 |
| crawl | M1 crawl share | +0.1121 | +0.0640 | +0.1609 | separated | 93 | 48 | 0.0032 | 0.0574 |

Seed floor = the dominance card's seed gap of that arm against P10 (|delta_s1 - delta_s0|); `none` = the floor P10 has no seeds. The interval answers another draw of SCENES only; a separated row is necessary, not sufficient, for a lever claim (H-ESTIM-SEED-1).

Disclosure: 62 scenes selected on the human's logs only, launch scenes over-represented (42 of 62); a result here is a result on these scenes at T2. A T2 pair is not a reactive-traffic result.
