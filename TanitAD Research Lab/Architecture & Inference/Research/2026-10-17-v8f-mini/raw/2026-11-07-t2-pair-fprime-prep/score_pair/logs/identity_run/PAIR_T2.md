# PAIR_T2: C0C - P10 (READ)

Card PREREG_T2_ATTRIB (sha256 f7d2cfa1...); script sha256 cbd0fdea284f1747eb3828bb0c6fbf74275c2eae297c149c5751adffd2126874; pinned dominance read sha256 7fd65ffd... (now 7fd65ffd...).
A = C0C (stage 4), B = P10 (stage None). argv: `--root2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/pull_box2_run2/pull2/rollouts --root3 C:/Users/Admin/t2v8f_pulls/stage3/pull_box2/pull3/rollouts --root4 C:/Users/Admin/t2v8f_pulls/stage4/pull_box2/pull4/rollouts --sources2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/SCENE_SOURCES.json --sources3 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage3/SCENE_SOURCES3.json --g1 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/C0C_ID_RESULT.json --result1 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage1/pull_box2/pull/STAGE1_RESULT.json --manifest1 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage1/t2v8f_stage1.tar::pkg/MANIFEST.json --result2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/pull_box2_run2/pull2/STAGE2_RESULT.json --result3 C:/Users/Admin/t2v8f_pulls/stage3/pull_box2/pull3/STAGE3_RESULT.json --result4 C:/Users/Admin/t2v8f_pulls/stage4/pull_box2/pull4/STAGE4_RESULT.json --exclusions2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/EXCLUSIONS2.json --exclusions3 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage3/EXCLUSIONS3.json --exclusions4 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage4/EXCLUSIONS4.json --out-dir E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/score_pair/logs/identity_run --a-arm C0C --b-arm P10 --identity-record E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage4/score/DOMINANCE_T2.json`
Tier: T2 re-perception (agents replayed at log time; not a reactive-traffic result). Estimator: paired scene-cluster bootstrap, B 2000, seed 0 (answers another draw of SCENES only; the seed floor is the training-run term, read beside).

## Gates

* ATTRIB: PASS 
* G1: PASS 
* G2: PASS 
* G3: PASS 
* G4: PASS 

## Excluded scenes (rule R-SCN, the union over every stage read)

* Scenes read on: 48 of 62 (14 excluded). Starts read on: 93 of 121 (lost 28).

| scene | idx | excluded in (stage: reason) | starts lost |
|---|---|---|---|
| dffd4c04 | 9 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED | 2 |
| 00169207 | 13 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED | 2 |
| 1aa48b8a | 17 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED | 2 |
| c67bdfef | 18 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED | 2 |
| 90a3d349 | 27 | 2: NO_DYNAMIC_LAYER, 4: NO_DYNAMIC_LAYER | 2 |
| aa094af8 | 28 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED | 2 |
| 42a44314 | 33 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED | 2 |
| 829c55af | 42 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED | 2 |
| 964e7935 | 43 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED | 2 |
| b0ca8b13 | 44 | 3: NO_DYNAMIC_LAYER, 4: NO_DYNAMIC_LAYER | 2 |
| aace43c1 | 45 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED | 2 |
| 359e435d | 46 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED | 2 |
| ca2cb16e | 54 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED | 2 |
| fbbadc60 | 61 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED | 2 |

## C0C - P10, seed 0 (paired over the same starts, per row)

| family | row | A - B | lo | hi | separated | n windows | n scenes | seed floor A | seed floor B |
|---|---|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | +0.7893 | +0.3313 | +1.2386 | separated | 93 | 48 | 0.1494 | none |
| LONGITUDINAL | speed MAE | -0.5660 | -0.9096 | -0.1990 | separated | 93 | 48 | 0.1170 | none |
| LONGITUDINAL | along-track MAE | +1.4164 | +0.8567 | +1.9535 | separated | 93 | 48 | 0.1202 | none |
| LONGITUDINAL | accel MAE | -3.4593 | -4.7769 | -2.1896 | separated | 93 | 48 | 0.1958 | none |
| LONGITUDINAL | DK-VIOL | +0.0000 | +0.0000 | +0.0000 | not separated | 31 | 25 | 0.0000 | none |
| LONGITUDINAL | DK-HW | -0.5740 | -1.3929 | +0.1305 | not separated | 40 | 26 | 0.1439 | none |
| LONGITUDINAL | DK-TG | -0.0111 | -0.3898 | +0.4535 | not separated | 33 | 25 | 0.0051 | none |
| LATERAL | heading MAE | +11.3782 | +4.2750 | +18.9838 | separated | 93 | 48 | 5.1153 | none |
| LATERAL | yaw-rate MAE | -66.2993 | -87.3509 | -45.6714 | separated | 93 | 48 | 2.9756 | none |
| LATERAL | curvature MAE | -0.1772 | -0.2385 | -0.1156 | separated | 93 | 48 | 0.0084 | none |
| LATERAL | cross-track MAE | -1.1891 | -1.5496 | -0.8309 | separated | 93 | 48 | 0.0963 | none |
| LATERAL | L1 lateral-accel MAE | -15.4085 | -20.6528 | -10.4454 | separated | 93 | 48 | 0.2805 | none |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0354 | -0.0677 | -0.0055 | separated | 93 | 48 | 0.0138 | none |
| TACTICAL | TAC-LAT | +0.1075 | -0.0106 | +0.2366 | not separated | 93 | 48 | 0.0215 | none |
| TACTICAL | TAC-LON | -0.2258 | -0.3517 | -0.0968 | separated | 93 | 48 | 0.0323 | none |
| STRATEGIC | S-POS@30 | -2.3356 | -4.3371 | -0.2416 | separated | 90 | 48 | 1.9151 | none |
| STRATEGIC | S-POS@50 | -2.8637 | -6.2061 | +0.7062 | not separated | 74 | 48 | 3.5407 | none |
| STRATEGIC | S-POS@80 | -3.7931 | -10.3190 | +1.9946 | not separated | 44 | 48 | 3.4955 | none |
| crawl | M1 crawl share | +0.4073 | +0.3180 | +0.5002 | separated | 93 | 48 | 0.0032 | none |

## C0C - P10, seed 1 (paired over the same starts, per row)

| family | row | A - B | lo | hi | separated | n windows | n scenes | seed floor A | seed floor B |
|---|---|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | +0.6399 | +0.1070 | +1.1504 | separated | 93 | 48 | 0.1494 | none |
| LONGITUDINAL | speed MAE | -0.4490 | -0.7713 | -0.1455 | separated | 93 | 48 | 0.1170 | none |
| LONGITUDINAL | along-track MAE | +1.2962 | +0.6504 | +1.8949 | separated | 93 | 48 | 0.1202 | none |
| LONGITUDINAL | accel MAE | -3.6551 | -4.9924 | -2.3817 | separated | 93 | 48 | 0.1958 | none |
| LONGITUDINAL | DK-VIOL | +0.0000 | +0.0000 | +0.0000 | not separated | 31 | 25 | 0.0000 | none |
| LONGITUDINAL | DK-HW | -0.7179 | -1.2745 | -0.1974 | separated | 45 | 26 | 0.1439 | none |
| LONGITUDINAL | DK-TG | -0.0060 | -0.3167 | +0.3594 | not separated | 33 | 25 | 0.0051 | none |
| LATERAL | heading MAE | +6.2630 | -0.6923 | +14.2532 | not separated | 93 | 48 | 5.1153 | none |
| LATERAL | yaw-rate MAE | -69.2749 | -90.8690 | -47.5371 | separated | 93 | 48 | 2.9756 | none |
| LATERAL | curvature MAE | -0.1857 | -0.2479 | -0.1249 | separated | 93 | 48 | 0.0084 | none |
| LATERAL | cross-track MAE | -1.2854 | -1.6165 | -0.9377 | separated | 93 | 48 | 0.0963 | none |
| LATERAL | L1 lateral-accel MAE | -15.6890 | -20.9633 | -10.6268 | separated | 93 | 48 | 0.2805 | none |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0491 | -0.0820 | -0.0178 | separated | 93 | 48 | 0.0138 | none |
| TACTICAL | TAC-LAT | +0.0860 | -0.0211 | +0.1915 | not separated | 93 | 48 | 0.0215 | none |
| TACTICAL | TAC-LON | -0.1935 | -0.3085 | -0.0832 | separated | 93 | 48 | 0.0323 | none |
| STRATEGIC | S-POS@30 | -0.4205 | -2.5083 | +1.5886 | not separated | 90 | 48 | 1.9151 | none |
| STRATEGIC | S-POS@50 | +0.6770 | -3.0058 | +4.7153 | not separated | 74 | 48 | 3.5407 | none |
| STRATEGIC | S-POS@80 | -0.2976 | -6.2975 | +5.7409 | not separated | 44 | 48 | 3.4955 | none |
| crawl | M1 crawl share | +0.4041 | +0.3033 | +0.5027 | separated | 93 | 48 | 0.0032 | none |

Seed floor = the dominance card's seed gap of that arm against P10 (|delta_s1 - delta_s0|); `none` = the floor P10 has no seeds. The interval answers another draw of SCENES only; a separated row is necessary, not sufficient, for a lever claim (H-ESTIM-SEED-1).

Disclosure: 62 scenes selected on the human's logs only, launch scenes over-represented (42 of 62); a result here is a result on these scenes at T2. A T2 pair is not a reactive-traffic result.
