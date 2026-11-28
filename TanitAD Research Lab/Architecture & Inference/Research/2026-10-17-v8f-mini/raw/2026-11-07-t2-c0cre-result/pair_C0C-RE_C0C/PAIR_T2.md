# PAIR_T2: C0C-RE - C0C (READ)

Card PREREG_T2_ATTRIB (sha256 f7d2cfa1...); script sha256 cbd0fdea284f1747eb3828bb0c6fbf74275c2eae297c149c5751adffd2126874; pinned dominance read sha256 7fd65ffd... (now 7fd65ffd...).
A = C0C-RE (stage 5), B = C0C (stage 4). argv: `--root2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/pull_box2_run2/pull2/rollouts --root3 C:/Users/Admin/t2v8f_pulls/stage3/pull_box2/pull3/rollouts --root4 C:/Users/Admin/t2v8f_pulls/stage4/pull_box2/pull4/rollouts --sources2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/SCENE_SOURCES.json --sources3 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage3/SCENE_SOURCES3.json --g1 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/C0C_ID_RESULT.json --result1 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage1/pull_box2/pull/STAGE1_RESULT.json --manifest1 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage1/t2v8f_stage1.tar::pkg/MANIFEST.json --result2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/pull_box2_run2/pull2/STAGE2_RESULT.json --result3 C:/Users/Admin/t2v8f_pulls/stage3/pull_box2/pull3/STAGE3_RESULT.json --result4 C:/Users/Admin/t2v8f_pulls/stage4/pull_box2/pull4/STAGE4_RESULT.json --exclusions2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/EXCLUSIONS2.json --exclusions3 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage3/EXCLUSIONS3.json --exclusions4 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage4/EXCLUSIONS4.json --out-dir E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage5/C0C-RE/pair_C0C-RE_C0C --a-arm C0C-RE --a-stage 5 --a-root C:/Users/Admin/t2v8f_pulls/stage5_C0C-RE/pull5_C0C-RE/rollouts --a-exclusions E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage5/C0C-RE/EXCLUSIONS5.json --a-overlay E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage5/C0C-RE/t2v8f_stage5_overlay_C0C-RE.tar::pkg/MANIFEST5.json --a-gate-log E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/logs/pytest_full_Q.log --b-arm C0C`
Tier: T2 re-perception (agents replayed at log time; not a reactive-traffic result). Estimator: paired scene-cluster bootstrap, B 2000, seed 0 (answers another draw of SCENES only; the seed floor is the training-run term, read beside).

## Gates

* ATTRIB: PASS 
* G1: PASS 
* G2[C0C-RE@5]: PASS 
* G2[C0C@4]: PASS 
* G3[C0C-RE@5]: PASS 
* G3[C0C@4]: PASS 
* G4: PASS 

## Excluded scenes (rule R-SCN, the union over every stage read)

* Scenes read on: 48 of 62 (14 excluded). Starts read on: 93 of 121 (lost 28).

| scene | idx | excluded in (stage: reason) | starts lost |
|---|---|---|---|
| dffd4c04 | 9 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:C0C-RE: ACTOR_REFUSED | 2 |
| 00169207 | 13 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:C0C-RE: ACTOR_REFUSED | 2 |
| 1aa48b8a | 17 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:C0C-RE: ACTOR_REFUSED | 2 |
| c67bdfef | 18 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:C0C-RE: ACTOR_REFUSED | 2 |
| 90a3d349 | 27 | 2: NO_DYNAMIC_LAYER, 4: NO_DYNAMIC_LAYER, 5:C0C-RE: NO_DYNAMIC_LAYER | 2 |
| aa094af8 | 28 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:C0C-RE: ACTOR_REFUSED | 2 |
| 42a44314 | 33 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:C0C-RE: ACTOR_REFUSED | 2 |
| 829c55af | 42 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:C0C-RE: ACTOR_REFUSED | 2 |
| 964e7935 | 43 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:C0C-RE: ACTOR_REFUSED | 2 |
| b0ca8b13 | 44 | 3: NO_DYNAMIC_LAYER, 4: NO_DYNAMIC_LAYER, 5:C0C-RE: NO_DYNAMIC_LAYER | 2 |
| aace43c1 | 45 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:C0C-RE: ACTOR_REFUSED | 2 |
| 359e435d | 46 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:C0C-RE: ACTOR_REFUSED | 2 |
| ca2cb16e | 54 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:C0C-RE: ACTOR_REFUSED | 2 |
| fbbadc60 | 61 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5:C0C-RE: ACTOR_REFUSED | 2 |

## C0C-RE - C0C, seed 0 (paired over the same starts, per row)

| family | row | A - B | lo | hi | separated | n windows | n scenes | seed floor A | seed floor B |
|---|---|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -0.1956 | -0.4391 | +0.0895 | not separated | 93 | 48 | 0.0539 | 0.1494 |
| LONGITUDINAL | speed MAE | -0.1234 | -0.2157 | -0.0268 | separated | 93 | 48 | 0.1122 | 0.1170 |
| LONGITUDINAL | along-track MAE | -0.0965 | -0.3476 | +0.2074 | not separated | 93 | 48 | 0.0441 | 0.1202 |
| LONGITUDINAL | accel MAE | -0.0156 | -0.1567 | +0.1346 | not separated | 93 | 48 | 0.2591 | 0.1958 |
| LONGITUDINAL | DK-VIOL | +0.0323 | +0.0000 | +0.1034 | not separated | 31 | 25 | 0.0323 | 0.0000 |
| LONGITUDINAL | DK-HW | +0.5474 | +0.0413 | +1.2069 | separated | 40 | 26 | 0.0476 | 0.1439 |
| LONGITUDINAL | DK-TG | +0.0534 | -0.0365 | +0.1796 | not separated | 33 | 25 | 0.1270 | 0.0051 |
| LATERAL | heading MAE | -0.5722 | -5.5319 | +6.2260 | not separated | 93 | 48 | 1.6881 | 5.1153 |
| LATERAL | yaw-rate MAE | +0.4323 | -1.2665 | +2.4124 | not separated | 93 | 48 | 3.5520 | 2.9756 |
| LATERAL | curvature MAE | -0.0059 | -0.0147 | +0.0028 | not separated | 93 | 48 | 0.0052 | 0.0084 |
| LATERAL | cross-track MAE | -0.1169 | -0.2777 | +0.0245 | not separated | 93 | 48 | 0.0978 | 0.0963 |
| LATERAL | L1 lateral-accel MAE | +0.1189 | -0.0692 | +0.3413 | not separated | 93 | 48 | 0.4465 | 0.2805 |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0073 | -0.0181 | +0.0035 | not separated | 93 | 48 | 0.0060 | 0.0138 |
| TACTICAL | TAC-LAT | +0.0753 | -0.0106 | +0.1596 | not separated | 93 | 48 | 0.0753 | 0.0215 |
| TACTICAL | TAC-LON | +0.0538 | -0.0435 | +0.1563 | not separated | 93 | 48 | 0.0645 | 0.0323 |
| STRATEGIC | S-POS@30 | -0.2978 | -0.9465 | +0.3670 | not separated | 90 | 48 | 0.6721 | 1.9151 |
| STRATEGIC | S-POS@50 | -0.4754 | -1.8072 | +0.9356 | not separated | 74 | 48 | 0.0041 | 3.5407 |
| STRATEGIC | S-POS@80 | +1.6176 | -2.9892 | +7.1253 | not separated | 44 | 48 | 2.7471 | 3.4955 |
| crawl | M1 crawl share | -0.0523 | -0.0852 | -0.0249 | separated | 93 | 48 | 0.0348 | 0.0032 |

## C0C-RE - C0C, seed 1 (paired over the same starts, per row)

| family | row | A - B | lo | hi | separated | n windows | n scenes | seed floor A | seed floor B |
|---|---|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -0.1002 | -0.3435 | +0.1502 | not separated | 93 | 48 | 0.0539 | 0.1494 |
| LONGITUDINAL | speed MAE | -0.1282 | -0.2440 | -0.0142 | separated | 93 | 48 | 0.1122 | 0.1170 |
| LONGITUDINAL | along-track MAE | -0.0204 | -0.2695 | +0.2473 | not separated | 93 | 48 | 0.0441 | 0.1202 |
| LONGITUDINAL | accel MAE | -0.0788 | -0.1755 | +0.0051 | not separated | 93 | 48 | 0.2591 | 0.1958 |
| LONGITUDINAL | DK-VIOL | +0.0000 | +0.0000 | +0.0000 | not separated | 31 | 25 | 0.0323 | 0.0000 |
| LONGITUDINAL | DK-HW | +0.6043 | -0.0051 | +1.3520 | not separated | 45 | 26 | 0.0476 | 0.1439 |
| LONGITUDINAL | DK-TG | +0.1753 | -0.0152 | +0.5060 | not separated | 33 | 25 | 0.1270 | 0.0051 |
| LATERAL | heading MAE | +2.8549 | -1.8623 | +8.6949 | not separated | 93 | 48 | 1.6881 | 5.1153 |
| LATERAL | yaw-rate MAE | -0.1441 | -1.2305 | +0.9448 | not separated | 93 | 48 | 3.5520 | 2.9756 |
| LATERAL | curvature MAE | -0.0026 | -0.0123 | +0.0065 | not separated | 93 | 48 | 0.0052 | 0.0084 |
| LATERAL | cross-track MAE | -0.1184 | -0.2894 | +0.0407 | not separated | 93 | 48 | 0.0978 | 0.0963 |
| LATERAL | L1 lateral-accel MAE | -0.0471 | -0.1406 | +0.0430 | not separated | 93 | 48 | 0.4465 | 0.2805 |
| LATERAL | L2 curvature MAE (GT>=5.2) | +0.0004 | -0.0100 | +0.0111 | not separated | 93 | 48 | 0.0060 | 0.0138 |
| TACTICAL | TAC-LAT | +0.0215 | -0.0543 | +0.1064 | not separated | 93 | 48 | 0.0753 | 0.0215 |
| TACTICAL | TAC-LON | +0.0860 | -0.0215 | +0.2000 | not separated | 93 | 48 | 0.0645 | 0.0323 |
| STRATEGIC | S-POS@30 | -1.5408 | -2.5800 | -0.5158 | separated | 90 | 48 | 0.6721 | 1.9151 |
| STRATEGIC | S-POS@50 | -4.0202 | -6.1074 | -2.0622 | separated | 74 | 48 | 0.0041 | 3.5407 |
| STRATEGIC | S-POS@80 | -4.6250 | -8.1186 | -1.1719 | separated | 44 | 48 | 2.7471 | 3.4955 |
| crawl | M1 crawl share | -0.0142 | -0.0450 | +0.0112 | not separated | 93 | 48 | 0.0348 | 0.0032 |

Seed floor = the dominance card's seed gap of that arm against P10 (|delta_s1 - delta_s0|); `none` = the floor P10 has no seeds. The interval answers another draw of SCENES only; a separated row is necessary, not sufficient, for a lever claim (H-ESTIM-SEED-1).

Disclosure: 62 scenes selected on the human's logs only, launch scenes over-represented (42 of 62); a result here is a result on these scenes at T2. A T2 pair is not a reactive-traffic result.
