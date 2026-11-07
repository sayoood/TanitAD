# DOMINANCE_T2: NOT-DOMINANT

Card PREREG_T2_DOMINANCE (sha256 38aa9df3..., card sha ok: True); script sha256 7fd65ffdf62ae74f5609e3ac259588826bb84b51e3a3a84968b299332a3f35f9.
Arm read: **C0C-RE** (stage 5, cells C0C-RE_s0 and C0C-RE_s1) against the floor P10 (stages 2 and 3). argv: `--root2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/pull_box2_run2/pull2/rollouts --root3 C:/Users/Admin/t2v8f_pulls/stage3/pull_box2/pull3/rollouts --root4 C:/Users/Admin/t2v8f_pulls/stage4/pull_box2/pull4/rollouts --sources2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/SCENE_SOURCES.json --sources3 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage3/SCENE_SOURCES3.json --g1 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/C0C_ID_RESULT.json --result1 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage1/pull_box2/pull/STAGE1_RESULT.json --manifest1 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage1/t2v8f_stage1.tar::pkg/MANIFEST.json --result2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/pull_box2_run2/pull2/STAGE2_RESULT.json --result3 C:/Users/Admin/t2v8f_pulls/stage3/pull_box2/pull3/STAGE3_RESULT.json --result4 C:/Users/Admin/t2v8f_pulls/stage4/pull_box2/pull4/STAGE4_RESULT.json --exclusions2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/EXCLUSIONS2.json --exclusions3 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage3/EXCLUSIONS3.json --exclusions4 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage4/EXCLUSIONS4.json --out-dir E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage5/C0C-RE/score_dom --arm C0C-RE --arm-stage 5 --arm-root C:/Users/Admin/t2v8f_pulls/stage5_C0C-RE/pull5_C0C-RE/rollouts --arm-exclusions E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage5/C0C-RE/EXCLUSIONS5.json --arm-overlay E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage5/C0C-RE/t2v8f_stage5_overlay_C0C-RE.tar::pkg/MANIFEST5.json --arm-gate-log E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/logs/pytest_full_Q.log`
Tier: T2 re-perception (agents replayed at log time; not a reactive-traffic result). Estimator: paired scene-cluster bootstrap, B 2000, seed 0 (answers another draw of SCENES only).

## Gates

* G1: PASS 
* G2: PASS 
* G3: PASS 
* G4: PASS 

## Excluded scenes (rule R-SCN)

Rule R-SCN, AMENDMENT_T2_SCENE_REFUSAL.md (frozen 2167a7eb). A scene the renderer refuses before any planner runs leaves the read for BOTH arms; a scene excluded in ANY stage is out of every pairing.
The excluded scenes' starts are never re-drawn and never replaced. The result is a result on the refusal-surviving scenes, never on the 62.

* Scenes read on: 48 of 62 (14 excluded in the union of stages 2, 3, 4 and 5).
* Starts read on: 93 of 121 (launch 67 of 81, guard 26 of 40); starts lost 28 (launch 14, guard 14).

| stage | scene | idx | reason | failed cells (deciding + beside) of total | starts lost |
|---|---|---|---|---|---|
| 2 | 00169207 | 13 | ACTOR_REFUSED | 10 + 4 of 14 | 2 |
| 2 | 1aa48b8a | 17 | ACTOR_REFUSED | 10 + 4 of 14 | 2 |
| 2 | 42a44314 | 33 | ACTOR_REFUSED | 10 + 4 of 14 | 2 |
| 2 | 90a3d349 | 27 | NO_DYNAMIC_LAYER | 10 + 4 of 14 | 2 |
| 2 | aa094af8 | 28 | ACTOR_REFUSED | 10 + 4 of 14 | 2 |
| 2 | c67bdfef | 18 | ACTOR_REFUSED | 10 + 4 of 14 | 2 |
| 2 | dffd4c04 | 9 | ACTOR_REFUSED | 10 + 4 of 14 | 2 |
| 3 | 359e435d | 46 | ACTOR_REFUSED | 10 + 4 of 14 | 2 |
| 3 | 829c55af | 42 | ACTOR_REFUSED | 10 + 4 of 14 | 2 |
| 3 | 964e7935 | 43 | ACTOR_REFUSED | 10 + 4 of 14 | 2 |
| 3 | aace43c1 | 45 | ACTOR_REFUSED | 10 + 4 of 14 | 2 |
| 3 | b0ca8b13 | 44 | NO_DYNAMIC_LAYER | 10 + 4 of 14 | 2 |
| 3 | ca2cb16e | 54 | ACTOR_REFUSED | 10 + 4 of 14 | 2 |
| 3 | fbbadc60 | 61 | ACTOR_REFUSED | 10 + 4 of 14 | 2 |
| 4 | 00169207 | 13 | ACTOR_REFUSED | 2 + 2 of 4 | 2 |
| 4 | 1aa48b8a | 17 | ACTOR_REFUSED | 2 + 2 of 4 | 2 |
| 4 | 359e435d | 46 | ACTOR_REFUSED | 2 + 2 of 4 | 2 |
| 4 | 42a44314 | 33 | ACTOR_REFUSED | 2 + 2 of 4 | 2 |
| 4 | 829c55af | 42 | ACTOR_REFUSED | 2 + 2 of 4 | 2 |
| 4 | 90a3d349 | 27 | NO_DYNAMIC_LAYER | 2 + 2 of 4 | 2 |
| 4 | 964e7935 | 43 | ACTOR_REFUSED | 2 + 2 of 4 | 2 |
| 4 | aa094af8 | 28 | ACTOR_REFUSED | 2 + 2 of 4 | 2 |
| 4 | aace43c1 | 45 | ACTOR_REFUSED | 2 + 2 of 4 | 2 |
| 4 | b0ca8b13 | 44 | NO_DYNAMIC_LAYER | 2 + 2 of 4 | 2 |
| 4 | c67bdfef | 18 | ACTOR_REFUSED | 2 + 2 of 4 | 2 |
| 4 | ca2cb16e | 54 | ACTOR_REFUSED | 2 + 2 of 4 | 2 |
| 4 | dffd4c04 | 9 | ACTOR_REFUSED | 2 + 2 of 4 | 2 |
| 4 | fbbadc60 | 61 | ACTOR_REFUSED | 2 + 2 of 4 | 2 |
| 5 | 00169207 | 13 | ACTOR_REFUSED | 2 + 2 of 4 | 2 |
| 5 | 1aa48b8a | 17 | ACTOR_REFUSED | 2 + 2 of 4 | 2 |
| 5 | 359e435d | 46 | ACTOR_REFUSED | 2 + 2 of 4 | 2 |
| 5 | 42a44314 | 33 | ACTOR_REFUSED | 2 + 2 of 4 | 2 |
| 5 | 829c55af | 42 | ACTOR_REFUSED | 2 + 2 of 4 | 2 |
| 5 | 90a3d349 | 27 | NO_DYNAMIC_LAYER | 2 + 2 of 4 | 2 |
| 5 | 964e7935 | 43 | ACTOR_REFUSED | 2 + 2 of 4 | 2 |
| 5 | aa094af8 | 28 | ACTOR_REFUSED | 2 + 2 of 4 | 2 |
| 5 | aace43c1 | 45 | ACTOR_REFUSED | 2 + 2 of 4 | 2 |
| 5 | b0ca8b13 | 44 | NO_DYNAMIC_LAYER | 2 + 2 of 4 | 2 |
| 5 | c67bdfef | 18 | ACTOR_REFUSED | 2 + 2 of 4 | 2 |
| 5 | ca2cb16e | 54 | ACTOR_REFUSED | 2 + 2 of 4 | 2 |
| 5 | dffd4c04 | 9 | ACTOR_REFUSED | 2 + 2 of 4 | 2 |
| 5 | fbbadc60 | 61 | ACTOR_REFUSED | 2 + 2 of 4 | 2 |

| scene | idx | excluded in (stage: reason) | starts lost |
|---|---|---|---|
| dffd4c04 | 9 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5: ACTOR_REFUSED | 2 |
| 00169207 | 13 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5: ACTOR_REFUSED | 2 |
| 1aa48b8a | 17 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5: ACTOR_REFUSED | 2 |
| c67bdfef | 18 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5: ACTOR_REFUSED | 2 |
| 90a3d349 | 27 | 2: NO_DYNAMIC_LAYER, 4: NO_DYNAMIC_LAYER, 5: NO_DYNAMIC_LAYER | 2 |
| aa094af8 | 28 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5: ACTOR_REFUSED | 2 |
| 42a44314 | 33 | 2: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5: ACTOR_REFUSED | 2 |
| 829c55af | 42 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5: ACTOR_REFUSED | 2 |
| 964e7935 | 43 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5: ACTOR_REFUSED | 2 |
| b0ca8b13 | 44 | 3: NO_DYNAMIC_LAYER, 4: NO_DYNAMIC_LAYER, 5: NO_DYNAMIC_LAYER | 2 |
| aace43c1 | 45 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5: ACTOR_REFUSED | 2 |
| 359e435d | 46 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5: ACTOR_REFUSED | 2 |
| ca2cb16e | 54 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5: ACTOR_REFUSED | 2 |
| fbbadc60 | 61 | 3: ACTOR_REFUSED, 4: ACTOR_REFUSED, 5: ACTOR_REFUSED | 2 |

## Actor verdict per scene (G2 e)

* 00169207: {'P10@stage2': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'C0C-RE_s0@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'C0C-RE_s1@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)'}
* 0caa5e02: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 10eef229: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 132ae6cb: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 15eaddae: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 1aa48b8a: {'P10@stage2': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'C0C-RE_s0@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'C0C-RE_s1@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)'}
* 1b024ad1: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 26b07382: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 359e435d: {'P10@stage3': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'C0C-RE_s0@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'C0C-RE_s1@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)'}
* 3d6a22f7: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 3e455540: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 41dadaf2: {'P10@stage3': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 42a44314: {'P10@stage2': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'C0C-RE_s0@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'C0C-RE_s1@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)'}
* 4611c801: {'P10@stage3': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 5894f7ff: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 5ab1d9cb: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 5b94e1f5: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 5cdc4c2f: {'P10@stage3': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 5da7524e: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 6140cba2: {'P10@stage3': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 6bffbe09: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 6eaeeb45: {'P10@stage3': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 764c8a4a: {'P10@stage3': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 769e6724: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 76f7e025: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 77d5887d: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 7c72937c: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 7fd4d554: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 829c55af: {'P10@stage3': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'C0C-RE_s0@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'C0C-RE_s1@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)'}
* 904af7cf: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 90a3d349: {'P10@stage2': 'EXCLUDED (R-SCN): NO_DYNAMIC_LAYER (stage 2), NO_DYNAMIC_LAYER (stage 4), NO_DYNAMIC_LAYER (stage 5)', 'C0C-RE_s0@stage5': 'EXCLUDED (R-SCN): NO_DYNAMIC_LAYER (stage 2), NO_DYNAMIC_LAYER (stage 4), NO_DYNAMIC_LAYER (stage 5)', 'C0C-RE_s1@stage5': 'EXCLUDED (R-SCN): NO_DYNAMIC_LAYER (stage 2), NO_DYNAMIC_LAYER (stage 4), NO_DYNAMIC_LAYER (stage 5)'}
* 90ab8c24: {'P10@stage3': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 951ce3ff: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 964e7935: {'P10@stage3': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'C0C-RE_s0@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'C0C-RE_s1@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)'}
* 9d73b364: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 9e17ce05: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 9e6753c2: {'P10@stage3': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* 9e792fc4: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* a2a4322c: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* a4353805: {'P10@stage3': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* a686c7d0: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* aa094af8: {'P10@stage2': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'C0C-RE_s0@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'C0C-RE_s1@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)'}
* aace43c1: {'P10@stage3': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'C0C-RE_s0@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'C0C-RE_s1@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)'}
* b0ca8b13: {'P10@stage3': 'EXCLUDED (R-SCN): NO_DYNAMIC_LAYER (stage 3), NO_DYNAMIC_LAYER (stage 4), NO_DYNAMIC_LAYER (stage 5)', 'C0C-RE_s0@stage5': 'EXCLUDED (R-SCN): NO_DYNAMIC_LAYER (stage 3), NO_DYNAMIC_LAYER (stage 4), NO_DYNAMIC_LAYER (stage 5)', 'C0C-RE_s1@stage5': 'EXCLUDED (R-SCN): NO_DYNAMIC_LAYER (stage 3), NO_DYNAMIC_LAYER (stage 4), NO_DYNAMIC_LAYER (stage 5)'}
* b12fa656: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* b303ebe3: {'P10@stage3': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* bfb44da0: {'P10@stage3': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* c67bdfef: {'P10@stage2': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'C0C-RE_s0@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'C0C-RE_s1@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)'}
* c8c798be: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* ca2cb16e: {'P10@stage3': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'C0C-RE_s0@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'C0C-RE_s1@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)'}
* cda8a096: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* d078258b: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* d3267951: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* dd9fd9d1: {'P10@stage3': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* dffd4c04: {'P10@stage2': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'C0C-RE_s0@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'C0C-RE_s1@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)'}
* e34fcbdc: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* e848c843: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* ec41061e: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* f58c6fa3: {'P10@stage3': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* fa035232: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}
* fbbadc60: {'P10@stage3': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'C0C-RE_s0@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'C0C-RE_s1@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)'}
* fc41d9e9: {'P10@stage2': 'ACCEPTED', 'C0C-RE_s0@stage5': 'ACCEPTED', 'C0C-RE_s1@stage5': 'ACCEPTED'}

## Decision

Outcome: **NOT-DOMINANT**. UNDERPOWERED is tested before NOT-DOMINANT: with the card's literal order it would be unreachable (no separated row => D1 fails => NOT-DOMINANT)

* s0: DOMINANT False (D1 False, D2 False, D3 True), three-family form False, rows separated 14 of 19; failing_terms: ['D1 (ADE6 not separated-better with point <= -0.2415 m; point +0.5937)', 'D2 (separated-worse: ADE6, M1 crawl share, TAC-LON, along-track MAE, heading MAE)']
* s1: DOMINANT False (D1 False, D2 False, D3 False), three-family form False, rows separated 11 of 19; failing_terms: ['D1 (ADE6 not separated-better with point <= -0.2415 m; point +0.5397)', 'D2 (separated-worse: M1 crawl share, along-track MAE, heading MAE)', 'D3 (STRATEGIC: no separated-better row)']

## Per-family table, s0

| family | row | delta | lo | hi | verdict |
|---|---|---|---|---|---|
| LATERAL | heading MAE | +10.8060 | +2.8542 | +19.2939 | separated-worse |
| LATERAL | yaw-rate MAE | -65.8670 | -86.2199 | -45.3236 | separated-better |
| LATERAL | curvature MAE | -0.1831 | -0.2434 | -0.1205 | separated-better |
| LATERAL | cross-track MAE | -1.3060 | -1.6366 | -0.9647 | separated-better |
| LATERAL | L1 lateral-accel MAE | -15.2896 | -20.4467 | -10.3974 | separated-better |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0427 | -0.0725 | -0.0141 | separated-better |
| LONGITUDINAL | speed MAE | -0.6894 | -1.0388 | -0.3357 | separated-better |
| LONGITUDINAL | along-track MAE | +1.3198 | +0.7791 | +1.8677 | separated-worse |
| LONGITUDINAL | accel MAE | -3.4749 | -4.7371 | -2.2683 | separated-better |
| LONGITUDINAL | DK-VIOL | +0.0323 | +0.0000 | +0.1034 | unseparated |
| LONGITUDINAL | DK-HW | -0.0659 | -1.1113 | +0.8725 | unseparated |
| LONGITUDINAL | DK-TG | +0.0423 | -0.3826 | +0.5262 | unseparated |
| PRIMARY | ADE6 | +0.5937 | +0.0883 | +1.0880 | separated-worse |
| STRATEGIC | S-POS@30 | -2.6334 | -4.7620 | -0.5144 | separated-better |
| STRATEGIC | S-POS@50 | -3.3391 | -6.7468 | +0.4373 | unseparated |
| STRATEGIC | S-POS@80 | -2.1755 | -7.7609 | +2.9173 | unseparated |
| TACTICAL | TAC-LAT | +0.1828 | +0.0753 | +0.2947 | separated-better |
| TACTICAL | TAC-LON | -0.1720 | -0.2979 | -0.0532 | separated-worse |
| crawl | M1 crawl share | +0.3550 | +0.2628 | +0.4511 | separated-worse |

## Per-family table, s1

| family | row | delta | lo | hi | verdict |
|---|---|---|---|---|---|
| LATERAL | heading MAE | +9.1179 | +1.7150 | +17.5802 | separated-worse |
| LATERAL | yaw-rate MAE | -69.4190 | -90.4740 | -48.0409 | separated-better |
| LATERAL | curvature MAE | -0.1883 | -0.2489 | -0.1255 | separated-better |
| LATERAL | cross-track MAE | -1.4038 | -1.7032 | -1.0727 | separated-better |
| LATERAL | L1 lateral-accel MAE | -15.7361 | -21.0043 | -10.6863 | separated-better |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0487 | -0.0809 | -0.0153 | separated-better |
| LONGITUDINAL | speed MAE | -0.5772 | -0.8990 | -0.2597 | separated-better |
| LONGITUDINAL | along-track MAE | +1.2758 | +0.6192 | +1.9148 | separated-worse |
| LONGITUDINAL | accel MAE | -3.7340 | -5.0411 | -2.4651 | separated-better |
| LONGITUDINAL | DK-VIOL | +0.0000 | +0.0000 | +0.0000 | unseparated |
| LONGITUDINAL | DK-HW | -0.1135 | -0.6426 | +0.3840 | unseparated |
| LONGITUDINAL | DK-TG | +0.1693 | -0.0220 | +0.4771 | unseparated |
| PRIMARY | ADE6 | +0.5397 | -0.0459 | +1.0951 | unseparated |
| STRATEGIC | S-POS@30 | -1.9613 | -3.9972 | +0.1168 | unseparated |
| STRATEGIC | S-POS@50 | -3.3432 | -6.8580 | +0.3073 | unseparated |
| STRATEGIC | S-POS@80 | -4.9226 | -12.0885 | +1.7767 | unseparated |
| TACTICAL | TAC-LAT | +0.1075 | +0.0107 | +0.2043 | separated-better |
| TACTICAL | TAC-LON | -0.1075 | -0.2418 | +0.0316 | unseparated |
| crawl | M1 crawl share | +0.3898 | +0.2816 | +0.4931 | separated-worse |

## Seed gap beside (section 7)

* speed MAE: s0 -0.6894, s1 -0.5772, gap 0.1122, margin 0.2597, margin < gap: False
* accel MAE: s0 -3.4749, s1 -3.7340, gap 0.2591, margin 2.2683, margin < gap: False
* yaw-rate MAE: s0 -65.8670, s1 -69.4190, gap 3.5520, margin 45.3236, margin < gap: False
* curvature MAE: s0 -0.1831, s1 -0.1883, gap 0.0052, margin 0.1205, margin < gap: False
* cross-track MAE: s0 -1.3060, s1 -1.4038, gap 0.0978, margin 0.9647, margin < gap: False
* L1 lateral-accel MAE: s0 -15.2896, s1 -15.7361, gap 0.4465, margin 10.3974, margin < gap: False
* L2 curvature MAE (GT>=5.2): s0 -0.0427, s1 -0.0487, gap 0.0060, margin 0.0141, margin < gap: False
* TAC-LAT: s0 +0.1828, s1 +0.1075, gap 0.0753, margin 0.0107, margin < gap: True

Rule R-SCN applied: this read is on the 48 refusal-surviving scenes of the 62 (93 of 121 starts).

Disclosure: 62 scenes selected on the human's logs only, launch scenes over-represented (42 of 62); a result here is a result on these scenes at T2, never on PhysicalAI-AV at large.
The render gap (gate T1 of the crawl card) is reported beside by its own record and is never deciding. A T2 DOMINANT is not a reactive-traffic result.
