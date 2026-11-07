# DOMINANCE_T2: NOT-DOMINANT

Card PREREG_T2_DOMINANCE (sha256 38aa9df3..., card sha ok: True); script sha256 7fd65ffdf62ae74f5609e3ac259588826bb84b51e3a3a84968b299332a3f35f9.
Arm read: **MC** (stage 5, cells MC_s0 and MC_s1) against the floor P10 (stages 2 and 3). argv: `--root2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/pull_box2_run2/pull2/rollouts --root3 C:/Users/Admin/t2v8f_pulls/stage3/pull_box2/pull3/rollouts --root4 C:/Users/Admin/t2v8f_pulls/stage4/pull_box2/pull4/rollouts --sources2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/SCENE_SOURCES.json --sources3 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage3/SCENE_SOURCES3.json --g1 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/C0C_ID_RESULT.json --result1 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage1/pull_box2/pull/STAGE1_RESULT.json --manifest1 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage1/t2v8f_stage1.tar::pkg/MANIFEST.json --result2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/pull_box2_run2/pull2/STAGE2_RESULT.json --result3 C:/Users/Admin/t2v8f_pulls/stage3/pull_box2/pull3/STAGE3_RESULT.json --result4 C:/Users/Admin/t2v8f_pulls/stage4/pull_box2/pull4/STAGE4_RESULT.json --exclusions2 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/EXCLUSIONS2.json --exclusions3 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage3/EXCLUSIONS3.json --exclusions4 E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage4/EXCLUSIONS4.json --out-dir E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage5/MC/score_dom --arm MC --arm-stage 5 --arm-root C:/Users/Admin/t2v8f_pulls/stage5_MC/pull5_MC/rollouts --arm-exclusions E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage5/MC/EXCLUSIONS5.json --arm-overlay E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage5/MC/t2v8f_stage5_overlay_MC.tar::pkg/MANIFEST5.json --arm-gate-log E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/logs/pytest_full_Q.log`
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

* 00169207: {'P10@stage2': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'MC_s0@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'MC_s1@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)'}
* 0caa5e02: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 10eef229: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 132ae6cb: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 15eaddae: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 1aa48b8a: {'P10@stage2': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'MC_s0@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'MC_s1@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)'}
* 1b024ad1: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 26b07382: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 359e435d: {'P10@stage3': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'MC_s0@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'MC_s1@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)'}
* 3d6a22f7: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 3e455540: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 41dadaf2: {'P10@stage3': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 42a44314: {'P10@stage2': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'MC_s0@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'MC_s1@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)'}
* 4611c801: {'P10@stage3': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 5894f7ff: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 5ab1d9cb: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 5b94e1f5: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 5cdc4c2f: {'P10@stage3': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 5da7524e: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 6140cba2: {'P10@stage3': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 6bffbe09: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 6eaeeb45: {'P10@stage3': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 764c8a4a: {'P10@stage3': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 769e6724: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 76f7e025: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 77d5887d: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 7c72937c: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 7fd4d554: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 829c55af: {'P10@stage3': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'MC_s0@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'MC_s1@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)'}
* 904af7cf: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 90a3d349: {'P10@stage2': 'EXCLUDED (R-SCN): NO_DYNAMIC_LAYER (stage 2), NO_DYNAMIC_LAYER (stage 4), NO_DYNAMIC_LAYER (stage 5)', 'MC_s0@stage5': 'EXCLUDED (R-SCN): NO_DYNAMIC_LAYER (stage 2), NO_DYNAMIC_LAYER (stage 4), NO_DYNAMIC_LAYER (stage 5)', 'MC_s1@stage5': 'EXCLUDED (R-SCN): NO_DYNAMIC_LAYER (stage 2), NO_DYNAMIC_LAYER (stage 4), NO_DYNAMIC_LAYER (stage 5)'}
* 90ab8c24: {'P10@stage3': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 951ce3ff: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 964e7935: {'P10@stage3': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'MC_s0@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'MC_s1@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)'}
* 9d73b364: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 9e17ce05: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 9e6753c2: {'P10@stage3': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* 9e792fc4: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* a2a4322c: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* a4353805: {'P10@stage3': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* a686c7d0: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* aa094af8: {'P10@stage2': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'MC_s0@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'MC_s1@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)'}
* aace43c1: {'P10@stage3': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'MC_s0@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'MC_s1@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)'}
* b0ca8b13: {'P10@stage3': 'EXCLUDED (R-SCN): NO_DYNAMIC_LAYER (stage 3), NO_DYNAMIC_LAYER (stage 4), NO_DYNAMIC_LAYER (stage 5)', 'MC_s0@stage5': 'EXCLUDED (R-SCN): NO_DYNAMIC_LAYER (stage 3), NO_DYNAMIC_LAYER (stage 4), NO_DYNAMIC_LAYER (stage 5)', 'MC_s1@stage5': 'EXCLUDED (R-SCN): NO_DYNAMIC_LAYER (stage 3), NO_DYNAMIC_LAYER (stage 4), NO_DYNAMIC_LAYER (stage 5)'}
* b12fa656: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* b303ebe3: {'P10@stage3': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* bfb44da0: {'P10@stage3': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* c67bdfef: {'P10@stage2': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'MC_s0@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'MC_s1@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)'}
* c8c798be: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* ca2cb16e: {'P10@stage3': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'MC_s0@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'MC_s1@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)'}
* cda8a096: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* d078258b: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* d3267951: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* dd9fd9d1: {'P10@stage3': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* dffd4c04: {'P10@stage2': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'MC_s0@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'MC_s1@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)'}
* e34fcbdc: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* e848c843: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* ec41061e: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* f58c6fa3: {'P10@stage3': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* fa035232: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}
* fbbadc60: {'P10@stage3': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'MC_s0@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)', 'MC_s1@stage5': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4), ACTOR_REFUSED (stage 5)'}
* fc41d9e9: {'P10@stage2': 'ACCEPTED', 'MC_s0@stage5': 'ACCEPTED', 'MC_s1@stage5': 'ACCEPTED'}

## Decision

Outcome: **NOT-DOMINANT**. UNDERPOWERED is tested before NOT-DOMINANT: with the card's literal order it would be unreachable (no separated row => D1 fails => NOT-DOMINANT)

* s0: DOMINANT False (D1 False, D2 False, D3 False), three-family form False, rows separated 11 of 19; failing_terms: ['D1 (ADE6 not separated-better with point <= -0.2415 m; point +1.4552)', 'D2 (separated-worse: ADE6, M1 crawl share, TAC-LON, along-track MAE, heading MAE)', 'D3 (TACTICAL: no separated-better row)']
* s1: DOMINANT False (D1 False, D2 False, D3 False), three-family form False, rows separated 11 of 19; failing_terms: ['D1 (ADE6 not separated-better with point <= -0.2415 m; point +0.9463)', 'D2 (separated-worse: ADE6, M1 crawl share, TAC-LON, along-track MAE, heading MAE)', 'D3 (TACTICAL: no separated-better row)', 'D3 (STRATEGIC: no separated-better row)']

## Per-family table, s0

| family | row | delta | lo | hi | verdict |
|---|---|---|---|---|---|
| LATERAL | heading MAE | +16.5860 | +8.2295 | +25.1287 | separated-worse |
| LATERAL | yaw-rate MAE | -63.4213 | -83.5992 | -42.8631 | separated-better |
| LATERAL | curvature MAE | -0.1539 | -0.2167 | -0.0914 | separated-better |
| LATERAL | cross-track MAE | -1.2526 | -1.5704 | -0.9344 | separated-better |
| LATERAL | L1 lateral-accel MAE | -15.0852 | -20.2894 | -10.1636 | separated-better |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0099 | -0.0458 | +0.0247 | unseparated |
| LONGITUDINAL | speed MAE | -0.1410 | -0.5407 | +0.2801 | unseparated |
| LONGITUDINAL | along-track MAE | +2.1273 | +1.4923 | +2.7365 | separated-worse |
| LONGITUDINAL | accel MAE | -2.4485 | -3.9165 | -1.0373 | separated-better |
| LONGITUDINAL | DK-VIOL | +0.0000 | +0.0000 | +0.0000 | unseparated |
| LONGITUDINAL | DK-HW | -0.0836 | -0.8314 | +0.7395 | unseparated |
| LONGITUDINAL | DK-TG | +0.0931 | -0.1290 | +0.4633 | unseparated |
| PRIMARY | ADE6 | +1.4552 | +0.9442 | +1.9479 | separated-worse |
| STRATEGIC | S-POS@30 | -3.2007 | -5.2636 | -1.1765 | separated-better |
| STRATEGIC | S-POS@50 | -3.2321 | -7.4001 | +0.7596 | unseparated |
| STRATEGIC | S-POS@80 | -1.7529 | -9.3973 | +5.2146 | unseparated |
| TACTICAL | TAC-LAT | +0.0645 | -0.0645 | +0.2021 | unseparated |
| TACTICAL | TAC-LON | -0.2796 | -0.3914 | -0.1684 | separated-worse |
| crawl | M1 crawl share | +0.3493 | +0.2681 | +0.4233 | separated-worse |

## Per-family table, s1

| family | row | delta | lo | hi | verdict |
|---|---|---|---|---|---|
| LATERAL | heading MAE | +11.5272 | +3.8572 | +19.2901 | separated-worse |
| LATERAL | yaw-rate MAE | -63.9442 | -84.9026 | -43.7226 | separated-better |
| LATERAL | curvature MAE | -0.1809 | -0.2414 | -0.1207 | separated-better |
| LATERAL | cross-track MAE | -1.2808 | -1.5608 | -0.9815 | separated-better |
| LATERAL | L1 lateral-accel MAE | -14.8617 | -20.0531 | -9.9357 | separated-better |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0400 | -0.0697 | -0.0113 | separated-better |
| LONGITUDINAL | speed MAE | -0.2881 | -0.6866 | +0.1290 | unseparated |
| LONGITUDINAL | along-track MAE | +1.6883 | +1.0591 | +2.2603 | separated-worse |
| LONGITUDINAL | accel MAE | -2.3604 | -3.7675 | -0.9706 | separated-better |
| LONGITUDINAL | DK-VIOL | +0.0000 | +0.0000 | +0.0000 | unseparated |
| LONGITUDINAL | DK-HW | -0.3893 | -0.9729 | +0.3313 | unseparated |
| LONGITUDINAL | DK-TG | +0.0492 | -0.1968 | +0.4081 | unseparated |
| PRIMARY | ADE6 | +0.9463 | +0.3548 | +1.4572 | separated-worse |
| STRATEGIC | S-POS@30 | -0.7681 | -2.7531 | +1.0773 | unseparated |
| STRATEGIC | S-POS@50 | +0.1816 | -4.4682 | +4.4641 | unseparated |
| STRATEGIC | S-POS@80 | +2.0493 | -7.0915 | +10.6445 | unseparated |
| TACTICAL | TAC-LAT | +0.0323 | -0.0638 | +0.1355 | unseparated |
| TACTICAL | TAC-LON | -0.1398 | -0.2553 | -0.0319 | separated-worse |
| crawl | M1 crawl share | +0.2919 | +0.2109 | +0.3736 | separated-worse |

## Seed gap beside (section 7)

* accel MAE: s0 -2.4485, s1 -2.3604, gap 0.0882, margin 0.9706, margin < gap: False
* yaw-rate MAE: s0 -63.4213, s1 -63.9442, gap 0.5229, margin 42.8631, margin < gap: False
* curvature MAE: s0 -0.1539, s1 -0.1809, gap 0.0271, margin 0.0914, margin < gap: False
* cross-track MAE: s0 -1.2526, s1 -1.2808, gap 0.0282, margin 0.9344, margin < gap: False
* L1 lateral-accel MAE: s0 -15.0852, s1 -14.8617, gap 0.2235, margin 9.9357, margin < gap: False

Rule R-SCN applied: this read is on the 48 refusal-surviving scenes of the 62 (93 of 121 starts).

Disclosure: 62 scenes selected on the human's logs only, launch scenes over-represented (42 of 62); a result here is a result on these scenes at T2, never on PhysicalAI-AV at large.
The render gap (gate T1 of the crawl card) is reported beside by its own record and is never deciding. A T2 DOMINANT is not a reactive-traffic result.
