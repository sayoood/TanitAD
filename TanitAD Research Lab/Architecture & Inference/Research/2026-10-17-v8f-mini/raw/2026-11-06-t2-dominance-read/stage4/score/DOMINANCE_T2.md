# DOMINANCE_T2: NOT-DOMINANT

Card PREREG_T2_DOMINANCE (sha256 38aa9df3..., card sha ok: True); script sha256 d66647180a040318fc48b65bbdd068b637cf5d5a3e8b2ca0f91c4cee6bf1280a.
Tier: T2 re-perception (agents replayed at log time; not a reactive-traffic result). Estimator: paired scene-cluster bootstrap, B 2000, seed 0 (answers another draw of SCENES only).

## Gates

* G1: PASS 
* G2: PASS 
* G3: PASS 
* G4: PASS 

## Excluded scenes (rule R-SCN)

Rule R-SCN, AMENDMENT_T2_SCENE_REFUSAL.md (frozen 2167a7eb). A scene the renderer refuses before any planner runs leaves the read for BOTH arms; a scene excluded in ANY stage is out of every pairing.
The excluded scenes' starts are never re-drawn and never replaced. The result is a result on the refusal-surviving scenes, never on the 62.

* Scenes read on: 48 of 62 (14 excluded in the union of stages 2, 3 and 4).
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

## Actor verdict per scene (G2 e)

* 00169207: {'P10@stage2': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4)', 'C0C_s0@stage4': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4)', 'C0C_s1@stage4': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4)'}
* 0caa5e02: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 10eef229: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 132ae6cb: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 15eaddae: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 1aa48b8a: {'P10@stage2': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4)', 'C0C_s0@stage4': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4)', 'C0C_s1@stage4': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4)'}
* 1b024ad1: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 26b07382: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 359e435d: {'P10@stage3': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4)', 'C0C_s0@stage4': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4)', 'C0C_s1@stage4': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4)'}
* 3d6a22f7: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 3e455540: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 41dadaf2: {'P10@stage3': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 42a44314: {'P10@stage2': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4)', 'C0C_s0@stage4': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4)', 'C0C_s1@stage4': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4)'}
* 4611c801: {'P10@stage3': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 5894f7ff: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 5ab1d9cb: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 5b94e1f5: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 5cdc4c2f: {'P10@stage3': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 5da7524e: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 6140cba2: {'P10@stage3': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 6bffbe09: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 6eaeeb45: {'P10@stage3': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 764c8a4a: {'P10@stage3': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 769e6724: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 76f7e025: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 77d5887d: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 7c72937c: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 7fd4d554: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 829c55af: {'P10@stage3': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4)', 'C0C_s0@stage4': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4)', 'C0C_s1@stage4': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4)'}
* 904af7cf: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 90a3d349: {'P10@stage2': 'EXCLUDED (R-SCN): NO_DYNAMIC_LAYER (stage 2), NO_DYNAMIC_LAYER (stage 4)', 'C0C_s0@stage4': 'EXCLUDED (R-SCN): NO_DYNAMIC_LAYER (stage 2), NO_DYNAMIC_LAYER (stage 4)', 'C0C_s1@stage4': 'EXCLUDED (R-SCN): NO_DYNAMIC_LAYER (stage 2), NO_DYNAMIC_LAYER (stage 4)'}
* 90ab8c24: {'P10@stage3': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 951ce3ff: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 964e7935: {'P10@stage3': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4)', 'C0C_s0@stage4': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4)', 'C0C_s1@stage4': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4)'}
* 9d73b364: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 9e17ce05: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 9e6753c2: {'P10@stage3': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* 9e792fc4: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* a2a4322c: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* a4353805: {'P10@stage3': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* a686c7d0: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* aa094af8: {'P10@stage2': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4)', 'C0C_s0@stage4': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4)', 'C0C_s1@stage4': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4)'}
* aace43c1: {'P10@stage3': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4)', 'C0C_s0@stage4': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4)', 'C0C_s1@stage4': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4)'}
* b0ca8b13: {'P10@stage3': 'EXCLUDED (R-SCN): NO_DYNAMIC_LAYER (stage 3), NO_DYNAMIC_LAYER (stage 4)', 'C0C_s0@stage4': 'EXCLUDED (R-SCN): NO_DYNAMIC_LAYER (stage 3), NO_DYNAMIC_LAYER (stage 4)', 'C0C_s1@stage4': 'EXCLUDED (R-SCN): NO_DYNAMIC_LAYER (stage 3), NO_DYNAMIC_LAYER (stage 4)'}
* b12fa656: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* b303ebe3: {'P10@stage3': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* bfb44da0: {'P10@stage3': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* c67bdfef: {'P10@stage2': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4)', 'C0C_s0@stage4': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4)', 'C0C_s1@stage4': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4)'}
* c8c798be: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* ca2cb16e: {'P10@stage3': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4)', 'C0C_s0@stage4': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4)', 'C0C_s1@stage4': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4)'}
* cda8a096: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* d078258b: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* d3267951: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* dd9fd9d1: {'P10@stage3': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* dffd4c04: {'P10@stage2': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4)', 'C0C_s0@stage4': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4)', 'C0C_s1@stage4': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4)'}
* e34fcbdc: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* e848c843: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* ec41061e: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* f58c6fa3: {'P10@stage3': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* fa035232: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}
* fbbadc60: {'P10@stage3': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4)', 'C0C_s0@stage4': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4)', 'C0C_s1@stage4': 'EXCLUDED (R-SCN): ACTOR_REFUSED (stage 3), ACTOR_REFUSED (stage 4)'}
* fc41d9e9: {'P10@stage2': 'ACCEPTED', 'C0C_s0@stage4': 'ACCEPTED', 'C0C_s1@stage4': 'ACCEPTED'}

## Decision

Outcome: **NOT-DOMINANT**. UNDERPOWERED is tested before NOT-DOMINANT: with the card's literal order it would be unreachable (no separated row => D1 fails => NOT-DOMINANT)

* s0: DOMINANT False (D1 False, D2 False, D3 False), three-family form False, rows separated 13 of 19; failing_terms: ['D1 (ADE6 not separated-better with point <= -0.2415 m; point +0.7893)', 'D2 (separated-worse: ADE6, M1 crawl share, TAC-LON, along-track MAE, heading MAE)', 'D3 (TACTICAL: no separated-better row)']
* s1: DOMINANT False (D1 False, D2 False, D3 False), three-family form False, rows separated 12 of 19; failing_terms: ['D1 (ADE6 not separated-better with point <= -0.2415 m; point +0.6399)', 'D2 (separated-worse: ADE6, M1 crawl share, TAC-LON, along-track MAE)', 'D3 (TACTICAL: no separated-better row)', 'D3 (STRATEGIC: no separated-better row)']

## Per-family table, s0

| family | row | delta | lo | hi | verdict |
|---|---|---|---|---|---|
| LATERAL | heading MAE | +11.3782 | +4.2750 | +18.9838 | separated-worse |
| LATERAL | yaw-rate MAE | -66.2993 | -87.3509 | -45.6714 | separated-better |
| LATERAL | curvature MAE | -0.1772 | -0.2385 | -0.1156 | separated-better |
| LATERAL | cross-track MAE | -1.1891 | -1.5496 | -0.8309 | separated-better |
| LATERAL | L1 lateral-accel MAE | -15.4085 | -20.6528 | -10.4454 | separated-better |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0354 | -0.0677 | -0.0055 | separated-better |
| LONGITUDINAL | speed MAE | -0.5660 | -0.9096 | -0.1990 | separated-better |
| LONGITUDINAL | along-track MAE | +1.4164 | +0.8567 | +1.9535 | separated-worse |
| LONGITUDINAL | accel MAE | -3.4593 | -4.7769 | -2.1896 | separated-better |
| LONGITUDINAL | DK-VIOL | +0.0000 | +0.0000 | +0.0000 | unseparated |
| LONGITUDINAL | DK-HW | -0.5740 | -1.3929 | +0.1305 | unseparated |
| LONGITUDINAL | DK-TG | -0.0111 | -0.3898 | +0.4535 | unseparated |
| PRIMARY | ADE6 | +0.7893 | +0.3313 | +1.2386 | separated-worse |
| STRATEGIC | S-POS@30 | -2.3356 | -4.3371 | -0.2416 | separated-better |
| STRATEGIC | S-POS@50 | -2.8637 | -6.2061 | +0.7062 | unseparated |
| STRATEGIC | S-POS@80 | -3.7931 | -10.3190 | +1.9946 | unseparated |
| TACTICAL | TAC-LAT | +0.1075 | -0.0106 | +0.2366 | unseparated |
| TACTICAL | TAC-LON | -0.2258 | -0.3517 | -0.0968 | separated-worse |
| crawl | M1 crawl share | +0.4073 | +0.3180 | +0.5002 | separated-worse |

## Per-family table, s1

| family | row | delta | lo | hi | verdict |
|---|---|---|---|---|---|
| LATERAL | heading MAE | +6.2630 | -0.6923 | +14.2532 | unseparated |
| LATERAL | yaw-rate MAE | -69.2749 | -90.8690 | -47.5371 | separated-better |
| LATERAL | curvature MAE | -0.1857 | -0.2479 | -0.1249 | separated-better |
| LATERAL | cross-track MAE | -1.2854 | -1.6165 | -0.9377 | separated-better |
| LATERAL | L1 lateral-accel MAE | -15.6890 | -20.9633 | -10.6268 | separated-better |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0491 | -0.0820 | -0.0178 | separated-better |
| LONGITUDINAL | speed MAE | -0.4490 | -0.7713 | -0.1455 | separated-better |
| LONGITUDINAL | along-track MAE | +1.2962 | +0.6504 | +1.8949 | separated-worse |
| LONGITUDINAL | accel MAE | -3.6551 | -4.9924 | -2.3817 | separated-better |
| LONGITUDINAL | DK-VIOL | +0.0000 | +0.0000 | +0.0000 | unseparated |
| LONGITUDINAL | DK-HW | -0.7179 | -1.2745 | -0.1974 | separated-better |
| LONGITUDINAL | DK-TG | -0.0060 | -0.3167 | +0.3594 | unseparated |
| PRIMARY | ADE6 | +0.6399 | +0.1070 | +1.1504 | separated-worse |
| STRATEGIC | S-POS@30 | -0.4205 | -2.5083 | +1.5886 | unseparated |
| STRATEGIC | S-POS@50 | +0.6770 | -3.0058 | +4.7153 | unseparated |
| STRATEGIC | S-POS@80 | -0.2976 | -6.2975 | +5.7409 | unseparated |
| TACTICAL | TAC-LAT | +0.0860 | -0.0211 | +0.1915 | unseparated |
| TACTICAL | TAC-LON | -0.1935 | -0.3085 | -0.0832 | separated-worse |
| crawl | M1 crawl share | +0.4041 | +0.3033 | +0.5027 | separated-worse |

## Seed gap beside (section 7)

* speed MAE: s0 -0.5660, s1 -0.4490, gap 0.1170, margin 0.1455, margin < gap: False
* accel MAE: s0 -3.4593, s1 -3.6551, gap 0.1958, margin 2.1896, margin < gap: False
* yaw-rate MAE: s0 -66.2993, s1 -69.2749, gap 2.9756, margin 45.6714, margin < gap: False
* curvature MAE: s0 -0.1772, s1 -0.1857, gap 0.0084, margin 0.1156, margin < gap: False
* cross-track MAE: s0 -1.1891, s1 -1.2854, gap 0.0963, margin 0.8309, margin < gap: False
* L1 lateral-accel MAE: s0 -15.4085, s1 -15.6890, gap 0.2805, margin 10.4454, margin < gap: False
* L2 curvature MAE (GT>=5.2): s0 -0.0354, s1 -0.0491, gap 0.0138, margin 0.0055, margin < gap: True

Rule R-SCN applied: this read is on the 48 refusal-surviving scenes of the 62 (93 of 121 starts).

Disclosure: 62 scenes selected on the human's logs only, launch scenes over-represented (42 of 62); a result here is a result on these scenes at T2, never on PhysicalAI-AV at large.
The render gap (gate T1 of the crawl card) is reported beside by its own record and is never deciding. A T2 DOMINANT is not a reactive-traffic result.
