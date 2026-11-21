# ROLLHIST next arm RHp3 (share p = 0.3 on the same bank): verdict of record ROLLHIST-HARMS (MM, 2026-11-21)

Card `PREREG_V8F_ROLLHIST.md` (781eea87) with Amendments 1 to 7 (Amendment 7 sha256 81f8c5ac, frozen 10:44Z before any RHp3 step). Arms RHp3_s0 and RHp3_s1 against the existing B0_s0 and B0_s1. Code of record 97fe0a56. Evidence class: MEASURED (ours). Tier: T1-kin/hb0.2 (kinematic rollout on the model's own actions); G-T0 at T0. Estimator: paired episode-cluster bootstrap, B 2,000, seed 0; it answers "another draw of EPISODES"; the two seeds answer "another TRAINING run". Floor F = the larger of the two arms' seed spreads per cell. Result: `RESULT_ROLLHIST_P03.json`.

## P1, RHp3 - B0 under readout C* (`R0C10XHb20`), both seeds
| family | cell | RHp3 - B0 [interval] | F | verdict |
|---|---|---|---|---|
| LONGITUDINAL | speed_mae | -0.314 m/s [-0.455, -0.184] | 0.126 | BETTER |
| | along_mae | -0.226 m [-0.430, -0.034] | 0.122 | BETTER |
| | headway U | +0.035 [+0.014, +0.059] | 0.023 | **WORSE** |
| LATERAL | heading_mae | -1.38 deg [-1.62, -1.18] | 0.70 | BETTER |
| | yaw_rate_mae | -8.37 deg/s [-9.96, -6.99] | 3.06 | BETTER |
| TACTICAL | ade_tac | -0.406 m [-0.699, -0.128] | 0.164 | BETTER |
| STRATEGIC | position at tick 2 | -2.01 m [-2.55, -1.54] | 0.71 | BETTER |
Families: LONGITUDINAL MIXED (headway is the one WORSE cell); LATERAL, TACTICAL and STRATEGIC BETTER.

## Beside, no bar: RHp3 - RH under C* (the dose p = 0.3 against p = 0.1)
LATERAL better (heading -0.56 deg [-0.69, -0.43], yaw-rate -2.93 deg/s [-3.79, -2.18]); STRATEGIC better (-0.40 m [-0.56, -0.26]); speed -0.050 m/s [-0.076, -0.023]; along, headway and ade_tac not separated.

## The guard and the other conditions
* **G-T0** (RHp3 - B0 at T0 ADE6, at most +0.0705 m; F_T0 INHERITED, one seed pair): **fails at both seeds**. s0 +0.1124 m [0.0337, 0.1991]; s1 +0.0939 m [0.0309, 0.1628]. The RH arm at p = 0.1 failed it at s0 only.
* P2 False, P3 False, P4 False. The reader's answer: "p = 0.3 repairs NEITHER P2 nor P3".

## Label
By the card's precedence (BLOCKED, then HARMS), G-T0 failing at both seeds is **ROLLHIST-HARMS**, whatever P1 shows. Amendment 7 committed this in advance. It is reported as written.

## What it says
* The larger share improves every family the T1 read can see except headway, and it improves on p = 0.1 at the same time. The cost appears in the teacher-forced T0 forecast on logged history, and it grows from p = 0.1 to p = 0.3: s0 +0.050 to +0.112 m, s1 +0.022 to +0.094 m. Training on more of its own rolled history pulls the model away from the logged distribution that T0 scores.
* By the eval doctrine, T0 is a world-model diagnostic, not driving performance, and T1 is the tier for capability claims. The card made G-T0 a binding guard, so the label stands; this reading is beside it.

## Next lever (card section 3.4 HARMS row, Amendment 7 section 5)
Localise with the beside UE.B if it exists (it needs STAGEB); otherwise a bank generated under readout K, or the recovery label (`H-ST-RECOVERYLABEL-1`). Each is a new arm and needs a PI go.
