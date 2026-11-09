# VERDICT_PT3F (Master Mind, 2026-11-09)
Card `PREREG_V8F_PT3F.md` (one folder up in the repo), frozen prefix sha256 `6c826c39...f3ef2dfe9`. Tier **T0**, an oracle-goal ceiling with no world-model rollout and no planner, never a driving result. MEASURED: `RESULT_PT3F.md`, `RESULT_PT3F.json`; arrays and readers PRIVATE on HF, `HF_BANK_PT3F.json`. Paired episode-cluster bootstrap over the 71 SCORE episodes (994 windows), B 2,000; it answers "another draw of episodes"; both training seeds are required. TACTICAL: not computed (n = 0, no decision output). STRATEGIC: PAUSED (PI item 25).

## Validity
All controls pass. Reader tie to the full-n probe difference 0.0000 on all six rows; PT3M tie, HOLD tie and ORACLE-END tie largest window difference 0.0; split guard and its mutation pass; 4,512 unique clips present; PT3M's files unchanged.

## Outcome: NOT-READ in both families
| family | HOLD | F-DECIDE s0 | minus HOLD s0 | F-DECIDE s1 | minus HOLD s1 | label |
|---|---|---|---|---|---|---|
| LONGITUDINAL (m/s^2) | 0.3348 | 1.0049 | +0.6701 [+0.5138, +0.8426] worse | 1.0988 | +0.7640 [+0.6121, +0.9311] worse | **NOT-READ** |
| LATERAL (rad/s) | 0.0256 | 0.0228 | -0.0028 [-0.0061, +0.0002] none | 0.0228 | -0.0028 [-0.0058, +0.0001] none | **NOT-READ** |

* **Speed.** Fitting on 4,565 episodes in place of 1,053 lowers the mean read error at knot 12 from 2.68 / 2.60 m/s (PT3M, `DIAG_RESPONSE.json`) to 2.30 / 2.57 m/s, and acceleration MAE from 1.151 / 1.115 to 1.005 / 1.099 m/s^2, still about three times HOLD. Held-out R2: MLP 0.778 at s0, ridge 0.696 at s1.
* **Yaw rate.** The point estimate is 11 % below HOLD at both seeds and each interval just includes zero, so the committed label is NOT-READ. It stays well above ORACLE-END (0.0142 rad/s). Held-out R2 0.510 / 0.463.
* **The learning rate was not what held the yaw rate back in PT3M.** N1053 with the yaw-rate MLP at 1e-3 (held-out 0.570 / 0.441) reads 0.0255 / 0.0258 rad/s, against PT3M's 0.0255 / 0.0261 at the frozen rule's 1e-4.
* **Differencing two reads does not rescue the speed at N4565 either** (0.803 / 0.774 m/s^2, worse; beside, no bar).
* ADE at 2.4 s (beside): 0.996 / 1.063 m, against HOLD 0.228 m and ORACLE-END 0.207 m.

## Trigger applied
The card's row for NOT-READ, in both families: the reader lever on z1 at this scale is spent. No planner run follows from this card. Next, for both families: the harness read on the ladder world models, and the measured-speed world-model card.

## What the planner chain now says (PLANTEST3, PT3N, PT3M, PT3F; T0, oracle goal)
Four reader settings have been tried inside or under the end-point cost: a 53-episode ridge, a 1,053-episode ridge, a four-frame per-target reader, and readers fitted on 4,565 episodes. None clears HOLD in the planner or at the READ-END ceiling, while the same end-point form with exact values beats HOLD by 35 % and 45 %. The absolute speed held in this model's z1 latent is the bottleneck: about 2.3 to 2.6 m/s of error against 0.69 m/s for HOLD. Inference may use the measured current ego state (R-EGO), and the measured-speed world model puts that signal in the state the planner reasons over.

## Limits
* SCORE is one set of 71 episodes; two training seeds of one world model family (M2b).
* The readers are a ridge and one MLP form on z1; the pooled token reads speed better on held-out episodes (0.840 / 0.801) but is not what the world model rolls out, so it is not a cost for this planner.
* The goal is the encoded true future. A goal the tactical layer predicts is a different object.

## Next (Rule Zero)
1. **Harness read on the ladder world models** when they finish (run 9 of 9 started 15:39), to see whether wider predictors change the action response.
2. **Measured-speed world model:** card frozen after the ladder's T0 read; runs on box 2 after the band retrain (PI, 2026-11-09).
