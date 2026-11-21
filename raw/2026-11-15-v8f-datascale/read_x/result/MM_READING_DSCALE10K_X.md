# MM reading: DSCALE10K X_s0, the 10,000 clips alone (2026-11-18)

Source: `out/RESULT_DSCALE10K_X.json`. Rules frozen in `blockx/MM_RULINGS_X_READ.md` (sha256 e6e3f24c), written before any X_s0 number, re-hashed by the reader.
* Controls: no problems.
  * K1 0.0 [0, 0].
  * K-RE: every block-1 and block-3 row reproduced, 0 mismatches.
  * K-EVALX PASS at both stages.
  * K-VLM-ZERO PASS: X_s0 VLM macro-F1 0.0953 against B_s0 0.1257 (bar 0.0984). GEO macro-F1 is 0.5936 against 0.5928.
  * C-JOIN-EQ True in the banked gate records.
* Estimator: paired episode-cluster bootstrap, B 2,000, seed 0, 141 episodes / 2,399 windows. It answers "another draw of EPISODES" only.
* **ONE seed, one-pair floor.**

## Outcome as committed: X-NO-COST (one seed), not stage-sensitive

| TAC-LAT (T1-kin, higher is better) | stage A (decides) | stage B |
|---|---|---|
| X_s0 − B_s0 | +0.0317 [−0.0192, +0.0796], F 0.0767 | +0.0071 [−0.0133, +0.0305], F 0.0300 |
| X_s0 − U_s0 (beside) | **+0.1188 [+0.0796, +0.1583], separated, above F** | **+0.0575 [+0.0304, +0.0851], separated, above F** |

* **The 10k clips alone teach the lateral tactical decision as well as the corpus does.**
* **The union of the two is worse than EITHER source alone.** Block 3 measured the cost against the corpus: U − B_s0 −0.0871. Here it is against the 10k set as well: X − U_s0 +0.1188.
* So the cost is an interaction of the mixture. It is not the content of either source.

## Four families, per pair, never pooled

| pair | LONGITUDINAL | LATERAL | TACTICAL | STRATEGIC |
|---|---|---|---|---|
| X − B_s0, A | gain (speed MAE 0-2 s) | none | none | cost (S1 pos_err@80) |
| X − B_s0, B | none | cost (yaw-rate MAE 0-2 s) | gain (TAC-LON) | gain (S2 skill@80) |
| X − U_s0, A | none | none | gain (TAC-LAT) | cost (S1 pos_err@80) |
| X − U_s0, B | gain (speed MAE 0-2 s) | none | gain (TAC-LAT, TAC-LON) | none |

* Fog is unscored (PI ruling 2026-11-17). STRATEGIC is in full.

## What the mixture cost is NOT (MEASURED, eliminations)

* **Epochs per clip** (block 3): UE − BL carries the same TAC-LAT cost at 3.2x the steps.
* **The content of the 10k clips** (this read): X ≈ B.
* **A label-convention conflict between the two builders** (GEO10K `CONTROL_RESULTS.md`):
  * 20 corpus clips were rebuilt through the 10k label path. Every v9 field matched the release to float noise (max 3.56e-12).
  * The only exceptions are, by design, the VLM goal bits and flags, and the lane-change diagnostics.
  * Not controlled there: the lead pass. It feeds the longitudinal labels, not TAC-LAT.

## Next lever (as committed in R2): UM_s0

* UM_s0 is the per-source tactical-loss weight arm (card `CARD_DSCALE_MIX.md` sha256 0bc241f5, code 42ef66da).
* It is the discriminating experiment for a mixture effect. It waits for the PI's go (about 4.4 pod-h, ESTIMATED).
* HYPOTHESIS for the mechanism: the masked VLM goal bits on 69 % of union rows change what the tactical block learns for the corpus scenes. Each source alone is self-consistent; the mixture is not.
  * UM_s0 restores the corpus rows' tactical share.
  * If it removes the TAC-LAT cost, the share is the mechanism. If it does not, the next candidate is the goal-bit mask itself, through an arm that masks the VLM bits on the corpus rows too.
