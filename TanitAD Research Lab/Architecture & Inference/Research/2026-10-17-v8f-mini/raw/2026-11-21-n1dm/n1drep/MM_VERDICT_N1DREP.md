# MM verdict: N1DREP (same-code, same-seed replicate of N1D_s0) — N1DREP-COST-HOLDS

Card `PREREG_V8F_N1DREP.md` (sha256 0c641197, frozen before training) and Amendment 1 (ba600897, tree count, written before training). PI go "Run both", 2026-11-22 about 07:12Z.
Result `out/RESULT_N1DREP.json` (sha256 d9222bf381d3c017). Evidence class MEASURED (ours). Tier T1-kin/hb0.2 (executor of record), eval-141, 141 episodes, 2,399 windows. Estimator: paired episode-cluster bootstrap, B 2,000, seed 0; it answers "another draw of EPISODES", and the replicate answers "another TRAINING run" at fixed seed, init and code (inference deterministic).

## The arm (pod)
* N1DREP_s0 trained in N1D's own tree at 0c3b374f, from 07:27:26Z, 692.8 s (N1D_s0: 672 s); lock held 07:23:39Z to 07:40:19Z, about 0.28 pod-h. Checkpoint sha256 f4f04acc5ed95e23.
* K-REP-CODE: the argv equals N1D_s0's but for `--out`; the gate token's commit, trainer, checker, tree manifest, G-SUITE result, nav reports and data equal N1D_s0's. Post check PASS; zero-grad gate PASS (253 of 253 parameters trained).
* Attempt 1 (07:20:15Z) stopped at the chain's own tree count before any training (Amendment 1).

## Controls (dev box)
* K-REDUMP PASS: N1D_s0 dumped again in the same session equals the banked hb0.2 dump in all 1,269 arrays. The read tree had been deleted after the N1DM read and was rebuilt from 573ad839 plus the hb wrapper blob 35425445 (sha256 74fd8445); K-REDUMP shows the rebuilt tree reproduces the banked dump.
* K1 (self-pair exactly 0) PASS; K-SAME (model-free arrays equal the reference) PASS for all three dumps.
* IDENTICAL test: 282 of 1,269 arrays differ, so the trainer is not deterministic at this seed (the DAgger swap), and F_REP is a floor, not an identity.

## The deciding cells
| cell | F_REP = abs(N1DREP_s0 - N1D_s0) [paired CI of the difference] | F2 | floor max(F2, F_REP) | N1DM - N1D, s0 / s1 | ratio to floor |
|---|---|---|---|---|---|
| ade_m (0-2 s) | 0.00208 [+0.0007, +0.0035] | 0.00047 | 0.00208 | +0.01223 / +0.01238 | 5.9 / 6.0 |
| along_mae_m (0-2 s) | 0.00160 [+0.0003, +0.0029] | 0.00053 | 0.00160 | +0.01232 / +0.01237 | 7.7 / 7.7 |
| speed_mae_mps (0-2 s) | 0.00200 [-0.0001, +0.0042] | 0.00306 | 0.00306 | +0.00836 / +0.00813 | 2.7 / 2.7 |

All three exceed at both seeds: **N1DREP-COST-HOLDS**. As the card commits: F_REP becomes the floor of record for operative cells on the N1 line at hb0.2; N1DM stays not adopted (verdict of record N1DM-COST, unchanged). On all 24 guard cells x 2 seeds, 14 of 48 N1DM - N1D deltas exceed max(F2, F_REP); no cost cell that cleared F2 falls inside the new floor.

## Families
LONGITUDINAL, LATERAL and TACTICAL: F_REP on all 24 guard cells (operative 0-2 s, tactical 2-6 s, TAC-LAT, TAC-LON) in the result file. STRATEGIC: not measured by this card, because the guard carries no strategic cell on this line (card section 3).

## Beside the card (read from the same result; not a card outcome)
* **The replicate alone separates from N1D_s0 on 6 of 24 cells, with no lever moved:** operative ade +0.00208 m [+0.0007, +0.0035], along +0.00160 m [+0.0003, +0.0029], headway error +0.0145 m [+0.0017, +0.0310], time-gap error +0.0055 s [+0.0007, +0.0119]; tactical accel +0.0613 [+0.0191, +0.1180], tactical speed +0.0287 m/s [+0.0057, +0.0557]. This is `H-ESTIM-SEED-1` on the N1 line at hb0.2: a separated interval smaller than these magnitudes is not evidence of a lever effect here. It is one replicate pair (one draw of training noise), so 6 of 24 is a single reading, not a rate.
* **N1DH's guard costs re-read against the floor of record** (`read_n1dh/eval141/out/RESULT_N1DH_EVAL141.json` sha256 cdcf16f0): s0 ade +0.00184 is now inside the floor (0.00208); s0 along +0.00165 clears 0.00160 by 0.00005 m; s1 along +0.00273 clears it; the s0 tactical time-gap cost (+0.0480 s, F_REP 0.0386) is outside the card's operative scope and is reported only. N1DH-COST (not adopted) stands on along at both seeds, s0 at the floor's edge.

## Next
* The N1 line's operative cells read against max(F2, F_REP) from now on; tactical cells have F_REP in the file but no floor of record (a tactical floor would need its own card).
* The decoder-path family (N1DH, N1DM) stays not adopted; no further arm on it is proposed here.
