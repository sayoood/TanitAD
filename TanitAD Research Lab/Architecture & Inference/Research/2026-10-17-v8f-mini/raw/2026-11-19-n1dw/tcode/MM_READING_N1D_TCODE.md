# MM reading: PREREG_N1D_TCODE. N1D's tactical level hands O the roundabout decision through its code; the loss is in O's decoder (2026-11-19)

**Inputs**
- Card: `../PREREG_N1D_TCODE.md` (sha256 4cec3917), frozen and landed at 673035e2 before any feature was extracted.
- Result: `RESULT_TCODE.json` (sha256 6999107f), with the printout `READ_TCODE.txt` (sha256 31efeb21) and per-window out-of-fold correctness `oof_correct.npz` (sha256 68cbd9e2).
- Code: `code/`.
- Features: banked PRIVATE on HF `Sayood/tanitad-v8f-strat1-runs`, `n1dw_tcode/n1dwtcode_feats.tar` (sha256 c93c25b4, hub equal; record `BANK_N1DWTCODE.json`).

**Evidence and estimator**
- MEASURED: banked N1D checkpoints, CPU, zero training, on the pod from 15:18Z to 15:44Z.
- Tier: T0 decodability of frozen features, never driving performance.
- Population: the D1-HO grid, 5,736 windows / 497 held-out episodes. Class counts 1,934 / 1,862 / 1,940.
- Estimator: paired episode-cluster bootstrap, B 2,000, seed 0. It answers "another draw of EPISODES" only; training variance is carried by the two N1D seeds.
- Bar: 0.0383.
- Probe seed 1 is not run, because the tree makes no random draw (PROBE-N1D-FEAT Amendment 1).

## Outcome as committed (tree probe decides): CODE-CARRIES at both N1D seeds

| tree accuracy | F-TEXT | F-CODE | F-TTRUNK | F-PLAN | F-CODE-LAT (beside) | F-CODE zeroed |
|---|---|---|---|---|---|---|
| N1D_s0 | 0.6044 | **0.6702** | 0.6757 | 0.7028 | 0.6531 | 0.3040 |
| N1D_s1 | 0.6044 | **0.6783** | 0.6935 | 0.6937 | 0.6642 | 0.3040 |

| contrast (tree) | N1D_s0 | N1D_s1 |
|---|---|---|
| **F-CODE - F-TEXT** | **+0.0657 [+0.0402, +0.0901]** | **+0.0739 [+0.0469, +0.1001]** |
| F-TTRUNK - F-TEXT | +0.0713 [+0.0467, +0.0954] | +0.0891 [+0.0640, +0.1144] |
| F-PLAN - F-CODE (beside) | +0.0326 [+0.0173, +0.0478] | +0.0153 [-0.0005, +0.0308], not separated |
| F-PLAN - F-TTRUNK (beside) | +0.0270 [+0.0123, +0.0415] | +0.0002 [-0.0139, +0.0148], not separated |
| F-CODE - F-TTRUNK | -0.0056 [-0.0173, +0.0060], not separated | -0.0152 [-0.0262, -0.0035] |
| F-CODE-LAT - F-CODE | -0.0171 [-0.0287, -0.0052] | -0.0141 [-0.0255, -0.0028] |

**How the outcome is reached**
- F-CODE clears F-TEXT by more than the bar at both seeds: each CI's lower end (0.0402, 0.0469) lies above 0.0383.
- T-HAS-CODE-LACKS and T-LACKS are not held.
- The logistic probe, beside, reads CODE-CARRIES at both seeds too.

**What it says**
- N1D's tactical level forms the roundabout decision: T's trunk reads 0.68 / 0.69.
- It hands nearly all of it to O through the 60-column code O reads: 0.67 / 0.68.
- O's plan-head input adds at most +0.03 on top of the code (s0), and nothing separated at s1.
- The 14 lateral and turn columns carry most of the code's share. The full code reads 0.014 to 0.017 more, so the goal and longitudinal columns add a little.

**Together with the two reads before it:**

| stage | s0 | s1 |
|---|---|---|
| route text alone | 0.6044 | 0.6044 |
| T's code | 0.6702 | 0.6783 |
| O's plan feature | 0.7028 | 0.6937 |
| decoded tick-0 plan (T0) | 0.6266 | 0.6349 |
| rolled track (T1) | 0.6196 | 0.6499 |

- The decision is carried into O.
- It is lost when O's plan head decodes the track: 0.67 to 0.70 decodable, 0.63 delivered at T0.
- The last row and the rows above it come from two different estimators: a tree on features, and the kinematic factor of the decoded track. Their differences are point differences, printed as such.

## Controls: all card controls read
| control | value | status |
|---|---|---|
| K-CONST | 0.338215 = the card's | PASS |
| K-DET | the second full extraction is bit-identical on every key (F-CODE and F-TTRUNK included), both checkpoints, n 5,736 | PASS |
| K-PARITY | re-extracted F-TEXT and F-PLAN equal the banked PROBE-N1D-FEAT features, 0.0 on 5,736 / 5,736 windows, both seeds | PASS |
| Regression arm (binding): F-CODE all 60 columns zeroed | -0.3661 [-0.3994, -0.3320] (s0), -0.3743 [-0.4058, -0.3417] (s1) | PASS |
| SHUF (beside) | max deviation from 0.338215 is 0.0248; one cell (N1D_s1 F-TTRUNK, 0.3135) is outside the parent's 0.02 band | beside; same pattern as the parent's SHUF, whose permutation null is still running |

The extractor asserted that the code it read equals T's own prediction (`_code_pred`) and that `forward_infer` runs with no override, mix or teacher forcing.

## What it means, and the next lever (RULE ZERO)
- **Committed next (CODE-CARRIES):** the lever is O-side.
  - N1DW is in build now: its four arms are staged on the pod and wait on the test suite and the mint.
  - Beyond N1DW, the card names a direct code-to-decoder path: the code conditioning `head_o`, zero-init, so step 0 equals N1D. It needs its own card and a PI go.
- **Arithmetic beside, never deciding:**
  - The decoder's room on these windows is about +0.04 to +0.07, from the code (0.67 / 0.68) or from the plan feature (0.70 / 0.69) against the delivered 0.63 / 0.63. That is about 1 to 2 bars.
  - The decision itself (T's 0.68 / 0.69) sits 0.31 below a perfect score.
  - But the route text alone reads only 0.60, which suggests the 6 s kinematic label is not fully determined at NOW (for example, an exit beyond the 6 s horizon). So T's real ceiling is not measured here.
  - That ceiling is the larger lever if it exists. Its cheapest test would be the same tree on the route text with an oracle exit column. It is HYPOTHESIS and gets its own card.
- **N1DW is unaffected:** none of its bars or outcomes depend on this read.
