# MM reading: STRAT1 C0 same-seed replicates (2026-11-17 ~22:45Z)

Evidence class MEASURED (ours). Tiers: the driving cells are T1-kin; STRATEGIC is T0 at the window's first tick. Estimator: paired episode-cluster bootstrap B 2,000 seed 0 ("another draw of EPISODES"). Addendum `PREREG_ADDENDUM_C0REP.md`, sha256 2f41cf5c, frozen before the arms trained. Result `out/RESULT_STRAT1_C0REP.{json,md}`.

## Label as committed: **COST-HOLDS**

* S1's TAC-LAT cost (S1 - C0) is COST4 at seeds 0, 10 and 11 under F2 = max(F, F_rep), exactly as under F.
* For TAC-LAT, F_rep is 0.0046, below the card's F of 0.0067, so F2 equals F.
* F2 changed no COST4 or GAIN4 call in any arm pair (S1-C0, N1-C0, S1d-C0, S1d-S1, S1-N1).
* S1d stays **PARTIAL** on TAC-LAT under F and F2. N1's attribution stays **NEITHER** under both.
* Controls all pass:
  * K1 self-pair exactly 0.0;
  * K-RE: 960 rows against the four earlier STRAT1 results, 0 mismatches;
  * K-REFLAG: 576 rows;
  * K-ARGV for both replicates;
  * K-PIN.

## What the replicates add

* **Training is not deterministic at a fixed seed on the LoRA pod for this trainer.** The replicate checkpoints differ from the originals (sha256 ae28eb76 vs f69c100a at s10, 887438234 vs 8d2d5923 at s11). The cells measure what that variance does.
* **The rig's own noise floor shows here too.** Two runs with the same flags and seed give separated differences on **5 of 48** single cells (24 per pair):
  * s10r: operative heading, tactical headway, tactical curvature;
  * s11r: operative yaw rate, operative curvature.
  * That is 10.4 %, the same order as the 9.5 % false-positive rate measured for `separated` on the v7-tiny rig (H-ESTIM-SEED-1).
  * Neither aggregate (TAC-LAT, TAC-LON) separates.
  * So a single-cell separated difference between two trained STRAT1 arms is not by itself a lever effect. The aggregate COST4 rule, which needs 3 of 4 seeds, is what carries the claim, and it holds.
* **STRATEGIC (T0):** route following of each replicate and its original, real minus donor and minus zero per arc (30 / 50 / 80 m), and RC-A with F and F_rep are all in the JSON.
  * C0_s10 follows the real route: real minus donor is +0.43 / +0.60 / +0.56, separated at all three arcs, on 75 / 149 / 244 windows.
  * Detected info is not computable for C0, which has no context heads (n 0).

## Next (Rule Zero)

The S1 heads' TAC-LAT cost is real, and the stop-gradient (S1d) removes it without a gain over S1 (PARTIAL). The open lever on the strategic line is the route channel: NAVRB2 (N1R / S1R at s0 / s1).
* Committed f95de67a.
* The pod G-SUITE is running, and the arms queue behind DSCALE block 3.
* Its read uses this F2 as the replicate floor for the C0-relative cells.
