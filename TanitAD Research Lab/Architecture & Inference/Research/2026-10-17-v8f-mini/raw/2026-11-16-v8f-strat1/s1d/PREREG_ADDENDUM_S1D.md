# PREREG addendum to PREREG_V8F_STRAT1: the S1d read rule (FROZEN by the MM 2026-11-17 at about 09:00Z, before any S1d number exists)

**Status.** FROZEN. On the pod, S1d_s0, S1d_s1 and S1d_s10 were DONE and S1d_s11 was minting. No S1d checkpoint had been pulled, dumped or read. The builder's proposal is `PREREG_ADDENDUM_S1D_READ.md`. This file freezes it with the changes stated below. The reader is `s1d_read.py`, and its label names stay as coded.

## Why a rule is needed
* The card (PREREG_V8F_STRAT1 section 6, row COSTS-HEADS) names the arm S1d: S1 with `--ctx-detach on`, at the same seeds and with the same reads. It gives no outcome rule for that arm.
* The N1 read at 4 seeds (`n1s10/out/RESULT_STRAT1_N1S10.json`, MEASURED, T1-kin, 141 episodes / 2,399 windows per seed, paired bootstrap B 2,000 seed 0) reads:
  * TAC-LAT is COST4 for S1 − C0, at seeds {0, 10, 11};
  * N1 − C0 costs at seeds {1, 11} only, so NAV4 is empty;
  * S1 − N1 costs at no seed, so HEADS4 is empty;
  * attribution: **NEITHER**.
* So the card's rule sends the attribution to S1d: "reported as written, and S1d is read as the card says".

## Definitions (as proposed, frozen)
* Seeds {0, 1, 10, 11}; init 0 = {0, 10}, init 1 = {1, 11}.
* Count rule r2: ≥ 3 of the 4 seeds, with ≥ 1 in each init.
* F per cell is the card's floor, as in the SEEDS and N1 reads. The same 24 driving cells and the same reader constants apply.
* COST4_S1(cell): S1 − C0 costs under the count rule.
* COST4_S1d(cell): S1d − C0 costs under the count rule.
* GAIN4_DET(cell): S1d − S1 is separated-better with |Δ| > F, under the count rule.

## Labels, per COST4_S1 cell (today: TAC-LAT)
| label (as coded) | condition | what it means |
|---|---|---|
| **CURED** | not COST4_S1d, and GAIN4_DET | Detaching the context-loss gradient removes the cost, and S1d is separated-better than S1 at the count rule. |
| **PARTIAL** | not COST4_S1d, and not GAIN4_DET | **Read as UNRESOLVED.** The cost is not seen in S1d at the count rule, but S1d is not shown better than S1 either. This is NOT evidence of a cure; at 4 seeds it is indistinguishable from training variance. |
| **REMAINS** | COST4_S1d | The cost does not ride on the context-loss gradient into the adapter. Qualifier **REDUCED** if GAIN4_DET also holds (computed from the result's `GAIN4_DET` set and printed by the MM). |

**Also listed, never hidden:**
* NEW-COST: a COST4_S1d cell that is not a COST4_S1 cell.
* The GAIN4 cells of S1d − C0.
* DETECT-LOST: a variable that DETECTS in S1 and not in S1d (strategic K4, card sections 4.4 / 4.5), at the pairs {0, 1} and {10, 11}.

## Beside, no bar (added by the MM)
* **S1d − N1 per seed, on all 24 cells** (the reader already computes these pairs).
  * With the detach exact, the adapter of S1d receives N1's gradient. The two arms then differ in the context heads' presence and the random stream, not in what the driving path is trained on.
  * These pairs are therefore read as a 4-seed reference for training-run noise (`H-ESTIM-SEED-1`). It is a HYPOTHESIS that the detach makes them equivalent. G-LIVE's probe `adapter_grad_ctx == 0.0` is the gate evidence for the detach.
* The two replicate references already measured on TAC-LAT are printed beside it:
  * C0_s10 − C0_s0: −0.0475 [−0.0692, −0.0275].
  * C0_s11 − C0_s1: +0.0075 [−0.0088, +0.0255].
  * These differ in sign, against F 0.0067.

## Next step per label (Rule Zero; the next lever named, with its blocker)
* **CURED, and no DETECT-LOST on a variable the card reads:** S1d replaces S1 as the route-text arm. The card's KEEPS / ADDS next step (the N2 card) is read on S1d.
  * With DETECT-LOST: S1d replaces S1 for driving. The lost variable is reported, and detection is read from S1 until a card fixes it.
* **REMAINS (REDUCED or not):** the context-loss gradient is not the carrier, and the N1 read attributes the cost to neither the nav text nor the heads.
  * The next lever is the floor itself: a STRAT1 TAC-LAT floor from ≥ 2 same-seed replicate pairs. That means C0 replicates at s10 and s11: 2 arms, about 30 pod-min, ESTIMATED from the N1 s10/s11 pair's wall time.
  * It then re-reads COST4 for S1, N1 and S1d against that floor.
  * **Blocked:** it adds arms, so it goes to the PI with this result.
* **PARTIAL (UNRESOLVED):** the same floor lever, also blocked on the PI. S1 stays the route-text arm of record until the floor read; S1d is reported as the candidate.
* **NEW-COST on any cell:** listed, and S1d does not replace S1 on any label until a card reads that cell.

## Controls
As coded:
* K-RE: every recomputed pair that exists in a banked result equals it exactly, and each file has ≥ 1 compared row.
* The K1 self-pair reads exactly 0.0.
* The S1d tags are accepted at 4dedac45, which differs from a4357e79 in `strat1_check.py` and its test only.
