# MM rulings: how X_s0 is read (2026-11-18 ~10:10Z, written before any X_s0 evaluation number exists)

State when written:
* X_s0 stage A is trained (step 19,164, ckpt sha256 1dd437f1) and its bank step is running on the LoRA pod.
* No X_s0 evaluation, dump or trainer validation number has been read by the MM or by any reader.

Card `CARD_DATASCALE_TRAIN.md` section 2.3 / 6 says X_s0 runs to "diagnose dilution". Block 3 (`../read3/MM_READING_DSCALE10K_B3.md`) localised the question. The 10k clips cost TAC-LAT (T1-kin) by the same amount at both training lengths:
* U − B_s0: −0.0871 [−0.1297, −0.0446];
* UE − BL: −0.0855 [−0.1216, −0.0492].

The question X_s0 answers is: **do the 10k clips by themselves teach a worse lateral tactical decision (CONTENT), or does the cost appear only when they are mixed with the corpus clips (MIXTURE)?**

## R1. Deciding cell, pairs, estimator

* Cell: **TAC-LAT, T1-kin**, the block-1/3 instrument, unchanged.
* Pairs:
  * **X_s0 − B_s0** (deciding);
  * **X_s0 − U_s0** (beside).
* Read at **stage A** (matched to block 3) and at **stage B** (the recipe of record).
* Estimator: paired episode-cluster bootstrap over the 141 eval episodes, B 2,000, seed 0. It answers "another draw of EPISODES" only.
* Floor: F_cell = |TAC-LAT(B_rep) − TAC-LAT(B_s0)| at the same stage. Stage A: 0.0767 (block 3). Stage B: recomputed by the reader from the banked block-1 stage-B arms.
* ONE seed, one-pair floor: every label carries "(one seed)".

## R2. Labels, committed now (stage A decides; stage B is reported beside with its own label)

| label | condition (X − B_s0 at stage A, TAC-LAT; higher is better) | next lever |
|---|---|---|
| **X-CONTENT** | separated AND Δ ≤ −F_cell (worse by at least the floor) | Zero-training audit of the 10k clips' lateral labels and inputs against the corpus clips. Compare their class shares, label builder fields and nav alignment. UM_s0 (rebalancing) would not be expected to remove a content defect, so the MM tells the PI this before UM_s0 runs. |
| **X-NO-COST** | not separated, OR \|Δ\| < F_cell | The 10k clips alone teach TAC-LAT as well as the corpus. The U cost appears only in the mixture, so the mixture arm UM_s0 is the discriminating lever (it is waiting for the PI's go). |
| **X-GAIN** | separated AND Δ ≥ +F_cell | As X-NO-COST, but stronger: the cost is a mixture effect. UM_s0 first. |

* A stage-B label that differs from stage A's is printed as **STAGE-SENSITIVE**, with both labels shown. The stage-A label still names the next lever.
* X − U_s0 is printed beside, with no bar. Δ(X − U_s0) > 0 separated means removing the corpus clips helps TAC-LAT. Δ < 0 separated means the corpus clips carry lateral skill the 10k clips do not.

## R3. Four families and STRATEGIC, in full

* Every pair, both stages: LONGITUDINAL, LATERAL, TACTICAL and STRATEGIC. Use the block-3 cells and rules, F_cell from B_rep, never pooled.
* STRATEGIC is reported in full (the item-25 pause ended 2026-11-16).
* Fog is unscored (PI ruling 2026-11-17).

## R4. Controls that must read known values

* **K-VLM-ZERO (reach, in the other direction).** X_s0 trains with the VLM goal bit masked on every row, so it never sees a VLM-provenance positive.
  * Its VLM-class macro-F1 at stage A must be at most B_s0's minus the block-1 floor, OR at most 0.05.
  * If X_s0 scores VLM classes like B_s0, the mask did not hold. The read is then marked CONTROL FAILED and is not quotable.
* **K1** X − X self-pair reads 0.0 [0, 0].
* **K-RE**: B_s0, U_s0 and B_rep TAC-LAT at both stages reproduce the banked block-1/3 values exactly.
* **K-EVALX** on X_s0 at both stages, as block 3.
* **C-JOIN-EQ** evidence from the chain's log is quoted: the X join equals the union join on the 10k clips.

## R5. What X_s0 cannot answer

* One seed.
* X has 10,000 clips against B's 4,512, at the same steps. X sees each window about 1.8 times (19,164 x 64 / 678,482 bank windows, ESTIMATED) against B's 4.0. This is the same confound direction as U against B (fewer passes per window), and block 3 showed that passes do not explain U's TAC-LAT cost. A CONTENT label is therefore read with that caveat, and a NO-COST is conservative.
* The geometry goal classes and the STRATEGIC cells are read, but they do not decide.
