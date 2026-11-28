# PREREG_V8F_PROOF_ADD42_N1 : the first run of the trained gap head is not named; a fit whose quantile must hold on clips it never saw

Note to card `PREREG_V8F_PROOF_ADD42.md`. Frozen after the first run of the card was read (`qhead/read/`, `qhead/pulled/`, read on 2026-11-19 at 03:45Z) and before any fit or rollout of this note. The outcome differences of the first run were withheld by its reader and were not computed. Tier: T0 for the fits (heads on frozen states of training clips; no model is trained), T1-kin for the rollouts (kinematic rollout on the model's own actions, logged frames, the lead does not react); never driving performance. STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and the reader has no strategic instrument).

## 1. The first run (MEASURED)

* Queue: four fits (51 s each on the pod CPU), then the control rollout and four rollouts, done 2026-11-19T03:39Z. A first start had stopped at the control rollout on a script fault before any rollout (one-token flag, unbound argument); fixed and restarted, stated in `TIMES_NOTE.md`.
* Controls: h1 PASS (the control rollout equals the banked head-fed dump: 141 files, 0 differing), h2 PASS, h4 PASS, h6 PASS, h7 PASS.
* **h5 FAIL at all four checkpoints.** Share of rows with `q_head` above the label gap: on the 1,000 fitted clips 0.186, 0.188, 0.208, 0.174; on the 200 held-out clips 0.446, 0.421, 0.420, 0.478 (band 0.15 to 0.25). The head learned the quantile of the rows it was fitted on and not a quantile that transfers (19,164 steps on 26,327 rows of 1,000 clips).
* **h3 FAIL in its card form.** In the rollout the gap the rule used equals `min(g_head, q_head)` exactly (0.0 at all four arms). Recomputed on the dev box from the banked states, it differs by at most 4.3, 3.2, 5.1 and 5.8 mm (mean 0.7 to 0.9 mm) against the bar 1e-4 m: the rollout ran on the pod GPU, the recomputation on another machine's CPU.
* By card section 6 no arm is named. Names: none.

## 2. The lever of this note: the same head and loss, with the fit made to answer for unseen clips

The head, its inputs, the pinball loss at quantile 0.20 and the optimiser recipe are those of the card. Three changes, all on training clips only:

* **Split** of the 1,200 train probe clips in manifest order: fit clips 0 to 799, select clips 800 to 999, check clips 1,000 to 1,199 (the held-out clips of the card; they enter no fit and no selection).
* **Early stopping:** the head is evaluated on the select clips every 250 steps; the step with the lowest pinball loss on the select clips is kept (ties: the earlier step).
* **Shift:** one constant c per checkpoint, the value for which the share of select rows with `q_head - c` above the label gap is 0.20 (the 0.80 quantile of `q_head - label` on the select rows). The used output is `q_head - c`; it is written into the head file, so the rollout tool of the card is unchanged.

## 3. Arms and read

* Arms: `LSQE_s0`, `LSQE_s1`, `LSDQE_s0`, `LSDQE_s1`: the rollouts of the card with the head of this note, thresholds as in `qhead_arms.txt` (0.4, 0.5, 0.4, 0.3). The control rollout of the first run stands for h1 if the rollout tool file is byte-identical (its sha256 is compared); else it is run again.
* Read: as card section 5 with QE in place of QH, second-version headway reader and third-version family reader.
* Cost (MEASURED on the first run): fits about 1 min each on the pod CPU, rollouts about 4 min each on one GPU slot.

## 4. Controls

* h1, h2, h4, h6, h7 as in the card.
* **h5 (bar unchanged):** on the check clips the share of rows with the used output above the label gap lies in [0.15, 0.25], per checkpoint; outside it the arm of that checkpoint is not named. Reported beside: the same share on the fit and the select clips, the selected step, c, and the pinball loss on the select clips over the steps.
* **h3 in two parts.** (a) In the rollout: the gap the rule used equals `min(g_head, q_head - c)` at the first tick, exactly (bar 0.0). (b) Recomputed on the dev box from the banked states: within 0.02 m, a tolerance stated here because the two machines differ by up to 5.8 mm on the first run; a planted difference of 0.10 m in the recomputation must fail the check, or part (b) does not count.
* h8 (new): the check clips are used by no fit and no selection: the record of each head lists the clip indices of its three sets, and a test fails when a check row is added to the selection.

## 5. Outcomes

The names of card section 7, read on the arms of this note and written with the suffix `-QE` (for example `QHEAD-REACHES-ORACLE-QE`). The bars are unchanged.

## 6. What follows (committed before the data)

* Named outcomes: card section 8 applies.
* h5 fails again on the check clips for an arm type: the state does not carry a transferable low quantile at 800 fitted clips. Then the dose step of the card is run under the fit of this note: states of all 4,572 training clips for the four checkpoints (ESTIMATED 0.6 GB per checkpoint from 156 MB for 1,200 clips; a pod job, named with its volume before it runs), the same three-way split by manifest order in the same proportions.
* h5 passes and the outcome is `QHEAD-U-EQUAL-QE` with `QHEAD-COST-EQUAL-QE`: as card section 8, second bullet.

## 7. Limits

One fit per checkpoint and seed; 200 check clips, so a coverage inside the band says the quantile holds on those clips and not more; the shift is a constant, so it repairs the level of the quantile and not its shape; the tolerance of h3 part (b) was chosen after the first run and is guarded by the planted difference; four rollouts of two arm types.
