# PREREG_V8F_PROOF_ADD132 — GAPSPREAD-T0: does the disagreement of the three gap fits say where the candidate's gap reads long?

Frozen before any tool of this card is written and before any value of it exists. Open at this time: card ADD131 (row 200, landing). Tier **T0** (recorded frames, no rollout; a diagnostic of a perception read and not driving performance). CPU only, on files that exist on the fork pod; no training, no fit, no encode, no detector, no rollout.

## 1. Why this card

Row 181 reads that at readout K with no pre-roll the unsafe rate of the candidate above the label form comes with the camera gap value. Rows 183 and 193 read that a flat margin of 10 % on that gap closes it on four model seeds (unsafe -0.0116 [-0.0175, -0.0063]) at a price in falling behind (+0.0062 [+0.0019, +0.0112]). Row 200 reads that a more exact gap by a detector box moves nothing. A flat margin shortens every gap, also where the read is right. The candidate's gap file is the mean of three fits of the same head (`VGAPL_s2`, `VGAPL_s3`, `VGAPL_s4`; card ADD106); their disagreement on a frame costs nothing and has not been read. If it marks the frames where the mean reads long, a margin placed by it buys the safety of the flat margin at a lower price. This card reads on recorded frames whether it does, before any rollout is spent on it.

Stated guess before data: three fits of one head on the same clips share most of their error; the spread marks the long reads weakly (rank correlation between 0.05 and 0.2), the ratio of the long part lies between 0.93 and 1.00, and the name is `ALLOCATION-NOT-SEPARATED` or `ALLOCATION-BETTER` above the bar of 0.90.

## 2. Files, rows and arms

Files: the base file `VGAPLE.npz` (sha256 begins 55de2f7c) and the three fit files `VGAPL_s2.npz`, `VGAPL_s3.npz`, `VGAPL_s4.npz` of card ADD106, with the hashes of that card's record. Label rows: the rows of the base file with an output and a label gap, selected exactly as the T0 read of card ADD131 selects them (3,841 rows of 141 clips). The label gap is used for scoring alone; every arm is computed from the three fit files with no label.

Per row with an output: m = the mean of the three fits (the base gap), s = (largest - smallest of the three) / m, the relative spread.

* **A90** — the flat margin: 0.90 m.
* **SP90** — the margin placed by the spread: m times max(0.60, 1 - k s), with k the one number for which the mean of the factor over all rows of the file with an output (no label enters) is 0.90 within 1e-6, found by bisection. Same mean dose as A90, placed where the fits disagree.
* **ENV** — the lower envelope, the smallest of the three fits (beside, no name).

## 3. Measures and estimator

On the label rows, for a gap g: the long part L(g) = mean of max(g - label, 0), which lets the rule follow closer than the label form; the short part S(g) = mean of max(label - g, 0), which lets it fall behind. RL = L(SP90) / L(A90) and RS = S(SP90) / S(A90), each with a cluster bootstrap over the 141 clips, 2,000 draws, seed 132, 95 % interval. Beside, no name: the rank correlation of s with the signed relative error (m - label) / label and its interval by the same bootstrap; k; the mean of s; the share of rows with an output where the factor of SP90 is below 0.90 and where it is at the floor of 0.60; L and S of the base, A90, SP90 and ENV; the mean factor of ENV; RL and RS by label gap (0 to 10, 10 to 20, 20 to 40, 40 to 60 m). The interval answers what another draw of clips would say, not what another fit of the head or another model would say.

Families: this is a read of one perception input on recorded frames. LONGITUDINAL: the long and the short part of the gap, as above. LATERAL, TACTICAL, STRATEGIC: not read by this card (no rollout, no plan, no route; n = 0), stated in the row.

## 4. Names

One name:

* `GAPSPREAD-T0-ALLOCATION-BETTER` — the interval of RL lies below 1 and the interval of RS does not lie above 1;
* `GAPSPREAD-T0-ALLOCATION-TRADES` — the interval of RL lies below 1 and the interval of RS lies above 1 (less of the long part at more of the short part: another point on the same trade);
* `GAPSPREAD-T0-ALLOCATION-WORSE` — the interval of RL lies above 1;
* `GAPSPREAD-T0-ALLOCATION-NOT-SEPARATED` — else.

With `ALLOCATION-BETTER` a second name: `GAPSPREAD-T0-BAR-MET` when RL is at most 0.90, else `GAPSPREAD-T0-BAR-NOT-MET`.

## 5. Controls (all have to pass before the name or any value is opened)

* **s0 files** — the four files have the hashes of the record of card ADD106 and equal arrays `clip_index`, `frame`, `has_out` and `clips`; on every row with an output the mean of the three fits equals the base gap within 1e-3 m.
* **s1 rows** — the tool selects 3,841 label rows of 141 clips and reproduces the mean absolute error of the base file of row 200 (3.132 m) within 1e-3 m.
* **s2 dose** — the mean factor of SP90 over all rows with an output is 0.90 within 1e-6; with the spread set to one constant on every row SP90 equals A90 within 1e-9 m on every row; a second routine, row by row, gives the gap of SP90 within 1e-9 m.
* **s3 the measure can read a mark** — on the real rows with a planted spread (s replaced by the positive part of the signed relative error plus 0.01) RL lies below 0.90; with the real spread permuted within the label rows and over the other rows (seed 132) the interval of RL holds 1 or RL lies within 0.03 of 1.
* **s4 tests** — the tool's test file passes in the lead's run on the dev box and on the pod before the run of record; outputs outside the kit folder.

This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

## 6. Committed readings

* **ALLOCATION-BETTER and BAR-MET** → the next card puts SP90 into the candidate's lead rule at readout K with no pre-roll on the four model seeds with banked runs (four rollouts, about 0.6 pod-hours on the fork pod, existing models), read against A90 and the candidate on the unsafe rate and on falling behind. A margin placed by the spread is a parameter of the rule's input: measured as a study arm, adopted by the PI and the other session alone.
* **Any other name** → the disagreement of the three fits does not place a margin better than a flat one; no rollout is spent on it. The gap line is then read out as far as existing files go (more clips, direction of the window, smoothing, a map of the value, a detector box, the spread of the fits), and the next card turns to the larger term: the unsafe rate of the label form itself (0.0369 on four seeds at readout K with no pre-roll, against a gap part of 0.0111), read on the banked label-form runs with no new rollout: which windows are unsafe with label inputs, at what range and closing speed.

## 7. Cost and limits

ESTIMATED under 5 minutes of CPU on the fork pod and no GPU time. Limits: T0 only; the long and the short part of the gap are measures of the input and not of the rule's unsafe rate or of falling behind, which a rollout alone reads; three fits of one head, one floor (0.60) and one dose (0.90); the evaluation clips score the arms and the dose k is set with no label, yet the form of SP90 was chosen with the results of rows 172 to 200 known.
