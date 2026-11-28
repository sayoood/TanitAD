# PREREG_V8F_PROOF_ADD118 — PRESHEAD-T0: does a presence head on the existing narrow-window tokens reach the bar of row 180?

Frozen before any tool of this card is written and before any value of it exists. Open at this time: cards ADD114 (row 181), ADD116 (row 180) and ADD115 (row 182 in writing); no value of card ADD117 is open (its rollouts are running). Tier **T0** (a head's reads on recorded frames against the label; a diagnostic of the head, not driving performance). Three head fits on existing tokens of the NVIDIA train pool (1,679 clips of the list of card ADD106) and three dumps on the 141 evaluation clips; no encode, no rollout, no model training.

## 1. Why this card

Card ADD115 reads `PRESATTRC10K-CONV-BOTH-CARRY-UNSAFE`: at the convention the unsafe rate of the candidate above the label form (0.0569 against 0.0320) comes with both inputs of the rule. The reader's presence in the place of label presence raises it by +0.0080 [+0.0009, +0.0171] at the camera gap and by +0.0107 [0.0000, +0.0232] at the label gap, and it lowers falling behind at the label gap: the pattern of leads that are not marked. Card ADD116 measured that presence at T0: it misses 0.1558 of the frames with a lead of the label and marks 0.2244 of the frames with none, and a threshold has no room. Row 180 committed the lever and its bar before this result: a presence head whose miss rate is at most half of 0.1558 with at most twice 0.2244 extra marks on these frames. By the committed reading of card ADD115 that lever is taken up now.

The inputs exist: the tokens of the two narrow windows straight ahead (arm D of card ADD99; on every frame, placed with no label) for 1,679 train clips and the 141 evaluation clips. The gap head of card ADD106 reads them on frames with a lead only and has no presence output; the token reader of the candidate reads a 4x10 grid of the full frame. The misses of that reader sit at large gaps (row 180), where a narrow window has more pixels on the lead.

Stated guess before data: the head alone does not reach the bar (it sees only the lane straight ahead); the head joined with the reader by OR cuts the misses by a third to a half and lands near the bar, on either side of it.

## 2. Fits and forms

Fit tool: a generated variant of `crop_fit.py` (61bf2f4e8e7ec61f, unchanged) made by asserted substitutions and nothing else: the label of a frame is 1 when the label marks a lead with a finite gap from 0 to 60 m and 0 else; the samples of a role are all frames f >= 4 of its clips (not only frames with a lead); the head's output is one logit (no 60 * sigmoid); the loss is binary cross-entropy on the logit with weight 1 and no class weight; selection by the loss on the sel clips. Arm D, head H5 (frames f-4 to f), roles fit and sel as in the train list of card ADD106. Hyper-parameters of card ADD99 (learning rate 1e-3, weight decay 0.05, batch 256, token dropout 0.1, width 256, 4 attention heads, warm-up 1 epoch) with **8 epochs** in the place of 40 (three times the samples per epoch; the gap fits of card ADD106 kept epochs 13, 1 and 2 of 40). Fits: seed 0, seed 1, and seed 0 with the labels permuted across the train frames. p = sigmoid(logit).

Dump tool (new, imports the generated fit tool): applies a head to the tokens of the 141 evaluation clips and writes p per row of the evaluation join (row = off[clip] + frame; not a number for frames 0 to 3).

Frames of the read: those of card ADD116, per model seed m the 10,865 frames with a full history of the two rule logs of the banked reader-fed run `I10RQ50KF5Hb20_s<m>`; head seed m is read on the frames of model seed m. Label of a frame as in card ADD116. Forms, each marking a frame when its value is at least 0.5:

* **READER** — the reader's p (identity with row 180);
* **HEAD** — the new head's p;
* **HEAD-OR-READER** — the larger of the two.

## 3. Measures and estimator

Per form: miss rate M (share of frames with a lead of the label that are not marked), extra-mark rate X (share of frames with none that are marked), per seed and pooled (mean of the two seeds); M per gap bin (0-10, 10-20, 20-40, 40-60 m); area under the curve for READER and HEAD. Differences HEAD − READER and HEAD-OR-READER − READER in M and in X. Intervals: bootstrap over the evaluation clips, 2,000 draws, the same draw for both seeds and all forms. Printed with no name: for HEAD the miss rate at the threshold of the grid 0.1 to 0.9 whose extra-mark rate is nearest to the reader's; the permuted fit's values.

Four families: this card reads no driving measure. LONGITUDINAL, LATERAL, TACTICAL and STRATEGIC are not computed at T0 for a presence head (reason: no rollout; n = 0 windows); they are read in the T1-kin card that follows (section 6).

## 4. Names

The bar of row 180: pooled M at most 0.0779 (half of 0.1558) and pooled X at most 0.4488 (twice 0.2244). Per form F in {HEAD, HEAD-OR-READER}:

* `PRESHEAD-T0-<F>-MEETS-BAR` — both parts of the bar hold;
* `PRESHEAD-T0-<F>-CUTS-MISSES-NOT-TO-BAR` — the bar does not hold, the interval of M(F) − M(READER) lies below 0, and pooled X is at most 0.4488;
* `PRESHEAD-T0-<F>-NO-CUT-OF-MISSES` — else.

One overall name: `PRESHEAD-T0-BAR-MET` (a form meets the bar; of two the one with the lower pooled X is named) / `PRESHEAD-T0-MISSES-CUT-BAR-NOT-MET` (no form meets it, a form cuts the misses) / `PRESHEAD-T0-NO-CUT`.

## 5. Controls (all have to pass before any name or value is opened)

* **p0 tool** — the generator's record: the source has the hash named above, every substitution was applied exactly once, and the variant differs from the source in the listed lines only.
* **p1 lists** — each fit record names the train list and the evaluation list with the hashes of the fit records of card ADD106 (the tool itself refuses an evaluation clip in the train list).
* **p2 fits** — three finished fits with the tool hash of the generated variant, the hyper-parameters of section 2, both classes among the fit samples and among the sel samples, and a finite kept loss.
* **p3 permuted labels** — the area under the curve of the permuted fit on the frames of the read lies between 0.40 and 0.60 for both frame sets.
* **p4 identity** — form READER reproduces row 180 within 1e-4 (pooled M 0.1558, X 0.2244) and its counts (10,865 frames per seed, 3,383 with a lead, 7,482 with none).
* **p5 coverage** — each dump has one row per row of the evaluation join, names the hash of its head, and has a finite p on every frame of the read.
* **p6 tests** — the fit kit's test passes in the lead's run before the fit queue starts; the reader's test passes in the lead's run before the controls stage; outputs outside the kit folders.

This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

Order of work, stated before data: the fit kit is built, tested and dry-run by the lead and placed on the pod to start when the queue of card ADD117 has written its done marker (a stop marker of that queue starts nothing); the reader is built while the fits run. No value of a fit beyond its log lines of loss per epoch is read before the reader's tests pass and the controls are logged.

## 6. Committed readings

* **BAR-MET** → the named form goes to a T1-kin card at the convention and at readout K with no pre-roll: the candidate with its presence input exchanged for the named form (a study wrapper generated from the unchanged wrapper, with its own identity control), against the candidate and against the cell with label presence and the camera gap, which bounds what a presence input can give (+0.0080 on the unsafe rate at the convention).
* **MISSES-CUT-BAR-NOT-MET** → reported as a missed bar. The same T1-kin card is still run with the form of the lower miss rate, as a study arm with the missed bar stated beside it, because the driving measure decides and four rollouts cost about 0.35 pod-h.
* **NO-CUT** → the existing narrow-window tokens do not carry a better presence; the presence part of the unsafe rate at the convention stays a stated limit, and what remains is a new encode (the PI's decision, with its cost).

A presence input of the rule is a design decision: as a study arm it is measured; its adoption goes to the PI and the other session. Inference stays vision only: the label enters the fits as the target and the read as the reference, never the rule.

## 7. Cost and limits

ESTIMATED 1.5 pod-h GPU for three fits of 8 epochs (about 335,000 samples per epoch; the gap fits of card ADD106 passed about 1,560 samples per second) and three dumps; seeds 0 and 1 may run side by side. Stop rule 4 pod-h from the queue start. Limits: T0 only, not driving performance; the label marking is a reference and not ground truth (a frame the label does not mark may hold a vehicle ahead); two head seeds read on the frames of two model seeds; one head form and one token arm; the bar was set on the same 141 clips on which it is read; a form with OR can only lower the misses and raise the extra marks, so its test is the bar and not the sign; two forms have two chances at the bar.
