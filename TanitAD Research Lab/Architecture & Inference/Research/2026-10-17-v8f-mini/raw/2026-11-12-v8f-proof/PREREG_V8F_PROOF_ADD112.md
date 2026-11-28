# PREREG_V8F_PROOF_ADD112 — GAPMAP-T0: can a monotone map of the camera gap remove the slow part of its error?

Frozen before any tool of this card is written and before any value of it exists. Tier **T0** (recorded evaluation frames; a diagnostic of the gap reader, **not driving performance**). CPU only: no rollout, no fit of a head, no encode, no pod GPU.

## 1. Why this card

Card ADD111 (row 177) reads `GAPERR10K-PARTS-NOT-SEPARATED`: neither the slow nor the fast part of the camera gap error carries the price in falling behind alone at T1-kin/hb0.2, the trailing median pays safety and the trailing line fit gives no gain. Its T0 block puts the slow part at 5.41 m rms and the fast part at 2.65 m rms on 3,841 error rows, and both smoothing forms leave the error where it was. The reading committed in card ADD111 for this outcome is a recalibrated and smoothed gap as one study arm. A recalibration is a map of the gap value alone. Before fold fits on the train clips (ESTIMATED 1.5 pod-h) and four rollouts (ESTIMATED 0.7 pod-h) are paid, this card measures on the evaluation frames how much of the slow part ANY monotone map of the gap value can remove.

Stated guess before data: little. A head fitted by least squares is already near calibrated in the mean; the slow part is expected to be an offset per scene that the gap value does not carry.

## 2. Inputs (all exist; none is changed)

* base gap file of the candidate form: `VGAPLE.npz` (sha256 begins 55de2f7c), the mean gap of the three large fits of card ADD106;
* evaluation join `join_eval141.npz` (141 clips; label gap and label validity);
* the gap tool of card ADD111, `make_gaperr_gapfiles.py` (sha256 begins 08dbe57f2b6c4107), imported unchanged: its routines give the error rows (label rows with a base output and a finite base gap), the error e = base gap minus label gap, the slow part s (mean of e over the rows of the error run within 5 frames on each side) and the fast part f = e − s;
* the gap-file record of card ADD111 (`GAPERR_GAPFILES_RECORD.json`), for control c0 and c4.

## 3. Maps

Each map sends the base gap g to a mapped gap g'. The label gap enters the fit of every map, and the fit rows are evaluation frames: **no map of this card is admissible at inference and none is a candidate.** They are a bound (IN) and estimates (CF, LIN) of what a map fitted elsewhere could deliver.

* **IN** — isotonic least-squares fit (pool adjacent violators, non-decreasing) of the label gap on the base gap over all error rows, applied to the same rows. In-sample: an upper bound for any monotone map on these frames.
* **CF** — the same fit, cross-fitted over clips. The clips with at least one error row are sorted by name; the fold of a clip is its position modulo 5. The rows of fold k are mapped by the fit on the other four folds. A fitted map is evaluated by linear interpolation between its fitted points (block mean of the base gap, block value), with the end values held outside the fitted range.
* **LIN** — the least-squares line label = a + b·g, cross-fitted over the same folds. Printed with no name.

After a map, e' = g' − label gap on the error rows, and s', f' come from the same routine as s and f, applied to e'.

## 4. Measures (T0)

For the identity (no map), IN, CF and LIN, per range of the label gap as in card ADD111 (near: below 20 m; middle: 20 to below 40 m; far: from 40 m; all): the number of error rows and the rms of e', s' and f'.

`cut_s(M) = 1 − rms_s'(all) / rms_s(all)` and `cut_e(M)` likewise.

Interval: bootstrap over the clips with error rows, 2,000 draws with replacement, generator seed 20261122, percentiles 2.5 and 97.5, the fitted maps held fixed (not refitted per draw). This interval answers what another draw of evaluation clips would say; it does not answer what another fit of the head or another model would say.

Families: this is a T0 read of one input of the lead rule. LONGITUDINAL: the gap error above, the only family this card measures. LATERAL, TACTICAL, STRATEGIC: not measured by this card — no trajectory, manoeuvre or route output exists at T0 on a gap file (n = 0 for each); they are read at T1-kin in the card that follows if it runs.

## 5. Names

Exactly one of:

* `GAPMAP-T0-MAP-CUTS-SLOW-PART` — the lower end of the interval of cut_s(CF) is at least 0.10;
* `GAPMAP-T0-MAP-DOES-NOT-CUT-SLOW-PART` — the upper end of the interval of cut_s(IN) is below 0.10 (even the in-sample bound stays below);
* `GAPMAP-T0-NOT-DECIDED` — neither, or both conditions at once.

The bar of 0.10 is a design choice of this card, stated before data: a cut of the slow part below one tenth is not worth fold fits and rollouts at a resolution of two model seeds.

## 6. Controls (all have to pass before any name or value is opened)

* **c0 identity** — with no map the tool reproduces the T0 block of card ADD111's record (`n_error_rows` equal; `rms_e`, `rms_s`, `rms_f` within 1e-6) for near, middle, far and all.
* **c1 planted worlds** — on a planted world with a known monotone distortion of the gap and noise, IN and CF both give cut_e of at least 0.5; on a planted world whose error is an offset per clip that does not depend on the gap, cut_s(CF) is below 0.05. Both known readings have to come out.
* **c2 isotonic routine** — on planted series of at most 60 points the pool-adjacent-violators output is non-decreasing and equals a second routine (the max-min formula over block means) within 1e-9.
* **c3 folds** — every clip with an error row is in exactly one fold; for each fold the fit rows hold no row of a clip of that fold (count 0); each error row is mapped exactly once.
* **c4 inputs** — the sha256 of the base gap file equals the one in card ADD111's record; the imported gap tool has the sha256 named in section 2.
* **c5 tests** — the test file of the kit passes in the lead's run (a line with a count of passed tests and no failed test), written before the run on the real files; the lead's test output is written outside the kit folder.

This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

## 7. Committed readings

* **MAP-CUTS-SLOW-PART** → the fold card runs: three-fold fits of the head on the train clips with the existing tokens (ESTIMATED 1.5 pod-h, own stop rule), an isotonic map from the out-of-fold gaps of the train clips, gap files recalibrated and recalibrated with the trailing line fit, four rollouts on the two 10k models at T1-kin/hb0.2 (ESTIMATED 0.7 pod-h). A recalibrated gap as input of the rule would be a design decision for the PI and the other session.
* **MAP-DOES-NOT-CUT-SLOW-PART** → the recalibration lever ends here with no pod spend: no fold fits and no rollouts. The slow part is then not a function of the gap value; what is left for it is the reader's own input (where the head looks, row 171's bound; more clips, row 172), each a new encode and a PI decision. The price in falling behind of the camera-fed form stays in the spec as a stated limit of its gap input. The fork then leaves the gap value: the next card attributes the unsafe rate, which rows 176 and 177 leave at 0.0839 with the label gap on every label row against 0.0714 for the label form, to the reader's presence output (label presence with the camera gap of this spec, two rollouts, ESTIMATED 0.35 pod-h, an attribution arm and no candidate; if a banked rollout of that form with the mean gap of the large fits exists on the two 10k models, it is read instead of run).
* **NOT-DECIDED** → the fold card runs as under MAP-CUTS-SLOW-PART: the estimate then does not exclude a useful cut and the fits are the cheapest read that decides it.

## 8. Cost and limits

CPU minutes on the fork pod; no pod GPU. Limits: T0 only; one base file (the mean of three fits); the maps are fitted on evaluation frames and are bounds or estimates, never candidates; the interval holds the maps fixed; 141 clips.
