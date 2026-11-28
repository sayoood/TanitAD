# PREREG_V8F_PROOF_ADD130 — BOXFUSE-T0-LOCFIT: the joined read of card ADD129 under the two further fits of the locator

Tier T0: recorded frames, no rollout; a diagnostic of a perception read and not driving performance. Written after the names of card ADD129 (row 198: `BOXFUSE-T0-BETTER`, R(FG) 0.8961, interval [0.8449, 0.9407], with the locator's first fit `LOC_s0`). Frozen before any tool of this card. CPU only on banked files; no GPU, no download, no fit.

## 0. What is known when this card is written

Every value of rows 197 and 198, all of them with the pick of the locator's fit `LOC_s0`. Not known: any value of the box read or of the joined read with the picks of the fits `LOC_s1` and `LOC_s2` of card ADD107, whose prediction files are banked on the pod and have not been read by any tool of cards ADD127 to ADD129.

## 1. Question

The interval of row 198 answers what another draw of clips would say with the fits held fixed. The pick comes from one fit of the locator. Does the joined read stay better than arm D when the pick comes from another fit of the locator?

## 2. Material and arms

Those of card ADD129 unchanged (detector files, map, three fits of arm D, the rule of `FG` with the factor 1.5, the 3,841 evaluation frames of 141 clips), with the locator's prediction file exchanged: `LOC_s1`, then `LOC_s2`. One read per fit.

## 3. Measures and names

All controls of section 4 have to pass, for both fits, before any name or value is opened. This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

Measure and estimator as in card ADD129 (R(FG) against arm D, cluster bootstrap over the 141 clips, 2,000 draws, seed 129, fits of arm D held fixed), per fit of the locator.

* `BOXFUSE-T0-LOCFIT-REPLICATED` when for both further fits the interval of R(FG) lies below 1; `BOXFUSE-T0-LOCFIT-NOT-REPLICATED` else.
* `BOXFUSE-T0-LOCFIT-BAR-3-OF-3` when, further, R(FG) is at most 0.90 for both further fits (the bar that the first fit met); `BOXFUSE-T0-LOCFIT-BAR-NOT-3-OF-3` else.
* Beside, naming nothing: R(FG), R of the box read alone, the share of picks that are the label's pick and the gate shares, per fit; the spread of R(FG) over the three fits of the locator.

## 4. Controls, per fit

* g0 inputs and identity: the controls file of card ADD128 says all pass and its input hashes are the present ones except for the locator's file and its record, which are those of the fit read here and differ from `LOC_s0`'s; the errors of arm D equal the banked values of card ADD128's result within 1e-6 m for each of the three fits; the locator's file hashes to its own banked record of card ADD107.
* g1 the gate is live, g2 vision only, g3 instrument (gate share at least 0.30), g4 tests before read, g5 one evaluation set: as f1 to f5 of card ADD129.
* g6 the tool of card ADD129 is the one of row 198 (sha256), imported unchanged; with the file of `LOC_s0` this card's tool reproduces the errors of `FG` of row 198 within 1e-6 m for each of the three fits (run before the two reads; it opens no new value).

The first controls run on the pod is the run of record. A clause that fails for a reason of construction goes into a note before any value is opened; a missed bar of g1 or g3 fails the card as written.

## 5. Committed readings

* `REPLICATED`: the joined read is better than the token head for each of the three fits of the locator; the claim of row 198 does not hang on one fit of the pick. With `BAR-3-OF-3` the bar of 0.90 holds for each fit; with `BAR-NOT-3-OF-3` the gain is read as "better, about ten per cent, at the bar for some fits and short of it for others".
* `NOT-REPLICATED`: row 198 is read as a result of one fit of the locator, and the decisions of the PI named there wait for a pick that holds across fits.

## 6. Families

LONGITUDINAL (distance keeping) is the family this read serves; at T0 it is measured as the gap error only. LATERAL and TACTICAL are not measured. STRATEGIC is not computed, n = 0.

## 7. Limits

Three fits of one locator on the same train clips, one detector, the same 141 evaluation clips read a third time; the interval per fit answers what another draw of clips would say.
