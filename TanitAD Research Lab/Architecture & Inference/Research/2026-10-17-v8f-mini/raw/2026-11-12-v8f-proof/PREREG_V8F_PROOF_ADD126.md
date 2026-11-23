# PREREG_V8F_PROOF_ADD126 — DSCALE-PRES-T0: does data scale lift the read of lead presence through pooled tokens

Tier T0: a probe on cached tokens of logged frames. It says what the tokens allow, not what a model does with them, and it is not driving performance. Written after the names of card ADD125 were opened (row 194 written, not yet landed) and before any tool of this card. No new run and no GPU: a CPU read of the banked state files of the runs of 2026-11-20 and of cards ADD123 and ADD125.

## 1. Question

At equal passes (4.0 on both sides), do the union clips (14,512) give a better read of the presence of a lead through pooled tokens than the corpus clips (4,512)?

## 2. What was seen before this card

The verdict files of cards ADD123 and ADD125 print the presence AUROC of every probe as a point value, naming nothing: corpus probes 0.870, 0.883, 0.860, 0.875; union probes at 1.25 passes 0.903, 0.899, 0.910, 0.905; at 2.5 passes 0.917, 0.918, 0.919, 0.914; at 4.0 passes 0.916, 0.914, 0.901, 0.923. So the sign of the per-seed differences is known before this card. Not seen: any interval on a difference of AUROC, any count behind the AUROC, and any value by label gap. This card is therefore a read of a seen pattern in a form fixed before its interval is computed, and the row says so.

## 3. Measures

* A: presence AUROC of `P16E_sK` minus that of `P16C_sK`, per seed, on the same windows, with the AUROC defined exactly as the measure M4 of the banked reader chain (same scores, same labels, same windows). Interval: two-level bootstrap as in the reader of card ADD123 (seed indices, then episodes, the same episode draws of the banked chain), 2,000 draws, with the AUROC computed again on every episode draw.
* Beside, the same form: `P16Q_sK` minus `P16C_sK` and `P16U_sK` minus `P16C_sK`.
* Beside, descriptive: per probe, the share of windows with a lead that the probe scores below 0.5, in all and by label gap (0 to 20 m, 20 to 40 m, 40 m and more), and the share of windows with no lead scored at 0.5 or more.

## 4. Names

All controls of section 5 have to pass before any name or value is opened. This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

* `LEAD-PRES-EQP-SCALES`: the interval of A lies above 0, at least three of the four seeds are above 0, and the mean of A is at least 0.020.
* `LEAD-PRES-EQP-SMALL`: the interval of A lies above 0 and at least three of the four seeds are above 0, but the mean of A is below 0.020.
* `LEAD-PRES-EQP-WORSE`: the interval of A lies below 0 and at least three of the four seeds are below 0.
* `LEAD-PRES-EQP-FLAT`: none of the three above.

## 5. Controls

* p0 inputs: the state files read are those of row 194 (the hashes of the done files and of the states equal the input hashes of `DSCALE_EQP_RESULT.json` 927b06ff7e4a69e9).
* p1 identity: for each of the 18 probes the tool's AUROC point equals the M4 AUROC that the banked reader chain prints for it, within 1e-9.
* p2 estimator: a probe against itself reads exactly 0 with the interval [0, 0]; a swapped pair reads the negative of the pair; scores replaced by the labels read an AUROC of 1; scores replaced by a fixed permutation of themselves read an AUROC whose episode interval includes 0.5.
* p3 instrument: the AUROC rests on at least 50 windows with a lead and at least 50 windows with none, in at least 20 episodes; the counts are printed.
* p4 tests: the tests of the tool pass before the read, with the test output older than the read.
* p5 one evaluation set: all probes are read on the same windows of the same 48 episodes.

Each control clause is rehearsed before the read; a clause found to fail by construction or an open definition goes into a note written before any value is opened.

## 6. Committed readings

* `SCALES`: data scale lifts the read of presence through pooled tokens while it leaves the read of the gap flat (row 194). The models of the 10k line are trained on the union clips, so this is a statement about what the larger clip set gives the reader of presence, not a new lever of the candidate. The beside table by label gap is set against row 180 (presence misses at large gaps) to say where the gain sits. Next question, for a card of its own and with no run asked here: whether the presence input of the candidate's own reader shows the same step against a reader trained on the corpus alone.
* `SMALL`: a gain that is separated and below 0.020; reported as measured, no lever claimed.
* `FLAT`: the per-probe pattern does not survive the interval over seeds and episodes; nothing is claimed.
* `WORSE`: as written.

## 7. Families

LONGITUDINAL (distance keeping) is the family this read serves and is not measured by a token probe. LATERAL and TACTICAL are not measured. STRATEGIC is not computed, n = 0: a token probe reads no strategic quantity.

## 8. Limits

One frozen encoder; pooled tokens; the evaluation windows come from the corpus only; four seeds, and the two-level interval treats them as the draw of seeds. The point values were seen before the card (section 2). An interval above 0 on four seeds is necessary for a name and not sufficient for a claim about training runs at large. A probe is not a model.
