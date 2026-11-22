# Note N1 to card ADD106 (CROPSCALE10K): open definitions, fixed before any read

Card: `PREREG_V8F_PROOF_ADD106.md` (sha256 begins 4a56153b9a18a12f). This note is frozen with its own hash file before the encode of the card has ended, before any fit of the card exists and before any reader of the card has run on data. It changes no control, no bar and no name of the card. Where the card leaves a definition open, the reading below is the one the reader (`cropscale/read/read_cropscale10k.py`, sha256 begins 752c06df6bcb9c7c) carries; where the reader adds a clause to a control, the clause makes the control stricter and is listed here.

## 1. The T0 read (tier T0, a diagnostic of the head, not driving performance)

1. The four T0 names of the card are four values of one name, chosen in the order the card gives them. An upper end exactly at 1.0 is not below 1.0; a lower end exactly at 1.0 is not above 1.0.
2. Row set of the ratio: the rows of the evaluation join with a lead in the label (a finite label gap) and a finite gap output of all eight heads (the 500-clip fits of fit seeds 0 and 1 of card ADD99 and of fit seeds 2, 3, 4 of card ADD104; the large fits L2, L3, L4). The row count is printed.
3. The second clause of `CROPSCALE-T0-GAP-ERROR-CUT` ("each of the three large fits has an error below the smallest error of the five 500-clip fits on record") is read on the row set of point 2, so that a large fit and a 500-clip fit are compared on the same rows. The recorded value of 3.913 m on the 3,841 rows of card ADD105 is printed beside it and carries no name.
4. Interval: two-sided 95 % percentile interval; `numpy.random.RandomState(106)`; 2,000 draws; in each draw 141 evaluation clips are drawn with replacement from all 141 (a clip with no row of the row set adds nothing); statistic: the sum over the drawn clips of the mean absolute error of L2, L3, L4 divided by the same sum for the 500-clip fits of seeds 2, 3, 4. The fits are held fixed, as the card says: the interval speaks about another draw of evaluation clips and not about another fit; the second clause of point 3 is the card's guard for the fit noise.
5. No interval is given when fewer than half of the draws have a finite ratio; the reader then stops and gives no T0 name.

## 2. The T1-kin read (tier T1-kin/hb0.2, kinematic rollout on the model's own actions, pre-roll of 30 steps (6 s))

1. The count rule of the card name (`CROPSCALE10K-HOLDS`, `-DOES-NOT-HOLD`, `-NOT-DECIDED`) counts G2, G3 and G4 only, as the card writes. GEc has its own arm names and does not enter the count.
2. Half the gap H is half of the pooled point value of contact of RR minus contact of RLc on the banked rollouts (0.0477), as in cards ADD104 and ADD105. The names of record use H from the point values; the reader also prints, with no name, each arm's half-the-gap reading with H = 0.0477 as the card prints it and says whether the two differ. Control g6 holds the two contact values to 0.0954 and 0.0000 at four decimals.
3. The mean pair GEc minus E3c has no count in the card. Reading: per measure, `MEAN-LOWERS` when the upper end of the interval of the pair is below 0, `MEAN-RAISES` when the lower end is above 0, else `MEAN-<M>-NOT-SEPARATED`. The interval is the engine's two-level bootstrap (model seeds, evaluation episodes; 2,000 draws), as for every other pair of the card.
4. The pair E3c against RLc is read in the grid so that falling behind against the label gap form is printed for all five arms. It carries no name.
5. All intervals of this tier come from two model seeds. A separated interval of a pair says that another draw of evaluation episodes and of these two model seeds would read the same; three fit seeds stand for the fit noise of the head. The planner of the tier does not sample.

## 3. Controls: what is read again, and added clauses (each makes its control stricter)

* g0: read again in full from the files on disk; the flags of the encode kit's record have to say PASS as well.
* g1: the flags of the encode kit's record have to say PASS, and the cheap clause is read again from disk: the first three clips of the pilot's train list, encoded again by the call of this card, against the pilot's tokens of arm D within 1e-3. The hash of every token file (about 49 GB) is taken once, by the encode kit; the reader does not take it again. The check folder on the pod stays in place until the read is done.
* g2 and g3: derived again from the fit records, head files, lists, dump metas, gap files and dumps on disk; the hashes have to be those of the build record.
* g4: ten run texts are required (five arms by two model seeds).
* g5: the rule log of every arm is compared with the gap file of that arm as it lies on disk, and its meta has to name the hash of that file and the closing input as switched on.
* g6: the seven recorded contact values are compared after rounding to four decimals.
* g7: three test outputs (encode kit, fit kit, reader kit), each with a count of passed tests above 0 and none failed.
* g8: the engine's controls on the ten rollouts, the reference rollouts and the control run. Stated limit, as in cards ADD103 and ADD104: whether an engine control assumes a closing column of zero for rule logs of the reader-fed form could not be checked from here.
* g9, added clause: the five dumps of the 500-clip fits have to carry the same label fields on the join rows (0 rows differ) and the row count has to be 3,841. The five recorded errors are the values of card ADD105 at three decimals; the tolerance of 1e-3 m of the card is applied to them.
* g10, added clauses: the record of the fit with permuted labels has to say that the labels were permuted, with fit seed 2, arm D, head D-H5 and the fit and selection clips of L2; its head file has to be on disk as recorded and has to differ from the eight heads on record; its dump meta has to name that head file. The ratio is read on the rows with a lead in the label and an output of both heads, with the bootstrap of section 1 point 4 (`RandomState(106)`, 2,000 draws).

## 4. Order of the read

Stage `controls` prints the eleven control lines and no value of the card. The lines are logged in `TIMES_NOTE.md` before stage `names` is run. Stage `names` opens nothing unless g0 to g10 are all present and PASS. The key `t0_diagnostic` of the gap file build record is not opened before the names. With one control failed nothing of the card is opened, the card is reported as failed with no name, and a changed form is a new card.
