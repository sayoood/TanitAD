# PREREG card ADD107 (LOOKAT-T0): can a head on the tokens of the fixed windows tell where the lead is?

Fork v8proof, 2026-11-22. Frozen with its hash file before any tool of it exists and before any fit of it is made. Tier T0 only (recorded frames, no rollout; a diagnostic of a perception head, not driving performance). No new encode. All controls of section 5 have to pass before any value or name of this card is opened. This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

## 1. Why (MEASURED, tier T0, rows 169 and 170 and three diagnostics logged in `TIMES_NOTE.md` on 2026-11-22 before this card)

* The lead rule with no label at inference pays for its cut in contact with more falling behind (rows 164, 167, 168). Row 169 ranks a fed gap lying short as the candidate; row 170 finds the short reading at far leads (40 to 60 m: 6.35 m short on average), the same for the mean of five fits, not lowered by a filter over frames.
* `t0_gap_bins_arm.py`: with both windows centred on the lead by the label tracks (arm B of card ADD99, a bound) the gap error of the same head is 2.137 and 1.910 m against 3.960 and 3.913 m for the fixed windows (arm D), and the short reading at 40 to 60 m is -2.99 and -2.18 m against -5.90 and -6.50 m.
* `t0_offcentre.py`: in 486 of 3,841 rows the centre of the lead's box is more than 8 degrees off the axis, outside the fixed 16-degree window; there arm D has an error of 11.34 m (arm B 3.73 m), and at 40 to 60 m it reads 15.29 m short in 107 rows. Those 486 rows carry 0.36 of the summed absolute error of arm D.
* So where the head looks is a ranked lever. Arm B is not admissible at inference (the label says where to look). The question of this card is the first half of an admissible form: a head that reads the direction of the lead from the tokens of the fixed windows themselves. The second half (windows placed at that direction, a new encode) is not part of this card and is the PI's call.

## 2. Data, target, head, fit (nothing new is encoded)

* Clips, lists and tokens of the pilot of card ADD99, unchanged: 500 train clips (450 fit, 50 selection), 141 evaluation clips, tokens of arm D (the two fixed windows; W32 at azimuth 0 and elevation -6 degrees, W16 at azimuth 0 and elevation -2 degrees).
* Target: the direction of the centre of the lead's box, `ctr_az` and `ctr_el` in degrees, of the box files of the pilot (`enc_out/box/<clip>.npz`, from the label tracks), on the frames where `has_lead` holds and both are finite. Labels may use ego and other agents; the input at inference is the tokens only.
* Head: the head D-H5 of card ADD99 (token reader, one cross-attention query, five frames) with the last layer giving two values and no sigmoid. Targets `t_az = clip(az, -32, 32) / 16` and `t_el = clip(el + 2, -16, 16) / 8`; loss: smooth L1 with beta 0.25, mean over both values and the batch. Read-out: `az = 16 a`, `el = -2 + 8 e`.
* Fit: the settings of card ADD99 (AdamW, learning rate 1e-3, weight decay 0.05, batch 256, 40 epochs, one warm-up epoch then cosine to 0, the epoch with the lowest loss on the 50 selection clips kept, token dropout as there). Fit seeds 0, 1, 2. One more fit with fit seed 0 and the targets permuted across the train frames (deliberate regression).
* The fit tool of card ADD99 is not edited; the locator fit is a tool of its own that imports the token store and the token reader of that tool, or a variant generated from it by counted edits. An identity control (k5) ties it to the tool of card ADD99.

## 3. Read (tier T0, the 141 evaluation clips)

Row set: the 3,841 rows of row 168 (a lead in the label of the join with 0 to 60 m and an output of the five 500-clip gap fits); all of them have a box centre (logged). Per row and fit: `d_az = |az_pred - az|`, `d_el = |el_pred - el|`.

* `S4`: share of rows with `d_az <= 4` and `d_el <= 4` (a 16-degree window placed at the prediction has the centre of the lead's box in its central half). `S8`: the same with 8 (inside that window).
* Reference with no fit, the fixed window of arm D (azimuth 0, elevation -2): `S4_fixed`, `S8_fixed` from `|az|` and `|el + 2|`.
* Outside rows: the rows with `|az| > 8` (486 on record). `S8_out`: share of the outside rows with `d_az <= 8` and `d_el <= 8`.
* Interval of `S4 - S4_fixed` per fit: paired cluster bootstrap over the 141 evaluation clips, 2,000 draws, `numpy.random.RandomState(107)`, two-sided 95 % percentile; the fit is held fixed, so the interval speaks about another draw of evaluation clips and three fit seeds stand for the fit noise.
* Printed with no name: medians and 90th percentiles of `d_az` and `d_el`; `S4`, `S8` by bin of the label gap (0-10, 10-20, 20-40, 40-60 m) and by bin of `|az|` (0-4, 4-8, 8-16, above 16 degrees) with the row counts; the same for the fixed window.

Names (one of four):

* `LOOKAT-T0-LOCATES`: for each of the three fits the lower end of the interval of `S4 - S4_fixed` is above 0, and for each of the three fits `S8_out` is at least 0.50.
* `LOOKAT-T0-CENTRES-ONLY`: the first clause holds for each of the three fits and the second fails for at least one.
* `LOOKAT-T0-DOES-NOT-LOCATE`: for at least one fit the upper end of the interval of `S4 - S4_fixed` is below 0.
* else `LOOKAT-T0-NOT-SEPARATED`.

Why 0.50: the outside rows carry 0.36 of the summed absolute gap error of arm D; a locator that brings half of them into its window can remove, at the error of the bound arm on those rows (3.73 m against 11.34 m), about 0.5 m of the 3.96 m. Below one half the second encode would mostly re-centre leads that the fixed window already holds.

## 4. Families

The card reads no T1-kin measure. LONGITUDINAL: not read here; the head is one step before an input of the lead rule (distance keeping), and its effect on that family needs the second encode and rollouts of a later card. LATERAL, TACTICAL, STRATEGIC: not read; a direction head at T0 measures none of them (n = 0 for each, reason: no rollout and no plan in this card). A later card with rollouts carries the four families and STRATEGIC in full.

## 5. Controls (all have to pass before anything is opened)

* k0 (lists and targets): the fits use the 450 fit clips and 50 selection clips of the pilot's train list (list hash as in the fit records of card ADD99); a box file exists for each of the 641 clips with the frame count of its token file.
* k1 (tokens): the token files of arm D of the 641 clips are unchanged against the pilot's manifest.
* k2 (fits): four fit records (seeds 0, 1, 2 and the permuted one) with the settings of section 2, the head files on disk equal to their records, the three real fits not permuted.
* k3 (deliberate regression): the fit with permuted targets locates worse than the fit of seed 0: the upper end of the interval of `S4_perm - S4_seed0` (same bootstrap) is below 0.
* k4 (the reference recomputes): the reader finds 3,841 rows, all with a box centre, 486 of them outside rows, and `S8_fixed` equal to (3,841 - 486) / 3,841 within 1e-9 when no row has `|el + 2| > 8`; if a row has, the count of such rows is printed and `S8_fixed` is computed as defined.
* k5 (identity with the tool of card ADD99): with the gap as the target and the output layer of card ADD99, the locator tool's token path and head reproduce the selection loss of the recorded fit D-H5 seed 0 of card ADD99 at its kept epoch within 1e-4 when loaded with that head file (no fit is made for this control).
* k6 (kit tests): the lead's run of the kit's tests on the dev box passes with none failed.

## 6. Budget and stop rule

ESTIMATED: four fits of about 13 minutes each on the fork pod (the recorded fit D-H5 seed 0 of card ADD99 took 771 s; one GPU, beside the jobs of card ADD106, never more than three GPU jobs), the read on the CPU, under 1 GB of disk. Stop rule: 1.5 pod-hours of this card's jobs.

## 7. What follows (committed before the read)

* `LOCATES`: a decision card to the PI for the second half: the encode of two windows placed at the predicted direction on the 641 pilot clips (ESTIMATED by the pilot's own figures, 0.35 to 1.46 GPU-hours and about 17 GB), then the gap head on those tokens at T0 against arms D and B, then the lead rule at T1-kin with the four families. Nothing of it starts without the PI's go.
* `CENTRES-ONLY`: the same decision card with the alternative beside it: three narrow windows side by side (the outside rows are outside any single predicted window), also a new encode.
* `DOES-NOT-LOCATE` or `NOT-SEPARATED`: reported as written; the next candidate is the direction from the reader's own tokens or a wider narrow window, each a new card.
