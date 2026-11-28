# Card ADD108 (LOOKGAP-T0): does the gap head read the gap better with its windows placed at the direction a head finds from the camera alone

Pre-registration, written and frozen with its hash file before any tool of it exists and before any value of it. Fork of the v8 flagship proof line. Date 2026-11-22.

## 1. Question and why

Row 171 (tier T0): with both windows placed on the lead by the label tracks (arm B of the pilot of card ADD99, a bound that is not admissible at inference) the gap head has about half the gap error of the fixed windows (arm D): 2.137 and 1.910 m against 3.960 and 3.913 m. Row 173 (card ADD107, `LOOKAT-T0-LOCATES`): a head on the tokens of the fixed windows finds the direction of the lead with no label. This card asks the second half: with both windows placed at that head's direction, on every frame and with no label at inference (arm P), does the gap head read the gap better than with the fixed windows, and how much of the bound's gain does it reach.

Tier of every value of this card: T0 (recorded frames, no rollout); a property of the head's gap output, not driving performance. A rollout read at T1-kin with the lead rule fed by arm P is not part of this card; it would be a new card, written only if this card reads `LOOKGAP-T0-CUTS`.

## 2. Stages

Stage A, directions, no new encode (tokens of arm D that exist):

* Evaluation clips (141): the direction (azimuth, elevation) on every frame from the locator fit of seed 0 of card ADD107, the head file named in its fit record, read out as in that card (`az = 16a`, `el = -2 + 8e`).
* Train clips (500): directions from fits that did not see the clip. The 500 clips of the pilot's train list are sorted by clip id; clip number i (from 0) is in fold i mod 5. For each fold one locator fit with the tool and settings of card ADD107, seed 0: its fit clips are the clips with the role fit that are not in the fold, its selection clips are the clips with the role selection that are not in the fold. The direction of a clip comes from the fit of its own fold.
* A frame for which the locator has no output (no full history of five frames) gets the direction azimuth 0, elevation -2.
* No filter over frames and no label enters a direction file. The direction files have the format of the pilot's box files, with the direction in place of the box centre and a direction on every frame.

Stage B, encode (a new encode; it starts only with the PI's go): for the 641 clips, the tokens of two windows (32 and 16 degrees, as in arm B) both centred at the direction of the frame, by a generated variant of the pilot's encode tool that writes arm P only. The pilot's encode tool is not edited.

Stage C, fits (on the tokens of arm P): the gap head H5 with the fit tool and settings of card ADD99, the pilot's lists (450 fit clips, 50 selection clips), fit seeds 0, 1, 2; and one fit of seed 0 with permuted labels. Dumps on the evaluation join as in card ADD99.

Stage D, read (CPU).

## 3. Row set and measures

Row set: the 3,841 rows of rows 170 to 173 (frames of the 141 evaluation clips with a lead in the label and an output of the five 500-clip fits of arm D) that also have a finite output of the three fits of arm P and of the two fits of arm B. The reader prints the count; when it is not 3,841 the counts of what is missing are printed.

* E(x): mean absolute error of the gap of fit x against the label gap on the row set. P0, P1, P2: the three fits of arm P. D0, D1, D2: the fits of arm D of fit seeds 0, 1 (card ADD99) and 2 (card ADD104). B0, B1: the two fits of arm B.
* R = mean(E(P0), E(P1), E(P2)) / mean(E(D0), E(D1), E(D2)). Interval of R: paired cluster bootstrap over the 141 evaluation clips, 2,000 draws, `numpy.random.RandomState(108)`, two-sided 95 % percentile; the fits are held fixed, so the interval speaks about another draw of evaluation clips and three fit seeds stand for the fit noise.
* Printed with no name: E of every fit; the share of the bound's gain reached, (mean E(D) - mean E(P)) / (mean E(D) - mean E(B)) with D0, D1 and B0, B1 and P0, P1; the correlation of each fit's gap with the label gap; E, the signed error and the share more than 5 m short by bin of the label gap (0-10, 10-20, 20-40, 40-60 m) for P0, D0, B0; E by bin of the absolute azimuth of the lead (0-4, 4-8, 8-16, above 16 degrees) for P0, D0, B0, with the row counts; E of the two fits of arm B read on the tokens of arm P of the evaluation clips (a head fitted with the windows on the lead by the label and read with the windows at the head's direction; an importing tool, shown first to reproduce the dump of B0 on the tokens of arm B); for stage A, the measures S4, S8 and `S8_out` of card ADD107 on the frames with a lead of the 500 train clips, from the fold fits, beside those of the evaluation clips.

## 4. Names

* `LOOKGAP-T0-CUTS`: the upper end of the interval of R is below 1, and each of E(P0), E(P1), E(P2) is below the smallest error of the five 500-clip fits of arm D on the row set (3.913 m on record).
* `LOOKGAP-T0-CUTS-NOT-EACH-FIT`: the upper end of the interval of R is below 1, and at least one fit of arm P is not below that smallest error.
* `LOOKGAP-T0-WORSE`: the lower end of the interval of R is above 1.
* `LOOKGAP-T0-NOT-SEPARATED`: every other case with an interval.
* With no interval the reader stops with no name.

## 5. Controls (all have to pass before any name or value is opened)

* m0 (directions): a direction file exists for each of the 641 clips with the frame count of its token file; for each train clip the record of its fold fit lists neither the clip as a fit clip nor as a selection clip; the directions of the evaluation clips come from the head file whose hash the fit record of seed 0 of card ADD107 names; no direction file has a field from a label.
* m1 (encode): given the pilot's box files in place of direction files, the variant of the encode tool reproduces the pilot's token files of arm B for three clips byte for byte (the hashes of the pilot's manifest); every token file of arm P has the shape and type of the pilot's token files and its hash is in the manifest of this card; the token files of arm D and arm B of the pilot are unchanged against the pilot's manifest.
* m2 (fits): each of the three fit records names its seed (0, 1, 2), head H5 on arm P, the settings of card ADD99, the pilot's fit and selection lists and no permuted labels; the fourth record names permuted labels and seed 0.
* m3 (deliberate regression): the fit with permuted labels has more gap error than the fit of seed 0 of arm P: the lower end of the interval of E(permuted) / E(P0) (same bootstrap) is above 1.
* m4 (the reference recomputes): the reader finds E(D0) 3.960, E(D1) 3.913, E(D2) 4.112, E(B0) 2.137 and E(B1) 1.910 m on the 3,841 rows within 1e-3 m.
* m5 (the directions are those of card ADD107): on the 3,841 rows the directions of the evaluation clips give S4 0.8748 and `S8_out` 0.5844 of the fit of seed 0 of card ADD107 within 1e-4.
* m6: the tests of the kit pass in the lead's run on the dev box before stage B.

This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

Limits by construction, stated before the freeze: m3 depends on the subject in one case only, a gap head of arm P that is no better than a fit on permuted labels; the card then fails with no name where `LOOKGAP-T0-WORSE` would be the plain reading. m4 and m5 do not depend on the subject. The fold fits see about four fifths of the fit clips, so the directions of the train clips are somewhat less exact than those of the evaluation clips; this makes the fit of arm P harder, not easier.

## 6. Families

T0, no rollout and no plan: LONGITUDINAL, LATERAL, TACTICAL and STRATEGIC are not read (n = 0 for each, with this reason); the measure is a T0 read of one input of the lead rule, which acts on LONGITUDINAL (distance keeping) once it feeds the rule.

## 7. Cost, stop rules, what follows

* Stage A: five fold fits of about 13 minutes each and the direction dumps, on the fork pod; stop rule 1.5 pod-hours of its jobs. No new encode; it may run before the PI's go for stage B.
* Stage B: ESTIMATED from the pilot's own figures 0.35 to 1.46 GPU-hours and about 17 GB; stop rule 2 GPU-hours or 20 GB, whichever comes first. It starts only with the PI's go.
* Stages C and D: four fits of about 13 minutes each and a CPU read; stop rule 1.5 pod-hours.
* `LOOKGAP-T0-CUTS`: a card for ten rollouts at T1-kin/hb0.2 with the lead rule fed by the gap of arm P (the form of card ADD106), to read contact and the price in falling behind. `LOOKGAP-T0-CUTS-NOT-EACH-FIT`, `LOOKGAP-T0-NOT-SEPARATED` or `LOOKGAP-T0-WORSE`: reported as written; the next candidate is named in the row from the values printed with no name (among them the head fitted on the label-placed windows and read with the windows at the head's direction, which needs no further encode).
* Adoption of any input into the candidate spec is not part of this card; it stays a decision of the PI and the other session.
