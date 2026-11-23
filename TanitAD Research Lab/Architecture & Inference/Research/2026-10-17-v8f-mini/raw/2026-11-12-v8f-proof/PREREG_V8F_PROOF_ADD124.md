# Card PREREG_V8F_PROOF_ADD124 (BOXSIZE-T0): does a camera head that reads the angular size of the lead's box give a better gap than the head that regresses metres?

Card of the v8F proof study, written by the coordinator fork. Frozen before any tool of this card exists and before any of its fits (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Tier: T0 (recorded frames, no rollout; a diagnostic of a perception head, not driving performance). No new encode, no download: head fits on the tokens that card ADD99 wrote for its pilot (500 train clips, 141 evaluation clips), on the pod of this fork.

## 1. Why

Row H-V8F-PROOF-189 (MEASURED): a head that also reads the tele camera is worse than the wide-only head, and where the lead lies inside the tele window four times the pixels change nothing. The same row reads, post hoc and on label tracks alone, that the angular size of the lead's box fixes the gap to 1.611 m (1.45 m short at 40 to 60 m) against 3.9 to 4.1 m for the camera head on the same pilot and 1.9 to 2.1 m for the head with windows placed by the label (row 171, a bound). The camera heads so far regress metres from pooled crop tokens; none is asked for the box. The locator of card ADD107 showed that a head on these tokens can read where the box is (row 173). This card asks whether it can read how large the box is, and whether the gap that follows from the size is better than the regressed gap.

## 2. Arms

* Arm D (reference, banked, read from the banked dumps): the three fixed-window fits of the gap head on the pilot, 3.960, 3.913 and 4.112 m on 3,841 frames.
* Arm S: the size head. Tool: the locator tool of card ADD107 (`locator_fit.py`, used unchanged: the head of card ADD99 with two outputs, smooth L1, its sample builder, its fit settings), imported by a new tool that sets ONE value of the imported module, the function that loads the box file. In the two places where the locator reads the direction of the box centre it is given the encoded size of the box: 16 u_w and -2 + 8 u_h with u = ln(angle in degrees) - 1.6, for the angular width and height of the lead's box. The locator's targets are then u_w and u_h, limited to -2 to +2 (0.67 to 36.6 degrees; 0.3 per cent of the widths and 0.4 per cent of the heights of the label tracks lie outside), and its read-out returns the same encoding. Samples: the frames the locator takes (a lead in the label, a finite box centre, a full five-frame history) that also have a finite positive width and height. Fit seeds 0, 1, 2: `S_0`, `S_1`, `S_2`.
* Arm SP (control): the size head fitted with permuted targets (the locator tool's own flag), fit seed 0.
* The map G from size to gap, fixed before any fit: on the label tracks of the 500 train clips, the frames with a lead, a finite label gap and a finite positive width and height; features ln(height) and ln(width), each scaled by its mean and standard deviation over those frames; for a query the 25 nearest train frames by Euclidean distance; the read is the median of their label gaps. The gap of arm S on a frame is G at the decoded width and height of the head's output. No label of an evaluation clip enters G or a fit.
* Budget: ESTIMATED 0.5 GPU-h (four fits of the size of the locator fits of card ADD107, each with its prediction), beside the trainers of the other session on the same pod. Stop rule: 1.5 GPU-h.

## 3. Measures

On the 3,841 evaluation frames of control t3 of card ADD122 (a lead of the label within 60 m and a finite gap in the three banked dumps of arm D), 141 evaluation clips:

* E: mean absolute gap error per fit; ratio of the mean of the three fits of arm S to that of arm D, with a cluster bootstrap over the evaluation clips (2,000 draws, fits held fixed). The interval answers what another draw of clips would say, not what another fit would say; the three fits are printed singly.
* Signed error and share more than 5 m short at 40 to 60 m; error by bin of the label gap (0 to 10, 10 to 20, 20 to 40, 40 to 60 m).
* Printed beside, judged by nothing: the error of the head's size read (mean absolute error of ln width and ln height against the label, as per cent of size), by bin of the label gap; the error of G on the label sizes of the same frames (the bound of a perfect size read); the share of frames whose label size lies outside the limits of the encoding.
* Four metric families: LONGITUDINAL, LATERAL, TACTICAL and STRATEGIC are not read by this card (n = 0 each; a T0 read of a perception head produces no trajectory). A rollout card that reads all four follows only if the bar is met.

## 4. Names (committed before the data)

* `BOXSIZE-T0-CUTS-GAP-ERROR` when the interval of the ratio S to D lies below 1; `BOXSIZE-T0-WORSE` when it lies above 1; else `BOXSIZE-T0-NOT-SEPARATED`.
* `BOXSIZE-T0-FAR-LEADS-READ` when the mean of the three fits of the signed error at 40 to 60 m of arm S is at most half of arm D's in size and the paired difference is separated; else `BOXSIZE-T0-FAR-LEADS-STAY-SHORT`.
* Bar for going on: a ratio of at most 0.85 with its interval below 1.

## 5. Controls (all have to pass before any name or value of section 3 is opened)

This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

* s0 (lists and tokens): the train and evaluation lists and the token folder are those of the banked fits of arm D (hashes of the banked fit record).
* s1 (identity): the locator tool's own identity run, started through the new tool, reproduces the selection loss of the banked fit of arm D within 1e-4.
* s2 (one value changed): the record of every fit lists exactly one changed value of the imported module, the box loader, and the hash of the imported tool; encoding followed by decoding returns the label sizes of the evaluation frames within 1e-9 inside the limits.
* s3 (reference): the banked dumps of arm D give 3.960, 3.913 and 4.112 m on the 3,841 frames within 0.001 m.
* s4 (the map): G on the label sizes of the 3,841 frames has a gap error of at most 2.2 m (the post-hoc read of row 189 gave 1.611 m on a frame set close to it; a larger value says the map or the join of rows is wrong), and G on label sizes permuted across the frames has an error of at least 0.95 of that of the constant (the median label gap of the 3,841 frames).
* s5 (permuted targets): the gap through arm SP has an error of at least 0.95 of that of the same constant.
* s6 (tests and counts): the tests of the kit have passed before the fits and before the read (file times); every one of the 3,841 frames has a finite size output of each fit.

## 6. What follows (committed before the data)

* **Bar met** → the size read is the gap source to carry on: the same head fitted on all 1,679 eligible train clips (tokens exist, card ADD106), a read of the size head on the tele tokens of card ADD122 (size is where pixels count: at 50 m the box is 28 pixels high in the wide camera and 112 in the tele camera), and a T1-kin rollout study arm of the lead rule fed by it (four families). The adoption of a changed gap source in the candidate is a decision of the PI and the other session.
* **Cuts the error but misses the bar, or FAR-LEADS-READ alone** → reported as written; the fit on 1,679 clips is run once as the next arm.
* **Not separated or worse** → the size read printed beside says which part fails: if the head reads the size with an error that G turns into more than the error of arm D, pooled crop tokens of a frozen encoder do not carry the box size finely enough, and the lever is a box read at pixel level (a detector on the camera frames, a design step for the PI and the other session with its cost); the camera gap of the candidate stays the wide head of row 172.

## 7. Order of work

Kit (the new tool, a queue script, a reader with its tests) after this freeze; tests on the pod; identity run; four fits; the reader's controls are logged in `TIMES_NOTE.md` first and names and values are opened after that line.

## 8. Limits

T0 only; 500 train clips; the two fixed wide windows (a lead outside both is read from the 32-degree window alone); three fit seeds; the map G is fitted on label boxes and applied to read sizes, so an error of the read is not corrected by it; the label box is a projected cuboid, not a drawn box, so its size is exact where a drawn box would not be; the frames of the read are those where the label has a lead, and presence is not part of this card.
