# PREREG_V8F_PROOF_ADD122_N2 — note 2 to card ADD122 (TELEGAP-T0): the fit tool, the dropped frames, the constant of control t5

Written before any fit of this card is started and before any value is opened; the staging and the three-clip smoke of the encode have run, the pixel control t2 is running, no token of the full encode exists. The card (91c3b8283730030b) and note N1 (c12a3e5fde570cec) are unchanged.

## 1. The fit tool of the arms M and F

The card speaks of a generated variant of the fit tool with four windows. The kit instead imports the fit tool and the dump tool of card ADD99 unchanged (`crop_fit.py` 61bf2f4e8e7ec61f, `crop_dump.py` b892ef27f3c3ae51) and, for the time of one run, sets five values of the imported module; every other line, the model and all fit settings are those of card ADD99. These five are what control t4 calls the declared lines:

* `N_WIN` and `N_TOK` (arm F only: four windows, 260 tokens per frame);
* `ARMS` (the names M, F and D2 are admitted);
* the token store (it reads the banked wide tokens and the new tele tokens through memory maps and joins them per clip, because joined token files would pass the card's limit of 40 GB of new files);
* the sample builder (wrapped, see section 2).

Arm M reads the windows W16 and T8 in that order, arm F reads W32, W16, T8, T4. Control t4's run is the arm D2: the two wide windows read through the same wrapper with the last three values set and no tele token, fit seed 0. The record of each fit lists the values it set and the hash of the imported tool; t4 fails if a record lists any other value.

The head is H5 of card ADD99, the head of its fixed-window arm. For fit seed 2 the dump tool takes the present-reader dump of seed 0, as the queue of card ADD108 did; that dump enters no measure of this card.

## 2. Frames with no tele frame

A sample of the head is a run of consecutive frames. For the arms M and F a sample whose run holds a frame with no matched tele frame is dropped in the fit and marked in the dump; arm D2 and the banked arm D drop nothing. The measures of section 3 of the card are read on the frames that every arm holds; control t3 is read on arm D before that set is applied, as the card says.

## 3. The constant of control t5

The constant read is the median of the label gap over the evaluation frames of control t3 (the frames with a lead of the label within 60 m in the banked dumps), the constant with the smallest mean absolute error on those frames. Control t5 passes when the head fitted on permuted labels (arm F, fit seed 0) has a mean absolute gap error of at least 0.95 of the error of that constant on the frames it holds.

## 4. Inputs of the fit queue

The lists are those of the banked fits (`cropread/run1/lists` on the pod, equal by hash to the lists the encode used). The fits and dumps are written under the card's own root; nothing is written under the root of card ADD99. The queue's own stop rule (3 hours, waiting for a free slot on the shared pod included) lies inside the card's stop rule.
