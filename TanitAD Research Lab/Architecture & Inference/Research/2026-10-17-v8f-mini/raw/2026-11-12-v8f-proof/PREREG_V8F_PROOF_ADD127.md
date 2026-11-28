# PREREG_V8F_PROOF_ADD127 — BOXREAD-T0: the gap to the lead from a box read at pixel level

Tier T0: recorded frames, no rollout; a diagnostic of a perception read and not driving performance. Written after row 191 (card ADD124), whose committed reading names a box read at pixel level as the lever of the camera gap, and after the go of the PI of 2026-11-23 (verbatim: "Yes", to the question "Do you want me to download the detector weights and run the box-read study arm?"). Frozen before any tool of this card, before the weights are on the pod and before any detector output exists. A study arm: its adoption into the perception path is a decision of the PI and the other session and is not part of this card.

## 1. Question

Does the box of the lead, read from camera pixels by a pretrained detector with no training on our data, give a better gap than the head that regresses metres from pooled crop tokens (arm D of row 172: 3.960, 3.913, 4.112 m on 3,841 frames of 141 evaluation clips)?

## 2. Material

* Detector: `fasterrcnn_resnet50_fpn_v2` of torchvision with its COCO weights (one file of about 170 MB, the one download the PI approved), used as it is. Classes counted as a vehicle: car, motorcycle, bus, truck. Score at least 0.5.
* Frames: the staged videos of the 641 pilot clips of card ADD99 (500 train, 141 evaluation), the frames of the token cache (5 Hz), the lists of the crop study unchanged.
* Wide read: the fixed window of 32 degrees of arm D (centre 0 degrees left, 6 degrees down), rendered from the wide camera by the window tool of card ADD99 at 768 pixels (24 pixels per degree). Tele read: a fixed window of 16 degrees at the centre of the tele camera of card ADD122, rendered at 1,024 pixels (64 pixels per degree), where the clip has a tele video.
* Labels (fitting and bound arms only): the label box of the lead (direction and angular size, the box files of cards ADD99 and ADD124) and the label gap.
* Direction of the lead at inference: the locator of card ADD107 (vision only), its banked predictions on the evaluation frames.

## 3. Arms

* Pick. Among the vehicle detections of a frame, the one whose box contains the direction, or else the one whose box edge is nearest to it within 1.5 degrees; if several contain it, the one with the smallest box. No such detection: no pick.
* Size. The angular width and height of the picked box, from its window pixels through the window geometry.
* Map. Size to gap by the form of the map G of card ADD124: the 25 nearest train frames in the two log sizes, the median of their label gaps. It is fitted on the train clips on detector boxes picked by the LABEL direction (labels are allowed in fitting), separately for the wide and the tele read.
* `BW` (candidate form, vision only): on the evaluation frames the pick by the locator's direction in the wide read, the gap from the wide map; a frame with no pick takes the gap of arm D.
* `BT` (candidate form, vision only): the pick in the tele read where the frame has one, the gap from the tele map; else as `BW`.
* `BL` (bound, not a candidate): as `BW` with the pick by the label direction.
* Arm D enters in its three banked fits; `BW`, `BT` and `BL` are formed against each of the three, and the three values are printed singly and as their mean.

## 4. Measures and names

All controls of section 5 have to pass before any name or value is opened. This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

Measure: the mean absolute gap error on the 3,841 evaluation frames with a label gap, the rows of row 191. R(X) is the mean over the three fits of the error of arm X divided by the mean of the three errors of arm D. Interval: cluster bootstrap over the 141 evaluation clips, 2,000 draws, fits held fixed (the estimator of the reader of card ADD124). It answers what another draw of clips would say, not what another fit would say.

* Wide: `BOXREAD-T0-WIDE-BETTER` when the interval of R(BW) lies below 1 and R(BW) is at most 0.90; `BOXREAD-T0-WIDE-SMALL-GAIN` when the interval lies below 1 and R(BW) is above 0.90; `BOXREAD-T0-WIDE-WORSE` when the interval lies above 1; `BOXREAD-T0-WIDE-NOT-SEPARATED` else.
* Tele: the same four names with `TELE` for R(BT), and `BOXREAD-T0-TELE-ADDS` when the interval of the error of `BT` divided by that of `BW` lies below 1, `BOXREAD-T0-TELE-DOES-NOT-ADD` else.
* Far leads (label gap 40 to 60 m, where arm D reads 6.25 m short): `BOXREAD-T0-FAR-LEADS-LESS-SHORT` when the paired difference of the signed error, `BW` minus D, has an interval above 0 and the signed error of `BW` is at most half of that of D in size; `BOXREAD-T0-FAR-LEADS-STAY-SHORT` else.
* Beside, naming nothing: the bound arm `BL`; the share of evaluation frames with a pick, in all and by label gap bin; the error on the picked frames alone against arm D on the same frames; the error of the detector's sizes against the label sizes.

## 5. Controls

* b0 inputs: the sha256 of the weights file begins with the eight characters that its published file name carries; the clip lists are those of the crop study; the three banked predictions of arm D give their errors of row 191 within 1e-3 m on the same 3,841 rows.
* b1 geometry: a rectangle drawn at a known direction and angular size into a synthetic camera frame comes back from the window render and the size formula within 0.1 degree (tool test), and on the real train frames the label direction of the lead falls inside the wide window on at least 0.90 of the frames with a lead.
* b2 the detector is live: on train frames with a lead at a label gap of at most 20 m the pick by the label direction exists on at least 0.80 of the frames in the wide read; with the label directions mirrored left to right the share with a pick falls to at most half of that.
* b3 the map is live: fitted on four fifths of the train clips and read on the fifth (five folds by clip), the wide map's gap error on picked train frames is at most 0.90 of the error of the constant (the median train gap); fitted on permuted gaps it is at least 0.95 of the constant's.
* b4 vision only: the arms `BW` and `BT` read no label at inference (the tool takes the locator's files and refuses a label file in that place; tool test), and the locator's predictions are the banked ones of card ADD107 (hash).
* b5 instrument: at least 0.50 of the 3,841 evaluation frames have a pick in the wide read.
* b6 tests: the tests of the kit pass before the read, with the test output older than the read.
* b7 one evaluation set: every arm is read on the same 3,841 rows of the same 141 clips.

Each control clause is rehearsed before the read on the train clips or on stand-in data; a clause found to fail by construction or an open definition goes into a note written before any value of the evaluation clips is opened. The bars of b2, b3 and b5 are set without knowledge of the detector's output on these frames; if one fails, the card fails as written and the reading beside it says why.

## 6. Cost and stop rule

Two detector passes over about 17,000 frames each. Estimate 0.5 GPU-h on the fork pod; stop rule 1.0 GPU-h in all. One download of about 170 MB into the card's own folder on the fork pod; nothing is installed into a Python environment. The fork pod is shared: the state file is set to busy for the GPU passes.

## 7. Committed readings

* `WIDE-BETTER` (or `TELE` better with `TELE-ADDS`): a pixel-level box read is a lever of the camera gap at T0. Next, each a decision of the PI: the gap files of the candidate's rollouts from this read, so that the lead rule can be run with it at T1-kin (a new encode pass over the rollout windows), and whether a detector enters the perception path at all.
* `SMALL-GAIN` or `NOT-SEPARATED`: a detector with no training on our data does not beat the metre head by the bar; the beside values (picked frames alone, size error against label sizes, the bound arm) say whether the pick, the size or the coverage limits it, and the next arm is named from them.
* `WORSE`: as written; the same beside values name the cause.
* `FAR-LEADS-LESS-SHORT` or `STAY-SHORT`: set against rows 189 and 191.

## 8. Families

LONGITUDINAL (distance keeping) is the family this read serves; at T0 it is measured as the gap error only. LATERAL and TACTICAL are not measured. STRATEGIC is not computed, n = 0: a gap read on recorded frames reads no strategic quantity.

## 9. Limits

One detector, one set of weights, no training on our data; 141 evaluation clips; the map is fitted on 500 train clips; the tele read exists only where the lead is inside the tele window; frames with no pick take arm D's gap, so the arms are mixtures and the share of picked frames is part of every reading. Sheets with camera frames and boxes are private and stay off the repository.
