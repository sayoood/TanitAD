# PREREG_V8F_PROOF_ADD127_N2 — note to card ADD127 (BOXREAD-T0): open definitions found while the reader was built

Written after the reader kit was built and tested on synthetic data, while the detector passes run on the pod, before the reader is on the pod and before any detector file has been read by any tool but the pass itself. The card (2510dd8e853d4873) and note N1 (94b3de310cff47bf) are unchanged. No bar and no name is touched.

## 1. Which fit of the locator (sections 2 and 3)

Card ADD107 has three fits of the locator. The card of this study names none. Fixed here: the first fit, `LOC_s0`, its banked prediction file on the pod (`lookat/pred/LOC_s0.npz`, sha256 begins 94e6321943f89df4). The locator reads five frames, so the first four frames of a clip have no locator output and no pick; such a row, if it is among the 3,841, takes arm D's gap like every row with no pick.

## 2. The tele read needs the two extrinsics files

Label and locator directions are in the wide camera's frame (note N1, point 3). The reader carries them into the tele camera's frame with the extrinsics files of the two staged cameras (`cropread/stage/calibration/front_wide_extrinsics.csv`, `telegap/stage/calibration/front_tele_extrinsics.csv`), the files the tele encode of card ADD122 used.

## 3. Control b3

"The constant" is the median label gap of the four fitting folds. The five folds are the train clips by their index in the train list modulo 5.

## 4. Controls b1 and b2, the frames counted

Both count train frames on which the detector was run (frames with a labelled lead, note N1). b2 further keeps frames at a label gap of at most 20 m, as the card says. A detector file with no record of where the detector ran fails b0.

## 5. Control b7

The reader is given the expected counts of 3,841 rows and 141 clips; b7 passes when every arm is read on exactly these rows, the detector was run on the frame of every one of them in the wide read (note N1), and the counts are met.

## 6. The bootstrap

The cluster bootstrap over the 141 evaluation clips uses the draw form of the reader of card ADD124 with 2,000 draws; the seed of the draws is 127 in this reader (the card names none).

## 7. Order of work

The first run of the controls stage on the pod is the run of record. If a clause fails there for a reason of construction (a path, a file form), the reason is written into a further note before any value of the evaluation clips is opened and the controls are run again; if a bar of b2, b3 or b5 is not met, the card fails as written.
