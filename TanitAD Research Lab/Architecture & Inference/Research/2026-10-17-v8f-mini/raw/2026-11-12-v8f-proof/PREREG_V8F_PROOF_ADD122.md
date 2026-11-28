# PREREG_V8F_PROOF_ADD122 — TELEGAP-T0: does a gap head that also reads the front tele camera read the lead gap more exactly? (pilot)

Frozen before any tele frame is encoded and before any tool of this card is written. PI decision of 2026-11-22 (verbatim: "yes for the new encode", in answer to the fork's item "A new encode for a more exact camera gap"). Tier **T0**: a read of a perception head on recorded frames, a diagnostic and not driving performance. No model of the 10k line is trained or changed; no rollout.

## 1. Why this card, and what was looked at before the freeze

The camera gap of the label-free lead rule comes from a head on crops of the front wide camera through the frozen encoder (cards ADD99, ADD106). Its error is 3.96 to 4.11 m at the pilot's 500 train clips and leads at 40 to 60 m are read about 6 m short (rows 170, 171); more clips cut the error to 0.82 of it and left far leads short (row 172); windows at the head's own direction were not separated (row 174). Row 170 left one lever: more resolution on a far lead.

Looked at before this freeze, structure only and no outcome: the crops of card ADD99 are cut from the decoded wide frame of 1080 by 1920 pixels over 120 degrees, 16 pixels per degree, so its 16-degree window of 256 pixels is already at the camera's own resolution and a narrower wide window adds nothing. The source dataset has a front tele camera (30 degrees, 1920 pixels wide, 64 pixels per degree) that the programme's corpus does not hold. One tele clip was fetched by a range read out of its chunk file (6.8 MB, 1920 by 1080, 605 frames) to see that single clips can be staged without the chunk files; the 641 pilot clips lie in 405 chunks. No tele frame was encoded and no gap was read.

Stated guess before data: the four-window head cuts the gap error, by less than the bound of the label-placed windows (ratio about 0.5), and most of the cut is at leads beyond 30 m; fixed tele windows miss leads off the axis, so the mixed pair of two windows may not separate.

## 2. Data, windows and arms

Clips: the pilot lists of card ADD99 unchanged (141 evaluation clips, 500 train clips with its fit and check parts), the frames of each clip that the pilot encoded, labels as in card ADD99 (label gap from the logged tracks; labels may use ego and other agents, inference is vision only). The tele frame of a pilot frame is the tele frame nearest in time, at most 17 ms away; a frame with none is dropped from every arm of the comparison.

Tele windows, fixed in the rig frame like the wide windows (azimuth 0 is the forward axis), nothing but the image: **T8** = 8 degrees, azimuth 0, elevation -2 degrees (32 pixels per degree after sampling to 256 pixels); **T4** = 4 degrees, azimuth 0, elevation -1 degree (64 pixels per degree, the camera's own resolution). Tele intrinsics and extrinsics from the source dataset's calibration tables. Encoder, sampling routine and token format as card ADD99.

| arm | windows | head |
|---|---|---|
| D (reference) | W32, W16 of the wide camera | the three banked fits of the fixed windows on the pilot clips that card ADD108 read against (3.960, 3.913 and 4.112 m), read from the banked dumps, not fitted again |
| M | W16 of the wide camera and T8 | the fit tool of card ADD99 unchanged (two windows) |
| F (the candidate of this card) | W32, W16, T8, T4 | a generated variant of the fit tool with four windows; every other setting as card ADD99 |

Three fit seeds per fitted arm (0, 1, 2), fit on the 450 fit clips of the pilot, read on the 141 evaluation clips. No label enters at inference in any arm.

## 3. Measures and estimator

On the evaluation frames with a lead of the label within 60 m that all arms hold: the mean absolute gap error per fit; the ratio of the means of arm X to arm D over the three fit seeds, with an interval from a bootstrap over the evaluation clips (2,000 draws, clips as units), as in card ADD108; the mean signed error and the share of frames read more than 5 m short for label gaps of 40 to 60 m; the error by gap bin (0 to 10, 10 to 20, 20 to 40, 40 to 60 m). Printed with no name: the share of lead frames whose label box centre lies inside T8 and inside T4, the error of arm F on the frames inside and outside T8, and the correlation of the slope of the read gap with the label closing speed over 25 frames (the requirement of r 0.9 of row 147 is not a claim of this card).

The interval answers what another draw of evaluation clips would say for these fits; three fit seeds on one set of train clips is a small number, and a separated interval is necessary, not sufficient.

Four metric families: this is a T0 read of a perception head. LONGITUDINAL, LATERAL, TACTICAL and STRATEGIC are not read, because no trajectory is produced; they are read by the rollout card that follows if the bar below is met.

## 4. Names and the bar

* For X in (F, M): `TELEGAP-T0-<X>-CUTS-GAP-ERROR` when the interval of the ratio to D lies below 1; `TELEGAP-T0-<X>-WORSE` when it lies above 1; else `TELEGAP-T0-<X>-NOT-SEPARATED`.
* `TELEGAP-T0-FAR-LEADS-READ` when for arm F the mean of the three fits of the signed error at 40 to 60 m is at most half of arm D's in size and the paired difference is separated; else `TELEGAP-T0-FAR-LEADS-STAY-SHORT`.
* Bar for going on, committed here: arm F with a ratio of at most 0.85 and its interval below 1.

## 5. Controls (all have to pass before any name or value is opened)

* **t0 lists** — the clip lists and the frame lists are those of the pilot (hashes of the banked list record).
* **t1 staging** — every staged tele clip is read out of its chunk file with the archive's own check of each member passing, has 1920 by 1080 pixels, and at most 5 per cent of the pilot clips have no tele clip or fewer matched frames than 0.95 of the pilot's frames; the count is logged. Whether every pilot clip has a tele clip was not looked at before this freeze.
* **t2 geometry** — the tele windows are built with the tele camera's own calibration rows of each clip; a check on pixels with no label and no outcome value, run on the pod on two frames each of 100 evaluation clips (the pixels stay on the pod): the T8 window, sampled down to 128 pixels, is compared with the central 128 by 128 pixels of the W16 window of the same frame (the same 8 degrees of the scene seen by the other camera) by a normalised correlation of the grey values; in at least 0.9 of the frames the correlation at the built direction is higher than at each of four tele windows turned by 2 degrees (left, right, up, down). A wrong calibration row or a wrong time match fails this.
* **t3 reference** — arm D's values are recomputed from the banked dumps and reproduce the banked errors of the fixed windows (3.960, 3.913 and 4.112 m on 3,841 frames) within 0.001 m before the common frame set is applied.
* **t4 the variant** — the four-window fit tool differs from the fit tool of card ADD99 only in the declared lines (number of windows and the token count that follows from it); run with two windows on arm D's tokens at fit seed 0 it gives a gap error inside the range of the banked fits of that kind, widened by 0.15 m.
* **t5 permuted labels** — arm F fitted with permuted labels (the switch of the fit tool) reads the gap no better than 0.95 of the error of the constant read.
* **t6 tests and counts** — the kits' tests pass in the lead's run before the encode and before the fits, outputs outside the kit folders; token files exist for every staged clip and window with the format of card ADD99.

This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

Order of work, stated before data: staging and encode kit first (generated from the kits of card ADD99, with the tele camera and the two windows as the only declared differences), then the fit variant and the reader (generated from the kits of cards ADD99 and ADD108); the reader may be built while the encode runs. A smoke on three clips may run before the full encode and opens no value. No value is opened before the reader's tests pass and the controls are logged.

## 6. Committed readings

* **F meets the bar** → the next card stages and encodes the tele windows for all 1,679 eligible train clips and fits the head at that size, then reads the lead rule with that gap file at T1-kin/hb0.2 against the present candidate (a study arm; four families). A tele input would change the sensor set of the lead rule; its adoption is a decision of the PI and the other session.
* **F cuts the error but misses the bar, or FAR-LEADS-READ alone** → reported as written; the next arm is the tele window at the direction of the locator of card ADD107 (a further encode of the pilot clips), not the larger fit.
* **F not separated or worse** → the tele camera at fixed windows is not the lever; the same next arm (locator direction) is run once, and if that is not separated the gap line at T0 ends with the wide head of row 172 as its best form.
* **M alone separates** → reported; a two-window form is cheaper at inference and is carried beside F.

## 7. Cost and limits

ESTIMATED: staging about 5 GB of tele clips by range reads (the chunk files themselves are not downloaded), 0.3 h; encode of two tele windows on 641 clips about 0.7 pod-h and about 17 GB of tokens; six fits and two control fits about 1 pod-h. Stop rule: 4 GPU-h or 40 GB of new files on the fork pod. Limits: T0 only; a pilot of 500 train clips; fixed windows; three fit seeds; the evaluation clips are the 141 clips every card of this line reads on; the tele camera is a second sensor that the programme's models do not read.
