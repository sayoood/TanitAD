# PREREG_V8F_PROOF_ADD127_N1 — note to card ADD127 (BOXREAD-T0): open definitions found while the detector pass was built

Written after the detector-pass kit was built and tested on synthetic data, before the kit is on the pod, before the detector has run on any frame and before any value exists. The card (2510dd8e853d4873) is unchanged. No bar and no name is touched.

## 1. Which frames the detector reads (sections 2 and 6 of the card)

The card says "the frames of the token cache (5 Hz)" and costs the passes at "about 17,000 frames each". The token cache holds about 101 frames per clip, about 64,700 frames for the 641 clips; the figure of 17,000 is the number of cache frames with a labelled lead, which are the only frames that enter a measure, a map fit or a control of the card (the 3,841 evaluation rows with a label gap, and the train frames with a lead). Reading fixed here: the detector is run on the cache frames with a labelled lead (the lead flag of the box file of the clip) and on no other frame. Every per-clip file records on which frames the detector was run. An evaluation row of the 3,841 on whose frame the detector was not run in the wide read is a missing input and fails control b7; it is not counted as a frame with no pick. This choice of frames uses a label to decide where the study reads, as the measure itself does (it is defined on rows with a label gap); what the detector returns on a frame does not depend on any label, and the pick at inference stays the locator's.

## 2. The detector's input size (section 2)

The detector of torchvision, used as it is, resizes its input to a shorter side of 800 pixels. The wide window rendered at 768 pixels is therefore read at about 25 pixels per degree, and the tele window rendered at 1,024 pixels at about 50 pixels per degree, not 64. The windows are rendered as the card says and the detector's own resize is left as it is; box sizes are taken in window pixels through the window geometry. The wide source has about 16 pixels per degree, so the wide window is an up-sampling by about 1.5.

## 3. Directions (sections 2 and 3)

Elevation zero is the optical axis of the camera, as in the windows of arm D. The tele window is at the centre of the tele camera's own frame, as the card words it. Label and locator directions are given in the wide camera's frame; for the tele read they are carried into the tele camera's frame through the extrinsics, the way the tele encode of card ADD122 carries directions.

## 4. Score

The detector files keep vehicle detections from a score of 0.3; the reader applies the card's bar of 0.5. Values at other score bars, if printed, are beside and name nothing.

## 5. Cost

With the frames of point 1 the estimate of the card stands (two passes over about 17,000 frames each). The smoke stage projects the GPU time from the counted frames; a projection above 1.0 GPU-h stops the queue and goes back to the PI, as the card says.
