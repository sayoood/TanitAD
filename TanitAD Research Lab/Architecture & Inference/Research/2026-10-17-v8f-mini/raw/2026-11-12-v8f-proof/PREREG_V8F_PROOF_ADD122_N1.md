# PREREG_V8F_PROOF_ADD122_N1 — note 1 to card ADD122 (TELEGAP-T0): the frame of the tele windows, the frame time, control t2

Written before any tele frame is encoded, before any clip beyond the one probe clip is staged and before any value is opened. The card (91c3b8283730030b) is unchanged; this note states one statement of the card that the kit of card ADD99 does not bear out, and what the card left open.

## 1. The frame of the windows

Section 2 of the card says the tele windows are fixed in the rig frame like the wide windows, with azimuth 0 the forward axis. Read from the kit of card ADD99 after the freeze: its wide windows are fixed in the wide camera's own rectified frame (azimuth 0 and elevation 0 are that camera's optical axis; no mount pitch is applied), not in the rig frame.

Definition used: T8 and T4 are centred on the directions (azimuth 0, elevation -2 degrees) and (azimuth 0, elevation -1 degree) of that frame of the WIDE camera, carried into the tele camera's own frame by the extrinsics of both cameras of the clip. Only the centre direction is carried; a roll between the two cameras is not applied. So T8 looks where the wide 16-degree window looks, and T4 one degree above it, whatever the two mounts are. The directions per clip are stored with the tokens. A clip with no extrinsics row of either camera is not encoded and counts under control t1.

## 2. The time of a pilot frame

"The frames of each clip that the pilot encoded" are taken at the time of the wide video frame that the pilot decoded for each of its frames (the time table of the staged wide clip), and the tele frame is the one nearest to that time, at most 17 ms away. Seen on the one probe clip, structure only: both time tables are on one clip clock and 30 frames per second, the tele table starting four frame periods before the wide one, so the nearest tele frame is expected at the same instant. A constant offset between the clocks on other clips would show in controls t1 and t2.

## 3. Control t2

The control reads the T8 window at the direction of section 1 against the central 128 by 128 pixels of the W16 window at (0, -2) of the wide frame, as the card words it. The form with both windows at (0, -2) of their own camera is printed for information and decides nothing. The card has no texture condition: a frame of a flat scene counts as not passed, and the control passes at a share of 0.9 as frozen.

## 4. Sharing of the pod

From 2026-11-22 the pod is shared with another session by the PI's decision. The queue of this card starts a GPU step only with fewer than 3 GPU processes and more than 28 GB of container memory free; time spent waiting for that counts into the stop rule of 4 hours, and a stop for that reason is reported as a stop of the queue, not as a result.
