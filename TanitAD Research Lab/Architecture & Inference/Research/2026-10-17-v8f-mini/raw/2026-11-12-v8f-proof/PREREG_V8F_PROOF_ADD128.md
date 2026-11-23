# PREREG_V8F_PROOF_ADD128 — BOXREAD-T0-C: the box read of card ADD127, read under a changed set of controls

Tier T0: recorded frames, no rollout; a diagnostic of a perception read and not driving performance. Written after the controls run of record of card ADD127 (2510dd8e853d4873, notes N1 94b3de310cff47bf and N2 60ac04cf4d5d48f2), which failed three of its eight controls. Card ADD127 is reported as failed with no name (row 196). Frozen before any tool of this card is written.

## 0. What is known when this card is written

Of card ADD127's run only the control lines: b0, b3, b5, b6 and b7 passed; b1 failed (the label direction lies inside the wide window on 0.8934 of the train frames with a lead, bar 0.90); b2 failed (pick by the label direction on 0.8859 of the train frames with a lead within 20 m, bar 0.80 met; with mirrored directions 0.7583, above half of 0.8859); b4 failed (the reader compared the full hash of the locator file with the 16 characters given to it; the file's hash begins with those 16 characters); the share of evaluation frames with a pick in the wide read is 0.8995 (b5, bar 0.50). No error, no ratio, no share by gap bin and no name of the evaluation clips has been opened by anyone. The control forms below were written with knowledge of these control values. The row of this card says so, and it is read by the other session before it is landed.

## 1. Question, material, arms, measures, names, readings

Those of card ADD127, sections 1 to 4 and 7 to 9, with its notes N1 and N2, unchanged: the same detector and weights, the same windows, the same pick, size and map, the same arms `BW`, `BT`, `BL` against the three banked fits of arm D, the same measure on the 3,841 evaluation frames of 141 clips, the same cluster bootstrap (2,000 draws, seed 127), the same bars and the same names (`BOXREAD-T0-WIDE-…`, `BOXREAD-T0-TELE-…`, `BOXREAD-T0-TELE-ADDS` or `-DOES-NOT-ADD`, `BOXREAD-T0-FAR-LEADS-…`), the same beside values, the same committed readings. The detector files are those of card ADD127's two passes, used as they are; this card runs no detector, uses no GPU and downloads nothing.

All controls of section 2 have to pass before any name or value is opened. This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

## 2. Controls

* c0 inputs: as b0 of card ADD127, and the detector files are the ones the failed run read (the listing hashes of its controls file).
* c1 geometry, tool test: a rectangle drawn at a known direction and angular size into a synthetic camera frame comes back from the window render and the size formula within 0.1 degree. The share of train lead frames whose label direction lies inside the wide window is printed beside and decides nothing. Why changed: that share is a property of the fixed window's coverage and not of the geometry (a window of 32 degrees holds both signs of a small elevation); coverage is guarded on the evaluation frames by c5, and the geometry on real frames by c2 (ii).
* c2 the detector is live and its boxes agree with the label geometry, on train frames with a lead at a label gap of at most 20 m on which the detector was run in the wide read:
  (i) the pick by the label direction exists on at least 0.80 of these frames (unchanged);
  (ii) on the frames with such a pick the label direction is displaced to the left and to the right by half the angular width of the label box plus 4 degrees, and upward by half its angular height plus 4 degrees. For each of the three displacements the share of frames on which the pick is the same detection as under the label direction is at most 0.10. The clause needs at least 1,000 such frames of at least 100 train clips; with fewer it fails.
  The mirrored share of card ADD127 is printed beside and decides nothing. Why changed: a mirror moves a direction by twice its azimuth. A lead within 20 m sits near the image centre and spans several degrees, so the mirrored direction stays inside its box or inside the tolerance of 1.5 degrees; the clause asked whether some pick exists and could not tell a pick of the lead from a pick anywhere. Clause (ii) asks whether the pick is the same box when the direction has left the label box by more than the tolerance; it fails for a detector whose boxes do not sit where the labels are (a sign or scale fault of the geometry) and for boxes that cover the window.
* c3 the map is live: as b3.
* c4 vision only: as b4; the given hash of the locator file is accepted as a beginning of at least 16 characters of the file's sha256 (the form in which note N2 states it), beside the unchanged check against the banked record of card ADD107.
* c5 instrument: as b5 (at least 0.50 of the 3,841 evaluation frames have a pick in the wide read).
* c6 tests: the tests of the kit of card ADD127 and of this card's tool pass before the read, with the test outputs older than the read. The tests of this card's tool hold two planted defects that have to make c2 fail: boxes that cover the window, and a window geometry with the azimuth sign turned.
* c7 one evaluation set: as b7.

The first controls run of this card on the pod is the run of record. A clause that fails for a reason of construction (a path, a file form) goes into a note before any value of the evaluation clips is opened. A missed bar of c2, c3 or c5 fails the card as written; after a second failed set of controls no third form is written by this fork alone: the private sheet of frames with boxes and the control values go to the PI and the other session first.

## 3. Cost

CPU only on the fork pod, minutes. No GPU, no download, no install.

## 4. Families

LONGITUDINAL (distance keeping) is the family this read serves; at T0 it is measured as the gap error only. LATERAL and TACTICAL are not measured. STRATEGIC is not computed, n = 0: a gap read on recorded frames reads no strategic quantity.

## 5. Limits

Those of card ADD127, and: the set of controls was changed after a failed run whose control values were known; the detector passes read 46,900 frames per camera (card ADD127 estimated about 17,000), at 0.73 GPU-h in all against its stop rule of 1.0.
