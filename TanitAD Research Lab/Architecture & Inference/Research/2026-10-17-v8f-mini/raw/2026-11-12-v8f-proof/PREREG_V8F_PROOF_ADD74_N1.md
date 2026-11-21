# Note N1 to card PREREG_V8F_PROOF_ADD74 (HANDSRC60): part 2 of the first read stopped on control a1; the card asked for a headway name where the headway instrument is not qualified

Written by the coordinator fork after the first read of the card ended and before any value of part 2 exists and before the names of part 1 were opened (time: the line of this note in `TIMES_NOTE.md`, after the line that logs the stop; hash in the sidecar file).

## 1. What the read reported

Part 1: every control passed. Part 2: the engine controls of the grid read and the strategic stages passed; control a1 ended with an error of the tool (a TypeError while reading the pair AGE-K@P60 of the banked re-read of card ADD70), so no family name was written.

## 2. Cause

The banked re-read of card ADD70 holds no headway share for its pairs on the P = 60 grid. Its note N3 says why: the headway instrument is not qualified there (73 windows with a lead; the instrument's own bar is 200 windows and 20 episodes), and that entry of the banked file is a sentence, not numbers. Control a1 of this reader took numbers from it.

Behind the fault of the tool stands an error of this card: section 3 asked for `HANDSRC60-<M>-VS-OWN-U-…` names and for "LONGITUDINAL with the headway read" on a grid where the headway instrument does not qualify. The fork had fixed that limit in note N3 of card ADD70 and did not carry it into this card.

## 3. Form of record

* On a grid where the headway instrument is not qualified by its own bar (fewer than 200 windows with a lead or fewer than 20 episodes with one) part 2 reports no headway share, no level and no U name; the result says so with the counts. On the P = 60 grid that is the case. The LONGITUDINAL reading there rests on the family cells (speed, along-track) only, and the row says that the headway side is missing and why.
* a1 on that grid: the family cells of the pair own minus fresh read here equal those of the pair AGE-K@P60 of the banked re-read (1e-9); the headway side is not compared because the banked result holds none.
* Nothing changes on the P = 30 grid (325 windows with a lead in 46 episodes): card ADD73 and its row 132 stand as read.
* Added control p1 for the re-read: the names and every cell block of part 1 of the re-read equal those of the first read (1e-9). The names of part 1 are opened from the re-read only.

## 4. The re-read

Reader `integ10k/read_handsrc_decomp.py` with these changes (hash logged in `TIMES_NOTE.md` before the run), new folder `out/read_handsrc60_decomp_n1` on the pod, pod script `integ10k/podread_handsrc60_decomp_n1.sh` (the script of the first read with the folder name changed and the first read given for p1). The first read stays in its folder as read.
