# Note N3 to card PREREG_V8F_PROOF_ADD70 (LONGROLL10K): three faults of the read tool found at its first run on data, and the read of record

Written by the coordinator fork after the first pod read of the card ended with three controls not passed and with no name, no level and no difference printed or written (`integ10k/read_longroll/pod/READ_LONGROLL10K.json`: the blocks names, pairs, cells, families, strategic and beside are empty), and before any outcome value of the card has been opened (time: the line of this note in `TIMES_NOTE.md`; hash in the sidecar file). Arms, pairs, estimators, names, consequences and limits of the card and of notes N1 and N2 are unchanged. The first read stays on record as it is.

## 1. What the first read reported

Passed: e1, l1 (tool check on the pod), l2 (P = 0 equals the banked run, 141 files), l3 (state carry: 141 files, 0 differing, largest difference 0.0), l4 to l10, the reader identity l6 (largest difference 0.0 against card ADD67), the reader controls on the P = 30 grid and on the full grid, the strategic rows. Not passed: l12, l11 and the headway reader's own controls on the P = 60 grid. Each is a fault of the read tool, not of a run:

1. l12 (recorder frame, state-carry run against fresh run). The tool compared every key of the two tick files, including the text scalar `tag`, which is the name of the run and differs between two runs by construction; a differing text reads as an infinite difference. The numeric arrays were not shown to differ.
2. l11 (segments). Both P = 60 convention runs report two pre-roll segments per window and the expected 568 segment rows. The tool also required the counter `clipped_windows_in_preroll` to be 0. That counter is the pinned tool's count of plans beyond its speed envelope during the pre-roll (83 and 63 model windows in the convention at P = 60). It is not an index of the rule's arc tables, which cannot run past their end with 30-step segments (note N1, R2). The counter is an observation of the run and belongs to the beside block.
3. The headway reader's own controls on the P = 60 grid. That grid holds 73 lead windows in 37 episodes; the headway reader's count bar is 200 lead windows, and its planted-regression check has no window with a lead label at 6 s there. The instrument is not qualified on that grid. The card gives P = 60 as a beside block that names nothing.

## 2. The read of record (wrapper `integ10k/read_longroll10k_n3.py` around the unedited reader)

* l12 compares every key of the two tick files except the checkpoint scalar and the run name `tag`; tolerance 1e-6 as before; a key present in one file only fails it.
* l11 requires two segments per window and the expected number of segment rows in every P = 60 run. The envelope counter of the pre-roll is reported per run in the beside block.
* On the P = 60 grid the headway shares are not reported (no U, C, F, no level, no spread from that grid): the headway instrument is not qualified there. The family cells and the strategic read of the P = 60 pairs stay in the beside block as the card gives them, provided the family reader's own controls and the window identity pass on that grid; they name nothing.
* Everything else is the reader of note N2, unchanged: the named pairs at P = 30, their controls on the P = 30 grid and the full grid, the names, the consequence rules.

The first result of each replaced control is kept inside the new result under `as_first_read`. If a control of this read fails, nothing is named and the fault is written down before any further step.

## 3. Tools and order

`integ10k/read_longroll10k_n3.py` (the wrapper; the reader `read_longroll10k.py` is not edited and its hash is recorded beside the result) and `integ10k/podread_longroll_n3.sh` (pod, CPU only; reads into the new folder `out/read_longroll_n3/`, result file `READ_LONGROLL10K_N3.json`; control e1 is the file the first read wrote). Their hashes are in the line of this note in `TIMES_NOTE.md`. The wrapper's own check with planted faults passes 16 of 16 items on synthetic files (an array difference and a key in one file only fail l12; one segment and short segment rows fail l11; a failed family control and a window mismatch fail the P = 60 grid; the headway controls stay required on the P = 30 grid; the headway shares of the P = 60 grid are removed and nothing at P = 30 is touched). The headway spread of the P = 60 grid in the block of grid details is removed with the shares. Order: this note and its hash first, then the push to the pod with a hash compare, then the read.
