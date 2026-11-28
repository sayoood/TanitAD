# PREREG_V8F_PROOF_ADD39_N2 : control d4 of note N1 failed as written; its measured cause and the form in which it is read

Note to `PREREG_V8F_PROOF_ADD39_N1.md`. Written after the first run of the read of note N1 (`leadscr/n1/read/`, script sha256 f8478fa86738c0f7d1a27f9fb483009e1bf0dce02d08a5b29f3a5808ddb565a8): the values of GL and GV of that run had been seen. This note changes no statistic, no input and no bar of section 3 of note N1; it concerns control d4 only. Tier T0. STRATEGIC: PAUSED (PI item 25).

## 1. What failed

d4 of note N1: "the head's gap read correlates with the label gap at the start frame more than at the frames 5 steps earlier and 5 steps later (both seeds), and the correlation at the start frame is above 0.3". Read: 0.827 at the start frame against 0.795 five steps earlier and 0.829 five steps later (LS_s0); 0.860 against 0.824 and 0.870 (LS_s1). Above 0.3 and above the earlier frame, not above the later frame: FAIL as written.

## 2. Cause (MEASURED, `leadscr/n1`, a scan of shifts and of label frames run after the failure)

* The comparison across shifts did not use the same windows: the lead head needs 5 frames of history, so the windows that start at frame 3 of a clip have no head read at the start frame (41 of 642 lead windows) and do have one 2 or more frames later (601 windows at shifts up to 0, 642 from shift +2).
* The head reads from 5 frames of history and is smooth in time: its read at neighbouring frames correlates with one label almost equally (0.821, 0.827, 0.834 at shifts -2, 0, +2 for LS_s0; 0.849, 0.860, 0.869 for LS_s1).
* On fixed windows the frames are aligned: the head's read at the start frame correlates most with the label gap of the same frame and less with every later label frame (LS_s0: 0.827 at the same frame, 0.819, 0.807, 0.800, 0.791, 0.771, 0.749 at 2, 4, 5, 6, 8, 10 frames later; LS_s1: 0.860, then 0.852, 0.840, 0.833, 0.824, 0.803, 0.779).

## 3. The form in which d4 is read (d4b)

On the lead windows that have a head read at the start frame (the same windows for every comparison): the correlation of the head's gap read at the start frame with the label gap k frames later is highest at k = 0 among k = 0, 2, 4, 5, 6, 8, 10 at both seeds, and above 0.3 at k = 0. The label frames before the start are not used (the windows at the start of a clip have none).

## 4. How the result is reported

d4 as written: FAIL, with this cause, beside every name. The names of section 3 of note N1 are given with d1, d2, d3, d4b and d5. Since the values had been seen when this note was written, the name carries that statement.
