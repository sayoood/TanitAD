# PREREG_V8F_PROOF_ADD126_N1 — note to card ADD126 (DSCALE-PRES-T0): open definitions and one control clause found in the rehearsal

Written after the tool was built and tested on synthetic data and after one rehearsal of its controls stage on the real files (controls only; that stage computes and prints no value of the measure A, no beside value and no per-probe share), and before any value of the card is opened. The card (51de009306207eb1) is unchanged. No bar and no name is touched.

## 1. The window set of the measure (sections 3 and 5 of the card)

The banked reader chain computes its measure M4, the presence AUROC, on all evaluation windows with a presence label and not on the scored gap windows: 1,844 windows, 534 with a lead and 1,310 with none, from 139 evaluation episodes (counts from the rehearsal, control p3). The 429 scored windows of 48 episodes that cards ADD123 and ADD125 name are the windows of the gap measures. The card fixes A "exactly as the measure M4 of the banked reader chain (same scores, same labels, same windows)", so A is read on the 1,844 windows, with the episode draws that the chain uses for M4 (over the 139 episodes).

## 2. Control p5 (one evaluation set)

The card's clause says "the same windows of the same 48 episodes". For the window set of M4 the number 48 is wrong by construction (139 episodes), and the rehearsal failed on that clause alone within p5. Reading fixed here: p5 passes when all 18 probes are read, their states carry the same evaluation ids in the same order and the same count (the form of control q6 of card ADD123), and the scored gap windows come from the 48 episodes. The number of episodes of the M4 window set is recorded and printed, not judged. The tool is changed in this one clause (read_dscale_pres.py 2732b75f42d49a35 before, 8dc888b593408f90 after; its tests pass before and after).

## 3. Control p0 (inputs)

The rehearsal also failed p0, because the lead called the tool with one input path written in another spelling than in the read of card ADD125 (a path through a parent folder), so the path keys of the two input lists did not match. That was a fault of the rehearsal call. The read of record uses the same path strings as the read of card ADD125 (through `read_args.txt`); p0 is unchanged and has to pass as written. The test output file of the previous card is the one input that is in its list and not in this tool's list; the tool allows that one.

## 4. Control p1 (identity) and ties

The chain's AUROC breaks ties between equal scores by the order of the windows; the tool's weighted AUROC on the episode draws counts a tie as a half. The rehearsal found 0 tied windows on every probe, so the two agree at the point, and p1 passed in the rehearsal. p1 is unchanged.

## 5. Control p2 (estimator)

p2 runs on synthetic scores with the real labels, episodes and draws; it involves no score of a probe. Its permutation clause uses one fixed permutation and passed in the rehearsal. Unchanged.

## 6. Definitions for the values beside

* The stored presence score is a logit; "below 0.5" in the card means a logit below 0.
* The label gap of a window for the bins (0 to 20 m, 20 to 40 m, 40 m and more) is the label gap at the reference step that the gap measures use. Windows with a lead and no finite label gap are counted apart.
