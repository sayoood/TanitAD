# Note N2 to card PREREG_V8F_PROOF_ADD73 (HANDSRC10K): the form of the read, fixed while the rollouts run and before any dump of the card is opened

Written by the coordinator fork while the queue of the card is running on the pod (two of six rollouts done, no dump, tick file or run log of the card read; time: the line of this note in `TIMES_NOTE.md`; hash in the sidecar file). The reader is `integ10k/read_handsrc_decomp.py`; its hash is logged in `TIMES_NOTE.md` before it is run on the pod. Nothing in the card is changed; this note says how its sentences are computed.

## 1. Part 1, the cells of section 3

* Windows. The card reads each cell "on the windows where the cell has a value in all tracks used". Form: per mode M and model seed, the split of each run keeps a cell on the windows where it has a value in its three tracks (F, D, A; note N2 of card ADD71), and the cell block of mode M keeps the windows common to the own run and to M. The state part of own (A_own minus D), the state part of the mode (A_M minus D), the removed part (A_own minus A_M) and the statistic Q are read on that one set of windows; NO-STATE-COST is judged on it. The split of the own run on its own windows is written beside (`own_split`) and names nothing.
* Control q1, added: on the differences of the means, state part of own minus state part of the mode equals the removed part, and Q equals the removed part minus one half of the state part of own, both to 1e-9, on every cell block. If q1 does not pass, no name and no value of part 1 is written. (This is the lesson of control b4 of card ADD71: the engine pairs windows where a difference has a value, so parts read on different windows do not add up.)
* The split tool is run on each mode run with 300 bootstrap draws for its controls b4, b5 and b6 only; every reported value of a mode comes from the cell blocks with B 2,000.
* First replan segment (reported, no bar): the mean over steps 1 to 10 of the distance between a track and the logged track, for A_own, A_M and D, read through the same cell block. It is a position cell; it does not use a step-derived quantity.
* c0, form: the largest distance, over all windows and steps 1 to 10, between the aged model track of the `logged` run and D (the fresh model track moved by the offsets of that run), per model seed; bar 1e-3 m, both seeds.
* c1, form: dx, dy and the heading offset at t0 of each mode run equal those of the own run on every window, largest difference exactly 0.
* c2, form: from `preroll_log_<run>.json` of each mode run and seed, for every planner kind in the log: the log names the mode; under `logged` the position or heading group and the speed differ from the own batch; under `speed_logged` positions and headings differ by exactly 0 and the speed differs; under `path_logged` the speed differs by exactly 0 and a position or heading differs.

## 2. Part 2, the four families for the pairs mode minus own

The card asks for them "when the family reader of card ADD71 can be run on the new runs with its controls passing". Form:

* The grid read of that reader (`read_longroll10k.read_grid` under its notes N3 and N4, not edited) on the 1,271 windows with the runs R0Hb20 (spread), R0KHb20, R0KP30Hb20 and the three mode runs, pairs LOGGED-VS-OWN, SPEED-LOGGED-VS-OWN, PATH-LOGGED-VS-OWN: headway (unsafe rate U, contact, falling behind) and the family cells of LONGITUDINAL, LATERAL and TACTICAL, with the engine controls of that read (same windows in every dump, the headway controls, self pair 0, swap, planted regression).
* STRATEGIC: `strategic_grid` of that reader (instrument of card ADD66, route goal at 30, 50 and 80 m at the tick t0 + 2.0 s) on the runs R0, R0KP30Hb20 and the three mode runs, with its stage A and stage B controls.
* Control a1: the pair own minus fresh read in this grid read equals the pair AGE-K of the banked re-read of card ADD70 on every headway share and family cell to 1e-9.
* Not run: the file controls of that reader (wrapper hash, run-log arguments, readout log). They are bound to the wrapper generation before this card. In their place stand the pod check of the new wrapper (note N1: with `own` it returns every dump array of the wrapper of record) and c1 and c2 of part 1.
* Names by the rules of card ADD71 (family reading over the named cells, U by its interval and the spread of R0Hb20). If a control of part 2 does not pass, no family name is written and the row says per family what is missing; the names of part 1 stand by themselves, as the card says. A failure of part 2 does not hold back part 1, and a failure of part 1 does not hold back part 2.

## 3. What this note does not change

The runs, the cells, the name rule, the bar of one half, the controls c0 to c2, h1 and the fixed consequences of the card.
