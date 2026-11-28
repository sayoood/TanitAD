# PREREG_V8F_PROOF_ADD140_AM2 — amendment 2 of card ADD140 (RHK10K): the run that control n1 reads

Frozen after the first read of card ADD140 stopped at a failed control and before any value of the card was opened. The reader (`rhacc10k/read_rhk/read_rhk10k.py`) wrote no name, level or difference; only the control record was read. Card ADD140 (d5f7108f7272081c) and amendment 1 (47be0af85a5bdde5) stay as written except for this point.

What failed: control n1 of the pre-roll reader of record (`integ10k/read_longroll10k_n4.py`, `controls_n4`, lines 187 to 206) needs at least one run without a pre-roll among the runs it loads ("no-preroll runs start at the origin"); it passes only on a non-empty list. Both arms of R3 are pre-rolled by 6 s, so the list was empty and n1 read `PASS false` with `runs {}`. The other controls of that reader (n2 on all six pre-rolled runs, n3, and the engine controls) and every control of R1 and R2 passed.

The point: n1 is read on the three runs `H_RLcA_P0_s0..s2` of card ADD139 (the same 1,271-window grid of card ADD70, the same box 2, tree and pre-roll wrapper, no pre-roll; controls passed in card ADD139), loaded beside the two R3 arms and registered as card ADD139's reader registered them (`("RLc", "A", 0, "deadreckon", "30", "plan", 5, 0.2)`). They enter no pair and no bar of this card; R3 stays `H_RLcK_P30` minus `U_RLcK_P30` with the spread of `U_RLcK_P30` as the bar. Their files are checked against card ADD139's pulled copies before the read (sha256 of every file).

If n1 fails on these runs, the card stops again and no value is read.
