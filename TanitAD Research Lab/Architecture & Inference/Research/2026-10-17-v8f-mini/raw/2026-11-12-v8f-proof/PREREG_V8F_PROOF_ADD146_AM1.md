# PREREG_V8F_PROOF_ADD146_AM1 — amendment 1 of card ADD146 (FCREAD10K): the tool's stated choices

Frozen after the tool `rulefc/fcread/fcread10k.py` was written and tested on synthetic data (6 passed) and before it has run on any real file; the runs of card ADD145 are still going and none of their files is opened. Card ADD146 (91c4c523898ce5c4) stays as written except for these points.

1. A fifth control, k5 alignment: the reader side files line up with their logs (equal lengths), the logged gap of the rule g_src equals the reader's used gap within 1e-4 m on every row with a lead, a frame read in two windows gives the same values (no conflicts), and the two runs that share the reader TK16_s0 (model seeds 0 and 2) give the same reads of a frame. A failure stops the card.
2. Reader groups: the rows of runs s0 and s2 are merged into one group (TK16_s0), run s1 is the group TK16_s1. "For every reader seed" means both groups.
3. The names: `FCREAD10K-CARRIED` if A >= 0.30 in both groups; else `FCREAD10K-SPEED-ONLY` if S >= 0.30 in both groups; else `FCREAD10K-NOT-CARRIED`.
4. The label side uses the logged speeds of the RR runs' logs (the logged speed of a clip's frame, the same in every run) and the label closing of `LabelSource`; control k1 compares the label a_L with the forecast log of `H_RLcE1FC_s0` (card ADD144) exactly on every shared (clip, frame).
