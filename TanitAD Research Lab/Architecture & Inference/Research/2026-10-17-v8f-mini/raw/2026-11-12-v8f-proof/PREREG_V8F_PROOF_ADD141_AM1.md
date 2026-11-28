# PREREG_V8F_PROOF_ADD141_AM1 — amendment 1 of card ADD141 (CONTACTRH10K): the tool's stated choices

Frozen after the tool `contactrh/read_contactrh10k.py` was written on synthetic data and before it has run on any real file; no value of card ADD141 exists. Card ADD141 (9d7e419072bfa399) stays as written except for these points.

1. Control c2: the reader's gap of a head cannot be derived from a dump, so c2 checks the logged `g_src` against the rule log's own identity `g0 = g_src + a_l - a_m` (g_src recomputed as g0 - a_l + a_m) within 1e-3 m on the planted sample of 20 ticks, together with the existence of a tick at or before every first contact step. It checks the log's consistency, not the reader against the dump.
2. The tick of a window is the main tick log's row with the largest k at or before the first contact step; the second-look log is not read.
3. The gap error is taken only where the reader reports a lead (`present`) and the label has one; a missed lead counts in Q2 only.
4. Q3 uses each arm's own tick; the closing rate of record is -c_src (c_src = lead speed minus own speed, positive when the gap opens, as `v_L = max(0, v_l + c_src)` in `dist_rule.py`), with v0 minus v_L beside it.
5. |X| below 10 windows over three seeds reads `CONTACTRH10K-UNRESOLVED` (too small to read), before the other names are tested.
6. Control c4 reads the lead's own test run of the tool on the dev box.
