# PREREG_V8F_PROOF_ADD143_AM1 — amendment 1 of card ADD143 (CONTACTRH10KC): the tool's stated choices

Frozen after the tool `contactrh/c143/read_contactrh10kc.py` was written on synthetic data (33 tests) and before it has run on any real file; no value of card ADD143 exists. Card ADD143 (a8185c372dbfeb2a) stays as written except for these points.

1. Control c2's "dump's own rolled speed at the tick's step": for k >= 2 the chord of the own step k of the dumped track; for k = 1 the chord from the track's own start point p0; for k = 0 the speed `pose[3]` of the tick file at tick 0. The tolerance stays 1e-6; if the dump stores positions in float32 and that tolerance fails on real data, c2 fails and a successor is frozen (the tolerance is not widened after the read).
2. The tick is the last main-phase row of the rule log at or before the first contact step; c2 also fails if a row lies outside the grid windows or has j different from t0 + k. The c2 sample draws from X and S of RH and S of UE.
3. The contact cells and the gap of the headway reader depend on positions only, so the pre-roll patch of `read_longroll10k_n4.py` does not change them; the patch is installed and checked, and the dump windows must equal the grid windows in grid order (part of c1).
4. Control n2 of `read_longroll10k_n4.py` is reported beside and does not gate; the card's c3 names n3 only.
