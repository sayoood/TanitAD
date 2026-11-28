# PREREG_V8F_PROOF_ADD140_AM1 — amendment 1 of card ADD140 (RHK10K): the reader's stated choices

Frozen after the reader of card ADD140 was written (on synthetic data) and before it has run on any real file of this card; no value of card ADD140 exists in any open file. Card ADD140 (d5f7108f7272081c) stays as written except for these points.

1. R3 route goal. The route-goal reader of card ADD66 (`strat_roll_read.py`) has two controls, s2 and s3, that need a run started at the logged state. Both arms of R3 are pre-rolled by 6 s, so neither run can meet them, by construction. For R3 only, s2 and s3 are not applied; the reader writes this into its output. Its controls s1, s4, s6, s9 and the episode map still gate. R1 and R2 keep every control.
2. `RHK10K-FRESH-EQUAL` also needs no family read MIXED; a fresh read with a MIXED family and none WORSE is `RHK10K-FRESH-MIXED`.
3. A family counts WORSE only when its reading is WORSE (as in card ADD139).
4. For R3 the unsafe rate U is reported beside the AGED name, not inside it, as section 5 of the card writes the AGED names from the families alone.
5. Control q3 fails on a missing or differing listed partner file; a partner folder's extra file is reported, not failed.
6. The controls k4 (environment) and k5 (reader input) of card ADD139 are not repeated: the environment, the tree and the reader are those of card ADD139.
7. For a combination of names that section 5's consequence sentence does not cover, the output records "not stated by the card".
