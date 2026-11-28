# PREREG_V8F_PROOF_ADD96_N1: note 1 to card ADD96 (ERRCUT10K), written before any join of the card is built

State when this note is written: the card is frozen (4325c5963528497e); a helper has built the kit in `errcut/` on synthetic data; the lead reviews it in code. No join of the card is built, no control of the card is evaluated on data, no rollout exists. Of the per-frame dump the lead has seen the names, types and shapes of its arrays and no value.

## What was found, and where

Control e1 of the card has the clause "no mixed value is below 0". The lead wrote it as a check that a mix of two gaps stays a gap. Reading the code of the dump shows that the clause does not hold by construction: the reader's raw gap is a linear output times 50 m with no bound at 0 (`closeread/closeread_dump.py` line 140, `o[:, 1] * GAP_SCALE_M`), and the map of the run scales a value under its first knot in proportion (`closeread/closeread_fit.py` lines 354 and 355), so a read below 0 stays below 0. A mixed value M_k = L + k * (g - L) can then be below 0, first of all in arm K100. Whether it happens on the frames of this card is not known to the lead; the clause would then stop the card for a property of the reader that the card is about.

## The change

In control e1 the clause "no mixed value is below 0" is replaced by: the number of mixed values below 0 is recorded per arm and reader seed and printed in the row; such a value is left as it is. The rule's label source takes a gap as a lead only from 0 to the rule's largest gap (`distrule/v8f_t1_eval_rule.py`, class LabelSource), so on such a frame the rule sees no lead, as it would with that reader. This is the treatment that section 2 of the card already gives to a mixed value above the rule's largest gap.

Nothing else of the card changes: arms, measures, names, the other clauses of e1 and the controls e0 and e2 to e5 stand as frozen.
