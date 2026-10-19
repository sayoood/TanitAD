# Amendment 5 to PREREG_HEADROW_PROBE: the proposal-SLOT prior, a zero-training POST-HOC diagnostic (it decides nothing)

**Parent:** `PREREG_HEADROW_PROBE.md` sha256 `1800705ef49333dffe0162ea5e51eb72fa1fe8fba57a0ea2f2869bf156762c55`; Amendments 1 to 4 (sha256 in their `.sha256` files).

**Written 2026-10-19 ~10:09, after the outcome tables and the C2x, C2y, C2z and index numbers were read, BEFORE any slot-prior number exists.** Nothing here changes, rescues or voids a call.

## Why

The index diagnostic read 0.49 to 0.51 for all rows (a monotone function of the slot index carries nothing), yet the cross-set label permutation C2x (which keeps each slot's label marginal and destroys everything else) read +0.10 to +0.33 above 0.5 in 8 of 8 cells, and the entry-level permutation C2y / C2z (which destroys the slot marginal too) read about 0 with a sign-mixed spread of +-0.05 to +0.08. The reading that fits all of it: the 64 proposals of a set are NOT exchangeable. Slot j carries a scene-independent TYPE (a non-monotone function of j, for example a grid of speed levels by lateral offsets), the features encode the slot, and the label distribution depends on the slot. A probe trained on noise labels that kept the slot marginal then learns the slot prior and ranks proposals within a set by it.

## The diagnostic (exact)

For each row, the SLOT PRIOR score of entry (set, slot j) is the mean label of slot j over the valid entries of the sets of the OTHER four folds (leave-fold-out, so no set scores itself; EP uses safe entries only, the registered convention); the score has no scene information at all. Report its all-64 OOF concordance against the real labels with the registered estimator (paired log-cluster bootstrap, B = 10,000, seed 20260927), per row, next to the P2 reference and the L arm of the same row. No threshold is committed; the reading, committed now: if the slot prior reaches at least 0.80 concordance on NC and on TTC, the C2 / C2x REDs are explained by the slot prior and say nothing about a pipeline artefact; and every concordance of this probe (P2's, L's, M's) must be read as including that scene-independent component.
