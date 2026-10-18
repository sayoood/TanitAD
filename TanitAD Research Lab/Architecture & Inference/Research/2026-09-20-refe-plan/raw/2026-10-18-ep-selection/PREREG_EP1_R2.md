# PREREG_EP1_R2: one guarded candidate, written after the fit read of C1-C3 and before any of its numbers

Written 2026-10-18 by the EP-1 agent. Amends nothing in `PREREG_EP1.md` (sha256 in `PREREG_EP1.sha256`); it is the next arm that file committed for the "PDMS-positive but ADVERSE" pattern (O-B), applied at the fit stage because that is where C2 showed it. **Navtest is still unopened; nothing of this candidate has been computed on any table.** RULE ZERO: C1, C2, C3 failed their fit gates, so the cheapest experiment that could still make a zero-training rule work is run before the turn ends.

## 0. What the fit read of C1-C3 showed (`raw/fit_read.json`, `raw/mech.json`; all fit table, 1,705 tokens / 24 logs)

| candidate | fit gain dPDMS x100 [95 %] | gate | what blocked it |
|---|---|---|---|
| C1 XPDMS | +0.10 [-0.17, +0.35] in-sample, positive in 3 of 6 folds | F1 fails (needs 4 of 6) | EP channel +0.33 [0.21, 0.45] is eaten by the gate channel -0.21: 366 picks move; 226 gain +2.9 each, but 9 lose -54.5 each (5 newly zeroed) |
| C2 BAND (delta 0.02) | +0.42 [-0.22, +1.10] in-sample; nested out-of-fold +0.23 [-0.42, +0.84], 3 of 6 folds | F1 fails, **F2 fails: TTC -1.58 [-2.45, -0.77] separated adverse** | EP channel +1.48 [1.26, 1.73], TTC channel -0.71 [-1.06, -0.38], gate channel -0.35: 1,437 picks move; 872 gain +3.5 each, 46 lose -49.8 each (12 newly zeroed) |
| C3 FLOOR | every tau loses (-0.35 to -5.47) | identity chosen, F1 fails | picking the longest plan above a predicted-safety floor ignores the EP/C terms and the head's ranking inside the floor |

Controls held: identity reproduces the baseline pick 1,705 / 1,705 for all three; shuffled-proxy controls read -3.31 (C1), -0.06 (C2), -1.80 (C3), all <= +0.05. Informative ceilings (not gates): the band rule with the TRUE EP as its progress proxy reads +2.75 at best, so a perfect progress signal inside the rule's own near-ties is worth +2.75 of the +8.10 oracle gap.
**The mechanism, MEASURED:** the payoff of a move is asymmetric. A longer pick gains about +3 PDMS on most moved tokens and costs about -50 on the few that become violations, so a move pays only if the extra violation rate it buys stays below about 5 % of the moved tokens. Measured on the moved picks, realised `NC*DAC` falls from 0.978 to 0.967 (C1) and from 0.980 to 0.976 (C2) while the heads predicted 0.961 to 0.953 and 0.971 to 0.961: the moves buy EP and pay violations at a rate the heads foresee on average but cannot localise (the 9 and 46 losing picks), which leaves a small net gain that its interval does not separate from 0.

## 1. Candidate C4, BANDGUARD (2 parameters)

Notation as in `PREREG_EP1.md` section 0-1; `b` = the baseline pick (`argmax S0`), `g_j = pNC_j pDAC_j`, `L_j` the plan's end distance.
`A(delta, eps) = {j : ln S0_j >= ln S0_max - delta  AND  g_j >= g_b - eps  AND  pTTC_j >= pTTC_b - eps}`; pick `argmax_{j in A} L_j`. `b` is always in `A`, so `A` is never empty. `eps` is a non-inferiority margin on the two predicted safety terms that carried the margin in step 1 (pDAC/pNC through `g`, and TTC).
Grid, ordered by (delta ascending, eps ascending): `delta in {0.005, 0.01, 0.02, 0.05, 0.1, 0.2}`, `eps in {0, 0.005, 0.01, 0.02, 0.05}`, plus the identity `delta = 0`. 31 cells.

## 2. Procedure (identical to `PREREG_EP1.md` section 2, restated)

1. Identity control: `delta = 0` reproduces the baseline pick on 1,705 / 1,705.
2. Deliberate-regression control: `L` replaced by a within-token permutation (seed 20260927); the best-over-grid dPDMS must be <= +0.05 (x100) or the instrument is broken.
3. `theta* = (delta, eps)`: the cell with the largest in-sample paired dPDMS; among cells within 0.05 of the maximum the first in the grid order above.
4. Nested grouped CV: the same 6 folds of 4 logs (`default_rng(20260927).permutation(24).reshape(6, 4)`); `theta` chosen on 20 logs, scored on the held-out 4.
5. **Fit gate: F1** pooled out-of-fold dPDMS > 0 AND positive in >= 4 of 6 folds; **F2** at `theta*` no NC, DAC or TTC component separates adversely in-sample (paired 95 % interval entirely below 0); **F3** the control of step 2.
6. If C4 fails the fit gate, navtest stays unopened (O-D below).
7. If C4 passes, `FROZEN_R2.json` (theta*) is written and hashed, then navtest is opened ONCE (m = 1): gates (table sha256 prefix `33441082b7044c95`, `pick` reproduced 12,146 / 12,146, PDMS formula max abs diff 0.0, DDC control 1.0, identity control), then the bar: **B1** paired dPDMS lower bound > 0 at the ordinary 95 % level (m = 1); **B2** no NC, DAC or TTC component separates adversely (95 %). Families by `families6.py`; STRATEGIC PAUSED; distance-keeping UNAVAILABLE.
- Estimator: paired log-cluster bootstrap, 10,000 resamples, seed 20260927, another draw of LOGS only, one training seed.

## 3. Committed outcomes and next arms (before any number)

| outcome | rule | committed next arm |
|---|---|---|
| **O-A PASS** (navtest B1 and B2) | deployable zero-training rule for P2; deploying it is the PI's call | transfer read on P6's table when it exists (ASSUMPTION until then) |
| **O-B ADVERSE** (B1 holds, B2 fails) | buys progress with safety | not deployable; zero-training rules are then exhausted at the head's information; training-side arm below |
| **O-C NULL** (B1 fails, B2 holds) | the guarded rule has no reach | training-side arm below |
| **O-D ELIMINATED AT FIT** (F1, F2 or F3 fails) | navtest stays unopened | training-side arm below |

**Training-side arm (the lever every outcome except O-A hands on; not run here, no training allowed):** a progress-aware scorer loss on P2's heads: (i) an EP target conditional on geometry-safe (so no double charge), (ii) a pairwise EP ranking term among safe plans of the fan, and (iii) the SR-TTC TTC ranking term on contenders. Cost ESTIMATED: head-only refit on P2's cached trunk features; the pre-head features for the 1,705 held-out tokens exist (`prehead_features_p2_heldout.npz`, 51.4 MB) but the navtrain training sets do not, so it needs the P2 forward over navtrain that SR-TTC already budgets (about 1.0-1.3 GPU-h per 3,137 sets, ~7 GPU-h for the SR-TTC arm in total, `D-REFE-SELRANK-1`); the EP terms share that forward, so their marginal GPU cost is ~0 and the added work is CPU head training. Dev-box GPU contention is the blocker named in `D-REFE-SELRANK-1`.

No grid, gate or bar changes after a number; a change is a new pre-registration.
