# RESULT_GATES - L2-OFFPROP build gates (2026-10-19, after the MM rulings)

Gates, not results: none trains H, H0 or H-null on real labels or reads an A1-OOF of an unplanted head. Real arrays = the 400 tokens / 24 logs / 38,400 pairs of L1 Step A. Stamp: SEEN-SCENE, SELECTION tier. Every PASS is stamped with the sha256 of the recipe file and of `l2lib.py`, `gates.py`, `synthetic.py`; `run_l2.py` refuses a PASS earned on a different recipe or different code.

## C-FEAT - PASS
P2's EP head applied (float64) to the EXPORTED tokens vs Step A's banked EP scores (`scores.npz`); bar: max abs diff <= 1e-4 on >= 99 % of rows.

| rows | n | max abs diff | share <= 1e-4 | p99 abs diff | all six columns, max abs | export-run logits vs banked, max abs |
|---|---|---|---|---|---|---|
| in_fan | 25600 | 2.093e-06 | 1.000000 | 1.067e-06 | 7.701e-06 | 0.000e+00 |
| perturbed | 38400 | 2.145e-06 | 1.000000 | 1.116e-06 | 6.612e-06 | 0.000e+00 |

bundle sha256 matches the run's: True; perts sha256 matches: True. Evidence: MEASURED, `raw/gates/cfeat.json`.

## C-ID - PASS
Zero optimiser steps => P2's EP scores bit for bit, for H and for H0, both seeds; then A1-OOF equals Step A's A1 on the same pairs. Recipe stamp: `c7f5b75b0a8f25f9`.

* zero-step scores bit-identical to the P2 arm (in-fan AND perturbed): {'H_seed0': True, 'H_seed1': True, 'H0_seed0': True, 'H0_seed1': True}.
* A1 (POOLED lon+lat (MM ruling E2), den 9888.0) of the zero-step arm 0.6851 [0.6144, 0.7610] vs Step A `RESULT.json` 0.6851 [0.6144, 0.7610]; |diff of points| = 0.00e+00; worst |diff| over every reported A-quantity (point, lo, hi, den; all/lon/lat) = 0.00e+00; L1's own analyse on the banked logits reproduces RESULT.json to 0.00e+00.
* pairs whose banked EP-logit gap is below 1e-4 (could flip between fp32 and float64): 13.
* P2's in-fan top-8 EP concordance on these 400 tokens: 0.8776 [0.8269, 0.9340] (REPORTED, never gated). Evidence: MEASURED, `raw/gates/cid.json`.

## G-LIVE - PASS
Gradient probes on the REAL fold-0 training set (7836 pairs, 19648 in-fan rows; no optimiser step is taken on a real label); recipe stamp `c7f5b75b0a8f25f9`.

* finite losses on all 3 probe batches: True; EP-row gradient |sum| min 4.754e+01, bias gradient min 8.479e-02 (> 0).
* frozen rows (0,1,3,4,5) gradient max abs: W 0.0, b 0.0 (must be EXACTLY 0.0).
* control: a loss term through row 3 reads a NON-zero frozen gradient (the probe can read a non-zero): True; inside training the same mutation raises: True.
* **H0 lever reaches the head:** with `w_pair = 0` the EP-row gradient equals the gradient of `lam_bce * BCE` alone BIT FOR BIT, with `w_pair = 1` it differs, frozen rows stay exactly 0.0: True.
* **prox lever reaches the head:** its gradient is exactly 0 at P2's init, changes the EP-row gradient off the init, frozen rows stay exactly 0.0: True.
* a 40-step training segment (H) and a 40-step one (H0) ran with the per-step frozen-gradient assertion on, on REAL tokens + SYNTHETIC labels (semi-synthetic, beta 0.5, alpha 0.25, seed 21): no real label trains a head here; H and H0 trained to different heads: True. Evidence: MEASURED, `raw/gates/glive.json`.

## C-REACH (planted) - PASS
A feature equal to `10 * sign(true EP difference)`, appended to the perturbation token, 5-fold-by-log OOF, **both seeds**. Bar: A1-OOF >= 0.98 on each. Recipe stamp `c7f5b75b0a8f25f9`.

* seed0: A1-OOF (pooled) **0.9984 [0.9961, 1.0000]**; lon 0.9983 [0.9960, 1.0000]; lat 1.0000 [1.0000, 1.0000]; steps per fold [465, 480, 480, 420, 510].
* seed1: A1-OOF (pooled) **0.9992 [0.9982, 1.0000]**; lon 0.9992 [0.9982, 1.0000]; lat 1.0000 [1.0000, 1.0000]; steps per fold [465, 480, 480, 420, 510].

Reads the PLANTED arm only. Evidence: MEASURED, `raw/gates/reach.json`. (The chain's superseded recipe read 0.9048 here: `raw/superseded/`.)

## C-LEAK (STRUCTURAL) - PASS
REDEFINED by the MM addendum: a linear 257-parameter head cannot memorise shuffled labels, so the old statistical test (clean 0.502 vs leaky 0.504) was UNINFORMATIVE, not passed (`raw/superseded/leak_statistical_uninformative.json`). `oof_scores` now calls `fold_split` on every fold, which ASSERTS (fails closed, no off-switch) that the training token set is disjoint from the held-out token set, that held-out logs are disjoint from training logs, and that every index array the optimiser consumes (pair anchors, pair perturbations, in-fan rows) references only training tokens; after the folds, every token must have been scored exactly once. Run on the REAL tokens with SHUFFLED labels (no real label is read).

| fold | train tokens | held-out tokens | pairs | token overlap | log overlap | held-out tokens referenced by the training set |
|---|---|---|---|---|---|---|
| 0 | 307 | 93 | 22198 | 0 | 0 | 0 |
| 1 | 343 | 57 | 24885 | 0 | 0 | 0 |
| 2 | 332 | 68 | 24029 | 0 | 0 | 0 |
| 3 | 292 | 108 | 21183 | 0 | 0 | 0 |
| 4 | 326 | 74 | 23653 | 0 | 0 | 0 |

RED mutations - each must make the production path raise `C-LEAK`:

* `one_held_out_token_injected_fold0`: fires
* `one_held_out_token_injected_fold1`: fires
* `one_held_out_token_injected_fold2`: fires
* `one_held_out_token_injected_fold3`: fires
* `one_held_out_token_injected_fold4`: fires
* `whole_fold_injected_fold0`: fires
* `oof_scores_trips_before_any_step`: fires
* `held_out_row_routed_into_pair_indices`: fires
* `held_out_row_routed_into_infan_indices`: fires

A clean production `oof_scores` ran end to end ({'folds': 5, 'every_fold_recorded_a_leak_check': True}). Evidence: MEASURED, `raw/gates/leak.json`; unit tests in `test_l2.py` (`test_oof_scores_trip_on_ONE_injected_held_out_token_in_every_fold` and three more).

## Outcome mapping - RED mutations
`python test_outcome.py` -> `ZZOUTCOME_MUTATIONS OK 27 mutations`. 31 scenarios (0 failures of the original module); each textual mutation (every threshold, every branch, both VOID clauses, both null-pool / void-scope switches, and the in-fan clause: dropped, compared with P2 instead of H0, `>=` to `>`, sign flipped) fails at least one scenario. `raw/test_outcome_mutations.log`.

## KNOWN FAILURE - `test_semi_synthetic_hits_its_two_calibration_targets_and_reads_no_real_label` (NOT loosened)
The test asserts A1(H) > A1(P2) + 0.02 on a 72-token planted-signal semi-synthetic world (beta 3). Under the SELECTED recipe: A1(P2) 0.8766, A1(H) 0.8810, gain +0.0044, **margin over the bar -0.0156 -> FAILS**. Under the source default `Recipe(epochs=12)` (what the chain's run used - the test never read the chain's recipe): gain +0.01996, margin -0.000036 (100 of the 101 pairs needed) -> FAILS by one pair. Cause (MEASURED, `raw/planted_margin.json`): the selected recipe is in-fan-weighted by design (lam_bce 10, prox 0.3: the head moves ||dW|| ~0.15 from P2's), so on a world with only 57 training tokens per fold it finds only +22 concordant pairs of 5,009 (the source default finds +100); a recipe that finds it (lam 0.1, ||dW|| ~1.7) degrades in-fan ordering. It is the A1-gain versus in-fan trade-off the MM's rule resolves in favour of in-fan non-degradation. The test was not changed; a companion REGRESSION test on the REAL tokens (`test_selected_recipe_finds_a_planted_signal_on_REAL_tokens`: sigA, beta 0.5) passes with the selected recipe (gain +0.045).

## Unit tests
`pytest test_l2.py -q` (CPU, synthetic only) -> `1 failed, 34 passed in 38.57s`. Covers folds, the A1 pair definition = L1's A1 denominator, the GLOBAL H-null (default) and the anchor diagnostic, arm H0 (a dead pairwise term: flipping every pair label changes nothing), structural C-LEAK + its RED mutations, G-LIVE and its mutation, reachability of every Recipe field (incl. prox) into the trained head, C-ID/C-REACH on synthetic worlds, the SELRANK concordance against a brute force, EP-score reachability into the aggregate, pooled-A1 and H0-in-fan plumbing of the read, the seal guard (RED on a draft, a wrong sha, a missing or failing gate, a gate earned on a different recipe or different code).

