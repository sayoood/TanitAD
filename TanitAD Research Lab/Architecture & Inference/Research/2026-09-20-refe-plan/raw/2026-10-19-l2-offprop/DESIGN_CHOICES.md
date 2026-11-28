# DESIGN_CHOICES - L2-OFFPROP (build agent, 2026-10-19; amended by MM_RULING_L2_BUILD.md E1-E7 + addendum)

Everything here was fixed on SYNTHETIC or SEMI-SYNTHETIC data (REAL tokens, SYNTHETIC labels). **No real unplanted label was read to make any choice below**, and no real arm (H, H0, H-null) was trained. Evidence class of every number: MEASURED on synthetic worlds (artifact paths in `raw/`), i.e. a property of the optimiser on a planted model, not of REFe. The recipe and its selection table are in `RECIPE_SELECTION.md`.

## 1. What the draft and the MM rulings fix (not chosen here)
* The EP head is row 2 of P2's `score_head = Linear(256, 6)`: **257 parameters**, input = the 256-d token the head consumes (MEASURED from the checkpoint; exported once, `feats_manifest.json`). Initialised from P2's row; every other row and everything upstream frozen.
* Loss (arm H) = `softplus(-y (z_p - z_a))` pairwise logistic on Step A (anchor, perturbation) pairs whose TRUE EP differs by >= 0.05, + `lam_bce` x pointwise BCE on in-fan EP labels, + `prox` x squared distance of the EP row from P2's. Both data terms are batch means. Folds: 5 by log (rank mod 5 over the 24 sorted names). Seeds {0, 1}.
* **E1 (null):** H-null permutes the perturbations' true EP GLOBALLY (across every perturbation of the dataset); replicates 100-104. The draft's within-(token, anchor) permutation is WITHDRAWN (it read +0.09 above P2 on a planted world) and survives only as a `--null-group anchor` diagnostic. VOID = >= 2 of 5 GLOBAL-null replicates exceed P2's A1-OOF by > 0.05.
* **E2 (A1):** the POOLED lon + lat quantity Step A's gate used (0.6851, den 9,888); lon-only and lat-only are reported beside it, ungated.
* **E3 / E4:** strictest defaults (null pool = max over all 10 null cells; VOID if EITHER seed has >= 2 exceeding).
* **In-fan clause:** in-fan EP concordance (OOF, SELRANK top-8 instrument) of H >= that of **H0 - 0.02**, H0 = the SAME recipe with the pairwise weight 0 (same init, folds, seed, minibatch stream), compared at the same seed; P2's in-fan value is REPORTED, never gated.
* **C-LEAK** is a STRUCTURAL reach check inside `oof_scores` (no switch turns it off), see `RESULT_GATES.md`.
* Estimator: L1's own paired log-cluster bootstrap (24 logs, 10,000, seed 20260927) - reused from `stepA.py`, not re-implemented.

## 2. The recipe
`lam_bce = 10`, `lr = 0.001`, `epochs = 15`, `prox = 0.3`, `batch = 256`, Adam betas [0.9, 0.999], eps 1e-08, weight decay 0, no schedule, no early stopping (nothing in the training may look at a validation label). Chosen by the rule in `RECIPE_SELECTION.md` -> `lam10_lr0.001_ep15_prox0.3`. The recipe is a DATA file (`raw/recipe_chosen.json`): `run_l2.py run` refuses to start without it and refuses gates that were earned on a different recipe file or different code.

## 3. Other fixed choices
* **Seed** = the minibatch order only (the head starts from P2's EP head; the generator is seeded from `(seed, n_pairs)`). H-null replicate = the permutation seed (100-104), crossed with the training seed. H0 shares H's generator stream, so it takes exactly as many steps.
* **Scoring** is float64 on the float32 token and weights (`l2lib.ep_apply`), so an arm's score does not depend on summation blocking; P2's own row gives the banked EP to <= 3e-6 (C-FEAT).
* **G-LIVE is enforced inside training**: the full [6, D] weight is the optimised leaf; at every step the gradient of the five frozen rows is asserted EXACTLY 0.0 and at the end they are asserted bit-identical to P2's. The prox term acts on the EP row only.
* **C-REACH planted amplitude 10** (the per-dim std of a token is ~1): one appended column equal to `10 * sign(true EP difference)` on perturbation tokens, 0 on in-fan tokens.
* **A1 pair rule** is Step A's (`>= 0.05`, scorer tie counts 0.5); the **in-fan pair rule** is the SELRANK instrument's (`> 0.05`, strict). They differ by definition in the two source instruments; both are kept as registered.

## 4. The semi-synthetic planted-signal test (the chain's one pytest failure)
`test_semi_synthetic_...` asserts A1(H) > A1(P2) + 0.02 on a planted-signal world (beta 3). Margins over the bar (MEASURED, `raw/planted_margin.json`):

* selected_recipe: A1(P2) 0.8766, A1(H) 0.8810, gain +0.0044, margin over the 0.02 bar -0.01561 ->FAILS
* superseded_chain_recipe_lam10_lr3e-4_ep5: A1(P2) 0.8766, A1(H) 0.8810, gain +0.0044, margin over the 0.02 bar -0.01561 ->FAILS
* source_default_REC_epochs12_used_by_the_other_tests: A1(P2) 0.8766, A1(H) 0.8966, gain +0.0200, margin over the 0.02 bar -0.00004 ->FAILS

The bar (strictly above +0.02) and the world are unchanged. The only change is the recipe the test runs under: the source default `Recipe(epochs=12)` before, the SELECTED recipe (`CHOSEN`) now, as the MM instructed. Both recipes with lam_bce 10 land on the same 4,413 of 5,009 concordant pairs; the two heads differ (||dW|| 0.152 vs 0.153) but flip the same pairs. The test is NOT loosened and it FAILS (see `RESULT_GATES.md`).

## 5. Seed-to-seed spread of H (E8; semi-synthetic, the chosen recipe)

| world | A1-OOF seed 0 | seed 1 | abs diff | in-fan H-H0 seed 0 | seed 1 | max abs EP-score diff (in-fan) | share of pairs whose sign differs |
|---|---|---|---|---|---|---|---|
| sigA | 0.7800 | 0.7794 | 0.0006 | -0.0042 | -0.0045 | 0.175 | 0.0039 |
| null | 0.7104 | 0.7100 | 0.0004 | +0.0006 | +0.0003 | 0.139 | 0.0051 |

The seed moves only the minibatch order of a (near-)convex fit from a fixed start, so a small spread is expected; it is NOT a replicate of P2's own training seed (H-ESTIM-SEED-1 still applies to the P2 token).

## 6. Cost of the full real run (measured on REAL tokens + SYNTHETIC labels, 1 torch threads, a box at 60-90 % CPU load from other jobs, CPU only; recipe `lam10_lr0.001_ep15_prox0.3`)

* data build 1.2 s; one H cell (5 folds) **7.5 s**, one H0 cell 5.5 s (pairs per fold [9107, 10036, 9807, 8712, 9850], steps per fold [540, 600, 585, 525, 585]); one H-null cell 13.1 s; one `evaluate_arm` 0.1 s; peak RSS ~668 MB.
* full run = **15 cells** (P2 + 2 H + 2 H0 + 10 global-null H-null) and 15 evaluations: **~2.6 min on one process** ({'fits_2_H': 15.0, 'fits_2_H0': 11.0, 'fits_10_Hnull': 131.0, 'evals_15': 1.5, 'total': 158.5}); cells are independent, so N processes divide it.
* GPU: none. Export (already done): 2 workers x ~100 min-equivalent of CPU, ~45 min wall.

