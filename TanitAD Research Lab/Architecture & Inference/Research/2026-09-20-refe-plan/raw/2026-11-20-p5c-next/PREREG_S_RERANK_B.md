# PREREG S-RERANK-B: a within-token, conservative switcher over C1's frozen outputs (0 GPU). Second and LAST design on this split.

**Written 2026-11-20 ~09:55Z (UTC) by the MM.** S-RERANK (design A) read **NO SIGNAL**, and both its controls passed
(`propscore/S_RERANK_READ.json`):
* C1 EVAL Δ −0.45 [−0.74, −0.14], with NC and EP separated adverse.
* RED −16.1; POS regret 0.002.
* It overrode the pick on 4,385 / 6,251 tokens.

Diagnosis: design A regressed **absolute** PDMS, and that objective is dominated by between-token variance (scene difficulty), not the
within-token ranking that selection needs. B changes **only** the objective and adds a switch threshold. The sha256 is frozen in the MM
loop state before the run.

⚠️ **Multiplicity, stated now.** This is the second re-ranker design read on the same TUNE/EVAL split. A SIGNAL from B is labelled
*second design on the split*. It is never adopted on this read alone. Adoption needs the navtrain-trained replicate (the deployable
version) read on the full navtest. No third design is read on this split.

## Arm (fixed now)
* **Target.** Gain over the pick, y_k = PDMS_k − PDMS_pick (tracked, T1). It is computed on TUNE tokens of C1 + C0 + base, over the
  63 non-pick proposals.
* **Features.**
  * design A's per-proposal features (model outputs only);
  * the proposal-minus-pick differences of the 6 head outputs and of the navsim_v1 aggregate;
  * the pick's own 6 head outputs and aggregate (token context).
  * No column from the scorer table enters the features.
* **Model.** The same HistGradientBoostingRegressor hyper-parameters as A: max_iter 300, learning_rate 0.05, max_leaf_nodes 31,
  min_samples_leaf 200, l2 1.0, random_state 0.
* **Switch rule.**
  * k* = argmax over the non-pick proposals of the predicted gain. Switch to k* iff pred(k*) > τ; otherwise keep the pick.
  * τ is chosen **on TUNE only**, by 5-fold GroupKFold over the TUNE logs (out-of-fold predictions), from the grid
    {0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, ∞}. The choice maximises out-of-fold mean tracked PDMS. Ties go to the larger τ.
  * τ = ∞ means never switch, so Δ = 0 exactly. The model is then refit on all of TUNE and applied to EVAL with the chosen τ.

## Read, bar and controls: the same as A (PREREG_S_RERANK.md)
* **Estimator.** Paired log-cluster bootstrap, 10,000 draws, seed 20260927, on EVAL logs. It answers *another draw of logs* only.
* **SIGNAL iff all three hold:**
  1. CI-lo(Δ) > 0 on C1 EVAL;
  2. no sub-score NC, DAC, TTC, EP or C is separated adverse;
  3. Δ point > 0 on C0 EVAL and on base EVAL.
* **RED.** Labels are permuted within each token. It must NOT read SIGNAL: required Δ point ≤ 0 on C1 EVAL.
* **POS.** The true gain is added as a feature. It must reach regret < 1.0 on C1 EVAL.
* **Leak guards.** Zero EVAL logs among the training rows; no feature from `tr` or `tp`.

## Both outcomes, committed now
* **SIGNAL** (second design on the split):
  * Meaning: the margin is learnable within-token from the frozen outputs.
  * Next: the deployable version. This is either a navtrain per-proposal table with this switcher trained on it, or a heads-only
    SCORER-FT with a within-token pairwise loss. The PI card C is re-cut with the measured navtrain inference and labelling cost.
* **NO SIGNAL:**
  * Meaning: two designs fail. The frozen outputs plus geometry do not carry the margin.
  * Next: SCORER-FT must reach the representation, or the lever is generation (TEACHER-N7). Both need the PI and the GPU. The MM then
    measures their costs for the card, and the selection line closes on this split.
