# PREREG S-RERANK: do C1's frozen outputs carry the selection margin a fixed rule could not use? (0 GPU)

**Written 2026-11-20 ~10:58Z by the MM**, after S-RULE read **NOT ADOPTED** (`propscore/S_RULE_READ_A1.json`) and before any re-ranker
is trained. The sha256 is frozen in the MM loop state before the run. This is the cheapest test of the question that SCORER-FT
(PREREG_P5C_NEXT §5 C, GPU, PI) pays to answer. It is a **diagnostic**. Its EVAL number is never a navtest score, because the re-ranker
trains on navtest TUNE logs.

## Measured basis (MEASURED, T1 LQR-tracked, EVAL logs, `srule_fam/selection_quality_eval.json`, paired log-cluster bootstrap 10k)
* C1's pick scores 92.59 [91.66, 93.43]. The oracle of its 64 proposals scores **99.41**.
* Regret is 6.83 [6.08, 7.67]. The pick attains the oracle on only 47.7 % [44.3, 50.9] of tokens.
* S-RULE (2,5): Δ −0.17 [−0.32, −0.01], with EP separated adverse.
* ⇒ Headroom exists, and every monotone reweighting of the 6 heads loses it on EVAL.

## Arm
* **Inputs (inference-admissible).** Model outputs only. Nothing from the scorer table enters the features. Per proposal:
  * the 6 sigmoid head outputs and the navsim_v1 aggregate;
  * the aggregate's rank and its gap to the set maximum;
  * geometry of the 8 poses: endpoint x, y and heading, the 1 s pose, path length, max |Δheading|, and max second difference
    (accel proxy) and third difference (jerk proxy) at 0.5 s;
  * set context: mean L2 distance to the aggregate-weighted consensus trajectory, distance to the navsim_v1 pick, and endpoint distance
    to the set median.
  * No situation-classifier output exists in this model, and no ego privileged channel is added.
* **Label.** Tracked PDMS of each proposal (T1), from `propscore/full_{C1,C0,base}_s*of2.jsonl`.
* **Model (fixed now, no tuning).** sklearn HistGradientBoostingRegressor: max_iter 300, learning_rate 0.05, max_leaf_nodes 31,
  min_samples_leaf 200, l2_regularization 1.0, random_state 0.
* **Training rows.** TUNE logs (sha1(log) last hex digit even) of all three arms, C1 + C0 + base, 64 rows per token.
* **Selection.** argmax of the predicted PDMS. Ties go to the lower index.

## Read (same as S-RULE)
* Δ = PDMS(re-ranker pick) − PDMS(navsim_v1 pick), tracked, ×100, on C1 EVAL.
* Estimator: paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile. It answers *another draw of logs* only.
* The sub-scores NC, DAC, TTC, EP and C are read the same way.

**Bar: SIGNAL iff all three hold:**
1. CI-lo(Δ) > 0 on C1 EVAL;
2. no sub-score is separated adverse (CI-hi < 0) on C1 EVAL;
3. Δ point > 0 on C0 EVAL and on base EVAL.

## Controls (the run refuses a verdict unless both read as required)
* **RED.** Labels are permuted across the 64 proposals within each TUNE token. Signal is destroyed, so the arm must read
  Δ point < 0 on C1 EVAL.
* **POS.** The true tracked PDMS is added as a feature, a deliberate leak. The arm must reach regret < 1.0 on C1 EVAL, which proves the
  pipeline can find a margin when one is present.
* **Leak guards (asserted).** Zero EVAL logs among the training rows. No feature column is derived from `tr` or `tp`.

## Both outcomes, committed now
* **SIGNAL.**
  * Meaning: the frozen outputs plus geometry carry a learnable margin, so a heads-only SCORER-FT (or a small learned aggregator) is
    expected to work.
  * Next: a deployable version trained on a **navtrain** per-proposal table, then a full navtest read. This is PI card C, and the
    navtrain inference and labelling cost is measured first.
  * The EVAL Δ here is reported as a diagnostic upper-bound estimate, labelled "trained on navtest TUNE".
* **NO SIGNAL.**
  * Meaning: the margin is not in the outputs plus geometry, so a heads-only fine-tune is unlikely to close it.
  * Next: SCORER-FT must reach the representation (a backbone or BEV feature fine-tune), or the lever is generation (TEACHER-N7).
    The PI card is re-cut with that design.
