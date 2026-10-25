# PREREG_L1 — AMENDMENT 3: the follow-up after O2 (EXPLORATORY, a new pre-registration; sealed BEFORE these arms are computed)

*2026-10-25 ~12:45 Berlin. The sealed L1 read (`PREREG_L1.md` + Amendments 1–2) returned **O2** with every control passing. These arms
were NOT in the base prereg; they are registered here, before computation, with both outcomes committed. They never re-label O2.*

## What O2 showed that decides the next lever (MEASURED, `result/L1_RESULT.json`)

* H2 (z_tac + scalars, ridge) fits TRAIN well **episode-held-out** (OOF R² at 0.5/1/1.5/2 s 0.54 / 0.25 / 0.27 / 0.33 vs the scalars'
  0.51 / 0.10 / 0.07 / 0.10) and **collapses on eval** (eval R² 0.41 / −0.27 / −0.49 / −0.55; E1 +0.026 SEP worse than the prior).
  Train139 is IN-SAMPLE for refcv8, so its z_tac can carry memorised clip information that the episode-CV inside train cannot expose
  (every train episode shares it). Hypothesis **MEM**: the failure is the fit split, not the absence of information in z_tac.
* The scalar head H2s ties the stage on E1 (H2s − H0 −0.0029 / −0.0034, n.s.) but discriminates braking far worse (AUC 0.657 vs
  refcv8 + stage 0.7445, AUC-DOWN): refcv8's own selected speed carries braking information the scalars do not. Hypothesis **STACK**: the
  two progress sources are complementary.

## Arms (S2 primary, S6 beside; same windows, estimator, tiers and four-family read as PREREG_L1 §5)

| arm | definition | deployable? |
|---|---|---|
| **H3** (stack) | per instant k: s' = s_P + a_k + α_k ĥ_s,k + β_k (s_os − s_P), (a, α, β) by OLS of h_k on [1, ĥ_s,k, d_os,k] on TRAIN, ĥ_s = H2s's 5-fold episode out-of-fold predictions (fold seed 0), d_os from refcv8-48,852's TRAIN os plans (bank, seed 0). Inputs at inference: t0 scalars + fed ceiling + the selected path. | yes (a deterministic in-forward layer; mechanism (b) extended with the scalar head) |
| **CF2 / CF2s** | H2's / H2s's recipe CROSS-FITTED on EVAL episodes: 5 folds by eval episode (seed 0); each fold scored by a head fit on the other 4 folds (inner (k, λ) CV on those folds only). Scored on refcv8's path. | NO — a measurement of whether z_tac on UNSEEN clips carries progress information; never a deployable arm |
| **D-shift** | domain AUC train-vs-eval z_tac: logistic regression on standardised PCA-32 z_tac, 5-fold episode CV over the pooled episodes | diagnostic, no bar |

## Committed outcomes

* **STACK-WINS**: H3 − H0 separated < 0 at both inference seeds AND H3 − H2s separated < 0 at both seeds AND no four-family row of
  H3 − H0 separated worse at both seeds ⇒ refcv9 L1 = the stacked deterministic re-timing layer (no latent); next: port it into
  `ProgressRetime` (mode "stack") and read B9-1 in-model. **STACK-NO** otherwise ⇒ the layer stays the shipped table (a).
  Beside (not in the outcome): H3's braking AUC vs refcv8 + stage on S6 (Amendment 2 definition).
* **MEM-YES**: CF2 − CF2s separated < 0 at both seeds ⇒ z_tac on unseen clips carries linearly decodable progress information beyond the
  scalars; L1's failure is the in-sample fit split ⇒ next lever: fit the head on a CLEAN split (V-A 203 refcv8-only clips under a refcv7
  warm start, or fresh V-C clips — both need Thor / a corpus pull: PI) or train it jointly in refcv9 with stop-grad.
  **MEM-NO** otherwise ⇒ z_tac does not carry linear progress information beyond the scalars even on unseen clips ⇒ the latent lever is
  training-side (full-grad progress head, a refcv9 ladder arm with a replicate) or a different readout (planner / BEV features).
