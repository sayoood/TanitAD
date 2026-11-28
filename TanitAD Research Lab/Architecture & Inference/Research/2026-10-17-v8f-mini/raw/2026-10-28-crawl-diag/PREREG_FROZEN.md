# PREREG_FROZEN: verbatim copy of H-CRAWL-01, -02, -03

Source: dagland/research_crawl/HYPOTHESES.md (sha256 cd3e1d7089acae7eaf8522f074ded57ce2501b54b2323407ec9e79004d42ab65), lines 24-52, copied byte-for-byte before any computation. Falsifiers (both outcomes) are the committed readings; not to be changed after data.

---

## H-CRAWL-01. The outcome-blind floor
- **Hypothesis.** Part of the −0.85 m/s gap is produced by selection plus regression to the conditional median/mean, not by a policy defect. The 327 windows are chosen because the human launched; any predictor that cannot read the trigger scores negative on them, and L1/L2 regression on a stay/go mixture interpolates (the same mechanism as [2306.07957 p.5-6]). *The selection argument is my inference, not a citation.*
- **Predicted effect vs floors (ESTIMATED).** The outcome-blind conditional-median baseline reads between −0.3 and −0.8 m/s on the same 327 windows, i.e. 35-95 % of the observed gap. The range is wide because I do not know what share of stopped-state windows launch within the horizon.
- **Cheapest discriminating experiment.** Zero training, CPU, minutes; label arrays plus the existing T1-kin model outputs.
  1. Bucket all training windows by current speed (<0.5 m/s, 0.5-1.5 m/s) and short speed history; take the conditional median and mean of `v_GT` at the first ticks.
  2. Apply that table to the 327 launch windows → outcome-blind gap.
  3. Compute the model's signed gap on matched **GT-stay** windows (stopped, no launch within the horizon). An outcome-blind model reads positive there.
  4. Report model gap minus outcome-blind gap with the paired episode-cluster bootstrap (that answers "another draw of episodes" only; say so).
  - Controls: the conditional-median table applied to its own bucket must read exactly 0.0 median bias; P10 gets the same treatment.
- **Falsifier (both outcomes committed).** (a) model gap − outcome-blind gap ≤ −0.4 m/s with CI excluding 0 **and** a non-positive GT-stay gap ⇒ the model crawls beyond what its inputs allow ⇒ shortcut/copycat explanations (02, 04-06) stay alive. (b) the difference is within ±0.2 m/s ⇒ the crawl is information-limited; copycat fixes will not close it, and the lever is trigger perception (06 probe) or the output representation (09).
- **Admissibility.** Measurement only.

## H-CRAWL-02. Speed-shortcut strength
- **Hypothesis.** The planner's first-tick output speed tracks its input speed/history, with sensitivity `S_v = Δv_out / Δv_in ≥ 0.6`, and swapping the images barely changes the launch output. Analogue: VAD with input v = 0 collapses to a near-stationary plan (avg L2 6.16 m vs 0.37 m) while blank images give 0.46 m [2312.03031 p.5, Table 2]. The copycat channel is previous acceleration recoverable from the speed history [2010.14876 p.2].
- **Predicted effect (ESTIMATED).** `S_v ∈ [0.6, 1.0]` on stopped-start windows. `S_v` differs between the two runs of the pair by an amount that tracks their 0.556 m/s launch-gap difference.
- **Experiment.** Inference only, minutes, both runs of the pair, 327 windows, first tick (no rollout):
  1. shift the speed history by δ ∈ {−0.5, +0.5, +1.0} m/s;
  2. zero the history except the current speed;
  3. replace the images with another window's images, same speed history.
  - Controls: δ = 0 reads exactly 0.0; setting v_in = 0 is the positive control (output must drop). Counterfactual inputs are measurement tools and are never deployed.
- **Falsifier.** `S_v ≤ 0.3` with upper CI < 0.5 on **both** runs ⇒ output speed is largely independent of input speed ⇒ 04 and 05 lose their mechanism. Separately: if `S_v` is equal across the two runs while their launch gaps differ by 0.556, the run-to-run difference is not shortcut strength.
- **Admissibility.** Measurement only; no deployed input changes.

## H-CRAWL-03. Within-run vs between-run variance
- **Hypothesis.** The 0.556 m/s run-to-run difference is mostly between-run (basin and data order), not checkpoint noise inside a run, so tail-averaging will not help but pooling K runs will. Support: runs commit to a linearly connected region early [1912.05671 p.1]; seeds change results "substantially" and init and data order contribute comparably [2002.06305 p.1]; averaging the latest checkpoints removes within-run noise only [2209.14981 p.1]; checkpoint choice alone swings results 10-100 % [2108.03298 p.7]. Aggregate metrics over ~12k scenes show <0.02 PDMS spread over 3 seeds [2605.15120 p.31], so a 327-window subset is where run variance concentrates (ESTIMATED).
- **Predicted effect (ESTIMATED).** Within-run SD of the launch gap over the last 5 checkpoints ≈ 0.10-0.25 m/s, well under 0.556. Pooled SD for K = 4 runs ≈ 0.4/√4 = 0.2 m/s.
- **Experiment.** Zero training. Evaluate the launch gap and 6 s ADE on the last k = 5 saved checkpoints of each run (10 evals), plus an EMA/LAWA tail average. If intermediate checkpoints were not saved, save the last 5 in the next fine-tune (cost ≈ 0).
- **Falsifier.** Within-run SD ≥ 0.4 m/s ⇒ checkpoint noise dominates ⇒ tail-averaging (EMA/SWA/LAWA) becomes the first variance lever and cross-run soups (07) are secondary. Within-run SD ≤ 0.15 with the between-run gap persisting ⇒ pool seeds, do not extend steps.
- **Admissibility.** Measurement only.
