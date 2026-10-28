# Hypothesis cards: launch crawl and run-to-run variance (research_crawl)

Date 2026-10-28. Status of every card: **UNRUN, PROPOSED**. Thresholds below are proposals written before any of these measurements exist; the Master Mind should adopt or change them **before** running, then freeze them.
Sources are `[arXiv-id p.N]` PDFs in `pdf/` (see `REVIEW.md`, `BANKING_MANIFEST.txt`). Scope: inertia / copycat and variance reduction only. Data selection and training stages (SFT, DAgger, RL) belong to `dagland/research_data/`.

**Floors used (INHERITED from the Master Mind, 2026-10-28):** launch gap `mean(v_own − v_GT)` ≈ −0.84…−0.87 m/s on 327 GT-launch windows (T1-kin, open loop); same-recipe same-init pair differs by −0.556 [−0.699, −0.398] m/s; 6 s ADE run floor ≈ 0.40 m; `--launch-intent-w` moved the gap +0.03 m/s, not separated. From one pair I derive a per-run SD of the launch gap of ≈ 0.4 m/s (ESTIMATED, n = 1 pair, plausible range 0.25–0.8).
**Conventions.** "Effect" = change in the launch gap, positive = closer to the human. Every effect is ESTIMATED because no paper measures this quantity. A separated paired-bootstrap CI is necessary and not sufficient for a lever claim (`H-ESTIM-SEED-1`): each trained-arm card needs its replicate arm, and reports the four metric families per family (ADE alone is never the result).
**Execution order** (differs from the ranking in `REVIEW.md` §3): 01, 02, 03, 07 first (all zero-training, minutes), then 04 and 05 (box-2), then 06 and 08 (PI / overlap), then 09 only if 01 supports it.

| id | kind | cost | rank among levers |
|---|---|---|---|
| H-CRAWL-01 | diagnostic: outcome-blind floor | CPU, minutes | – |
| H-CRAWL-02 | diagnostic: speed-shortcut strength | inference only, minutes | – |
| H-CRAWL-03 | diagnostic: within- vs between-run variance | existing checkpoints | – |
| H-CRAWL-04 | lever: kinematic-token dropout + speed noise | box-2, ~40 min/run | 1 |
| H-CRAWL-05 | lever: history truncation | box-2 (flag) or PI | 3 |
| H-CRAWL-06 | lever: loss-only vision-speed + stop/go heads | PI (architecture) | 2 |
| H-CRAWL-07 | lever: weight / output averaging | zero-training CPU | 4 (run first) |
| H-CRAWL-08 | lever: copycat-error (APE) weights | CPU + box-2 | 5 |
| H-CRAWL-09 | lever: longitudinal speed-bin head | PI (architecture) | 6 |

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

## H-CRAWL-04. Kinematic-token dropout + speed noise (SDE-style)  [lever, rank 1]
- **Hypothesis.** If `S_v` is high (02) and the model is worse than the outcome-blind floor (01), randomly dropping the kinematic history tokens in training, plus Gaussian noise on the speed input, forces the planner to use vision and scene context at low speed and raises launch speed. Inference keeps all tokens.
- **Support.** PlanTF SDE (kinematic tokens dropped with p; pose kept): NR-CLS 86.48 at p = 0.75 vs 77.28 without [2309.10443 p.7, Table VIII], +3.29 in Table II [p.4]; the two no-SDE baselines differ ~6 points and no seeds are reported. State-only input beats history input on closed loop, 81-86 vs 56.5 [p.3, Table I]. Speed-state noise σ = 1 (their units) lifts offline/online correlation −0.64 → −0.92 [2310.20650 p.8, Fig 7]. ChauffeurNet's past-motion dropout has no isolated ablation [1812.03079 p.13-14] and its re-implementation is ≈ baseline on CARLA (34.0±2.6 vs 33.0±4.2) and worse on HalfCheetah [2106.06452 p.7].
- **Predicted effect (ESTIMATED).** +0.05…+0.30 m/s (central +0.15); `ΔADE_6s` within ±0.40 m; `S_v` falls by ≥ 0.15. Probability that a 2-seed read separates it from the 0.556 floor ≈ 15 % (effect sits below the floor).
- **Experiment.** Box-2, ~40 min/run (INHERITED figure). Arm S: per-token dropout p = 0.5 on history speed/yaw-rate tokens (current speed always kept; current yaw rate kept with prob 0.75), plus N(0, σ = 0.5 m/s) on the speed input (σ is a design choice; Cultrera's unit is unstated). Same init, 2 seeds, against the existing baseline and its same-seed replicate pairs. ≈ 1.3 box-hours for the arm, ≈ 2.7 with replicates.
  - Reachability controls (per the "built, tested, unreachable" rule): p = 0 reproduces baseline bit-for-bit; p = 1.0 for 50 steps drives `S_v` to ≈ 0 and changes the loss; the new flags appear non-default in the argv and have a G-DVB entry.
- **Falsifier.** |arm − baseline| launch gap < 0.556 m/s on both seeds **and** `S_v` falls by < 0.1 ⇒ no effect at this resolution. Also fails if ADE_6s worsens by > 0.40 m (reported per family, never pooled).
- **Admissibility.** Training-only. Inference inputs unchanged (vision + measured speed + yaw rate). No steering-angle token (PlanTF's and Cultrera's states include steer/brake: excluded). No goal-path involvement.

## H-CRAWL-05. History truncation  [lever, rank 3]
- **Hypothesis.** The copycat channel is previous acceleration recovered from the speed history; a current-only input (speed, yaw rate) removes it.
- **Support.** History vs state-only, nuPlan: NR-CLS 56.5 vs 81-86, OLS 90.2 vs 81-88 [2309.10443 p.3, Table I]; history adds little and lowers CLS [2306.07962 p.4]. Counter-evidence: with speed given, copycat fixes gain less (reweighting only reaches the single-frame level) [2106.06452 p.7].
- **Predicted effect (ESTIMATED).** +0.05…+0.35 m/s; ADE_6s risk +0.1…+0.5 m (OLS −2…−9 points in Table I).
- **Experiment.** One arm with history length → 1, same init, 2 seeds, ≈ 1.3 box-hours (≈ 2.7 with replicates). If history length is not a flag, it is an architecture change and goes to the PI. Skip entirely if 02 returns `S_v ≤ 0.3`.
- **Falsifier.** Launch gap within 0.556 m/s of baseline on both seeds, or ADE_6s worse by > 0.40 m.
- **Admissibility.** Strict subset of allowed inputs.

## H-CRAWL-06. Loss-only vision-speed + stop/go heads  [lever, rank 2]
- **Hypothesis.** The trunk does not strongly encode "go soon" (trigger perception); supervising a vision-only ego-speed head (ego-speed token masked) and a stop/go head (stopped now, moving within 1 s) pushes it to, and raises launch speed.
- **Support.** Speed branch: up to +14 pp NoCrash success [1904.08980 p.6], but inertia remains and grows with data [p.7]. Stop/go loss + state noise: inertia failures 17/9/16 % → 1/1/6 % (train), 40/20/22 % → 8/2/4 % (new weather) [2310.20650 p.7, Table V]; residual stalls are perception errors in 56 % of 50 inspected episodes [p.8, Table VII]. Counter-evidence: our similar-signal launch up-weighting gave only +0.03 (INHERITED).
- **Predicted effect (ESTIMATED).** +0.10…+0.50 m/s, the widest range because the mechanism match is partial (they count binary 8-s stalls; we measure a continuous bias).
- **Experiment.** Precursor, zero training: linear probe for stop/go on frozen trunk features, held-out windows. AUC ≥ 0.8 ⇒ information present but unused (heads may help). AUC ≈ 0.5 ⇒ **inconclusive**, not negative (a linear-probe negative is not a negative about learnability). Then one arm with both heads (loss weight 0.1 each), 2 seeds, ≈ 1.3 box-hours (+ replicates), after PI approval and G-DVB entries.
- **Falsifier.** Gradients on the new heads and trunk are non-zero (zero control reads exactly 0.0) yet the launch gap moves < 0.556 m/s on both seeds ⇒ the representation is not the bottleneck.
- **Admissibility.** Labels from the GT future are fine (offline). **The stop/go output must have no inference consumer.** Cultrera's design conditions stage 2 on it, which would put a situation signal into the planner path: inadmissible. Speed-head target is measured ego speed predicted from vision features. No steering, brake or throttle inputs. Architecture change ⇒ PI.

## H-CRAWL-07. Weight / output averaging of same-init runs  [lever, rank 4, run first]
- **Hypothesis.** The run-specific component of the launch gap (σ_run ≈ 0.4 m/s, ESTIMATED from n = 1 pair) is removable by averaging same-recipe runs from the same init, while the shared bias (≈ −0.85) is untouched.
- **Support.** Same-init fine-tunes share a basin [2008.11687 p.1] and a linearly connected region [1912.05671 p.1]; averaging works when the variance term dominates [2205.09739 p.1]. Measured: greedy soup +0.7 pp ImageNet, **uniform soup 79.97 below best individual 80.38** when ingredients vary [2203.05482 p.6]; EMA/SWA models still gain from soups [p.30]; WiSE-FT +1.6 pp [2109.01903 p.1]; ensembles ≈ soups (81.19 vs 81.03) [2203.05482 p.6]; M = 5 default [1612.01474 p.5]. No driving-planner measurement found.
- **Predicted effect (ESTIMATED).** Uniform average of K = 4: SD of the soup's launch gap ≈ 0.2 m/s; the pair difference 0.556 → ≈ 0.28; mean gap ≈ mean of members; ADE_6s ≤ best member + 0.1 m. Its main value is **shrinking the floor** so 04-06 effects of 0.1-0.3 m/s become readable (compare soup-of-2-seeds per arm).
- **Experiment.** Zero training, minutes to tens of minutes:
  - interpolate the two existing runs at α ∈ {0, .25, .5, .75, 1} (launch gap, ADE_6s, eval loss);
  - uniform soup of all same-recipe same-init runs held (≥ 4 with replicates);
  - output ensemble (mean trajectory) of the same;
  - WiSE-style α sweep between the pre-fine-tune init and each run.
  - If the model has normalisation running statistics, recompute them (engineering note, ESTIMATED).
- **Falsifier.** (1) Barrier: ADE_6s(α = 0.5) > max(endpoints) + 0.40 m ⇒ not linearly connected ⇒ weight averaging invalid, use outputs only. (2) Soup gap outside [min, max] of members by > 0.2 m/s ⇒ non-linear behaviour. (3) "Bias unchanged" is falsified if the soup beats the best member by > 0.3 m/s; that would be a surprise and must be replicated before use.
- **Admissibility.** Post hoc; inference inputs unchanged. Output ensembling costs M× inference (INHERITED PI ruling from the auto-memory: 100 ms end-to-end on Thor, but never gate research on latency; I did not re-verify it).

## H-CRAWL-08. Copycat-error (APE) weights replacing `--launch-intent-w`  [lever, rank 5]
- **Hypothesis.** Weights from a history-only kinematic-extrapolation predictor's error select the copycat-failing keyframes better than a hand-set launch weight.
- **Support.** CARLA 33.0±4.2 → 43.4±0.8; but BCPD 28.7±2.5 and ActFreq 20.3±5.8 weights fall **below** unweighted BC-OH, and with speed available the gain only recovers the single-frame level [2106.06452 p.5-7, Table 1].
- **Predicted effect (ESTIMATED).** ≤ +0.10 m/s (central +0.03, because the hand-set version gave +0.03, INHERITED).
- **Experiment.** CPU precursor: compute APE (constant-acceleration extrapolation or a small MLP on speed history only) on training windows and its overlap with the launch-intent set. Overlap ≥ 70 % ⇒ already tested, drop. Otherwise one softmax-weighted fine-tune pair, 2 seeds, ≈ 1.3 box-hours (+ replicates).
- **Falsifier.** Overlap ≥ 70 %, or launch gap within 0.556 m/s on both seeds.
- **Admissibility.** Training-only. **Boundary:** sample weighting also belongs to `dagland/research_data/`; coordinate before running.

## H-CRAWL-09. Longitudinal speed-bin head  [lever, rank 6, conditional on 01]
- **Hypothesis.** If the stopped-bucket conditional distribution of future speed is bimodal (stay vs go), an explicit speed-bin classification head with confidence-weighted output removes median/mean interpolation at launch.
- **Support.** Waypoint regression interpolates multimodal velocity and the chosen mode "depends on the training run" [2306.07957 p.5-6]; but the fix was driving-score neutral (50±3 vs 49±8) and argmax raised collisions [p.6, Table 3].
- **Predicted effect (ESTIMATED).** 0…+0.20 m/s.
- **Experiment.** No training until 01. Test bimodality from 01's histograms: ≥ 25 % of mass near 0 and ≥ 25 % at ≥ 2 m/s one second ahead, in the stopped bucket (proposed thresholds). Cost: CPU.
- **Falsifier.** ≥ 85 % of the conditional mass within 1 m/s of its median ⇒ interpolation cannot explain the gap; drop the card.
- **Admissibility.** Architecture change ⇒ PI. The head must be computed from vision + measured ego only, never from the situation classifier.
