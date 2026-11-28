# HYPOTHESES: training data and stages for the v8F planner (8 cards)

Research Lab agent (Sonnet), 2026-10-28. Companion to `REVIEW.md` (read its section 1 first: what is already tried; section 5: noise floor).
Nothing here is registered or pre-registered; thresholds below are proposals for the Master Mind's PREREG. Evidence classes as in REVIEW: [PUB] banked PDF page, [MI] repo path at efb6223c, [INH] inherited, [EST] estimated, [HYP] hypothesis.

**Floors used in every card [MI]:** F_M2b = 0.1207 m ADE6 (2F = 0.2415); same-init replicate REP - CTRL ADE6 +0.4025 / +0.0116 m; launch speed training draw -0.556 m/s vs lever effect ~0.03 m/s; crawl bars (2 x max(F, RF0)): yaw-rate 1.1070, curvature 0.0526, L2 0.0136, M1 0.0096. Tier T1-kin unless stated; paired episode-cluster bootstrap B = 2,000 over 141 episodes; the training-run question is carried by a replicate arm (H-ESTIM-SEED-1).
**Constraints applied to every card:** inference = vision + measured ego history; no steering-angle input; no situation-classifier output in any goal path; labels may use anything; selection and tuning on TRAIN only (eval-141 and holdout `540:600` are never a selector); a new trainer flag needs a G-DVB entry (launch-gate rule).

## Ranking (expected effect relative to the floor, then cost)

| rank | id | kind | expected effect [EST] | in floors | P(clears its bar) | cost class | parity |
|---|---|---|---|---|---|---|---|
| 0a | H-DD-FITGAP | decider | decides whether cards 4-6 run | n/a | n/a | zero-training, dev box | none |
| 0b | H-DD-SEMLOC | decider | decides targeting for cards 3, 5, 6 | n/a | n/a | zero-training CPU | none |
| 1 | H-ST-RECOVERYLABEL | stage / labels | ADE6 -0.12 m (80% PI -0.35..+0.15) | 1.0 F | 0.15 (HARMS 0.3) | code + 4 box-2 runs | within-corpus |
| 2 | H-ST-DAGDOSE | stage / dose | ADE6 -0.08 m (-0.30..+0.10) | 0.7 F | 0.12 (guard breach 0.4) | 2-4 box-2 runs, flag exists | within-corpus |
| 3 | H-ST-REGRET | RL-lite weighting | crawl rows ~0.3 of bar; ADE6 +-0.1 | 0.3 bar | 0.20 (one of M1 / L2); 0.05 (all four) | zero-training pre-step + flag + 4 runs | within-corpus |
| 4 | H-DD-LAUNCHMIX | ADDED corpus | launch speed +0.10..+0.35 m/s closed of 0.85 | 0.2-0.6 of 0.556 | 0.15 (0.30 if GEN-GAP) | new data + PI + 4 runs | **PI decision** |
| 5 | H-DD-INFLUENCE | data attribution | ADE6 -0.05 m (-0.20..+0.10) | 0.4 F | 0.08 | zero-training GPU, then 2-4 runs | within-corpus |
| 6 | H-DD-SEMBAL | sampler reweighting | ADE6 -0.03 m (-0.20..+0.15) | 0.25 F | 0.05 | zero-training + flag + 4 runs | within-corpus (drop variant: PI) |

Order = expected effect in floors, with cost and P(clear) as tie-breakers; REGRET sits above LAUNCHMIX and INFLUENCE because its pre-step is free and its pass / fail is read on the crawl rows themselves. No card has an expected effect >= 2F. The reason is the floor, not the ideas (REVIEW section 5): at sigma 0.1-0.2 m, a 0.1 m effect needs 16-63 runs per arm. Cards 1-3 extend the only lever class that has cleared a bar (valid on-policy supervision); cards 4-6 are data levers the evidence puts below the floor and are gated by the two deciders.
Dependencies: FITGAP = FIT-LIMITED drops 4, 5, 6 (keep 1-3); FITGAP = GEN-GAP promotes 4 to rank 1. SEMLOC supplies the strata for 3, 4's query and 6.

---

## H-DD-FITGAP (rank 0a): is the launch crawl a fit gap or a generalisation gap?

**Hypothesis.** On the models' own TRAIN launch windows, the T1-kin launch-speed deficit is at least half its size on held-out windows: the crawl is fit- / loss-limited, not data-limited, so no data-amount or data-selection lever can clear a crawl bar.
**Instrument.** The M-LAUNCH statistic, mean(v_own - v_GT) at ticks 1-2 over GT-launch windows [MI REG H-V8F-LAUNCH-INTENT-1: 327 windows / 49 episodes on eval-141], computed on three window sets for 6 checkpoints (M2b s0/s1, DAG-OACC s0/s1, REP s0/s1): (i) TRAIN launch windows (M2b's own bank already holds 613,132 TRAIN own states with the matched route point: a file read [MI RAW/2026-10-27-dagger-run/bank_s0/bank.json]; other checkpoints need rollouts on a ~10% TRAIN clip subset); (ii) the 60-clip holdout `540:600`; (iii) eval-141 as the reference. r = C_train / C_eval, episode-cluster bootstrap.
**Proposed decision rule (commit before any number).** FIT-LIMITED if r >= 0.5 at >= 5 of 6 checkpoints; GEN-GAP if r <= 0.25 at >= 5 of 6; MIXED otherwise (holdout decides the tie: r_holdout within 0.1 of r_eval).
**Predicted [EST].** FIT-LIMITED, P = 0.6. Basis: the register already records "at full data T is fit-limited, eval ~ train" [MI REG H-V8F-DSCALE-1, R70 correction]; a 1.58926x weight on exactly the TRAIN launch windows left the eval crawl unmoved [MI LAUNCH-INTENT, M-LAUNCH +0.03 m/s]; point-estimate regressors saturate early [PUB 2603.15185 p6] and interpolate between trajectory modes [PUB 2603.15185 p1 Fig 1c] (stay vs go as the mode pair is my HYP).
**Falsifier.** r <= 0.25 at >= 5 of 6: the crawl is a generalisation gap; promote H-DD-LAUNCHMIX and H-DD-SEMBAL.
**Cheapest experiment.** Zero-training, dev-box inference only; ~1 agent-day. No box 2, no PI decision.
**Admissibility.** Read-only; eval used as a reference, never to select; no new input. Parity: none.

---

## H-DD-SEMLOC (rank 0b): the crawl and the run-to-run variance have a semantic address

**Hypothesis.** The crawl excess (and the REP - CTRL excess) is concentrated: >= 50% of each lies in <= 10% of TRAIN windows, and those windows fall in <= 5 strata nameable from labels the programme already holds (traffic-light state from the v7.x tactical labels, lead-vehicle departure from `lead_train4572.npz` / `obstacle.offline`, turn / junction from ego kinematics).
**Measure.** (a) k-means (k = 100-300; SemDeDup recipe at our scale [PUB 2303.09540 p4]) on [PCA-64 of mean-pooled DINOv3 `pool4x10` ; ego-history kinematics], fitted on TRAIN only. (b) Per cluster: crawl excess over the 613,132 TRAIN own states of the M2b bank (own speed shortfall vs the human at the matched route point; curvature chord noise), plus the across-checkpoint variance of the same quantity over >= 8 same-recipe checkpoints (DAG-OACC, REP, ORC A, LAUNCH-INTENT A / A_rep: all banked). (c) Top-decile share, Gini, and the enrichment of each nameable stratum. (d) **Confirmation, pre-registered, on eval-141** (same clusters, no refit): the top-decile share must reproduce within +-15 points.
**Predicted [EST].** Top-decile share >= 50%: P = 0.7. Basis: on eval the top 30 windows (1.4% of 2,106) carry 66% / 98% of the L2 excess, from 12 / 18 episodes [MI RAW/2026-10-26-lcdiag/DIAGNOSIS.md]. A nameable stratum with enrichment > 2: P = 0.5 (green-light go; lead departure). Alpamayo-R1 treats "resume at traffic light" as a first-class decision clip [PUB 2511.00088 p10].
**Falsifier.** TRAIN top-decile share < 30% and no label stratum with enrichment > 2: the crawl is diffuse; semantic selection cannot target it; drop cards 3 (targeting), 4 (query) and 6.
**Cheapest experiment.** Zero-training CPU on cached features and banked dumps; 1-2 agent-days.
**Admissibility.** Labels and strata only; none enters the goal path at inference (BINDING goal rule). The selector-leak trap is closed by fitting and choosing on TRAIN and treating eval as confirmation only. Parity: none.

---

## H-ST-RECOVERYLABEL (rank 1): give the discarded off-route own states a valid recovery label

**Hypothesis.** The DAgger bank throws away the model's largest deviations, because the human label at the matched route point is invalid beyond 3 m lateral / 30 deg heading (tick 2: 12.3% + 12.0% of 306,566 records; tick 1: 0.2% + 6.9%; coverage 0.756) [MI bank.json]. Labelling those states with a kinematically feasible recovery plan onto the human path (geometric blend at a lookahead, human speed profile from r', bounded by the P10 accel / curvature limits) and adding them to the own-state draw pool lifts usable tick-2 states from 53.9% to ~78% and lowers T1-kin ADE6 versus DAG-OACC at p = 0.1.
**Why now.** (i) CAT-K's principle: closed-loop SFT works because visited states stay near GT so the GT label stays a valid supervision signal; a 7M model beat a 102M one [PUB 2412.05334 p1, p6]. (ii) OPTED (LTFv6): logged-state BC 12.8, closed-loop baseline with recovery supervision 32.9, privileged-teacher supervision 41.8; the authors credit "unbiased rollouts and better supervision signal" [PUB 2609.20756 p4 Table I, p6]. (iii) Our ORC arm moved states but **left targets unchanged** and harmed (R-ADE +0.1434 m, R-T0 fails both seeds) [MI REG H-V8F-DAG-ORC-1]: the label half is the untested half. (iv) Iteration 2 added nothing (DAG2 - DAG ~0) [MI REG H-V8F-DAGGER2-1], so more of the same states is not the lever; different states with valid labels is.
**Predicted [EST].** T1-kin ADE6 vs DAG-OACC -0.12 m (80% PI -0.35..+0.15), concentrated in turn windows; P(<= -2F at both seeds) = 0.15; P(HARMS: R-ADE or R-T0 breach) = 0.30 (label noise; ORC precedent).
**Cheapest discriminating experiment.** (1) Zero-training, agent 1-2 days: build the labeller on the existing bank; identity test (an on-route zero-offset state reproduces the human label bit-for-bit) and a deliberate-regression arm (a labeller that ignores the offset must fail the gate); report the kept fraction (predict 53.9% -> ~78%) and the feasibility violation rate (must be 0). (2) Box 2: A_0, A_1 (p = 0.1, recovery pool on) against banked DAG-OACC s0/s1, plus A_rep,0/1: 4 runs x 38 min = ~2.6 h [MI summary.json 2,273 s per fine-tune]; T1-kin reads on the dev-box CPU.
**Falsifier.** ADE6 point estimate >= DAG-OACC's at either seed, or R-T0 above +0.0604 m, or the identity test fails. Then label validity beyond 3 m is not the lever and the DAgger axis is closed with DAGDOSE.
**Admissibility.** Labels use the future route (allowed); inference unchanged; no classifier output. Parity: derived from canonical TRAIN windows, eval-141 excluded by the bank's K-B3 control; kind = within-corpus derived states. Caveat: the bank rollout equals the eval rollout, so the gain is T1-kin-matched; T2 untested [HYP].
**Cost.** New code + 4 box-2 runs; box-2 PI go.

---

## H-ST-DAGDOSE (rank 2): DAgger is dose-limited at p = 0.1

**Hypothesis.** Raising the own-state fraction from p = 0.1 to p = 0.3 (flag `--dagger-p 0.3`, same bank, same 3,000 steps, DAG-OACC recipe) lowers T1-kin ADE6 versus DAG-OACC by >= 2F at both seeds without R-T0 above +0.0604 m.
**Basis.** The p = 0.3 arm was never run (p fell to 0.1 by a pre-registered mapping, not by a dose test) [MI PREREG_V8F_DAGGER.md header and outcome table]. At p = 0.1 only ~15-19k own-state samples enter a 192k-sample fine-tune (0.63 epoch; 19.2k before the 0.756 coverage cut) [MI argv.json, summary.json], yet the effect was -0.54 / -0.40 m. Controlling the on-policy proportion is the main knob in DARB [PUB Prakash2020_DARB_cvlibs p5]; OPTED refreshes a FIFO buffer of 512 rollouts, 32 per iteration, from the current student [PUB 2609.20756 p6]. Against: iteration 2 saturated; DAG s1 already sits at R-T0 +0.0483 of 0.0604 [MI REG H-V8F-DAGGER-1].
**Predicted [EST].** ADE6 -0.08 m (80% PI -0.30..+0.10) vs DAG-OACC; P(<= -2F both seeds) = 0.12; P(R-T0 breach at >= 1 seed) = 0.40.
**Cheapest experiment.** 2 box-2 runs (s0, s1; 38 min each) against banked DAG-OACC s0/s1 and the banked REP pair for the floor; the programme's standard form adds A_rep,0/1 (+2 runs, ~2.6 h total). No bank regeneration, no new flag. Dev-box CPU T1-kin reads.
**Falsifier.** ADE6 point > -2F at either seed or R-T0 breach: the dose axis is closed (with DAG2), leaving label validity (H-ST-RECOVERYLABEL).
**Admissibility.** Resampling fraction within the bank; inference unchanged. Parity: within-corpus.
**Resolvability warning.** Expected 0.7 F is unresolvable with 2 seeds; run it with the variance-reduced rig (SOUP / more replicates) or accept an INCONCLUSIVE row.

---

## H-ST-REGRET (rank 3): outcome-weighted DAgger draws (RL-lite)

**Hypothesis.** Drawing bank states with probability proportional to the model's own deficit at that state (progress shortfall versus the human at r', plus curvature-chord noise), mixed 50 : 50 with uniform draws, concentrates the p = 0.1 own-state budget on crawl states, and moves the crawl rows (M1, L2) by >= their bars at both seeds with guards held.
**Why.** 88-96% of the L2 excess sits on states "that exist only in the model's own rollouts" [MI LCDIAG]; the uniform draw dilutes them. This is the cheapest RL-flavoured stage: an advantage-style weight on already-labelled on-policy data, no simulator, no reward model. Published forms: closed-loop-selected upsampling cuts collisions up to ~35% [PUB 2210.02174 p4]; difficulty-prioritised training -15% collisions at 10% of data [PUB 2212.01375 p1]; critical-state sampling +16% [PUB Prakash2020_DARB_cvlibs abstract]; Alpamayo-R1's RL data = disagreement samples plus an equal random share [PUB 2511.00088 p22]. Against: LAUNCH-INTENT's window-level weight read null / harmful [MI]; uncertainty-as-hardness is negative [PUB 2604.08366 via DSR; Prakash2020_DARB_cvlibs p8], so the weight must be an outcome (deficit), not a confidence.
**Zero-training pre-step.** Measure the crawl-state share of the 613,132-state bank (own speed < 0.5 x GT speed at r', GT >= 3 m/s) and the effective sample size of the uniform draw. If the share is >= 15% the uniform draw already covers it: card dies. If < 3%: the uniform draw yields ~450-600 crawl samples in the ~15-19k own-state draws; weighting gives >= 5x.
**Predicted [EST].** M1 / L2 move ~0.3 of their bars; P(M1 or L2 clears at both seeds) = 0.20; P(all four crawl rows) = 0.05; ADE6 +-0.1.
**Cheapest experiment.** Pre-step (CPU); then a sampler flag (`--dagger-weight deficit`, G-DVB entry required) and 4 box-2 runs (~2.6 h).
**Falsifier.** Crawl rows within RF0 at both seeds, or launch speed moves less than the 0.556 m/s training-draw floor in the M-LAUNCH instrument.
**Admissibility.** Weights computed from GT quantities at r' (labels); inference unchanged. Parity: reweighting within the bank.

---

## H-DD-LAUNCHMIX (rank 4): add independent launch events from the other 301k clips

**Hypothesis.** The launch skill is event-limited: our corpus holds ~2k independent launch events (1,891 of 4,512 clips contain a launch window [MI f_measure.json]; events [EST]). Adding ~2-3k launch-rich clips mined from PhysicalAI-AV by an ego-speed query, mixed at <= 1/3 of draws, closes >= 0.1 m/s of the 0.85 m/s launch-speed deficit at T1-kin versus a matched-steps canonical-only control.
**Basis.** At constant total volume, doubling a long-tail scenario type gave +9.7..16.9% and quadrupling +22.8..32.9% [PUB 2412.02689 p6]; observed other-agent miles transfer at 2-3 : 10 [PUB 2506.08228 p19]; retrieval of relevant behaviours from a large unlabeled set helps [PUB 2304.08742 p1; 2412.15182 p7]; the abundant-pool regime favours hard / rare examples [PUB 2206.14486 p2]. Against: halving our corpus moved nothing [MI DSCALE]; random 10% = full set [PUB 2212.01375 p1]; cross-corpus pre-training swung -6.5..+4.9 PDMS [INH DSR s3.5, DriveVLA-W0 Table 7].
**Predicted [EST].** Launch-speed deficit improves by 0.10-0.35 m/s (0.2-0.6 of the 0.556 training-draw floor); ADE6 -0.05..-0.15. P(clears floor with 4 runs) = 0.15; 0.30 if FITGAP = GEN-GAP and the rig is variance-reduced.
**Cheapest experiment, staged.** Stage 0 (zero-GPU, no PI): one `egomotion` chunk (~41 MB) or `labels/egomotion.offline` (4.94 GB total; content UNVERIFIED) -> count clips with v0 < 6 m/s and v(6 s) >= v0 + 1 m/s; if fewer than ~500 beyond our 1,891 launch-bearing clips, stop. Stage 1 (new data): fetch ~2,000 front-wide clips (~125 GB [EST, 133 TB / (306,152 x 7)]), build DINOv3-L `pool4x10` features with the existing cache builder, diversify inside the candidate set by embedding (SemDeDup / STRAP recipe), join ego labels. Stage 2: box-2 fine-tune with the added pool at <= 1/3 of draws vs a canonical-only control at equal steps, 2 seeds + 2 replicates.
**Falsifier.** Stage 0 yields too few candidates; or Stage 2 launch-speed change is inside the training-draw floor at either seed.
**Admissibility.** Mining uses ego motion (label side); inference unchanged. **Parity: ADDED corpus.** Canonical TRAIN list, window-list sha and eval-141 untouched; added clips must be disjoint from eval-141 and the holdout by clip id (session / drive disjointness UNVERIFIED, needs `clip_index.parquet`); arms are not parity-comparable to banked arms, hence the mandatory canonical-only control. **PI decision:** ~125 GB download, new corpus kind.

---

## H-DD-INFLUENCE (rank 5): attribute the launch outcome to TRAIN windows

**Hypothesis.** A TRAK / CUPID-style influence score on the 7.2M-parameter model identifies <= 10% of TRAIN windows whose down-weighting raises launch speed on a TRAIN-side dev-launch set more than a size-matched random down-weight.
**Basis.** CUPID: < 33% curated data matches SOTA on Transport MH, using TRAK projections [PUB 2506.19121 p3 fn.1, p7]; DemInf +5-10% [PUB 2502.08623 p1]; Demo-SCORE +15-35% absolute [PUB 2503.03707 p1]; DataMIL: naive similarity selection can harm, a surrogate loss on target data replaces rollouts [PUB 2505.09603 p1]. All manipulation, small data; none in driving [HYP transfer]. Sorscher warns that pruning hard examples in a scarce set backfires [PUB 2206.14486 p2], so the arm down-weights negative-influence windows, never "hard" ones.
**Measure.** Per-window gradients of O's loss (adapter + O, ~2.4M params) projected to d = 4,096 at the M2b checkpoint (306,566 x 4,096 fp32 ~ 5 GB), scored against the gradient of the along-track loss on a dev-launch split of TRAIN clips (not eval-141). Then (i) rank stability across the 4 near-identical checkpoints (Spearman >= 0.5, else influence is noise at this variance); (ii) concentration in SEMLOC strata; (iii) only if stable: 2-4 box-2 runs, top-harmful 10% down-weight vs random 10% down-weight.
**Predicted [EST].** Stability >= 0.5: P = 0.5. Follow-on ADE6 -0.05 m (-0.20..+0.10); P(clear) = 0.08.
**Falsifier.** Rank correlation < 0.3, or the top-harmful down-weight is not better than the random control by > F/2.
**Cost.** Zero-training dev-box GPU gradient pass (~1-3 h [EST]); optional 2-4 box-2 runs.
**Admissibility.** Analysis; follow-on is down-weighting only (dropping windows would change the window-list sha: PI decision). Parity: within-corpus.

---

## H-DD-SEMBAL (rank 6): cluster-balanced sampler over the fixed corpus

**Hypothesis.** Sampling canonical windows with weights ~ n_cluster^-0.5 (capped 5x) over trajectory-pattern clusters (FlowDrive: k-means K = 20 on normalised ego futures) raises the launch / turn share of draws from 16.2% to >= 25% and does not worsen ADE6; unlike LAUNCH-INTENT it acts on the sampler, so S, T, the adapter and O all see the stratum.
**Basis.** FlowDrive Table 2a: none 81.91, scenario-type 80.08, cluster-based 85.37 (single run) [PUB 2509.21961 p9]; CAPS: driving score 59.5 -> 66.7, completion time 71.4 -> 50.0 s [PUB 2503.01650 p5]; D4: repeating intelligently beats baseline, repeating random is worse [PUB 2308.12284 p1]. Against: label-side scenario weighting is negative (80.08); LAUNCH-INTENT read HARMS; Sorscher (scarce data) [PUB 2206.14486 p2].
**Predicted [EST].** ADE6 -0.03 m (-0.20..+0.15); P(clear) = 0.05. Value is diagnostic: it separates "weight on one loss term" from "exposure of the whole stack".
**Cheapest experiment.** Zero-training: compute the clusters and the implied sampler shares and effective sample size; require launch share >= 25% before any run. Then a sampler flag (G-DVB entry) and 4 box-2 runs. The embedding-cluster variant (SemDeDup recipe) reuses SEMLOC's clusters.
**Falsifier.** ADE6 not better than uniform by > F/2 and launch / turn rows unmoved at both seeds.
**Admissibility.** Reweighting within the fixed corpus (window list unchanged). The SemDeDup drop variant re-selects windows: **PI decision, not proposed.** Gated by FITGAP.
