# REPORT_A: how JEPA-style predictors are validated and visualised across application areas (Research Lab agent A, 2026-11-07; written to file by the Master Mind from the agent's hand-back, because the harness blocked the agent's own Write)

Evidence key:
* `[PUBLISHED key]` = read in the banked primary.
* `[... abstract-level]` = banked, abstract or intro only.
* `[PUBLISHED-SECONDARY]` = not read in a banked primary.
* `[HYPOTHESIS]` = the agent's adaptation to our model.

Library key = arXiv id, except `areq-neurips2022` (alpha-ReQ). 26 new papers banked into the staging Library `F:/TanitAD-scratch/kbA` (`BANKED_A.txt`, sha256 re-checked against disk, 0 mismatches).

## Short answer
No single method proves a JEPA prediction is right. Papers combine four kinds of evidence:
1. **what the prediction carries** (decoders, probes, retrieval);
2. **whether it is right** (probes and task heads on predicted latents, with a persistence floor and an oracle ceiling);
3. **whether it depends on the action with the right sign and size** (counterfactual sweeps, inverse probes, matched interventions);
4. **whether the latent is healthy** (rank, collapse, retrieval against exact chance).

## Top 5 for our gaps (agent's ranking)
1. **Capture-gated matched-intervention audit** (Intervention Gap 2608.29998): persistence, zero-effect and oracle baselines, plus a gain and direction decomposition. It matches our 2× overshoot, the weak and wrong-sign steering and the wrong-sign lead gap. It also says propagation is readable only once capture is good, and our lead-gap capture fails.
2. **Probe transfer from encoded to predicted latents** (SG-JEPA 2609.10464 excess NMSE; UWM-JEPA 2605.25313 ΔR² teacher minus blind), with an action-shuffle control and a Copy-Last floor. It turns "levels 2 and 3 beat no-change" into a per-level, per-horizon table.
3. **Action-identifiability set**: ATM inverse-probe matrix (2606.09028), V-JEPA 2-AC action-energy sweep (2506.09985), IWM action-bank retrieval MRR (2403.00504; no conditioning 0.00 vs 0.82). Tests action dependence without a planner.
4. **Frozen task heads on predicted tokens with Oracle and Copy-Last bounds** (DINO-Foresight 2412.11673, driving data). The nearest precedent for imagining lead presence, agent count and lead gap. Decoders stay visual-only.
5. **Surprise (violation of expectation) on matched counterfactual clips with four controls** (Garrido 2502.11831, LeWM 2603.19312, scenario complexity 2606.28383): random encoder, shuffled scores, constant-velocity baseline, no-EMA collapse arm.

**Gate before all five:** rank plus retrieval against exact chance. Graph-JEPA 2608.20516 shows probe 0.871 and effective rank 18–47 on a latent that recovers 0.00 of 14.4 bits.

## Master table (12 families)

### F1 Stochastic post-hoc decoder (RCDM-style diffusion)
* Papers: RCDM 2112.09164; I-JEPA 2301.08243 §8; V-JEPA 2404.08471 §6; Sat-JEPA-Diff 2603.13943 (abstract-level).
* Validates what the prediction **carries**, not whether it is correct: features common across samples are in the prediction; features that vary were discarded.
* Protocol: freeze encoder and predictor; train a conditional diffusion decoder on the predicted features of the masked region only (it never sees context); draw several seeds. Faithfulness: re-encode the samples and rank the conditioning image among neighbours. MRR 0.99 (DINO), 0.97 (SimCLR), 0.69 (supervised); FID 19.0 on ImageNet-256 [PUBLISHED RCDM Tab. 1]. Sat-JEPA-Diff reports GSSIM 0.8984 and FID 0.1475 because PSNR/SSIM reward blur [abstract-level].
* Failure mode: the decoder's prior invents plausible detail. Neither I-JEPA nor V-JEPA reports a shuffled-condition control in the sections read. Qualitative only.
* Cost: hours to days of GPU [ESTIMATED].
* Map [HYPOTHESIS]: decode each level's 128-d latent to the DINOv3 token grid or a coarse agent raster; add a shuffled-condition decoder and a re-encode rank. Use to look, never as proof.

### F2 Deterministic decoder trained on encoder outputs only, applied to predictor outputs
* Papers: V-JEPA 2 2506.09985 B.3; DINO-WM 2411.04983; LeWM 2603.19312; H-JEPA 2610.06805 Fig. 4 (decoders "serve only for visualization").
* Protocol: V-JEPA 2 trains a ViT-L frame decoder (L2, 150k steps, 256×256×3) on **encoder** outputs, then applies it unchanged to action-conditioned predictor rollouts. DINO-WM cuts the decoder gradient from the predictor and reports LPIPS/SSIM next to planning success.
* Control: the decoder never trains on predictor outputs, so a good picture is not the decoder fitting the predictor.
* Failure modes: blur from low capacity; predicted latents off the decoder's training distribution (V-JEPA 2: the cup lands lower).
* Cost: hours. Map: one small decoder per level, trained on encoder latents only.

### F3 Probes on ENCODED latents
* Papers: V-JEPA, I-JEPA (Clevr count and depth), LeWM Tab. 1, H-JEPA, Point-JEPA 2404.16432, ECG-JEPA 2410.08559, Laya 2603.16281, US-JEPA 2602.19322, Brain-JEPA 2409.19407, S-JEPA 2403.11772, Audio-JEPA 2507.02915, A-JEPA 2311.15830, Graph-JEPA 2309.16014.
* Validates encoding only. H-JEPA: 2-layer MLP probes, variance-normalised MSE (1.0 = the marginal mean), 3 seeds. LeWM Push-T block angle: linear MSE 0.187 (r 0.902), MLP 0.021 (r 0.990).
* Controls: NMSE-1 floor; random-init encoder; Hewitt and Liang 1909.03368 control tasks and selectivity [abstract-level].
* Failure modes: probe capacity decides the answer; a probe can read the input when the target is in the input; a healthy probe can sit on a useless latent (F9).
* Map: our per-level reads. Add the NMSE-1 floor and a random-encoder row.

### F4 Probes and task heads on PREDICTED latents (probe transfer) — the core method for our gaps
* Papers: SG-JEPA 2609.10464; UWM-JEPA 2605.25313; DINO-Foresight 2412.11673 (Cityscapes, nuScenes); PSG-JEPA 2608.06799; PULS 2607.03558 (abstract-level).
* Protocol: SG-JEPA fits the probe on encoded latents, applies it to open-loop rollouts to 44 frames, and reports **excess** NMSE (rollout probe error minus probe error on encoded ground truth). UWM-JEPA reports ΔR² = R²(teacher) − R²(blind rollout): UWM 0.023 / 0.069 / 0.091 at h = 1/3/5, LSTM-JEPA 0.413 / 0.678 at h = 1/3; both reach noise by h = 10. DINO-Foresight runs **frozen** segmentation heads on forecast DINO features with Oracle (true future features, mIoU 77.0 ALL) and Copy-Last bounds.
* Controls: persistence floor; oracle ceiling; encoded-latent error subtracted; 3–5 seeds.
* Failure mode: a probe fitted on predicted latents re-learns the state from the copied context, so fit on encoded latents only. Below the persistence floor "readable" means nothing.
* Map: fit on level-k encoded latents, apply to level-k predicted latents at each horizon, report excess NMSE against Copy-Last and the oracle.

### F5 Nearest-neighbour retrieval
* Papers: IWM 2403.00504; RCDM; UWM-JEPA; Graph-JEPA failure 2608.20516; Audio-JEPA (kNN).
* IWM: a bank of 256 augmented targets; MRR with no conditioning 0.00, sequence conditioning 0.82, feature conditioning 0.79, plain I-JEPA predictor 0.00. Graph-JEPA failure: retrieval recovers 0.00 of 14.4 bits (MRR 1.9e-4 vs exact chance 1.99e-4, permutation p = 0.98).
* Failure mode: only as hard as the distractor bank; an easy bank gives a persistence copy MRR near 1.
* Map: bank = predicted latents for other actions or other windows of the same scene, with a persistence copy and an exact chance line.

### F6 Violation of expectation (surprise)
* Papers: Garrido 2502.11831; IntPhys 2 2506.09849; LeWM.
* Surprise = distance between predicted and encoded future latents per step, on paired clips with one impossibility. V-JEPA pairwise accuracy 98 % (CI 95–99) IntPhys, 66 % (64–68) GRASP, 62 % (60–63) InfLevel-lab; max over time beats mean. LeWM: teleportation spike significant (paired t, p < 0.01) in three environments; a colour change is not.
* Controls: untrained network near chance; pixel and multimodal-LLM baselines; a visual-only perturbation.
* Failure modes: needs minimal pairs; depends on context length and frame rate; tests the encoder as much as the predictor.
* Map: matched pairs from logs (lead removed or teleported, kinematically impossible cut-in); surprise per level.

### F7 Action-conditioned counterfactuals and planning
* Papers: V-JEPA 2-AC 2506.09985; DINO-WM; PLDM 2502.14819; JEPA-WMs 2512.24497; ATM 2606.09028; Intervention Gap 2608.29998; Delta-JEPA 2606.31232; UWM-JEPA; WM-in-RL probing 2603.21546.
* V-JEPA 2-AC sweeps one action with the rest fixed (energy minimum at (0, −0.05) vs truth (0, −0.1)). ATM: inverse probes on real and predicted transitions (2×2 matrix), ranking accuracy 98.8 % at a 5 % margin. Intervention Gap: (a) capture, a readout trained on real encoded states; (b) real-effect resolvability; (c) propagation, the same action sequence in the environment and in open-loop rollout, against persistence, zero-effect and the environment-endpoint oracle. LeWM checkpoints: imagined 5-step effects worse than predicting no effect, with "task-direction rotation with excess gain". UWM-JEPA: teacher-forced targets leave the action term inert; action sensitivity needs counterfactual targets.
* Failure modes: next-step loss does not predict control quality; planning success mixes planner and model; reconstruction-free JEPAs can collapse to action-insensitive latents; propagation is unreadable when capture fails.
* Map: our weak and wrong-sign steering, the 2× overshoot and the wrong-sign lead gap. Per-level audit on logged action contrasts: gain, direction cosine, sign.

### F8 Distance maps, attention, cost geometry, probe-direction interventions
* Papers: H-JEPA Fig. 7–8; LeWM; WM-in-RL 2603.21546; AIM probe 2603.20327.
* H-JEPA maps latent distance to a fixed anchor; a useful cost rises smoothly with path length; every multilevel model had at least one upper-level cost that beat the native level-1 cost. WM-in-RL shifts hidden states along probe directions and sees correlated prediction changes.
* Map: distance-to-goal maps per level; steer a probe direction (lead gap, speed) and watch the output.

### F9 Collapse and rank diagnostics
* Papers: RankMe 2210.02885; LiDAR 2312.04000; alpha-ReQ (abstract-level); dimensional collapse 2110.09348; SC-JEPA 2602.04643; Graph-JEPA failure.
* Failure mode: rank and probe both healthy on a useless latent (Graph-JEPA). RankMe is capped by min(samples, dimension); our cap is 128.
* Map: per level and per horizon on predicted latents. A gate, never a result.

### F10 Equivariance and invariance of the predictor
* Papers: IWM; seq-JEPA 2505.03176; Laya.
* IWM's retrieval MRR is the equivariance test; an 18-layer predictor reaches jitter equivariance in 4 of 5 runs, 12 layers in 1 of 5.
* Map: steering and acceleration must be equivariant in the predictor; F5 with an action bank is the test.

### F11 Predictor error as anomaly or complexity score
* Papers: scenario complexity 2606.28383 (nuPlan mini); 2602.09985, SC/MTS-JEPA, PULS (abstract-level).
* 2606.28383: AP 0.512 vs chance 0.436 (577 of 1,322 scenarios "complex"). Controls: shuffled scores (ρ −0.126, p 0.348), random encoder (ρ 0.024, p 0.860), constant-velocity baseline (ρ 0.314), no-EMA collapse (score spread 0.014 vs 0.602, cosine > 0.998 from epoch 3).
* Failure modes: modest effect; single-author preprint on structured states, not images.
* Map: per-level surprise on PhysicalAI windows against event tags, with the four controls verbatim.

### F12 Horizon consistency and composition
* Papers: SG-JEPA; H-JEPA.
* SG-JEPA: excess-NMSE curves over horizons and 25 gravity values, with a linear model separating local error from recursive amplification. H-JEPA re-anchors level-1 rollouts on level-2 subgoals.
* Failure mode: consistency is not accuracy; both paths can share one bias.
* Map: level-2 at 0.4 s vs two level-1 steps of 0.2 s in a common probed space; the 2× overshoot at level 1 should show as a slope above 1.

## Application areas
* Images: I-JEPA, IWM, RCDM, LiDAR. Video: V-JEPA, Garrido, IntPhys 2.
* Robotics: V-JEPA 2, DINO-WM, LeWM, PLDM, ATM, Intervention Gap, PSG-JEPA.
* Driving: DINO-Foresight, 2606.28383, H-JEPA, taxonomy 2603.09086.
* Audio: A-JEPA, Audio-JEPA. Time series: SC/MTS-JEPA, 2602.09985.
* Medical and EEG: ECG-JEPA, Laya, US-JEPA, Brain-JEPA, S-JEPA. Graphs: Graph-JEPA and its failure study. Point clouds: Point-JEPA, 3D-JEPA.
* Other (abstract-level): Sat-JEPA-Diff, RD-JEPA 2609.29403, VL-JEPA 2512.10942.

## What the literature does not settle
* No paper found reporting probe accuracy on predicted vs encoded latents per horizon **for a hierarchy** [PUBLISHED-SECONDARY: one web search plus the banked papers, no second-path check].
* No published control against prior hallucination for stochastic decoders in I-JEPA or V-JEPA, in the sections read.
* Driving JEPA validation is thin: 2606.28383 is a modest signal on nuPlan mini.

## Reading limits
* Read in banked text: RCDM, I-JEPA §8, V-JEPA §6, V-JEPA 2 (AC and B.3), IWM §4, Garrido, LeWM §5, UWM-JEPA, SG-JEPA, Intervention Gap, ATM, H-JEPA §3–4, DINO-WM, DINO-Foresight protocol, RankMe, LiDAR, Graph-JEPA failure, scenario complexity.
* Abstract-level only: alpha-ReQ, Hewitt and Liang, Audio-JEPA, A-JEPA, Brain-JEPA, S-JEPA, US-JEPA, 3D-JEPA, Sat-JEPA-Diff, RD-JEPA, VL-JEPA, PULS, 2602.09985, SC-JEPA.
* Banked earlier and not read here (follow-ups for F7 and F10): ACT-Bench 2412.05337, WorldLens 2512.10958, causal auditing of planners 2606.14438, invariant planning 2602.18639, Mind the Gap 2607.12547, rollout-decoded reconstruction 2608.25017.

## Manifest (agent A)
* `research_imagination/A/BANKED_A.txt`: keys, sha256, bytes, staging paths, deletion log.
* Staging Library `F:/TanitAD-scratch/kbA` (library.json and LIBRARY.md updated, 154 MB; only the 26 new PDFs remain). Nothing staged, committed or pushed.
* Deleted by the agent (re-creatable): 30 PDFs pulled from the tip to read, and 5.5 MB of extracted text; each logged in BANKED_A.txt.
