# LIT_B - literature for v8 (agent report, 2026-11-12)
Evidence: PRIMARY = arXiv abs page fetched this session (abstract only; bodies NOT read, so theorem statements and tables are UNSURE unless quoted). SECONDARY = search snippet only.

## T1. Hierarchy vs flat
1. When to Trust Your Model: MBPO. Janner, Fu, Zhang, Levine. NeurIPS 2019. arXiv 1906.08253. PRIMARY (abstract). Finding: monotonic-improvement analysis is overly pessimistic; an empirical estimate of model generalisation justifies model use; short model rollouts branched from real data improve sample efficiency and scale to long horizons. Exact form of Thm 4.1-4.3 (return gap vs rollout length k, model error, policy shift): UNSURE, not on the abs page; read the PDF before quoting. Use for v8: treat rollout length k as a tuned knob; branch imagined rollouts from real clip states and pick k by measured per-step val error vs horizon.
2. Near-Optimal Representation Learning for HRL. Nachum, Gu, Lee, Levine. ICLR 2019. arXiv 1810.01257. PRIMARY. Finding: defines a suboptimality measure of a goal representation by the reward of the best hierarchical policy using it, derives bounds on it and turns them into a representation-learning objective. Exact bound form: UNSURE. Use for v8: train or select the operative-to-tactical goal space for low suboptimality (goal must keep what the low level needs); log a proxy (goal-conditioned low-level ADE gap).
3. Why Does Hierarchy (Sometimes) Work So Well in RL? Nachum, Tang, Lu, Gu, Lee, Levine. arXiv 1909.10618. PRIMARY. Finding: in fully observed Markov settings most hierarchy gains come from better exploration, not easier policy learning or imposed structure; simpler exploration tricks match HRL. Use for v8: on offline imitation data there is no exploration channel, so any tactical-level gain must be tested against a flat model with the same horizon and parameters.
4. On Efficiency in Hierarchical RL. Wen, Precup, Ibrahimi, Barreto, Van Roy, Singh. NeurIPS 2020 (no arXiv id checked). SECONDARY. Finding: finite-time regret bound for model-based Thompson-sampling HRL separating planning error from learning error; big gain when the MDP decomposes into small repeating subMDPs with few exit states. Use for v8: check that the manoeuvre vocabulary has few exit profiles.
5. Deep Hierarchical Planning from Pixels (Director). Hafner, Lee, Fischer, Abbeel. arXiv 2206.04114. PRIMARY. High-level policy picks latent goals decoded to images, low level reaches them; beats exploration baselines on sparse-reward tasks (no numbers in abstract). Use for v8: decode tactical goals to image/BEV for audit.
6. Hierarchical Planning with Latent World Models (HWM). Zhang, Terver, Zholus, Chitnis, Sutaria, Assran, Balestriero, Bar, LeCun, Ballas. arXiv 2604.03208 (Apr 2026, v3 Oct 2026, CoRL 2026). PRIMARY. Multi-timescale world models in a shared latent; long-horizon predictions are subgoals for short-horizon planning via latent matching; an action encoder compresses action chunks into latent macro-actions; real Franka pick-and-place 70% vs 0% single-level; up to 3x less planning compute. Use for v8: closest published analogue of operative+tactical; try a macro-action encoder for the 6 s level and compare latent-matching subgoals against our vocabulary head.
7. H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning. Zhang, Terver, Rabbat, LeCun, Balestriero. arXiv 2610.06805 (Oct 2026, v2 Nov 2026). PRIMARY. Stack of action-conditioned JEPAs, each at a longer stride; top-down planning; Visual AntMaze success 18% to 73% with three levels at lower planner compute; ablations credit both temporal decomposition and higher-level goal representations. Use for v8: copy the ablation design (flat vs 2-level; decomposition vs goal representation). Simulated nav/manipulation only, not driving.
8. V-JEPA 2 / 2-AC. Assran et al. arXiv 2506.09985. PRIMARY. 1M+ h pretraining, SSv2 77.3, EK100 recall@5 39.7; 2-AC post-trained on under 62 h unlabeled Droid video, zero-shot planning with image goals. Use for v8: 62 h is a data-scale anchor for action-conditioned latent prediction.
9. DINO-WM. Zhou, Pan, LeCun, Pinto. arXiv 2411.04983. PRIMARY. Predicts DINOv2 patch features, plans by optimising actions toward goal features, six environments, offline data only. Use for v8: same recipe on frozen features; compare our CEM/iCEM settings.
Driving: I found no source where a multi-level driving world model beats a flat one at matched data (search not exhaustive). Treat as an open gap, not as a negative.

## T2. Rule injection and encoding
1. Hydra-MDP. Li et al. arXiv 2406.06978. PRIMARY. Student distills human and rule-based teachers via a multi-head decoder, one head per metric; no non-differentiable post-processing; 1st in Navsim challenge. Use for v8: per-metric heads (NC, DAC, EP, TTC, comfort) over the trajectory vocabulary, targets from a PDM-style scorer on every vocabulary entry.
2. GTRS. Li, Yao, Wang, Sun, Chen, Chang, Shen, Wu, Lan, Alvarez. arXiv 2506.06664. PRIMARY. Diffusion generator + vocabulary dropout so a scorer trained on very dense sets works on small subsets + sensor augmentation; won Navsim v2 challenge. Use for v8: vocabulary dropout when training the scorer.
3. DriveDPO. Shang et al. arXiv 2509.17940 (NeurIPS 2025). PRIMARY. Unified distribution from imitation similarity and rule safety scores, then trajectory-level DPO; NAVSIM PDMS 90.0. Use for v8: cheap post-training stage on existing vocabulary scores.
4. Safety filters. PC-Diffuser arXiv 2603.10330 (Mar 2026): CBF filter inside the denoising loop, capsule-distance barrier. Xu et al. arXiv 2505.02395: CBF safety filter (QP) with formal road-boundary guarantees, up to 40 Hz. DriveSafer arXiv 2605.16737: 48% fewer PDMS=0 failures than DiffusionDrive. All SECONDARY. Use for v8: hard guarantees come only from a post-hoc QP or shield; the driving-score cost of such filters is not measured in anything I fetched (UNSURE), so measure ADE and PDMS with and without filter on our set before adopting.
5. Rulebooks (Censi et al.) and STL losses: not fetched, no claim.

## T3. Imagination and test-time planning
1. Dreamer 4. Hafner, Yan, Lillicrap. arXiv 2509.24527. PRIMARY. RL inside a scalable world model trained mostly on unlabeled video; action conditioning from a small amount of data; first agent to obtain Minecraft diamonds from offline data only. DreamerV3 id 2301.04104 is from memory, UNSURE. Use for v8: action conditioning needs little paired data.
2. Think2Drive. Li, Jia, Wang, Yan. ECCV 2024. arXiv 2402.16720. PRIMARY. Latent world model as neural simulator for RL in CARLA v2; expert-level after 3 days on one A6000, 100% route completion. Reward form not in abstract.
3. CaRL. Jaeger et al. arXiv 2504.17838 (CoRL 2025). PRIMARY. Simple reward (route completion; infractions as termination or multiplicative cut); CARLA longest6 v2 DS 64; nuPlan Val14 91.3 non-reactive / 90.6 reactive; 300M (CARLA) and 500M (nuPlan) samples. Use for v8: if doing RL, use a multiplicative-infraction route reward, not a multi-term sum.
4. RAD. Gao et al. arXiv 2502.13144. PRIMARY. 3DGS in-the-loop RL with IL regulariser; 3x lower collision rate than IL baselines. Model-error containment: the IL regulariser.
5. WoTE arXiv 2504.01941 (BEV world model scores candidates online; NAVSIM and Bench2Drive SOTA, no numbers in abstract), LAW arXiv 2406.08481 (future-feature prediction as auxiliary, ICLR 2025), World4Drive arXiv 2507.00603 (-18.1% L2, -46.7% collision, 3.75x faster convergence per abstract). All PRIMARY. Use for v8: world model as selector over a candidate set is the cheapest containment: the model ranks, a fixed scorer supervises.
6. Epona arXiv 2506.24113 and DriveVLA-W0 arXiv 2510.12796 (ICLR 2026): SECONDARY. W0: future-image prediction as auxiliary gains more as data grows; simple autoregressive decoder beats flow matching at large scale, reversing the small-data trend. Use for v8: at 4.5k clips keep the small-data-friendly head.
7. Test-time CEM-style latent planning: V-JEPA 2-AC and DINO-WM (T1). Containment rule: short branched rollouts (MBPO).

## T4. Data selection
1. EL2N/GraNd. Paul, Ganguli, Dziugaite. NeurIPS 2021. arXiv 2107.07075. PRIMARY. Scores a few epochs in; can prune half of CIFAR-10 train set with slight accuracy gain. Use for v8: score clips by early per-clip loss or grad norm; check against the rig noise floor since runs are single-seed.
2. ActiveAD. Lu, Jia, Xie, Liao, Yang, Yan. arXiv 2403.02877. SECONDARY. Diversity metrics for the first batch, planning-oriented uncertainty after; comparable to SOTA with 30% of data on nuScenes open-loop and CARLA in-the-loop. Use for v8: diversity seed set (k-center on encoder latents) then uncertainty on tactical decision margin.
3. TRAK. Park, Georgiev, Ilyas, Leclerc, Madry. arXiv 2303.14186. PRIMARY. Attribution with a handful of trained models instead of thousands. Use for v8: attribute rare-manoeuvre val error to train clips and down-weight harmful ones.
4. Gradient-norm importance sampling, facility location, moderate coreset, VLM scenario mining: no primary fetched, no claim.

## T5. Decoding and physical-understanding probes
1. Representation Autoencoders. Zheng, Ma, Tong, Xie. arXiv 2510.11690. PRIMARY. Frozen DINO/SigLIP/MAE encoders with trained decoders; DiT reaches 1.51 FID (no guidance) and 1.13 (guided) on ImageNet 256/512. Use for v8: train a decoder on our frozen-encoder latents and decode predicted latents at 0.5/1/2/6 s; first check decoding of true-future latents.
2. IntPhys 2. Bordes, Garrido, Kao, Williams, Rabbat, Dupoux. arXiv 2506.09849. PRIMARY. Four principles (permanence, immutability, spatio-temporal continuity, solidity), violation-of-expectation; most models at chance (50%), humans near perfect. Use for v8: driving VoE test (vanishing lead car, pass-through) scoring latent surprise against an ego-motion-only control.
3. DINO-WM trains a decoder only for visualisation. Agent-state/BEV probe protocols with ego-motion controls: no primary fetched; proposal (ours, untested): probe agent positions from predicted latents vs from ego-shifted current latents.

## T6. Data-efficiency records
1. LFG: Learning to Drive is a Free Gift. arXiv 2602.22091. SECONDARY. NAVSIM PDMS at 1% / 10% / 100% labels: best 66.3 / 81.4 / 85.2; DiffusionDrive-DINOv2 baseline 57.3 / 74.4 / 81.5; at 10% labels matches DINOv3 full-data. Use for v8: report a label-fraction curve; label-free video pretraining first.
2. NoRD. arXiv 2602.21172. SECONDARY. 80k NAVSIM samples (over 60% fewer than a reasoning baseline); PDMS 76.66 base, 85.62 after Dr. GRPO; plain GRPO about +0.67%. Use for v8: reward normalisation matters in post-training.
3. Data Scaling Laws for End-to-End Driving. Naumann, Gu et al. arXiv 2504.04338 (CVPRW 2025). SECONDARY. 16 to 8192 h internal data. Use for v8: our 4.5k clips sit at the low end; quote slopes only with fit window, R2 and n.

## arXiv ids cited
1906.08253
1810.01257
1909.10618
2206.04114
2604.03208
2610.06805
2506.09985
2411.04983
2406.06978
2506.06664
2509.17940
2603.10330
2505.02395
2605.16737
2301.04104
2509.24527
2402.16720
2504.17838
2502.13144
2504.01941
2406.08481
2507.00603
2506.24113
2510.12796
2107.07075
2403.02877
2303.14186
2510.11690
2506.09849
2602.22091
2602.21172
2504.04338
