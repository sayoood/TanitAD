# Citation table — planner scorer inputs (2026-07-27)

Companion to `SCORER_INPUTS_RESEARCH.md`. **Read the "how it was read" column before quoting
anything.** Three access tiers, and the tier is part of the evidence class:

| tier | meaning |
|---|---|
| **PDF-VERBATIM** | PDF downloaded and text extracted; tables read line by line; quotes are exact |
| **HTML-SUMM** | arXiv HTML / ar5iv read through an automated summariser — **one extraction hop**, re-verify before spending a GPU-day |
| **ABS-ONLY** | abstract only |
| **INHERITED** | seen only inside another paper's table or related-work — **do not quote** |

---

## A. Systems with an explicit candidate scorer (the core of the deliverable)

| # | paper | link | how it was read | what we took from it |
|---|---|---|---|---|
| 1 | **Hydra-MDP** — End-to-end Multimodal Planning with Multi-target Hydra-Distillation (NVIDIA; 1st, CVPR 2024 E2E challenge) | https://arxiv.org/abs/2406.06978 | HTML-SUMM | scorer inputs (env tokens + ego status + vocab embedding); 4,096/8,192 vocab from k-means over 700 K nuPlan trajectories; the five distilled rule teachers; **distilling the SCALAR PDM score is worse (80.2) than imitation-only (80.9), while distilling the five metrics SEPARATELY is better (83.0)**; inference `f̃ = −(w₁log S_im + w₂log S_NC + w₃log S_DAC + w₄log(5S_TTC+2S_C+5S_EP))`, **weights by grid search** |
| 2 | **Hydra-MDP++** — Expert-Guided Hydra-Distillation | https://arxiv.org/abs/2503.12820 · https://arxiv.org/html/2503.12820v1 | HTML-SUMM | ⭐ **T.3: imitation-only 85.0 → + rule distillation 86.5 PDMS; T.4: 76.8 → 80.6 EPDMS** — matched vocabulary, target-only change. New teachers TL/DDC/LK/EC and what each reads. Rules are **training-time only**. |
| 3 | **WoTE** — End-to-End Driving with Online Trajectory Evaluation via BEV World Model | https://arxiv.org/abs/2504.01941 · https://arxiv.org/html/2504.01941v1 | HTML-SUMM | ⭐ **256 anchors** (same as ours), per-candidate BEV world-model rollout, reward head supervised by **BCE on simulator NC/DAC/TTC/Comf/EP**. T.3: **no evaluator 81.0 → current-state 83.2 → + world-model futures 85.6 PDMS**. Explicitly **does NOT** discuss WM failure to penalise implausible plans. |
| 4 | **VADv2** — Probabilistic Planning for E2E AD | https://arxiv.org/abs/2402.13243 | **PDF-VERBATIM** (v1) | ⭐⭐ **T.3: −image tokens 0.082→0.083; −map 0.086; −agent 0.089; −distribution LOSS 0.082→1.415 (17×).** 4,096 vocab by **furthest-point sampling over demonstrations** (feasible by construction). `L_distribution` = soft distance-weighted negatives; `L_conflict` = **a rule label over agent futures + boundary**. CARLA Town05 Long 85.1 DS — ⚠️ **a rule-wrapped number**. No vocab-size ablation, no oracle, `E_state` never ablated. |
| 5 | **DriveSuprim** (AAAI 2026) | https://arxiv.org/abs/2506.06659 | **PDF-VERBATIM** (v3) | ⭐ **T.1 oracle table over a 256 fan: top-1 91.9 / top-4 94.5 / top-16 96.1 / top-256 98.7 PDMS; human 94.8.** Diagnosis: *"easy-to-reject options dominate the training process and gradient."* Fixes: coarse-to-fine 2-stage scoring, rotation aug, self-distillation soft labels (δ_m=0.15). **89.9 → 93.5 PDMS (+3.6)** — narrows, does not close. |
| 6 | **GTRS** — Generalized Trajectory Scoring (NAVSIM v2 challenge **winner**) | https://arxiv.org/abs/2506.06664 · https://arxiv.org/html/2506.06664v1 | HTML-SUMM | scorer over a super-dense vocabulary; **random selection 25.6 → GTRS-Dense 39.7 EPDMS** (a scorer-vs-generator attribution); zero-shot to unseen proposals **+11.1 EPDMS**; navhard GTRS-E 49.4 vs **privileged PDM-Closed 51.3**. No oracle/coverage statement. |
| 7 | **GoalFlow** | https://arxiv.org/abs/2503.05689 · https://arxiv.org/html/2503.05689v1 | HTML-SUMM | goal-point scorer = **learned distance term + a LIVE rule DAC term**, `δ_final = w₁log δ_dis + w₂log δ_dac`. T.2 ladder **M₀ 85.6 → M₁ 88.5 (+goal) → M₂ 89.4 (+DAC) → M₃ 90.3 (+traj scorer)**. Human reference 94.8. |
| 8 | **DiffusionDrive** (CVPR 2025 Highlight) — **our REF-C's recipe** | https://arxiv.org/abs/2411.15139 · https://arxiv.org/html/2411.15139v3 | HTML-SUMM | **only 20 anchors**; scorer reads BEV + **agent queries** + **map queries** + trajectory features; **imitation-only supervision**. T.6: 10 → 84.9, **20 → 88.1**, 40 → 88.2 PDMS (**saturated at 20**). ⚠️ its T.3 "ID-2 55.1 / ID-3 97.1" as returned is internally implausible for a PDMS column — **UNVERIFIED, not used.** |
| 9 | **CLOVER** — Closed-Loop Value Estimation and Ranking | https://arxiv.org/abs/2605.15120 | HTML-SUMM | K=64: oracle 0.9933 vs selected 0.9369. **Expanding proposals moved the oracle (→0.9976) and left the pick where it was (→0.9413) — the gap did not move.** Closes ~12 % of its gap. |
| 10 | **SparseDriveV2** — *"Scoring is All You Need"* | https://arxiv.org/abs/2603.29163 | ABS-ONLY | denser anchors improve *"without exhibiting saturation before computational constraints"* ⚠️ **in tension with DiffusionDrive T.6 — both sides recorded.** **Reports no oracle or best-in-set number at all.** |

## B. nuPlan / rule-scored planners

| # | paper | link | how it was read | what we took from it |
|---|---|---|---|---|
| 11 | **PDM** — Parting with Misconceptions about Learning-based Vehicle Motion Planning (CoRL 2023; **1st, 2023 nuPlan Challenge**) | https://arxiv.org/abs/2306.07962 + **supplementary** https://www.cvlibs.net/publications/Dauner2023CORL_supplementary.pdf | **PDF-VERBATIM** (both) | ⭐ the full scorer term list + weights (**EP 5, TTC 5, Comfort 2**; NC/DAC/DDC multiplicative; **speed-limit and no-progress DROPPED because the generator enforces them**). Fan = **15** (3 lateral offsets × 5 IDM target speeds), **LQR + kinematic bicycle rollout 4 s @ 10 Hz**. T.4b: no-forecasting **−6 CLS-R**. T.4a: **letting the learned module own the first 2 s costs 34 CLS-R.** ⭐ *"PDM-Hybrid … evaluating proposals based on the **expected controller outcome**, causing it to match/outperform log replay"* — **Val14 CLS-R 92 vs human log-replay 80.** |
| 12 | **PDM official scorer code** | https://github.com/autonomousvision/tuplan_garage → `.../pdm_planner/scoring/pdm_scorer.py` | HTML-SUMM | independently confirms the term list and the 5/5/2 weights |
| 13 | **PLUTO** | https://arxiv.org/abs/2404.14327 | **PDF-VERBATIM** | ⭐ `π = π_rule + α·π₀`, **α = 0.3**. **T.VI rule-vs-learned head-to-head: rule-only 90.64 / mix 93.57 / learned-only 91.66** — the mix wins and the two fail oppositely. **T.VII: learned agent futures vs constant-velocity for scoring = +0.75 only.** T.II: +rule scorer worth **+4.17 CLS-NR / +12.05 CLS-R** (and **+4.66 / +11.01** on a third-party PlanTF generator). |
| 14 | **LLM-Assist** | https://arxiv.org/abs/2401.00125 | **PDF-VERBATIM** | ⭐⭐ **T.1: PDM-Closed at 15 proposals 92.51 → at 8,505 proposals 77.78 CLS-NR** (TTC 93.11→62.89, comfort 95.19→78.68, progress 91.75→**95.60**). Caption: *"PDMClosed fails to select the best proposal when presented with too many options."* |
| 15 | **PlanTF** — Rethinking Imitation-based Planners | https://arxiv.org/abs/2309.10443 | **PDF-VERBATIM** | the clean **pure-learning control arm** — deliberately no scorer, no post-optimisation |
| 16 | **DTPP** | https://arxiv.org/abs/2310.05885 | **PDF-VERBATIM** | ⚠️ **the pro-learned-scorer counter-example**: fixed generator + fixed predictor, hand-crafted → learned cost **CL-NR 0.7388 → 0.8964**. Caveats: not Val14; PDM still beats it there (0.9061); its "learned" cost keeps a **hand-crafted collision RBF** and reads ego-conditioned agent futures. ⭐ *"one could add additional cost terms, e.g. for lane keeping and route following. We did not find this necessary as our planner is constrained to follow a target lane through its trajectory generator."* |
| 17 | **hoplan** (2nd, 2023 nuPlan Challenge) | https://arxiv.org/abs/2306.15700 | **PDF-VERBATIM** | rule optimiser with 5 kinematic terms + occupancy raster + learned heatmap; post-solver **+0.060 cl-nr / +0.077 cl-r** |
| 18 | **Diffusion Planner** (ICLR 2025) | https://arxiv.org/abs/2501.15564 | **PDF-VERBATIM** | classifier guidance with **the same rule terms as PDM**, made differentiable; +rule refinement **+4.39 CLS-NR / +10.10 CLS-R** |
| 19 | **GameFormer** (ICCV 2023) | https://arxiv.org/abs/2303.05760 | **PDF-VERBATIM** | ⚠️ **its nuPlan cost term list is UNVERIFIED** — the paper does not enumerate it and the challenge tech report returned **HTTP 404** |

## C. Proposal-set construction / feasibility

| # | paper | link | how it was read | what we took from it |
|---|---|---|---|---|
| 20 | **CoverNet** | https://arxiv.org/abs/1911.10298 | HTML-SUMM | ⭐ **the only published ablation of feasibility-aware SET construction**: minADE₅ fixed ε=2 **2.62** → dynamic ε=3 **2.02** → hybrid **1.96**, and ~**half** the trajectories at equal coverage. Dynamic set = forward-integrating a kinematic model **from the current state**. |
| 21 | **PRIME** | https://arxiv.org/abs/2103.04027 | HTML-SUMM | the explicit **current-speed-anchored longitudinal band** `v ∈ [max(0, ṡ₀−δ⁻T), min(ṡ_max, ṡ₀+δ⁺T)]`; v_max 33.33 m/s, a ∈ ±8, κ ≤ 0.33. ⚠️ numeric δ⁺/δ⁻ **UNVERIFIED** |
| 22 | **TOAD** | https://arxiv.org/abs/2606.07170 | HTML-SUMM | control-space search through a kinematic bicycle → *"every sample is smooth and feasible"*; **94.7 PDMS navtest v1**. Argues **the SET is the bottleneck** — the opposing view to DriveSuprim. |
| 23 | **Apollo EM Motion Planner** | https://arxiv.org/abs/1807.08048 | HTML-SUMM | production lattice→QP with explicit speed/accel/**jerk** bounds; 68,000 km deployed; no numeric limits published |
| 24 | **"Slow Brain, Fast Planner"** | https://arxiv.org/abs/2606.20458 | HTML-SUMM | ⚠️ **sidewalk robots, not driving** — HYPOTHESIS tier. 64 feasible candidates, selected 1.64 m vs oracle 0.39 m (**4.2×**); selection plateaus at **K ≈ 18–24**; ⭐ **hiding the planner's own scores from the selector IMPROVED selection.** |

## D. End-to-end nuScenes family + the ego-status confound

| # | paper | link | how it was read | what we took from it |
|---|---|---|---|---|
| 25 | **UniAD** (CVPR 2023 Best Paper) | https://arxiv.org/abs/2212.10156 | HTML-SUMM (×2, agreeing) + source read | the collision cost reads **exactly two things**: distance to the regressed trajectory and predicted future occupancy. **Inference-only** (`if use_col_optim and not training`), σ=1.0, α_col=5.0. Its own T.10 says the optimiser buys collision at an L2 cost. |
| 26 | **PARA-Drive** (CVPR 2024) | https://openaccess.thecvf.com/content/CVPR2024/papers/Weng_PARA-Drive_..._CVPR_2024_paper.pdf | **PDF-VERBATIM** (via `curl --ssl-no-revoke`; CVF 403s WebFetch) | ⭐⭐ **removing UniAD's rule optimiser improves BOTH: collision 0.40→0.16, L2 0.83→0.74.** T.4: **command-only 4.66 m L2 vs BEV 0.53**; motion **queries** (0.54) recover BEV performance, motion **boxes** (1.10) do not. T.6: ego-only AD-MLP **L2 0.5568** ties the full stack. **Four protocol inconsistencies ⇒ the "VAD beats UniAD" margin is ~87 % artifact**; GT trajectories "collide" at **0.384 %** under axis-aligned boxes. |
| 27 | **VAD** (ICCV 2023) | https://arxiv.org/abs/2303.12077 | HTML-SUMM | the three constraints are **training losses only** (ego-agent collision δ_X=1.5/δ_Y=3.0; ego-boundary δ_bd=1.0; ego-lane direction). Worth 0.76→0.72 m L2, 0.28→0.22 % collision. ⚠️ "VAD is ego-status-free" is **UNVERIFIED** — the paper and BEV-Planner disagree, likely a revision artifact. |
| 28 | **BEV-Planner** — *Is Ego Status All You Need?* (CVPR 2024) | https://arxiv.org/abs/2312.03031 | HTML-SUMM (×2) | ⭐ ego status **halves** L2 on every architecture (UniAD 1.03→0.46, VAD 1.25→0.37, theirs 0.55→0.35). Perturbation: v×0.0 → L2 **6.16**; v=100 m/s → **208**. Their Curb Collision Rate **inverts** the L2 ordering. |
| 29 | **AD-MLP** — Rethinking the Open-Loop Evaluation | https://arxiv.org/abs/2305.10430 | HTML-SUMM | 21-dim input, **zero perception**, **L2 avg 0.29** beats VAD-Base 0.37 and UniAD 1.03. Its own ablation: **the high-level command alone** moves L2 0.49→0.29. |
| 30 | **NAVSIM** + `docs/metrics.md` | https://arxiv.org/abs/2406.15349 · https://github.com/autonomousvision/navsim/blob/main/docs/metrics.md | HTML-SUMM | the EPDMS term list and **exactly what each rule reads** (§2.3 of the main report); LQR-simulated ego; **Ego Status MLP 65.6 PDMS / 64.0 EPDMS** — the benchmark that is *not* fooled |
| 31 | **GenAD** | https://arxiv.org/abs/2402.11502 | HTML-SUMM | no scorer; ⚠️ **lowest confidence in the set** — its L2 appears to use UniAD's averaging convention, not comparable to VAD's |
| 32 | **TCP** (NeurIPS 2022) | https://arxiv.org/abs/2206.08129 | HTML-SUMM | two branches fused by a **hand-written rule**, α = 0.3, switch = "whether the ego is turning" |
| 33 | **Transfuser** (PAMI) | https://arxiv.org/abs/2205.15997 | HTML-SUMM | 64-d fused feature + position + **goal point**; creeping heuristic |
| 34 | **Transfuser++** — Hidden Biases of End-to-End Driving Models (ICCV 2023) | https://arxiv.org/abs/2306.07957 | HTML-SUMM | **decouples path from target speed**, the latter as a **4-way classification** with confidence-weighted averaging |

## E. Forecasting-side oracle gaps

| # | paper | link | how it was read | what we took from it |
|---|---|---|---|---|
| 35 | **LaneGCN** | https://arxiv.org/abs/2007.13732 | HTML-SUMM | Argoverse **test**: minADE₆ 0.87 vs minADE₁ 1.71 (**1.97×**); minFDE 1.36 vs 3.78 (**2.78×**) — gap never discussed |
| 36 | **TNT** · **DenseTNT** · **MTR** | 2008.08294 · 2108.09640 · 2209.13508 | HTML-SUMM | ⭐ **verified ABSENCE**: none reports an oracle / coverage / upper-bound ablation. **The brief's expectation that these frame scoring as the bottleneck is NOT supported.** |

---

## Verified absences — searched for and NOT found

`PUBLISHED-ABSENCE.` Full-text term search for `oracle | upper bound | ceiling | vocabulary coverage`
returned **nothing** in: **Hydra-MDP**, **GTRS** (the NAVSIM v2 winner), **VADv2**, **TNT**,
**DenseTNT**, **MTR**, **SparseDriveV2**. Across **nine** nuPlan papers (PDM, PLUTO, PlanTF, DTPP,
GameFormer, hoplan, Diffusion Planner, LLM-Assist, + code) there is **no oracle-selection ablation
at all**, confirmed by a second probe (two targeted web searches).

**⭐ Consequence:** the literature publishes only the **unrealisable** oracle. **Bar A's in-sample
ceiling — a re-scorer fitted on the very windows it scores, zero generalization gap — has no
published counterpart**, and neither does our ordering `0.4271 regression < 0.4907 best-achievable
ranker < 0.8563 actual`.

## Could not reach

| source | why |
|---|---|
| GameFormer nuPlan challenge tech report (`opendrivelab.com/e2ead/AD23Challenge/Track_4_AID.pdf`) | **HTTP 404** → GameFormer's cost terms **UNVERIFIED** |
| Hydra-MDP++ OpenReview PDF | verification wall; used the arXiv HTML instead |
| VADv2 post-v1 revision (NAVSIM / Bench2Drive / 3DGS tables) | v1 read in full; newer tables **UNVERIFIED** |
| pegasus_multi_path tech report | not attempted |

## Cross-paper reproduction discrepancies — do not treat Val14 numbers as interchangeable

- PLUTO's own Val14 CLS-R **92.06**; Diffusion Planner's reproduction of PLUTO **76.88**.
- PDM-Closed Val14 CLS-R reported as **92** (PDM), **93.20** (PLUTO), **92.12** (Diffusion Planner).
- PDM internal: §3.3 says forecasting at 10 Hz, supplementary §5.1 says 2 Hz. **Unreconciled.**
- DiffusionDrive T.6 (saturation at 20 anchors) vs SparseDriveV2 (no saturation). **Both recorded.**
