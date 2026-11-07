# LIT_REVIEW: how frontier AD trains and validates trajectory scorers, and where RL enters (stream SCORER-GYM, 2026-11-07)

**What this covers.** Each paper below is read from its primary PDF. All 19 PDFs are banked in `library/`, and each one's sha256 and one-line finding are in `BANK_LIST.json`. Seventeen were copied from `TanitAD Research Lab/Library/`, and each copy's sha256 matches `library.json`. Two are new: CaRL `2504.17838` and Gen-Drive `2410.05582`.

**Evidence classes.**
- Every number is **PUBLISHED**, read in the banked primary, unless it is marked otherwise.
- **INHERITED** means it was taken from the programme's own banked review `…/2026-10-11-refe-data-and-rl/RL_POSTTRAIN_REVIEW.md` and not re-read here.
- **MEASURED** means our artifact.

**Tiers.**
- NAVSIM navtest PDMS / EPDMS is a non-reactive, one-plan, 4 s simulation. The programme stamps it as T1 (`D-REFE-SFT-4`).
- navhard is NAVSIM v2's two-stage pseudo-simulation.
- nuPlan Val14 CLS is privileged closed loop: true multi-step, with no perception.
- RAD and CARLA are sensor closed loop.

⚠️ Do not compare numbers across these rows.

**Not done.**
- No web search for further 2026 work was run (context budget). The Library already holds the 2026 entries cited here (CLOVER, TOAD, DriveZero, the NAVSIM score-basis audit).
- FlowR2A `2606.24231` was **not read** and is not cited.
- Hydra-MDP++ `2503.12820` is cited only as a baseline row.

---

## 0 · Headline (five findings, each of which changes our design)

1. **Every NAVSIM-leading scorer trains with a full-information, one-step signal.**
   - The common form is per-proposal regression (BCE) of each simulator sub-score: Hydra-MDP, GTRS, DriveSuprim, WoTE, DrivoR, and CLOVER's scorer phase.
   - A listwise "expected reward" over all candidates is the same thing in a different form: ZTRS's EPO and our own SFT-4 `expreward`.
   - ⭐ **When every arm's reward is observed, "RL" (EPO) and "supervised listwise ranking" are the same gradient family.**
   - EPO's gradient `Σ_a Ψ(s,a) ∇π(a|s)` (ZTRS eq. 7) is the gradient of `E_π[Ψ]` with π a softmax over the candidates. That is exactly `scorer_finetune.py::expected_reward_loss` (`:113-119`).
   - ⇒ **Genuine RL only adds something when the consequence is multi-step**: the state you reach depends on your choice.
2. **Distilling the aggregate is worse than distilling the parts.**
   - Hydra-MDP regressing overall PDMS reads **80.2**, against **83.0** for per-sub-score targets (navtest, Table 1).
   - Every successor keeps per-sub-score heads. REFe already does (6 BCE heads plus lane).
3. **Validation of selection is regret against the set's oracle, not the planner score alone.**
   - CLOVER defines ranking regret `G = O − V`, the oracle minus the selected, and decomposes top-1 gain as *(oracle gain) + (regret reduction)*. It validates with pairwise ranking accuracy and top-K enrichment.
   - DriveSuprim reports its own top-K oracle curve (K = 1/4/16 → **91.9/94.5/96.1**).
   - ⛔ TrajHF's NAVSIM **94.3** is a **PDMS-selector (GT-scored oracle)** row. It is not a deployable selector, and it is cited as such elsewhere (`D-RL-NAVSIM-PRIOR-1`).
4. **Learned scorers are exploitable off their training distribution.**
   - TOAD: using GTRS's scorer as a search reward drives iPad **34.7 → 23.9** EPDMS on navhard. DrivoR's disentangled scorer reaches **49.8**.
   - Gen-Drive's learned reward over-optimises: **1,000 RL steps 0.8753 → 2,000 steps 0.8123** (nuPlan reduced Val14).
   - ⇒ Any scorer trained against its own optimiser needs a held-out true-metric check at every read, with a kill rule. RL-S1's K3 is that rule, and it fired (`D-REFE-RLS1-R0`).
5. **Where genuine multi-step RL works, the reward is simple and multiplicative, and the policy is a privileged teacher.**
   - CaRL: route completion, terminate on infractions, multiplicative soft penalties. The dense shaped (Roach) reward gets stuck in **a "not driving" local minimum** at large mini-batch.
   - DriveZero: hard-event penalty, goal bonus, and **(1/H)·∏ of six soft criteria**. It uses a **decomposed per-term critic** and a camera student distilled from the teacher's rollouts (**95.3 PDMS navtest**).
   - This is the template for the PI's scorer-gym.

---

## 1 · Per-paper table

| paper (arXiv, banked) | scorer training signal | credit | how selection is validated | reported numbers (benchmark, tier) |
|---|---|---|---|---|
| **PDM-Closed** (2306.07962) | none (rule-based). 15 IDM proposals = 5 target speeds {20…100 %} of the limit × 3 lateral offsets {−1, 0, +1 m}; constant-velocity forecasts over 8 s; LQR simulation; nuPlan-metric scoring; argmax | multi-step in the *simulator* (4 s proposal horizon), re-planned every step | closed-loop CLS | Val14 **CLS-R 92 / CLS-NR 93**, OLS 42 (privileged closed loop) |
| **Hydra-MDP** (2406.06978) | softmax imitation over the 8,192 vocab (L2 to the human) **+ per-entry BCE on each offline PDM sub-score** (eq. 10). Inference: weighted −log sum (eq. 11), weights grid-searched | one-step, full information | the planner score only | navtest PDMS: V8192 **83.0**; aggregate-PDMS target **80.2**; +weighted confidence **85.7**; +EP target **86.5** (NAVSIM v1, T1) |
| **GTRS** (2506.06664) | Hydra-style BCE, trained on a **16,384 super-dense vocab with 50 % vocabulary dropout**, scored on other sets | one-step, full information | **generalisation to UNSEEN proposals**: scores diffusion proposals it never trained on | navhard EPDMS: random diffusion pick **25.6**, zero-shot scorer **36.7**, dropout **43.4**, ensemble **49.4** (pseudo-sim; NAVSIM v2 challenge winner) |
| **DriveSuprim** (2506.06659) | coarse stage: BCE + imitation over the vocab. **Top-k filter → refinement decoder trained only on the filtered hard negatives** (BCE per layer); rotation augmentation; **EMA-teacher soft labels** | one-step | **own top-K oracle curve**: K = 1/4/16 → 91.9/94.5/96.1 PDMS; human 94.8 | navtest EPDMS ablation 81.4 → **83.1**; headline 93.5 PDMS v1 / 87.1 EPDMS v2 (T1) |
| **WoTE** (2504.01941) | reward model reads the **BEV world model's predicted future states** per proposal; targets = simulator sub-scores + an imitation reward (CE) | one-step labels; the WM supplies future context | ablations | navtest **88.3**; imitation-only 83.5 / sim-only 83.6 / both 85.6; 0→4 s single step 84.0 vs 0→2→4 s 85.6 (T1) |
| **DrivoR** (2601.05083), REFe's reference | **per-sub-score BCE on its own proposals**, scorer detached from the trunk ("disentangled") | one-step | planner score | navtest PDMS **93.1** (train) / **93.7** (trainval) (T1) |
| **CLOVER** (2605.15120), REFe's family | scorer phase: per-sub-score loss against **true evaluator sub-scores on the CURRENT generator's proposals** (eq. 5; loss form ℓ not checked); then generator self-distillation toward scorer-selected Pareto targets | one-step, alternating | ⭐ **ranking regret G = O − V** (App. D.5); pairwise ranking accuracy **94.72 %** on the ≥ 0.95 vs ≤ 0.50 groups; enrichment; Top-8/16 best-true 0.9656/0.9730 | Stage 1 94.13 → Stage 2 94.48 PDMS navtest **(INHERITED, RL_POSTTRAIN_REVIEW §1)** |
| **ZTRS / EPO** (2510.24108) | ⭐ **EPO: one-step offline policy gradient summed over ALL 16,384 actions.** Advantage Ψ = EPDMS − λ·1[EC violated vs the previous pick], normalised to zero mean and unit variance (eqs. 7-8). No human data | one-step; the only temporal term is the previous-pick consistency penalty | navtest, navhard, **HUGSIM closed loop** | navtest EPDMS **85.3** vs human-IL 86.2 (**EP 84.1 vs 88.5**); argmax-EPDMS imitation **76.7**; Ψ = E without b: 84.2 (EC 53.8 → 77.2 with b). GRPO "fluctuates", attributed to sampled subsets |
| **DriveDPO** (2509.17940) | DPO over PDMS-ranked candidate pairs; **rejected = the imitation-closest unsafe candidate** | one-step, pairwise | ablation over the rejected-pair rule | navtest **90.0** (imitation-based rejected) vs 89.7 (distance-based) (T1) |
| **CaRL** (2504.17838) | (planner, not scorer) **PPO, reward = route completion; terminate on major infractions; multiplicative soft penalties**; survival bonus in nuPlan | **multi-step**, closed loop | closed-loop CLS | Val14 **91.3 NR / 90.6 R** CLS (privileged closed loop); 500 M samples on one 8-GPU node; the Roach dense reward hits a **"not driving" local minimum** at mini-batch 1,024 |
| **DriveZero** (2609.06055) | privileged PPO teacher in mixed-agent nuPlan worlds (log-replay / IDM / self-play). **r_t = h_t + (1−d_t)g_t + (1/H)∏_k q_{t,k}** over six soft criteria, H = 110. Per-term decomposed critic; value-guided test-time reranking; camera student distilled from teacher rollouts | **multi-step** | nuPlan closed loop + navtest + navhard + HUGSIM | student navtest **95.3 PDMS** with no human trajectories; navhard 57.1 EPDMS (T1 / pseudo-sim) |
| **RAD** (2502.13144) | PPO in 3DGS digital twins; rewards for dynamic/static collision and position/heading deviation; IL as a regulariser | multi-step, sensor closed loop | own 3DGS closed-loop benchmark | collision rate **0.089 (RL+IL) vs 0.229 (IL)** (sensor closed loop, own benchmark) |
| **Gen-Drive** (2410.05582) | scene evaluator trained on **VLM-curated pairwise preferences**; RL fine-tunes a diffusion generator | multi-step generation, one evaluation | nuPlan closed loop | reduced Val14: learned-reward selection 0.8512 vs PDM-scored 0.8087; RL **0.8753 vs PDM-Closed 0.9121**; ⚠️ **over-optimisation at 2,000 steps: 0.8123** |
| **TrajHF** (2503.10434) | RLHF for style personalisation | — | — | ⛔ NAVSIM 94.3 is the **PDMS-selector (oracle)** row; no controlled NAVSIM RL gain |
| **TOAD** (2606.07170) | (consumer of scorers) test-time optimisation with a scorer as the search reward | — | ⭐ **adversarial validation: is the scorer still right off its training set?** | iPad 34.7 → **23.9** with the GTRS scorer, **49.8** with DrivoR's (navhard) |
| **NAVSIM** (2406.15349) / **pseudo-sim** (2506.04218) | — | — | closed-loop correlation of the open-loop metric | PDMS correlates with CLS better than displacement metrics; pseudo-sim R² **0.8** vs 0.7 for the best open-loop metric |

---

## 2 · Synthesis for REFe (what the literature does and does not settle)

**2.1 The one-step problem is already solved in form, and our u400 already uses that form.**
- MEASURED from source: SFT-4 arm B, i.e. u400, was trained with `--b-mode expreward --lam 1.0` (`refe/sft4_pod.sh:10`).
  - That is per-head BCE plus `−E_π[R]` with π = softmax(log agg / τ_s) and R = **true** PDMS (`scorer_finetune.py:113-119, 1147, 1158`).
  - This is EPO's estimator with a fixed temperature.
- RL-S1 then added EPO with a shaped reward and a KL anchor (`--epo`, `:1161-1172`). It bought PDMS **+1.16 [+0.59, +1.74]** and lost TTC **−1.00** (`D-REFE-RLS1-R0`, MEASURED, RL-val selection tier).
  - ⇒ The literature's main one-step levers have **been run on this rig**. ⛔ Proposing "train the scorer with EPO/listwise on true PDMS" as new would be wrong.
- What remains open is **why a listwise-on-truth scorer still reads speed as danger**. P1 Q4 shows the predicted-vs-true within-token Spearman against v4/v0: TTC −0.48 vs −0.05.
  - The likely reason, a HYPOTHESIS: the listwise term acts on the **aggregate**, at τ_s 0.1, i.e. on the top of each set. The per-head BCE is **pointwise across tokens**, so a head can satisfy BCE by learning speed as a cross-token proxy for danger.
  - O2-CALIB already measured that the heads are well calibrated (slope 0.90–1.02), and that **a monotone map cannot change within-scene ranking**.
  - No published scorer trains its *safety heads* with a within-token pairwise objective. DriveSuprim's hard negatives and DriveDPO's rejected-pair rule are the closest relatives.

**2.2 Multi-step credit is the genuinely new thing, and the published route to it is a privileged teacher.**
- CaRL and DriveZero train privileged closed-loop policies with simple multiplicative rewards. DriveZero distils into a camera student by imitating teacher rollouts and gains over human-trajectory training.
- No paper trains *only a selector* in closed loop. PDM-Closed is the closest: a selector over IDM proposals, rule-scored on a simulated 4 s future. It reaches **92/93 CLS**, which is the bar a privileged learned selector must at least match.

**2.3 Validation practice to adopt.** Our programme already has 1-3, 5 and 6 in places, but not as one battery.
1. Regret against the set's oracle, per stratum.
2. Pairwise ranking accuracy on separated groups.
3. Top-K oracle curve.
4. **Within-token rank correlation of each head against the behaviour covariates** (speed, gap). This is our P1 Q4. It is not in the literature, and it is the instrument that found this defect.
5. Adversarial / off-distribution scoring (TOAD).
6. Closed-loop behaviour with a kill rule on the true metric (Gen-Drive's over-optimisation curve is the warning).

**2.4 Risks the literature measured.**
- Reward hacking by not driving (CaRL).
- Over-optimisation of a learned reward (Gen-Drive).
- Scorer exploitation by a search (TOAD).
- EP traded for safety under RL: ZTRS EP −4.4; our R0 TTC −1.00.
- Oscillation without a consistency term: ZTRS EC 53.8 → 77.2.
