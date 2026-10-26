# HiCAP stream R-C — prediction structure and cheap imagination inside a contrastive selector (2026-09-29)

**Status:** COMPLETE — staged in the index, not committed. No pod, no GPU, no torch (numpy only). Scope: what to add to the REF-F / HiCAP
selector (frozen VLM -> 512-d state embedding `s`; cached action embeddings; score `tau*<s,a> + prior terms`; prefill-only) so that it has
**prediction structure and imagination-based reasoning at maximal compute efficiency**.

**Evidence classes.** `PUBLISHED` (arXiv id + title seen in a search-result list *this session*; **content is search-snippet level** because
arxiv.org / huggingface.co / alphaxiv are egress-blocked for WebFetch — no paper was read as primary) · `PUBLISHED-RECALLED` (statement from memory,
re-verify before quoting) · `INHERITED` (programme doc or the brief, not re-verified; registry not opened — no pod) · `MEASURED (toy)` (this stream:
`rc_latent_imagination_toy.py` -> `rc_latent_imagination_toy_result.json`) · `ESTIMATED` · `HYPOTHESIS` · `UNVERIFIED`.
**Nothing measured on the toy is a driving result.**

## Headline findings

1. **Target shape, not imagination, is the biggest measured lever.** Replacing the one-hot expert label by per-candidate outcome-aware soft targets
   (zero inference cost) cuts toy regret **7.00 -> 3.43** (n=1200; paired Delta 3.57 [2.99, 4.19]). A *z-blind* risk-sensitive lookup that has **no access to
   the hidden consequence** reaches **2.17**. So most of "imagination beats the CLM baseline" is risk-sensitive expected-cost selection (MEASURED toy, §5).
2. **Hidden-consequence information is worth little at low cue SNR and moderately at high SNR.** SNR 1: the best learned arm (A5m 1.81 / 1.91 at n = 1200 / 4800) beats the z-blind floor
   (2.17 / 2.04) by only 0.1-0.4 and every unstructured arm is at or above it (A2 2.17 / 2.90, QM 2.97). SNR 3 (mode accuracy 0.91): re-rank on a strong proposer 1.42, full rollouts 1.74 vs floor 2.17 — a 20-35 % gain (MEASURED toy).
3. **Unstructured latent imagination is catastrophic without counterfactual data** (expert-only rows: pure imagination regret **71.8**, 40 % unsafe; model-free Q
   **47.0**; distilled selector **48.2**; A0 7.3). A **factorised** imagination (learn only the exogenous mode posterior from *logs alone*, evaluate candidates
   analytically) does not care: **1.80**. Its ceiling with the true visible state is **0.46** (SNR 3); **the whole gap to that ceiling is readout error of visible
   metric state from the embedding** — the same failure as the programme's LF0 (lead distance not decodable from the v5f latent) (MEASURED toy).
4. **Compute:** on cached 512-d embeddings the top-8 imagination path costs ~0.10 GFLOP = **0.013 %** of the smallest prefill; all-K iterated rollouts cost ~10 %
   and are not admissible (ESTIMATED, §3). Imagination FLOPs are not the constraint — **identifiability, echo and calibration are.**
5. **Gating:** a disagreement or margin gate captures no more of the gain than a random gate (30-39 % of the gain at 30 % budget vs 35 % random); imagination *hurts* on
   13.4 % of ticks and helps on 18.6 %. Two-head disagreement is a good **out-of-vocabulary veto** (AUROC 0.88, 64 % recall at 5 % FPR), not a benefit gate. A learned
   benefit gate reaches 65 % of the gain at 30 % budget (MEASURED toy; 400 validation ticks).
6. **Anti-calibration is reproduced qualitatively:** adding an imagined-cost term flips Spearman(confidence, regret) from **-0.11 to +0.19** (3/3 seeds each).
7. **Theory:** with a future-state positive the InfoNCE optimum is a density ratio of the *dynamics* (action-independent denominator, no PMI-vs-likelihood
   bias, but unidentified off the data manifold). With per-state negatives it equals `log p(a|s,f) - log pi(a|s)` = conditional pointwise MI; adding the action critic
   gives `log p(a|s,g)` — planning as inference. One **state tower** can be shared; one objective cannot (§1).
8. **2026 literature corroborates the programme's own defects:** DA-WAM, D-JEPA, DA-LeWM, AD-WM, Temporal-Distance-JEPA, IMWM all report that a JEPA latent can decode task
   variables yet **rank candidate plans wrongly** (snippet level, §2).
9. **Pre-registered readings:** H1 supported but confounded · H2 refuted as worded · H3 supported · H4 refuted · H5 refuted as worded (§5.3).

---

## 0. What the programme already knew — this document reports the delta

| already known (INHERITED, `origin/handoff` docs / brief) | what this document adds |
|---|---|
| `MPC_WM_DESIGN`: sampling-MPC over the frozen WM; E0 cost-term probes (ridge from z-hat to lead gap); latent-plausibility cost | cost structure is **factorised** (exogenous learned, ego analytic); FLOP arithmetic at cached-embedding scale; toy evidence that the readout probe (E0) is the gate |
| `DIFFUSION_MPC_SYNTHESIS` L1 (per-candidate rolls on top-8), L4 (distil MPC into the selector; trigger on margin / ensemble disagreement / flags) | toy: L4-style distillation works **only with counterfactual rows**; margin/disagreement triggers are ~random for *benefit*; a learned trigger works |
| `HIERARCHICAL_WM_REDESIGN` §2.1 counterfactual-supervision problem, §2.3 B2 ensemble pessimism | measured size of the failure (regret 47-72, 19-40 % unsafe) and of the disagreement veto (AUROC 0.88 / 0.71) |
| `PBATTERY_V6_FIRST_RESULT`: R2(z-hat, speed) 0.995 collapses to -0.72 when v0 is shuffled | the same split in the toy: ego-speed R2 0.62 -> 0.61 under shuffled cue (echo, invariant) vs lead-speed 0.39 -> -0.33 |
| `JEPA_PHYSICS_SURVEY` LF0: RC1 refuted, lead not decodable from the latent | maps to risk R1 below: factorised imagination lives or dies on a metric readout |
| F-R doc §3.1 (`2026-09-29-reff-prior-art-and-theory.md`): PMI vs likelihood; contrastive RL contrasts over goals | extends to a **future-state positive**, the Bayes link between the two critics, the shared-space question (§1) |
| brief: anti-calibrated imagination (H15); WM-rollout scoring ~22x over Thor budget; own-WM-scores-own-fan +0.21 vs reference roll -0.29; 2-head features R2 0.5526 vs 0.070 | all INHERITED, used as constraints; none re-measured here |

---

## 1. Contrastive prediction as the unifying objective

**Notation.** `s` = state embedding (frozen VLM + trained tower `u`, unit 512-d); `a` = candidate; `f_k = sg(u(o_{t+k}))` = embedding of the frame `k` steps later (same tower, stop-grad
target); `g` = goal embedding; `pi_E` = expert policy; `P_k(f|s,a)` = k-step dynamics. Evidence for all optima below is `PUBLISHED-RECALLED` (CPC 1807.03748, Poole 1905.06922; ids
confirmed by the F-R stream), with the algebra checked by hand here; F-R's toy B3 verified the general-`q` form numerically to 4 d.p.

### 1.1 Three critics and their optima

| critic | positive | negatives | optimum (InfoNCE, `+c(.)` = unidentified offset) | ranks candidates by |
|---|---|---|---|---|
| **(A) action critic** (CLM / HiCAP score) `<u(s), v(a)>` | `a+ ~ pi_E(a\|s)` | `a- ~ q(a)` | `log pi_E(a\|s) - log q(a) + c(s)` | evidence vs the action prior (**PMI, not likelihood** — F-R §3.1) |
| **(B) outcome critic** `<phi(s,a), w(f)>` | `f+ ~ P_k(f\|s,a+)` (future under the *executed* action) | `f- ~ q_F` | `log P_k(f\|s,a) - log q_F(f) + c(s,a)` | dynamics likelihood ratio |
| **(C) occupancy / goal critic** (contrastive RL, 2206.07568) | `f = s_{t+K}, K ~ Geom(1-gamma)` | `f- ~ p(f)` | `log p^pi_+(f\|s,a) - log p(f)`, `p^pi_+ = (1-gamma) sum_k gamma^k P_k` | discounted reach probability of `g` (a goal-conditioned Q for reward `delta_g`) |

**What the optimum is when the positive is a future state (answer to the brief).**
1. It is the density ratio of the **dynamics/occupancy**, not of the policy. The denominator `q_F(f)` does not depend on `a` at fixed `s` and `g`, so `argmax_a` **is not prior-biased**
   — the opposite of (A), where `q(a)` depends on `a` and `argmax` picks evidence, not the modal action (F-R toy: 0 % agreement with MAP in ambiguous scenes).
2. With **per-state negatives** `q_F = p_pi(f|s) = sum_a pi(a|s) P_k(f|s,a)` (futures of *other candidates from the same state*), Bayes gives
   `h*_B(s,a,f) = log P_k(f|s,a) - log p_pi(f|s) = log p(a|s,f) - log pi(a|s)` (1) — the **pointwise conditional mutual information** `i(a; f | s)`; its expectation is `I(a; f | s)`, capped at
   `log N` for `N` negatives. AD-WM's "normalised action-recovery objective motivated by conditional mutual information" (README read, no numbers) estimates the same quantity by the inverse route.
   **Diagnostic (HYPOTHESIS, derived here):** if the InfoNCE bound on `I(a;f|s)` is ~0 nats the imagination is *action-blind* — the programme's measured longitudinal under-controllability
   of v1arch (INHERITED `HIERARCHICAL_WM_REDESIGN` §2.1) stated in information terms.
3. **Identifiability limit.** (B) is identified only for actions that occur in the data at that state. On expert-only logs the action varies little *within* a state, so an in-batch critic learns
   cross-state structure, not action effects. This is not a technicality — it is the toy's 47-72 regret (§5).
4. **Planning as inference.** For an outcome/goal `g`: `log p(a|s,g) = log pi(a|s) + h_B(s,a,g) + const(s,g)`. **HiCAP's score = [PMI-corrected action critic] + [outcome critic on a predicted goal]
   = log posterior over candidates given the goal** — control-as-inference (Levine, 1805.00909, `PUBLISHED`, tutorial; Attias 2003 `PUBLISHED-RECALLED`). Keep the two log-terms **additive with separate weights and a
   pre-registered zero-ablation each** (attribution rule; the `--v2` failure). Admissibility: `g` must be a predicted geometric goal and information-disjoint from the situation classifier (CLAUDE.md binding rule).
   Where the two critics **disagree** (expert prior vs outcome value) is exactly where imagination should matter — a candidate gate feature, `HYPOTHESIS` (not tested).

### 1.2 The published contrastive value-learning cluster — what transfers

| work (id, status) | what it gives | transfer to HiCAP |
|---|---|---|
| TD InfoNCE / Contrastive Difference Predictive Coding, Zheng, Salakhutdinov, Eysenbach, ICLR 2024, **2310.20141** `PUBLISHED` (snippet) | bootstrapped estimator of the discounted occupancy; stitches pieces of different series; tabular: ~20x more sample-efficient than successor representations, ~1500x than Monte-Carlo CPC (snippet, **tabular only, do not extrapolate**) | needed for horizons >= 3 s across 10-Hz clips (Monte-Carlo pairs never cross clips). **Bootstrap on the logged next action, never on the selector's own policy** — own-WM-scores-own-fan measured +0.21 (INHERITED) |
| C-learning, Eysenbach, Salakhutdinov, Levine, ICLR 2021, **2011.08909** `PUBLISHED` (snippet) | classify "is this observation from the future?"; Bayes rule turns the classifier into a future-state density; off-policy variant | the (B)/(C) critic *is* this classifier; gives the odds-to-density conversion for a calibrated score |
| Learning Temporal Distances, Myers, Zheng, Dragan, Levine, ICML 2024, **2406.17098** `PUBLISHED` | contrastive successor features (after a change of variables) define a temporal distance that **satisfies the triangle inequality even in stochastic settings**; stitching | label-free hierarchy consistency: `d(s,g) <= d(s,w) + d(w,g)` (§4.3) |
| Horizon Generalization, Myers, Ji, Eysenbach, ICLR 2025, **2501.02709** `PUBLISHED` | horizon generalisation <-> invariance to planning (same action towards a waypoint as towards the goal) | justifies training the tactical level on near goals and testing far — only if the metric structure holds |
| 1000-Layer Networks for Self-Supervised RL, **2503.14858** (F-R `CONFIRMED`) | depth 2x-50x in contrastive RL | critic heads may be depth-starved; `HYPOTHESIS`, cheap on cached features (one more layer = 1 MFLOP) |
| Temporal-Distance-JEPA, **2607.25337** `PUBLISHED` (snippet + README read, no numbers) | mines a directed temporal cost from reward-free logs (same-trajectory order = positive, cross-trajectory = heuristic negative, rollout-consistency term); uses it as planning cost on topology-dominated tasks but **latent L2 on contact-rich tasks** | no universal winner: keep latent-distance and temporal-cost heads side by side |
| Temporal Representation Alignment (Myers, Zheng, Dragan, Fang, Levine, NeurIPS 2025), Contrastive Representations for Temporal Reasoning **2508.13113** | titles/ids seen only; TRA arXiv id `UNVERIFIED` | not used |

None of these has been shown on **expert-only driving logs**; every result above assumes behaviour data with action diversity or a simulator. `HYPOTHESIS` that the (B)/(C) critic transfers.

### 1.3 Can one shared embedding space serve action selection **and** future-state prediction?

**Partially.** *Shareable:* the frozen VLM and the state tower `u` — a future frame is a state, so `u(o_{t+k})` lives on the same unit sphere and `<phi(s,a), u(f)>` is a cosine in the same 512-d space as
`<u(s), v(a)>` (JEPA precedent: V-JEPA 2 predicts in a frozen-encoder space). *Not shareable as one relation:* (A) is a bilinear compatibility that ranks actions given `s`; (B) is a transition `(s,a) -> f`.
The successor-feature form `phi(s,a) ~ E[sum_k gamma^k psi(s_{t+k})]` would let one critic `<phi(s,a), psi(g_E)>` with `g_E` = the *predicted expert future* play both roles, but that is imitation in outcome space and,
by 1.1-3, its ranking of off-manifold actions is unidentified. **Recommendation:** share `u(s)`; separate small `v(a)` and `phi(s,a) = normalise(s + R v(a))` towers (0.26 M MACs); one cosine each; test interference before
merging (risk R5). Toy evidence: none (the toy's selector and imagination are separate models). `HYPOTHESIS`.

---

## 2. Latent-space prediction heads on frozen foundation encoders

| system (id) | head / use | verified gain | compute / cost note | status |
|---|---|---|---|---|
| V-JEPA 2 / 2-AC, **2506.09985** | action-conditioned predictor post-trained on <62 h unlabeled Droid on a frozen encoder; MPC with CEM on a goal-conditioned latent energy | zero-shot pick-and-place on Franka arms in two labs, 65-80 % (snippet) | **16 s/action vs 4 min for Cosmos** (snippet) | `PUBLISHED` snippet |
| DINO-WM, **2411.04983** | dynamics on frozen DINOv2 **patch** features, predicts future patch embeddings, latent-MSE goal cost | outperforms baselines on 6 environments (snippet; the quoted 0.34 vs 0.18 is ambiguous, not used) | pooled vectors lose geometry (INHERITED lesson) | `PUBLISHED` snippet |
| PLDM, **2502.14819** | reconstruction-free JEPA latent dynamics for planning | across 23 datasets / 6 methods: best generalisation to unseen environments and tasks; model-free HILP/GCIQL better at stitching (snippet) | — | `PUBLISHED` snippet |
| Dreamer 4, **2509.24527** | RL inside a world model, shortcut forcing | diamonds in Minecraft from offline data with ~100x less data than VPT; real-time on one GPU; >25x faster generation (snippet) | — | `PUBLISHED` snippet |
| TD-MPC2, **2310.16828** | decoder-free latent, MPPI, latent-consistency + reward + value losses | one 317 M agent on 80 tasks; 104 tasks with one hyper-parameter set (snippet) | latent shaped by reward/value, not reconstruction | `PUBLISHED` snippet |
| LAW, **2406.08481** (ICLR 2025) | predicts future scene features from current features + planned trajectory; self-supervised | NAVSIM PDMS 84.6 perception-free (snippet) | auxiliary loss, no test-time cost | `PUBLISHED` snippet |
| World4Drive, **2507.00603** (ICCV 2025) | intention-driven future latents + a world-model selector | avg L2 0.61 -> 0.50 vs LAW, collision 0.30 -> 0.16 % (snippet) | annotation-free | `PUBLISHED` snippet |
| WoTE, **2504.01941** | per-candidate BEV world-model rollout + reward head on 256 anchors | 81.0 -> 83.2 -> 85.6 PDMS (F-R, V1-checked) | per-candidate rollout — the cost we are avoiding | INHERITED |
| DriveFuture, **2605.09701** | predicts the future latent from current latent + ego action, refines against the GT future latent in training, conditions a diffusion planner on the predicted future | 90.7 PDMS navtest v1, 89.9 EPDMS v2, 55.5 EPDMS navhard (snippet) | — | `PUBLISHED` snippet |
| ForeDrive, **2609.26299** | multi-horizon latent futures from a JEPA-style WM; **planning gradients update the encoder, the predictor gets forecasting losses only** (stop-grad routing) | 89.9 PDMS v1, 90.0 one-stage EPDMS v2, front camera only, no RL, no external scorer (snippet) | — | `PUBLISHED` snippet |
| DA-WAM, **2608.19085** | action-conditioned predictor gives **one future latent per candidate**, scored by a future-latent-conditioned factorised scorer; expert-matched candidate supervised by the observed future; safety-critical **hard negatives**; online encoder + momentum target | ablation over {no future, shared global future, current latent, action-conditioned future}; **numbers not read** | — | `PUBLISHED` snippet |
| Epona **2506.24113**, DriveVLA-W0 **2510.12796** | pixel-generating world models (trajectory + video; future-image prediction as dense supervision) | numbers not read | pixel decoding — outside our budget | `PUBLISHED` snippet |

**Reading (evidence-limited).** (i) All driving numbers are **NAVSIM PDM-scored replay**, not closed loop (INHERITED doctrine: T1/T2 tiers). (ii) Nearly every gain comes from *training-time* future prediction
(LAW, World4Drive, DriveVLA-W0) or from a *per-candidate* future feeding a scorer (WoTE, DA-WAM, DriveFuture); none reports FLOPs in the snippets, so compute is `UNVERIFIED`. (iii) **No head-to-head of MLP vs small transformer
on cached pooled embeddings was found** (`UNVERIFIED` absence at limited depth, 30 searches). Our toy needed only residual MLPs; the binding question is information, not capacity (DINO-WM lesson, LF0).
(iv) ForeDrive's stop-grad routing (predictor never receives the planning gradient) matches our guard against imagination that adapts to the selector.

**The 2026 decision-alignment cluster (all snippet level; none read in full).** *Decoding a variable is not ranking by it:* Decision-Metric Alignment / DA-LeWM (**2608.18746**: "information sufficiency vs decision-metric alignment",
Plan-Real Spearman as diagnostic), D-JEPA (**2609.24749**: a "decision-local prediction gap" — among the few futures competing for execution, the one predicted closest to the goal can execute worse), AD-WM (**2609.30264**: inverse-dynamics +
action-recovery head so the predictor retains action-dependent differences), Temporal-Distance-JEPA (**2607.25337**), **IMWM (2606.01626)**: a *contrastive scalar score over (start latent, goal latent, action chunk)* **complements** the
world-model rollout cost with retrieval initialisation, a hybrid cost and a **reliability gate** (+11.5 pp Two-Room, +28.5 pp OGBench-Cube over world-model-only; snippet) — HiCAP's selector plays the "intuition" role and the imagination
the world-model role, i.e. the roles are reversed relative to REF-F's first framing. RISE (**2608.20430**): a learned Roll/Stop gate weighing expected planning benefit against rollout cost, with a counterfactual risk dataset (CounterDrive);
"When to Trust Imagination" (**2605.06222**), "When and How Much to Imagine" (**2602.08236**), Biased Dreams (**2604.25416**: **ensemble disagreement captures local epistemic uncertainty but not compounding error; rollouts are drawn to
well-supported attractor regions where uncertainty falls while error grows** — the abstract-level match to H15; the snippet's "July 27, 2026" date conflicts with the 2604 id, date `UNVERIFIED`); ELVIS **2605.04709**, The Intervention Gap
**2608.29998** (titles only).

---

## 3. Imagination on cached embeddings — options (a)-(e)

**FLOP arithmetic** (ESTIMATED; `d = 512`, MLP width `W = 1024`, horizons `H = 4`, `K_i = 8` re-ranked, one MAC = 2 FLOP; prefill = `2·P·n_tokens` as in the sibling `hicap_cost_model.py`; no Thor timing exists —
`UNVERIFIED`). Smallest prefill used as denominator (conservative): 1 camera x 128 tokens + 64 text, `P = 2 B` -> `2·2e9·192 = 0.77 TFLOP`.

| option | FLOPs per tick (real scale) | what it can fix | what it cannot | how it re-creates echo / anti-calibration | guard |
|---|---|---|---|---|---|
| **(a) direct multi-horizon head**, action-free (exogenous): `s -> [H x 512]` | `512·1024 + 1024·2048 = 2.62 M MAC = 5.2 MFLOP`, once | supplies the lead/scene future the candidate cannot change -> longitudinal selection; identifiable from **logs alone** | effect of the ego action; anything `s` lacks (R1) | future embedding carries `v0` by construction (PBATTERY: 0.995 -> -0.72); toy: ego-speed R2 0.62 vs 0.61 under shuffled cue | report R2 on the **exogenous residual** with a shuffled-cue / shuffled-`v0` control; never the headline R2 |
| **(a') candidate-aware direct head** `[s;a] -> [H x 512]` | `1024·1024 + 1024·2048 = 3.15 M MAC = 6.3 MFLOP` per candidate; top-8 x 2 heads = **100.7 MFLOP**; all 4096: 25.8 GFLOP (3.4 % of prefill) | action-dependent outcomes | needs counterfactual futures (§5: without them 47-72 regret) | mean-latent + convex cost = biased risk (Jensen); confidence sign flips | counterfactual rows or refuse; analytic feasibility mask after; sign audit |
| **(b) one-step transition** `s' = g(s,a)` on top-K | `1024·1024 + 1024·512 = 1.57 M MAC = 3.1 MFLOP` per candidate-step; top-8 x 3 steps: **75.5 MFLOP**; all-K x 6 steps: **77.3 GFLOP (10 % of prefill)** | ego-consequence chains; multi-step effects | compounding error; SNR-0 (toy: 7.31 vs model-free 3.70) | "consistency is not plausibility": obediently simulates infeasible over-braking (§5) | top-K only; feasibility mask; cost head from real states |
| **(c) imagined-outcome critic** `cos(g(s,a), g_E(s))` against the *predicted expert future* | one (a) pass (5.2 MFLOP) + `8·2·512 = 8 kFLOP` | soft similarity kernel between candidates | by 1.1-2 it is **imitation in outcome space**; off-manifold ranking unidentified | reads `v0`/expert echo (target and candidate both computed from `s`) | use it as a **training-time soft-target kernel** (toy A4), not as an inference score (`HYPOTHESIS`; the cosine form itself was **not** tested) |
| **(d) 2-head disagreement** | second head: x2 of (a') (+50 MFLOP on top-8) | flags out-of-vocabulary candidates | does **not** track candidate-cost error (rho 0.03) nor benefit | Biased Dreams: agreement can rise while error grows | use as a **veto** on candidates, not as a benefit gate |
| **(e) learned "when to imagine" gate** (margin, entropy, disagreement, cost spread -> predicted benefit) | < 10 kFLOP | saves the top-K path on ticks where it hurts or is inert | needs paired episode-level validation | trained on the arm's own errors -> self-referential | train on held-out episodes; budget cap; oracle-gate upper bound reported |
| **(f) factorised imagination (this stream's addition)**: exogenous descriptors + mode posterior from `s`, ego motion + cost analytic | `~1.5 kFLOP` per candidate (toy accounting: 3 modes x 3 onset quantiles x 6 steps x ~14 = 0.76 k, plus ~0.2 k trajectory decode); 2,048 beam-surviving leaves (sibling cost model) ~ **3 MFLOP** (a 10x error margin still leaves < 0.05 % of prefill) | longitudinal/gap decisions with an *expected-cost* rule; robust to **no counterfactual data** | needs a metric readout of gap / closing speed from `s` (R1) | Jensen over hidden factors if only the mean is used (onset: +0.7 regret at SNR 3) | marginalise mode **and** timing; probe first (E-I0) |

**Versus the world-model roll it replaces:** a 91 M-parameter predictor step costs >= `2·91e6 = 182 MFLOP` per token-step against 3.1 MFLOP for the MLP transition, i.e. **>= 58x cheaper per candidate-step**
(params-only, one token; several tokens widen the gap). Weights are not the bottleneck either: 2 heads x 3.1 M params x 2 B = 12.6 MB per tick if re-streamed = 0.05 % of Thor's published 273 GB/s at 10 Hz.
**Batch-1 latency on Thor is launch- and bandwidth-bound, not FLOP-bound — no measurement exists (`UNVERIFIED`).**

---

## 4. Prediction structure for the hierarchy

**4.1 What each level predicts (targets are hindsight labels; labels may use ego and other agents, inference is vision-only).**

| level | horizon / clock | imagined object | label source | note |
|---|---|---|---|---|
| operative | 0.1-1 s, 10 Hz | ego motion **analytic** (unicycle, `kinematic.py` reused) + exogenous descriptors at 0.5/1 s | logged futures | the WM is a good ego integrator (INHERITED) — do not learn what is analytic |
| tactical | 1-5 s, 1 Hz + triggers | `f_k` for `k = 10..50` + **mode posterior and onset quantiles** of the lead / interacting agents + hindsight goal point / speed | logged futures, hindsight relabelling (every window supervises every `k`) | toy A5m: timing uncertainty matters (§5) |
| strategic | 5 s-minutes, 0.2 Hz | corridor / goal class at 6-15 s | hindsight ego heading; **no map in PhysicalAI** (INHERITED, settled at five probes) — no route-level outcome labels exist | do not claim strategic outcome quality without an external corpus |

**4.2 Level-wise objectives.** `L_l = InfoNCE(phi_l(s, a_l), sg u(f_{t+k_l}))` with the level's native action `a_l` (residual code / tactical token / goal corridor), per-state negatives where counterfactuals exist,
plus a small inverse-dynamics term `L_inv` (AD-WM-style) so the level's imagination stays action-discriminative. Hindsight relabelling doubles as augmentation (INHERITED design).

**4.3 Consistency between levels (parent teaches child; all with stop-grad on the parent).**
(i) *Outcome compatibility:* hinge `max(0, cos_min - cos(phi_child(s,a_c)@k_p, sg phi_parent(s,a_p)@k_p))` at the parent's horizon `k_p`. HWM's inference-time analogue: the first predicted latent of the high-level plan is the low-level
subgoal (**2604.03208** `PUBLISHED` snippet: 70 % vs 0 % on pick-and-place with V-JEPA 2-AC as single-level baseline, three backbones incl. PLDM and DINO-WM, "substantially reducing inference-time compute"; content not read).
(ii) *Triangle slack:* `max(0, d(s,g) - d(s,w) - d(w,g))` with a contrastive temporal distance (2406.17098) — label-free. (iii) *Veto, not score:* drop child candidates whose imagined outcome at `k_p` falls outside the parent's predicted outcome set.
**Information-disjointness:** the parent outcome must not be computed from the situation classifier's output (CLAUDE.md binding rule); state its inputs per arm.

**4.4 Published anchors.** Director (Hafner, Lee, Fischer, Abbeel, **2206.04114** `PUBLISHED`: manager picks discrete latent goals through a goal autoencoder, worker acts, planning inside latent space); HWM 2604.03208; FF-JEPA **2606.09311** (action-free latent
subgoal planner); Drive-HWM **2609.03572** (slow model predicts multi-step future representations with optical-flow "dynamic-aware latents"; fast model predicts next frame + one-step action; NAVSIM v1/v2, numbers not read); "Mind the Gap:
Promises and Pitfalls of Hierarchical Planning in LeWorldModel" **2607.12547** (title only — a caution flag, unread). H-JEPA / HRM / TRM: `PUBLISHED-RECALLED`, **not searched, ids and numbers withheld**.

**4.5 Amortised compute.** A tactical head at 1 Hz and a strategic head at 0.2 Hz cost 1/10 and 1/50 of an operative head per second; a 15-s strategic roll is 3 steps of a small model (INHERITED design), so hierarchy adds ~10 % of the
operative imagination FLOPs (ESTIMATED arithmetic).

---

## 5. The toy — `rc_latent_imagination_toy.py` (numpy, 86-103 s wall on a shared 4-core box; deterministic: max regret delta 0 between two runs)

**World** (design in the script docstring): 1-D longitudinal driving; lead mode `z in {steady, brake 3.0, brake 6.5 m/s^2}` with hidden onset; observation = visible state + a noisy 'brake-light' cue x SNR, passed through a fixed random tanh
encoder to a unit 32-d `s`; K = 32 candidates; exact simulator cost; the expert uses `z` (privileged label). **Metric = regret** on the true simulator (heavy-tailed: collision = 40/step). 3 seeds; 900 test ticks each; interval estimator =
**paired bootstrap over pooled test ticks** (i.i.d. by construction; no episode clusters exist in the toy).
**Forking paths, disclosed:** the world was re-calibrated twice before any arm ran (first version: `z` did not change the best action; criteria = z-blind regret materially > 0, oracle 0). Only A0, A1, A2 and A4 (plus the fixed gates, H1-H5 and the echo control) were in the original design. After the first sweep showed A0 worse than a trivial z-blind lookup and A2's win
confounded with risk-sensitivity, **A0s, QM, A1s, REF, A1c/A2c, the A5 family, the SNR-3 config and the learned/oracle gates were added** — all of those are post-hoc.

### 5.1 Main table — regret (mean ± sd over 3 seeds; lower is better; ORACLE = 0)

| arm (inference-time inputs = `s` only) | n=300 | n=1200 | n=4800 | SNR 0 (n=1200) | SNR 3 (n=1200) | expert-only rows (n=1200) | toy MACs / tick |
|---|---|---|---|---|---|---|---|
| **A0** myopic, hard one-hot label (CLM-style) | 9.88 ± 0.94 | 7.00 ± 1.92 | 6.53 ± 0.79 | 10.50 ± 3.11 | 3.72 ± 1.01 | 7.27 ± 1.50 | 2,816 (1x) |
| **A0s** soft targets from TRUE per-candidate cost (privileged teacher) | 4.92 ± 1.26 | 3.43 ± 0.47 | 3.25 ± 0.28 | 4.24 ± 0.59 | 2.03 ± 0.17 | 3.43 ± 0.47 | 2,816 |
| **QM** model-free Q head, same rows | 3.90 ± 1.79 | 2.97 ± 0.27 | 2.99 ± 0.83 | 3.70 ± 0.23 | 2.69 ± 0.45 | **47.04 ± 5.50** | 151,552 (54x) |
| **A1** A0 + direct head on top-4 | 4.75 ± 0.26 | 3.79 ± 0.92 | 3.49 ± 0.47 | 6.99 ± 2.48 | 1.87 ± 0.31 | 7.27 (beta=0 chosen) | 176,000 (62x) |
| **A1s** the same on top of A0s | 3.31 ± 0.84 | 2.28 ± 0.42 | 2.14 ± 0.15 | 3.37 ± 0.36 | **1.42 ± 0.16** | 3.43 (beta=0) | 176,000 |
| **A2** iterated rollouts, all K | 2.90 ± 0.31 | 2.17 ± 0.06 | 2.90 ± 1.55 | **7.31 ± 2.78** | 1.74 ± 0.08 | 7.27 (beta=0) | 2,208,512 (784x) |
| **A4** distilled from the learned imagination | 3.10 ± 0.24 | 2.89 ± 0.13 | 2.91 ± 0.14 | 5.09 ± 0.96 | 2.68 ± 0.14 | **48.23 ± 3.27** | 2,816 (1x) |
| **A5m** factorised (mode + onset quantiles, analytic) | 2.74 ± 0.25 | **1.81 ± 0.06** | **1.91 ± 0.04** | 2.17 ± 0.08 | 3.00 ± 0.40 | **1.80 ± 0.09** | 28,672 (10x) |
| DIAG: A5m with the **true** visible state | 1.61 ± 0.14 | 1.37 ± 0.10 | 1.36 ± 0.10 | 1.85 ± 0.09 | **0.46 ± 0.03** | 1.37 ± 0.11 | — |
| **REF** z-blind risk-sensitive lookup on the true visible state | 3.86 ± 1.01 | 2.17 ± 0.06 | 2.04 ± 0.07 | 2.17 ± 0.06 | 2.17 ± 0.06 | 2.17 ± 0.06 | — |

Unsafe rate (selected collides while a collision-free best exists), n=1200: A0 2.9 %, A1 1.1 %, A1s 0.3 %, A2 0.2 %, QM 0.7 %, A4 0.0 %, A5m 0.15 %, REF 0.26 %. Expert-only rows: A2 pure-imagination 71.83 (40.1 %), QM 19.2 %, A4 27.4 %, A0 2.9 %, A5m 0.1 %.
Mode accuracy of the A5 posterior: 0.53-0.57 at SNR 0 (prior 0.55), 0.63-0.67 at SNR 1, 0.90-0.92 at SNR 3.
Recall bound (oracle re-rank of the proposer's top-4): A0 3.84 / 3.10 / 2.74 (n = 300 / 1200 / 4800), A0s 2.40 / 1.47 / 1.27 — a re-ranker cannot recover candidates the proposer never surfaced.

### 5.2 Paired deltas that carry the argument (mean [95 % CI], regret units; positive = first arm worse)

| contrast | n=300 | n=1200 | n=4800 | SNR 0 | SNR 3 |
|---|---|---|---|---|---|
| A0 - A0s (label-side only) | 4.96 [4.16, 5.77] | 3.57 [2.99, 4.19] | 3.28 [2.69, 3.86] | 6.25 [5.39, 7.23] | 1.69 [1.35, 2.04] |
| A0 - A1 | 5.14 [4.45, 5.85] | 3.21 [2.75, 3.72] | 3.04 [2.54, 3.50] | 3.51 [2.84, 4.24] | 1.84 [1.52, 2.16] |
| A0 - A2 | 6.98 [6.08, 7.84] | 4.83 [4.10, 5.59] | 3.63 [3.00, 4.29] | 3.19 [2.25, 4.19] | 1.98 [1.52, 2.42] |
| QM - A2 (latent rollouts vs model-free Q, same rows) | 1.00 [0.64, 1.35] | 0.79 [0.55, 1.03] | 0.09 [-0.27, 0.46] | **-3.62 [-4.40, -2.88]** | 0.96 [0.75, 1.16] |
| A0s - A1s (imagination on a strong proposer) | 1.62 [1.26, 1.98] | 1.14 [0.87, 1.43] | 1.11 [0.84, 1.41] | 0.87 [0.54, 1.24] | 0.61 [0.42, 0.81] |
| A0s - A4 (distilled vs true-cost teacher) | 1.83 [1.25, 2.42] | 0.53 [0.08, 0.98] | 0.35 [-0.04, 0.77] | -0.85 [-1.43, -0.22] | -0.66 [-0.95, -0.35] |
| QM - A5m | 1.16 [0.76, 1.59] | 1.16 [0.95, 1.37] | 1.08 [0.84, 1.35] | 1.52 [1.25, 1.80] | -0.31 [-0.70, 0.08] |

### 5.3 Pre-registered readings vs outcomes

| reading | verdict | measured evidence |
|---|---|---|
| **H1** A1/A2 beat A0 at small n, gap shrinks with n | **supported as worded, but confounded** | A0-A1 5.14 -> 3.21 -> 3.04; A0-A2 6.98 -> 4.83 -> 3.63 (CIs of n=300 vs n=1200 disjoint). The **same** gap appears with zero-inference-cost label arms (A0-A0s 4.96 / 3.57 / 3.28; A0-A4 6.79 / 4.10 / 3.62) => it is a **target-shape** effect. The latent-rollout increment over model-free Q is 0.8-1.0 at n <= 1200 and **0.09 [-0.27, 0.46] at n=4800**: a low-data effect |
| **H2** at SNR 0 no arm beats A0 | **refuted as worded** | at SNR 0 A1, A2, QM, A0s and A5m have lower regret than A0 by 3.5, 3.2, 6.8, 6.3 and 8.3 — with **no information** in `s` (A5 posterior = prior). Correct control is REF (2.17): nothing learned beats it (A5m 2.17), and **A2 is far worse (7.31)** — rollouts hurt when the future is unpredictable from `s` |
| **H3** expert-only training hallucinates benign futures for out-of-envelope candidates; counterfactual rows + feasibility mask fix it | **supported** | expert-only rows: A2 pure 71.8 (40 % unsafe), QM 47.0, A4 48.2; imagination-only picks the hard-accel candidate (4,4) in **49.8 %** of ticks. With counterfactual rows imagination-only picks an infeasible **over-braking** candidate in **89.8 %** (56.3 % / 33.6 %); analytic mask -> 0 %. Imagined vs obedient-simulator cost of the over-braking types: -2.5 / -0.6 vs 9.5 / 15.0 |
| **H4** need x trust gate keeps >= 80 % of A1's gain at <= 30 % budget; anti-gate worse | **refuted** | share of A1's gain at 30 % budget: random 35 %, margin 39 %, trust-low-disagreement **30 %** (worse than random), need x trust **35 %**, anti (imagine where heads disagree most) 32 %; **learned benefit gate 65 %** (31 / 52 / 65 / 84 % at 10 / 20 / 30 / 50 %); oracle gate 97-106 % (>100 % because imagination hurts on 13.4 % of ticks, helps on 18.6 %) |
| **H5** disagreement tracks error for direct heads; decouples for rollouts at long horizon | **refuted as worded** | cost-level Spearman(disagreement, abs(Jhat - J)) = 0.027 (per seed 0.077 / 0.011 / -0.009). State level: direct 0.17 / 0.17 / 0.22 at 1 / 2 / 3 s; **rollout 0.11 -> 0.33 over 6 steps (grows)**. Mean state error is flat (0.815 -> 0.833) because of an irreducible cue-noise floor, so the toy **cannot show compounding** — it is not evidence against Biased Dreams |

### 5.4 Other measured facts

- **Confidence sign flip:** Spearman(top-1 probability, regret): A0 **-0.113** (-0.126 / -0.052 / -0.161), A1 **+0.190** (+0.182 / +0.270 / +0.118). Mechanism `HYPOTHESIS`.
- **Out-of-vocabulary candidates, disagreement as veto:** heads disagree 20.8 on WILD vs 4.7 on vocabulary candidates (AUROC **0.877**; recall **63.8 %** at 5 % vocabulary FPR; by type 0.57 / 0.51 / 0.65 / 0.84). Expert-only rows: AUROC 0.71, recall 36.5 %.
  A0 selects a WILD candidate in 24.1 % of ticks without the mask (A1 31.0 %); the mask -> 0 %. A5 with the envelope penalty off picks 7.6 %.
- **Echo split:** R2 of the direct head's t=6 prediction, real cue vs shuffled cue: ego speed 0.620 / 0.613 (invariant = action-determined), gap 0.644 / 0.350, lead speed **0.387 / -0.329**. The headline mixture (0.55 vs 0.21) would mask that only the second and third are decision-relevant.
- **Jensen guard refuted:** refitting the cost head on imagined latents made rollouts worse (A2 - A2c = -1.79 [-2.19, -1.39] at n=1200) and the top-4 head slightly worse (-0.21 [-0.32, -0.11]). The SNR-0 rollout failure mechanism remains `HYPOTHESIS` (latent blur, mean-collapse).
- **Factorised arm:** onset marginalisation helped (A5 - A5m = 0.70 [0.54, 0.87] at SNR 3; ~0.15 at SNR 1); temperature scaling of the mode posterior did nothing (NLL selects T = 1 in 16 of 18 runs, 1.5 in two). With the true visible state A5m reaches 0.46 (SNR 3) — the
  **3.00 vs 0.46 gap is entirely visible-state readout error amplified by a steep cost**. That is why A5m loses to A1s / A2 / REF at SNR 3 and wins everywhere else.
- **Validation matters:** with expert-only rows the validation-selected beta = 0 for A1/A2 (they collapse onto A0) — correct *only because the validation ticks carry counterfactual labels*, which real logs do not.

### 5.5 Limits — read before quoting anything above

1-D longitudinal world; fixed random encoder; exact simulator as the counterfactual source (a real system supplies counterfactual **labels/descriptors** from rules or non-reactive replay, **not counterfactual frames** unless AlpaSim/NuRec is in the loop — so A1/A2/A4-style
embedding imagination is *optimistic* here, and QM / A0s-style label supervision is the realistic analogue); A5 is given the analytic simulator and the mode -> trajectory law (the "structure prior" is the point, but it is knowledge); regret is dominated by a collision tail, so means are noisy (A2 at n=4800:
1.89 / 2.13 / 4.68); 3 seeds; small towers (48 hidden units) — capacity was checked only for A0, in an exploratory run not in the shipped JSON (128 units / 4000 steps: 5.4-5.7 vs 6.5-7.0 at 48 units, still >> 2.2); `s` is a scrambled embedding of noisy observations, so there is a prediction floor; no result transfers to VLM scale, real cameras or closed loop.

---

## 6. Recommendation — "HiCAP-I" imagination for HiCAP

```
frames + ego (+audio) -> frozen VLM -> u(s) [512, unit]  ------------------------------------------.
                                          |                                                        |
  TIER 0  (training only, 0 inference FLOPs)                                                       |
     soft listwise targets pi_T ∝ exp(-J_teacher/T) mixed 0.7/0.3 with the expert one-hot;         |
     per-family teacher heads (safety, comfort, progress, lateral) [labels use ego/others: allowed] |
  TIER 1  (every tick, factorised, analytic)                                                       |
     E-head: u(s) -> exogenous descriptors x {0.5,1,2,3 s} (lead gap, closing speed, lead accel)   |
             + mode posterior (3-5 modes) + onset quantiles (2)      [~20-24 numbers, not 2048]     |
     candidate ego motion: unicycle integration (kinematic.py verbatim); cost = E_{mode,onset} J   |
  TIER 2  (only where counterfactual embeddings exist; gated, top-K_i = 8)                         |
     phi(s,a) = normalise(s + R v(a)); two heads; disagreement VETO; learned benefit gate           |
  score = tau<u(s),v(a)> + log-priors  -  beta1·J_tier1  -  beta2·J_tier2   (each with a zero-ablation)
```

**6.1 Dimensions.** `d = 512`. E-head MLP `512 -> 1024 -> 24` (24 output numbers instead of 4 x 512 = 2048: descriptors, not embeddings, because counterfactual *frames* are not available). Tier-2 `phi`: `1024 -> 1024 -> 4 x 512`, two heads. Action tower `v(a)`: cached (`K x 512 x 2 B`; 4.2 MB at K = 4096; beam retrieval touches ~2.4 k nodes, sibling cost model).
Gate: 7 features (margin, entropy, mean disagreement, cost spread, top logit, two logs) -> ridge/MLP.

**6.2 Losses.** `L_sel` soft listwise CE over the candidate set (+ per-family heads) · `L_out = sum_k InfoNCE(phi(s,a), sg u(f_{t+k}))`, `k in {5,10,20,40}` at 10 Hz, per-state negatives where counterfactuals exist, in-batch otherwise, learnable `tau_out` ·
`L_inv` inverse-dynamics action recovery from `(s, f)`, weight <= 0.1 (AD-WM-style, keeps `I(a;f|s) > 0`) · `L_exo` Huber on descriptors + CE on mode + quantile loss on onset · `L_cons` hinge on parent-child outcome cosine + triangle slack (§4.3).
**No gradient from any imagination head into the frozen VLM; imagination parameters are separate from the selector's and trained/frozen independently** (own-WM-scores-own-fan = +0.21, INHERITED; ForeDrive's stop-grad routing).

**6.3 When it runs.** Tier 0: training only. Tier 1: every tick on every beam-surviving candidate (~3 MFLOP). Tier 2: only when the learned benefit gate fires (target <= 30 % of ticks) **and** the candidate passes the disagreement veto; skipped entirely for any candidate family without counterfactual supervision.
Hierarchy clocks: strategic 0.2 Hz, tactical 1 Hz + event triggers (TTC drop, goal invalidation), operative 10 Hz.

**6.4 FLOPs per tick (ESTIMATED, real scale; prefill denominator 0.77 TFLOP).**

| item | FLOPs | share of smallest prefill |
|---|---|---|
| score matmul (K = 4096 x 512) | 4.2 MFLOP | 0.0005 % |
| Tier 1 E-head (once) | `2·(512·1024 + 1024·24) = 1.1 MFLOP` | 0.0001 % |
| Tier 1 analytic evaluation, 2,048 leaves x ~1.5 kFLOP | ~3 MFLOP | 0.0004 % |
| Tier 2, top-8 x 2 heads (only on <= 30 % of ticks) | 100.7 MFLOP -> ~30 MFLOP amortised | 0.004 % |
| gate | < 10 kFLOP | ~0 |
| **total imagination overhead** | **~38 MFLOP amortised (Tier 2 dominates)** | **~0.005 %** |
| *for contrast:* Tier 2 on all 4096 candidates / iterated rollouts | 25.8 / 77.3 GFLOP | 3.4 % / 10 % — **not admitted** |

**6.5 Guards (each one is a measured failure above).**
G1 analytic feasibility + comfort mask applied **after** every learned term (imagination-only picks infeasible over-braking 89.8 %). G2 **counterfactual-coverage audit**: no counterfactual supervision for a candidate family => Tier 2 refuses it (47-72 regret otherwise); Tier 1 needs only logs.
G3 shuffled-cue / shuffled-`v0` control and R2 reported on the **exogenous residual** only (echo split). G4 disagreement = **veto** on out-of-vocabulary candidates (AUROC 0.88), never a benefit gate. G5 **confidence-sign audit**: Spearman(confidence, regret) must remain negative after adding any imagination term; do not expose the combined softmax as confidence.
G6 select `beta` on validation that carries **counterfactual labels** (else it silently returns beta = 0 and hides the failure). G7 Tier 1 marginalises **every** hidden factor of the consequence (mode **and** timing); mean-plugging costs +0.7 regret at SNR 3. G8 every term carries a zero-weight ablation and a **paired, episode-cluster** estimate — a single scalar composite would hide the trade-off.
G9 goal path and situation path stay information-disjoint; the E-head reads `s` only and never the situation classifier's output.

**6.6 Ranked risks and the cheapest discriminating experiment (both outcomes committed in advance).**

| # | risk | experiment | outcome A | outcome B |
|---|---|---|---|---|
| R1 | **the embedding does not carry metric gap / closing speed** (LF0 precedent) -> Tier 1 inert; toy: readout error is the whole 3.00 vs 0.46 gap | **E-I0 (~0.1 GPU-h + cached VLM embeddings):** ridge and 2-layer MLP probes `s -> {lead gap now, closing speed, gap at +2 s, time-to-brake}` on val40, with a `v0`-only baseline and shuffled-cue control | R2(gap) >= 0.3 above the `v0`-only baseline with a cluster-bootstrap CI excluding 0 => fund Tier 1 | R2 < 0.1 => Tier 1 is dead for this encoder; ship Tier 0 only and reopen the encoder question (patch tokens, DINO-WM lesson) |
| R2 | hard-argmin / one-hot InfoNCE target is the defect (toy: 3.6-6.3 regret) | **E-I1:** one-hot vs soft per-candidate multi-family targets, same data and towers; paired selected-ADE and four families on val40 | soft lowers the selection gap >= 15 % (paired CI excludes 0) => Tier 0 is the first spend (matches registry §4.1, INHERITED) | no difference => the defect does not appear at VLM scale; drop Tier 0 |
| R3 | counterfactual coverage missing for non-expert candidates (toy 47-72) | **E-I2:** Tier-2 head trained on expert-only vs expert + rule-teacher-labelled counterfactuals; pure-imagination unsafe rate on the vocabulary | unsafe > 5 % with expert-only => G2 is binding and Tier 2 waits for AlpaSim/NuRec labels | unsafe <= 1 % => the toy's collapse is a toy artefact; relax G2 |
| R4 | adding imagination flips confidence sign (toy -0.11 -> +0.19) | **E-I3:** Spearman(confidence, regret) on val40 with / without each term | sign flips positive => G5 binding; add a validation calibrator | stays negative => no action |
| R5 | shared state tower degrades selection | **E-I4:** matched-step selector-only vs joint, `lambda_out in {0, 0.1, 0.5}` (matched-step ratio, no exponents) | selector within CI => share `u(s)` | degrades beyond CI => separate towers |
| R6 | echo: `v0`/ego in `s` makes outcome prediction look good (toy ego-speed R2 invariant to cue) | **E-I5:** PBATTERY-style shuffled-`v0` and shuffled-cue controls on the E-head and Tier-2 head | exogenous R2 collapses under shuffle => real signal | does not collapse => echo; the head is inadmissible |
| R7 | the learned gate overfits (400 validation ticks in the toy) | **E-I6:** train on val40-A episodes, test on held-out episodes, paired cluster bootstrap | >= 50 % of A1's gain at 30 % budget => keep | < 50 % => always-on top-K or drop Tier 2 |

---

## 7. Implications for HiCAP (<= 12)

1. **Fix the target before adding imagination.** One-hot InfoNCE positives were the worst target in the registry (INHERITED) and the largest toy lever (3.6 regret at zero inference cost); make Tier 0 (outcome-aware soft listwise targets, per-family teachers) the first spend.
2. **Never report an imagination gain without the risk-sensitive floor** (REF) and the model-free Q on the same rows; otherwise risk-sensitivity is credited to imagination.
3. **Imagine the exogenous part, integrate the ego part analytically.** It is identifiable from logs, ~10x cheaper than unstructured imagination, and the only arm that survives the absence of counterfactual data.
4. **Run E-I0 first (0.1 GPU-h).** If `s` cannot read gap / closing speed, no predictor on `s` can help; the 3.00 -> 0.46 toy gap is exactly this.
5. **Unstructured action-conditioned latent imagination needs counterfactual futures.** Without them it is anti-conservative (47-72 regret, 19-40 % unsafe) — refuse those candidate families (G2).
6. **Imagination FLOPs on cached embeddings are ~0.01 % of prefill for top-8; iterated all-K rollouts are ~10 %.** Efficiency is not the constraint; keep imagination on the top-K and never call the large WM per candidate.
7. **Disagreement is a veto, not a gate.** It flags out-of-vocabulary candidates (AUROC 0.88) but is random as a benefit trigger; a learned benefit gate is the working "when to imagine".
8. **Imagination hurts on ~13 % of ticks** even when it helps on average — the gate is a safety device before it is a compute device.
9. **Audit confidence sign** whenever a term is added: the toy flips it positive (anti-calibration), matching H15.
10. **Marginalise every hidden factor of a consequence** (mode and timing); a sharp mode posterior with a mean timing is over-confident (+0.7 regret at SNR 3).
11. **Share the state tower, not the objective.** Action selection (PMI over actions) and outcome prediction (dynamics ratio) are different relations; their disagreement is a candidate gate feature (`HYPOTHESIS`).
12. **Hierarchy: parent outcomes teach child outcomes with stop-grad; consistency is a veto and a triangle-slack term**, with strategic outcomes unlabelable without a map (settled INHERITED fact) — do not claim strategic outcome quality.

---

## 8. Access log, citation status, manifest

**Access log (fail-loud).** 30 of 30 WebSearch calls used; 4 WebFetch calls (github.com only): AD-WM README, D-JEPA README, Temporal-Distance-JEPA README (all three: **no numbers in the pages**), and one alphaxiv attempt that was **EGRESS_BLOCKED**. `origin/handoff` docs read via `git show`: `MPC_WM_DESIGN`, `DIFFUSION_MPC_SYNTHESIS`, `JEPA_PHYSICS_SURVEY`, `HIERARCHICAL_WM_REDESIGN`, `PBATTERY_V6_FIRST_RESULT`, first 140 lines of `DIFFUSION_PLANNER_COMPARISON`; local `2026-09-29-reff-prior-art-and-theory.md` (F-R) sections 2-3 and 5.3; sibling `hicap_cost_model.py` header. **Not opened:** `MODEL_REGISTRY.md` (no pod; programme numbers are INHERITED), the CLM blog, any arXiv PDF.

**Citation status** (`CONFIRMED` = id and title appeared in a result list this session; content = snippet only unless stated).

| item | id | status |
|---|---|---|
| TD InfoNCE (Contrastive Difference Predictive Coding) | 2310.20141 | CONFIRMED; ICLR 2024; tabular claims are snippet |
| C-Learning | 2011.08909 | CONFIRMED; ICLR 2021 |
| Learning Temporal Distances | 2406.17098 | CONFIRMED; ICML 2024 |
| Horizon Generalization in RL | 2501.02709 | CONFIRMED; ICLR 2025 |
| Temporal Representation Alignment (NeurIPS 2025); Contrastive Representations for Temporal Reasoning | id `UNVERIFIED`; 2508.13113 title only | titles only |
| Contrastive RL; 1000-layer RL; CPC; Poole et al. | 2206.07568; 2503.14858; 1807.03748; 1905.06922 | inherited-CONFIRMED (F-R); statements RECALLED |
| V-JEPA 2 / 2-AC; DINO-WM; PLDM | 2506.09985; 2411.04983; 2502.14819 | CONFIRMED (snippet) |
| Dreamer 4; TD-MPC2; Director | 2509.24527; 2310.16828; 2206.04114 | CONFIRMED (snippet) |
| LAW; World4Drive; Epona; DriveVLA-W0 | 2406.08481; 2507.00603; 2506.24113; 2510.12796 | CONFIRMED (snippet; Epona/W0 numbers not read) |
| WoTE | 2504.01941 | INHERITED (F-R, V1-checked) |
| DriveFuture; ForeDrive; DA-WAM; Drive-HWM | 2605.09701; 2609.26299; 2608.19085; 2609.03572 | CONFIRMED (snippet); DA-WAM ablation numbers **not read** |
| HWM; FF-JEPA; Temporal-Distance-JEPA; IMWM | 2604.03208; 2606.09311; 2607.25337; 2606.01626 | CONFIRMED (snippet) |
| AD-WM; D-JEPA; Decision-Metric Alignment (DA-LeWM) | 2609.30264; 2609.24749; 2608.18746 | CONFIRMED (snippet); AD-WM / D-JEPA READMEs read, no numbers |
| RISE; When to Trust Imagination; When and How Much to Imagine | 2608.20430; 2605.06222; 2602.08236 | CONFIRMED (snippet) |
| Biased Dreams | 2604.25416 | CONFIRMED (snippet); **date conflict (id 2604 vs "July 27, 2026") UNVERIFIED** |
| ELVIS; Intervention Gap; Mind the Gap (hierarchical LeWM); Traj-LeWM; Latent-WAM | 2605.04709; 2608.29998; 2607.12547; 2608.14125; 2603.24581 | titles only, unread |
| Control as inference (Levine) | 1805.00909 | CONFIRMED; Attias 2003 RECALLED, no id |
| H-JEPA, HRM, TRM | — | RECALLED, not searched, no ids or numbers used |

**Deliverable manifest** (all staged; `git ls-files --stage` verified)

| artifact | where it lives | single-copy? |
|---|---|---|
| `RC_imagination.md` (this report) | repo:`TanitAD Research Hub/Architecture & Inference/Research/2026-09-29-hicap/RC_imagination.md` | repo only |
| `rc_latent_imagination_toy.py` | repo:`.../2026-09-29-hicap/rc_latent_imagination_toy.py` | repo only |
| `rc_latent_imagination_toy_result.json` (per-seed + extras + paired CIs, ~150 KB) | repo:`.../2026-09-29-hicap/rc_latent_imagination_toy_result.json` | repo only |

Integration needed: none to merge. **Decision escalated to the PI in the headline:** fund E-I0 (0.1 GPU-h) before any imagination work; it gates Tier 1.
