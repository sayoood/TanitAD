# Planner + hierarchy SOTA — and the one measurement that re-opens a closed lever

**Date:** 2026-08-04 (Europe/Berlin) · **Stream:** arch-inf · **GPU cost: 0.** Dev-box CPU,
banked artifacts only. No pod touched, no training launched, nothing pushed.

---

## 0. LEAD — the ranked shortlist, and the single cheapest discriminating experiment

⭐ **The finding.** Every selection lever this programme has killed — including
`S1_IS_DEAD_AFTER_ALL` yesterday — is **a re-ranking of a FIXED fan**. K1 bounds that at
**≤ 8.4 %** of the 0.3075 m oracle gap, in-sample, not separated. **That bound does not apply to
operators that leave the fan.** Three independent 2026 papers refine *off* the candidate set, and
all three factorise the refinement into an along-path and a cross-path axis.

**So I ran it, today, on banked fans at 0 GPU** (E-EXP-1, pre-registered and content-pinned
before any statistic). Applying HAD's published 5-point radial grid to the **already-selected**
trajectory has a ceiling of:

| | `refc-base-30k` (128) | `refc-xl-30k` (256) |
|---|---|---|
| shipped `ade_sel` | 0.4728 | 0.4714 |
| **ceiling with a per-window longitudinal scale λ** | **0.3138** | **0.3064** |
| paired Δ (`sel − sel+λ`) | **+0.1590 [+0.1196, +0.2006]** ✅ separated | **+0.1651 [+0.1244, +0.2049]** ✅ separated |
| the **matched-DoF lateral control** (5 angular offsets) | +0.0161 [+0.0080, +0.0255] | +0.0141 [+0.0069, +0.0230] |
| longitudinal ÷ lateral | **9.9×** | **11.7×** |
| share of the arm's own oracle gap | **56.5 %** | **53.7 %** (gap = 0.3075) |

**Ceiling-to-ceiling, against K1's in-sample re-ranking bound of ≤ 8.4 %, this is 6.4×.**

⛔ **And the equally important negative, from the same run.** A **single global λ**, fitted
leave-one-episode-out, recovers **nothing** (+0.0010 [−0.0014, +0.0036] base; −0.0011 XL, neither
separated) and lands at λ = 0.999–1.001. **There is no systematic speed-scale bias to calibrate
away** — the correction is genuinely per-window. And in the post-hoc follow-up, a **`v0` decile
lookup for λ is separably WORSE than doing nothing** on both arms (−0.0039 [−0.0097, −0.0006];
−0.0021 [−0.0045, −0.0006]).

⇒ **The reachability question is answered PASS-L. The findability question is open, and the two
cheapest feature sets both failed.** That is exactly what makes the #1 item a probe, not a build.

### The ranked shortlist — ranked by expected effect on the LONGITUDINAL family

| # | item | cost | why here |
|---|---|---|---|
| **1 ⭐** | **E-EXP-2 — probe λ\* from REF-C's latents** against a shuffled-latent control **and** the `v0`-echo control below | **1 feature dump, 0 training** | The *only* thing that decides whether item 2 is fundable. Everything else waits on it. |
| **2** | **AlignDrive-style lateral→longitudinal cascade**: predict the path, then a **1-D displacement along it** | ~+1.3 k params for a 5-way λ head | The published version of un-mixing our 5-way lat+lon softmax. **Conditional on #1.** |
| **3** | **Mimir-style Laplace uncertainty gate** on any goal/λ signal | ~+2 params per output | Cheapest published guard against a wrong high-level signal poisoning the low level. |
| **4** | **HAD-style select-then-locally-expand** (K=2, not flat K=20) | 0 new params at inference | Structural form for #2; HAD's own ablation is the warning against a flat fan. |
| **5** | TOAD / DriveVer test-time refinement | 34 M params (DriveVer) or a CEM loop | Same family as #2 but 26× the parameter cost and gains that shrink on stronger planners. |
| **6** | Schedule advances (BridgeDrive, DiffusionDriveV2, MeanFuser, MISTY) | varies | **Latency and proposal quality, not longitudinal.** Correctly ranked last *by this criterion*. |
| ⛔ | GuideFlow constraint guidance | — | **Blocked**: requires BEV map tokens + drivable-area. We have no map. |

### ⭐⭐ The measurement that reframes the goal-point question (P2)

`MEASURED (ours, raw/e_exp1_axis_reach.json, frame_check)`: on our 881 val windows the **2 s
along-track endpoint correlates with `v0` at r = 0.9973**.

⇒ **A predicted geometric goal point on PhysicalAI would be ~a `v0` echo.** GoalFlow's +2.9/+4.7
PDMS and Mimir's +2.2 EPDMS are measured where a goal carries map and route information we do not
have; on our corpus its dominant component is already an input the planner receives. This is the
same defect class as flagship-v1's route head being an exact nav bijection — and it **explains K5**
(oracle route: +0.0024, not separated) rather than leaving it an anomaly.

⇒ **And it explains why E-EXP-1b failed in the useful direction:** the arms already consume the
`v0`-explainable part of the longitudinal signal. **λ\* is the residual that `v0` cannot explain** —
which is precisely why it is worth probing, and precisely why a goal point is not.

---

## 1. P1 — diffusion-planner advances for a 2-step truncated decoder over 256 anchors

⚠️ **Protocol scepticism, applying to every row below.** **Not one of these papers reports a
confidence interval, a standard deviation, or a significance test on its headline** — the single
exception is AlignDrive, which reports **3 independent runs** (89.07 / 87.80 / 88.05 DS). NAVSIM is
**non-reactive**; PDMS deltas are not comparable in kind to our paired episode-cluster bootstrap.
No number here is a target we "beat".

| paper | what it changes | headline | survives our constraints? |
|---|---|---|---|
| **BridgeDrive** (arXiv 2509.23589) | Names the defect in our own decoder: the truncated schedule *"introduces an asymmetry between the forward and denoising processes, diverging from the core principles of diffusion models."* Replaces it with a **diffusion bridge** anchor→plan, ODE-solver compatible | +7.72 % over PDM-Lite on Bench2Drive | ⚠️ **Diagnosis yes, magnitudes no.** The abstract page gives no step count, no latency, no ablation. Our defect is **selection**, not schedule fidelity. |
| **DiffusionDriveV2** (arXiv 2512.07745) | RL (intra-/inter-anchor GRPO) on the **proposal** step; collisions get −1 | NAVSIM v1 **91.2 vs 88.1** PDMS; **Ego Progress 87.5 vs 82.2** | ⚠️ Needs a **rasterised BEV/LiDAR** input. **But `+5.3 Ego Progress` is the largest published longitudinal-family move in this sweep** — worth watching. Still **2 truncated steps**, same as ours. |
| **GuideFlow** (arXiv 2511.18729) | Constraint guidance at inference: CVF (gradient), **CF (replace the flow state at step 50/100 with a constraint-adhering anchor)**, RFE (EBM) | Navhard EPDMS 43.0 vs 42.1. Ablation: **CF alone +4.7**, RFE alone **+5.3** Stage-2 — but RFE alone **HURTS Stage-1, 56.7 → 53.3** | ⛔ **Blocked.** Requires BEV + **map tokens** and a drivable-area notion. Also: its "aggressiveness score" is distance along a **lane centerline** — no lane graph, no such feature. |
| **HAD** (arXiv 2604.03581) | **Select K=2 of 20 anchors, then locally expand** by polar λ ∈ {0.92…1.08} × δ ∈ {−6°…+6°} → 50 candidates | NAVSIM v1 **90.2 vs 88.1**; ablation **K=1 → 88.1, K=2 → 88.5–88.6, flat K=20 → 79.8** | ✅ **The operator transfers and I used it verbatim** (§0). ⚠️ Its *model* needs BEV. **The K=20 → 79.8 collapse is the published warning against a flat large fan** — and we run 256 flat. |
| **TOAD** (arXiv 2606.07170) | CEM search in **control space (accel, yaw-rate)**, warm-started in a trust region around the proposals; **does not change the candidate set** | navhard EPDMS 54.6 → 56.3; **weak planner iPad 34.7 → 49.8 (+43.6 %)**; +1.9–20.4 ms | ✅ **No HD map required** — the only unblocked one in this row. ⚠️ Needs a competent frozen scorer as the reward; **ours is K1-bounded**, so this predicts to Goodhart. |
| **DriveVer** (arXiv 2607.00399) | 34 M verifier, **refines** rather than ranks: `τ + α·û`, α = 0.5, direction-only | DiffusionDrive 88.1 → **89.0**; directional beats absolute-residual **89.0 vs 88.3** | ⚠️ 34 M ≈ **26× our largest accepted lever**. The *direction-only* finding is the cheap import. |
| **MeanFuser** (2602.20060) · **MISTY** (2604.21489) | One-step MeanFlow / single-step generation | — | **Latency levers.** We are not latency-bound in this programme; ranked last **by the longitudinal criterion**, not dismissed. |

**On the selection step specifically** — the thing the brief asked for, because it is our measured
defect: **the field has largely stopped trying to fix selection and started refining instead.**
TOAD, DriveVer, HAD's low level and AlignDrive's longitudinal stage are all *post-selection
corrections*, not better rankers. Our own S1/K1 results are the same conclusion reached
independently, and they say the *ranking* half is closed here too. ⇒ **The productive reading of
K1 is not "the fan is exhausted" but "ranking within the fan is exhausted".**

⚠️ **The standing counterweight**, and it must travel with any expansion proposal: LLM-Assist
(arXiv 2401.00125, Table 1) — handing PDM-Closed 8,505 proposals instead of 15 drops it
**92.51 → 77.78**, with **TTC 93.11 → 62.89** and comfort 95.19 → 78.68 while *progress rises*.
A larger set is an adversarial search against the scorer's error. **This is why item 2 is a
regressed 1-D residual and never a search.**

---

## 2. P2 — hierarchical goal-setting, screened against the information-disjointness rule

| system | what the high level emits | predicted or supplied? | passes our screen? |
|---|---|---|---|
| **Mimir** (arXiv 2512.07130) | goal point at 4 s, from an **8,192**-point vocabulary; scored by DAC + proximity to the GT endpoint; **Laplace (μ, b)** uncertainty; `b` → sigmoid confidence that **modulates the guidance**. Ablation: goal alone **28.9 → 31.1 (+2.2)**, +uncertainty **→ 33.3 (+2.0)**. **2 diffusion steps**, same as ours | **PREDICTED** ✅ | ⚠️ **Partly.** Goal path reads perception + ego state; it never reads a situation classifier ⇒ **passes the disjointness rule**. But **DAC needs a drivable area we do not have** (same halving as GoalFlow ℳ₁ vs ℳ₃), and §0's r = 0.9973 says the remaining half is a `v0` echo here. |
| **AlignDrive** (arXiv 2601.01762) | a **drive path** (15 waypoints at 2 m spacing), then **1-D displacement along it** (M = 5 longitudinal anchors at 0.25/1.7/4.0/6.0/8.5 m) | **PREDICTED**, both stages ✅ | ✅ **Passes, and it is the only one that does cleanly.** **Vision-only, no HD map.** The high level is *geometry*, never a class label ⇒ structurally incapable of carrying a classifier posterior. |
| **HAD** | K=2 coarse anchors, then a local refinement region | PREDICTED ✅ | ✅ on the rule; ⚠️ model needs BEV. |

⭐ **Why AlignDrive is the right hierarchy for us and Mimir is not.** Our binding rule is that the
goal path and the situation path stay **information-disjoint at inference**. A *goal point* is a
compact summary that a shared trunk could easily make classifier-correlated, and the correlation
would be invisible. **A drive-path + along-path-displacement factorisation has no such surface**:
the high level emits raw geometry with no categorical bottleneck, so there is nothing for a class
posterior to hide inside.

**The engineering constraint that makes this checkable:** the goal/path head and the situation
classifier may share the **vision trunk** — a shared trunk is not the classifier's *output* — but
there must be **no feature or gradient path from the classifier HEAD into the path head**. That is
a one-line assertion in the model and it should be tested, not asserted.

**AlignDrive's isolating ablation** (its Table 5), which is the number that earns item 2 its rank:
parallel path+trajectory **83.21 DS / 63.18 SR / 22.7 % collision** → path-conditioned 1-D
displacement **85.82 / 66.81 / 16.3 %** = **+2.61 DS, +3.63 SR, −28.2 % relative collision from the
factorisation alone**, against a run-to-run spread of ~±0.6 DS across its 3 seeds.

⚠️ **Cost caveat, stated plainly:** AlignDrive's longitudinal stage is **+19.8 M parameters** — far
outside our band (+897 / +385 / +128 / 0). **We would import the parameterisation, not the module.**
E-EXP-1 shows the entire ceiling is reachable with a **5-way choice ≈ 2.32 bits**, so the honest
head is a 5-way softmax over λ (~1.3 k params), and K7's capacity sweep — which peaked at **129
parameters** — says even that may be generous.

---

## 3. P3 — combinable techniques, with what each would cost us

| technique | source | our param cost | what we lack | cheapest discriminating test |
|---|---|---|---|---|
| **1-D along-path displacement head** | AlignDrive | **≈ +1.3 k** (5-way λ softmax); K7 suggests ~129 may suffice | nothing — the selected trajectory is already there | **E-EXP-2** (item 1) |
| **Laplace uncertainty on a high-level signal, gating its own guidance** | Mimir | **+1 param per output** (the scale `b`) | nothing | Add `b` to any λ head; falsifier = the gate never closes ⇒ it is decoration |
| **Direction-only refinement** (predict a unit vector, fixed magnitude) | DriveVer (89.0 vs 88.3 vs absolute residual) | ~0 — a parameterisation choice | nothing | Free A/B inside item 2: λ-as-direction vs λ-as-magnitude |
| **Select-then-locally-expand, K small** | HAD (K=2 → 88.5; K=20 → 79.8) | 0 at inference | nothing | Already partly measured: our fan is 256 **flat**, and 73.8 %/72.1 % of it is never selected |
| **FPS anchor vocabulary over the canonical corpus** | DriveAnchor | 0 | nothing (parity-safe — selects among existing GT trajectories) | Carried over from the 2026-08-03 scan; **still unrun** |
| **Intervention probing** (swap a latent, watch the plan move) | arXiv 2606.31106 (36–40 of ~59–64 cases) | 0 | nothing | The right form for H26; pre-register the threshold, since **~62 % is the published expectation** |
| ⛔ **Constraint guidance / drivable-area terms** | GuideFlow, Mimir DAC, PLAN-S | — | **a map** | **Blocked on PhysicalAI.** Would have to come from AlpaSim/NuRec `map.xodr`. Say so; do not smuggle it in. |

---

## 4. E-EXP-1 — the measurement, in full

**Pre-registration:** `PREREG_E_EXP1.md`, git blob **`000ed1cdc45da59c7b4ca406f921ba18c024ce4e`**,
sha256 `01b4d6d6…`, pinned in `raw/prereg_pin.json` **before** any statistic here existed.
`git ls-files -s` must print that same blob id.

**n = 881 canonical windows / 40 val episodes**, both arms. **Estimator: paired episode-cluster
bootstrap**, unit = episode, `n_boot = 2000`, `seed = 0`. ⛔ `overlapping_holdout_se` never called.

**Instrument checks — all passed, none waived:**
- frame convention **verified, not assumed**: 98.41 % of GT 2 s displacements are forward-positive;
  `corr(gt_x, v0) = 0.9973`;
- λ = 1.00, δ = 0° reproduces the fan **bit-identically**;
- ⭐ my `ade_oracle_fan` reproduces the registry's banked oracle-in-fan **exactly**: **0.1914**
  (128) and **0.1639 vs 0.1640** (256). The instrument agrees with the bank before it is trusted.

**Verdict: PASS-L on both arms** — Δ(A − L) = **+0.0301 [+0.0215, +0.0390]** (base),
**+0.0194 [+0.0128, +0.0266]** (XL), both separated, both the pre-registered sign.

⚠️ **What this does NOT say, restated because the numbers are large enough to be misread:**
1. **It is an ORACLE — reachability, not findability.** λ\* is chosen with the ground-truth future.
   The 53.7 % is subject to the **same aleatoric inflation** as the oracle gap it is compared to.
   It is a fair ceiling-to-ceiling comparison against K1's in-sample ≤ 8.4 %, and nothing more.
2. **Open-loop only.** Open-loop does not predict closed-loop here (0.45 m → 1.69 m, MEASURED).
3. **TACTICAL and STRATEGIC families: NOT-APPLICABLE**, n = 0 — this instrument cannot see a
   manoeuvre decision or a route. Declared per the binding rule's clause 5, not silently dropped.
4. **REF-C only.** The flagship has no fan at all (`anchor_decoder is None`).

**LONGITUDINAL / LATERAL read** (2 s waypoint, shipped selection): mean |along-track| error
**0.8661 m** vs mean |cross-track| **0.3050 m** (base); 0.8706 / 0.3072 (XL). Direction is
consistent with the programme's 88.7 %-longitudinal figure. ⚠️ These are mean-absolute components
and are **not** the same statistic as that 88.7 %; I do not claim a numeric reproduction.

### 4.1 E-EXP-1b — ⚠️ POST-HOC, decides nothing

Labelled post-hoc everywhere because it was **not** pre-registered. Leave-one-episode-out, scored
as realised ADE:

| arm | base | XL |
|---|---|---|
| MAJORITY (no features) | +0.0000 — reproduces the incumbent, as it must | +0.0000 |
| **`v0` decile lookup** | **−0.0039 [−0.0097, −0.0006]** ✅ separated **ADVERSE** | **−0.0021 [−0.0045, −0.0006]** ✅ separated **ADVERSE** |
| `v0` × along-distance deciles | −0.0073 [−0.0173, +0.0022] not separated | −0.0032 [−0.0183, +0.0117] not separated |

λ\* distribution: **437/881 (49.6 %) at λ = 1.00**, the remainder spread near-evenly over the four
off-grid values. ⇒ λ\* is **not** a `v0` echo, and the cheap features do not find it.

---

## 5. PRE-REGISTRATION — E-EXP-2, the top item, both outcomes committed

**Question.** Is λ\* predictable from REF-C's own representation, out-of-episode, above **both**
controls?

**Arms** (all leave-one-episode-out, realised ADE, paired episode-cluster bootstrap, n_boot = 2000):
- **P-LATENT** — λ\* from REF-C's `fmap` tokens at the selected anchor's query.
- **C-SHUFFLE** — identical head, latents shuffled **across windows within an episode**.
  ⚠️ *Not* a permute-then-argmax control, which is vacuous here.
- **C-V0** — the `v0` lookup, already measured **adverse** (§4.1). It is carried as a floor.
- **C-MAJORITY** — λ = 1.00 always, i.e. exactly the shipped incumbent (measured: +0.0000).

**Primary + DIRECTION predicate.** Δ = mean(ADE `C-SHUFFLE`) − mean(ADE `P-LATENT`) must be
**> 0** and separated, **and** `P-LATENT` must separably beat `C-MAJORITY` — a head that beats a
shuffle but not the incumbent is not a lever.

| outcome | verdict |
|---|---|
| **P-LATENT beats BOTH, separated** | **FUND item 2.** Recovers *r* of the 0.159 m ceiling; ship as a 5-way λ softmax (~1.3 k params) with a Mimir Laplace gate. Report all four families; primary read is LONGITUDINAL. |
| **beats shuffle but NOT majority** | **DO NOT FUND.** The signal exists and is not reachable through the head. Report as a bound; revisit only with a representation change. |
| **fails the shuffle control** | ⛔ **Item 2 is DEAD and K7 generalises**: the along-track residual is absent from the representation, not from the parameterisation. **Say so publicly** and redirect to the encoder. |

**Cost:** one feature dump on an idle pod, **0 training**. **Falsifier is symmetric and the
"dead" branch is committed in advance.**

---

## 6. What I did NOT cover — stated plainly

- **No PDF-level verification.** Every PUBLISHED number here reaches me through **one automated
  extraction hop** (HTML/PDF → summariser). Class them `PUBLISHED (cited) — SINGLE-HOP` and
  re-verify against the PDF before any of them decides a GPU-day. **BridgeDrive is the weakest**:
  I only reached its abstract page, so its step count, latency and ablations are **unread**.
- **No VLA / VLM-planner sweep**, no RL-from-simulation sweep beyond DiffusionDriveV2 and HAD, no
  re-derivation of the encoder question (settled 2026-07-22).
- **E-EXP-1 is REF-C only**, open-loop, on 881 windows. Nothing here is a closed-loop claim.
- **I did not re-verify K1/K2/K7** — they are `INHERITED` from `MODEL_REGISTRY.md` and the REF-C
  brief. ⚠️ The **≤ 8.4 %** figure the brief itself flags as resolving to a prose note rather than a
  JSON is load-bearing in §0's "6.4×" comparison. **If that number is wrong, the ratio is wrong.**
- **Item 5 (FPS anchors) and the intervention-probing method** are carried forward unrun from the
  2026-08-03 scan; I did not duplicate that scan's §1–§7.

---

## 7. Deliverable manifest

| artifact | where it lives |
|---|---|
| this report | `TanitAD Research Hub/Architecture & Inference/Research/2026-08-04-planner-hierarchy-sota/PLANNER_HIERARCHY_SOTA.md` — **repo, staged** |
| pre-registration (pinned before results) | `…/PREREG_E_EXP1.md` — **repo, staged**, blob `000ed1cd…` |
| content pin | `…/raw/prereg_pin.json` — **repo, staged** |
| E-EXP-1 runner | `…/code/e_exp1_axis_reach.py` — **repo, staged** |
| E-EXP-1b runner (post-hoc) | `…/code/e_exp1b_lambda_echo.py` — **repo, staged** |
| E-EXP-1 results | `…/raw/e_exp1_axis_reach.json` — **repo, staged** |
| E-EXP-1b results | `…/raw/e_exp1b_lambda_echo.json` — **repo, staged** |

**Hosts touched:** none — dev-box CPU only. **GPU spent: 0.** **Nothing is stranded** on a pod or
in a worktree. Inputs were the banked fans in
`…/Implementation/incoming/2026-08-03-s1-climbout/raw/`, read-only.

### Escalations — decisions that are not mine and must not sit in a file

1. **E-EXP-2 needs a feature dump on an idle pod** (0 training, ~minutes). It is the gate on
   everything in §2–§3. **Owner needed.**
2. **Item 2 changes REF-C's decoder output contract** (a trajectory plus a longitudinal scale).
   That is a **D-018 escalate** before it becomes a trained config.
3. ⚠️ **The ≤ 8.4 % re-scorer bound should be promoted from prose to a JSON artifact.** It is now
   load-bearing in a headline ratio, and `CLAUDE.md` forbids exactly this shape of dependency.

### Sources

- [BridgeDrive: Diffusion Bridge Policy for Closed-Loop Trajectory Planning (arXiv 2509.23589)](https://arxiv.org/abs/2509.23589)
- [DiffusionDriveV2: RL-Constrained Truncated Diffusion (arXiv 2512.07745)](https://arxiv.org/html/2512.07745)
- [GuideFlow: Constraint-Guided Flow Matching (arXiv 2511.18729)](https://arxiv.org/html/2511.18729v1)
- [HAD: Hierarchical Diffusion with Metric-Decoupled RL (arXiv 2604.03581)](https://arxiv.org/html/2604.03581v1)
- [TOAD: Test-Time Trajectory Optimization (arXiv 2606.07170)](https://arxiv.org/html/2606.07170v1)
- [DriveVer: Lightweight Trajectory Evaluator as Test-Time Verifier (arXiv 2607.00399)](https://arxiv.org/html/2607.00399)
- [AlignDrive: Aligned Lateral-Longitudinal Planning (arXiv 2601.01762)](https://arxiv.org/html/2601.01762v2)
- [Mimir: Hierarchical Goal-Driven Diffusion with Uncertainty Propagation (arXiv 2512.07130)](https://arxiv.org/pdf/2512.07130)
- [MeanFuser: One-Step Multi-Modal Trajectory Generation via MeanFlow (arXiv 2602.20060)](https://arxiv.org/pdf/2602.20060)
- [MISTY: High-Throughput Motion Planning via Mixer-based Single-step Drifting (arXiv 2604.21489)](https://arxiv.org/pdf/2604.21489)
- [SimpleVSF: VLM-Scoring Fusion (arXiv 2510.17191)](https://arxiv.org/html/2510.17191v1)
- [DiffusionDrive (CVPR 2025, arXiv 2411.15139)](https://arxiv.org/abs/2411.15139)
- [LLM-Assist (arXiv 2401.00125)](https://arxiv.org/abs/2401.00125) — the large-fan Goodharting counterweight
- [What Probing Reveals about Autonomous Driving (arXiv 2606.31106)](https://arxiv.org/html/2606.31106)
