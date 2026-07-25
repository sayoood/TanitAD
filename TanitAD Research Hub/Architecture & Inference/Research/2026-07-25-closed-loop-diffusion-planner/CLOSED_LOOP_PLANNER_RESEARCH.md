# Is the anchored-diffusion planner (REF-C) good enough to drive closed-loop? — verdict, and the design that fixes it

**2026-07-25 (Europe/Berlin), Architecture & Inference research stream. Pod-free: no GPU launched, no pod
written to, no training started.** This is research + design for a Sayed decision. It does **not** auto-commit
GPU-days.

**Evidence-class legend (CLAUDE.md operating standard).** `MEASURED` (ours + artifact path) · `PUBLISHED`
(external, cited) · `INHERITED` (another of our docs, not re-verified here) · `ESTIMATED` · `HYPOTHESIS`.
A claim that would decide a GPU-day is MEASURED or PUBLISHED. Every interval carries its estimator; all of
ours are the **episode-cluster bootstrap** over val episodes (`taniteval/ci.py`), **paired** for two arms on
identical windows.

**Read before writing:** `RETRACTION_LOG.md` (C1–C6 + the 07-23/07-24 closed-loop entries, incl. entries 57
and 58 — the two most recent over-claimed-closure retractions in this exact direction),
`AGENT_OPERATING_STANDARD.md`, `MODEL_REGISTRY` §0/§1.2/§4.4, and the full closed-loop arc:
`Research/2026-07-23-closed-loop-wm-training-verdict.md`, `Research/2026-07-23-planner-is-the-bottleneck.md`,
`Research/2026-07-24-low-ood-closedloop-renderer.md`, and the four RESULTS files in
`…/incoming/2026-07-24-refccl/`.

---

## 0. TL;DR — the verdict, the reframe, and the ranking

**Verdict (§1): NO.** REF-C is our best planner by a decisive margin and it is the right *architecture*, but
it is **not closed-loop deployment-grade**, and the reason is mechanistically MEASURED, not stylistic: it is
**covariate-shift blind** (a 1 m off-path view moves its plan **7 mm**), and its plan **degrades on-policy**
(plan cross-track 0.57 → 12.98 m over a junction crossing) while the controller tracks that degrading plan
faithfully (0.49 m). It fails *where* it matters — junctions — at a rate that is not a rounding error:
**36.8 % of junction windows leave the corridor** even on our friendliest instrument.

**The reframe (the load-bearing contribution of this doc).** The program concluded, over four experiments
(D2 → RefcCL → LOWOOD-CL → powered n=40), that the closed-loop-improvement direction is **BOUND**. Re-reading
the *same* artifacts against their source code, that conclusion carries **two un-named confounds**, and both
are cheap to remove:

1. ⭐ **HORIZON.** Our low-OOD closed-loop instrument rolls out for **2.0 seconds** — `K = max(WP_STEPS) = 20`,
   `DT = 0.1` (`MEASURED`, source-read: `lowood_closedloop.py:59-60`). The failure we are chasing was measured
   over a **20-second** junction crossing on AlpaSim. Behaviour-cloning compounding error scales as **T²ε**
   (`PUBLISHED`, Ross & Bagnell) → the two instruments differ by ~**100×** in the compounding-error bound.
   "On-policy drift is only 0.13–0.27 m so the objective is starved of informative failures" — the stated
   mechanism of the LOWOOD-CL **BOUND** verdict — is very largely a *mechanical consequence of a 2 s horizon*,
   not a property of the objective.
2. ⭐⭐ **STRATUM.** LOWOOD-CL fine-tuned on **all** windows (`max_windows=None`, `--collect-windows 0=all`;
   `MEASURED`, source-read `lowood_cl_train.py:217,292`) of a corpus that is **59.6 % lane-keep**. On the
   stratum that actually fails, the on-policy objective **improved every metric** — junction `dCDR` **+0.0159**,
   `dADE` **+0.092**, `dPEAK` **+0.302** (all n.s.) — and it was destroyed by the stratum that never fails —
   longitudinal `dCDR` **−0.130 S**, `dADE` **−0.825 S**. The overall "BOUND" is an **average over a
   population that is ~98.7 % non-failing**. Every config in the arc applied the recovery objective
   *globally*; none was failure-gated.

Both confounds point the same way, and the 2026 literature has independently converged on the fix:
**train the closed-loop objective only on the states where the policy actually fails, and protect the states
where it does not** (R2LPL, `PUBLISHED`, arXiv:2606.30537 — mines *recoverable pre-failure* states, scores
**anchors** with a rule-based drivability score, and adds anti-forgetting replay: nuPlan Test14-hard
**60.67 → 83.51**, largest gains exactly where the base model failed most).

**Ranked interventions** (expected closed-loop gain ÷ (cost + risk)); **none of the three needs a renderer**:

| # | intervention | mechanism it escapes | needs renderer? | first cost |
|---|---|---|---|---|
| **1** | **Failure-targeted closed-loop FT on a horizon-extended instrument** (stratified + failure-gated CL-SFT with replay) | escapes #2 (the closed configs were global + 2 s) — and re-tests the arc's central claim at ~5× the failure density | **NO** | **E1a ≈ 4 GPU-h, PURE EVAL, zero training** → E1b ~1 pod-day |
| **2** | **Restore covariate-shift OBSERVABILITY** (localize the 0.0074, then fix upstream of any recovery objective) | escapes #1 at its root: every closed config changed the *plan objective*; none asked whether the offset is *perceivable or representable* | **NO** | **E2a ≈ 2–3 GPU-h, ZERO training** |
| **3** | **A drivable-corridor channel** (monocular road/lane seg + GeoCalib ground plane → BEV corridor) → a PDM-style anchor scorer | supplies the one ingredient shared by the fair metric, the CL reward, TL compliance and offset observability | **NO** (but a real build) | probe first (~1 day); build 1–2 wk |

**Cheapest discriminating experiment, to run first:** **E2a** (§4.2) — a zero-training, ~2–3 GPU-hour probe
suite that decomposes `recovery_ratio = 0.0074` into *perception* / *representability* / *truncation* /
*conditioning*. It costs no training and it **conditions the value of everything else**: if the lateral offset
is not linearly decodable from the encoder features, then no recovery objective — global, stratified,
on-policy or synthetic — can ever work, and the whole four-experiment arc was arguing about the wrong layer.

---

## 1. The blunt verdict

**Is the anchored-diffusion planner good enough to drive closed-loop? No — and the honest form of "no" has
four parts.**

**(a) It is decisively our best planner, and the architecture choice is right.** `MEASURED`, three independent
instruments agreeing on the ordering: AlpaSim n=12 paired (pass **8/12 vs 2/12**, mean score **0.496 vs
0.066**, paired Δ **−0.430 [−0.646, −0.215]**, sign-test 8–0, `plan_deviation` **0.342 vs 1.125**); real-footage
low-OOD n=40 (closed ADE@2s **0.564 [0.452, 0.676] vs 1.488 [1.329, 1.647]**; corridor departure **0.0134
[0.0059, 0.0223] vs 0.0318 [0.0152, 0.0531]**, paired CI excludes 0 on both). The anchored form is *why*: the
output lives near the convex hull of real GT trajectories and cannot swing arbitrarily, which is exactly the
property v1's unconstrained regression head lacks. This is also the field's answer (`PUBLISHED`: DiffusionDrive
arXiv:2411.15139; AnchDrive arXiv:2509.20253; DiffusionDriveV2 arXiv:2512.07745 — 91.2 PDMS / 85.5 EPDMS on
NAVSIM). **Do not change the decoder family.**

**(b) It is nonetheless not deployment-grade, and the mechanism is measured.** `MEASURED`:
- **Covariate-shift blindness.** `recovery_ratio` = **0.0074 [0.0036, 0.0115]** at a 1.0 m lateral offset
  (n=881 windows / 40 eps) — the plan corrects **<1 %** of the offset; yaw response is negligible and
  wrong-signed. A 1 m displacement of the world moves the plan by ~**7 mm**.
- **On-policy plan collapse.** Over 15 AlpaSim junction scenes / 675 steps the ego **tracks its own plan to
  0.49 m** while the *plan itself* degrades from cross-track **0.57 m → 12.98 m**. This is planner
  covariate-shift, faithfully executed — not a controller problem.
- **Junctions are the failure stratum, and the rate is high.** On the low-OOD instrument, junction
  (182 win / 22 ep): REF-C corridor departure **0.064 [0.036, 0.090]**, peak XTE **1.458 m**,
  **window departure rate 0.368**. Longitudinal (374 win / 24 ep): corridor departure **0.0004**. The
  headline 0.0134 is an average of a **0.0004** regime and a **0.064** regime.

**(c) Every free inference-time lever is already exhausted for the junction failure.** `MEASURED`: cost-guided
**selection** (Gate 0) leaves junction off-road unchanged; per-denoise **gradient guidance** (Gate 0b)
synthesizes trajectories outside the anchor set and makes **0 planned trajectories off-road** — *the plan is
on-road at every junction* — **yet the ego still departs** (junction off-road 0.73 → 0.73 [0, 0]); **WM-MPC**
(rung 3) ties (+0.005 to +0.011, none separated; lateral off-road separated **worse**, +0.136, from
stochastic-selection jitter). Three independent lines converge: **only closed-loop-aware training shapes the
executed path.** (Ship the gradient nudge anyway — it is a free strict improvement: intersection at-fault
collisions 0.71 → 0.43.)

**(d) But "closed-loop training is BOUND" is over-claimed.** Four experiments (D2, RefcCL, LOWOOD-CL, powered
n=40) reported the direction closed. Two of the three closure claims in this direction have **already been
retracted** by cheap follow-ups (RETRACTION_LOG entries 57 and 58: "needs a renderer" → C3; "decisively
closed" → C3, the ADE cost was ~74–95 % a knife-edge-L2 artifact). §2 shows a third confound survives in the
current reading. **The bar the operating standard sets — run the cheapest metric-or-power check *before*
declaring a direction closed — has not yet been met for the horizon and stratum axes.**

**One-line answer to Sayed:** *No — REF-C plans blind to its own lateral error and its plan collapses on-policy
at junctions; but the four experiments that "closed" the fix were all run at a 2-second horizon on a
98.7 %-non-failing population, and on the failing stratum the on-policy objective actually improved every
metric. The fix is to stop training recovery everywhere and train it where the policy fails — and to first
spend 3 GPU-hours finding out whether the planner can even perceive that it is off-path.*

---

## 2. Our own evidence, re-read — the two confounds

### 2.1 The instrument's closed loop is 2.0 seconds long

`MEASURED` (source-read, `…/incoming/2026-07-23-lower-ood-closedloop-source/lowood_closedloop.py`):

```
K  = max(WP_STEPS)   # 20 = 2 s
DT = 0.1
```

`corridor_departure_rate` is defined as the fraction of these **K=20 on-policy steps (0–1.9 s)** whose |XTE|
exceeds a lane half-width. So:

- **Every low-OOD closed-loop number in the program is a 2-second number.** REF-C's 0.0134, flagship's
  0.0318, the junction 0.064/0.368, the ±1.16× OOD envelope, the LOWOOD-CL on-policy drift of 0.13–0.27 m,
  and the frac_inside 0.948 — all at T = 2 s.
- **Compounding error is quadratic in the horizon.** `PUBLISHED`: for a BC policy with per-state error ε,
  `J(π) ≤ J(π*) + T²ε` (Ross & Bagnell; the H² term is the standard motivation for DAgger). Between T = 20
  steps and T = 200 steps the bound differs by **100×**. `HYPOTHESIS` (well-grounded): most of REF-C's
  closed-loop competence at 2 s is the *absence of time to diverge*, not robustness.
- **The comparison instrument agrees.** The plan-collapse we care about (0.57 → 12.98 m) is a **20 s**
  AlpaSim junction measurement. Our low-OOD instrument cannot express it: at 2 s and 25 m/s the ego travels
  ~50 m — barely into the junction.
- **It is cheaply extensible.** Episodes are **199 frames ≈ 20 s** (`MEASURED`, `clip_len: 199` in the VLM
  production pass), so an **8–10 s** rollout is feasible from early windows without new data, new code paths,
  or a renderer. The cost is that windows-per-episode drops and OOD grows with drift — both are *measurable*
  (the harness already reports `frac_inside` the envelope), so the honest product is a **departure-rate and
  OOD-ratio curve vs horizon**, with the horizon at which the instrument stops being low-OOD read off it.

⚠️ This does **not** invalidate the REF-C > flagship ordering (paired, same windows, same horizon — the
horizon cancels). It bounds what "REF-C departs only 1.3 % of the time" means.

### 2.2 The closed-loop objective was trained everywhere, and only helped somewhere

`MEASURED` (`…/incoming/2026-07-24-refccl/LOWOOD_CL_TRAIN_RESULTS.md`), held-out 28:40, paired
episode-cluster bootstrap, positive = FT better:

| stratum | dCDR@1.75 | dADE@2s | dPEAK |
|---|---|---|---|
| overall | −0.0460 [−0.120, +0.005] n.s. | **−0.329 [−0.732, −0.034] S (worse)** | −0.649 S |
| **junction** | **+0.0159** [−0.008, +0.042] n.s. | **+0.092** [−0.110, +0.241] n.s. | **+0.302** n.s. |
| longitudinal | **−0.130 [−0.269, −0.018] S** | **−0.825 [−1.482, −0.195] S** | −1.677 S |

Two facts, both MEASURED:
- **The training set was unstratified.** `lowood_cl_train.py` collects `max_windows=None` — *all* FT windows
  (`--collect-windows`, help text `0=all FT windows`). The parity corpus is **59.6 % lane-keep, 14.25 % turns**
  (`CORPUS_PROFILE.md`). So a *recovery* objective was applied overwhelmingly to states where the base policy's
  corridor departure is **0.0004**.
- **The stratum signature is consistent across the whole arc.** The LOWOOD-CL results file names it itself:
  *"the recurring signature is junction ADE improves (+0.09..+0.13) while longitudinal/straight ADE degrades."*

`HYPOTHESIS` (grounded in the two MEASURED facts): the arc's global "BOUND" is **not** evidence that
closed-loop training fails on the failure stratum. It is evidence that a globally-applied recovery objective
buys reactivity where heading error dominates and pays for it with over-reaction on straights — i.e. the
classic catastrophic-forgetting/over-generalization failure that R2LPL's *knowledge replay* and failure-gating
exist to prevent (`PUBLISHED`, arXiv:2606.30537).

⚠️ **Honest counter-evidence I must not hide:** the junction deltas are **not CI-separated** (n=12 held-out;
the powered lesson from the n=40 cross-fit is that this program's held-out sets are underpowered for ~1 pp
effects). The junction improvement is a *direction*, not a win. That is precisely why intervention 1 is
proposed as an experiment with both outcomes committed, not as a conclusion.

### 2.3 What is genuinely, durably closed (do not re-propose)

`MEASURED`, do not spend GPU-days re-testing these:
- **Global recovery-augmentation FT** (synthetic single-step, all configs incl. gentle/g1/g2/g3 + speed terms):
  at n=40 cross-fit the departure benefit **reverses** (−0.0302 [−0.0595, −0.0088] S; departs 3.3× more).
- **Unfreezing the encoder** does not dissolve it (RefcCL-s2, material move feat_cos 0.966, canary holds,
  dCDR +0.0002 n.s.). *Positive banked: the encoder is safely fine-tunable.*
- **Inference-time floors** (selection guidance, per-denoise gradient guidance, WM-MPC) for junction off-road.
- **Selection / re-ranking** as a lever (oracle gap ~92 % irreducible).
- **DAgger over the WM's own imagined latents** (`DAGGER_HURTS`: closed-loop ADE +0.266 [0.008, 0.550],
  off-road proxy +0.548) — self-referential rollouts are a trap. Any WM-as-simulator proposal must be
  non-self-referential or heavily guarded.

---

## 3. State of the art, 2024–2026 — what the field knows that we can use

*(Compressed to what changes a decision. Citation hygiene note in §9.)*

### (a) Why diffusion planners are brittle on-policy
- **The prior dominates when conditioning is weak.** DiffusionDrive (`PUBLISHED`, arXiv:2411.15139) works by
  *truncating* the schedule and starting from an **anchored Gaussian** — the design intent is that the output
  stays near the anchor set. Our REF-C runs **2 denoise steps over 256 anchors**. The same property that gives
  us `plan_deviation 0.342` (a virtue) mechanically bounds how far a plan can move in response to an OOD
  observation. `HYPOTHESIS`: this is a candidate explanation of `recovery_ratio = 0.0074` that is *independent*
  of training, and it is testable at inference with zero training (E2a-D, §4.2).
- **Mode collapse is data-scale-bound.** `PUBLISHED` (arXiv:2602.22801): at ~100 K frames a diffusion planner
  shows "negligible multimodal capability" with trajectories collapsing to a single mode; multimodality emerges
  between **20 M and 70 M frames**; closed-loop success 59.00 % → 79.53 % across that scaling. Our parity corpus
  is ~473 K frames (2,376 eps × ~199), the new balanced corpus ~1.8 M. **We are 1–2 orders of magnitude below
  the published multimodality threshold** — which independently predicts our MEASURED "~92 % irreducible oracle
  gap" and says *selection is not where the headroom is*.
- **Replanning injects jitter.** `PUBLISHED`: temporal inconsistency, where "small perturbations across frames
  accumulate into unstable trajectories," is now a named failure mode with three 2025–26 fixes — pivot-conditioned
  sampling (CoPlanner, arXiv:2509.17080), history-annealed segment noise (Diffusion Forcing Planner,
  arXiv:2606.11019), asymmetric temporal guidance (arXiv:2603.25462). We MEASURED the matching symptom:
  rung-3's lateral off-road was separated **worse** (+0.136) *from stochastic-selection jitter*.
- **Ego-status shortcut.** `PUBLISHED` (arXiv:2312.03031, CVPR 2024; AD-MLP arXiv:2305.10430): in E2E planners
  "decision-making is disproportionately influenced by ego status," and a ~20-parameter no-vision MLP matches
  perception-based models on nuScenes. `MEASURED` (ours, `2026-07-17-openloop-l2-egostatus-shortcut.md`): on our
  own comma val a no-vision ego-status regressor scores **avg L2 0.658 m**, tied with the CTRV baseline, on a
  corpus that is **73.9 % straight-cruising — identical to nuScenes**. A planner that leans on ego state is
  *definitionally* covariate-shift blind, because a lateral offset does not appear in `v0`.

### (b) Closed-loop-consistent training **without** a photoreal reactive sim
- **CAT-K** (`PUBLISHED`, arXiv:2412.05334, CVPR'25 Oral): unrolls the policy so visited states stay **close to
  GT**; a 7 M CL-tuned model beats a 102 M open-loop one. ⚠️ Note the design intent — *staying close to GT* is
  the mechanism, so CAT-K does **not** manufacture large-deviation failures. Our LOWOOD-CL faithfully
  reproduced CAT-K behaviour (95 % of states inside a ±1.16× envelope, drift 0.13–0.27 m) and therefore
  inherited its blind spot on a source where the base policy does not fail.
- **RoaD** (`PUBLISHED`, arXiv:2512.01993): rollouts-as-demonstrations closed-loop SFT on the **NVIDIA Physical
  AI AV NuRec** corpus — our exact data ecosystem. Driving score **0.444 → 0.630 (+41 %)**, collisions
  **0.0525 → 0.0239 (−54 %)**, off-road **0.283 → 0.210**. Crucially it uses a **recovery blend**: when a
  sampled action diverges beyond a threshold it *interpolates toward the expert continuation*. And note the
  base rates: RoaD's policy fails at **28.3 % off-road**; ours fails at **1.3 %** on the instrument we
  fine-tuned on. **Same recipe, ~21× the failure density.**
- **R2LPL** (`PUBLISHED`, arXiv:2606.30537, 2026) — the best-matched paper to our situation. Rollout →
  event-detect (**Failure / Risk / Conflict**) → **mine recoverable states in a temporal window preceding the
  failure** → score **anchor candidates** with a rule-based safety/drivability/route/progress/comfort score →
  store as **sparse anchor-score supervision** → learn with **replay against forgetting**. It explicitly
  discards *unrecoverable* states where no valid candidate exists. nuPlan Test14-hard **60.67 → 83.51**,
  Val14 **75.39 → 91.26**, "notably larger gains on hard scenarios." Its simulator is **nuPlan — state-space +
  map, no renderer.** ⭐ Its architecture is *anchor-score supervision*, which is exactly REF-C's `sel_score`
  interface.
- **Latent-space closed-loop training is now real, but the latent is structured, not pixel-ish.**
  MAPLE (`PUBLISHED`, arXiv:2605.14201, 2026) does reactive multi-agent rollouts **in the latent token space of
  a VLA** — tokens are `⟨DYN⟩⟨TYPE⟩⟨MS⟩⟨TS⟩` (dynamics, type, map segment, traffic status) — explicitly to avoid
  simulators that are "computationally expensive… limited visual fidelity and distribution mismatch." Bench2Drive
  DS **85.2**, SR **67.1 %**, ablation **+12.3 DS** from the rollout. WorldRFT (arXiv:2512.19133) fine-tunes a
  policy over a latent world model with GRPO and collision-aware rewards (nuScenes collision 0.30 % → 0.05 %;
  NAVSIM 87.8 PDMS camera-only).
  ⚠️ **Direct answer to the brief's question "can OUR world model serve as the rollout simulator?"** —
  **Not as a free pixel-latent imagination loop.** Our own `DAGGER_HURTS` result is the counter-example, and it
  is consistent with the literature: the published successes roll out **low-dimensional, semantically grounded
  agent/scene state** (tokens, boxes, map segments) where the dynamics are near-exact, not a learned
  reconstruction latent the policy can exploit. Our low-OOD harness already does the *right* version of this for
  the ego (exact bicycle dynamics + a real frame); what is missing is the **map/agent state channel**, which is
  §4.3.
- **Latent Policy Barrier** (`PUBLISHED`, arXiv:2508.05941): treat expert latents as an implicit barrier
  separating in- from out-of-distribution states; a learned dynamics model predicts future latents and
  optimizes them to stay inside. **Requires no recovery demos and no simulator**, and its dynamics model is
  trained on **suboptimal policy rollouts** — which we already banked (32 k on-policy states from LOWOOD-CL).
  A genuinely different inference-time objective from the drivable-area cost we already ruled out: it steers
  *away from states where the policy is unreliable* rather than *toward on-road plans*.

### (c) Traffic lights — see §6.

### (d) Junction competence with scarce junction data
- **Curation beats volume for the long tail.** `PUBLISHED`: WOD-E2E (arXiv:2510.26125) mined 6.4 M miles for
  eleven long-tail event types each **below 0.03 % corpus frequency**; Semantic-Drive (arXiv:2512.12012) does
  the same with an open-vocabulary detector + a reasoning **VLM** — the pipeline shape we already own.
- **Perturbation-with-recovery-losses is the classic junction/road-keeping recipe, and it needs a map.**
  ChauffeurNet (`PUBLISHED`, arXiv:1812.03079) perturbs the agent off lane-center, refits a return trajectory,
  and adds explicit **on-road / collision losses** — the ancestor of our D2. It runs on a **mid-level
  representation with a map**, which is what lets it say *"you are fine here"*. Our D2 had no map and so
  penalized any deviation from the exact recorded path — the same limitation the LOWOOD-CL results file
  independently diagnoses.
- ⚠️ Our data reality is the binding constraint: **~13–22 distinct real junction episodes**, semantic scenarios
  **0 % labeled**; the new 49.742 h corpus lifts junction-clip presence 38 % → 61 % but adds no semantics.

### (e) Guided / safety-filtered diffusion at inference
`PUBLISHED`, a mature 2025–26 family: real-time CBF safety filters with arbitrary road-boundary constraints
(arXiv:2505.02395); **PC-Diffuser** (arXiv:2603.10330) embeds a certifiable path-consistent capsule-CBF
*inside the denoising loop* with a path-consistent correction to minimize distributional deviation;
**DualShield** (arXiv:2601.15729) uses reachability; HOCBF+Diffuser at signal-free intersections
(arXiv:2412.00162). This corroborates our own MEASURED result that a **rule barrier scores TLC 1.0 while a soft
prior scores 0.0** — hard constraints belong as *barriers composed with* the learned prior, never as costs
inside it. ⚠️ **But**: Gate 0/0b MEASURED that guidance does not fix our junction failure (the plan was already
on-road and the ego still departed), and **all of these need a geometric constraint set** — road boundaries,
a corridor, a stop line — which is exactly the artifact §4.3 builds. Guidance is a **beneficiary** of
intervention 3, not an alternative to it.

---

## 4. The three interventions, ranked — with pre-registered experiments

Ranking metric: **(expected closed-loop gain) ÷ (cost + risk)**. Priority order is strict: if this agent is
killed after one item, item 1 (specifically E2a, which is cheapest and gates the rest) is the one to keep.

### 4.1 Intervention 1 — Failure-targeted closed-loop fine-tuning on a horizon-extended instrument

**Mechanism.** Stop treating the corpus as homogeneous. (i) Extend the rollout horizon so failures can
actually compound (T² — the 2 s instrument cannot express the failure); (ii) run the closed-loop rollout only
to *detect* failures, and construct supervision **only from recoverable states in the window preceding a
departure** (R2LPL's Failure/Risk/Conflict detectors); (iii) supervise as **anchor-score** preferences
(REF-C's native `sel_score` interface) rather than a global waypoint-regression pull; (iv) **replay** the
non-failing longitudinal distribution to prevent the over-reaction that destroyed every previous config.

**Why it escapes our measured failures.** It escapes **#2** because every closed config was global and 2 s:
D2/RefcCL/LOWOOD-CL trained recovery on a population that departs **0.04 %** of the time (longitudinal),
and MEASURED, that is precisely where they broke (dADE −0.825 S) while the failing stratum improved on all
three metrics. It escapes the "starved of informative failures" diagnosis directly — that diagnosis is a
*failure-density* statement, and both (i) and (ii) raise failure density (junction windows already depart at
**0.368** window-rate at 2 s, ~5× the overall 0.078). It does **not** escape **#1** — see §4.2, which is why
E2a runs first.

**What it needs.** Nothing new: `lowood_closedloop.py` + `lowood_cl_train.py` + the junction/longitudinal
strata + the 32 k banked on-policy states + `band_ade2d` + `taniteval/ci.py`. **No renderer. No map. No new
data.** One eval pod.

#### Pre-registered experiment E1a — `CL-HORIZON-CURVE` (PURE MEASUREMENT, no training) ⭐ run early
- **Setup.** Re-run the existing low-OOD closed-loop eval for REF-C base **and** flagship v1 at
  K ∈ {20, 40, 60, 80, 100} (2/4/6/8/10 s), all 40 val eps, reporting for each K: `corridor_departure_rate`,
  `window_departure_rate`, `band_ade2d(1.0)`, peak XTE, **and `frac_inside` the MEASURED ±1.16× OOD envelope**,
  per stratum, with episode-cluster bootstrap CIs. Zero training; ~4 GPU-h.
- **Committed outcomes.**
  - **HORIZON-BOUND** — REF-C's junction `window_departure_rate` rises materially with K (e.g. ≥0.6 by 6 s)
    **while** `frac_inside` stays ≥0.85 ⇒ the instrument was under-measuring the failure by construction;
    **every standing low-OOD closed-loop number is re-stated as a 2 s number** (registry + leaderboard edit),
    the failure-dense training source for E1b exists, and the LOWOOD-CL BOUND verdict is **re-opened as
    horizon-confounded** (a C6-class entry is owed).
  - **HORIZON-NULL** — departure rates are flat in K, or `frac_inside` collapses below ~0.7 before failures
    appear ⇒ the 2 s instrument was *not* hiding the failure, or the homography envelope forbids the longer
    horizon. Then the low-OOD source is genuinely exhausted for road-keeping, the renderer paths are
    **necessary** (not merely nicer), and E1b is **not run**. This outcome kills my own headline reframe, and I
    commit to it.
- **Either way it is decision-useful**, and it costs no training. It also produces the first
  **departure-rate-vs-horizon curve** in the program, which every future closed-loop claim should cite.

#### Pre-registered experiment E1b — `CL-FAILGATE` (gated on E1a = HORIZON-BOUND, and on E2a ≠ BLIND-PERCEPTION)
- **Setup.** Decoder-only (WM-safe, no canary needed), identical held-out protocol to D2/RefcCL/LOWOOD-CL
  (episode-disjoint, **plus** the n=40 2-fold cross-fit that the powered eval established as this program's
  standard — the n=12 single split is retired). Roll out at the E1a-selected horizon. Build supervision from:
  (1) states in the 1–2 s window **preceding** a corridor departure; (2) states with high plan-vs-GT
  disagreement (Conflict); **excluding** unrecoverable states (no anchor in the 256-vocabulary returns to the
  corridor — we MEASURED that ~21 % of off-road junction moments have no on-road anchor, so this filter is not
  hypothetical). Supervise **anchor scores**, not waypoints. **Replay** an equal budget of untouched
  longitudinal windows each step. Primary read `band_ade2d(1.0)` + `corridor_departure_rate`, **per stratum**.
- **Committed outcomes.**
  - **WIN** — junction `dCDR` **CI∌0 positive** at n=40 cross-fit **AND** longitudinal `dCDR` and
    `band_ade2d` both CI∋0 (no collateral damage) ⇒ the arc's "BOUND" was a **global-objective artifact**;
    failure-gated CL training is the lever; scale it (more junction episodes from the 49.7 h corpus) and
    promote to the v4 curriculum.
  - **BOUND-CONFIRMED** — junction `dCDR` CI∋0 or still negative even when the objective is applied *only*
    where the policy fails, on a failure-dense horizon ⇒ the closed-loop road-keeping direction is closed on
    this instrument **for the right reason this time** (not a stratum or horizon artifact). Redirect the entire
    closed-loop budget to intervention 3 and the reactive-agent renderer (Path 3).
  - **COLLATERAL** — junction improves but longitudinal degrades despite replay ⇒ the trade is real and
    *representational*, not a sampling artifact; that is a direct escalation to intervention 2's fix.
- **Cost** ~1 eval-pod-day. **No renderer, no training pod.**

### 4.2 Intervention 2 — Restore covariate-shift OBSERVABILITY (the 0.0074, attacked upstream) ⭐ cheapest

**Mechanism.** Every closed experiment changed the **planning objective** under recovery supervision. **None
asked the prior question: can this model perceive, or represent, that it is 1 m off-path?** `recovery_ratio =
0.0074` has (at least) four mutually-distinguishable causes, and they imply completely different fixes:

| cause | fix if true | falsifier |
|---|---|---|
| **P — perception**: lateral offset is not encoded in the features | auxiliary **lane-relative pose head** (predict δ_lat, δψ). **The labels are free**: the harness *applies* a known warp δ, so `(warped frame → δ)` is a self-supervised pair at zero labeling cost | linear probe for δ on frozen features of warped frames |
| **R — representability**: no anchor in the 256-vocabulary returns to the corridor from 1 m off | re-fit the anchor vocabulary on junction-balanced + laterally-offset trajectories (cheap; anchors are a 0-param buffer) | offline geometric check, **zero model calls** |
| **T — truncation**: 2 denoise steps cannot move the sample far from the anchored Gaussian | more denoise steps at inference, or an offset-conditioned anchor selection | inference-only denoise-step sweep |
| **C — conditioning**: the plan is dominated by `v0`/`nav_cmd` and the ego-status shortcut (`PUBLISHED`, arXiv:2312.03031; our own MEASURED 0.658 m no-vision ceiling on a 73.9 %-straight corpus) | ego-state dropout / re-balance the conditioning; note REF-C already ships `ego_dropout 0.5`, so this may already be partly handled — measure, don't assume | ablate `v0` at inference and re-measure the probe |

**Why it escapes our measured failures.** It escapes **#1** at its root rather than downstream of it, and it
escapes **#2** because it is **not a recovery objective at all** — an auxiliary perception head does not touch
the plan loss, so it cannot reproduce the departure↔ADE trade that every previous config hit. And RefcCL
already MEASURED that REF-C's encoder is **safely fine-tunable at a material move** (feat_cos 0.966, canary
holds), which de-risks the only architectural change involved.

**What it needs.** The deployed REF-C checkpoint, the existing `recovery_probe.py`, the anchor buffer, one GPU
for a few hours. **No renderer, no new data, no training** for the diagnosis.

#### Pre-registered experiment E2a — `RECOVERY-LOCALIZE` (ZERO training, ~2–3 GPU-h) ⭐⭐ **RUN THIS FIRST**
- **Setup.** Four probes over the same n=881 windows / 40 eps used for the original `recovery_ratio`, at
  lateral offsets {0.5, 1.0, 1.75} m and yaw {±5°}:
  - **A (perception).** Ridge/linear probe from frozen REF-C encoder features (both the pooled feature **and
    the conv fmap the decoder cross-attends** — RefcCL MEASURED these can diverge) → applied warp δ. Report R².
  - **B (representability).** Offline: for each probe state, does any of the 256 anchors, executed from the
    offset pose, return |XTE| below the corridor within 2 s? Report the fraction with ≥1 valid anchor. Zero
    model calls.
  - **C (conditioning).** Re-run the recovery probe with `v0` ablated / randomized and with `nav_cmd` varied;
    report Δ`recovery_ratio`.
  - **D (truncation).** Re-run the recovery probe with denoise steps ∈ {2, 4, 8, 16} (inference only, no
    retraining); report `recovery_ratio` vs steps.
- **Committed outcomes** (these are branches, and each names its own next action):
  - **BLIND-PERCEPTION** (A: R² < ~0.3 on the fmap) ⇒ the offset is **not in the representation**. This is the
    highest-value finding available: it explains all four failed experiments in one stroke and **retroactively
    predicts that no recovery objective could have worked**. Next action: E2b, the aux lane-relative pose head
    with free warp labels. *This branch would also mean intervention 1's E1b is likely to fail, and E1b should
    be deferred behind E2b.*
  - **PERCEIVABLE** (A: R² ≥ ~0.6) ⇒ the information is there and the planner ignores it ⇒ the fault is in
    conditioning/objective; look to C and D, and E1b's failure-gated objective becomes the primary lever.
  - **UNREPRESENTABLE** (B: < ~50 % of offset states have a returning anchor) ⇒ **re-fit the anchor vocabulary
    first**; no objective can select a trajectory that does not exist. (Consistent with the MEASURED Gate-0
    ceiling: ~21 % of off-road junction moments have no on-road anchor.)
  - **TRUNCATION-BOUND** (D: `recovery_ratio` rises ≥5× from 2 → 16 steps) ⇒ a **free inference-time
    improvement** exists that the whole program has not tested; measure its latency cost against the 100 ms
    budget (batch-1 tick was MEASURED ~50 ms at K8, so headroom plausibly exists) and ship it.
  - **NULL** (all four flat) ⇒ the blindness is diffuse; my decomposition is wrong; fall back to intervention 1
    and 3 and record the null. I commit to this outcome.
- **Cost: ~2–3 GPU-hours, zero training, zero renderer.** It is the cheapest experiment in the program that
  can change the closed-loop plan, and it *re-prices every other experiment*.

#### E2b — `LANE-POSE-AUX` (gated on E2a = BLIND-PERCEPTION)
Add an auxiliary head predicting (δ_lat, δψ) from the decoder-facing fmap, trained on the harness's **own
known warps** (free labels, arbitrary quantity, no renderer), with the plan loss unchanged; encoder-in-loop at
the RefcCL-cleared lr, integrity canary gating every checkpoint. **WIN** = `recovery_ratio` rises ≥10× **and**
held-out `band_ade2d` CI∋0 (no cost) **and** canary holds. **BOUND** = `recovery_ratio` unmoved ⇒ the offset is
not recoverable from a monocular frame without lane geometry ⇒ escalate to intervention 3. ~0.5 pod-day.

### 4.3 Intervention 3 — A drivable-corridor channel → a PDM-style anchor scorer

**Mechanism.** The single artifact that four separate walls all reduce to is **lane/road geometry**:
- the LOWOOD-CL results file's own diagnosis: *"the map-free source's inability to say 'you're actually fine
  here' … is the deeper limit"*;
- the tolerance-band re-score, which showed the ADE "trade" was ~74–95 % an artifact of scoring against a
  **knife-edge GT path** rather than a lane;
- ChauffeurNet's on-road loss, R2LPL's drivability scorer, NAVSIM's PDM/EPDM score, and every CBF safety
  filter in §3(e) — **all of them require a geometric constraint set, and none of them requires a renderer**;
- and, if the E2a **BLIND-PERCEPTION** branch fires, the planner's own missing input.

**Build (renderer-free).** Monocular road + lane-marking segmentation from a pretrained model, lifted to BEV
through the ground plane we can now estimate — **GeoCalib landed on 2026-07-25** with a MEASURED qualified pass
(~6.8 % median focal error on comma with known GT, resolution-robust, confidence-gated fallback; correctly
rejects the 120° f-theta fisheye as out-of-model). Product: a per-frame **drivable corridor polygon** in ego
BEV. That single artifact yields (i) a fair road-keeping metric (departure from the *drivable area*, not from
the recorded path), (ii) a **PDM-style anchor score** = the reward for R2LPL/GRPO-style FT, (iii) the geometric
constraint set for a CBF/barrier safety filter, and (iv) a candidate planner input.

⚠️ **PROBE BEFORE BUILDING — this is an absence-at-one-location risk.** The registry states *"the data has no
lead-agent boxes or HD map"* — but **RETRACTION C2 already overturned the first half of that exact sentence**
(`obstacle.offline` carries real 3D agent tracks on **96.90 %** of our corpus; our ingest reads **2 of 36
features**). The map half has **not** been re-probed. **Step 0 is a ~1-day probe of all 36 PhysicalAI-AV
features + the HF dataset card + the loader source for lane/map/drivable-area/traffic-signal annotations.**
If any exists, most of this build disappears and the fair metric is available immediately.

**Committed outcomes for the probe.** **ASSET-EXISTS** ⇒ wire it; the fair metric, the reward and the TL labels
may be nearly free, and a RETRACTION entry is owed for the "no HD map" claim. **ASSET-ABSENT (at ≥3 probe
points)** ⇒ build the segmentation+IPM corridor; budget 1–2 eng-weeks; validate against the recorded ego path
(the ego was, by construction, in its lane ~always) before any metric uses it.

**Honest cost/risk.** This is the only item here that is a *build*, and monocular IPM is wrong wherever the
road is not flat — the same limitation that makes our homography envelope "optimistic." It should be validated
and reported with an explicit failure fraction, not adopted silently.

---

## 5. Renderer or no renderer — the explicit split

| item | needs a reactive photoreal renderer? | can start immediately? |
|---|---|---|
| **E2a `RECOVERY-LOCALIZE`** (probe suite) | **NO** | ✅ yes — ~2–3 GPU-h, zero training |
| **E1a `CL-HORIZON-CURVE`** (measurement) | **NO** | ✅ yes — ~4 GPU-h, zero training |
| **E2b `LANE-POSE-AUX`** | **NO** (labels come from our own warp) | ✅ after E2a |
| **E1b `CL-FAILGATE`** | **NO** | ✅ after E1a/E2a |
| **Anchor-vocabulary re-fit** | **NO** | ✅ any time (anchors are a 0-param buffer) |
| **Denoise-step increase at inference** | **NO** | ✅ immediately if E2a-D fires |
| **Intervention 3 corridor channel** | **NO** (segmentation + IPM, not rendering) | ✅ after the 36-feature probe |
| **Traffic-light labels + TLC on real data** (§6) | **NO** (VLM over real frames) | ✅ yes |
| **Reactive-agent collision / off-road rates** | **YES** — this is the only truly renderer-bound item | ❌ Path 3 (`obstacle.offline` + IDM overlay) or AlpaSim |

**Everything in the ranking above is renderer-free.** The renderer remains binding for exactly one thing —
**reactive-agent collision safety (B)** — unchanged from the 07-24 survey.

---

## 6. Traffic lights — zero measurement, zero labels, and the cheapest real path

**Where we stand, stated honestly.**
- `MEASURED (synthetic fixtures only)`: the **TLC** metric and the **SC-14** scenario exist and are
  test-covered (8 analytic metric fixtures + 16 scenario tests, 24 green; full suite 836 passed). The design
  oracle separates cleanly: `rule_barrier` **TLC 1.000 / red-run rate 0.0**; `soft_prior` **TLC 0.000 /
  red-run rate 1.0**, and the soft prior's line-crossing speed grows **2.4 → 9.0 m/s** as apparent
  cross-clearance opens, while the barrier never enters on red.
- **We have ZERO measurement of REF-C's signal behaviour and ZERO signal labels.** TLC on real data is
  currently blocked on per-step signal-state telemetry; the intake's proposed unblock was a MetaDrive
  signalized junction (renderer-gated on the dev box).

**The cheapest real path — and it is cheaper than the intake assumed, because the labels are nearly already
there.** `MEASURED` (this session, reading `…/2026-07-21-vlm-production-semantic/val_full.jsonl`): our
production VLM pass (`nvidia/Cosmos-Reason2-8B`) **already emits traffic-light state in its free-text
evidence** — e.g. *"the traffic light is red and applies to the ego vehicle"*, *"the traffic light remains
green"*, *"turns green"* — on **18 of 30** val records, unprompted, as a by-product of route labeling. It is
unstructured today, so it is not a label; but the pipeline, prompts, enum validation, cross-validation
protocol and audit tooling all already exist.

**Recommended sequence (renderer-free throughout):**
1. **Add a `SIGNAL_STATE` field to the existing VLM production pass** — enum `{none, green, yellow, red,
   red_to_green, green_to_yellow, unknown}` + `applies_to_ego` + `stop_line_distance_bucket`, reusing the
   `enum_ok` validation and the existing prompt-A/B harness. Cost: one prompt revision + one re-run of a pass
   we already run. This is the **only** item that unblocks everything else.
2. **Human-audit ~200 frames** to get a real precision/recall for the signal label. ⚠️ **Mandatory, because of
   RETRACTION C4/C6**: our last VLM accuracy headline ("89.3 % turn detection") did not reproduce (80.6 %) *and*
   was agreement, not recall — on a 74 %-straight corpus a constant answer scores ~74 %. **Report per-class
   recall on a class-balanced audit set, never aggregate agreement.** A "red" recall number is the only one
   that matters for a safety metric.
3. **Run TLC on real footage** — the first non-synthetic signal number in the program, and the first
   measurement of REF-C's actual red-light behaviour. `HYPOTHESIS`, worth pre-registering: REF-C will be
   **near-chance** on red, because the corpus is 0 % signal-labeled, the planner has no signal input, and
   `PUBLISHED` evidence says models "find it hard to correctly identify traffic light states when labeled
   training data is lacking."
4. **Then, and only then, add the ingredient.** `PUBLISHED`, this is settled in the literature: an **explicit**
   traffic-light/sign recognition module cuts **red-light violations −64 % and stop-sign violations −81 %**
   versus the implicit-pixel baseline (arXiv:2511.14391); collisions and red-light running are the two
   dominant E2E failure modes and both trace to noisy monocular state estimation. The minimal ingredient is
   therefore an **explicit signal-state input** (a head or a distilled VLM label), **composed with a rule
   barrier at the stop line** — which our own oracle MEASURED as the difference between TLC 1.0 and 0.0. Do
   **not** attempt to obtain signal compliance from a soft cost inside the diffusion prior; we have measured
   that exact design scoring 0.000.
5. **Evaluation target.** `PUBLISHED`: NAVSIM v2's **EPDMS** already includes traffic-light compliance
   alongside lane-keeping and direction compliance; Bench2Drive scores traffic-sign compliance as a named
   multi-ability axis. Our TLC is compatible in spirit; align the reducer names so a future comparison is
   possible.

**Bound:** none of this gives *reactive* signal behaviour (yellow-phase dilemma-zone decisions under following
traffic) — that needs agents, i.e. the renderer. It gives red-run rate, stop quality and phantom-braking on
real footage, which is the honest 80 %.

---

## 7. Honest bounds — what my recommendation does NOT fix

1. **It does not produce a safety rate.** Everything above is **within-instrument RELATIVE**, low-OOD
   **lane-keeping**. Off-road and collision are structurally unmeasurable on a map-free, agent-free source.
   Intervention 3 supplies drivable-area (hence off-road) but **never collision** — that stays renderer-bound.
2. **It does not fix reactive-agent safety (B).** Unchanged: `obstacle.offline` + IDM overlay (Path 3) or
   AlpaSim at 3.2× reconstruction OOD.
3. **The horizon extension buys failure density at the cost of fidelity.** The homography is ground-plane-only
   and the lateral envelope is *optimistic*; at 8–10 s the ego may leave ±3 m/±12°. E1a reports `frac_inside`
   precisely so this is measured, not assumed — and HORIZON-NULL is a committed outcome that kills the idea.
4. **Data scale caps multimodality.** `PUBLISHED`: diffusion planners show mode collapse below ~100 K frames
   and only reach clear multimodality at 20–70 M. At ~473 K frames (~1.8 M in the new corpus) **we are 1–2
   orders of magnitude short**. No objective fix substitutes; this bounds how much of the ~92 % irreducible
   oracle gap is reachable at all.
5. **Junction data is genuinely scarce.** ~13–22 distinct real junction episodes. A failure-gated fine-tune
   concentrates on junctions, which **increases memorization risk** — the exact reason the Gate-1 clean run
   held. The n=40 cross-fit protocol is mandatory, not optional; a single favourable split has already
   produced one retracted headline in this direction.
6. **The stratum reframe is a HYPOTHESIS built on non-separated deltas.** The junction improvements in
   LOWOOD-CL are directional (CI∋0). I am proposing an experiment, not asserting a result — and this doc will
   be wrong if E1b returns BOUND-CONFIRMED.
7. **It does not settle whether our WM can be the simulator.** The published latent-rollout successes (MAPLE,
   WorldRFT) roll out **structured agent/map tokens**, not reconstruction latents; our own `DAGGER_HURTS`
   result stands against free imagination. Nothing here proposes closing that loop.
8. **Traffic lights: no reactive behaviour, and the label is a VLM** — an unaudited VLM label is not a
   measurement, and this program has already retracted one VLM accuracy headline.
9. **I did not re-verify every external magnitude.** See §9.

---

## 8. Deliverable manifest

| # | artifact | where it lives | status |
|---|---|---|---|
| 1 | **This research + design doc** | `TanitAD Research Hub/Architecture & Inference/Research/2026-07-25-closed-loop-diffusion-planner/CLOSED_LOOP_PLANNER_RESEARCH.md` (repo) | **STAGED** (`git add`), not committed, not pushed |
| 2 | Verdict (§1), the two-confound re-read (§2), ranked design (§4), renderer split (§5), traffic lights (§6), bounds (§7) | this file | complete |
| 3 | Pre-registered experiments E1a / E1b / E2a / E2b + the intervention-3 probe, **both outcomes committed in advance** for each | §4 | complete, **not launched** |
| 4 | Source artifacts read | in-repo paths cited inline §1–§6 | unchanged, read-only |

**No GPU launched. No pod touched. No training started. No commit, no push. No sub-agent spawned**
(fan-out discipline observed — all research performed in-agent via WebSearch/WebFetch).

### Escalations (need an owner/decision — not a paragraph in a doc)
1. ⭐ **Every standing low-OOD closed-loop number is a 2-second number** (`K = max(WP_STEPS) = 20`, MEASURED
   source-read). `MODEL_REGISTRY`, `LEADERBOARD.md` and `LOOP_STATE` quote REF-C's 0.0134 / 0.564 and the
   junction strata without a horizon qualifier. **Registry/leaderboard owner: add the horizon to the metric
   name or its caption.** I do not edit those files.
2. ⭐ **The LOWOOD-CL "BOUND" verdict is stratum- and horizon-confounded** (MEASURED: unstratified training on
   a 59.6 %-lane-keep corpus; junction stratum improved on all three metrics). A **C6-class RETRACTION_LOG
   entry is owed** if E1a returns HORIZON-BOUND. RETRACTION_LOG owner.
3. **The "no HD map" claim in `MODEL_REGISTRY` §(G1 cost note) has never been second-probed**, while the
   adjacent "no lead-agent boxes" half of the same sentence was already overturned as C2. **Data-engineering
   owner: probe all 36 PhysicalAI-AV features for lane/map/drivable-area/signal annotations.** This is the
   textbook absence-at-one-location risk and it gates intervention 3's entire cost.
4. **The VLM production pass already emits traffic-light state in free text** (MEASURED, 18/30 val records)
   but does not capture it as a field. **Data-engineering owner: add `SIGNAL_STATE` to the next pass** — it is
   a prompt change on a pipeline we already run, and it unblocks the program's first real signal measurement.

---

## 9. Sources

**Ours (`MEASURED`, artifact paths):**
`…/incoming/2026-07-23-lower-ood-closedloop-source/{lowood_closedloop.py,lowood_probe.py,P1_DECISION_GRADE_FINDINGS.md}`
(the `K = max(WP_STEPS) = 20`, `DT = 0.1` source-read) ·
`…/incoming/2026-07-23-lowood-lanekeeping-refc/LANEKEEPING_REFC_REPORT.md` (the n=40 arms + strata) ·
`…/incoming/2026-07-24-refccl/{RESULTS.md,LOWOOD_CL_TRAIN_RESULTS.md,TOLERANCE_RESCORE_RESULTS.md,POWERED_DEPARTURE_RESULTS.md,lowood_cl_train.py}`
(the four-experiment arc + the `max_windows=None` source-read) ·
`…/incoming/2026-07-23-refc-planner-closedloop/{provenance.json,PRE_REGISTRATION.md,DESIGN.md}` (`recovery_ratio
0.0074 [0.0036,0.0115]`, n=881) · `…/incoming/2026-07-23-gate0-freefloor/` (Gate 0 / 0b) ·
`…/incoming/2026-07-23-freefloor-rung3-wm-mpc/` · `…/incoming/2026-07-22-alpasim-closedloop-evalpod/{flagship_vs_refc_suite_*,GATE1_ROLLOUTS_NOTE.md}`
(n=12 paired; plan_xte 0.57→12.98) · `…/incoming/2026-07-23-dagger-closedloop-aware/` (`DAGGER_HURTS`) ·
`…/incoming/2026-07-24-traffic-light-scenario-metric/{NOTE.md,traffic_light_oracle_results.json}` (TLC + SC-14) ·
`…/incoming/2026-07-21-vlm-production-semantic/val_full.jsonl` (the unprompted traffic-light free text) ·
`…/incoming/2026-07-24-parity-corpus-profile/CORPUS_PROFILE.md` (59.6 % lane-keep) ·
`…/incoming/2026-07-25-geocalib/{NOTE.md,VALIDATION_REPORT.md}` ·
`Benchmarks & Eval/Research/2026-07-17-openloop-l2-egostatus-shortcut.md` (0.658 m no-vision ceiling; 73.9 % straight) ·
`Architecture & Inference/Research/{2026-07-23-closed-loop-wm-training-verdict.md,2026-07-23-planner-is-the-bottleneck.md,2026-07-24-low-ood-closedloop-renderer.md}` ·
`Project Steering/{MODEL_REGISTRY.md,RETRACTION_LOG.md,LOOP_STATE.md,GATE_PROTOCOL.md}`.

**External (`PUBLISHED`):**
- *Anchored / truncated diffusion planners:* DiffusionDrive arXiv:2411.15139 (CVPR'25) · DiffusionDriveV2
  arXiv:2512.07745 (intra/inter-anchor GRPO; 91.2 PDMS / 85.5 EPDMS) · AnchDrive arXiv:2509.20253 ·
  "Unleashing the Potential of Diffusion Models for E2E AD" arXiv:2602.22801 (mode collapse ≤100 K frames;
  multimodality at 20–70 M; SR 59.00→79.53).
- *Closed-loop-consistent training:* CAT-K arXiv:2412.05334 (CVPR'25 Oral, NVlabs/catk) · RoaD arXiv:2512.01993
  (Physical AI AV NuRec; +41 % DS, −54 % collisions, off-road 0.283→0.210; sim2sim 0.58→0.35) ·
  **R2LPL arXiv:2606.30537** (rollout-retrieval lifelong learning; nuPlan Test14-hard 60.67→83.51) ·
  MAPLE arXiv:2605.14201 (latent multi-agent rollouts, no external simulator; Bench2Drive DS 85.2 / SR 67.1 %,
  +12.3 DS ablation) · WorldRFT arXiv:2512.19133 (latent WM + GRPO; nuScenes collision 0.30→0.05 %) ·
  Latent Policy Barrier arXiv:2508.05941 · ChauffeurNet arXiv:1812.03079.
- *Theory:* Ross & Bagnell / DAgger — the `J(π) ≤ J(π*) + T²ε` compounding bound (AISTATS'11) ·
  "Is Behavior Cloning All You Need? Understanding Horizon in Imitation Learning" arXiv:2407.15007.
- *Ego-status shortcut:* "Is Ego Status All You Need for Open-Loop E2E AD?" arXiv:2312.03031 (CVPR'24) ·
  AD-MLP arXiv:2305.10430.
- *Temporal consistency / replanning:* CoPlanner arXiv:2509.17080 · Diffusion Forcing Planner arXiv:2606.11019 ·
  Temporally Decoupled Diffusion Planning arXiv:2603.25462 · Contractive Diffusion Policies arXiv:2601.01003.
- *Guided / safety-filtered diffusion:* CBF safety filter with arbitrary road-boundary constraints
  arXiv:2505.02395 · PC-Diffuser arXiv:2603.10330 · DualShield arXiv:2601.15729 · HOCBF+Diffuser at signal-free
  intersections arXiv:2412.00162 · Diffusion-Planner arXiv:2501.15564 (ICLR'25).
- *Traffic lights:* "Enhancing LLM-based Autonomous Driving with Modular Traffic Light and Sign Recognition"
  arXiv:2511.14391 (**−64 % red-light, −81 % stop-sign** vs implicit baseline) · Implicit Affordances
  arXiv:1911.10868 · NAVSIM/EPDMS (traffic-light compliance in v2) · Bench2Drive arXiv:2406.03877 ·
  Bench2Drive-Robust arXiv:2605.18059.
- *Long-tail curation:* WOD-E2E arXiv:2510.26125 · Semantic-Drive arXiv:2512.12012 · Hidden Biases of E2E
  Driving Datasets arXiv:2412.09602.
- *Calibration:* GeoCalib (ECCV 2024, `cvg/GeoCalib`).

⚠️ **Citation hygiene.** Individually **fetched and read** this session: arXiv 2508.05941, 2605.14201,
2512.19133, 2512.01993, 2602.22801, 2512.07745, 2606.30537. All other ids come from live search-result titles
and URLs — the ids are reliable, the abstracts were **not** individually re-fetched, so treat their
**magnitudes** as directional single-source. Several ids postdate this agent's training cutoff and were
verified via live search only. The Ross & Bagnell `T²ε` statement was confirmed through multiple secondary
sources this session, not from the primary PDF. **No external magnitude in this doc is used to decide a
GPU-day on its own** — every proposed experiment is gated on one of OUR measurements.
