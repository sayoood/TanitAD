# PLAN — v8F and refav1 with H-JEPA's proven edges: the fastest path to positive, dominant driving

**Written:** 2026-10-17 (Europe/Berlin), by a research–engineering session started by the PI.
**Inputs:** the handover `Project Steering/Handover/2026-10-15-refa-refd-flagship/` (A–D + HANDOVER), the
landed tip `e8cadf2` (fetched 2026-10-17; 40 commits after the handover tip `460508ed`), and **H-JEPA**,
arXiv **2610.06805** (Zhang, Terver, Rabbat, LeCun, Balestriero, 5 Oct 2026). The primary is banked in the Library
as `lib:2610.06805` (PDF 14.9 MB, sha256 in `library.json`).
**Status of this document:** a PLAN plus proposals. The architecture change (§4) needs the PI's go before
any training. Everything in §7 "starts now" is 0-GPU work that needs no decision.
**Evidence classes:** M = MEASURED (ours, artifact) · P = PUBLISHED (library key) · P-sec = published,
read only in a secondary source · I = INHERITED (not re-verified here) · E = ESTIMATED · H = HYPOTHESIS.
**Tiers:** T0 = teacher-forced WM diagnostic · T1 = self-action OPEN loop on logged frames · NAVSIM = open-loop
plan scoring · T2 = closed loop (not provisioned).

> **RULE ZERO governs every line below.** No rung ends at a refutation. Each failed bar names its next lever
> (§6), and that lever runs in the same work package. A rung is done only when it clears its bar, or when every
> remaining lever is blocked and named.

---

## 0. The plan in twelve lines

1. **Where we stand (M):** no TanitAD arm beats every trivial floor. The best driver on PhysicalAI is **refcv7-50,400**.
   It beats the echo control at 6 s by **−0.98 m** (T1), but not at 2 s.
   * At its 15k milestone, refcv8 ties refcv7 on ADE, LON and TAC. It is **worse on lateral heading** (+0.071°, PROVISIONAL), and the loss sits in **sharp turns**.
   * The best NAVSIM driver is **REFe P2: 88.82 PDMS**. Its gain replicated on seed 1, but TTC is adverse.
   * ~~No arm has been read against the strongest kinematic floor, **P10**: 6 s ADE 2.93 m, versus 5.36 m for the echo.~~ **RETRACTED 2026-10-17 (§14):** that 5.36 m echo integrated a steer angle as a curvature; in the correct unit it is 2.838 m.
   * **NOW READ (S1, M, T1 open loop, eval139 S6, paired episode-cluster bootstrap B = 2,000):** `refcv7 − P10` ADE 0–6 s = **−0.4061 m [−0.5936, −0.2161]** (inference seed 0) / −0.4054 (seed 1), lateral. ⚠️ At 0–2 s **P10 wins**: +0.0276 [+0.0163, +0.0398], on speed and along-track. `P10 − ha0_ext` = −0.5743 m @ 6 s, −0.0378 m @ 2 s, both separated.
2. **Our most persistent failure is the world model ignoring actions (M).**
   * In the v7-tiny census, **0 of 32 arms** change more than 0.1 % under action shuffle.
   * refav1's WM varies along κ by **1.99e-06**.
   * **97 %** of the conditioning response is the speed channel.
   * The imagination planner built on top of these WMs failed: refav1 FAIL-WORSE.
3. **H-JEPA explains this failure class and fixes it (P, outside driving).**
   * On diverse-scene data, prediction plus a regularizer on the marginal distribution (SIGReg or VICReg) has an **optimum the objective cannot exclude**: the encoder keeps scene identity and the predictor becomes the identity. Action sensitivity is **< 0.05 in 15 of 15 seeds**.
   * An **inverse-dynamics (IDM) term** is an O(1) barrier against this collapse. With it, sensitivity is **0.280**.
   * SIGReg alone restores representation rank, but **not planning fidelity**.
   * Our v7 recipe is O5 L1 + O6 SIGReg. That is exactly the configuration the theory says collapses.
4. **H-JEPA's other two proven edges are exactly what refav1's planner lacked (P):**
   * **Goal costs in an abstract latent.** Changing only the cost space lifts AntMaze from 23.3 % to 31.3 %.
   * **Subgoal decomposition.** Cost monotonicity is 0.89–1.00, against 0.48–0.75 for a full-horizon goal cost.
   * Three levels take AntMaze from **18 % to 73 %** with *less* planner compute.
   * refav1, measured: the goal term was negligible, the κ argmin was a float tie on 45 % of windows, and the vocabulary was the bottleneck.
5. **The thesis survives, the recipe changes.** The 4-brain hierarchy becomes an **H-JEPA stack**: each brain has its own latent, its own stride, its own predictor, plus SIGReg **and IDM** at every level. Planning runs top-down, and each level's predictions are the subgoals for the level below.
6. **Fastest path (Rule Zero):** do **not** build a new imagination planner from scratch.
   * Put the H-JEPA stack on the chassis that already drives (refcv8, REF-C). That definition is **v8F**.
   * Climb a gated ladder: **α** (prior swap, H-JEPA WM as auxiliary) → **β** (the WM scores the proposal fan in the tactical latent) → **γ** (the WM refines the chosen plan by gradient, aimed at the sharp-turn under-correction) → **δ** (NAVSIM/REFe port and RL-in-imagination).
7. ~~**The prior swap is the largest measured lever we have not used (M):** `damp50 − ha0_ext` = −1.90 m @ 6 s.~~ **RETRACTED 2026-10-17 (§14): a unit artifact.** In the correct unit, `damp50 − ha0_ext` = +0.0956 [−0.0761, +0.2901] (n.s., refav1 grid), and on eval139 damp50 is **worse at 2 s** (+0.1239, separated). ⇒ damp50 is dropped as a prior.
   **What survives, smaller and measured (S1):** `P10 − ha0_ext` = **−0.5743 m [−0.7744, −0.3753] @ 6 s** and **−0.0378 m @ 2 s**, both separated. The 2 s side is exactly where refcv7 loses to P10. refcv7/8 still sit on `ha0_ext_pose`, so **the prior swap to P10** remains v8F-α's first arm. It is now a *2 s / longitudinal* lever, not a −1.90 m one.
8. **refav1 becomes the frozen-encoder H-JEPA reference (refav2).**
   * A dev-box mini-arm on pooled frozen DINOv3 tokens decides **frozen vs LoRA vs trained** before any big spend.
   * Pooling also shrinks the 303 GB storage blocker to **~19–76 GB**.
9. **Frozen vs unfrozen, our evidence (M):**
   * Fully frozen DINOv3 + a WM adds nothing beyond kinematics.
   * A from-scratch ViT trunk learned 91× less scene content than DINOv2-B.
   * **Frozen DINOv3-L + LoRA is our best driver anywhere (REFe, 88.8 PDMS).**
   * ⇒ **Default trunk: frozen foundation encoder + LoRA + trainable adapter**, decided by the trunk ladder (§3).
10. **Five 0-GPU streams start today:**
    * **S1** P10/damp50/kd_x floors on refcv7 and refcv8;
    * **S2** the H-JEPA collapse diagnostic on our own checkpoints;
    * **S3** the `hjepa` module, its tests and gate entries;
    * **S4** pooled DINOv3 encodes for the refav1 mini-arm, which needs a GPU when one frees;
    * **S7** banking the competitor primaries and T2 feasibility.
11. **Thor (free ≈ 2026-10-20) runs the v8F-α tiny rung first:** about 7 arms × 4.5 h, two training seeds each, plus a deliberate-regression arm. A full run follows only after rung α clears its bars.
12. **PI decisions:** six, each with a recommendation (§10). The main one is **v8F = refcv8 chassis + H-JEPA stack + kinematic prior**.

---

## 1. H-JEPA — what it proves, and what it does not (all P, `lib:2610.06805`, unless marked)

| # | edge | evidence in the paper | what it does **not** prove |
|---|---|---|---|
| E1 | **Distinct latent per level**: each level has its own encoder (an MLP over the latent of the level below), its own action encoder (a pooled **macro-action**), its own predictor, a coarser stride (s = 2–3) | Visual AntMaze: LeWM (flat) **18.0 ± 3.5 %** → 2 levels **39.3 ± 3.7 %** → **3 levels 73.3 ± 3.5 %** (Tab. 1, Fig. 6). On AntMaze, 3-level H-JEPA ≈ **2× HWM** (shared-latent hierarchy). Gains on FourRoom and Cube. Push-T degrades at ≥ 3 levels | Not driving. Simulated tasks with a fixed camera, 3 seeds. Gains need a temporal-frequency gap ≥ 2× between fast and slow factors (Fig. 5) |
| E2 | **Abstract-level goal cost**: score a rollout by its distance to the goal in an *upper* latent | Change only the cost space, keep the dynamics fixed: 2-level L1 **23.3** → L2 projection **31.3**; at 4 levels, **10.0 → 26.7**. Level-1 distance is "nearly constant away from the anchor", so a gradient planner gets no signal (Fig. 7) | Needs a goal *observation*. Driving has none at inference; our substitute is a **predicted goal latent** (§4.3) |
| E3 | **Subgoal decomposition**: each level tracks the predicted states of the level above | Subgoal-tracking cost monotonicity **0.89–1.00**, vs full-horizon goal cost **0.48–0.75** (Tab. 10). Fewer rollout steps per candidate ⇒ success/compute Pareto shifts up and left | Planning is by gradient descent in simulation, then executed closed loop; the DROID result is offline only |
| E4 | **End-to-end training of all levels**; upper-level losses back-propagate into lower encoders; **no EMA, no stop-grad, no pretrained weights**; SIGReg at every level, grouping (T, N, D) | Level-1 LR 5e-5, upper levels 1e-5, AdamW, bf16, clip 1.0; "neither term degenerates at any level". On DROID, training the upper level jointly does not degrade the lower one | Small encoders (ViT-Tiny/S); not shown with a 100 M driving trunk |
| E5 | ⭐ **Slow-feature collapse and the IDM barrier.** Proposition 1: any regularizer on the *marginal* law (SIGReg, VICReg, whitening) admits an encoder that reads only the episode background, plus an identity predictor, at zero loss. Theorem 1: the objective selects by predictability, so this collapse is an optimum. Proposition 3: an IDM head regressing the action from (z_t, z_t+1) puts an O(1) floor under it | DROID: without IDM, planning ≈ the do-nothing floor at every SIGReg weight; with IDM (weight 100, SIGReg 0.08), fidelity is about 30 %. SIGReg ladder without IDM: action sensitivity **< 0.05 in 15/15 seeds**, within-episode variance share ρ **0.008–0.016**. IDM-on: sensitivity **0.280**, ρ **0.12**. SIGReg alone restores effective rank (1–14 → high) but **not** fidelity (Fig. 17). Target detachment makes no difference. IDM and SIGReg weights must be tuned *jointly* at level 1; level 2 is insensitive (Fig. 18) | Their collapse is to a *per-episode-constant* background. In driving the scene flows past within an episode, so our version may be **"advect the scene at the measured speed"** instead, which is consistent with our 97 % speed-channel finding. ⇒ S2 measures both (§7) |
| E6 | Frozen-encoder WMs plan worse per FLOP | DROID: DINO-WM and V-JEPA 2-AC "need orders of magnitude more compute for a lower fidelity" (Fig. 9c) | Weak comparison: native 1 fps, 15 clips, public checkpoints. It is **not** a frozen-vs-trained test at equal data |
| E7 | Data geometry | Deeper levels need longer clips; a 3-level model sees only 58 % of LeWM's transitions on Push-T (Tab. 3) | — |

**Read across to driving (H).** Driving has a large timescale gap: steering and pedal jitter are fast; lane geometry, route and the lead vehicle's state are slow. That gap is E1's precondition. Our 20 s clips at 10 Hz give ≈ 200 level-1 steps. A 3-level stack at stride 3 needs about 1/9 of that, so E7 does not bind.

---

## 2. Why H-JEPA is the right lever for us — mapping its mechanisms onto our measured failures

| our measured failure (M) | H-JEPA mechanism | consequence in this plan |
|---|---|---|
| **The WM ignores actions.** Census: 0/32 arms > 0.1 % nrmse under action shuffle (max 0.0919 %), REG §13.0c. refav1 κ-variation **1.99e-06** (`AI/2026-09-04-refav1-cost-scale/`). 97 % of the conditioning response is speed (`D-V7F-CONDITIONING-IS-SPEED`) | E5: prediction + SIGReg selects a slow-feature / identity-predictor optimum; **IDM is the barrier** | **IDM at every level, from day 1.** Gate **BAR-HJ-SENS** (§8) before any WM sits in a decision path |
| The rdw8p30k recipe has **healthy rank** (participation 25.58) yet is action-deaf | E5, Fig. 17: SIGReg keeps rank high but fidelity stays at the floor without IDM | "Healthy rank" is no longer accepted as WM health. **Action sensitivity is the gate** |
| refav1's goal term is negligible; the κ argmin is a **float tie on 45.4 %** of windows; **jerk** is the binding penalty; the vocabulary κ ∈ {0, ±0.08} cannot express R 100–1000 m (`AI/2026-09-05-refav1-goal-margin/`) | E2: in the fine latent the distance is flat away from the anchor; an abstract latent gives a graded cost. E3: subgoals give a monotone cost | Score and plan in the **tactical latent** toward a **predicted goal latent**. A hand-made curvature vocabulary is no longer the decision space |
| The WM-as-selector (roll-consistency) was **+5.98 m worse** | It was an action-deaf WM scoring in the fine latent | The "never a selector" ban is **conditional on action-deafness**. It is lifted only by BAR-HJ-SENS plus the β-rung bar, never by argument |
| The frozen refav1 trunk carries no info beyond 8 t0 kinematic features: `P9 − P10` **+0.0987 m @ 6 s** (worse) | The trunk was trained to predict *patch* features in one latent space, at three rates (≈ HWM, not H-JEPA), with no IDM | The refav1 mini-arm (RA-1) re-asks the question with H-JEPA levels plus IDM, against P10 |
| refcv8's lateral regression sits in **sharp turns** (slope shortfall −0.049 [−0.087, −0.018]). It is mostly **CANDIDATE** (−0.035) and **refinement under-rotates** (arm difference −0.065 [−0.092, −0.040]) (`H-R8-TURN-UNDERCORRECT-1`, GAC 2026-10-17, PROVISIONAL) | E3: gradient refinement of a plan toward subgoals in an action-sensitive latent | Rung γ targets exactly this: a WM-gradient refiner on the chosen candidate, read on the sharp-turn stratum |

---

## 3. Frozen vs unfrozen encoders — what the evidence says, and the experiment that decides it

| trunk regime | our evidence | published evidence |
|---|---|---|
| **Fully frozen foundation features** (DINOv3-L, patch tokens) + a trained WM | Content IS there: linear lead-gap R² **+0.4531**, vs **+0.0130** for v6F's trained trunk (E-TRUNK-2, M · T0). But refav1's WM on top adds **nothing** beyond kinematics at 2 s or 6 s (trunk probe, M · T1-equiv) | H-JEPA E6: frozen-encoder WMs plan worse per FLOP (weak comparison). DINO-WM works in sim (P-sec) |
| **Frozen + LoRA + trainable adapter** | **REFe:** frozen DINOv3 ViT-L/16 + LoRA, 322 M total / 19 M trainable. **88.25 → 88.82 PDMS** navtest; the best driver in the programme (REG §14.2, M · NAVSIM) | DriveZero 94.55 with the same pattern (P-sec, `2609.06055`) |
| **Trainable CNN from ImageNet init** (ResNet-101) | refcv7: navtest **70.20** PDMS zero-shot; beats the echo at 6 s (M) | DiffusionDrive / TransFuser lineage |
| **Trainable ViT from scratch or transplant** | v6F trunk: **91× less** lead-gap content than DINOv2-B through the same pool (S5). The DINOv3 → ViT transplant **loses** separated scene and ego-speed content (S6). Freezing the v7 encoder is **degenerate** (E-DEC-64) | H-JEPA trains small ViTs from scratch, but on single-scene sims and DROID |

**Synthesis (H, to be decided by measurement).** At our data scale (4.6k clips), training an encoder from scratch loses. Using frozen features with no adaptation keeps the content but does not turn it into a trajectory. **Frozen plus low-rank adaptation (LoRA) plus an H-JEPA stack trained end-to-end through LoRA and the adapter** is the regime that both our best result (REFe) and H-JEPA's E4 point to. A ViT-B/16 DINOv3 + LoRA keeps v8F inside the **sub-300 M** budget; ViT-L (303 M alone) does not.

**The deciding experiment is the TRUNK LADDER, run in two places with one bar.**

* **Cheap rung, on the dev box (RA-1, §5).** Pooled frozen DINOv3 tokens feed (a) a direct readout, (b) H-JEPA without IDM, (c) H-JEPA with IDM. Each is read against P10 at 6 s.
  * Positive ⇒ frozen + adapter suffices for the refav2 line.
  * Negative with sensitivity passing ⇒ adaptation is needed.
* **Driving rung, on Thor tiny rung (v8F-α′).** On the refcv8 chassis, compare:
  * `T-A` ResNet-101 trainable (today's chassis);
  * `T-B` DINOv3 ViT-B/16 frozen + LoRA r 32;
  * `T-C` DINOv3 ViT-B/16 fully frozen.
  
  Run two training seeds each, with identical heads. Read the four families against F, the rung's own training-seed floor.
  * Ships the trunk that wins. Ties go to `T-A`, because it is already gated.

---

## 4. v8F — definition (PROPOSAL; architecture change ⇒ PI go required)

**v8F = "HJ-Drive": the 4-brain hierarchy as an H-JEPA stack, on the proven REF-C chassis, with the measured
kinematic prior.** It extends the handover's option F1 (C §C.6) with H-JEPA.

### 4.1 Components

| brain / part | v8F choice | why (evidence) |
|---|---|---|
| **Chassis** | `refc_v3_train.py` / `refc_v3.py` at refcv8: 416×1024 cylindrical, BEV lift, 10 cm map head, A9 box head, 117 anchors, v9 per-frame tactical conditioning, `--w-r8-listwise 0` | Every measured driving gain is on this chassis. It has a working gate profile and the NAVSIM bridge |
| **Warm start** | refcv8 final EMA if it clears BAR-R8-E1/E3/NR, else refcv7-50,400. Every new path is zero-gated so step 0 reproduces the parent bit for bit | refcv8 §3.1 pattern |
| **Brain 0: perception trunk** | `T-A` ResNet-101 today; `T-B` DINOv3-B/16 + LoRA if the trunk ladder says so (§3) | §3 |
| **Brain 1: operative WM** (H-JEPA level 1) | z¹ = attention-pooled BEV/scene tokens (16–32 tokens × 256 d) at **Δt = 0.5 s**. Action = the (a, κ) block of 5 × 0.1 s. Causal-transformer predictor with **AdaLN action conditioning** (not FiLM-by-sum: the paper's H26 lesson on magnitude swamping). **SIGReg (T,N,D) + IDM** | E1, E4, E5 |
| **Brain 2: tactical WM** (H-JEPA level 2) | z² = MLP(z¹) at stride **s = 2 (1.0 s)**; macro-action = a learned pool of 2 level-1 action embeddings (≈ a manoeuvre embedding); own predictor; SIGReg + IDM on the macro-action (detached target) + action-embedding SIGReg (weight 0.005, as on DROID). **The v9 tactical label heads read z²** (R3), and nav + turn distance enter z² and z¹ (R2) | E1, E5; it makes the tactical latent supervised **and** predictive |
| **Brain 3: strategic** (level 3, s = 3 → 3 s) | **Built, OFF** (R6, PI 2026-09-19). Reported as PAUSED | R6 |
| **Residual prior** | Replace `ha0_ext_pose` (= the echo) with the **learned kinematic prior P10** (8 t0 features, fitted on TRAIN, correct steer unit). ~~fallback `damp50`~~: dropped, because damp50 is worse than `ha0_ext` at 2 s. Prior dropout kept | **MEASURED on eval139 (S1):** `P10 − ha0_ext` −0.5743 @ 6 s / −0.0378 @ 2 s, separated. ~~`damp50 − ha0_ext` −1.90 m @ 6 s; P10 2.93 vs 5.36 m (refav1 grid)~~ is **RETRACTED** as a unit artifact (§14) |
| **Max speed (R1)** | road-type label input + **hard cap on the EMITTED plan** | `SPEC_REFCV7.md` 26.1 |
| **Params** | ≈ 100.6 M (refcv8) + ≈ 6–12 M for the WM levels (E) | sub-300 M |
| **Step-time lever from H-JEPA** | future-frame targets computed under **`no_grad`** (paper K.1: target detachment changes nothing). Training cost ≈ one extra trunk forward per target frame, no backward ⇒ **≈ +10–15 % step** (E), instead of the handover's +20–45 % | keeps v8F near the PI's 10.5 s/step cap; **G-CLOCK measures it before launch** |

### 4.2 Training objective (per step)

The objective is the refcv8 loss (unchanged) plus a WM term:

**L = L_refcv8 + λ_wm · Σ_ℓ∈{1,2} [ L_pred^ℓ + β_ℓ · SIGReg(Z^ℓ) + γ_ℓ · L_idm^ℓ ] (+ λ_tac · L_tac-labels(z²))**

* `λ_wm` ramps from 0 over 1,000 steps.
* β and γ are tuned **jointly on level 1 only** (paper Fig. 18). The level-2 values are copied from level 1.
* The level-2 learning rate is 0.2× the level-1 rate (paper E4).

### 4.3 How the WM is allowed into decisions — the ladder (each rung gated; §6 has the fail branches)

| rung | the WM's role | entry condition |
|---|---|---|
| **α** | auxiliary only (representation shaping); selection unchanged | gate PASS (G-DVB / G-LIVE show gradient into all WM groups) |
| **β** | **hierarchical scorer**. For each of the 117 candidates, roll level 1 under the candidate's action blocks, lift with E², and score the distance to a **predicted goal latent ĝ²(+6 s)** plus subgoal monotonicity. The score is added to the learned selector through a scalar gate initialised at 0 | **BAR-HJ-SENS passed** on α's checkpoint; **goal admissibility** (below) |
| **γ** | **gradient refinement**: k = 5–10 AdamW steps on the selected plan's actions toward the level-2 subgoals (paper §2.2, K = 1). Clipped by Kamm μ 0.7 and the R1 cap | β ≥ α on every family; aimed at the sharp-turn stratum |
| **δ** | port to NAVSIM/REFe as an extra scorer component; RL-in-imagination on the action-sensitive WM | γ passes; an A40 slot (PI) |

**Goal-latent admissibility (binding rule, 2026-08-03).** ĝ² is predicted from vision plus nav only. It must not take the situation classifier's posterior, argmax or embedding. Check: ĝ²'s inputs are `{trunk features, nav token, turn distance}`, asserted by a `assert_no_future_at_inference`-style signature test, and its probe against the classifier's logits is reported. Training target = the level-2 latent of the frame **+6 s ahead** (privileged at label time, which is allowed).

### 4.4 R1–R6 compliance

| R | status |
|---|---|
| R1 | input + cap on the emitted plan |
| R2 | nav into z¹ and z² |
| R3 | all v9 tactical labels supervise the z² heads |
| R4 | z² and the macro-action condition the plan through the β/γ rungs |
| R5 | one 6 s plan |
| R6 | strategic level built OFF |

---

## 5. refav1 → refav2: the frozen-encoder H-JEPA reference

refav1 (ckpt 21,109) is **closed as a zero-training lever**: every planner lever has been eliminated, and the trunk adds nothing beyond P10 (M). Its job is now to answer the frozen-encoder question for the programme cheaply, on the dev box, before Thor or A40 time is spent.

| step | what | compute | bar (pre-registered in `PREREG_HJ_STAGE0.md`) | if it fails ⇒ next lever (automatic) |
|---|---|---|---|---|
| **RA-0** (S2) | Collapse diagnostic on the refav1 checkpoint, run on 40 of the 141 eval episodes. Measure: ρ (within-episode variance share) of the adapter latent and of the predicted latents; **action sensitivity** (paper Tab. 12 metric: the relative rollout change when the actions are swapped for another episode's), per channel (a / κ / v) | dev-box CPU, hours | descriptive. Pre-registered prediction: steer-sensitivity < 0.05 and speed-sensitivity ≫ steer-sensitivity | If **not** collapsed (sensitivity ≥ 0.10), H-JEPA's mechanism does not explain our deafness. Next lever: **copycat / action-from-history** (`2010.14876`, banked). Remedy: history-action dropout on the tiny rig |
| **RA-1a** (S4) | **Pooled frozen-DINOv3 cache**: 600 TRAIN clips first, then 4,572. `pull_encode.py` + pooling to 4×10 (and 8×20) tokens; fp8; stored on **C:** (59 GB free; D:/E: is full, 5.2 GB) | 4060 when its lock frees (M rate: 2,834 s per 600 clips) or Colab; ~2.5 GB / ~19 GB at 4×10 | integrity: bit-exact re-encode on 5 clips; clip-id disjointness from the 141 eval clips | — |
| **RA-1b** | **refav1-HJ mini**, arms on the pooled cache: **R0** direct MLP readout (no WM); **A0** 2-level H-JEPA, SIGReg only; **A1** = A0 + IDM (2×2 grid β ∈ {0.04, 0.08}, γ ∈ {50, 100}); **A2** best A1 + level 3. Every WM arm carries the same 6 s trajectory readout over [z¹, z², P10 features], with **P10 as the residual prior**, at 2 training seeds | 4060, about 1–2 h per arm (E) | **BAR-HJ-SENS** (A1 vs A0) and **BAR-HJ-K**: `best − P10` ADE @ 6 s < 0, separated, at both seeds, with \|Δ\| > 2× the seed floor; event strata reported (turn onset, stop, lead braking). **BAR-HJ-COST**: level-2 monotonicity > level-1 | SENS fails ⇒ γ up to 200; IDM at level 2 only; then the time-repulsion floor (paper L.6, \|λ_sim\| ≤ 0.06). K fails with SENS passing ⇒ **frozen ceiling reached** ⇒ RA-2 with LoRA (T-B). Both fail ⇒ refav2 retires as a WM-planner reference; the trunk question moves wholly to v8F's `T-A`/`T-B` |
| **RA-2** | **refav2** (only if RA-1 passes K): retrain refav1's design, swapping the 3-rate FiLM patch-predictor for the H-JEPA levels. Adds IDM, the R5 6 s trajectory head with the P10 prior, the strategic level OFF, and the top-down gradient planner used **as a refiner of the R5 trajectory** (never planning from scratch). Lands the banked R1/R3/R5/R6 + R1-VMAX builds first (A §5 B/C) | Thor or A40 + a `refav1` gate profile. Storage solved by the pooled cache (**19–76 GB, not 303 GB**) | inherits RA-1's bars at full scale + four families + inference-seed replicate | — |

---

## 6. The ladder and its automatic continuation tree (Rule Zero)

| rung | arms (tiny rung, Thor, batch 8, ≈ 4.5 h/arm, seeds 0 and 1) | bar | FAIL ⇒ next lever (runs in the same package) |
|---|---|---|---|
| **0** (S1, 0 GPU) | read refcv7-50,400 and refcv8 EMA-15k against **P10 / damp50 / kd_x** on eval139 S2/S6. Exploratory: a late-fusion `α·os + (1−α)·P10` with α fit by grouped 5-fold CV inside eval139 | **BAR-V8F-FLOOR**: `os − P10` ADE 0–6 s < 0, separated, both inference seeds | Negative ⇒ the prior is v8F's critical lever (rung α, arm α1 first) and every "beats doing nothing" row on LEADERBOARD gets re-read. **RESULT 2026-10-17: POS** at both inference seeds, `os − P10` −0.4061 / −0.4054 m @ 6 s; **but P10 wins at 2 s** (+0.0276, separated). refcv8 EMA-15k was not read (§14) |
| **α** | `α0` refcv8 recipe (rung baseline) + replicate `α0r`; `α1` prior swap → P10; `α2` α1 + H-JEPA L1+L2 **without** IDM; `α3` α1 + H-JEPA **with** IDM; `α3x` α3 with **time-shuffled targets** (deliberate regression: must NOT help) | **BAR-V8F-NR**: no family separated worse than α0, each \|Δ\| read against F = max(\|α0r−α0\|, …). **BAR-HJ-SENS** on α3 vs α2. ~~**α1 must win** S6 ADE and LAT cross by > F~~ **AMENDED 2026-10-17, before any α1 data (§14):** **α1 must win S2 ADE and LON speed by > F, and must not be separated-worse at S6 (NR).** The old S6/LAT bar was premised on the retracted −1.90 m @ 6 s; S1 places P10's measured advantage over refcv7 at 2 s and in the longitudinal family. S6/LAT are still reported | α1 fails ⇒ P10 becomes an **input feature** + a learned prior head with dropout (X2). α3 SENS fails ⇒ γ grid ×2, then IDM on z² only, then a **command-channel** action vocabulary (`D-V7F-COMMAND-CHANNEL`, never run). α3x helps ⇒ **instrument broken; stop and fix before reading anything** |
| **α′** | trunk ladder `T-A` / `T-B` / `T-C` on the best α arm (§3) | NR + K at F | all tie ⇒ keep `T-A` (already gated) |
| **β** | best α + the hierarchical scorer (gate init 0) + a **shuffled-goal** control (ĝ² taken from another window) | `β − α_best`: S6 ADE and the TAC selection families < 0 beyond F; the shuffled-goal control must lose | scorer gate stays ≈ 0 ⇒ score in z¹ only (E2 says it should be worse: a known-direction check); then a learned scorer over WM-imagined features (paper §3.4) |
| **γ** | best β + gradient refinement | sharp-turn stratum (\|yaw\| ≥ 0.15 rad/s, v0 5–10 m/s): slope shortfall vs refcv7 > 0 beyond F; no family worse | refinement does not move ⇒ more steps / an LR sweep; then a refinement **head** trained on WM gradients (amortised) |
| **full** | the winning recipe, one full pass of the 4,572-clip view (≈ 48.9 k steps, ≈ 5.5–6.5 d on Thor, E) + training-seed replicate | all BAR-V8F (§8) | per §8 row |
| **δ** | REFe scorer component + RL-in-imagination | navtest > 88.82 separated; navhard > STOP 29.85 | — |

---

## 7. Streams, owners, compute, start conditions (≥ 5 in flight; token-budget rules apply: Sonnet by default, hand off instead of waiting)

| stream | deliverable | compute | starts | package |
|---|---|---|---|---|
| **S1 Floors** | `taniteval` arms `kin_mlp` (P10), `damp50`, `kd_x` + known-value tests; the rung-0 read on refcv7-50,400 (and refcv8 EMA-15k if its dumps are banked) | dev-box CPU; TRAIN-window kinematics pulled read-only from Thor (MB), fallback grouped CV in eval139 | **now** | `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-17-p10-floors-refcv/` |
| **S2 Collapse diagnostic** | RA-0: ρ + per-channel action sensitivity on refav1-21,109 (+ v7-tiny `rdw8p30k` if a checkpoint is reachable read-only) | dev-box CPU (the GPU is held by another session: 97 % util) | **now** | `…/Architecture & Inference/Research/2026-10-17-hjepa-collapse-diag/` |
| **S3 H-JEPA module** | `stack/tanitad/models/hjepa.py`: level encoder, macro-action encoder, AdaLN causal predictor, SIGReg (T,N,D) with ECF all-reduce, IDM head, top-down gradient planner + scorer API. Tests: Prop. 1 construction (a background-only encoder reaches L_pred = 0), the IDM floor = 1 on unit-variance actions, an identity-predictor sensitivity = 0 control, the mutation "delete IDM ⇒ G-LIVE RED", and the sharding invariance of SIGReg | dev-box CPU | **now** | `…/Research/2026-10-17-hjepa-module/` → lands in `stack/` |
| **S4 Pooled DINOv3 cache + RA-1** | RA-1a cache, RA-1b arms | 4060 when free / Colab GPU | cache code **now**; GPU when free | `…/Research/2026-10-17-refav1-hj-mini/` |
| **S5 v8F gate + rung α** | `PROFILES["v8f"]` (template: the refcv8 profile), G-DVB entries for every new flag, the G-CLOCK measurement, then the α arms | Thor, after refcv8 (≈ 2026-10-20 17:20, E) | gate code after S3; arms after the **PI go** | `…/Research/2026-10-20-v8f-rung-alpha/` |
| **S6 Register + landing hygiene** | register rows for the H-HJ-* hypotheses; refav1 09-27 verdicts into REG §2.4 (handover step 0); land the banked refav1 R1–R6 builds | 0 GPU | with the Master Mind (it lands; this session stages) | — |
| **S7 Research Lab** | bank the primaries this plan cites: AD-E2E-JEPA `2609.34085`, ForeDrive `2609.26299`, WA-JEPA `2608.20974`, LeWM, LeJEPA/SIGReg `2511.08544`, HWM, PLDM, FROST-Drive `2601.03460`. Plus a **T2 feasibility note** (HUGSIM closed loop as used by WA-JEPA, NAVSIM v2 reactive, AlpaSim/NuRec) | 0 GPU | **now** | `…/Research/2026-10-17-hjepa-literature/` |

**Compute map (M, 2026-10-17).**
* Thor runs refcv8 until ≈ 10-20.
* The A40 is assigned to the safety-objective arm (PI 2026-10-17).
* The dev-box 4060 is busy (another session).
* D:/E: is full (5.2 GB free). C: has 59 GB free.

⇒ Until 10-20 this plan is **CPU + Colab only**, by construction. That is why S1–S3 and S7 come first.

---

## 8. Committed bars (register before compute; the numbers are the floors as measured)

| bar | definition | baseline (M) |
|---|---|---|
| **BAR-HJ-SENS** | level-1 action sensitivity (paper Tab. 12 definition: relative change of the k-step rollout under another episode's actions, same start state) **≥ 0.10** and separated from the no-IDM arm at both training seeds. Per-channel κ-sensitivity reported | our census max 0.0919 % nrmse; refav1 κ 1.99e-06; paper IDM-on 0.280 vs < 0.05 collapsed |
| **BAR-HJ-K** (vision helps) | `arm − P10` ADE @ 6 s < 0, separated at both seeds, \|Δ\| > 2× the training-seed floor; matched-capacity readout | P10 **2.9257 m @ 6 s / 0.2229 @ 2 s** (refav1 grid, 2,399 windows / 141 eps) |
| **BAR-HJ-COST** | level-2 goal-cost monotonicity (−Spearman along expert trajectories) > level-1, paired over episodes | paper 0.89–1.00 (subgoal) vs 0.48–0.75 (flat) |
| **BAR-V8F-FLOOR / K** | `os − P10` (and vs `damp50`, `kd_x`) ADE 0–6 s < 0, separated, both inference seeds, on S2 and S6 | P10 on eval139: computed by S1 |
| **BAR-V8F-E1 / E3** | `os − ha0_ext` < 0 separated at 0–2 s and 0–6 s | refcv7: S2 −0.0102 n.s.; S6 **−0.9804 [−1.294, −0.6815]** |
| **BAR-V8F-NR** | no family separated worse than the parent; lever effects only beyond the training-seed floor F | refcv8 15k: NR **FAIL on LATERAL** (PROVISIONAL) |
| **BAR-V8F-TURN** | sharp-turn slope shortfall vs refcv7 ≥ 0 | refcv8 15k **−0.049 [−0.087, −0.018]** |
| **BAR-V8F-N1 / N2** | navtest PDMS > refcv7 LEGAL 70.2048 and > STOP 61.82; navhard EPDMS > max(STOP 0.2985, CV, ECHO) | refcv7 navhard **0.1987 (loses to STOP)** |
| **BAR-V8F-LEAD** | LEAD-VIOL share ≤ echo, clipped progress ≥ 0.90 | refcv7 0.0590 vs echo 0.0434 |
| **Frontier references (not bars)** | REFe P2 88.82 · SimWAM 91.5 · Drive-JEPA 93.7 · DriveZero 94.55 · human 94.55 | REG §14.2; LEADERBOARD §0E |

Every row carries four families (STRATEGIC = PAUSED by the R6 ruling), a tier stamp, the paired episode-cluster bootstrap with the variance it answers named, plus a training-seed and (if sampling) inference-seed replicate.

---

## 9. Disruptive programme ideas folded in (not yet implemented or proven) — each with its entry rung

| idea | source | entry |
|---|---|---|
| **4-brain hierarchy as distinct-latent H-JEPA levels** | thesis + `lib:2610.06805` | α |
| **Imagine-and-select in the tactical latent** | paper §3.4; H-JEPA E2 | β |
| **WM-gradient refinement** of the chosen plan, aimed at refcv8's sharp-turn under-correction | H-JEPA §2.2 + `H-R8-TURN-UNDERCORRECT-1` | γ |
| **Self-knowledge:** per-level imagination error as a familiarity and fallback trigger | paper §3.6. Instrument: AUROC of predicting the model's own top-decile-error windows on eval139, vs a speed-only control | after α (free: needs only the WM) |
| **IDM head reused as P3's action labeller.** Label action-free dashcam video (YouTube/OpenDV class, research licence), then pre-train the H-JEPA stack at 10–100× our clip count | `TANITAD_PROGRAMME.md` P3 + H-JEPA E5 | after α passes SENS; corpus build is a Data FlyWheel item |
| **Train-time-only big prior (REF-D's economics), as feature distillation:** a frozen DINOv3-L teacher at training time only; the shipped trunk is `T-A`/`T-B`. Replaces a separate REF-D arm | `REFD_DESIGN.md` §1 | optional α′ arm |
| **Topological strategic memory** (VQ place codes on z³) | paper §3.3 | after the R6 pause is lifted (PI) |
| **Closed loop (T2)** | G1 requires it | S7 feasibility now; first T2 read after γ |

---

## 10. Decisions only the PI can make (each with a recommendation)

| # | decision | recommendation |
|---|---|---|
| D-A | **v8F definition** = refcv8 chassis + H-JEPA stack (L1 + L2, IDM, SIGReg) + P10 prior, rung ladder α → γ | **Yes** (§4). It supersedes the handover's F1 by naming the WM recipe |
| D-B | Thor queue after refcv8: the **rung-α tiny ladder (≈ 7 arms × 4.5 h ≈ 32 h)** before any full run | **Yes.** No full run before α clears NR + SENS |
| D-C | Step-time cap for the full v8F run | Decide on G-CLOCK's measurement. Expected ≈ +10–15 % with `no_grad` targets (E) ⇒ ≈ 9.8–11 s/step |
| D-D | refav1 storage: **pooled-token cache (19–76 GB on C:)** instead of the 303 GB full cache; refav1 R1–R6 build questions D1–D5 | **Pooled cache: yes.** Default prior = P10 (`kinmlp`) |
| D-E | **REF-D**: retire as a separate arm; fold its train-time-prior idea into v8F as the distillation arm | **Retire + fold** (it needs 2–3 weeks of build and has no forward pass) |
| D-F | GPU for the RA-1 mini-arm while the 4060 is busy | Colab GPU (no persistence: the cache is built on C: and uploaded in ≤ 40 MB chunks) or the 4060 when its lock frees |

v7F (F2) stays shelved; its only research question (does a trained-trunk WM become action-sensitive?) is now answered more cheaply by S2 + RA-1 + α3.

---

## 11. Risks and traps for this plan

* **Not-driving transfer.** H-JEPA is P in simulation and on DROID offline. Every claim here is H until our own bars pass. *The plan is built so the first two rungs pay off even if H-JEPA does not transfer:* the floors (rung 0) and the prior swap (α1).
* **Driving's collapse may be speed advection, not background.** S2 measures per-channel sensitivity, so speed-only sensitivity is not mistaken for health.
* **IDM learns past ego-motion, which P10 already has.** That is why BAR-HJ-K reads against P10 and not against `ha0_ext`; sensitivity alone is never reported as "vision helps".
* **Built ≠ reachable.** Every new flag gets G-DVB, a G-LIVE non-zero-gradient check with an exactly-zero control, and a mutation that must go RED (the CLAUDE.md "unreachable from its caller" class). `check_untrained_params.py` runs after every arm.
* **Goal leak.** ĝ² must not see the situation classifier's output (§4.3), and the shuffled-goal control must lose.
* **Grids.** The refav1 grid (2,399 / 282 windows), eval139 S2/S6 and NAVSIM are never differenced against each other.
* **Disk.** D:/E: is full. Never write caches or many small files there; C: holds S4.
* **Stale working tree.** `D:\Projects\TanitAD`'s HEAD is `37645fcc`, behind the remote `e8cadf2`. Read via the mirror; land through `qland`, never by committing from this tree.

## 12. Manifest

| artifact | where |
|---|---|
| this plan | repo: `Project Steering/PLAN_V8F_REFAV1_HJEPA_2026-10-17.md` |
| pre-registration for rungs 0 / RA-0 / RA-1 | repo: `Project Steering/PREREG_HJ_STAGE0.md` |
| H-JEPA primary | repo: `TanitAD Research Lab/Library/papers/2610.06805_…pdf` (`lib:2610.06805`) |
| stream packages | paths in §7 (created by each stream) |

---

## 13. AMENDMENT 1 — PI rulings of 2026-10-17 (afternoon): three layers, DINOv3-L + LoRA, ≤ 800 M, the data question, compute

### 13.1 The rulings (the PI, 2026-10-17; paraphrased; the chat transcript is the record)

1. **"I approve the DINOv3 L + LoRA direction."** ⇒ The v8F trunk is **DINOv3 ViT-L/16, frozen, plus LoRA**, plus a trainable adapter. The trunk ladder of §3 is no longer a decision; T-A (ResNet-101) remains only as the parent baseline that the NR bars read against.
2. **"The budget can be extended up to 800 million parameters if necessary."** ⇒ The sub-300 M constraint of §3 and §4.1 is lifted to **≤ 800 M** for v8F.
3. **The three layers** are the PI's definition. It supersedes §4.1 where they differ:
   * **Strategic.** Takes the nav command and the distance to the manoeuvre, and creates a long-term plan **in the form of checkpoints**. It will later extend to the country and its traffic rules, as strategic commands.
   * **Tactical.** The checkpoints condition the tactical layer, which works up to 6 s. It learns to generate the **tactical goals and constraints** of the training corpus.
   * **Operative.** The tactical goals and constraints condition the operative layer.
   * **Inputs, not learned labels:** measured ego dynamics, max target speed and nav commands.
   * Together, the tactical and operative layers output the target trajectory to 6 s. That trajectory **must be consistent with the tactical goals and constraints**.
   * **Each layer runs at its own cycle time.**
4. **"My priority is to achieve the fastest way to good driving results."**
5. **Compute.**
   * Use the PI's **second computer (8 GB GPU)** for experiments; Thor stays busy with refcv8 for the next days.
   * **An A40 pod for some days, only when we are sure the model works and is ready** (§13.5).
6. **"Approve the rest of the plan."**

⇒ **Two consequences:**
* **R6 (strategic OFF) is superseded for v8F.** The strategic layer is ON as a *checkpoint generator*, and the STRATEGIC metric family is **un-paused for v8F**.
* **The 2026-10-11 ruling stands.** No route checkpoint is ever an INPUT taken from labels. Checkpoints are the strategic layer's own OUTPUT, predicted at inference.

### 13.2 v8F — the three layers, made concrete

**Inputs (never learned):**
* measured ego dynamics at t0: v, a_long, yaw-rate, plus the 8 t0 kinematic features P10 uses;
* max target speed: the v8 sidecar, `limit_kmh ∈ {30, 50, 100, 120}`;
* nav command + distance to manoeuvre: v9 per-frame nav;
* camera: DINOv3-L/16 + LoRA.

| layer | cycle | conditioned on | learns to output | train-time labels | loss |
|---|---|---|---|---|---|
| **Strategic S** | **1 Hz** | scene summary + nav + distance + ego dynamics | **route checkpoints** at 30 / 50 / 80 m of arc ahead: (x, y, heading) in the ego frame | v9 **ROUTE CHECKPOINT** RC-A, already built: the driven path smoothed in arc length (σ 8 m), read at 30 / 50 / 80 m (`build_v9_labels.py` §6) | L1 on position + heading |
| **Tactical T** | **2 Hz** (0.5 s) | checkpoints + inputs + scene | the **v9 tactical goals and constraints** over [NOW+2 s, NOW+8 s]: goal tokens, lat/lon action tokens, constraint values (distance, time, target speed) | v9 per-frame labels (95.09 % of 746,946 train windows, `H-V9-COV`) | CE + masked regression; IGNORE cells masked |
| **Operative O** | **10 Hz** (0.1 s) | tactical goals + constraints + checkpoints + inputs + scene | **one 6 s trajectory** (60 × 0.1 s), a residual on the **P10 prior** | logged ego future | ADE / FDE + the **consistency losses** below |
| **World model (4th brain)** | each layer's own cycle | H-JEPA levels z¹ / z² / z³ aligned with O / T / S | latent prediction (auxiliary first, as in §4.3) | self-supervised | pred + SIGReg (+ IDM, pending the S2 mechanism read) |

**My interpretation, to confirm.** The PI's text gives the strategic layer nav and distance. I also give it the **scene summary**, because placing a checkpoint on the actual road needs the road. The **nav + distance-only S** is kept as the ablation, and it doubles as the strategic family's echo control (§13.4). If vision adds nothing at the strategic level, the simpler one ships.

**Consistency (binding: "must be consistent").**
* **Training:** differentiable penalties, one per active constraint:
  * speed ≤ min(max target speed, SPEED_BAND v_hi);
  * STOP: v → 0 before the stop distance;
  * TURN: the sign and band of the heading change match the goal;
  * LEAD: time gap ≥ the constraint.
* **Inference:** hard projection for the R1 speed cap, plus Kamm μ 0.7.
* **New metric, `CONSISTENCY`**, reported beside the four families:
  * (a) the fraction of windows whose plan violates any active **GT** constraint;
  * (b) the fraction violating the model's **own** predicted tactical goal (self-consistency).

**Cycle-time realism.** Training uses **sample-and-hold**: O sees T's output from its last tick (0–0.5 s old), and T sees S's output from its last tick (0–1 s old). **Scheduled sampling** of GT versus predicted upper-layer outputs (p_GT 1.0 → 0.2) removes the exposure bias between the layers (`1506.03099`, banked).

**Admissibility.** S, T and O are computed from vision + nav + ego dynamics + max speed only. None of them reads the situation classifier's output (binding rule of 2026-08-03).

**Parameters (E).**
* DINOv3-L 303 M
* LoRA r 32 on q / v: ≈ 6–25 M
* S / T / O heads ≈ 40–120 M
* WM levels ≈ 20–50 M

Total ≈ **370–500 M**, under the 800 M cap.

### 13.3 The data question — "do we have the right data?"

**Facts:**

| fact | class | source |
|---|---|---|
| We train on **B1 = 4,719 clips × 20 s = 26.2 h** (4,572 train / 147 eval) | M | `D-CORPUS-B1`; REG §16 |
| PhysicalAI-AV holds **306,152 clips × 20 s ≈ 1,700 h, 25 countries, 2,500+ cities** ⇒ **we use 1.54 % of it** | I | the NVIDIA card, quoted in `…/Data Engineering/Implementation/incoming/2026-07-26-nc-ingest/NC_INGEST_REPORT.md:155` |
| **The v9 labels are built from the ego log alone** (egomotion parquet → turn table, per-frame nav, route checkpoint) ⇒ they extend to **any** clip at near-zero cost, with the same quality pins (`H-V9-COV / GEOM / NAV`) | M | `stack/scripts/build_v9_labels.py` header |
| The colour-carrying traffic-light goals come from the Alpamayo CoT, so they exist only on the B1 records (779 / 4,572). Lead constraints need `obstacle.offline` (97.44 % coverage) | M | CLAUDE.md layer table; `D-TLIGHT-1` |
| **The mix is wrong for manoeuvres.** Lane-keep dominates: 247 / 282 windows on the refav1 eval grid. Sharp turns are ≈ 8 % of eval139 windows, which is **exactly where refcv8 regresses** | M | `RESULT-refav1-21109-openloop.md`; `H-R8-TURN-UNDERCORRECT-1` |
| Our runs did ≈ **one pass**: refav1 21,109 steps; refcv8 48,852 steps ≈ 781,635 windows. Windows overlap heavily, so one pass of windows already shows each frame many times | M | REG §2.4, §16 |

**Answer.** The data is the right *type*: front-wide 120°, ego log, nav, obstacles, and labels we can regenerate. It is the wrong *amount* and the wrong *mix*.
* The strategic layer needs many junction approaches.
* The tactical layer needs many examples of its rare goals.
* The H-JEPA world-model objective is self-supervised and benefits from new scenes more than from re-shown ones. DriveVLA-W0 (`lib:2510.12796`, banked, **not re-read here**) reports that world models amplify data scaling.

More epochs mostly re-show the same 26 h. **Recommendation: all three levers, in this order, each decided by measurement:**

1. **D1 — rebalance the existing 26 h (free, now).** Event-stratified sampling (turns, stops, lead braking, traffic lights, junction approaches) so that rare tactical classes are 15–25 % of every batch. Eval untouched.
2. **D2 — data-vs-epochs probe (`H-DATA-SCALE-1`, box 2, pre-registered).** On frozen pooled features, train v8F-mini on 25 / 50 / 100 % of the clips at matched steps, and for 1 / 2 / 4 epochs at 100 %.
   * The data curve still steep ⇒ D3.
   * Flat in data but improving with epochs ⇒ more epochs (cheaper).
3. **D3 — extend, targeted.** Mine the 306k clips **by egomotion alone**: no video needed to choose.
   * **Selection:** junction turns, roundabouts, stops, lane changes, low-speed manoeuvres, plus country diversity for the future strategic rule layer.
   * **Size:** start with **+5k–20k clips (×2–5)**.
   * **Labels:** run the same v9 builder, with identity checks on the B1 clips.
   * **Encode:** with DINOv3-L.
   * **Cost drivers:**
     * download: the dev box's internet is 1.2 MB/s, too slow; do it on the A40/HF side at ≈ 118 MB/s, or on box 2 if its link is fast;
     * storage: pooled 4×10 fp8 ≈ 4 MB/clip, so 20k clips ≈ 80 GB.
   * **Leak guard:** new clips must be disjoint from the eval clips **by clip id and by recording session**.
   * **Eval:** stays fixed (eval139 and the 141-episode refav1 grid).
   * **Comparability:** extended-data arms are a new comparability domain (B1 is already non-parity) and are only ever compared on the same eval windows.

### 13.4 Bars added for v8F (pre-registered in `PREREG_V8F_MINI.md`)

The trajectory bars in §8 stand. The new ones are:

* **K-MINI:** beats P10 at 6 s.
* **CONSISTENCY:** at or below P10's violation rate, separated; self-consistency ≥ 0.90.
* **STRATEGIC:**
  * checkpoint error at 30 / 50 / 80 m beats a nav-only kinematic baseline;
  * a **nav-shuffle control must degrade** (it proves nav is used);
  * the vision arm beats the nav + distance-only arm.
* **TACTICAL:** v9 goal macro-F1 beats both the majority-class and the nav-only baselines.

### 13.5 Compute: the fastest path and the A40 readiness gate

* **Thor** (after refcv8, ≈ 10-20): v8F-α on the refcv8 chassis as approved, starting with the prior swap at 2 seeds. ~~It is the largest measured lever on the proven chassis.~~ (RETRACTED, §14.) It is the cheapest measured lever on the proven chassis: P10 beats `ha0_ext` by 0.038 m @ 2 s and 0.574 m @ 6 s, both separated, and refcv7 loses to P10 at 2 s.
* **Box 2** (the PI's 8 GB machine, over SSH from the dev box): **track M**.
  * **M0** — the pooled DINOv3-L cache (S4).
  * **M1** — **v8F-mini**: the three layers + the P10 prior on the **cached frozen** features. No LoRA (impossible on cached features), no WM. Minutes to hours per arm.
  * **M2** — M1 + the H-JEPA WM, with and without IDM. This executes RA-1.
  * **M3** — D2, the data-vs-epochs probe.
  * **M4** — a DINOv3-L + LoRA smoke at batch 1 with gradient checkpointing and bf16. It proves the full model runs and measures step time and memory.
* **A40 (the PI)** is requested only when **all four** readiness checks pass:
  * **AR-1:** M1 or M2 beats P10 at 6 s, separated at 2 seeds; no family worse than the floors; CONSISTENCY ≤ P10's rate.
  * **AR-2:** M4's 200-step real-data smoke: every loss finite, a non-zero gradient on LoRA and on every head, an exact-zero control. The projected A40 hours are stated.
  * **AR-3:** a `PROFILES["v8f"]` launch-gate PASS on the exact commit and argv.
  * **AR-4:** the data is on HF private, so the pod trains within 1 h of start. A stop rule is set: abort if the in-run 6 s ADE at 25 % of the budget is not below M1/M2's.

  The A40 days then run the full LoRA training on B1 (on-the-fly encode) **and** the D3 download and encode in parallel.

### 13.6 Rule-Zero branches for track M

| failure | next levers, in order |
|---|---|
| M1 misses K-MINI | 8×20 pooling (more spatial detail) → more history frames → D1 rebalancing → LoRA (needs AR-2) |
| CONSISTENCY misses | raise the penalty weights → inference projection → a goal-anchored operative decoder (anchors per goal class) |
| STRATEGIC loses to the nav-only baseline | ship the PI's literal nav + distance-only strategic layer |
| TACTICAL loses to the nav-only baseline | the tactical heads read the z² WM latent (M2) → D1 rebalancing → D3 data |

## 14. ERRATUM 1 (2026-10-17, same day): lever 7 was a unit artifact, and the S1 floors result

**What was wrong.** §0 item 7 and the §4 residual-prior row cited `damp50 − ha0_ext` = −1.90 m @ 6 s and "P10 2.93 vs 5.36 m echo". Both come from the refav1 trunk-probe package. Its `extract.py:71-78` calls `paths_from_controls(...)` at the default `action_units="kappa"` (`taniteval/tools/refav1_arm.py:471`). But `actions[:, 0]` is a road-wheel **steer angle**, arctan(L·κ) with L = 2.9, per `refav1_loader.py`. The echo control therefore integrated the angle as a curvature and over-rotated.
* MEASURED, `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-17-p10-floors-refcv/raw/refav1_units_check.json`, 141 episodes / 2,399 windows.
* The rebuild with `kappa` reproduces the banked arrays to 9.6e-5 m; the rebuild with `steer` differs by up to 87 m.

| 6 s ADE, refav1 grid | banked unit (`kappa`) | correct unit (`steer`) |
|---|---|---|
| `ha0_ext` | 5.356 | **2.838** |
| `damp50` | 3.454 | 2.934 |
| `kd_x` | 3.558 | 2.940 |
| `damp50 − ha0_ext` | −1.9025 [−2.3605, −1.4971] | **+0.0956 [−0.0761, +0.2901], n.s.** |

**What survives.** The S1 floors read on eval139, where every floor is in the correct unit. Estimator: paired episode-cluster bootstrap, B = 2,000; it answers the **episode** variance. Tier: T1 open loop. Artifacts: `…/2026-10-17-p10-floors-refcv/raw/summary.json`, `raw/tables.md`.

| pair | 0–2 s (S2) | 0–6 s (S6) |
|---|---|---|
| `refcv7 − P10` (inference seed 0) | **+0.0276 [+0.0163, +0.0398] SEP** (P10 better) | **−0.4061 [−0.5936, −0.2161] SEP** |
| `refcv7 − P10` (inference seed 1) | +0.0282 [+0.0167, +0.0401] SEP | −0.4054 [−0.5911, −0.2231] SEP |
| `P10 − ha0_ext` | −0.0378 [−0.0478, −0.0278] SEP | −0.5743 [−0.7744, −0.3753] SEP |
| `damp50 − ha0_ext` | +0.1239 [+0.0985, +0.1506] SEP (worse) | −0.0140 [−0.2162, +0.2197] n.s. |

* refcv7's 6 s gain over P10 is **lateral**: cross-track −0.598 m, heading −2.06°. It is not in speed or along-track.
* At 2 s refcv7 is separated-worse on LON speed (+0.035), along-track (+0.027) and accel (+0.067).
* The P10 training-seed floor is 0.0213 m (one refit); the 6 s margin is 18× it.
* ⚠️ **refcv7's own training-seed variance is NOT measured** (one run). refcv8 was not read.

**What changes in the plan:**
1. **H-V8F-FLOOR-1 is POS.** refcv7-50,400 beats P10 at 6 s at both inference seeds. The sentence "refcv7 beats the learned kinematic floor" is **false without the horizon**: at 2 s, P10 wins.
2. **The prior swap α1 keeps P10 and drops damp50.** Its bar was amended *before any α1 data*: it must win S2 ADE and LON speed by > F and not regress at S6. The swap is now a **2 s / longitudinal** lever, not a −1.90 m one.
3. **v8F-mini (§13, `PREREG_V8F_MINI.md`) is unaffected.** Its operative layer was already a residual on P10, and its P10 reference numbers (2 s 0.2229 / 6 s 2.9257 m) are P10's own, not margins against the defective echo.
   ⚠️ **But the refav1-grid P10 was fit as a residual on the defective `kd_x`.** The correctly-integrated echo (2.838 m) beats it (2.926 m) on that grid. Before M1 is scored, **refit P10 on the correct-unit `kd_x`** of the refav1 grid. The S1 version already does that: `taniteval/taniteval/floors_kin.py`, 18 tests.
4. **Exposure, not a retraction.** §0 item 9, "fully frozen DINOv3 + a WM adds nothing beyond kinematics", compares two probes that both sit on the same defective `kd_x` base. The comparison is internally consistent, but its absolute level is wrong. It needs a re-read on the correct-unit base before it is quoted again.
5. **Root-cause class** (`RETRACTION_LOG.md`): *a control inherits the unit of the nearest default*. Same family as `C-REFCV3-ARM-SAME-DEFECT`. The fix is mechanical: `paths_from_controls` should refuse to run without an explicit `action_units` (BACKLOG item).

## 15. Amendment 2026-10-17 (night, before any α1 data): α1's prior is the POSE-ONLY P10 (`p10_pose`)

* **Why.** α0 (the refcv8 recipe) uses `ha0_ext_pose`, which reads **past poses only**. The P10 measured in S1 reads the **recorded steer**, and the steer channel at inference has **no PI ruling** (SPEC_REFCV7 §10, A5); NavSim has no steer channel either. α1 = `p10` would therefore move **two** things: the prior and the input set. That breaks "one lever".
* **The lever as built:** `--residual-prior p10_pose`.
  * It is P10 refit on the 4,572-clip train view, with features and kd_x base computed from past poses only (POSE_MODES).
  * It is frozen and sha-pinned: `stack/tanitad/configs/v8f_p10/p10_pose_prior.pt`.
  * The `v8f` launch-gate profile refuses every other prior by name.
  * The steer variant is built as arm α1s (profile `v8f_steer`). **The PI REJECTED it on 2026-10-17:** the model gets measured ego motion and its history, including yaw rate, never the steering angle (`D-PI-YAWRATE-1`, CLAUDE.md).
* **Premise re-measured for the pose variant** (`D-PRIOR-P10POSE-2S`): `os − P10pose` at 0–2 s is +0.0221 [+0.0105, +0.0346], separated, and the gap is longitudinal. The pose-only prior keeps the 2 s advantage the α1 bar rests on.
* **Bar unchanged** (§6 as amended before data): α1 must win S2 ADE and LON speed by > F against α0, and must not be separated-worse at S6.
* **Still open: D2, the tiny-rung geometry.** The Master Mind sets it at launch from refcv8's measured Thor step time, and the gate token is issued then.
* Package: `TanitAD Research Lab/Architecture & Inference/Research/2026-10-20-v8f-rung-alpha/`.
