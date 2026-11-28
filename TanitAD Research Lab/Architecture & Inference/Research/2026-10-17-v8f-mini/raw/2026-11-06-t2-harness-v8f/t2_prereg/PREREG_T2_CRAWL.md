# PREREG_T2_CRAWL: does v8F's launch crawl survive T2 re-perception? (DRAFT, Master Mind, 2026-11-06; to be frozen before any v8F T2 rollout, stage 1 included)

**Status: DRAFT, NOT FROZEN.** It freezes, with its sha256 in `t2_prereg/FROZEN_SHA.txt`, once three things hold:
* gates T0 to T5 below are built and pass on the dev-box CPU;
* the M1 metric is pinned against the T1 reader;
* no v8F planner has yet been rolled out in the renderer.

Every change between this draft and the frozen card is listed in the card's own change log.

**PI approval (2026-11-06, in chat):** "Approve T2 path", answering *"Approve the T2 path (about 26 GB scene download plus box-2
jobs)?"*. It covers the NuRec scene download (≤ 80 GB) and the T2 box-2 jobs, run one at a time in SAFE MODE. Claim row:
`H-T2-CRAWL-1`.

## 1. Why
* **The one failing row.** v8F-mini's adopted line beats the P10 kinematic floor at T1-kin, and its composites fail Read A's D2 on the
  M1 launch crawl alone. C0 against P10: M1 excess +0.0533 / +0.0596 under `reindexed`, the same under logged frames (ratio
  1.002 / 1.003). Source: `R/raw/2026-11-05-motion-c-ri/ri_read/RESULT_RI_C0.md`, [M], T1-kin.
* **Every CPU lever has been read.** The progress-floor family ended with A6 (outcome 4, `c59e7b85`).
* **T1-kin cannot show the cue.** It replays logged cameras. A car that lags the human never sees the lead pull away or the scene
  move at its own pace, so a T1-kin crawl may be partly an artefact of the camera clock.
* **T2 can.** It re-renders the camera at the car's own pose and time, which separates "the model crawls" from "the replayed camera
  tells the model to crawl".

## 2. Hypotheses
* **[H1] CRAWL-PERSISTS.** Under T2, C0 still crawls more than P10 on launch scenes.
* **[H2] CRAWL-CLOSES.** Under T2 the excess largely vanishes, so the T1-kin crawl is mostly a logged-camera artefact.

## 3. Design (zero training)
* **Scenes.** The 42 launch scenes and 20 guard scenes chosen by `t2_scenes/SCREEN_RULE.md`, which was frozen before any scene log was
  read. The rule selects on the human's logs only and excludes v8F's training clips. No model output enters the selection.
* **Starts.**
  * Launch scenes: two starts, at t_c − 4.0 s and t_c − 2.0 s (t_c from the screening table).
  * Guard scenes: two starts drawn by `numpy.random.default_rng(1)`, uniform over the renderable span, leaving 6 s.
  * Each start takes the logged history (16 rows at 10 Hz). Then come 3 replans of 10 executed knots (2 s each, 6 s), each plan
    30 knots at 0.2 s and scored over all 30: the T1-kin geometry.
* **Executor.** PLAYBACK: the car executes its plan's knots exactly, as T1-kin assumes. The harness actuator is reported beside.
* **Inputs.** As in T1-kin: measured ego history including yaw rate from the executed track, nav from the log route, and the CAUSAL
  speed bucket (past 4 s of measured speed).
* **Conditions.**
  * **T2 (deciding):** the camera rendered at the executed pose and time; NuRec agents replayed at log time (`objects`).
  * **RR, render-replay (deciding, for attribution):** the camera rendered at the LOGGED pose, re-indexed by the car's own along-track
    progress. This is `reindexed` inside the same renderer, and it removes the render domain gap from the T2 − RR contrast.
  * **`empty` (beside):** no agents.
* **Arms.**
  * **C0_s0 and C0_s1:** the floor reads' planner, `v8f-recover/M2b_s{s}_recft_C0/ckpt_last.pt`.
  * **DAG-OACC_s0 and _s1 (beside):** the base.
  * **P10:** the frozen prior from the same forward.
  * **ZERO:** a wrapper returning the prior. It must equal P10 exactly.
  * **BLIND (instrument):** C0 with the scene tokens replaced by their banked training mean.
* **Metric.**
  * The M1 crawl share is computed by the T1 reader's own function, per rollout window: logged route speed ≥ 5.2 m/s at the car's route
    position, own v0 < 5.2 m/s. Gate T4 pins it.
  * The four families of Read A are beside: every row the T1 reader computes, STRATEGIC as Read A defines it.
* **Estimator.** Paired scene-cluster bootstrap, B = 2000, seed 0, over launch scenes for M1 and over all 62 for the families. It
  answers *another draw of scenes* only. The training-seed question is answered by |X_s1 − X_s0|. Inference: the planner is
  deterministic, and the renderer is checked deterministic by gate T5.

## 4. Gates before any deciding rollout (each with a deliberate-regression arm that must go RED)
* **T0, policy reproduction.** Fed the banked cached features, the policy reproduces the T1-kin tool's tick-0 plan (E1 G1).
* **T1, encoder fidelity.** The in-loop encoder on a logged frame reproduces the banked tokens (E2 G2): rel-L2 and cosine reported.
  On a rendered frame at the logged pose, the render gap is reported beside, never deciding.
* **T2g, playback identity.** RR with the logged plan executed by playback reproduces the logged ego track to 1e-6 m.
* **T3, zero control.** ZERO equals P10 exactly. An attribution of P10 against itself reads exactly 0.
* **T4, metric pin.** The scorer reproduces the T1 reader's M1 crawl share and its four family rows bit for bit on a banked T1-kin dump.
* **T5, determinism.** One C0 start rendered twice gives identical plans.
* **Actor falsifier (box 2, before stage 2).** `objects` draws the replayed lead in the rendered frame. A planted no-lead scene must read
  absent.
* **Stage 1 (box 2, the 2 scenes on hand).** Plumbing only. Its rollouts are never read for H1 or H2, and they are reported as such.

## 5. Contrasts (deciding scene set: launch scenes, condition T2, both seeds)
* **X1 = C0(T2) − P10(T2)**, M1 crawl share.
* **X2 = C0(T2) − C0(RR)**, M1 crawl share: the camera-clock effect, measured on the same renderer.
* **Reference:** half the T1-kin reindexed excess, h = 0.0267 at s0 and 0.0298 at s1.

## 6. Outcomes, committed now (Rule Zero; read in this order)
1. **CRAWL-CLOSES.** At both seeds, X1's upper bound is below h, and X2 is separated-better (negative).
   * The T1-kin crawl is largely a logged-camera artefact, so T2 becomes the longitudinal tier for v8F.
   * The next card is a T2 dominance read of C0's composite against P10 over all 62 scenes, all four families, with Read A's rules.
     If D holds there at both seeds, that is a counted DOMINANT at T2 on these scenes. It carries the scene-selection disclosure and the
     video evidence, and goes to the PI as the scale signal.
2. **CRAWL-PERSISTS.** At both seeds, X1's lower bound is above 0, X1 ≥ h, and X1 exceeds |X1_s1 − X1_s0|. The crawl is not a camera
   artefact. Sub-branch on BLIND − C0(T2) for M1:
   * **(a) Not separated:** the frozen tokens do not carry the cue. The perception lever (the A40 LoRA run, `D-V8F-SPEED-PERCEPTION`
     (a)) goes to the PI, with this tier as its evaluation and this read as the case.
   * **(b) Separated:** the camera is used and does not lift the crawl. Next lever, in its own card: on-policy launch states from T2
     rollouts as DAgger data (box 2).
3. **MIXED.** Either outcome holds at one seed only, or X1 is separated and below h. The excess shrinks under re-perception and
   remains. Report the share explained by X2. Next: (2a)'s perception lever with T2 as its tier, plus a seed replicate of C0 if the seed
   gap exceeds |X1|.
4. **UNDERPOWERED.** Neither bound decides. Extend the screening in the frozen permutation to the next 42 launch scenes, within the
   80 GB approval. Not a result.
5. **Any gate fails:** the instrument is fixed first. No contrast is read.

## 6b. Draft change log (Master Mind; every entry made before any v8F planner was rolled out in the renderer)
* **2026-11-06 ~12:2x, the start clamp.** A start needs f0 ≥ 18 logged frames of history. Some launch scenes have t_c − 4 s or
  t_c − 2 s earlier than that; three would lose every start. **Ruling:** each start is max(t_c − Δ, the first frame with full history)
  for Δ ∈ {4.0, 2.0} s, and two starts that clamp to the same frame collapse to one. All 42 launch scenes keep at least one start.
  Inputs seen: the human-log screening table only, no model output.
* **The guard starts.** Two per guard scene from `default_rng(1)`, as section 3 says. (The first build drew one per guard scene.)
* **The BLIND mean.** BLIND's constant tokens are the mean over the eval-141 cache's tokens, because the training cache lives only on
  box 2. BLIND is an instrument that is never deciding, and a test-time constant shapes no training. Its sha256 is recorded in
  `t2v8f/PINS_E4.txt`.
* **The render-gap references.** The 2 on-hand scenes' `camera_front_wide_120fov.mp4` (about 38 MB each, same release and revision) are
  added for gate T1's render gap, which is beside and never deciding. They fall inside the PI's 80 GB approval.

* **2026-11-06 ~13:3x, outcome edge cases (the scorer's `outcome()`; still before any v8F renderer rollout):**
  * **The BLIND sub-branch of CRAWL-PERSISTS, split across seeds** (separated at one seed only) reads **SPLIT**. It proceeds as (2a),
    because the cue is not reliably used, and the split is disclosed.
  * **UNDERPOWERED** means no bound decides at either seed. Every other case that is neither 1 nor 2 is MIXED.
  * **Under `empty`**, distance keeping is scored against the logged agents. `empty` is beside, never deciding.
  * **Gates first:** any failed gate returns outcome 5 before any contrast is read.

* **2026-11-06 ~12:3x, the planted no-lead read of the actor falsifier (still before any v8F renderer rollout).** The planted read
  is the `empty` condition rendered at the same pose and the same tick as the `objects` read, so the two frames differ only by the
  replayed agents. The first design (an actor clock 30 s past the end of every track) is kept beside: a renderer that clamps its
  clock would make that read fail for a reason about the renderer, not the scene. The thresholds (PRESENT >= 200 px and >= 1 % of
  the projected-lead ROI, ABSENT <= 5 px) are fixed before the first box-2 render.

* **2026-11-06 ~13:1x, the actor falsifier's lead ticks and the stage-1 package (last entry before the freeze).** The `objects`
  read uses ticks 31, 44 and 58 of scene `7c72937c`, where the logged lead is 9.2, 11.0 and 12.6 m ahead. They were chosen from
  the scene's own tracks (state LEAD, 8 <= x <= 40 m, |y| <= 3 m, spread through the candidates), with no render and no model
  output seen. The stage-1 package is `t2v8f/stage1/t2v8f_stage1.tar`, sha256 `1785337f...d3421c`, rehearsed on the CPU stub
  renderer only (37 tests passed, 6 of 6 mutations RED). No v8F planner has been rolled out in the real renderer.

## 7. Variance and validity named
* **Episodes → scenes.** The bootstrap answers another draw of scenes, and launch scenes are a selected population. So the X1 level is a
  claim about launches, not about eval-141.
* **Training.** Two seeds of C0, and one LLR-free planner per seed. The seed gap is reported beside every contrast.
* **Render gap.** NuRec reconstructions are out of distribution for the frozen encoder (REF-C read 3.21× OOD on them, [I]). RR shares
  the gap, which is why X2 is the attribution contrast and not T2 against T1-kin.
* **Interaction.** Agents replay and do not react. The read says whether the crawl is perceptual; it never says "safe".
* **Not T1-kin.** This tier is T2 re-perception rollouts on reconstructed scenes. The numbers are stamped T2 and are never pooled with
  T1-kin numbers.
