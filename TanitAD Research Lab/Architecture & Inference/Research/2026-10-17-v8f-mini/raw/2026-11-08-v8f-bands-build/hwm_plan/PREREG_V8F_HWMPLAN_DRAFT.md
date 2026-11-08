# PREREG_V8F_HWMPLAN (DRAFT, not frozen; the Master Mind freezes it): does the multi-hierarchy world-model planner beat holding the action and beat a flat world-model planner, on the ladder world models? (agent draft, 2026-11-08, written before any HWM-PLAN eval window was scored)

## Why
The PI made multi-hierarchy world-model planning the baseline of the redesign (`REQ_HORIZONS.md` R-PLAN, 2026-11-08) and asked to compare it with other approaches. `PREREG_V8F_PLANTEST.md` read NO-GAIN for TD3 and FLAT1 on M2b at a 2.4 s plan (tier T0, oracle goal) and `PREREG_V8F_PLANTEST2.md` asks whether a seeded and regularised solver changes that. `PREREG_V8F_LADDER.md` trains world models whose levels reach 2 s, 6 s and 18 s. This card reads the planner of `SPEC_HWM.md` section 4 on those models, over the whole 6 s trajectory of R-TRAJ, per band, in the four families. Tiers: the true-future goal is **T0, a diagnostic ceiling, never a driving number**; a predicted goal read is **T1-kin** (the model conditioned on its own actions, integrated by the kinematic model). STRATEGIC is reported PAUSED (PI item 25).

## Objects
* Planner code `theory/hwm_plan/hwm_plan.py`, controls `test_hwm_plan.py`, choices `IMPL_CHOICES_HWMPLAN.md` (sha256 in its sidecar), tested before any window is scored.
* World models: the ladder runs that `PREREG_V8F_LADDER.md` names the world model of record (R35 or R35H), both training seeds, and C0 (2,2) as the same planner on the earlier steps. Zero training; adapter, readers and the planner network are frozen.
* Windows: PLANTEST's 1,976 windows (NOW = t - 15), minus the windows with fewer than 31 or 33 frames inside the clip (counted per arm, identical for every arm). **Deciding set: the SCORE half of PLANTEST2's split (salt `plantest2-v1`)**; the solver setting is the one PLANTEST2 chose on TUNE, so nothing is tuned on the deciding set.
* Priors: fitted on TRAIN windows of the same world model (never eval windows).
* Arms (all on the same windows and the same prior):
  | arm | what | tier |
  |---|---|---|
  | HWM-ORACLE | L3 -> L2 -> L1, goal = encoded true-future latent at the goal frame | T0 |
  | FLAT-ORACLE | level 1 only, same goal, same capacity | T0 |
  | HWM-PLAN | L3 -> L2 -> L1, goal from the goal head (separate card) | T1-kin |
  | FLAT-PLAN | level 1 only, same predicted goal | T1-kin |
  | HOLD, PRIOR | a0 held 30 steps; the level-1 prior mean held | T1-kin |
  The reactive arms (REACTIVE, HYBRID, REL, P10) of `SPEC_HWM.md` section 5 are scored by their own harness on the same windows and enter through the same paired bootstrap; this card does not run them.
* **Band levers (added 2026-11-08, `REVIEW_HJEPA_CONSISTENCY.md` gaps G1 to G7; each is a parameter whose default is today's planner, bit for bit).** Each lever is one arm against its control, run on the same checkpoint, windows and prior:
  | lever | arm of record | control | what it asks |
  |---|---|---|---|
  | T3 (G2) | `T3 = 4` (12 s, the ladder's L3 band end) | `T3 = 2` (6 s) | does a goal inside the 6 to 30 s band change the plan? Window counts per T3 are reported: oracle arms need the true-future frame at NOW + 60 or NOW + 30 steps, predicted-goal arms need 31 frames only |
  | plan_ctx (G1) | 3 (the context the ladder trained with) | 1 (the H-JEPA planning context) | does planning at the trained context change the plan? A retrain with `--wm-ctx 1 1 1` is the paper-equal arm if the two reads differ |
  | stitch (G3) | `l1_full` (level 1 plans all 30 knots against all level-2 sub-goals) | `decode` (L1 knots 1-10, L2 decoded 11-30) | is the 2-6 s band better planned by level 1 than read out of the decode? |
  | vocabulary seeds (G6) | level-2 candidates seeded from the nine tactical-vocabulary templates | HOLD seed only | does the tactical vocabulary help level 2? Top classes of a per-window posterior if one is banked for these windows, else all nine classes |
  | constraints (G7) | weight `w > 0` on T's constraints in the level-2 and level-1 cost (decoded kinematics: speed cap, lateral acceleration, deceleration) | weight 0 | do the constraints change the plan, and do they cost accuracy? |
  | goal head (G4, G5) | HWM-PLAN with the head-only trained level-3 goal, strategic token on | stub goal; shuffled goal; context swap; token off | does a predicted goal carry anything over the stub? |
  The goal-head card itself (training, its own gate) is separate; this card reads its output as the HWM-PLAN goal.

## The solver setting (fixed by PLANTEST2, filled in by the Master Mind at freeze)
`SolverCfg(seed_mode, lam, ...)`: HOLD seed with lambda from PLANTEST2's TUNE choice if PLANTEST2 reads PLANNABLE; PLANTEST's initialisation otherwise. Placeholder until then: `seed_mode = hold`, `lam = 0`. Macro decode: `decode_lam 0`, 300 AdamW iterations from HOLD.

## Read (paired episode-cluster bootstrap over the SCORE episodes, B 2,000, seed 0, `taniteval/ci.py`; both training seeds required; inference-seed replicate for every sampling arm)
Per band (0-2 s, 2-6 s, full 6 s), per family, reported separately and never pooled:
* LONGITUDINAL: acceleration MAE (deciding), speed error at the band end. Distance-keeping: headway, time gap and TTC to the lead agent (`taniteval.lead_metrics`, b1 eval lead block), errors |plan - logged| on windows in state LEAD, in every band: 0-2 s on the K = 10 lead block, 2-6 s and full 6 s on the K = 30 block (`data/b1_eval_lead_block_k30.npz`, IMPL section 14); window counts per lead state and plan-lost counts reported.
* LATERAL: yaw-rate MAE (deciding), heading error at the band end, curvature error (steps with logged speed >= 1 m/s, n reported), cross-track offset of the plan end from the logged-action trajectory end. ADE beside, no bar.
* TACTICAL: plan-derived decision, not a vocabulary output. The planned and the logged 6 s trajectory are classified by one geometry-only rule (the programme factored manoeuvre labeller, per band on the path from the band start pose; 2-6 s is the tactical band). Reported per band: lateral, longitudinal and collapsed 5-way confusion (logged rows, planned columns), per-class recall and n, kappa, and the per-window disagreement through the same paired bootstrap. Goal and anchor selection: n = 0 (not exposed by the planner).
* STRATEGIC: PAUSED (PI item 25).
Deciding quantities: full 6 s band, acceleration MAE and yaw-rate MAE. The 0-2 s and 2-6 s bands are read beside to locate an effect.
The inference-seed replicate (seed 1 against seed 0, same checkpoint) is the floor: a contrast smaller than the replicate difference is not an effect (`H-ESTIM-SEED-1`; the bootstrap answers the episode question only).

## Committed outcomes (per ladder arm and training seed; a label needs both training seeds to agree, else SEEDS-DISAGREE)
* **HWM-DRIVES:** HWM-PLAN minus HOLD is separated-better on both deciding quantities (full 6 s), and the contrast exceeds the replicate difference of HWM-PLAN.
* **GOAL-LIMITED:** HWM-ORACLE minus HOLD is separated-better, HWM-PLAN minus HOLD is not.
* **PLANNER-LIMITED:** HWM-ORACLE minus HOLD is not separated-better (T0 ceiling absent).
* **T3-ADDS:** HWM-ORACLE (T0) with `T3 = 4` minus the same with `T3 = 2` is separated-better in the full 6 s band on both deciding quantities, at both training seeds, on the windows both arms share. **T3-NEUTRAL** / **T3-HURTS:** not separated / separated worse. The window counts at each T3 are part of the row.
* **CTX-MATTERS:** HWM-PLAN at `plan_ctx = 3` minus at `plan_ctx = 1` is separated-better in the full band and exceeds the replicate difference. **CTX-NEUTRAL** / **CTX-HURTS** as above.
* **L1-FULL-ADDS:** `stitch = l1_full` minus `decode` is separated-better in the 2-6 s band without being separated-worse in 0-2 s. **L1-FULL-NEUTRAL** / **L1-FULL-HURTS** as above.
* **VOCAB-ADDS:** vocabulary seeds minus HOLD seeds is separated-better in the 2-6 s band and the tactical disagreement does not rise; the HOLD-COLLAPSE flag is read on both. **VOCAB-NEUTRAL** / **VOCAB-HURTS** as above.
* **CONSTRAINTS-HOLD:** at weight `w`, the constraint exceedance (the penalty of the executed plan) falls and the deciding quantities are not separated-worse than at weight 0. **CONSTRAINTS-COST:** exceedance falls and a deciding quantity is separated-worse. **CONSTRAINTS-INERT:** exceedance does not fall.
* **GOAL-HEAD-ADDS:** HWM-PLAN with the trained goal minus HWM-PLAN with the stub goal is separated-better on the deciding quantities (T1-kin), and the shuffled-goal and context-swap controls read worse than the trained goal. **GOAL-HEAD-INERT:** not separated from the stub, or the shuffled goal reads equal. **CONTEXT-USED:** the context-swap control reads worse than the trained goal; **CONTEXT-UNUSED** otherwise.
* **HIER-ADDS:** HWM minus FLAT is separated-better at the same goal (oracle at T0, predicted at T1-kin), at both training seeds, per family.
* **HIER-NEUTRAL** / **HIER-HURTS:** not separated / separated worse.
* **BAND-SPLIT (flag):** the full-band effect sits in 0-2 s only, or the 2-6 s band reads worse than HOLD while 0-2 s reads better.
* **DECODE-WEAK (flag):** the median relative macro-decode residual exceeds 0.1; the 2-6 s band is then read as a property of the decode, not of the level-2 plan.
* **COVERAGE-LIMITED (flag):** a lever arm has fewer than half of its control's windows (T3 = 4 is the case at risk: the clips end at about 20 s); it is read only on the windows both arms share.
* **READOUT-WEAK (flag):** the level-2 constraint term uses a linear readout of the macro-action; if its R^2 on the fitting windows is below 0.8 in either channel the level-2 term is read as a property of the readout.
* **HOLD-COLLAPSE (flag):** the mean distance of the plan from HOLD is below 10 % of the prior std; the row does not count as planning.

## What each outcome triggers (committed now, Rule Zero)
* **HWM-DRIVES:** HWM-PLAN is the baseline of record; it is compared with REACTIVE, HYBRID, REL and P10 on the same windows (the DOMINANT rule applies unchanged), and the next arm is the goal head's own ablation.
* **GOAL-LIMITED:** the goal head is the lever: its card runs next, at the oracle ceiling's gap; the planner is not changed.
* **PLANNER-LIMITED, with HIER-NEUTRAL or HIER-HURTS:** the cost space moves from raw latent distance to decoded action-relevant quantities (speed, yaw rate at the knots), read on the same world model; if that does not clear the bar, the world-model action sensitivity (inverse-dynamics weight, action-diverse sampler) is next, as `PREREG_V8F_PLANTEST2.md` commits.
* **HIER-ADDS:** the hierarchy is kept, and the level whose band carries the gain gets the next training lever.
* **BAND-SPLIT:** the failing band's level is replanned alone (L2 re-seeded from the macro encoding of the L1 plan's end state, with a continuity term) before any retrain is requested.
* **DECODE-WEAK:** a learned macro-action decoder (a head on the frozen model) is built before the 2-6 s band is read again.
* **T3-ADDS:** `T3 = 4` becomes the arm of record. **T3-NEUTRAL / T3-HURTS:** the goal head's horizon is the lever: its target steps are re-read at 6 s and 9 s before any retrain is requested.
* **CTX-MATTERS:** the planner runs at the trained context for every later arm. **CTX-NEUTRAL:** the paper-equal retrain (`--wm-ctx 1 1 1`) is not requested for this reason.
* **L1-FULL-ADDS:** `l1_full` becomes the stitch of record and the macro decode leaves the 2-6 s path. **L1-FULL-HURTS:** the decode stays and the level-2 re-plan from level 1's end state (the BAND-SPLIT lever) is run.
* **VOCAB-ADDS:** the vocabulary seeds stay on, and a per-window posterior of T replaces the all-class seeding. **VOCAB-NEUTRAL / VOCAB-HURTS:** the template amplitudes (ESTIMATED) are re-read first, then the seeds are dropped.
* **CONSTRAINTS-HOLD:** the weight is fixed at the smallest value that reads CONSTRAINTS-HOLD. **CONSTRAINTS-COST:** the weight is swept at the next three values before the lever is set aside. **CONSTRAINTS-INERT:** the cost scale is read first (penalty against latent cost), then the readout.
* **GOAL-HEAD-ADDS:** the head is the HWM-PLAN goal of record. **GOAL-HEAD-INERT:** the head's inputs are re-read (nav and context are absent from the harness windows today; their wiring is the first suspect), then a longer history.
* **Every outcome:** a refutation is a waypoint (Rule Zero): the cheapest next lever above is run in the same turn, or what blocks it is named.

## Controls (the read is INVALID if one fails; each has a deliberate-regression arm that goes red; `test_hwm_plan.py`)
1. **Reduction:** with PLANTEST's solver values, PLANTEST's initialisation and PLANTEST's geometry (top plan 3 steps, K = 3 and 6, level 1 plans all 12 knots) the plans equal PLANTEST's TD3, FLAT and FLAT1 plans bit for bit on a 16-window subset (plans and level-1 costs). Mutations: HOLD seed, lam_sub 0.4, S 8.
2. **Planted dynamics:** a three-level plan with a known linear action effect recovers the planted actions (MAE <= 0.05 and <= 0.2 x the prior baseline) in the level-1 knots and in the decoded level-2 knots. Mutations: actions disconnected, decode removed.
3. **Goal plumbing:** a different goal changes the plan in both bands; a mutation that cuts the sub-goal hand-off from level 3 to level 2 makes the first band goal-independent (check goes red). Oracle and shuffled goals differ.
4. **Band stitching:** knots 1-10 equal level 1's actions exactly, knots 11-30 equal the decoded level-2 macro-actions, and moving the level-2 macro-actions moves only the second part. Mutations: decode replaced by zeros or by HOLD.
5. **Bit-repeat:** the same inference seed gives bit-identical plans and costs; another seed gives different plans. Mutation: a planner ignoring the seed.
6. **No-leak:** replacing every logged action except a0 leaves the plan unchanged; a mutant seeded from the logged window changes. a0 is the measured ego state at NOW.
7. **Kinematic model and band metrics:** closed form on constant acceleration; bounds act; an identical plan reads exactly 0.0 in every metric and band; a plan that differs only in knots 11-30 reads 0.0 in the 0-2 s band. Mutation: swapped band edges.
8. **Geometry from the built model:** strides and counts read from the built stack, refused when the macro encoder's stride and the level config disagree.
9. **Admissibility:** the harness source holds no situation-classifier identifier (`hwm_plan.py`, `hwm_eval.py`, `hwm_data.py` and `goal_head.py` are scanned); the goal slot receives latents up to NOW, nav, measured ego history and the detached vision-only strategic token (`GoalInputs.ctx`) only.
10. **plan_ctx (G1):** the default and an explicit 1 give bit-identical plans and level-1 costs; context 3 changes the plan on the built (3,5) model; every level receives its own context in the rollout (spy); a context above the predictor's is refused. Mutation: the shadow not installed makes context 3 a no-op.
11. **T3 (G2):** the geometry reads `T3 = 4` as a 60-step (12 s) goal and the top plan as four level-3 steps; the goal at 12 s reaches the plan; the per-T3 window counts equal a hand count on a synthetic clip store. Mutation: T3 pinned to K2 plans two steps.
12. **stitch (G3):** with `l1_full` all knots are level 1's actions and no decode runs; with `decode` knots 7-12 differ; on a planted world `l1_full` recovers the planted actions; a level-1 horizon too short for the knots is refused.
13. **Vocabulary seeds (G6):** on a planted vocabulary world the best level-2 candidate starts at the goal with the vocabulary seeds and not with the HOLD seed; top classes of a posterior beat the fixed order when few candidates are seeded. Mutation: the seeds replaced by HOLD loses the advantage.
14. **Constraints (G7):** the penalty reads exactly 0.0 inside the caps, is positive outside, carries a gradient, and acts at level 1 and level 2; weight 0 with the hooks installed equals no constraints bit for bit; a planted goal that asks for a speed above the cap gives a plan under the free plan's speed. Mutation: the penalty cut leaves the plan equal to the free plan.
15. **Goal head (G4, G5):** selfcheck before step 1 (built levers read back, non-zero gradient on every head parameter, exact-0.0 control with every step masked, no gradient on the world model or on the strategic token); the world model is bit-equal after a run (sha256 of its state); the trained head beats the stub, while a shuffled goal and a context swap read worse; the context edge reaches the plan in both bands, and cut it the plan stops reacting. Mutations: output multiplied by 0.0, mask removed, detach removed, planted classifier identifier, a trainer that updates the world model, a head without a context path.

## Implementation choices
`theory/hwm_plan/IMPL_CHOICES_HWMPLAN.md`, sha256 in `IMPL_CHOICES_HWMPLAN.sha256`, written before any window is scored.

## Log
* 2026-11-08, Stream D: the band levers, the goal head and their controls were added to this draft (arms, outcomes, controls 10 to 15); it stays a draft. The Master Mind freezes it after PLANTEST2 sets the solver; the template amplitudes, constraint caps and the constraint weight are placeholders until then.
* DRAFT, unfrozen. No HWM-PLAN eval window was scored when this draft was written; the smoke run (16 windows, M2b s0, CPU) printed no metric.
