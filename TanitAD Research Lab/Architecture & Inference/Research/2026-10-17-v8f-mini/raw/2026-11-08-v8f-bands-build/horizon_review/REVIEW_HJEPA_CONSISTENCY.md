# REVIEW_HJEPA_CONSISTENCY: the three-band design against H-JEPA (arXiv 2610.06805) and the PI's top-down rule (Master Mind, 2026-11-08)

PI, verbatim (chat, 2026-11-08): "Review the implementation regarding consistency of the implementatio to jepa hierarchacy end2end planning october paper. Dont forget to condition the lower levels from goals of the higher levels"

Sources read: the banked primary (`lib:2610.06805`, sha prefix `4a4a799c50d2`), sections 2.1, 2.2, 3 (IDM), appendices C and E; `stack/tanitad/models/hjepa.py` and `v8f_mini.py` at ladder commit `1faa1a06`; `theory/hwm_plan/hwm_plan.py`; `theory/plantest2/plantest2_lib.py`; `SPEC_HWM.md`; `AUDIT_MODEL.md`; PI rulings of 2026-10-17 in `Project Steering/PLAN_V8F_REFAV1_HJEPA_2026-10-17.md` section 13.1. Every row below is a code or paper read (MEASURED by reading); no planner result is claimed.

## 1. How the paper conditions a lower level
A world-model predictor at any level reads its own latent state and its own action. No predictor reads a goal, in the paper or here. The top-down path is the **planner's cost**: level L plans to the goal (Eq. 4), and each lower level is optimised so that its rollout, lifted by E^(l+1), matches the predicted states of the level above (Eqs. 5, 6). So "condition the lower level on the higher level's goal" has one meaning inside the world-model planner (the sub-goal cost), and a second one in our reader cascade (S -> T -> O, the PI's ruling of 2026-10-17: checkpoints condition the tactical layer; tactical goals and constraints condition the operative layer). Both have to hold.

## 2. Paper against implementation
| # | paper | ours | state |
|---|---|---|---|
| 1 | Own encoder, macro-action encoder, predictor per level; w = 1 | `HJEPAStack`; w = 1 asserted by `Geometry.from_stack` | MATCH |
| 2 | Upper losses propagate through lower encoders; no EMA, no stop-gradient | inside the world model yes (`_apply_encoder` holds no detach; `rollout_loss` `detach_target` False) | MATCH inside the WM |
| 3 | Encoder trained end to end from pixels | frozen DINOv3 + frozen adapter + trainable `wm.proj`; the WM-only ladder cannot move the adapter | DEVIATION: PI ruling 2026-10-17 (DINOv3 + LoRA). The LoRA arm on the pod is the lever |
| 4 | SIGReg and IDM at every level, IDM target detached above level 1 | `wm = "b"`, gamma 100 on every level, read back at `v8f_mini_train.py:1500`; `hjepa.py:251` | MATCH |
| 5 | One-step teacher-forced loss | the same, plus a k-step free-run term | DEVIATION: required by R-OP / R-TAC / R-STR. Arm S35 is the paper-equal control |
| 6 | Planning context c = 1 at every level, "training covers this regime" | **planner `rollout(context=1)`, but the ladder's k-step term runs with `--wm-ctx 3 3 3`**: the free run feeds up to 3 of its own predictions back. Steps 2 onward of the trained free run are not the planner's regime | **GAP G1** |
| 7 | Top level: Eq. 4, final predicted state to the goal | `plan_topdown2`; last state at upper levels, last two at level 1 (App. E) | MATCH |
| 8 | Lower level: Eqs. 5, 6, lifted rollout against the upper rollout, weight lambda, H = K s + w - 1 | `plan_topdown2:119-126`; lambda 0.5 as on DROID; mutation test `c goal plumbing` | MATCH |
| 9 | Candidates, AdamW at most 90 iterations, macro prior fitted on training, DROID 1.5x width and +-2 std | `SolverCfg`, IMPL section on the solver | MATCH |
| 10 | Top horizon in the level's own steps, reaching the goal | **default `T3 = K2 = 2` on the ladder: the goal sits at 6 s**, so the strategic level never plans inside its 6 to 30 s band | **GAP G2** |
| 11 | Output: level-1 primitive actions, a prefix executed, then replan | knots 1-10 from level 1; knots 11-30 from an **inversion of level 2's macro-actions**, which the paper does not have | DEVIATION: R-TRAJ asks for 6 s. Paper's open-loop form (K = H, level 1 plans the whole horizon) is the nearest paper-equal arm and is not built: **GAP G3** |
| 12 | Commit schedule, replan from the encoded observation | one 6 s plan per window (the paper's DROID form) | DEVIATION: offline T1-kin. Replanned execution needs T2, not provisioned |
| 13 | Goal: an encoded goal observation | none exists at inference: `stub_goal_fn` returns the current latent; true-future goal only at T0 | **GAP G4** (the goal head, known) |

## 3. Top-down conditioning, edge by edge (PI's instruction)
| edge | mechanism | where | state |
|---|---|---|---|
| goal -> L3 | Eq. 4 | `plan_topdown2` | built; the goal itself is G4 |
| detected context, derived goals -> goal | input of the goal head | `GoalInputs` has no context field | **GAP G5** |
| L3 -> L2 | sub-goal cost, Eq. 6 | `plan_topdown2:119-126` | built, mutation-tested |
| L2 -> L1 | sub-goal cost, Eq. 6 | same | built, mutation-tested |
| T's vocabulary -> L2 | SPEC section 4 step 3: candidates seeded from the tactical vocabulary | `hold_seeds_g` seeds HOLD only | **GAP G6** |
| T's constraints -> planned trajectory | R: the trajectory must be consistent with the tactical goals and constraints (PI 2026-10-17) | no such cost in `hwm_plan` | **GAP G7** |
| S checkpoints -> T, O | `_ck_cond`, `v8f_mini.py:910, 933` | built; arc-indexed 30 / 50 / 80 m only |
| T code -> O | `e_code`, `v8f_mini.py:933` | built |
| strategic goals, context, time-indexed checkpoints -> T | AUDIT section 3.3 proposed a token **default off** | **GAP G8**: not built; it must be on in the arm, with the off arm as control |
| T -> O inside the 6 s trajectory | SPEC section 2 and AUDIT 2.1 (ii): tactical head **conditioned on the operative end state** | **GAP G9, the design points the wrong way**: that is the lower level steering the higher one |

## 4. What changes (SPEC_HWM section 9)
* G9: the tactical level decides first. It outputs knots 11-30 **and a hand-over state at knot 10** (position, heading, speed). The operative head reads T's code and the **detached** hand-over and plans knots 1-10 to reach it; a continuity term trains O only. This is the reader-side form of Eq. 6. O's losses put exactly 0.0 gradient on T's parameters.
* G8: the strategic token enters T (detached, on in the arm, off in the control), and through T's code reaches O.
* G6, G7: L2 candidates are seeded from the vocabulary; T's constraints enter the level-2 and level-1 costs through the decoded kinematics, with weight 0 as the control.
* G5, G4: the goal head reads vision latents, nav and the vision-only context; situation-classifier output stays out.
* G2: `T3 = 4` (12 s, the ladder's L3 band end) is the arm of record, `T3 = 2` the control, with window counts reported.
* G1: the planner runs at the context the model was trained with (`plan_ctx`), read at both 1 and 3. A later retrain with `--wm-ctx 1 1 1` is the paper-equal arm if the read differs.
* G3: `stitch = l1_full` (level 1 plans all 30 knots against all level-2 sub-goals) is added beside the decode.

## 5. The PI's ruling on G9 (2026-11-08)
Section 4's hand-over state is not built. The PI ruled that all three layers read the current ego state, that T's goals and constraints are the actions conditioning O, and that the output is one or several trajectories up to 6 s (`REQ_HORIZONS.md` R-EGO, R-T2O, R-OUT; `SPEC_HWM.md` 9.1). G9 is met by the edge `e_code` that exists (`v8f_mini.py:933`), with T's labels re-cut to the band and per-band losses on O's single 6 s output.
