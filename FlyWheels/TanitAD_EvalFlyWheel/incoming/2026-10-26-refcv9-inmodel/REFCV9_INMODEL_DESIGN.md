# REFCV9_INMODEL_DESIGN -- the clean-split progress head and the factored lon decode, moved INTO the model

*Architecture FlyWheel for the Master Mind, 2026-10-26. Status: DESIGN v1, 0 GPU. Source read from branch tip
`agent/arch-inf-20260803` @ `425762053c75`. Tier of every quoted number: T1 self-action open loop on logged eval139 frames,
paired episode-cluster bootstrap over EPISODES (models fixed), ONE training seed of refcv8, unless stated.*

## 0. Recommendation (one design): **R9-PF -- trunk-blind, regime-factored progress head + in-forward re-timing**

* **Path** = the planner's native selected candidate (`fan[idx]`, unchanged). **Speed** = a new head that never trains the trunk
  and is fitted ONLY on clips the trunk never saw. The head makes a **3-way longitudinal decision** (decel / cruise / accel, the
  MD/FD metric's own GT definition) and emits **one progress profile per decision**; inference takes the profile of the argmax
  decision (the in-model form of "self-sharpened lon applied to the speed only"). The selected path is re-timed to that progress
  BEFORE `plan_vmax_clamp`, so every emitted point lies on the selected polyline or its end ray (lateral untouched by construction)
  and the fed ceiling stays the last word.
* **Schedule** = two stages. Stage 1 trains the trunk exactly as refcv8 does (flags off; bit-identical). Stage 2
  (`--r9-calib-stage`) loads the FINAL EMA checkpoint, freezes everything, runs the model in eval mode, and fits only the R9 heads on
  an EXTERNAL clean split (the 300-clip fresh set, disjoint from train4572 and eval139 -- the H2cn split, K-DISJ). The heads are
  written into the checkpoint and run in the forward: **no post-hoc stage at eval**.
* Why this and not a joint form: it is H2cn's MEASURED recipe (fit on fresh clips from frozen features) moved into the model, plus
  the one ingredient the post-hoc frontier says is missing (a clean-split DECISION). Every joint alternative either re-creates the
  L1 memorisation trap or costs parity (§1).

## 1. (a) Avoiding the L1C memorisation trap -- options weighed

The trap, MEASURED (PLAN_REFCV9 §9; L1 package `…/Research/2026-10-25-refcv9-l1/`): a progress head fitted on refcv8's z_tac over
refcv8's OWN training clips is separated WORSE than the shipped stage (E1 +0.0264 vs H0 -0.0343); held-out-EPISODE R² inside train
0.54 / 0.25 / 0.27 / 0.33 becomes 0.41 / -0.27 / -0.49 / -0.55 on eval while z_tac's distribution barely moves (domain AUC 0.547).
So **within-train cross-validation is fooled**: regularisation chosen on train clips cannot see the trap. Fitted on clips refcv8 never
saw, the same features win (H2cn: SHIP-R8 A4 PASS, E3 -1.198 vs RETIME -0.868, FD 0.137 -> 0.060 separated; A4d PASS at fit seeds
0/1/2 -- MEASURED, PLAN §11-§12). A second, independent sign: refcv8's jointly trained lon head reads decel AUC 0.685 where a z_tac
probe reads 0.762 (INHERITED from PLAN §10/§12, artifact not re-opened here).

| option | trap? | parity cost | compute | evidence | verdict |
|---|---|---|---|---|---|
| J0 joint, full grad, on training clips | **yes**, strongest form (head AND trunk fit in-sample) | none | +0 | L1 O2; the joint lon head 0.685 < clean probe 0.762 | rejected |
| J1 joint, stop-grad, + dropout / feature noise, training clips | **yes**: a regulariser tuned in-sample; L1's in-train CV already picked the regularisation and failed | none | +0 | L1 O2 | rejected |
| J2 reserved split INSIDE the corpus, trunk losses masked on it, head stop-grad | clean only if EVERY trunk loss term and every training-mode buffer update (BN stats, `update_tactical_prior`) is masked -- a silent-leak surface in a 13k-line trainer | **IS a re-selection** of the trunk's training set: -K clips (e.g. 300 = -6.6 % of the 4,572-clip view, INHERITED count). Manifest and skip-hash unchanged, but every refcv9-vs-refcv8 row then carries a corpus difference | -K/N trunk throughput | none | not recommended |
| J3 joint, stop-grad, on an EXTERNAL calibration stream | clean | none | +forward-only calib batches, ~5-10 % (ESTIMATED); a second loader | none direct; the head chases a moving trunk and is fit to RAW, not EMA, features | dominated by S2 |
| **S2 two-stage**: trunk as refcv8, then head-only fit on the external clean split from the FINAL EMA trunk | **clean** | **none** | dev-box roll of the split + minutes of fit (§5) | **H2cn is exactly this recipe, post-hoc** (MEASURED above) | **RECOMMENDED** |

**Parity.** The canonical corpus is not re-selected: stage 1 reads the same cache, manifest and skip-hash as refcv8; stage 2 reads
only the fresh split, which is outside the training view by construction (K-DISJ, `ship_r8_sub_common.fresh_disjointness`, with its
mutation arm). ⚠️ The fresh set is now a CALIBRATION split, so it can no longer serve as an unseen eval set for any model whose head
was fitted on it. ⚠️ The mapping between the 4,572-clip refc training view and the canonical `physicalai-train-e438721ae894`
(2,376 episodes, skip-hash `f09e44db`) is INHERITED, not re-verified here; nothing in this design touches either.

## 2. (b) The lon decision drives the SPEED only -- forward, loss, inference

**Inference path** (eval mode, one decoder pass):
`frames -> trunk -> z_tac` (unchanged) `-> planner -> fan -> E9 selection -> traj_sel` (unchanged) ; in parallel
`[z_tac.detach() ; x9] -> MLP(521 -> 128, GELU) -> {regime logits [3], profiles [3 x 8]}` ;
`r_hat = argmax(regime)` (or decel iff p_decel >= tau, tau fixed on the calibration split) ;
`s' = cummax(max(s_P + dS[r_hat], 0))` ; `traj = retime(traj_sel, s')` ; `traj = plan_vmax_clamp(traj)` (unchanged).

* `x9` = H2cn's measured scalar set (`progress_scalar.feats`): v0, a0, omega0, v0*omega0, v0^2, a0*v0, ceiling, ceiling - v0,
  ceiling-valid. v0 and omega0 from `ego_channels_from_poses` on the OBSERVED window, a0 from `prior_controls("ha0_ext_pose")` --
  all IMU-measurable at t0. **Curvature at t0 is NOT read** (`ego_state[:, 3]` is never touched; pinned by a mutation test).
* `s_P` = polyline arc of `residual_prior_path` (the prior the plan is composed on); `retime` = a torch port of
  `tanitad/retime_stage.polyline_geometry` / `eval_polyline` (float64-cross-checked against the numpy original).
* **Goal / situation rule.** The R9 heads read z_tac UPSTREAM of every head; they read no posterior, argmax or embedding of the
  lat/lon tactical heads, the scorer or any situation classifier, and their output feeds NOTHING but the re-timing (not `g2`, not
  the nav input, not selection, not the cap limit). Shared-trunk disclosure: z_tac also feeds `lat/lon_head_tac`, `tac_goal_head`
  and the E9 scorer (as today). The control arm `--r9-decision trunk_lon` reads the trunk lon head's argmax instead (the in-model
  SELFSHARP) -- a tactical decision head, not the situation classifier; it exists to attribute the decision SOURCE.

**Training loss** (stage 2; every row is a calibration row; inputs detached, trunk frozen in eval mode):
`L = w_prog * sum_k valid_k |s_P,k + dS[r*]_k - s_H,k| / sum_k valid_k  +  w_reg * CE(regime, r*)`, where `s_H` = polyline arc of
the human's own path (`traj_tgt`) and `r*` = GT regime from `dv = |p(6.0 s) - p(5.0 s)| / 1.0 s - v0` with +-1.5 m/s (the
`l2_common` definition, slots 50 / 60 of `V3_HORIZONS`). The profile is TEACHER-FORCED on r*: each profile is the conditional
median progress GIVEN the decision, so the L1-median shrink that made H2cn miss decels (MD 0.47 -> 0.53, n.s.) cannot average
decel windows into cruise ones; a wrong decision is a visible classification error, not a blurred speed.
**Why a decision of its own and not the trunk's lon head:** four post-hoc levers that re-weight refcv8's existing lon output (H2cn,
H2cn-L = A4c, SELFSHARP, SELCOR) all slide along ONE MD/FD frontier (A4c vs H2cn: MD -0.091, FD +0.044, both separated); the
GT-oracle pick of the same fan reads MD 0.075 / FD 0.057 (PLAN §12, MEASURED). The frontier is set by decision INFORMATION, so the
lever is a better decision fitted clean -- HYPOTHESIS until Phase A (DRAFT_PREREG) reads.

## 3. (c) Flags, reused modules, parameters

| flag (default) | effect |
|---|---|
| `--r9-progress-head` (off) | builds `model.r9` (`RefCV3Config.refcv9.enable`); off = nothing built, no code path entered |
| `--r9-decision {regime,trunk_lon}` (regime) | source of the speed decision |
| `--r9-hidden` (128), `--r9-decel-tau` (None = argmax), `--r9-seed` (20261026) | head width; operating point; private init stream |
| `--w-r9-prog` (0.0), `--w-r9-regime` (0.0) | loss weights; 0.0 default = the trainer's REFC_WEIGHT_GATES zero-default invariant; REQUIRED > 0 with the head (design 1.0 / 1.0; the regime CE may be 0 only under `trunk_lon`); gated on head + stage |
| `--r9-calib-stage` (off) | stage 2: freeze all non-R9 tensors, eval-mode forward, loss = R9 only. `--r9-progress-head` without it is REFUSED in training (a built head with no clean rows is the dead-seam class; J3 is not built) |

Reused: `retime_stage` geometry (ported), `kinematic_prior.prior_controls("ha0_ext_pose")`, `ego_history.ego_channels_from_poses`,
`residual_prior_path`, `refb_labels.waypoint_targets` (`traj_tgt`), `plan_vmax_clamp` (after, unchanged), `r8c.LON6_V7_IDS`, the
`enable_refcv8` fork-RNG init pattern (an OFF build and every module built after it are byte-identical), `declared_vs_built`.
New file: `stack/tanitad/refs/refcv9_progress.py`. **Parameters: 70,299 trainable at d_tac 512** (body 66,816 + regime 387 +
profiles 3,096; MEASURED by `count_params` in the tests; refcv8's d_tac 512 is UNVERIFIED in its config.json, annex §2.4) = 0.073 %
of refcv8's 96,435,990 (INHERITED, annex). Buffers: 2 x 9 scalar + 2 x d_tac latent standardisation stats, set by stage 2.

## 4. (d) Deliberate-regression (mutation) tests -- each must go RED

| change | test (literal expectation) | mutation that must go RED |
|---|---|---|
| flags off | every forward output and every state-dict tensor bit-equal to a build of the PRISTINE tip (frozen golden digests); flag ON leaves every non-R9 output and tensor bit-equal | build the head from the global RNG (moves the init of later modules) |
| stop-grad | after `L_r9.backward()`, every non-R9 parameter's grad is None or exactly 0 | `z_tac` not detached -> trunk grad > 0 |
| lateral | every re-timed point is on the selected polyline / end ray (distance < 1e-6 m) | a position re-scale (`RS_LON`-style) -> off-path on a curved plan |
| retime geometry | straight line: s' -> (s', 0) exactly; s' = own arc -> plan returned; torch == `retime_stage` (float64, 1e-9) | drop the end-ray extension |
| factored decision | inference profile == `profiles[argmax]` on a crafted head | soft (expected) profile |
| admissibility | perturbing `ego_state[:, 3]` (curvature) changes no R9 output | a feature builder that reads it |
| clean rows only | training refuses `--r9-progress-head` without `--r9-calib-stage`; stage 2 freezes every non-R9 tensor | a freeze that leaves one trunk tensor trainable |
| DVB | every new flag has a registry entry; built iff declared | flag declared, head not built |

## 5. (e) Inference cost vs the 100 ms Thor budget (ESTIMATED)

~70k MACs per window (two GEMMs) + a 9-scalar feature build + an S x S = 64-compare polyline evaluation: compute is negligible; the
cost is kernel launches, ~12-20 small kernels at ~5-20 us each on Thor = **~0.1-0.3 ms un-fused, <0.05 ms under a CUDA graph /
TensorRT fusion** (ESTIMATED; not timed on Thor -- Thor is off-limits). = **<= 0.3 % of 100 ms**. It REMOVES no decoder pass and adds
none; the composed two-pass FACTORED_DEPLOY8 form (SHIP-R8 A5) would instead cost one extra decoder pass (not measured here).
Stage-2 cost: one dev-box roll of the fresh split (~9.1k S2 windows at ~0.57 s/window, rate from annex §3 = ~1.5 h, ESTIMATED) or
the in-trainer stage 2 on the same rows; the CPU fit of H2cn took 392 s (S2) + 241 s (S6) (MEASURED, `H2CN_FREEZE_INFO.json`).

## 6. Risks and open items

* **The clean split is small** (297 clips / 9,135 S2 rows, MEASURED in `H2CN_FREEZE_INFO.json`); a 3-regime head has ~1/5 decel
  rows (S6 eval: 639 GT-decel of the compliant set). If Phase A reads the decision as under-powered, the cheapest lever is MORE
  fresh clips (V-C; availability UNVERIFIED), not a joint form.
* **Fit-seed variance is real** (A4d: seed 0 was the best of 3; others S6 ADE +0.07 m separated) -> fit seeds 0/1/2 are mandatory.
* **Cap interaction:** H2cn plans above the fed ceiling ~3x as often as RETIME (PLAN §11). The cap stays after the re-timing; the
  compliance row is carried.
* **PI rulings needed:** (i) does a stage-2 calibration fit written into the checkpoint satisfy B9-1 ("the gain must be in the
  model")? (ii) the fresh split becomes a calibration split; (iii) Phase B compute.

## 7. Implemented and measured (tree copy `F:/refcv9_inmodel/tree`, CPU)

* **Flags off = pristine, bit for bit (MEASURED):** `tools/bitid_digest.py` digests every forward output (eval + train) and every
  state tensor of the refcv8 CPU rig in three builds (refcv7 / refcv8 / refcv8 + vmax: 234 / 272 / 280 outputs, 391 / 408 / 408
  tensors) on the PRISTINE tip and on the modified tree: `ALL_BIT_IDENTICAL True` (`bitid_REPORT_pristine_vs_modified_flags_off.json`).
  Controls: pristine vs pristine re-run True (determinism); one digest mutated -> False (`bitid_COMPARATOR_RED_CONTROL.json`).
* **Flag ON** leaves every non-R9 state tensor and every training-mode output bit-identical, and at eval changes exactly
  `traj` / `wp_seq` (`traj_preretime` == the OFF plan) -- `tests/test_refcv9_progress.py`, with the global-RNG mutation RED.
* **Smoke (MEASURED, `smoke_steps/SMOKE_R9_STEPS.json`):** the trainer's own pin + `compute_losses_v3` + calibration freeze + G-DVB,
  10 AdamW steps on the CI-synthetic corpus with ego history (120 windows, batch 4): loss finite and equal to the R9 loss every step
  (1.1117 -> 0.9172), all 6 head tensors non-zero gradient every step, frozen-gradient leak exactly 0, every non-R9 tensor bit-equal
  after 10 steps (8,859 trainable / 161,489 frozen on the smoke config), G-DVB clean on every R9 lever and on the freeze.
  ⚠️ **`train()` end-to-end with R9 on `--synth-episodes` is BLOCKED**: the synthetic loop's batches carry no `pose_hist`, and the
  trainer refuses `--ego-history` without it (MEASURED, `smoke_r9.log`). A real-cache `train()` smoke is the next check.
