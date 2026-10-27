# PREREG_ANTICHEAT_REFCV8 -- does refcv8 (EMA-48,852 + the shipped H2cn stage) beat CAMERA-BLIND and the EGO-ONLY FLOOR?

EvalFlyWheel agent, 2026-10-27 (Europe/Berlin). Written and SEALED (sha256 + time in `PREREG_ANTICHEAT_REFCV8.sha256`) BEFORE any
score of the blind arm, the floor, or any REAL-vs-blind / REAL-vs-floor contrast was read. Amendments, if any, are separate files, dated, and
sealed before the number they could influence is read.

**Why.** PI 2026-10-27 (binding): refcv9 is warm-started from refcv8; the PI requires proof that the planner reacts to the CAMERA and does
not merely extrapolate ego speed / dynamics (W4 CAMERA-BLIND, W5 EGO-ONLY FLOOR in `DRAFT_PREREG_REFCV9_WARM.md`). This package builds both
instruments and reads them on the **refcv8 reference** first: if refcv8 already fails, the warm start inherits the problem.

**Tier.** T1 self-action OPEN LOOP on logged eval139 frames, per-frame nav feed. **Not driving performance.**
**Estimator.** paired episode-cluster bootstrap, B 2000, bootstrap seed 0, over EPISODES, models FIXED
(`pair_read_r8.paired_cell` -> `taniteval.ci.paired_episode_cluster_bootstrap`). It answers *"would another draw of EPISODES say this?"* only.
It does NOT answer *another training run* (one training seed) nor *another inference run* (this read is inference seed 0; seed 1 is beside).
**Disclosures carried by every number.** (i) refcv8's speed band (the fed ceiling `lim`) is **future-derived** (the ego's own max speed 2-6 s
ahead; leak 0.224 vs the 0.05 bar); (ii) its per-frame nav token is ego-future-derived (optimistic by construction); (iii) one training seed.

## 1. Arms (all scored by the SAME code on the SAME windows)

| arm | what | built from |
|---|---|---|
| **REAL_REG** | refcv8 EMA-48,852 + the shipped H2cn stage = the registered `H2CN_DEPLOY8` | battery tree plan (inference seed 0) + L1 os-only `z_tac` bank; reproduced bit for bit from the frozen stage `H2CN_STAGE_FROZEN.pt` (sha256 `f293b7bd...`) |
| **REAL_OS** | the same stage applied to the **os-only identity roll** | os-only plan8 + `z_tac` of `ztac_eval_s0.npz` (the BASE bank) |
| **BLIND** | the same stage applied to the **camera-blind os-only roll** | plan8 + `z_tac` of the blind roll (section 2) |
| **FLOOR** | EGO-ONLY ridge floor (section 3) | train139 windows only |
| **CONST** | train-mean trajectory (control beside the floor) | train139 windows only |
| beside | RAW_REAL / RAW_BLIND = the model's emitted plan WITHOUT the stage (os-only identity vs blind); HP = `ha0_ext_pose` (kinematic prior); FLOOR_NL = gradient-boosted floor on the same features | -- |

**Why REAL_OS exists.** At inference seed 0 the battery tree's `os` plan (the FULL arm-set draw) differs from the os-only child's plan on 4,630
of 4,754 S2 windows (`ship_r8_sub/README`, K-SSOS note) -- a different inference DRAW of the same checkpoint. REAL_REG contrasted with BLIND would
then carry a draw difference on top of the camera difference. **The BINDING W4 contrast is REAL_OS - BLIND** (identical draw, identical code
path, only the camera pixels differ). REAL_REG - BLIND is reported beside, and REAL_REG - REAL_OS is reported as the draw floor of that pair.
**The BINDING W5 contrast is REAL_REG - FLOOR** (REAL_REG is the registered shipped arm; REAL_OS - FLOOR beside).

## 2. W4 instrument: the camera-blind roll

* **Code:** `code/camblind_roll.py` = a COPY of `refcv9_l1/code/l1_ztac_roll.py` (eval mode, `--os-only`) with ONE addition, a subclass of the battery's
  eval dataset class whose `_window_u8` returns the **frames of a donor window**. No battery file is edited.
* **Derangement (fixed, seed 0).** Episodes = the sorted episode indices present in the roll's restricted window index (139). Sattolo's algorithm
  with `random.Random(0)` over that sorted list gives a single 139-cycle (no fixed point). Donor window of own window `(e, t)` = the donor
  episode `pi(e)`'s window of the **same rank** within that episode's t-sorted window list (`min(rank, n_donor - 1)`).
* **What is replaced:** ONLY the camera pixels `frames[t : t+W]` (u8, 9 channels, W = 8). **What is NOT replaced (all ego/label inputs):** pose
  history, `v0`, ego state, nav token (per-frame), the speed band (`v_max_ms`, `v_max_valid`), perception grid/geometry (per-clip rig
  extrinsics: a calibration, not scene content -- disclosed; donors may cross the two PhysicalAI rigs), labels, future poses, the sampler seed
  and the number / order of forwards (so the DDIM noise draws are those of BASE).
* **Controls that must read known values** (any failure VOIDS the W4 read):
  * **(a) identity:** the same script with `--camera identity` reproduces the BASE bank `ztac_eval_s0.npz` (`traj_own`, `z_tac`, `sel_own`)
    **bit-equal on all 4,754 windows**, keyed by `(e_i, t)`; max abs difference must read exactly 0.0.
  * **(b) changed fraction:** the fraction of windows whose fed-frame digest (blake2b-8 of the uint8 slice) differs between the blind roll and
    the identity roll must read **100 %** (4,754 / 4,754); plus, inside the blind roll, the own-frame digest of every 25th window differs from
    the fed digest (100 % of the sample). The derangement assertions (no fixed point; every episode a donor exactly once) are printed.
  * **(c) ego inputs:** a per-window digest of EVERY tensor of the dataset item except `frames` (+ `nav_cmd`, `v0`, `ego_state` as passed to the
    model) is equal between the blind roll and the identity roll on **all windows**.
  * **(d) mutation arm:** the control-(b) checker fed a half-swapped digest array must read RED (`tests/test_anticheat_controls.py`).

## 3. W5 instrument: the EGO-ONLY FLOOR

* **Inputs = refcv8's ego inputs AS FED** (`progress_scalar.feats`, the H-R9-SCALAR-FLOOR-1 instrument, unchanged): `[1, v0, a0, omega0,
  v0*omega0, v0^2, a0*v0, min(lim,40), min(lim,40)-v0, lim_valid]`, d = 10, where `v0, a0, omega0` come from the observed pose window via
  `kinematic_prior.prior_controls('ha0_ext_pose')` / `ego_channels_from_poses` (the model's own ego channels) and `lim` is the **future-derived
  speed band** refcv8 receives. NO camera pixel, NO situation-classifier output, NO model latent, NO nav token (PI spec; see "beside").
* **Targets.** Per surface (S2: instants 0.5/1/1.5/2 s = slots 0-3; S6: instants 1..6 s = slots 1,3,4,5,6,7) the ego-frame positions of the human
  future, flattened: `y = vec(G)` (parametrisation DIRECT) or `y = vec(G - HP)` (RESIDUAL over the causal prior `ha0_ext_pose`).
* **Fit set.** TRAIN windows only: the A6 train139 tree, ceiling-compliant windows (S2 and S6 each; expected n 4,211 / 3,301 over 137 episodes
  -- the stage fits' own set). The eval139 windows are never seen by the fit.
* **Selection (train only).** Ridge (intercept unpenalised, features standardised on the fit rows); `lambda` in {1e-3 ... 1e3} (7 values) AND the
  parametrisation {DIRECT, RESIDUAL} chosen jointly by **5-fold episode-grouped CV SSE on the train rows** (fold of an episode = rank of a
  seed-0 permutation mod 5). The binding FLOOR is the argmin; both parametrisations' CV scores are printed.
* **Controls.** `CONST` (train-mean trajectory) scored beside; `n`, `d`, lambda, parametrisation, CV SSE, train R^2 printed; **known values:** a ridge
  fitted on a target that IS a feature (v0) recovers it (R^2 = 1 +- 1e-9); a ridge on a column-shuffled target reads R^2 <= 0 on held-out folds.
* **Function class, stated:** ridge, linear in 9 features that include the quadratic terms above. A negative result against this floor is a
  statement about THIS floor, not about "any ego-only predictor"; `FLOOR_NL` (HistGradientBoostingRegressor, same features, same train rows,
  default hyper-parameters, one model per output, `random_state=0`) is scored BESIDE (never a bar) to size that gap.
* **What the floor cannot produce:** class logits / head decisions. The battery's TACTICAL rows are TRAJECTORY-DERIVED (`TAC_traj_{lat,lon}_correct`:
  the plan's kinematics through the programme's canonical labeller), so they ARE computable for the floor from its positions; the **tactical-head
  decision quality** (logit-based rows, `tactical_v6_*.json`) is NOT computable for a ridge (no logits): n = 0 for those rows, reason stated.
  STRATEGIC: n = 0 (`--no-strategic`; no strategic head, no route label in this battery).
* **Beside, not in the spec:** the nav token is also fed to refcv8; a floor WITHOUT it can lose on turning windows for non-camera reasons. This
  is exactly what W4 isolates; no nav-augmented floor is built in this package (stated as a limit).

## 4. Scoring set, rows, orientation

* Windows: eval139 **compliant** set (human max speed <= fed ceiling; the set SHIP-R8 reads), both surfaces S2 (0-2 s) and S6 (0-6 s);
  `all` windows beside. Inference seed 0 binding; seed 1 beside if its blind roll completes.
* Scorer: `pair_read_r8.components` (= `refav1_arm._components` + the SPEC-5 cells) on the arm's positions at the surface's instants against the
  logged future `G`, `dt` from the panel -- the battery's own four-family code; the registered TAC rows come out of the same call.
  Paired cell = `new - ref`.
* **PRIMARY ROWS (binding, fixed now):** per family, the first registered headline row (`pair_read_r8.METRICS` order):
  **ADE** `ade_m` (S2, S6) -- **LONGITUDINAL** `LON_speed_mae_mps` (S2) -- **LATERAL** `LAT_heading_mae_deg` (S2) -- **TACTICAL**
  `TAC_traj_lat_correct` and `TAC_traj_lon_correct` (S2, S6; clause-5 rows) -- **STRATEGIC** n = 0.
  Every other registered row (14 metrics x 2 surfaces x {compliant, all}) is tabulated BESIDE with its CI; none is a bar.
* **Orientation.** Error rows (ADE, LON_*, LAT_*): REAL better <=> `delta < 0`; **separated better** <=> CI upper bound < 0. Accuracy rows (TAC_*,
  `LON_speed_within_0p5`): REAL better <=> `delta > 0`; separated better <=> CI lower bound > 0. "Separated worse" is the mirror.

## 5. Verdict rule and both outcomes (committed before the data)

* **W4_ref8 / W5_ref8 per family:** the family PASSES iff **every** primary row of that family is separated better (REAL_OS vs BLIND for W4;
  REAL_REG vs FLOOR for W5) at inference seed 0. Overall PASS iff every family with n > 0 passes. A row separated WORSE is a FAIL of its family
  and is named.
* **Any control (a)-(d), K-BASE, K-ID below, or the floor's known-value controls failing => VOID: no W4/W5 statement is made.**
* **K-BASE:** REAL_REG built here equals the landed `H2CN_DEPLOY8` package (`out/arms/h2cn/H2CN_DEPLOY8/DEPLOY8_POS_*_s0.npz`, `S2_pos`, `S6_pos`) bit
  for bit on every window; and its cells reproduce the landed ones (E1 S2 `ade_m` minus `ha0_ext_pose`, compliant: -0.0471 [-0.0585, -0.0363];
  E3 S6: -1.1975 [-1.4942, -0.9168]) to the printed digits. **K-ID:** a REAL-vs-REAL contrast reads delta = 0 with a zero-width CI; GT scored
  against itself reads TAC = 1.0 exactly and zero error.
* **Outcome A -- refcv8 PASSES both W4 and W5.** The planner demonstrably uses the camera beyond ego dynamics at refcv8; the warm start inherits a
  PASS and **refcv9 must keep it**: REFCV9-W is held to the same rows, and a REFCV9-W that loses a family that refcv8 passes is reported first.
* **Outcome B -- refcv8 FAILS W4 and/or W5.** The failing families / rows are named in the first line of the result and **become refcv9's first
  problem**: the warm start inherits an unsolved defect, and the draft's own branch applies (FAIL on W4/W5 => no ship request, reported before any
  other row). The cheapest next experiment is then registered, not the verdict alone: (1) read seed 1 / the draw-floor to separate camera-use from
  inference noise; (2) per-family localisation (which instants, which windows -- compliant vs all); (3) the nav-augmented floor, to see whether the
  residual advantage over ego-only is nav, not pixels.
* **Mixed (passes one control, fails the other):** reported as such with both lists; no averaging.

## 6. What this does and does not show

A REAL > BLIND separation shows the emitted plan **depends on the scene content of its own window**, with the sampler draw held identical. It does
not show the dependence is *correct* (a mismatched scene is out-of-distribution, so BLIND can degrade through OOD brittleness beyond "no
camera"). A REAL > FLOOR separation shows the model carries information beyond **this** ridge on **these** ego features. The two together are the
PI's test; neither is a driving claim (T1 open loop; no re-perception sim).

## 7. Deliverable and run record

All paths under `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-27-refcv9-anticheat/` in the hold `C:/Users/Admin/qland/hold_anticheat_r8/code/`.
No git, no staging, nothing written to G:, no Thor contact, `launch_gate.py` untouched. GPU: dev-box RTX 4060 under the shared lock
(`C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock` via `with_gpu_lock.py`); a foreign lock is never broken. No clip id is printed or written:
clips appear as sha12 only.
