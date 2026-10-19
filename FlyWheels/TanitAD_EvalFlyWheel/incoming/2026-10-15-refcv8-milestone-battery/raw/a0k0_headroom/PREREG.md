# PREREG — a0/κ0 HEADROOM for refcv8 (zero training, CPU, banked panels only)

Written 2026-10-18, **before any blend number was computed**. Instrument: `PKG/code/a0k0_headroom_r8.py` (NEW file).
Status of every arm here: **REQUIRES PI RULING** — the measured longitudinal acceleration a0 and curvature κ0 at t0 are
NOT covered by the 2026-09-02 ruling (that admitted v0 at its cycle time and forbade future GT dynamics); the codebase
already records κ0-from-recorded-steer as *"unruled at inference"* (`stack/tanitad/channel_admissibility.py:185-187`,
SPEC_REFCV7 §10 A5). Nothing below is a refcv8 bar result.

## Question
How much would the measured (a0, κ0) at t0 be worth to refcv8 EMA-30k at inference, if combined with refcv8's own plan?
Operationalised as a **cross-fitted convex blend** of refcv8's plan (`os`, fed v0 only) and the echo `ha0_ext`
(constant a0 and κ0 from the measured t0 state, `tanitad/eval/echo_gate.py::ha0_ext`).

## Arms (all on the banked panels; plans in the ego frame at t0, x = along, y = cross — the battery's convention)
* **BLEND (primary)** — per instant k and per axis c ∈ {along x, cross y}: `p_k,c = w_k,c · os_k,c + (1 − w_k,c) · ha0_ext_k,c`,
  `w ∈ [0, 1]`. S2 has 4 instants × 2 axes = 8 weights; S6 has 6 × 2 = 12.
* **GATED** — the same, with a separate weight set per **v0 bin** (`s6_lon_decomp.V0_EDGES`: <2 | 2-8 | 8-15 | 15-25 | ≥25 m/s;
  v0 is admissible — refcv8 is fed it). A bin with < 30 fit windows in a fold falls back to that fold's BLEND weights.
* **BLEND_POS (secondary)** — one weight per instant applied to both axes (positions).
* **Fitting.** Objective = the scored metric itself: mean window ADE over the fit windows. The objective is separable over instants
  (ADE = mean over instants of a per-instant norm; w_k only touches instant k), so each instant's (w_x, w_y) is found by
  exhaustive grid search on {0.00, 0.01, …, 1.00}² (ties → the grid point with the larger w, i.e. toward refcv8). Deterministic, no optimiser.
* **Cross-fitting.** 5 EPISODE folds: the 139 eval episodes (identity = `clip_index`) sorted, permuted with
  `numpy.random.default_rng(20261018)`, fold = rank mod 5. Weights for fold f are fit on the scored set's windows of the other 4 folds
  and applied to fold f only. Fit set = the same window set as scored (compliant fits on compliant windows, all fits on all windows).
  Never fit on the scored episodes.
* Per inference seed separately (each seed's `os` is blended with the same `ha0_ext`).

## Cells (both trees: EMA-30k `step30000ema_unc` = headline, raw-30k `step30000_unc`; both inference seeds; S2 0-2 s and S6 0-6 s;
compliant set (A-R8-CLAMP-2 mask, sha `4ebc6b86…`) and all windows)
1. **E1 form**: BLEND − ha0_ext, `ade_m` (on S6 this is the E3 form).
2. **NR form**: BLEND − refcv7-50,400 (`b_refcv7_50400_s{seed}`), every `pair_read_r8.METRICS` row (families ADE, LONGITUDINAL,
   LATERAL, TACTICAL; STRATEGIC NOT APPLICABLE, n = 0, layer off — `--no-strategic`).
3. **What a0/κ0 adds**: BLEND − refcv8 `os`, every METRICS row.
4. Same three for GATED. BLEND_POS: cells 1 and 3 on `ade_m`.

Estimator: `pair_read_r8.paired_cell` → `taniteval.ci.paired_episode_cluster_bootstrap` (B 2000, seed 0, 95 % percentile) — answers
**"another draw of EPISODES" only**; single training seed; seed 1 is the inference replicate beside seed 0. The fold assignment is a
fourth source of variance this interval does not see — reported as a sensitivity (below), not folded into the CI.
Tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**. EARLY READ of banked step 30,000.
On all windows the read is the panel's **uncapped** `os` (the banked PAIRED_READ path); on the compliant set capped == uncapped
(`n_changed_plans_on_compliant_windows` 0, asserted from the compliant tree json, both trees, both seeds).

## Controls (the tool refuses to write A0K0_HEADROOM.json unless (a), (b), (e) pass)
* **(a) w = 1 identity** — the blend at w ≡ 1 reproduces the banked `os` cells EXACTLY (delta, lo, hi, n_windows, n_episodes,
  separated, n_dropped_nonfinite) against `raw/compliant_reread/trees/<tree>.json` (`compliant.uncapped` and `all_window.uncapped`):
  `ade_m` vs ha0_ext and every METRICS row vs refcv7-50,400.
* **(b) w = 0 identity** — the blend at w ≡ 0 equals `ha0_ext` bit for bit, and BLEND − ha0_ext reads delta = lo = hi = 0.
* **(c) in-sample weights** (fit on all scored episodes, applied to the same) reported beside every primary E1 and a0/κ0 cell:
  the optimism gap = cross-fitted − in-sample. By construction in-sample BLEND − ha0_ext ≤ 0 and BLEND − os ≤ 0 (w ≡ 0 and
  w ≡ 1 are feasible) — a known-sign control.
* **(d) fold-shuffled control** — folds assigned per WINDOW (rng 20261018) instead of per episode: overlapping windows of one
  episode then sit on both sides of the fit, so it leaks within-episode correlation. Expected ordering (not a bar):
  in-sample ≤ window-fold ≤ episode-fold in most cells. Plus a **fold-reassignment sensitivity**: 10 alternative episode-fold
  seeds (1..10), point estimate of the E1-form BLEND cell only; the range is reported.
* **(e) v0 cross-check** — the dump extras' v0 (used for GATED bins) equals the panel files' own `v0`, every window.
* **(f) analytic fit targets** (pytest): g = 0.3·o + 0.7·h exactly → fitted w = 0.30 on both axes; g = o → w = 1; g = h → w = 0.
* **(g) deliberate regression** (pytest): a LEAKY cross-fit (held-out fold included in its own fit) must be detected — the
  held-out fold's GT is perturbed and its applied weights must not move under the correct cross-fit and must move under the leak.

## Committed readings — BOTH outcomes, written before the data
* **E1-form BLEND − ha0_ext on EMA-30k compliant S2 < 0 and SEPARATED at both seeds** ⇒ refcv8 + (a0, κ0) at t0 clears the
  echo by a convex blend alone; the measured headroom is the magnitude. Circular: the arm consumes ha0_ext's own inputs, so it is
  **NOT a BAR-R8-E1 pass** and is never quoted as one. It is the evidence a PI ruling on admitting (a0, κ0) would rest on.
* **Not separated** ⇒ a linear convex blend does not buy separation. This is a **weak lower bound on a learned use**
  (CLAUDE.md: a negative from a linear probe is not a negative about learnability) — it does not say a0/κ0 are worthless.
* **BLEND − os separated < 0** ⇒ (a0, κ0) carry information refcv8 does not recover from vision on these windows; magnitude is
  compared to the inference-seed floor |mean ADE(os s0) − mean ADE(os s1)| on the same set. An effect within 2× that floor is
  reported as NOT PROVEN.
* **NR form**: any METRICS row where BLEND − refcv7 is separated WORSE (lo > 0) at both seeds is reported as a regression the
  blend introduces, per family.
* The blend is a test-time combination of two plans, not a trained model; a learned input (refcv8.x fine-tune / refcv9) could do
  better or worse. "Upper-bound-ish" for a linear combiner, lower bound for a learned one.

---

## AMENDMENT 1 — PI RULING 2026-10-18 ~23:58 Berlin (relayed by the Master Mind), written BEFORE any blend or ha0_imu number was computed

PI, verbatim (as relayed): *"measured accelration is ok and yaw rate is ok since they can be measured by imus but not curvature"*.
This **supersedes the (a0, κ0) framing above**. The admissible ego set at t0 is **(v0, a0, ψ̇0)**; κ0 (the egomotion curvature /
recorded steer = atan(L·κ)) is **INADMISSIBLE**.

* **New admissible echo `ha0_imu`.** Hold a0 and ψ̇0 constant over the horizon; integrate at the native 0.1 s tick with the
  programme unicycle's update ORDER (`tanitad/models/kinematic.py::rollout_unicycle`: position on the speed at the START of the step,
  heading advanced next, v updated LAST and floored at 0), but with **heading rate = ψ̇0 held** instead of `v·κ0`; index-select onto the
  panel grid exactly as `refcv3_arm.integrate_select` does.
  * **a0 (primary)** = the SAME backward speed difference `ha0_ext` holds: `(v[i0] − v[i0−1]) / 0.1` (`refav1_arm.hold_ext_controls`,
    stride 1), read from the panel decisions' `ep_poses[:, 3]` at i0 = `ws` (asserted: `ep_poses[ws] == pose_last`). So `ha0_imu` differs
    from `ha0_ext` ONLY in the lateral channel. Sensitivity beside it: `ha0_imu_ax` with a0 = the measured `ax` (`ego_t0[:, 1]`).
  * **ψ̇0** = BACKWARD difference of the pose yaw: `wrap(yaw[i0] − yaw[i0−1]) / 0.1` from `ep_poses[:, 2]`. Never forward, never centred.
* **Validation (written into the JSON; the tool refuses to write results if V1 or V2 fails):**
  V1 — `ha0_ext` rebuilt from the panel's own decisions (a0 backward difference, κ0 = `ego_t0[:, 3]`) through the battery's ONE integrator
  (`refa_v1_plan.unicycle_paths`, units kappa) reproduces the panel's `ha0_ext` (max abs diff ≤ 1e-3 m; stated).
  V2 — the `ha0_imu` integrator with ψ̇0 := κ0·v0 and a0 = 0 reproduces that same integrator's (a0 = 0, κ0) path (max abs diff stated;
  must be ≤ 1e-4 m). V3 — the rate of windows where ‖ha0_imu − ha0_ext‖ > 0.1 m at 2 s (and at 6 s on S6) is reported.
* **Arms are now built on `ha0_imu`, not `ha0_ext`:** BLEND_imu / GATED_imu / BLEND_imu_POS, cross-fitted, with IS / WF controls, exactly
  as specified above. w ≡ 0 now reproduces `ha0_imu` bit for bit. A BLEND on `ha0_ext` is kept as ONE reference row only, labelled
  INADMISSIBLE (curvature), to price what the curvature exclusion costs.
* **Reported cells (order fixed):** (i) **E1 AS REGISTERED, os − ha0_ext — the BAR OF RECORD, unchanged, reported first**; (ii) os − ha0_imu
  (the admissible-reference read); (iii) BLEND_imu − ha0_ext and BLEND_imu − ha0_imu; (iv) BLEND_imu − refcv7-50,400 on the four families;
  (v) BLEND_imu − os (what admissible a0 / ψ̇0 add). S2 and S6, compliant + all windows, both inference seeds, EMA-30k and raw-30k.
* **Disclosure (binding sentence):** *The admissible-reference read (os − ha0_imu) is PRE-REGISTERED NOW, before any final-checkpoint
  (step 48,852) data exist, as a BESIDE-ROW to E1 — it was motivated AFTER seeing the 30k read (E1 narrowly failing against an echo that
  holds the inadmissible κ0), and it never replaces E1 as registered, which remains the bar of record.*
* Committed readings carry over with ha0_imu in place of ha0_ext. Additionally: if os − ha0_imu is separated < 0 at both seeds while
  os − ha0_ext is not, the E1 failure is attributable to the curvature channel the PI has now ruled out of inference — a statement about
  the REFERENCE, not a pass of the bar. BLEND_imu − ha0_ext is still circular-adjacent (it uses a0, which ha0_ext also holds) and is
  never quoted as a BAR-R8-E1 pass.

---

## AMENDMENT 2 — premise correction + NEW PRIMARY ARM (Master Mind, 2026-10-19), written BEFORE any blend, prior or ha0_imu number

**The original brief's premise "refcv8 is fed v0 only" was WRONG.** Verified at source on the run record
(`E:/refcv8_eval_kit/ckpt/config.json['argv']`): the run carries **`--ego-history`** (the past ego track — speed, longitudinal
acceleration and yaw rate per step, past only; the PI-admissible set) and **`--residual-prior ha0_ext_pose`**: refcv8's plan is a
causal kinematic prior **plus** a learned residual. The prior (`stack/tanitad/models/kinematic_prior.py::prior_controls`, mode
`ha0_ext_pose`) holds a0 = the backward speed difference and κ0 = ω0 / max(v0, 2.0) clamped to ±0.3, with ω0 the backward yaw
difference — both from the ego-history encoder's own derivation (`ego_history.py:96-98`). It reads ONLY admissible measurements.
The E1 echo `ha0_ext` instead reads κ0 from the recorded STEER channel — the curvature the PI ruled inadmissible on 2026-10-18.
So "refcv8 recovers from vision what (a0, κ0) buy over v0" (the e1_decomp framing) over-credits vision: refcv8 already HOLDS a0 and a
pose-derived κ0 inside its own prior. The ha0_ext BLEND arm is DROPPED (it consumes the inadmissible κ0).

**Re-prioritised arms (supersede the arm list above; controls, folds, estimator, tier, sets unchanged):**
* **PRIMARY reference `ha0_ext_pose`** — built with the library functions on the panel's own pose track: the observed window
  `ep_poses[ws−7 : ws+1]` (W = 8, asserted `ep_poses[ws] == pose_last`) → `kinematic_prior.prior_controls('ha0_ext_pose', ·)` →
  `kinematic_prior.prior_path(a0, κ0, v0, horizons_steps)`. CONTROL: it must reproduce the 2026-09-26 measurement
  (`…/2026-09-26-refcv7-residual-prior/raw/measure_prior_vs_echo.json`: S2 ha0_ext_pose − ha0_ext **+0.0055 [+0.0036, +0.0076]**,
  4,754 / 139, refcv6 battery's identical eval windows). On S2 all-windows an EXACT match is expected (same windows, same estimator);
  the S6 point (+0.0223) was measured on the 8-slot 0-6 s grid, not this panel's 6-slot grid, so it is compared, not required equal.
  The max-abs path difference vs ha0_ext and the 0-2 s ADE difference are stated on the compliant and all-window sets.
* **PRIMARY arm RS — RESIDUAL SCALING** (deployable, no new input: rescales the model's own residual):
  `p_k,c = prior_k,c + g_k,c · (os_k,c − prior_k,c)` computed as `g·os + (1 − g)·prior`, prior = `ha0_ext_pose`, per instant k and axis c,
  **g ∈ [0, 2]**, 5-episode-fold cross-fitted exactly as above. **GATED_RS**: per v0 bin (admissible). **RS_POS**: one gain per instant.
  Controls: g ≡ 1 reproduces the banked `os` cells exactly; g ≡ 0 is `ha0_ext_pose` bit for bit; IS and WF beside.
  Reported against ha0_ext (E1 AS REGISTERED form), ha0_ext_pose (admissible reference), ha0_imu (secondary), refcv7-50,400 (four
  families) and os (every METRICS row).
* **Plain identity `os − ha0_ext_pose`** ("does the learned residual beat its own causal prior"), every METRICS row, S2/S6,
  compliant/all, both seeds, both trees.
* **SECONDARY reference `ha0_imu`** (Amendment 1), with `os − ha0_imu`; one BLEND_imu_XF row (w ∈ [0, 1]) kept beside, nothing more.
* **Fitter (changed BEFORE any number, for compute: 16 cells × 10 fold-seed re-fits × gated sub-fits on a 201² grid would run > 1 h):**
  every weight/gain is the CONTINUOUS minimiser of the scored objective (mean window ADE, separable per instant) by L-BFGS-B with the
  analytic gradient inside the box, started at g = 1; it is VERIFIED against a coarse 0.1-step grid over the box (the L-BFGS-B loss must
  not exceed the grid minimum; if it does, the minimiser restarts from the grid point and the event is counted in the JSON). The
  objective is convex in the weights (an affine map inside a norm), so the minimiser is global.
* **Fold-seed sensitivity** (10 alternative episode-fold seeds) now applies to the PRIMARY arms RS_XF and GATED_RS_XF (points).
* Committed readings for RS (before data): RS − ha0_ext_pose separated < 0 at both seeds ⇒ the shrunk/scaled residual beats its own
  admissible prior; RS − ha0_ext separated < 0 at both seeds ⇒ a DEPLOYABLE test-time change (no new input) would clear the E1 FORM —
  but it is a post-hoc calibration fit on eval episodes (cross-fitted), so it is a LEVER ESTIMATE, never a bar pass; a deployable
  version fits g on TRAIN. RS − os separated < 0 ⇒ refcv8's residual is mis-scaled on these windows (over- or under-confident).
  Not separated ⇒ residual scaling buys nothing measurable at this n; any effect within 2× the inference-seed floor is NOT PROVEN.

---

## AMENDMENT 3 — FROZEN gains + forward test on the final checkpoint (Master Mind, 2026-10-19 00:15), written BEFORE any step-40,000 or step-48,852 data exists

(Verified at writing: `C:/Users/Admin/refcv8_eval_archive/battery/` holds no step40000 / step48852 tree.)
**Disclosure:** this amendment FOLLOWS the 30k exploratory read — the EMA-30k seed-0 smoke (compliant S2, RS_XF − ha0_ext
−0.0428 [−0.0538, −0.0321]) was seen before it was written. The 30k cross-fitted numbers are an honest estimate on 30k; the
gains below are the only thing carried forward, and nothing is re-tuned on the scored checkpoints.

* **Frozen gains** (`raw/a0k0_headroom/FROZEN_GAINS_ema30k.json`, written by `a0k0_headroom_r8.py freeze`): the same fitter as
  Amendment 2 (L-BFGS-B, verified on a 0.1 grid), fit on **EMA-30k (`step30000ema_unc`), ALL windows of the surface (S2 4,754 / S6 3,668
  windows, all 139 eval episodes), BOTH inference seeds POOLED** (each seed's `os` stacked against the same prior and GT).
  * `RS`: g[k, c] per instant × axis, box [0, 2], prior = `ha0_ext_pose`.
  * `GATED_RS`: one such table per v0 bin (`s6_lon_decomp.V0_EDGES`); a bin with < 30 UNIQUE fit windows uses the RS table.
  * One table per SURFACE GRID (S2: 0.5/1.0/1.5/2.0 s; S6: 1-6 s), applied only to that surface's panel.
  * The file carries the fit set, n, the sha256 of the input arrays and of every panel / decisions file read, and the script sha256.
    A compliant-set fit is recorded beside it for information and is NOT used by any arm.
* **Arms `RS_FROZEN30k` and `GATED_RS_FROZEN30k`** apply those gains UNCHANGED (`a0k0_headroom_r8.py forward --tree <tree>`) to the
  EMA trees of **step 40,000 and step 48,852**: p = g·os + (1 − g)·ha0_ext_pose, the prior rebuilt from that tree's own panel decisions
  (model-free; must equal the 30k prior bit for bit on the same windows).
* **Bars for those arms** (both inference seeds; ceiling-compliant set PRIMARY, all windows beside; `pair_read_r8.bar_E` /
  `bar_NR_metric` / `rollup_NR` semantics, NOT PROVEN within 2× the arm's own inference-seed floor):
  * **FZ-E1** — arm − ha0_ext, S2 ADE 0-2 s: < 0 and separated at both seeds (E1 AS REGISTERED form).
  * **FZ-E1P** — arm − ha0_ext_pose, S2 ADE: < 0 and separated at both seeds (the admissible-reference form).
  * **FZ-NR** — arm − refcv7-50,400 on the four families: no NR headline row separated WORSE at either seed (S2: ADE, speed MAE,
    |along| at 2 s, heading MAE, curvature MAE, yaw-rate MAE; S6: ADE) — LATERAL and TACTICAL rows printed in full, because shrinking
    the residual must not buy ADE by killing turns.
  * **FZ-E3** — arm − ha0_ext, S6 ADE 0-6 s: < 0 and separated at both seeds.
* `refcv8 os` itself is still scored only on E1/E3 AS REGISTERED; whether a frozen-gain post-process counts as refcv8's deployable output
  is a PI decision, escalated, not assumed. A deployable gain table would be fit on TRAIN; this one is fit on eval-30k and is honest
  only for checkpoints it was NOT fit on.
* Path check run now (not a bar): `forward --tree step30000_unc` (raw-30k weights of the same run — a different checkpoint, NOT an
  independent test) and `forward --tree step30000ema_unc` (in-sample by construction; must read ≤ the cross-fitted cells).
