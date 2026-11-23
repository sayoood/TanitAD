# FACTS_KINLAB: the DAgger own-state label and kinematic inputs at commit df250a00

Evidence class: MEASURED-FROM-SOURCE (tree exported by `git archive df250a00` into `F:/TanitAD-scratch/kinlab/tree`). Nothing run for this file. Paths are repo paths; line numbers are those of the df250a00 files.
Abbreviations: D = `stack/tanitad/data/v8f_dagger.py`, MD = `stack/tanitad/data/v8f_mini_data.py`, RV = `stack/tanitad/data/v8f_recovery.py`, M = `stack/tanitad/models/v8f_mini.py`, TR = `stack/scripts/v8f_mini_train.py`, FD = `stack/tanitad/refs/feasible_decode.py`.

## 1. Inputs at NOW of a swapped / appended sample

**Grid.** One cache step = 0.2 s (`DT = 0.2`, MD:49). The kinematics are computed on the NATIVE 10 Hz pose track (`DT_NATIVE = 0.1`, MD:337) and sampled at every second native frame, cache step j = native frame 2j (MD:350-361).

**kin3 under `--kin-mode yaw` = (v0, a0, kappa0)**, float32 in the batch, row 0 NaN.
* Built by `yaw_kin_from_track` (MD:350-361), logged side at MD:585: `v0` = track speed column [m/s]; `a0` = ONE-FRAME BACKWARD speed difference / 0.1 s [m/s^2] (`ego_channels_from_poses`, `ego_history.py:98`); `kappa0 = clamp(omega0 / max(v0, 2.0), +-0.3)` [1/m] (`kinematic_prior.py:212,215,341-343`), with `omega0` = one-frame wrapped yaw difference / 0.1 s [rad/s] (`ego_history.py:100`).
* **There is NO yaw-rate channel in kin3.** Channel 2 is a speed-normalised curvature, in the vehicle frame (yaw difference is frame-free). Yaw rate is recoverable as `kappa * max(v, 2)` only while |kappa| < 0.3 (not clamped).
* The measured yaw rate `omega` (rad/s) exists separately as `ch1` (MD:589) and reaches the model only through `act` (below).

**p10f [B, hmax, 8]** = `kin_features(kin3)` = (v, a, kappa, v^2, v*a, v*kappa, v^2*kappa, a*kappa) per history row (`taniteval/taniteval/floors_kin.py:66-70`, built MD:587). Row 0 is NaN in the store and nan_to_num'd to 0 in the item (MD:673). `p10_t` = the p10f row at NOW (MD:682). The model reads `p10f` at the tick frames through `_ego` = `e_ego(prior.standardise(f8))` (M:874-875, tokens at M:939/957/976), and O's prior reads `p10f` at its tick plus `base` (M:994).

**base [B, 30, 2]** = `kdx_base(kin3, "kappa")` (MD:390, 593; swap path D:593): the kinematic-damped path from (v0, a0, kappa0) at 0.2 s knots. It is O's prior path (M:994). So the own kin3 sets O's PRIOR PATH directly.

**Pose history** `pose` [B, hmax, 3] (x, y, yaw), `pose_t`, `pose_s` (MD:677, 687-689). The loss does not read the pose history for kinematics; it is used for `transform_ckpt(ck, pose_s, pose_t / pose_now)` (M:1046, 1068), i.e. to move S's held checkpoints between tick frames.

**act [B, hmax, 2]** = (cache `a`, `ch1` = yaw rate omega rad/s) (MD:678). It is the WM transition record and goes only into the WM aux loss (`wm(adapt, act)`, M:1230-1234). It is in `EGO_PERTURB_RECORD` (MD:766) = NOT touched by swap or append: it stays the LOGGED action while kin3 is the own one.

**Swap partition** (D:19-24; MD:760-766): REBUILT = p10f, base, kin3, p10_t, pose, pose_t, pose_s, nav (ch 3 only). KEPT = feats (images at r'), frame_ok, limit, limit_ok, a_t, a_s, pose_now. RECORD = act.

**Own values come from** the bank record: `snapshot` (`stack/scripts/v8f_dagger_bank.py:97-112`): p10f rows j-hmax+1..j, world pose, arc, and `kin3 = st.d["kin3"][j]` of the rolled own T1-kin state. Ticks 1 and 2 of an m = 10 rollout (D:56 `TICKS`, `REPLAN_STEPS = 10`). The own speed/acceleration are read from the executed steps (readout `pinned` or `causal_g`, `v8f_t1_readout.py`); the own omega comes from the executed headings.

## 2. What `reanchor_rows` changes and leaves

`reanchor_rows` (D:167-178) is a rigid motion of the own WORLD POSES [N, F, 3] taking own NOW (x, y, yaw) onto the logged NOW at r'; it is applied only in `swap_rows` to `pose` (D:595-601) and thereby to `pose_t`, `pose_s`, and the `rc` frame change (D:613).
* **Changes:** position and heading of the own pose history (the own lateral offset and heading offset relative to the route vanish: own NOW == logged NOW pose by construction; own heading error up to 30 deg, `OFF_ROUTE_HEAD_DEG`, D:49).
* **Leaves as the own value (untouched):** `kin3` (v, a, kappa), `p10f`, `p10_t`, `base`: D:590-593 write them straight from the record, no reanchor. So **speed, acceleration AND yaw rate/kappa stay own.** Re-anchoring does NOT erase the yaw-rate gap; it only removes the pose/heading offset. Relative geometry among own rows is exact, so the own history's internal turn rate survives in `pose` too.

## 3. Labels of a swapped row (all at r', the human's unless stated)

| tensor | built from | after swap (`--dagger-label human`) | after swap (`speed_v1`) |
|---|---|---|---|
| `gt` [B,30,2] = ego-frame human waypoints at 0.2 s steps t+1..t+30 (6 s) in the logged frame at r' | `gt_future(pg)` MD:597 (`floors_kin.py:213`) | human | **own-speed-consistent**: D:621 |
| `s_h` [B,30] human arc to each knot | `human_arc` MD:598 | human | `arc_from_label` D:622 (RV:255) |
| `gt_ok` | MD:681 | human | human |
| `lab` (tactical goal_y/goal_w, lat/lon classes, `cons`), `lab_t` | `_lab_pack` MD:607-615, 685 | human | human |
| `lab_s.rc` (route checkpoints) | `_lab_pack` | human world points, re-expressed into own S frame: `rc_frame_change` D:187, 613 | same |
| `lead_gap_h` [B,30] | MD:683 | human | human |
| `limit`, `limit_ok`, `nav` ch 0-2,4 | join | logged | logged |
| `act` | MD:678 | logged | logged |
| level label `kin3[:,0]` | `level_stage` TR:2187-2192, called BEFORE `dag.apply` (TR:3705-3715) | LOGGED speed | LOGGED speed |

* There is **no yaw-rate, acceleration or speed HEAD**. The only kinematic targets are `gt` (and `s_h` in the penalty rules) and the level head's speed (logged, vision-only input).
* **speed_v1** (`--dagger-label speed_v1`, TR:246, load at TR:613-614; `label_bank` RV:333): per ok record `speed_label` (RV:287-330): human polyline from r' in the human frame at r' (RV:306), step speeds `u_h[k] = (s[k]-s[k-1])/0.2` (RV:308), label speeds `label_steps` (RV:266): start at `v_own = p10f[i,-1,0]` (RV:358), move toward `u_h[k]` at `A_LABEL = 3.0 m/s^2` (RV:263), then follow `u_h`; label point k = the human polyline at the label's arc (RV:322). **The label is the human's PATH shifted along itself: only speed (arc timing) is own. Heading/curvature is the human polyline's own at every point, including the first step. Own a0 and own omega are not read by the label.** Refusals keep the human label (path_exhausted, arcs_decreasing, refused_infeasible).
* The older `recovery_label` (RV:156-226, `--recover-*` lever, not speed_v1) is the one that blends from the OWN pose/heading onto the human path with a Hermite ladder (`LADDER = (5,10,15,20,25,30)` RV:33, `_hermite` RV:146): the only existing construction that expresses an own-heading start in a positions label.

## 4. Which losses read them (M:1156-1244)

* `ade` / `fde` (`traj_terms`, M:459, 1183) read `gt`; `acc` (`accel_term`, M:497, 1190: second difference of plan vs gt; in the total only when `w_acc > 0`, M:1239) reads `gt`; `op`/`tac` band terms (`band_metrics`, M:466, only when declared) read `gt`; `launch_along` reads `gt` + logged-v0 mask. All are position-only. Yaw-rate and acceleration enter ONLY through the second difference of the position polyline.
* `pen_*` / `pens_*` consistency (M:1214-1226) read `lab`, `lead_gap_h`, `s_h`, `limit`.
* `ckpt` reads `lab_s.rc`; `goal/lat/lon/cons` read `lab_t`.
* `wm` reads `adapt` and the LOGGED `act` (M:1230-1234).
* The own kin3 reaches the loss only as INPUT: through `e_ego(p10f)` (all of S/T/O) and through O's prior `base` (M:994).

## 5. Limits available for a feasibility envelope

* `FD`: `A_MAX_MPS2 = 4.0` (FD:74), `KAPPA_MAX_1PM = 0.2` (FD:75), `MU_KAMM = 0.7` (FD:76, friction circle `mu*g` with `G = 9.80665`), `MIN_SPEED_MPS = 0.5` (FD:78), `DT_S = 0.5` (FD:79), `MARGIN = 0.01` (FD:90). **No yaw-rate limit and no jerk limit** (grep jerk: 0 hits in FD). Curvature limit 0.2 1/m is a CURVATURE limit; a yaw-rate limit would be `kappa * v` (at 10 m/s: 2 rad/s).
* `RV` mirrors them: `A_LIM = 3.96`, `K_LIM = 0.198`, `F_LIM = 6.80` (RV:30-32), checker `violations` (RV:82-95): accel, |kappa| (heading-change / dt / v_mid), friction `hypot(accel, lat_acc)` over the label with a virtual predecessor `(-v_own*dt, 0)` (own heading = 0, own speed). `A_LABEL = 3.0` is the speed label's own rate (RV:263). `kin3` kappa clamp is 0.3 (cap on the INPUT, above the 0.2 envelope).

## 6. Consequences recorded as facts (no design)
* The own yaw rate/kappa and acceleration are model INPUTS (kin3, p10f, base); the label (gt) carries neither as a separate target.
* Under speed_v1 the first label segment has the human's heading and curvature; the own-vs-human yaw gap is therefore a pure input/label mismatch at NOW (the census measures it).
