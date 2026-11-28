# PREREG R8-PICKER: refcv8 EMA-48,852 plus a GT-free picker, against the shipped refcv8 + RETIME

* **Author.** EvalFlyWheel agent, 2026-10-21.
* **Clock read immediately before sealing:** 2026-10-21 21:01:51 +0200 (Berlin, `date '+%F %T %z'` on the dev box). The
  seal (sha256 of this file) is in `PREREG_R8_PICKER.sha256` beside it.
* **Sealed BEFORE any new roll.** No capture roll (R0, R1 below) has been run. No picker in this document has been
  read on a fan carrying the model's own speed-cap mask.
* **Status.** Pre-registration of a DECISION-GRADE read. It replaces nothing that is registered. SHIP-R8 (AMENDMENT 3,
  registered per-frame RETIME table `2aacc9ff…`) is applied UNCHANGED, as written.

## 0. Why: the post-hoc evidence (MEASURED, not a registered claim)

The source is `C:/Users/Admin/refcv8_pf/fanceil/RESULT.md`, banked at f2f604f under
`FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-21-refcv8-fanceil/`. Its stamps are: T1 open loop, one training
seed, CI answers EPISODES only.

* refcv8 EMA-48,852 has a 149-candidate fan.
  * Oracle-in-fan at 2 s is 0.084 m (T0).
  * The model's own pick is 0.297–0.299 m, at rank median 12–13 of 149.
  * 78 % of the selection loss is along-track.
* Two GT-free pickers on the same fan beat the own pick at 2 s. Both are POST-HOC. Values are full set, paired vs the
  own pick:
  * `base` (`sel_idx_base`): −0.0156 [−0.0219, −0.0100] on s1 and −0.0161 [−0.0226, −0.0106] on s0.
  * MBR inside the core top-32, T = 1, restricted to a slot-level speed-cap PROXY. The cell was frozen 19:51:51 in
    `FROZEN_PICKER_CELL.md`:
    * −0.0347 [−0.0430, −0.0268] on s1.
    * −0.0308 [−0.0411, −0.0216] on the fresh s0, which is the frozen replicate.
    * On S6, −0.29 / −0.26.
* Plain argmax of the core score exceeds the proxy cap on about 11 % of windows, against about 5 % for the own pick.
  That is the reason a cap mask is part of the picker.

What was NOT shown:

* the model's OWN cap mask, since a proxy was used;
* RETIME on top of a picker;
* SHIP-R8;
* the four metric families against the SHIP stack.

This prereg reads exactly those.

## 1. Rolls (Thor, after the PERC2PLAN chain; nothing else)

| | inference seed | form | why |
|---|---|---|---|
| **R0** | 0 | **FULL-ARM**: the battery's s0 child argv, no `--os-only` | The banked s0 was a full-arm roll. `nav_true`, `nav_shuffled` and `nav_flipped` are batched as 3 rows, and the DDIM base noise `eps_b = randn((b, 117, …))` (`refc.py:3036`) depends on b. Only a full-arm roll can reproduce the banked s0 draw. The fanceil s0, which was `--os-only`, was a fresh draw and matched its bank on 33.8 % of fan cells. |
| **R1** | 1 | `--os-only`, the battery's s1 child argv | The banked s1 is os-only. The fanceil s1 matched it: sel agreement 0.9958, 99.4 % of fan cells. |

**Inputs and environment.** Both rolls use the fanceil agent's md5-verified inputs on Thor, read-only:

* code: `/home/nvidia/r8ns/fanceil/{tree,code}`;
* checkpoint: `/home/nvidia/r8ns/ckpt/ckpt_ema_48852.pt`, md5 `b4364029b68f29a33eeb4aebbbbd002b`;
* config: md5 `4ce2c77230dc4e802e78b39ce05af916`;
* windows: `/home/nvidia/r8ns/fanceil/inputs/windows_s1.json`, md5 `a3a78dba889e70aa69a05271559794ad`, the same file as
  the battery's `windows_s0.json` and `windows_s1.json`;
* remap: `refcv8_eval_remap_thor_fanceil.json`, md5 `4dc5fb013cb1ed4533379816f0bc3a22`;
* environment: the same as `run_fanroll_thor2.sh`, which is `R8_NO_POSE_SYNC=1`, `--nav-feed perframe_full`,
  `OMP_NUM_THREADS=6`;
* venv: `tanitad-train`.

The run script refuses to start if the md5 of the checkpoint, the config or the windows differs from these values.

**Capture.** The capture wrapper is `fan_roll_r8_pk.py`. Its sha256 is recorded in each run record. It extends
`fan_roll_r8_pf.py` and calls `r7_roll.Refcv6Proxy.__call__` UNCHANGED, with the same RNG stream and the same model
call. It then copies the `nav_true` row of these keys out of the returned dict:

`anchor_traj, anchor_logits, refined_logits, sel_score, sel_score_v3, sel_idx, sel_idx_base, traj, traj_base,
reach_keep, ceil_keep, r8_emit_keep, goal_dist, r8_cand_src`

It also stores `e9_keep = isfinite(refc_v3.e9_rank(zeros_like(sel_score_v3), out))`. That is the model's own E9
admissible set: `reach_keep` ∧ `ceil_keep` ∧ `r8_emit_keep`, with the model's own dead-row fallbacks, evaluated by the
model's own function at roll time. The forward is not changed. Calls whose conds do not include `nav_true` are passed
through uncaptured.

**Queue.** The rolls run after `ZZWDONE` or `ZZWTIMEOUT` appears in `/home/nvidia/r8ns/perc2plan/out/waiter.log`,
under `flock -w 7200 /home/nvidia/refcv7_post/thor_gpu.lock`, with R0 first and then R1. No other agent's process is
touched.

## 2. Arms

For each seed s, every arm is built from ONE capture roll R_s. The arms therefore share one fan and one DDIM draw, and
their differences carry no inference-draw noise. Let F = `anchor_traj` [N, 149, 8, 2].

| arm | the emitted 8-slot plan | RETIME |
|---|---|---|
| **SHIP** | `F[sel_idx]`, asserted equal to `traj`: the model's own E9 pick | **RETIME_DEPLOY8**, registered table sha256 `2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80` (AMENDMENT 2/3), exactly as shipped |
| **BASE** | `F[sel_idx_base]`: the model's own CORE pick, with the reach, ceiling and emit masks applied by the core; it comes BEFORE the E9 goal-distance graft and the drivable critic | none (raw) |
| **MBR32** | `F[mbr_pick(F[:, :, :4], where(e9_keep, s, −1e30), 32, 1.0)]`, where `s = sel_score` with non-finite values mapped to −1e30 first, as in `picker_panel_r8.py` | none (raw) |
| **BASE+RT** | the BASE plan | RETIME_DEPLOY8 with the SAME registered table as SHIP, no refit |
| **MBR32+RT** | the MBR32 plan | RETIME_DEPLOY8 with the SAME registered table as SHIP, no refit |

**MBR32 is exactly the frozen cell.** The frozen cell (`FROZEN_PICKER_CELL.md`, sha256 `5bc51822…`) is score
`sel_score`, K = 32, T = 1.0, using `fan_mbr_x8.mbr_pick` (sha256 `94cef175…`) UNCHANGED. The ONLY change is that the
mask is the model's own `e9_keep` instead of the slot-level proxy (planned speed > `v_max_ms` + 0.5). No other cell is
read, and K, T and the score are not re-chosen.

⚠️ **One property is stated now, because it decides how a speed-cap readout must be read.** `mbr_pick` takes the
top-32 by score and returns the risk argmin over those 32. When fewer than 32 candidates lie in `e9_keep`, masked
candidates fill the top-32 with weight exp(−1e30) = 0, but they REMAIN ELIGIBLE as the argmin. This is the frozen
implementation and it is NOT changed. The rate at which MBR32's pick falls outside `e9_keep` is readout R5-ii.

**Raw arms (BASE, MBR32).**

* They are scored CAPPED, using `clamp_posthoc_r8.apply_clamp`, the landed `cap_plan_to_limit`, on ceiling-compliant
  windows. This is the refcv8 raw-plan convention of AMENDMENT A-R8-CLAMP-2. The uncapped read is printed beside.
* They are DIAGNOSTIC. They show what a picker does without the stage. They are never adopted by this prereg (see §7).

**SHIP_raw**, the own pick capped without RETIME, is printed BESIDE.

## 3. Pipeline (no instrument file is edited)

**Trees.** Each picker gets one tree holding both seeds:

| tree | arm |
|---|---|
| `pkship_pf` | SHIP |
| `pkbase_pf` | BASE |
| `pkmbr32_pf` | MBR32 |

Each tree is a copy of the banked tree `C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf` into
`C:/Users/Admin/refcv8_pf/picker/archive/<tree>`. Rows are joined on (clip_index, ws), and every one of the 4,754
windows must join. For each seed the arm's 8-slot plan is SPLICED into exactly these fields, and nothing else:

* `os` in the `ep*.npz` files of `dump_s{s}`, `panel_s{s}` and `s6_s{s}`, at the slots of each directory's manifest
  grid;
* `plan_full_nav_true`, `sel_idx_nav_true` and `sel_idx` in the three `decisions/` directories;
* `nav_true.traj` and `nav_true.sel_idx` in `dump_s{s}/refcv6_extras.npz`.

Everything else stays bit-identical to the bank: GT, the model-free arms, the refcv7 baselines, the feed sidecars and
the decisions telemetry.

**Aggregates.** Root aggregates that depend on the plan are deleted:

`PAIRED_READ_*`, `analysis*`, `criteria*`, `acceptance*`, `cross_paired*`, `inference_seed_replicate.json`,
`tactical_v6_*`

`PAIRED_READ_<tree>.json` is then regenerated by `pair_read_r8.py`, unchanged and with the `run_bcand.sh` invocation.

**Lane.** The lane is the `wise_ship_lane.sh` sequence:

* `pf_lane.py` stages `install`, `clamp`, `compliant`, `tf_forward`, `d8`, `df`, `tac` and `parity`;
* `feed_parity_pf.py`;
* `ship_r8_verdict_pf.py`.

All three are UNCHANGED code. Each tree gets its own private pf package. The archive root is re-pointed to
`C:/Users/Admin/refcv8_pf/picker/archive` the way `retime_trainfit_r8._patch_archive` already does it: through
`CP`, `CR`, `TAC` and `NR` `.ARCHIVE`, prior_parity's `R8_ARCHIVE`, and the verdict's `--archive`. This is a PATH, not
a number.

## 4. Readouts (per arm and per inference seed; all cells are compliant windows unless stated)

* **R1 SHIP-R8.** The overall verdict and every clause of AMENDMENT 3, for the SHIP, BASE+RT and MBR32+RT trees. It
  comes from each tree's `SHIP_R8_<tree>.json`.
* **R2 E1.** S2 ADE minus `ha0_ext_pose`.
  * +RT arms: the cell clause (2) reads, RETIME_DEPLOY8 − ha0_ext_pose.
  * Raw arms: compliant_reread's capped `os` − ha0_ext_pose.
  * Both carry the inference floor and the NOT PROVEN rule of `pair_read_r8.bar_E`.
* **R3 E3.** The same on S6.
* **R4 The four metric families, paired vs SHIP.**
  * Components are computed per window by `pair_read_r8.components`, the battery's own `refav1_arm._components` plus
    the SPEC section 5 cells. They are computed on:
    * +RT arms: the lane's RETIME_DEPLOY8 positions, `S2_pos` and `S6_pos` from `DEPLOY8_POS_<tree>_s{s}.npz`;
    * raw arms: the capped plan at the surface slots;
    * SHIP: its own `DEPLOY8_POS`.
  * GT is the panel's `g`, asserted bit-identical across trees.
  * The cell is `arm − SHIP`, computed by `pair_read_r8.paired_cell` on S2 and S6.
  * The headline set is the compliant windows; all windows are printed beside.
  * The components, all 14 keys of `pair_read_r8.METRICS`:
    * **ADE:** `ade_m`, `fde_m`
    * **LONGITUDINAL:** `LON_speed_mae_mps`, `LON_along_mae_m`, `LON_along_abs_final_m`, `LON_speed_within_0p5`,
      `LON_accel_mae_mps2`, plus the COMPLIANCE row (R5-i)
    * **LATERAL:** `LAT_heading_mae_deg`, `LAT_curvature_mae_1pm_valid`, `LAT_yaw_rate_mae_radps_valid`,
      `LAT_cross_mae_m`, `LAT_cross_abs_final_m`
    * **TACTICAL:** `TAC_traj_lat_correct`, `TAC_traj_lon_correct`
    * **STRATEGIC: NOT APPLICABLE, n = 0.** The run's argv carries `--no-strategic`, so the route head is untrained
      and unscored, and this battery has no route or goal readout. It is reported per family with that reason.
* **R5 Speed cap.**
  * (i) **COMPLIANCE**, per window: `planned_max_speed` of the full 8-slot plan the arm EMITS is greater than the fed
    ceiling. The emitted plan is the uncapped pick for raw arms, and `S2_pos8` from `DEPLOY8_POS` for +RT arms. The
    fed ceiling is `clamp_posthoc_r8.fed_ceiling`, applied strictly. The read gives a rate and the paired
    arm − SHIP cell. The capped value is printed beside.
  * (ii) The fraction of windows whose picked candidate lies OUTSIDE `e9_keep`.
  * (iii) Each tree's own lane compliance row, printed beside.

## 5. Estimator and stamps

* **Estimator.** Full-set means. Paired episode-cluster bootstrap (`taniteval.ci` via `pair_read_r8.paired_cell`),
  B 2000, bootstrap seed 0, 95 % percentile interval.
* **Which question the CI answers.** It answers ONLY "would another draw of EPISODES say this?" (139 clips). The models
  and the RETIME gains are fixed.
* **Seeds.** There is ONE training seed (refcv8-r101-s0). Inference seeds 0 and 1 are the inference replicate. The
  picker − SHIP difference is within one DDIM draw per seed.
* **Tier.** T1 self-action OPEN LOOP on logged eval139 frames. This is not driving performance. The picks use no GT,
  and GT enters only the scoring.

## 6. Controls (each reads a stated value; a failed BLOCKING control makes the read INCOMPLETE, never PASS or FAIL)

| id | control | expected | blocking? |
|---|---|---|---|
| K1 | CPU smoke, 2 episodes, both seed forms. The wrapped roll vs the UNWRAPPED child, same device (CPU), same seed and windows: `os`, `plan_full_nav_true` and `nav_true.traj`. | bit-identical | yes |
| K2 | R1 vs the fanceil Thor s1 roll (same device, code and seed): sel_idx agreement, and max \|Δtraj\| on windows where sel_idx agrees. | ≥ 0.99 (expect ≈ 1.0); report max | yes |
| K3 | `F[sel_idx] == traj`, and `traj ==` the roll's own sidecar `nav_true.traj`, on R0 and R1. | exactly 0.0 | yes |
| K4 | argmax of `e9_rank(sel_score_v3, captured masks)` == `sel_idx`. | every window; a tie at an equal score is counted, not failed | yes |
| K5 | argmax of the core rank (`sel_score` with reach, then the ceiling mask with dead-row fallback, then emit as the core does) == `sel_idx_base`. | every window (ties as K4) | yes |
| K6 | On windows with `v_max_valid == 0`, `ceil_keep` is all True. The SHIP pick lies in `e9_keep` on 100 %. | literal True / 1.0 | yes |
| K7 | sel_idx agreement with the banked `sel_idx_nav_true` (cross-device, dev-box RTX 4060 bank): R1 and R0. | ≥ 0.99 each | **no**: R1 failing is impossible if K2 holds; R0 failing means the full arm did not reproduce the bank's draw. This is reported, and the paired reads stay valid. |
| K8 | Identity splice: splicing the bank's own plans into a copy reproduces EVERY array of EVERY npz of the tree. A deliberate-regression splice (one window's plan moved 1 m) must be DETECTED. | array_equal / detected | yes |
| K9 | The lane through the archive re-point, run on the identity tree, reproduces the banked `SHIP_R8_step48852ema_pf.json`. | identical per-clause statuses and identical E1/E3 cells (delta, lo, hi) | yes |
| K10 | The SHIP tree's verdict. | overall PASS; E1/E3 per seed within 0.005 / 0.02 m of the banked cells | PASS is blocking. Drift beyond tolerance is reported, not blocking. |
| K11 | Pulled npz md5 on Thor == on the dev box; the checkpoint, config and windows md5 hold on Thor at launch. | equal | yes |
| K12 | Panel GT `g` is identical across `pkship_pf`, `pkbase_pf`, `pkmbr32_pf` and the bank. | bit-identical | yes |

K1 and K8/K9 run BEFORE the GPU rolls. Their results are banked in this directory.

## 7. Decision rule (both outcomes committed now)

**ADOPTABLE.** X ∈ {**BASE+RT**, **MBR32+RT**} is ADOPTABLE iff every blocking control passes AND all three of these
hold:

* **A1.** The SHIP-R8 overall on X's tree reads **PASS**: every clause of AMENDMENT 3 with the registered table.
* **A2.** No R4 component cell of X − SHIP is SEPARATED WORSE at EITHER inference seed. This covers:
  * 14 keys × {S2, S6} × {s0, s1} on compliant windows;
  * plus the COMPLIANCE row × {s0, s1}.

  Separated worse is defined as follows:
  * error metric: CI lo > 0;
  * accuracy metric (`LON_speed_within_0p5`, `TAC_traj_lat_correct`, `TAC_traj_lon_correct`): CI hi < 0;
  * COMPLIANCE, a lower-is-better rate: lo > 0.

  This is the programme's FZR-NR convention ("not separated worse at both seeds" = at neither seed), stated
  explicitly. The cells that are separated worse at BOTH seeds are printed beside and are not decisive.
* **A3.** The S2 ADE of X − SHIP (compliant) is SEPARATED BELOW 0 (CI hi < 0) at BOTH seeds, AND |Δ| > 2 × the SHIP
  inference floor (|mean SHIP S2 ADE s0 − s1| on the compliant windows). If it is separated but within 2× the floor,
  the reading is NOT PROVEN.

**Verdict words** are ADOPTABLE / FAIL / NOT PROVEN / INCOMPLETE:

* **FAIL** names the first failing clause or cell.
* **INCOMPLETE** means a blocking control failed or a required cell could not be computed.

**Raw arms (BASE, MBR32)** are read against the raw-plan bars:

* E1 and E3 of the capped `os` − ha0_ext_pose;
* NR vs refcv7-50,400 as in compliant_reread;
* plus A2 and A3 vs SHIP.

At most they can read **RAW-ELIGIBLE**. That is a PI item, because it would drop the shipped RETIME stage decided
under D-R8-SHIP-RETIME. They are never ADOPTABLE here.

**Committed outcomes:**

* **MBR32+RT ADOPTABLE** → reported as a ZERO-TRAINING improvement of the deployed refcv8 on the T1 open-loop eval139
  read, with one training seed. It goes to the Master Mind as a deploy item, because it changes the shipped selection
  rule. The runtime-stage port and a NavSim read are the next steps, and this prereg does not license them.
* **BASE+RT ADOPTABLE** → reported as written. **Its adoption is a PI decision WHATEVER the result.** BASE removes the
  E9 goal-distance graft and the drivable critic from the deployed selection, which touches the hierarchy thesis (the
  tactical goal's use in selection).
* **Both ADOPTABLE** → both are reported. The recommendation is **MBR32+RT**, because it keeps the E9 graft in the
  scoring and so leaves the hierarchy intact. BASE+RT goes to the PI as above. The preference is fixed now and not
  chosen on the numbers.
* **Neither ADOPTABLE** → each arm is reported FAIL / NOT PROVEN / INCOMPLETE as written, with its failing cell. The
  committed next lever (§8) is then executed in the same run when it needs no GPU.

## 8. Next levers, committed in advance (Rule Zero)

* **L1 (CPU, same capture).** Triggered when MBR32's pick lies outside `e9_keep` on > 0 windows AND MBR32+RT fails A2
  on the COMPLIANCE row. The next lever is **MBR32m**: the same cell with the risk argmin restricted to `e9_keep`. It
  is sealed as an ADDENDUM to this prereg BEFORE it is computed, and read by the same rules.
* **L2.** Triggered when a raw arm's S2 gain vs SHIP is separated but its +RT arm fails A3. In that case the frozen
  RETIME table absorbs the picker's longitudinal gain. The next lever is a RETIME re-fit on PICKER train rolls by the
  unchanged H-R8-RETIME-TRAINFIT-1 procedure. **It is blocked on** Thor train rolls and a PI decision (a new
  registered table).
* **L3.** Triggered when SHIP-R8 clause (4) or (5) fails vs refcv7, or A2 fails on a LATERAL or TACTICAL cell. The
  next lever is the refcv9 selector item (rank the extras: the oracle is an extra on 45.5 % of windows, against 12 %
  of picks). **It is blocked on** training compute and a spec.

## 9. What this prereg does NOT license

* No other MBR cell, score, K, T or mask variant is read as a candidate. MBR32m comes only via L1 with its own sealed
  addendum.
* No change to the SHIP-R8 clauses, the RETIME table, the compliant set or the bootstrap.
* A failed arm is reported as FAILED. The bar is not moved after the data.
* No pick of the "better" arm on the numbers (see §7).
* Nothing here is driving performance (T1 open loop), and nothing replicates across training seeds.

## 10. Frozen references

| artifact | sha256 / md5 |
|---|---|
| `C:/Users/Admin/refcv8_pf/fanceil/FROZEN_PICKER_CELL.md` | sha256 `5bc518224f304220ae2aaf914685188881088137aadf9e77125731c3974d7d1a` |
| `…/2026-09-28-refcv7-standard-tests/battery/code/fan_mbr_x8.py` (launch tree C:/lgs/Lfb47a58376) | sha256 `94cef175bcd4540e760c2d064219264f2d8c63b97783835d41711ff067059c5a` |
| `C:/Users/Admin/refcv8_pf/fanceil/code/picker_panel_r8.py` (the frozen cell's producer) | sha256 `2e00b031339a4c6a29c35afb4ca5faef1d6b369d245e03c4b1bbd66aa9c37b97` |
| `C:/Users/Admin/refcv8_pf/fanceil/code/fan_roll_r8_pf.py` (the wrapper this one extends) | sha256 `f1c80d5ed6a5c701d1f09610da2ca1092b72de744c40385d66ddf09499185d79` |
| registered per-frame RETIME table (`C:/Users/Admin/qland/hold_r8pf/REGISTERED_PF_TABLE.sha256`) | `2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80` |
| banked verdict `…/2026-10-20-refcv8-perframe-final/raw/ship_r8/SHIP_R8_step48852ema_pf.json` (overall PASS; E1 −0.0342 / −0.0329, E3 −0.8682 / −0.8676) | sha256 `893b69a8cd8da312b89c31206db2bf637232468d89711e8bd00e68e34cc48c39` |
| checkpoint `ckpt_ema_48852.pt` / config.json / windows_s{0,1}.json | md5 `b4364029b68f29a33eeb4aebbbbd002b` / `4ce2c77230dc4e802e78b39ce05af916` / `a3a78dba889e70aa69a05271559794ad` |
