# EGOWM read and planner harness (card: PREREG_V8F_EGOWM.md, frozen sha256 ddd6fc9b...f446e7)

All code here is a thin layer over `ladder_read/lr_lib.py` and `ladder_plantest/lp_lib.py`; neither is edited. CPU only, 2 threads. Env: `source env.sh`. Tier T0.

## Files
| file | role |
|---|---|
| `eval_speed.py`, `eval141_speed.npz` | measured speed of the 141 eval clips; kinematics measurement (`KINEMATICS_EGOWM.json`) |
| `lr_speed.py` | IA-0 and beside modules of lr_lib with the speed channel (`read_arm_speed`, CLI `name|run_dir|mode`) |
| `lp_speed.py` | planner harness: `prepare_arm_speed`, `plan_all_speed`, `unit_speed` (same arguments and results as `lp_lib.unit3`) |
| `controls_read.py` / `CONTROLS_READ.json` | controls c1, c2, c3 |
| `controls_plan.py` / `CONTROLS_PLAN.json` | controls pA, pB, pC, pD |

## Where the measured speed comes from
The eval frame caches (`ladder_read/cache/frames_s*.npz`) carry no speed. The trainer's speed is `kin3[:, 0]` of `ClipStore.get_kin(clip)` in kin_mode `yaw`: the speed of the clip's native pose track at raw frame 2j (`ego_channels_from_poses`), file `sfresh_inputs/inputs/native_tracks.pt`, joined by clip name into `eval141_speed.npz` (sha256 d9e4336eb8e255a24cdeea67ffceb4a66e2c3e5ad18774f58d86f134555272c7). Row 0 of each clip is NaN by construction; the read fills it with row 1 so the anchor at frame 0 exists and the cell set equals the control's (`speed_row0_filled` is in each result). It is the same quantity the zcache trainer path feeds (`v8f_wm_windows._kin_small`).

## How speed enters
Read: the encoded states are speed-free; the free run gets `anchored_speeds(mode, v, acc, anchors, kmax, cum)`, the trainer's own function (E-NOW: measured speed at the anchor frame repeated; E-KIN: floored integral of the logged `act[f, 0]`, and of the contrasted actions under an IA-3 contrast). The arm is refused unless the BUILT stack's rule reads back the argv's `--wm-speed`.
Planner: level-1 rollouts add `st.speed_embed(1, speed)`; E-KIN integrates the CANDIDATE plan's accelerations (`2.0 x scaled acc`) from the measured speed at NOW, differentiably; E-NOW holds it; context-3 history rows use the measured speed of their own frames. TD3 raises (not computable on strides (3, 5)).
The seed-2 replicate is read with the frames of M2b_s0 (its readers), per `IMPL_CHOICES_EGOWM.md` section 3.

## Kinematics (deliverable 3, MEASURED, `KINEMATICS_EGOWM.json`)
Integrating `act[..., 0]` at 0.2 s from the measured speed at the anchor, against the measured speed 10 and 30 steps later, over all ok anchors of the 141 clips (12,686 anchors at 2.0 s, 9,866 at 6.0 s):
| horizon | E-KIN integral MAE, per clip mean / pooled | hold-speed (E-NOW) MAE, per clip mean / pooled | integral shifted by one step (beside) |
|---|---|---|---|
| 2.0 s | 8.7e-10 / 8.7e-10 m/s | 0.894 / 0.894 m/s | 0.099 m/s |
| 6.0 s | 2.3e-09 / 2.3e-09 m/s | 2.275 / 2.277 m/s | 0.153 m/s |
The cache's `act[..., 0]` is the exact forward speed difference over 0.2 s, so on LOGGED actions E-KIN's speed channel equals the measured future speed to float precision (the 0 m/s floor never changed a value in the data). Consequence to carry into the read: on logged actions the E-KIN channel is a deterministic function of the action sequence it already receives; the arm tests whether giving the integrated form helps, and its planner form differs from the logged one only through the candidates.

## Controls (measured; see JSON for numbers)
* c1: R35H_s0 read through `lr_speed` against `ladder_read/results/arm_R35H_s0.json`, every leaf. STATUS: NOT PASSED YET. First run (2 threads, 313 s): 503 of 1,426 leaves differed; the five shown were all in the ridge-probe `beside` block at the 1e-14 relative level (the banked run used 3 threads), but the run did not record which sections differ, so IA-0 exactness is UNVERIFIED. `controls_read.py c1` was edited to report differing leaves by section and the max relative difference and to require ia0, ia0_planning_context1 and passes to be exactly equal; its rerun was refused by the 6 GB memory guard (2.7 GB free, other sessions). Rerun: `source env.sh; $PY controls_read.py c1` (about 5 min, needs 6 GB free).
* c2: speed-structured model with zero speed weights, kin and now: all 24 cells (3 clips) bit-equal to R35H; random weights: differ (max abs 2.3 and 2.8).
* c3: speeds handed to the stack equal a numpy hand integration (kin 4.8e-7 m/s, now 0.0), including the floor and under a +1 m/s^2 contrast; sign flipped: error 24.6 m/s, red; E-NOW with a future speed: 7.7 m/s, red.
* pA: speed off, R35H planned through `lp_speed` equals `lp_lib.plan_all3`: difference 0.0, 8 windows (c1) and 7 windows (c3), every returned array bit-equal.
* pB: zero speed weights (kin, now), both contexts: difference 0.0; random weights: 5.5 to 21.1 (the effect is visible).
* pC: speeds handed to the rollout equal a hand integration of the candidate plan (kin 8.5e-7, now 0.0, floor engaged, E-NOW constant over the rollout); sign flipped 4.3 m/s, red.
* pD: autograd directional derivative of the goal cost equals a central finite difference (relative error 0.0012); with the speed detached the relative error is 0.019. UNVERIFIED strength: the margin is 16x on random weights of scale 0.5; a trained speed weight may differ.

## Not verified
Trained E-KIN / E-NOW checkpoints do not exist, so no speed arm has been read or planned; every speed control runs on R35H weights with synthetic speed weights. The planner harness has not run a full 1,976-window unit.
