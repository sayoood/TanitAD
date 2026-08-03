# STREAM C — does the closed-loop headline survive a better render?

**Question, pre-registered before the panel ran:** the morning panel found *REF-C beats flagship v1 closed-loop and the separation is ENTIRELY LATERAL*. That was measured on the OLD render. The render improved by **+23.4 % grad-NCC** and the four shipped videos were re-rendered. Changing the render changes what the policy SEES. **Is the verdict a property of the ARMS, or of render artifacts?**

Both outcomes were committed in advance: *survives* ⇒ strong evidence it is the arms; *does not survive* ⇒ a bigger finding, and the closed-loop panel becomes render-conditional until a render-invariance check is part of the protocol.

## Provenance — the run directories these numbers come from

| | run dir | render flags |
|---|---|---|
| **HQ** (what the videos show) | `thor:/home/nvidia/cl_out_hq` | `--cull-scale-quantile 0.95 --sky-gain 0.3` |
| **MORNING** | `thor:/home/nvidia/cl_out` | `(defaults) layers=background,road, no cull, no sky` |
| **REPRO** (negative control) | `thor:/home/nvidia/cl_out_repro` | morning flags, re-run today |

Same scene `00040136-e651-4abd-991d-0655ccda9430`, same checkpoints (`flagship-v1-speedjerk` step 29999 · `refc-base` step 29999, 128 anchors), same 9 starts × 50 ticks, same scorer, same day. Geometry asserted before every rollout: `CANON f_eff=266.01 (F_REF=266.0) OK` on all four HQ panels.

Estimator: paired_episode_cluster_bootstrap over ROLLOUT STARTS — disjoint segments of ONE clip, not independent episodes.

⚠️ **The raw rollouts are banked in this directory**, under `rollouts/` — 12 files, one per (run × arm × condition), each 9 starts. Thor rebooted mid-session today; a result that exists only on the device is not a result.

## 0. Did the render change reach the policy at all?

If it did not, the whole comparison is vacuous. MEASURED, not assumed.

| arm | driven-path Δ (mean / p50 / max, m) | emitted-plan Δ (mean / max, m) | v_target Δ (mean / max, m/s) |
|---|---|---|---|
| flagship-v1 | 9.0513 / 5.9476 / 37.7809 | 7.7643 / 17.9132 | 6.1007 / 14.1707 |
| refc-base | 0.4259 / 0.186 / 3.2548 | 0.3782 / 1.6555 | 0.1853 / 1.0889 |

**At `k=0` both runs start from the identical logged pose**, so the plan difference there is the render acting on the policy with zero accumulated drift:

- `flagship-v1`: per-start [0.7265, 4.1077, 7.1716, 7.6121, 5.0949, 4.2837, 3.4087, 9.0885, 5.8533] m
- `refc-base`: per-start [0.1328, 0.1119, 0.2264, 0.3583, 0.228, 0.7231, 0.6563, 0.046, 0.0188] m

## 1. ⛔ THE NEGATIVE CONTROL — read this before any render claim

The morning **config** re-run today, paired against the morning **rollouts**. The renderer is a step function of pose (a 0.1 px camera rotation has been measured to move the 2 s waypoint 6.65 m), and `closedloop_drive.py` changed twice today after the morning run — so this control is what separates *the render* from *run-to-run noise and code drift*.

| arm | driven-path Δ (mean / max, m) | emitted-plan Δ (mean / max, m) |
|---|---|---|
| flagship-v1 | 0.0 / 0.0 | 0.0 / 0.0 |
| refc-base | 0.0 / 0.0 | 0.0 / 0.0 |

## 2. THE ANSWER — every morning separation, re-measured

`flagship v1 − REF-C base`, empty road, paired on the windows shared by **all** runs. Positive = flagship worse.

| family | metric | MORNING (rescored today) | HQ render | verdict | render moved it? |
|---|---|---|---|---|---|
| ADE | `ade_0_2s` | +0.7885 [-0.8653, +2.7282] | +7.1642 [+5.2654, +8.9661] **sep** | **GAINED** | **yes** |
| LATERAL(=|cross_track|) | `dist_to_gt_traj_m` | +1.1705 [+0.0296, +2.2438] **sep** | +4.5033 [+2.0194, +7.3718] **sep** | **SURVIVES** | **yes** |
| LONGITUDINAL | `abs_target_speed_err_ms` | +1.1242 [-0.1008, +2.5657] | +6.3971 [+4.9996, +7.8014] **sep** | **GAINED** | **yes** |
| LONGITUDINAL | `along_track_ade_m` | +0.6498 [-1.0166, +2.5903] | +7.1528 [+5.2402, +8.9528] **sep** | **GAINED** | **yes** |
| LATERAL | `heading_err_rad` | +0.0838 [+0.0278, +0.1750] **sep** | +0.1700 [+0.1303, +0.2119] **sep** | **SURVIVES** | no |
| LATERAL | `curvature_err_1pm` | +0.0050 [+0.0008, +0.0130] **sep** | +0.0166 [+0.0125, +0.0213] **sep** | **SURVIVES** | **yes** |
| LATERAL | `yawrate_err_rads` | +0.0378 [+0.0201, +0.0565] **sep** | +0.1066 [+0.0774, +0.1393] **sep** | **SURVIVES** | **yes** |
| LATERAL | `cross_track_abs_m` | +1.1705 [+0.0296, +2.2438] **sep** | +4.5033 [+2.0194, +7.3718] **sep** | **SURVIVES** | **yes** |
| TACTICAL | `manoeuvre_plan_eq_logged` | +0.0709 [-0.1241, +0.2600] | -0.0618 [-0.2165, +0.0801] | **SIGN FLIP** | **yes** |
| STRATEGIC | `route_corridor_departure_rate` | +0.2037 [-0.0023, +0.3982] | +0.5057 [+0.3822, +0.6289] **sep** | **GAINED** | **yes** |

*"render moved it?" is the **difference-in-differences**: `(flagship−REF-C)|HQ − (flagship−REF-C)|MORNING`, bootstrapped on the same windows, clustered by rollout start. Comparing two CIs by eye is not a test; this is.*

## 3a. FOUR FAMILIES + ADE on the improved render — `empty` road

`results\closedloop-hq-render\HQ_flagship_vs_refc_empty.json` · n_windows 450/437, 9 clusters · paired on 437 shared windows

| family | metric | flagship v1 | REF-C base | paired Δ (F−C) |
|---|---|---|---|---|
| ADE | ADE 0-2 s | 9.6955 [8.2293, 11.3271] | 2.6554 [1.7555, 3.5254] | +7.1642 [+5.2654, +8.9661] **sep** |
| ADE | 0.5 s | 3.9654 [3.3696, 4.6240] | 0.9142 [0.5733, 1.2551] | — |
| ADE | 1 s | 7.8166 [6.6354, 9.1433] | 1.9796 [1.2632, 2.6903] | — |
| ADE | 1.5 s | 11.6357 [9.8691, 13.5983] | 3.1967 [2.0754, 4.2691] | — |
| ADE | 2 s | 15.3642 [13.0242, 17.8990] | 4.5312 [3.0988, 5.9242] | — |
| LONGITUDINAL | target-speed err (signed) | -7.9970 [-9.3089, -6.8014] | 1.2105 [0.1938, 2.1742] | — |
| LONGITUDINAL | target-speed err (abs) | 7.9970 [6.8014, 9.3089] | 1.7048 [1.0553, 2.3675] | +6.3971 [+4.9996, +7.8014] **sep** |
| LONGITUDINAL | executed-speed err (signed) | -4.9149 [-5.6020, -4.1857] | 1.1902 [0.1803, 2.1416] | -6.0898 [-7.0324, -5.1225] **sep** |
| LONGITUDINAL | along-track ADE | 9.6385 [8.1623, 11.2734] | 2.6104 [1.6980, 3.4910] | +7.1528 [+5.2402, +8.9528] **sep** |
| LONGITUDINAL | headway to lead | 37.5657 [22.4080, 51.3685] | 46.9923 [40.5137, 56.7049] | — |
| LONGITUDINAL | time gap | 3.9948 [2.1437, 6.1428] | 2.9255 [2.1788, 4.0895] | — |
| LATERAL | heading err | 0.2037 [0.1649, 0.2450] | 0.0357 [0.0246, 0.0501] | +0.1700 [+0.1303, +0.2119] **sep** |
| LATERAL | curvature err | 0.0179 [0.0136, 0.0230] | 0.0016 [0.0010, 0.0022] | +0.0166 [+0.0125, +0.0213] **sep** |
| LATERAL | yaw-rate err | 0.1179 [0.0918, 0.1485] | 0.0132 [0.0103, 0.0164] | +0.1066 [+0.0774, +0.1393] **sep** |
| LATERAL | cross-track abs (== dist_to_gt_traj) | 5.1618 [2.8872, 7.8607] | 0.6304 [0.2960, 1.0208] | +4.5033 [+2.0194, +7.3718] **sep** |
| LATERAL | lateral ADE | 0.7843 [0.6257, 0.9626] | 0.3096 [0.2344, 0.3796] | +0.4751 [+0.3203, +0.6569] **sep** |
| TACTICAL | plan == logged manoeuvre | 0.2400 [0.1467, 0.3333] | 0.2998 [0.1854, 0.4165] | -0.0618 [-0.2165, +0.0801] |
| TACTICAL | executed == planned | 0.5519 [0.4593, 0.6408] | 0.8778 [0.7296, 0.9852] | — |
| TACTICAL | 5-way head == plan | 0.5289 [0.4311, 0.6244] | 0.5240 [0.3387, 0.7151] | — |
| TACTICAL | 5-way head == logged | 0.0644 [0.0289, 0.1044] | 0.1556 [0.0644, 0.2588] | — |
| STRATEGIC | route-corridor departure | 0.6111 [0.5400, 0.6823] | 0.0938 [0.0000, 0.1912] | +0.5057 [+0.3822, +0.6289] **sep** |
| STRATEGIC | route head == logged route | 1.0000 [1.0000, 1.0000] ⛔degenerate | 0.1796 [0.0065, 0.3201] ⛔degenerate | +0.8027 [+0.6565, +1.0000] **sep** |
| STRATEGIC | route-head side == dyaw sign (PROXY) | 0.0911 [0.0000, 0.2600] | 0.4508 [0.2814, 0.6111] | — |

- **flagship-v1 manoeuvre class shares** — planned {'lane_keep': 0.1911, 'turn_left': 0.1067, 'turn_right': 0.3356, 'accelerate': 0.1067, 'brake_stop': 0.26}, logged {'lane_keep': 0.3667, 'turn_left': 0.0, 'turn_right': 0.0, 'accelerate': 0.1756, 'brake_stop': 0.4578}, 5-way head {'lane_keep': 0.1, 'turn_left': 0.16, 'turn_right': 0.5844, 'accelerate': 0.1111, 'brake_stop': 0.0444}
- **refc-base manoeuvre class shares** — planned {'lane_keep': 0.8284, 'turn_left': 0.0, 'turn_right': 0.0, 'accelerate': 0.0, 'brake_stop': 0.1716}, logged {'lane_keep': 0.3753, 'turn_left': 0.0, 'turn_right': 0.0, 'accelerate': 0.2059, 'brake_stop': 0.4188}, 5-way head {'lane_keep': 0.5355, 'turn_left': 0.0, 'turn_right': 0.4622, 'accelerate': 0.0, 'brake_stop': 0.0023}
- **flagship-v1 route** — logged share {'left': 0.0, 'straight': 0.3556, 'right': 0.0, 'unknown': 0.6444}, head share {'left': 0.0, 'straight': 1.0, 'right': 0.0}, label valid 0.3556
- **refc-base route** — logged share {'left': 0.0, 'straight': 0.3822, 'right': 0.0, 'unknown': 0.6178}, head share {'left': 0.2288, 'straight': 0.3867, 'right': 0.3844}, label valid 0.3822

## 3b. FOUR FAMILIES + ADE on the improved render — `objects` road

`results\closedloop-hq-render\HQ_flagship_vs_refc_objects.json` · n_windows 450/437, 9 clusters · paired on 437 shared windows

| family | metric | flagship v1 | REF-C base | paired Δ (F−C) |
|---|---|---|---|---|
| ADE | ADE 0-2 s | 7.3558 [4.5647, 10.3873] | 2.6327 [1.7358, 3.4987] | +4.6848 [+1.6450, +7.8810] **sep** |
| ADE | 0.5 s | 3.0137 [1.8991, 4.2344] | 0.9019 [0.5592, 1.2434] | — |
| ADE | 1 s | 5.9253 [3.6838, 8.3655] | 1.9577 [1.2305, 2.6722] | — |
| ADE | 1.5 s | 8.8306 [5.4741, 12.4835] | 3.1718 [2.0652, 4.2368] | — |
| ADE | 2 s | 11.6536 [7.2068, 16.4410] | 4.4995 [3.0521, 5.8819] | — |
| LONGITUDINAL | target-speed err (signed) | -5.8512 [-8.4461, -3.4024] | 1.1883 [0.1726, 2.1404] | — |
| LONGITUDINAL | target-speed err (abs) | 6.0788 [3.8550, 8.5033] | 1.6793 [1.0221, 2.3346] | +4.3809 [+1.9986, +6.9622] **sep** |
| LONGITUDINAL | executed-speed err (signed) | -3.8141 [-5.0944, -2.4431] | 1.1701 [0.1609, 2.1234] | -4.8992 [-6.2840, -3.4735] **sep** |
| LONGITUDINAL | along-track ADE | 7.2529 [4.4281, 10.3135] | 2.5897 [1.6710, 3.4656] | +4.6220 [+1.5735, +7.8473] **sep** |
| LONGITUDINAL | headway to lead | 35.2068 [24.9390, 46.6755] | 46.1694 [38.9006, 56.4109] | — |
| LONGITUDINAL | time gap | 3.2500 [1.9959, 4.8764] | 2.9266 [2.1282, 4.0963] | — |
| LATERAL | heading err | 0.1670 [0.1188, 0.2247] | 0.0355 [0.0252, 0.0474] | +0.1353 [+0.0816, +0.2018] **sep** |
| LATERAL | curvature err | 0.0109 [0.0059, 0.0168] | 0.0015 [0.0010, 0.0021] | +0.0096 [+0.0042, +0.0159] **sep** |
| LATERAL | yaw-rate err | 0.1034 [0.0695, 0.1448] | 0.0125 [0.0097, 0.0153] | +0.0934 [+0.0574, +0.1363] **sep** |
| LATERAL | cross-track abs (== dist_to_gt_traj) | 3.7964 [1.2394, 7.2274] | 0.6720 [0.3446, 1.0498] | +3.2127 [+0.3615, +6.6626] **sep** |
| LATERAL | lateral ADE | 0.7829 [0.6206, 0.9730] | 0.3052 [0.2227, 0.3798] | +0.4958 [+0.2927, +0.7152] **sep** |
| TACTICAL | plan == logged manoeuvre | 0.3689 [0.2555, 0.4800] | 0.3089 [0.1945, 0.4222] | +0.0526 [-0.0852, +0.2037] |
| TACTICAL | executed == planned | 0.5926 [0.4926, 0.6889] | 0.8481 [0.6556, 0.9815] | — |
| TACTICAL | 5-way head == plan | 0.3622 [0.2556, 0.4889] | 0.5309 [0.3377, 0.7406] | — |
| TACTICAL | 5-way head == logged | 0.0867 [0.0489, 0.1222] | 0.1648 [0.0618, 0.2847] | — |
| STRATEGIC | route-corridor departure | 0.3689 [0.2000, 0.5512] | 0.0915 [0.0000, 0.1887] | +0.2883 [+0.0664, +0.5111] **sep** |
| STRATEGIC | route head == logged route | 1.0000 [1.0000, 1.0000] ⛔degenerate | 0.2036 [0.0085, 0.3597] ⛔degenerate | +0.7843 [+0.6269, +1.0000] **sep** |
| STRATEGIC | route-head side == dyaw sign (PROXY) | 0.1022 [0.0000, 0.2711] | 0.4485 [0.2667, 0.6224] | — |

- **flagship-v1 manoeuvre class shares** — planned {'lane_keep': 0.4067, 'turn_left': 0.02, 'turn_right': 0.18, 'accelerate': 0.12, 'brake_stop': 0.2733}, logged {'lane_keep': 0.3511, 'turn_left': 0.0, 'turn_right': 0.0, 'accelerate': 0.2044, 'brake_stop': 0.4444}, 5-way head {'lane_keep': 0.1267, 'turn_left': 0.1311, 'turn_right': 0.4422, 'accelerate': 0.2156, 'brake_stop': 0.0844}
- **refc-base manoeuvre class shares** — planned {'lane_keep': 0.8352, 'turn_left': 0.0, 'turn_right': 0.0, 'accelerate': 0.0023, 'brake_stop': 0.1625}, logged {'lane_keep': 0.3753, 'turn_left': 0.0, 'turn_right': 0.0, 'accelerate': 0.2037, 'brake_stop': 0.4211}, 5-way head {'lane_keep': 0.5561, 'turn_left': 0.0, 'turn_right': 0.4439, 'accelerate': 0.0, 'brake_stop': 0.0}
- **flagship-v1 route** — logged share {'left': 0.0, 'straight': 0.3689, 'right': 0.0, 'unknown': 0.6311}, head share {'left': 0.0, 'straight': 1.0, 'right': 0.0}, label valid 0.3689
- **refc-base route** — logged share {'left': 0.0, 'straight': 0.3822, 'right': 0.0, 'unknown': 0.6178}, head share {'left': 0.2426, 'straight': 0.3776, 'right': 0.3799}, label valid 0.3822

## 4. With-objects vs empty-road on the improved render

The morning panel called this **null for both arms**. The improved render draws two dynamic layers that were absent from every previous frame, so it is the first version of this contrast where the actors are actually all there.

| arm | n sep / n tested | separated metrics |
|---|---|---|
| flagship-v1 | 12/23 | `ade_0_2s` -2.3397 [-3.7861,-0.7471]; `dist_to_gt_traj_m` -1.3654 [-2.3892,-0.4017]; `abs_target_speed_err_ms` -1.9181 [-3.0656,-0.6424]; `along_track_ade_m` -2.3856 [-3.8621,-0.7569]; `curvature_err_1pm` -0.0070 [-0.0133,-0.0015]; `cross_track_abs_m` -1.3654 [-2.3892,-0.4017]; `manoeuvre_plan_eq_logged` +0.1289 [+0.0267,+0.2422]; `executed_speed_err_ms` +1.1008 [+0.3677,+1.8542]; `abs_executed_speed_err_ms` -1.0028 [-1.6348,-0.3482]; `real_lead_time_gap_s` -0.8101 [-2.2038,-0.0174]; `real_lead_ttc_s_when_closing` -8.8159 [-23.8684,-0.0689]; `route_corridor_departure_rate` -0.2422 [-0.4000,-0.1000] |
| refc-base | 1/23 | `abs_executed_speed_err_ms` -0.0241 [-0.0471,-0.0003] |

## 5. The render's effect on each arm, against the noise floor

| arm | metric | HQ − MORNING | CONTROL: REPRO − MORNING |
|---|---|---|---|
| flagship-v1 | `ade_0_2s` | +6.0522 [+4.1290, +7.7424] **sep** | +0.0000 [+0.0000, +0.0000] |
| flagship-v1 | `abs_target_speed_err_ms` | +5.0378 [+3.6373, +6.2675] **sep** | +0.0000 [+0.0000, +0.0000] |
| flagship-v1 | `along_track_ade_m` | +6.1943 [+4.3371, +7.8249] **sep** | +0.0000 [+0.0000, +0.0000] |
| flagship-v1 | `heading_err_rad` | +0.0669 [-0.0587, +0.1710] | +0.0000 [+0.0000, +0.0000] |
| flagship-v1 | `curvature_err_1pm` | +0.0105 [-0.0009, +0.0195] | +0.0000 [+0.0000, +0.0000] |
| flagship-v1 | `yawrate_err_rads` | +0.0624 [+0.0173, +0.1120] **sep** | +0.0000 [+0.0000, +0.0000] |
| flagship-v1 | `cross_track_abs_m` | +2.9225 [-0.0760, +6.2240] | +0.0000 [+0.0000, +0.0000] |
| flagship-v1 | `dist_to_gt_traj_m` | +2.9225 [-0.0760, +6.2240] | +0.0000 [+0.0000, +0.0000] |
| flagship-v1 | `manoeuvre_plan_eq_logged` | -0.1378 [-0.2267, -0.0400] **sep** | +0.0000 [+0.0000, +0.0000] |
| flagship-v1 | `route_corridor_departure_rate` | +0.2578 [+0.0599, +0.4622] **sep** | +0.0000 [+0.0000, +0.0000] |
| refc-base | `ade_0_2s` | -0.0786 [-0.2863, +0.0811] | +0.0000 [+0.0000, +0.0000] |
| refc-base | `abs_target_speed_err_ms` | -0.0548 [-0.2034, +0.0707] | +0.0000 [+0.0000, +0.0000] |
| refc-base | `along_track_ade_m` | -0.0764 [-0.2847, +0.0847] | +0.0000 [+0.0000, +0.0000] |
| refc-base | `heading_err_rad` | -0.0065 [-0.0119, -0.0005] **sep** | +0.0000 [+0.0000, +0.0000] |
| refc-base | `curvature_err_1pm` | -0.0002 [-0.0004, -0.0000] **sep** | +0.0000 [+0.0000, +0.0000] |
| refc-base | `yawrate_err_rads` | -0.0020 [-0.0052, +0.0011] | +0.0000 [+0.0000, +0.0000] |
| refc-base | `cross_track_abs_m` | -0.2552 [-0.5252, -0.0071] **sep** | +0.0000 [+0.0000, +0.0000] |
| refc-base | `dist_to_gt_traj_m` | -0.2552 [-0.5252, -0.0071] **sep** | +0.0000 [+0.0000, +0.0000] |
| refc-base | `manoeuvre_plan_eq_logged` | -0.0183 [-0.0889, +0.0366] | +0.0000 [+0.0000, +0.0000] |
| refc-base | `route_corridor_departure_rate` | -0.0366 [-0.0801, +0.0000] | +0.0000 [+0.0000, +0.0000] |

## 5b. WHICH render feature moves the policy — the 2×2

`empty` road has exactly two render changes, so they can be separated. Each cell is paired against the MORNING rollouts on identical windows; the noise floor for all of them is the exactly-zero control in section 1.

| arm | metric | scale-cull 0.95 only | gated sky 0.3 only | both (= the videos) |
|---|---|---|---|---|
| flagship-v1 | `ade_0_2s` | +4.4886 [+3.1457, +5.9990] **sep** | -0.4565 [-0.9987, +0.1179] | +6.0522 [+4.1290, +7.7424] **sep** |
| flagship-v1 | `abs_target_speed_err_ms` | +3.7336 [+2.5937, +4.9494] **sep** | -0.4646 [-0.9171, +0.0548] | +5.0378 [+3.6373, +6.2675] **sep** |
| flagship-v1 | `along_track_ade_m` | +4.6411 [+3.2949, +6.1799] **sep** | -0.4507 [-1.0273, +0.1695] | +6.1943 [+4.3371, +7.8249] **sep** |
| flagship-v1 | `heading_err_rad` | +0.0506 [-0.0591, +0.1279] | -0.0067 [-0.0234, +0.0059] | +0.0669 [-0.0587, +0.1710] |
| flagship-v1 | `curvature_err_1pm` | +0.0144 [+0.0056, +0.0221] **sep** | -0.0008 [-0.0027, +0.0003] | +0.0105 [-0.0009, +0.0195] |
| flagship-v1 | `yawrate_err_rads` | +0.0308 [+0.0029, +0.0565] **sep** | +0.0042 [+0.0012, +0.0067] **sep** | +0.0624 [+0.0173, +0.1120] **sep** |
| flagship-v1 | `cross_track_abs_m` | +0.4029 [-1.1489, +1.9781] | +0.2445 [-0.0292, +0.5183] | +2.9225 [-0.0760, +6.2240] |
| flagship-v1 | `manoeuvre_plan_eq_logged` | -0.0556 [-0.2046, +0.0978] | +0.0178 [-0.0445, +0.0756] | -0.1378 [-0.2267, -0.0400] **sep** |
| flagship-v1 | `route_corridor_departure_rate` | +0.0756 [-0.1489, +0.2844] | +0.0511 [-0.0133, +0.1267] | +0.2578 [+0.0599, +0.4622] **sep** |
| refc-base | `ade_0_2s` | -0.0719 [-0.3137, +0.1655] | +0.0261 [-0.0001, +0.0555] | -0.0786 [-0.2863, +0.0811] |
| refc-base | `abs_target_speed_err_ms` | -0.0636 [-0.2283, +0.0950] | +0.0260 [+0.0041, +0.0470] **sep** | -0.0548 [-0.2034, +0.0707] |
| refc-base | `along_track_ade_m` | -0.0682 [-0.3080, +0.1691] | +0.0305 [+0.0026, +0.0625] **sep** | -0.0764 [-0.2847, +0.0847] |
| refc-base | `heading_err_rad` | -0.0035 [-0.0091, +0.0019] | -0.0013 [-0.0035, +0.0009] | -0.0065 [-0.0119, -0.0005] **sep** |
| refc-base | `curvature_err_1pm` | -0.0001 [-0.0002, +0.0001] | -0.0000 [-0.0001, +0.0001] | -0.0002 [-0.0004, -0.0000] **sep** |
| refc-base | `yawrate_err_rads` | -0.0005 [-0.0032, +0.0022] | -0.0004 [-0.0012, +0.0004] | -0.0020 [-0.0052, +0.0011] |
| refc-base | `cross_track_abs_m` | -0.1515 [-0.4713, +0.1465] | -0.0146 [-0.0726, +0.0594] | -0.2552 [-0.5252, -0.0071] **sep** |
| refc-base | `manoeuvre_plan_eq_logged` | -0.0343 [-0.1076, +0.0245] | +0.0046 [-0.0133, +0.0206] | -0.0183 [-0.0889, +0.0366] |
| refc-base | `route_corridor_departure_rate` | -0.0297 [-0.0711, +0.0047] | +0.0023 [-0.0067, +0.0137] | -0.0366 [-0.0801, +0.0000] |

## 6. Controls on the instrument itself

| run/arm | ADE recomputed independently | max(lon,lat) ≤ ADE ≤ lon+lat | `dist_to_gt` == `|cross_track|` |
|---|---|---|---|
| HQ/flagship-v1 | max|Δ| 0.0 → PASS | PASS | IDENTICAL |
| HQ/refc-base | max|Δ| 0.0 → PASS | PASS | IDENTICAL |
| MORNING/flagship-v1 | max|Δ| 0.0 → PASS | PASS | IDENTICAL |
| MORNING/refc-base | max|Δ| 0.0 → PASS | PASS | IDENTICAL |
| REPRO/flagship-v1 | max|Δ| 0.0 → PASS | PASS | IDENTICAL |
| REPRO/refc-base | max|Δ| 0.0 → PASS | PASS | IDENTICAL |

⚠️ CONFIRMED IDENTICAL — `dist_to_gt_traj_m` and `cross_track_abs_m` are ONE measurement reported twice, under ADE and under LATERAL. A table that lists both is showing one separation, not two.

⚠️ Scope of the ADE control: PLUMBING + ARITHMETIC ONLY. Same definition, separate implementation: catches a mis-indexed pose or a stale field, does not validate the choice of metric.

## 7. Why "MORNING" here is a RE-SCORE, not the published morning file

The published panel (`results/metrics_empty.json`) was scored **before** the route-head key fix, and recorded REF-C as exposing no strategic route logits — which is false; the head was there and a key-name mismatch deleted a family for one arm. Comparing the published file against HQ would confound **the render** with **a scorer fix**. Every render comparison above therefore uses the morning ROLLOUTS re-scored with today's scorer.

| | published morning | morning re-scored today |
|---|---|---|
| REF-C `route_head_eq_logged` | n/a (this arm exposes no strategic route logits a…) | 0.2130 [0.0400, 0.3769] ⛔degenerate |
| paired deltas unchanged by the re-score | 10 of 10 | — |

## 8. NOT RESOLVED — pre-registered, with both outcomes committed

1. **Is flagship's objects-vs-empty gain agent-aware, or just appearance?** On this render 12 of 23 paired deltas separate for flagship, all saying *actors make it better*; REF-C stays null at 1 of 23. But flagship is the arm whose plan moves 9 m under **any** appearance change, so "116 k extra gaussians changed the frame" and "the model reasons about the agents" predict the same sign.
   **The discriminating run:** repeat `--condition objects` with the actors drawn at a WRONG TIME (shift `renderer._actor["tracks"].t0_us`; the render-quality falsifier already uses exactly this control and it separates by +0.2358 grad-NCC on the image). *Committed in advance:* if wrong-time actors recover the same gain, the effect is appearance and the objects-vs-empty result says nothing about agent reasoning; if only correctly-timed actors do, flagship is responding to the agents.

2. ⛔ **RESOLVED, AND IT CONFIRMS THE CONFOUND: the `objects` condition is NOT deterministic across today's code change, so no `objects` morning-vs-HQ number is admissible.** `closedloop_drive.py` gained `act["tracks"].t0_us = float(t_us)` today, a line that only executes with actors attached, so the exactly-zero `empty` control does not cover it. MEASURED by re-running the morning `objects` config today (`CONTROL_<arm>_objmorn_vs_morning.json`):

   - `flagship-v1`: **18 of 19** paired deltas non-zero, **4 separated** — from the CODE CHANGE ALONE, with the render held fixed.
   - `refc-base`: **18 of 19** paired deltas non-zero, **7 separated** — from the CODE CHANGE ALONE, with the render held fixed.
   Driven-path floor from the code change: flagship **1.536 m mean / 7.266 m max**, REF-C **0.165 / 1.299 m** (450 windows). ⇒ **The `empty` headline does not depend on this** (its control is exactly 0.0), and the HQ-internal objects-vs-empty contrast is still valid (same code, same day, same render for both cells) — but the morning `objects` panel and the HQ `objects` panel are not comparable.

3. **Which render is "right" is not settled by this panel.** HQ is closer to the shipped reference (grad-NCC 0.3424 vs 0.2774), so its flagship numbers are the better estimate of flagship on *this* scene — but both are within-sim and REF-C's open-loop ADE is 3.21× OOD here vs real footage. The transferable claim is the **ordering** and the **render-sensitivity ratio**, not either arm's absolute rate.

4. **A render-invariance check belongs in the closed-loop protocol.** Every future closed-loop panel should report the arm's sensitivity to a render perturbation next to its score, because an arm that moves 9 m under one is not measurable to 0.1 m.
