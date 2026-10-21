# READ_48852 -- bar-deciding cells, READ from banked artifacts (EARLY READ; T1 self-action open loop; single training seed)

## step48852ema_unc

* compliant re-read `raw/compliant_reread/trees/step48852ema_unc.json`: controls ALL_PASS = **True** (C4 pose reference valid: True), status WRITTEN (all controls passed)
  * BAR-R8-E1 / E3 reference (D-R8-E1-REF): **ha0_ext_pose (D-R8-E1-REF, PI 2026-10-19)**; the registered `ha0_ext` read is BESIDE
  * `compliant/capped`  [reference ha0_ext_pose (D-R8-E1-REF, PI 2026-10-19)]  E1 FAILED  (s0 +0.0059 [-0.0123, +0.0241]; s1 +0.0046 [-0.0121, +0.0215])
    E3 PASS (single training seed; training-seed floor unmeasured)  (s0 -0.5044 [-0.8708, -0.1598] sep; s1 -0.5085 [-0.8632, -0.1773] sep)
    E1 beside (ha0_ext) FAILED  (s0 +0.0110 [-0.0065, +0.0292]; s1 +0.0097 [-0.0063, +0.0265])
    E3 beside (ha0_ext) PASS (single training seed; training-seed floor unmeasured)  (s0 -0.4756 [-0.8323, -0.1406] sep; s1 -0.4797 [-0.8302, -0.1536] sep)
    NR per family {"ADE": "FAIL: S2:ade_m, S6:ade_m", "LATERAL": "FAIL: S2:LAT_heading_mae_deg", "LONGITUDINAL": "FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m"}
  * `compliant/uncapped`  [reference ha0_ext_pose (D-R8-E1-REF, PI 2026-10-19)]  E1 FAILED  (s0 +0.0059 [-0.0123, +0.0241]; s1 +0.0046 [-0.0121, +0.0215])
    E3 PASS (single training seed; training-seed floor unmeasured)  (s0 -0.5044 [-0.8708, -0.1598] sep; s1 -0.5085 [-0.8632, -0.1773] sep)
    E1 beside (ha0_ext) FAILED  (s0 +0.0110 [-0.0065, +0.0292]; s1 +0.0097 [-0.0063, +0.0265])
    E3 beside (ha0_ext) PASS (single training seed; training-seed floor unmeasured)  (s0 -0.4756 [-0.8323, -0.1406] sep; s1 -0.4797 [-0.8302, -0.1536] sep)
    NR per family {"ADE": "FAIL: S2:ade_m, S6:ade_m", "LATERAL": "FAIL: S2:LAT_heading_mae_deg", "LONGITUDINAL": "FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m"}
  * `all_window/capped`  [reference ha0_ext_pose (D-R8-E1-REF, PI 2026-10-19)]  E1 FAILED  (s0 +0.0738 [+0.0376, +0.1157]; s1 +0.0736 [+0.0375, +0.1149])
    E3 PASS (single training seed; training-seed floor unmeasured)  (s0 -0.4652 [-0.8394, -0.0955] sep; s1 -0.4520 [-0.8225, -0.0866] sep)
    E1 beside (ha0_ext) FAILED  (s0 +0.0794 [+0.0437, +0.1212]; s1 +0.0791 [+0.0434, +0.1202])
    E3 beside (ha0_ext) PASS (single training seed; training-seed floor unmeasured)  (s0 -0.4366 [-0.8069, -0.0708] sep; s1 -0.4234 [-0.7873, -0.0653] sep)
    NR per family {"ADE": "FAIL: S2:ade_m, S6:ade_m", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid", "LONGITUDINAL": "FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m"}
  * `all_window/uncapped`  [reference ha0_ext_pose (D-R8-E1-REF, PI 2026-10-19)]  E1 FAILED  (s0 +0.0136 [-0.0036, +0.0318]; s1 +0.0144 [-0.0022, +0.0318])
    E3 PASS (single training seed; training-seed floor unmeasured)  (s0 -0.5784 [-0.9219, -0.2247] sep; s1 -0.5587 [-0.8984, -0.2182] sep)
    E1 beside (ha0_ext) FAILED  (s0 +0.0192 [+0.0022, +0.0369]; s1 +0.0199 [+0.0038, +0.0368])
    E3 beside (ha0_ext) PASS (single training seed; training-seed floor unmeasured)  (s0 -0.5498 [-0.8872, -0.2035] sep; s1 -0.5301 [-0.8641, -0.1934] sep)
    NR per family {"ADE": "FAIL: S2:ade_m, S6:ade_m", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid", "LONGITUDINAL": "FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m"}

* forward `raw/a0k0_headroom/FORWARD_step48852ema_unc.json`: controls ALL_PASS = **True**
  * `compliant|RS_FROZEN30k` FZ-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0395 [-0.0514, -0.0287] sep; s1 -0.0402 [-0.0512, -0.0300] sep)
  * `compliant|RS_FROZEN30k` FZ-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0446 [-0.0573, -0.0333] sep; s1 -0.0453 [-0.0572, -0.0343] sep)
  * `compliant|RS_FROZEN30k` FZ-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.8163 [-1.0963, -0.5527] sep; s1 -0.8174 [-1.0915, -0.5617] sep)
  * `compliant|RS_FROZEN30k` FZ-NR: {"ADE": "PASS", "LATERAL": "FAIL: S2:LAT_curvature_mae_1pm_valid", "LONGITUDINAL": "PASS"}
  * `compliant|GATED_RS_FROZEN30k` FZ-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0403 [-0.0522, -0.0284] sep; s1 -0.0410 [-0.0521, -0.0300] sep)
  * `compliant|GATED_RS_FROZEN30k` FZ-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0454 [-0.0582, -0.0333] sep; s1 -0.0461 [-0.0579, -0.0344] sep)
  * `compliant|GATED_RS_FROZEN30k` FZ-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.9257 [-1.1974, -0.6732] sep; s1 -0.9244 [-1.1908, -0.6774] sep)
  * `compliant|GATED_RS_FROZEN30k` FZ-NR: {"ADE": "PASS", "LATERAL": "PASS", "LONGITUDINAL": "PASS"}
  * `all|RS_FROZEN30k` FZ-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0374 [-0.0482, -0.0272] sep; s1 -0.0374 [-0.0480, -0.0275] sep)
  * `all|RS_FROZEN30k` FZ-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0429 [-0.0545, -0.0319] sep; s1 -0.0429 [-0.0542, -0.0323] sep)
  * `all|RS_FROZEN30k` FZ-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.9041 [-1.1912, -0.6356] sep; s1 -0.8941 [-1.1697, -0.6279] sep)
  * `all|RS_FROZEN30k` FZ-NR: {"ADE": "PASS", "LATERAL": "FAIL: S2:LAT_curvature_mae_1pm_valid", "LONGITUDINAL": "PASS"}
  * `all|GATED_RS_FROZEN30k` FZ-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0394 [-0.0510, -0.0282] sep; s1 -0.0394 [-0.0504, -0.0289] sep)
  * `all|GATED_RS_FROZEN30k` FZ-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0450 [-0.0574, -0.0330] sep; s1 -0.0450 [-0.0568, -0.0334] sep)
  * `all|GATED_RS_FROZEN30k` FZ-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -1.0144 [-1.2956, -0.7586] sep; s1 -1.0034 [-1.2788, -0.7485] sep)
  * `all|GATED_RS_FROZEN30k` FZ-NR: {"ADE": "PASS", "LATERAL": "PASS", "LONGITUDINAL": "PASS"}
* forward `raw/a0k0_lon/FORWARD_LON_step48852ema_unc.json`: controls ALL_PASS = **True**
  * `compliant|RS_LON_FROZEN30k` FZL-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0214 [-0.0332, -0.0095] sep; s1 -0.0216 [-0.0330, -0.0109] sep)
  * `compliant|RS_LON_FROZEN30k` FZL-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0265 [-0.0389, -0.0144] sep; s1 -0.0267 [-0.0385, -0.0154] sep)
  * `compliant|RS_LON_FROZEN30k` FZL-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.7905 [-1.0741, -0.5234] sep; s1 -0.7902 [-1.0646, -0.5329] sep)
  * `compliant|RS_LON_FROZEN30k` FZL-NR: {"ADE": "PASS", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid", "LONGITUDINAL": "PASS"}
  * `compliant|RS_LON_FROZEN30k` FZL-LAT: ?  ()
  * `compliant|GATED_RS_LON_FROZEN30k` FZL-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0221 [-0.0344, -0.0092] sep; s1 -0.0222 [-0.0339, -0.0104] sep)
  * `compliant|GATED_RS_LON_FROZEN30k` FZL-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0273 [-0.0400, -0.0140] sep; s1 -0.0274 [-0.0396, -0.0152] sep)
  * `compliant|GATED_RS_LON_FROZEN30k` FZL-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.8892 [-1.1637, -0.6357] sep; s1 -0.8854 [-1.1494, -0.6348] sep)
  * `compliant|GATED_RS_LON_FROZEN30k` FZL-NR: {"ADE": "PASS", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid", "LONGITUDINAL": "PASS"}
  * `compliant|GATED_RS_LON_FROZEN30k` FZL-LAT: ?  ()
  * `compliant|RS_LAT1_FROZEN30k` FZL-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0214 [-0.0332, -0.0095] sep; s1 -0.0216 [-0.0330, -0.0109] sep)
  * `compliant|RS_LAT1_FROZEN30k` FZL-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0265 [-0.0389, -0.0144] sep; s1 -0.0267 [-0.0385, -0.0154] sep)
  * `compliant|RS_LAT1_FROZEN30k` FZL-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.7905 [-1.0741, -0.5234] sep; s1 -0.7902 [-1.0646, -0.5329] sep)
  * `compliant|RS_LAT1_FROZEN30k` FZL-NR: {"ADE": "PASS", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid", "LONGITUDINAL": "PASS"}
  * `compliant|RS_LAT1_FROZEN30k` FZL-LAT: ?  ()
  * `all|RS_LON_FROZEN30k` FZL-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0194 [-0.0312, -0.0079] sep; s1 -0.0186 [-0.0301, -0.0075] sep)
  * `all|RS_LON_FROZEN30k` FZL-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0249 [-0.0371, -0.0129] sep; s1 -0.0241 [-0.0358, -0.0124] sep)
  * `all|RS_LON_FROZEN30k` FZL-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.8791 [-1.1662, -0.6085] sep; s1 -0.8674 [-1.1451, -0.6009] sep)
  * `all|RS_LON_FROZEN30k` FZL-NR: {"ADE": "PASS", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid", "LONGITUDINAL": "PASS"}
  * `all|RS_LON_FROZEN30k` FZL-LAT: ?  ()
  * `all|GATED_RS_LON_FROZEN30k` FZL-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0211 [-0.0336, -0.0086] sep; s1 -0.0202 [-0.0318, -0.0085] sep)
  * `all|GATED_RS_LON_FROZEN30k` FZL-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0266 [-0.0395, -0.0138] sep; s1 -0.0257 [-0.0381, -0.0135] sep)
  * `all|GATED_RS_LON_FROZEN30k` FZL-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.9798 [-1.2603, -0.7190] sep; s1 -0.9662 [-1.2396, -0.7103] sep)
  * `all|GATED_RS_LON_FROZEN30k` FZL-NR: {"ADE": "PASS", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid", "LONGITUDINAL": "PASS"}
  * `all|GATED_RS_LON_FROZEN30k` FZL-LAT: ?  ()
  * `all|RS_LAT1_FROZEN30k` FZL-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0194 [-0.0312, -0.0079] sep; s1 -0.0186 [-0.0301, -0.0075] sep)
  * `all|RS_LAT1_FROZEN30k` FZL-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0249 [-0.0371, -0.0129] sep; s1 -0.0241 [-0.0358, -0.0124] sep)
  * `all|RS_LAT1_FROZEN30k` FZL-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.8791 [-1.1662, -0.6085] sep; s1 -0.8674 [-1.1451, -0.6009] sep)
  * `all|RS_LAT1_FROZEN30k` FZL-NR: {"ADE": "PASS", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid", "LONGITUDINAL": "PASS"}
  * `all|RS_LAT1_FROZEN30k` FZL-LAT: ?  ()
* forward `raw/a0k0_retime/FORWARD_RETIME_step48852ema_unc.json`: controls ALL_PASS = **True**
  * `compliant|RETIME_FROZEN30k` FZR-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0233 [-0.0352, -0.0116] sep; s1 -0.0236 [-0.0349, -0.0126] sep)
  * `compliant|RETIME_FROZEN30k` FZR-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0284 [-0.0408, -0.0163] sep; s1 -0.0288 [-0.0407, -0.0173] sep)
  * `compliant|RETIME_FROZEN30k` FZR-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.9160 [-1.2045, -0.6508] sep; s1 -0.9161 [-1.1899, -0.6612] sep)
  * `compliant|RETIME_FROZEN30k` FZR-NR: {"ADE": "PASS", "LATERAL": "PASS", "LONGITUDINAL": "PASS"}
  * `compliant|RETIME_FROZEN30k` FZR-LAT: ?  ()
  * `compliant|GATED_RETIME_FROZEN30k` FZR-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0240 [-0.0363, -0.0112] sep; s1 -0.0243 [-0.0358, -0.0124] sep)
  * `compliant|GATED_RETIME_FROZEN30k` FZR-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0291 [-0.0418, -0.0160] sep; s1 -0.0294 [-0.0415, -0.0175] sep)
  * `compliant|GATED_RETIME_FROZEN30k` FZR-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.9691 [-1.2530, -0.7121] sep; s1 -0.9676 [-1.2413, -0.7148] sep)
  * `compliant|GATED_RETIME_FROZEN30k` FZR-NR: {"ADE": "PASS", "LATERAL": "PASS", "LONGITUDINAL": "PASS"}
  * `compliant|GATED_RETIME_FROZEN30k` FZR-LAT: ?  ()
  * `all|RETIME_FROZEN30k` FZR-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0212 [-0.0331, -0.0094] sep; s1 -0.0206 [-0.0324, -0.0092] sep)
  * `all|RETIME_FROZEN30k` FZR-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0267 [-0.0390, -0.0144] sep; s1 -0.0261 [-0.0382, -0.0143] sep)
  * `all|RETIME_FROZEN30k` FZR-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -1.0065 [-1.3019, -0.7297] sep; s1 -0.9987 [-1.2912, -0.7263] sep)
  * `all|RETIME_FROZEN30k` FZR-NR: {"ADE": "PASS", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid", "LONGITUDINAL": "PASS"}
  * `all|RETIME_FROZEN30k` FZR-LAT: ?  ()
  * `all|GATED_RETIME_FROZEN30k` FZR-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0227 [-0.0354, -0.0101] sep; s1 -0.0219 [-0.0336, -0.0099] sep)
  * `all|GATED_RETIME_FROZEN30k` FZR-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0282 [-0.0413, -0.0150] sep; s1 -0.0275 [-0.0400, -0.0147] sep)
  * `all|GATED_RETIME_FROZEN30k` FZR-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -1.0579 [-1.3544, -0.7892] sep; s1 -1.0491 [-1.3418, -0.7844] sep)
  * `all|GATED_RETIME_FROZEN30k` FZR-NR: {"ADE": "PASS", "LATERAL": "FAIL: S2:LAT_heading_mae_deg", "LONGITUDINAL": "PASS"}
  * `all|GATED_RETIME_FROZEN30k` FZR-LAT: ?  ()

* heading drift: `raw/heading_drift/HEADING_DRIFT_step48852ema.json` present = True

## step48852_unc

* compliant re-read `raw/compliant_reread/trees/step48852_unc.json`: controls ALL_PASS = **True** (C4 pose reference valid: True), status WRITTEN (all controls passed)
  * BAR-R8-E1 / E3 reference (D-R8-E1-REF): **ha0_ext_pose (D-R8-E1-REF, PI 2026-10-19)**; the registered `ha0_ext` read is BESIDE
  * `compliant/capped`  [reference ha0_ext_pose (D-R8-E1-REF, PI 2026-10-19)]  E1 FAILED  (s0 +0.0120 [-0.0060, +0.0311]; s1 +0.0119 [-0.0057, +0.0304])
    E3 PASS (single training seed; training-seed floor unmeasured)  (s0 -0.4419 [-0.8172, -0.0940] sep; s1 -0.4483 [-0.8127, -0.1073] sep)
    E1 beside (ha0_ext) FAILED  (s0 +0.0171 [-0.0006, +0.0361]; s1 +0.0170 [-0.0003, +0.0353])
    E3 beside (ha0_ext) PASS (single training seed; training-seed floor unmeasured)  (s0 -0.4131 [-0.7805, -0.0711] sep; s1 -0.4195 [-0.7803, -0.0833] sep)
    NR per family {"ADE": "FAIL: S2:ade_m, S6:ade_m", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid", "LONGITUDINAL": "FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m"}
  * `compliant/uncapped`  [reference ha0_ext_pose (D-R8-E1-REF, PI 2026-10-19)]  E1 FAILED  (s0 +0.0120 [-0.0060, +0.0311]; s1 +0.0119 [-0.0057, +0.0304])
    E3 PASS (single training seed; training-seed floor unmeasured)  (s0 -0.4419 [-0.8172, -0.0940] sep; s1 -0.4483 [-0.8127, -0.1073] sep)
    E1 beside (ha0_ext) FAILED  (s0 +0.0171 [-0.0006, +0.0361]; s1 +0.0170 [-0.0003, +0.0353])
    E3 beside (ha0_ext) PASS (single training seed; training-seed floor unmeasured)  (s0 -0.4131 [-0.7805, -0.0711] sep; s1 -0.4195 [-0.7803, -0.0833] sep)
    NR per family {"ADE": "FAIL: S2:ade_m, S6:ade_m", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid", "LONGITUDINAL": "FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m"}
  * `all_window/capped`  [reference ha0_ext_pose (D-R8-E1-REF, PI 2026-10-19)]  E1 FAILED  (s0 +0.0790 [+0.0429, +0.1207]; s1 +0.0801 [+0.0439, +0.1213])
    E3 PASS (single training seed; training-seed floor unmeasured)  (s0 -0.4069 [-0.7823, -0.0299] sep; s1 -0.3983 [-0.7790, -0.0247] sep)
    E1 beside (ha0_ext) FAILED  (s0 +0.0845 [+0.0486, +0.1257]; s1 +0.0856 [+0.0499, +0.1264])
    E3 beside (ha0_ext) PASS (single training seed; training-seed floor unmeasured)  (s0 -0.3783 [-0.7485, -0.0043] sep; s1 -0.3698 [-0.7419, -0.0006] sep)
    NR per family {"ADE": "FAIL: S2:ade_m, S6:ade_m", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid", "LONGITUDINAL": "FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m"}
  * `all_window/uncapped`  [reference ha0_ext_pose (D-R8-E1-REF, PI 2026-10-19)]  E1 FAILED  (s0 +0.0189 [+0.0014, +0.0370]; s1 +0.0208 [+0.0036, +0.0391])
    E3 PASS (single training seed; training-seed floor unmeasured)  (s0 -0.5171 [-0.8707, -0.1674] sep; s1 -0.5018 [-0.8525, -0.1494] sep)
    E1 beside (ha0_ext) FAILED  (s0 +0.0244 [+0.0075, +0.0420]; s1 +0.0264 [+0.0095, +0.0442])
    E3 beside (ha0_ext) PASS (single training seed; training-seed floor unmeasured)  (s0 -0.4886 [-0.8234, -0.1408] sep; s1 -0.4732 [-0.8126, -0.1294] sep)
    NR per family {"ADE": "FAIL: S2:ade_m, S6:ade_m", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid", "LONGITUDINAL": "FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m"}

* forward `raw/a0k0_headroom/FORWARD_step48852_unc.json`: controls ALL_PASS = **True**
  * `compliant|RS_FROZEN30k` FZ-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0374 [-0.0491, -0.0265] sep; s1 -0.0373 [-0.0489, -0.0266] sep)
  * `compliant|RS_FROZEN30k` FZ-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0425 [-0.0551, -0.0311] sep; s1 -0.0424 [-0.0547, -0.0313] sep)
  * `compliant|RS_FROZEN30k` FZ-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.7830 [-1.0631, -0.5182] sep; s1 -0.7852 [-1.0624, -0.5246] sep)
  * `compliant|RS_FROZEN30k` FZ-NR: {"ADE": "PASS", "LATERAL": "FAIL: S2:LAT_curvature_mae_1pm_valid", "LONGITUDINAL": "PASS"}
  * `compliant|GATED_RS_FROZEN30k` FZ-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0384 [-0.0501, -0.0269] sep; s1 -0.0380 [-0.0495, -0.0265] sep)
  * `compliant|GATED_RS_FROZEN30k` FZ-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0435 [-0.0560, -0.0313] sep; s1 -0.0431 [-0.0552, -0.0310] sep)
  * `compliant|GATED_RS_FROZEN30k` FZ-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.9029 [-1.1746, -0.6492] sep; s1 -0.9050 [-1.1702, -0.6545] sep)
  * `compliant|GATED_RS_FROZEN30k` FZ-NR: {"ADE": "PASS", "LATERAL": "PASS", "LONGITUDINAL": "PASS"}
  * `all|RS_FROZEN30k` FZ-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0355 [-0.0464, -0.0255] sep; s1 -0.0348 [-0.0457, -0.0249] sep)
  * `all|RS_FROZEN30k` FZ-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0411 [-0.0528, -0.0301] sep; s1 -0.0403 [-0.0518, -0.0296] sep)
  * `all|RS_FROZEN30k` FZ-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.8722 [-1.1544, -0.5986] sep; s1 -0.8629 [-1.1441, -0.5925] sep)
  * `all|RS_FROZEN30k` FZ-NR: {"ADE": "PASS", "LATERAL": "FAIL: S2:LAT_curvature_mae_1pm_valid", "LONGITUDINAL": "PASS"}
  * `all|GATED_RS_FROZEN30k` FZ-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0377 [-0.0489, -0.0266] sep; s1 -0.0368 [-0.0479, -0.0257] sep)
  * `all|GATED_RS_FROZEN30k` FZ-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0432 [-0.0554, -0.0310] sep; s1 -0.0423 [-0.0542, -0.0304] sep)
  * `all|GATED_RS_FROZEN30k` FZ-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.9904 [-1.2695, -0.7335] sep; s1 -0.9827 [-1.2569, -0.7248] sep)
  * `all|GATED_RS_FROZEN30k` FZ-NR: {"ADE": "PASS", "LATERAL": "FAIL: S2:LAT_curvature_mae_1pm_valid", "LONGITUDINAL": "PASS"}
* forward `raw/a0k0_lon/FORWARD_LON_step48852_unc.json`: controls ALL_PASS = **True**
  * `compliant|RS_LON_FROZEN30k` FZL-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0191 [-0.0308, -0.0076] sep; s1 -0.0182 [-0.0299, -0.0071] sep)
  * `compliant|RS_LON_FROZEN30k` FZL-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0243 [-0.0364, -0.0123] sep; s1 -0.0233 [-0.0358, -0.0117] sep)
  * `compliant|RS_LON_FROZEN30k` FZL-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.7570 [-1.0352, -0.4869] sep; s1 -0.7572 [-1.0361, -0.4936] sep)
  * `compliant|RS_LON_FROZEN30k` FZL-NR: {"ADE": "PASS", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid", "LONGITUDINAL": "PASS"}
  * `compliant|RS_LON_FROZEN30k` FZL-LAT: ?  ()
  * `compliant|GATED_RS_LON_FROZEN30k` FZL-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0200 [-0.0322, -0.0075] sep; s1 -0.0188 [-0.0311, -0.0061] sep)
  * `compliant|GATED_RS_LON_FROZEN30k` FZL-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0252 [-0.0377, -0.0122] sep; s1 -0.0240 [-0.0365, -0.0111] sep)
  * `compliant|GATED_RS_LON_FROZEN30k` FZL-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.8653 [-1.1415, -0.6088] sep; s1 -0.8655 [-1.1337, -0.6124] sep)
  * `compliant|GATED_RS_LON_FROZEN30k` FZL-NR: {"ADE": "PASS", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid", "LONGITUDINAL": "PASS"}
  * `compliant|GATED_RS_LON_FROZEN30k` FZL-LAT: ?  ()
  * `compliant|RS_LAT1_FROZEN30k` FZL-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0191 [-0.0308, -0.0076] sep; s1 -0.0182 [-0.0299, -0.0071] sep)
  * `compliant|RS_LAT1_FROZEN30k` FZL-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0243 [-0.0364, -0.0123] sep; s1 -0.0233 [-0.0358, -0.0117] sep)
  * `compliant|RS_LAT1_FROZEN30k` FZL-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.7570 [-1.0352, -0.4869] sep; s1 -0.7572 [-1.0361, -0.4936] sep)
  * `compliant|RS_LAT1_FROZEN30k` FZL-NR: {"ADE": "PASS", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid", "LONGITUDINAL": "PASS"}
  * `compliant|RS_LAT1_FROZEN30k` FZL-LAT: ?  ()
  * `all|RS_LON_FROZEN30k` FZL-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0175 [-0.0290, -0.0059] sep; s1 -0.0155 [-0.0272, -0.0040] sep)
  * `all|RS_LON_FROZEN30k` FZL-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0230 [-0.0350, -0.0110] sep; s1 -0.0211 [-0.0333, -0.0090] sep)
  * `all|RS_LON_FROZEN30k` FZL-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.8472 [-1.1294, -0.5744] sep; s1 -0.8352 [-1.1171, -0.5643] sep)
  * `all|RS_LON_FROZEN30k` FZL-NR: {"ADE": "FAIL: S6:ade_m", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid", "LONGITUDINAL": "PASS"}
  * `all|RS_LON_FROZEN30k` FZL-LAT: ?  ()
  * `all|GATED_RS_LON_FROZEN30k` FZL-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0193 [-0.0316, -0.0068] sep; s1 -0.0171 [-0.0292, -0.0049] sep)
  * `all|GATED_RS_LON_FROZEN30k` FZL-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0248 [-0.0375, -0.0118] sep; s1 -0.0226 [-0.0351, -0.0099] sep)
  * `all|GATED_RS_LON_FROZEN30k` FZL-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.9550 [-1.2357, -0.6962] sep; s1 -0.9447 [-1.2163, -0.6838] sep)
  * `all|GATED_RS_LON_FROZEN30k` FZL-NR: {"ADE": "PASS", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid", "LONGITUDINAL": "PASS"}
  * `all|GATED_RS_LON_FROZEN30k` FZL-LAT: ?  ()
  * `all|RS_LAT1_FROZEN30k` FZL-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0175 [-0.0290, -0.0059] sep; s1 -0.0155 [-0.0272, -0.0040] sep)
  * `all|RS_LAT1_FROZEN30k` FZL-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0230 [-0.0350, -0.0110] sep; s1 -0.0211 [-0.0333, -0.0090] sep)
  * `all|RS_LAT1_FROZEN30k` FZL-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.8472 [-1.1294, -0.5744] sep; s1 -0.8352 [-1.1171, -0.5643] sep)
  * `all|RS_LAT1_FROZEN30k` FZL-NR: {"ADE": "FAIL: S6:ade_m", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid", "LONGITUDINAL": "PASS"}
  * `all|RS_LAT1_FROZEN30k` FZL-LAT: ?  ()
* forward `raw/a0k0_retime/FORWARD_RETIME_step48852_unc.json`: controls ALL_PASS = **True**
  * `compliant|RETIME_FROZEN30k` FZR-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0214 [-0.0328, -0.0101] sep; s1 -0.0205 [-0.0318, -0.0094] sep)
  * `compliant|RETIME_FROZEN30k` FZR-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0265 [-0.0385, -0.0149] sep; s1 -0.0256 [-0.0376, -0.0140] sep)
  * `compliant|RETIME_FROZEN30k` FZR-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.8891 [-1.1793, -0.6248] sep; s1 -0.8896 [-1.1699, -0.6291] sep)
  * `compliant|RETIME_FROZEN30k` FZR-NR: {"ADE": "PASS", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid", "LONGITUDINAL": "PASS"}
  * `compliant|RETIME_FROZEN30k` FZR-LAT: ?  ()
  * `compliant|GATED_RETIME_FROZEN30k` FZR-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0223 [-0.0346, -0.0097] sep; s1 -0.0211 [-0.0333, -0.0083] sep)
  * `compliant|GATED_RETIME_FROZEN30k` FZR-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0274 [-0.0399, -0.0145] sep; s1 -0.0262 [-0.0386, -0.0135] sep)
  * `compliant|GATED_RETIME_FROZEN30k` FZR-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.9471 [-1.2328, -0.6931] sep; s1 -0.9510 [-1.2258, -0.6971] sep)
  * `compliant|GATED_RETIME_FROZEN30k` FZR-NR: {"ADE": "PASS", "LATERAL": "FAIL: S2:LAT_heading_mae_deg", "LONGITUDINAL": "PASS"}
  * `compliant|GATED_RETIME_FROZEN30k` FZR-LAT: ?  ()
  * `all|RETIME_FROZEN30k` FZR-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0197 [-0.0313, -0.0081] sep; s1 -0.0177 [-0.0294, -0.0061] sep)
  * `all|RETIME_FROZEN30k` FZR-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0252 [-0.0373, -0.0132] sep; s1 -0.0232 [-0.0358, -0.0112] sep)
  * `all|RETIME_FROZEN30k` FZR-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.9827 [-1.2769, -0.7049] sep; s1 -0.9727 [-1.2668, -0.6981] sep)
  * `all|RETIME_FROZEN30k` FZR-NR: {"ADE": "PASS", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid", "LONGITUDINAL": "PASS"}
  * `all|RETIME_FROZEN30k` FZR-LAT: ?  ()
  * `all|GATED_RETIME_FROZEN30k` FZR-E1: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0213 [-0.0336, -0.0089] sep; s1 -0.0190 [-0.0311, -0.0065] sep)
  * `all|GATED_RETIME_FROZEN30k` FZR-E1P: PASS (single training seed; training-seed floor unmeasured)  (s0 -0.0268 [-0.0396, -0.0137] sep; s1 -0.0245 [-0.0372, -0.0113] sep)
  * `all|GATED_RETIME_FROZEN30k` FZR-E3: PASS (single training seed; training-seed floor unmeasured)  (s0 -1.0376 [-1.3283, -0.7685] sep; s1 -1.0310 [-1.3223, -0.7642] sep)
  * `all|GATED_RETIME_FROZEN30k` FZR-NR: {"ADE": "PASS", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid", "LONGITUDINAL": "PASS"}
  * `all|GATED_RETIME_FROZEN30k` FZR-LAT: ?  ()

* heading drift: `raw/heading_drift/HEADING_DRIFT_step48852raw.json` present = True

