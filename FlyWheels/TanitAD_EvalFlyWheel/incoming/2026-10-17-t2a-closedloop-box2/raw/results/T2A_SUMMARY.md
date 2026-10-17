TIER T2 (perception-closed loop, in-house harness). The loop is closed on PERCEPTION, not on INTERACTION: nothing in the scene responds to the ego (actors are replayed or absent), there is no map and no drivable-area polygon, and the vehicle model has no collision response. This measures HOW WELL THE MODEL DRIVES ITSELF; it cannot measure WHAT HAPPENS WHEN IT DRIVES BADLY, and it cannot say 'safe'. n = 2 scenes; clusters are disjoint rollout starts of ONE clip each, not independent episodes. WITHIN-SIM RELATIVE: absolute levels do not transfer to real footage (REF-C open-loop ADE 1.5157 on these reconstructions vs 0.4728 on real footage, 3.21x OOD) -- orderings and paired differences may.

| scene | arm | ceiling | n_win | ade_0_2s [95% CI] | dist_to_log_m | cross_track_m | |speed err| m/s | progress |
|---|---|---|---|---|---|---|---|---|
| 00040136 | cl_ha0 | - | 437 | +2.745 [+1.940, +3.562] | +2.678 [+1.450, +3.865] | +2.678 [+1.450, +3.865] | +1.648 [+1.054, +2.239] | 1.028 |
| 00040136 | cl_ha0_ext | - | 436 | +2.655 [+1.635, +3.987] | +0.937 [+0.483, +1.419] | +0.937 [+0.483, +1.419] | +1.591 [+0.962, +2.348] | 1.004 |
| 00040136 | cl_ha0_ext_live | - | 437 | +2.681 [+1.898, +3.484] | +2.239 [+1.202, +3.238] | +2.239 [+1.202, +3.238] | +1.628 [+1.047, +2.207] | 1.023 |
| 00040136 | cl_p10 | - | 436 | +3.672 [+2.204, +5.090] | +2.032 [+1.118, +2.950] | +2.032 [+1.118, +2.950] | +2.444 [+1.423, +3.407] | 1.02 |
| 00040136 | cl_p10_t0 | - | 437 | +2.524 [+1.681, +3.416] | +0.887 [+0.416, +1.391] | +0.887 [+0.416, +1.391] | +1.643 [+1.018, +2.276] | 1.003 |
| 00040136 | refcv7 | ORACLE per-rollout max logged speed over [t0+2 s, t0+6 s] (v_max_valid=1) | 436 | +3.337 [+2.022, +4.622] | +0.653 [+0.347, +1.004] | +0.653 [+0.347, +1.004] | +2.237 [+1.359, +3.099] | 1.0 |
| 00040136 | refcv7 | ORACLE per-rollout max logged speed over [t0+2 s, t0+6 s] (v_max_valid=1) | 436 | +3.371 [+2.083, +4.639] | +0.650 [+0.347, +0.996] | +0.650 [+0.347, +0.996] | +2.255 [+1.378, +3.102] | 0.999 |
| 00040136 | refcv7 | withheld (v_max_valid=0) -- deployable | 437 | +3.296 [+2.065, +4.512] | +0.651 [+0.344, +1.001] | +0.651 [+0.344, +1.001] | +2.209 [+1.355, +3.029] | 0.998 |
| 00040136 | refcv7 | withheld (v_max_valid=0) -- deployable | 437 | +3.308 [+2.067, +4.538] | +0.674 [+0.361, +1.028] | +0.674 [+0.361, +1.028] | +2.220 [+1.357, +3.054] | 0.999 |
| 7c72937c | cl_ha0 | - | 442 | +2.731 [+1.782, +3.642] | +2.525 [+0.783, +4.429] | +2.525 [+0.783, +4.429] | +1.126 [+0.451, +1.860] | 1.338 |
| 7c72937c | cl_ha0_ext | - | 437 | +1.889 [+0.993, +2.887] | +2.267 [+0.861, +3.752] | +2.267 [+0.861, +3.752] | +0.750 [+0.356, +1.220] | 1.279 |
| 7c72937c | cl_ha0_ext_live | - | 441 | +2.652 [+1.722, +3.542] | +2.316 [+0.623, +4.206] | +2.316 [+0.623, +4.206] | +1.100 [+0.433, +1.819] | 1.336 |
| 7c72937c | cl_p10 | - | 439 | +2.680 [+2.038, +3.299] | +2.603 [+0.838, +4.557] | +2.603 [+0.838, +4.557] | +1.055 [+0.513, +1.658] | 1.357 |
| 7c72937c | cl_p10_t0 | - | 440 | +2.220 [+1.643, +2.806] | +2.143 [+0.702, +3.766] | +2.143 [+0.702, +3.766] | +1.048 [+0.440, +1.690] | 1.247 |
| 7c72937c | refcv7 | ORACLE per-rollout max logged speed over [t0+2 s, t0+6 s] (v_max_valid=1) | 439 | +2.198 [+1.247, +3.129] | +1.961 [+0.453, +3.840] | +1.961 [+0.453, +3.840] | +0.980 [+0.375, +1.625] | 1.288 |
| 7c72937c | refcv7 | ORACLE per-rollout max logged speed over [t0+2 s, t0+6 s] (v_max_valid=1) | 439 | +2.203 [+1.247, +3.133] | +1.962 [+0.454, +3.841] | +1.962 [+0.454, +3.841] | +0.983 [+0.378, +1.628] | 1.288 |
| 7c72937c | refcv7 | withheld (v_max_valid=0) -- deployable | 439 | +2.201 [+1.251, +3.124] | +1.955 [+0.448, +3.829] | +1.955 [+0.448, +3.829] | +0.983 [+0.364, +1.633] | 1.297 |
| 7c72937c | refcv7 | withheld (v_max_valid=0) -- deployable | 439 | +2.201 [+1.257, +3.106] | +2.008 [+0.490, +3.892] | +2.008 [+0.490, +3.892] | +0.980 [+0.378, +1.607] | 1.28 |
