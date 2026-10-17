TIER T2 (perception-closed loop, in-house harness). The loop is closed on PERCEPTION, not on INTERACTION: nothing in the scene responds to the ego (actors are replayed or absent), there is no map and no drivable-area polygon, and the vehicle model has no collision response. This measures HOW WELL THE MODEL DRIVES ITSELF; it cannot measure WHAT HAPPENS WHEN IT DRIVES BADLY, and it cannot say 'safe'. n = 2 scenes; clusters are disjoint rollout starts of ONE clip each, not independent episodes. WITHIN-SIM RELATIVE: absolute levels do not transfer to real footage (REF-C open-loop ADE 1.5157 on these reconstructions vs 0.4728 on real footage, 3.21x OOD) -- orderings and paired differences may.

| scene | arm | ceiling | n_win | ade_0_2s [95% CI] | dist_to_log_m | cross_track_m | |speed err| m/s | progress |
|---|---|---|---|---|---|---|---|---|
| stub-str | cl_ha0 | - | 100 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 1.0 |
| stub-str | cl_ha0_ext | - | 100 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 1.0 |
| stub-str | cl_ha0_ext_live | - | 100 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 1.0 |
| stub-str | cl_p10 | - | 100 | +1.539 [+1.539, +1.539] | +0.734 [+0.734, +0.734] | +0.734 [+0.734, +0.734] | +1.106 [+1.106, +1.106] | 0.998 |
| stub-str | cl_p10_t0 | - | 100 | +0.107 [+0.107, +0.107] | +0.205 [+0.205, +0.205] | +0.205 [+0.205, +0.205] | +0.100 [+0.100, +0.100] | 1.006 |
