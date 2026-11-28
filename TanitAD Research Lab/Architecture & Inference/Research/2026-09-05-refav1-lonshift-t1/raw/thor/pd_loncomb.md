### paired episode-cluster bootstrap, n = 40 windows / 8 clusters, n_boot 2000
tier: `cl` T1 · floors T1 · `ol` T0 · known-value control PASS

#### ADE
| pair | ade_m | fde_m |
|---|---|---|
| T_loncomb.cl - T_lonshift.cl | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| T_lonshift.cl - ha | -0.1005 [-0.2224, +0.0263] | **-0.4178 [-0.8148, -0.0157]** |
| T_lonshift.cl - ha0 | **-0.1368 [-0.2147, -0.0599]** | **-0.3626 [-0.5431, -0.1797]** |
| T_lonshift.cl - ha0_ext | -0.0889 [-0.2049, +0.0137] | **-0.3706 [-0.7361, -0.0293]** |
| T_loncomb.cl - ha | -0.1005 [-0.2224, +0.0263] | **-0.4178 [-0.8148, -0.0157]** |
| T_loncomb.cl - ha0 | **-0.1368 [-0.2147, -0.0599]** | **-0.3626 [-0.5431, -0.1797]** |
| T_loncomb.cl - ha0_ext | -0.0889 [-0.2049, +0.0137] | **-0.3706 [-0.7361, -0.0293]** |
| T_lonshift.cl - T_lonshift.cl | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

#### longitudinal
| pair | LON_speed_mae_mps | LON_along_mae_m | LON_accel_mae_mps2 |
|---|---|---|---|
| T_loncomb.cl - T_lonshift.cl | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| T_lonshift.cl - ha | **+0.2535 [+0.1162, +0.4227]** | **-0.1735 [-0.3661, -0.0181]** | **+0.1922 [+0.0663, +0.3562]** |
| T_lonshift.cl - ha0 | **-0.1624 [-0.2676, -0.0557]** | **-0.1601 [-0.2326, -0.0829]** | **-0.1357 [-0.2523, -0.0266]** |
| T_lonshift.cl - ha0_ext | **+0.2535 [+0.1162, +0.4227]** | -0.1632 [-0.3535, +0.0044] | **+0.1922 [+0.0663, +0.3562]** |
| T_loncomb.cl - ha | **+0.2535 [+0.1162, +0.4227]** | **-0.1735 [-0.3661, -0.0181]** | **+0.1922 [+0.0663, +0.3562]** |
| T_loncomb.cl - ha0 | **-0.1624 [-0.2676, -0.0557]** | **-0.1601 [-0.2326, -0.0829]** | **-0.1357 [-0.2523, -0.0266]** |
| T_loncomb.cl - ha0_ext | **+0.2535 [+0.1162, +0.4227]** | -0.1632 [-0.3535, +0.0044] | **+0.1922 [+0.0663, +0.3562]** |
| T_lonshift.cl - T_lonshift.cl | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

#### lateral
| pair | LAT_cross_mae_m | LAT_heading_mae_deg | LAT_yaw_rate_mae_radps |
|---|---|---|---|
| T_loncomb.cl - T_lonshift.cl | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] (n=34) | +0.0000 [+0.0000, +0.0000] |
| T_lonshift.cl - ha | -0.0138 [-0.0856, +0.0567] | **-3.4391 [-6.3282, -0.1918]** (n=34) | **-0.0987 [-0.1557, -0.0376]** |
| T_lonshift.cl - ha0 | **+0.0275 [+0.0030, +0.0613]** | +0.2854 [-0.0658, +0.6593] (n=33) | **+0.0073 [+0.0030, +0.0122]** |
| T_lonshift.cl - ha0_ext | -0.0100 [-0.0633, +0.0299] | **-3.5545 [-5.6085, -1.1910]** (n=34) | **-0.0998 [-0.1447, -0.0524]** |
| T_loncomb.cl - ha | -0.0138 [-0.0856, +0.0567] | **-3.4391 [-6.3282, -0.1918]** (n=34) | **-0.0987 [-0.1557, -0.0376]** |
| T_loncomb.cl - ha0 | **+0.0275 [+0.0030, +0.0613]** | +0.2854 [-0.0658, +0.6593] (n=33) | **+0.0073 [+0.0030, +0.0122]** |
| T_loncomb.cl - ha0_ext | -0.0100 [-0.0633, +0.0299] | **-3.5545 [-5.6085, -1.1910]** (n=34) | **-0.0998 [-0.1447, -0.0524]** |
| T_lonshift.cl - T_lonshift.cl | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] (n=34) | +0.0000 [+0.0000, +0.0000] |

#### tactical
| pair | TAC_traj_lat_correct | TAC_traj_lon_correct |
|---|---|---|
| T_loncomb.cl - T_lonshift.cl | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| T_lonshift.cl - ha | -0.2000 [-0.4000, +0.0250] | **-0.2750 [-0.4000, -0.1250]** |
| T_lonshift.cl - ha0 | +0.0000 [+0.0000, +0.0000] | +0.0250 [-0.1250, +0.2000] |
| T_lonshift.cl - ha0_ext | **-0.2500 [-0.4000, -0.0750]** | **-0.2750 [-0.4000, -0.1250]** |
| T_loncomb.cl - ha | -0.2000 [-0.4000, +0.0250] | **-0.2750 [-0.4000, -0.1250]** |
| T_loncomb.cl - ha0 | +0.0000 [+0.0000, +0.0000] | +0.0250 [-0.1250, +0.2000] |
| T_loncomb.cl - ha0_ext | **-0.2500 [-0.4000, -0.0750]** | **-0.2750 [-0.4000, -0.1250]** |
| T_lonshift.cl - T_lonshift.cl | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

#### strategic
**UNAVAILABLE (n = 0).** no route/goal channel exists in this eval surface. The refav1 open-loop grid is evaluated with `--no-navshuf` on the v7.2 EVAL labels and the planner's own strategic input is the tactical head's imagined goal token, which is MODEL OUTPUT, not a route label: scoring the plan against it would score the model against itself. PhysicalAI-AV ships no map, lane graph, junction annotation or route signal (CLAUDE.md, the 6-of-36 read-set table), so no external strategic reference exists either. This is a WORK ITEM (the strategic reference must come from AlpaSim's map.xodr or an external corpus), NOT a pass and NOT an omission.

