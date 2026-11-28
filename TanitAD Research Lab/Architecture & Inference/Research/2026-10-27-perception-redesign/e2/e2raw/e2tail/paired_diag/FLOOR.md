# E1 replicate floor -- C0 vs C1 (same flags, training seed 0 vs 1), DEV

Tier: **T0 (label level)** (perception only). Estimator: **paired_episode_cluster_bootstrap_stat**, B = 2000, seed 0 -- it answers *another draw of EPISODES*, not another training run.
Windows: 2308 DEV windows / 70 episodes (clips), identical for both arms (paired alignment drops none). Fixed P=R gate for the census rows: 0.26599645614624023 (--gates-json).

## The keep rule (REFCV9_DESIGN 4.3)

Keep a lever iff its primary is separated in the good direction AND |delta| > 2 x |C1 - C0| on that metric AND no other perception primary is separated worse beyond 2 x its floor (REFCV9_DESIGN 4.3; taniteval.ci.e1_floor_read).

`floor` = abs(C1 - C0) is ONE seed pair (n = 1): a single draw of the training noise, not its spread. A floor of exactly 0 is **degenerate** (two runs agreeing to the last digit), and the keep decision then rests on separation alone (`taniteval.ci.e1_floor_read`; `r9_e1_floor.lever_read` calls it).

## The four metric families (CLAUDE.md, binding)

| family | status |
|---|---|
| LONGITUDINAL | not computable: the E1 rig builds no planner (perception-only, T0 label level), n = 0 |
| LATERAL | not computable: the E1 rig builds no planner (perception-only, T0 label level), n = 0 |
| TACTICAL | not computable: the E1 rig builds no planner (perception-only, T0 label level), n = 0 |
| STRATEGIC | PAUSED (PI 2026-09-19, PI_DECISION_QUEUE ITEM 25): the strategic layer is switched off until tactical and operative drive well; reported as paused, not as a pass |

Perception is the family this rig reads: boxes and the near-field map raster, below.

## Perception -- boxes (T0)

### AP@1 m, all and by range (head output, no NMS)

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2308 windows / 70 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `ap1m_0_20` | 0.34502 | 0.34908 | +0.004062 | [-0.0086, 0.0159] | no | 0.004062 | 0.008124 | 2308 windows / 70 episodes |
| `ap1m_20_40` | 0.10737 | 0.10453 | -0.002837 | [-0.0117, 0.0059] | no | 0.002837 | 0.005674 | 2308 windows / 70 episodes |
| `ap1m_20_60` | 0.07815 | 0.07630 | -0.001852 | [-0.0077, 0.0038] | no | 0.001852 | 0.003704 | 2308 windows / 70 episodes |
| `ap1m_40_60` | 0.02735 | 0.02672 | -0.000627 | [-0.0062, 0.0038] | no | 0.000627 | 0.001254 | 2308 windows / 70 episodes |
| `ap1m_all` | 0.19086 | 0.19153 | +0.000670 | [-0.0061, 0.0074] | no | 0.000670 | 0.001341 | 2308 windows / 70 episodes |

### AP@2 m, all and by range (head output, no NMS)

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2308 windows / 70 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `ap2m_0_20` | 0.50204 | 0.50492 | +0.002888 | [-0.0128, 0.0186] | no | 0.002888 | 0.005776 | 2308 windows / 70 episodes |
| `ap2m_20_40` | 0.20697 | 0.21202 | +0.005052 | [-0.0055, 0.0166] | no | 0.005052 | 0.010103 | 2308 windows / 70 episodes |
| `ap2m_20_60` | 0.16474 | 0.16493 | +0.000191 | [-0.0071, 0.0078] | no | 0.000191 | 0.000383 | 2308 windows / 70 episodes |
| `ap2m_40_60` | 0.09314 | 0.08263 | -0.010513 | [-0.0191, -0.0038] | yes | 0.010513 | 0.021026 | 2308 windows / 70 episodes |
| `ap2m_all` | 0.30766 | 0.30762 | -0.000042 | [-0.0077, 0.008] | no | 0.000042 | 0.000083 | 2308 windows / 70 episodes |

### P=R census at the FIXED gate (same gate for both arms)

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2308 windows / 70 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `boxes_per_object_noNMS` | 2.06253 | 2.03192 | -0.030602 | [-0.076, 0.0114] | no | 0.030602 | 0.061204 | 2308 windows / 70 episodes |
| `boxes_per_object_noNMS_0_20` | 2.15767 | 2.16501 | +0.007341 | [-0.046, 0.054] | no | 0.007341 | 0.014681 | 2308 windows / 70 episodes |
| `boxes_per_object_noNMS_20_40` | 1.79693 | 1.64489 | -0.152037 | [-0.2493, -0.0603] | yes | 0.152037 | 0.304074 | 2308 windows / 70 episodes |
| `boxes_per_object_noNMS_40_60` | 1.39286 | 1.00000 | -0.392857 | [-0.4764, 0] | no | 0.392857 | 0.785714 | 2308 windows / 70 episodes |
| `frac_objects_ge2_noNMS` | 0.68737 | 0.66874 | -0.018633 | [-0.0428, 0.0034] | no | 0.018633 | 0.037265 | 2308 windows / 70 episodes |
| `prec_gate_noNMS` | 0.36732 | 0.37163 | +0.004309 | [-0.0067, 0.0153] | no | 0.004309 | 0.008618 | 2308 windows / 70 episodes |
| `rec_gate_noNMS` | 0.32672 | 0.32348 | -0.003239 | [-0.0124, 0.0059] | no | 0.003239 | 0.006478 | 2308 windows / 70 episodes |

### localisation / class of the 2 m greedy TPs

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2308 windows / 70 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `dx_0_40_mean` | 0.79421 | 0.78520 | -0.009003 | [-0.0248, 0.0071] | no | 0.009003 | 0.018005 | 2308 windows / 70 episodes |
| `dx_0_40_median` | 0.73420 | 0.71670 | -0.017498 | [-0.0437, 0.0084] | no | 0.017498 | 0.034995 | 2308 windows / 70 episodes |
| `tp_class_acc` | 0.82438 | 0.82034 | -0.004040 | [-0.0161, 0.0069] | no | 0.004040 | 0.008081 | 2308 windows / 70 episodes |

### other

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2308 windows / 70 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `n_det_noNMS_0_20` | 1833.00000 | 1806.00000 | -27.000000 | [-76, 17] | no | 27.000000 | 54.000000 | 2308 windows / 70 episodes |
| `n_det_noNMS_20_40` | 586.00000 | 597.00000 | +11.000000 | [-44, 59] | no | 11.000000 | 22.000000 | 2308 windows / 70 episodes |
| `n_det_noNMS_40_60` | 28.00000 | 9.00000 | -19.000000 | [-42, -4] | yes | 19.000000 | 38.000000 | 2308 windows / 70 episodes |
| `n_pos` | 7410.00000 | 7410.00000 | +0.000000 | [0, 0] | no | 0.000000 (DEGENERATE: not a floor) | 0.000000 | 2308 windows / 70 episodes |

## Perception -- map raster, corridor |y| < 3 m (T0)

Map flags cut at each arm's OWN FIT-split thresholds (not 128): C0 `28e2385af70b1430529f7d6d5dc286bf6618923b38a43c1efebacb6b66a275a7`, C1 `13f014449219f3a18bf0af775dd736bf281a554c185a7a5cac1c438d0b0d7306` (sha256 of the files).

### edge-class P / R / F @ 0.3 m of {pedge >= the arm's level} vs GT edge (E0d thin-class protocol; BESIDE, never a primary)

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2329 raster files / 68 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `nf_edge03_f_0_5` | 0.17755 | 0.39817 | +0.220628 | [0.1855, 0.2568] | yes | 0.220628 | 0.441256 | 2329 raster files / 68 episodes |
| `nf_edge03_f_10_20` | 0.42797 | 0.46501 | +0.037038 | [0.0186, 0.0546] | yes | 0.037038 | 0.074076 | 2329 raster files / 68 episodes |
| `nf_edge03_f_20_40` | 0.31189 | 0.36398 | +0.052094 | [0.0395, 0.0646] | yes | 0.052094 | 0.104188 | 2329 raster files / 68 episodes |
| `nf_edge03_f_5_10` | 0.42983 | 0.48388 | +0.054046 | [0.0329, 0.0757] | yes | 0.054046 | 0.108091 | 2329 raster files / 68 episodes |
| `nf_edge03_f_5_20` | 0.42850 | 0.47144 | +0.042946 | [0.0255, 0.0602] | yes | 0.042946 | 0.085891 | 2329 raster files / 68 episodes |
| `nf_edge03_prec_0_5` | 0.25976 | 0.43277 | +0.173007 | [0.1348, 0.2129] | yes | 0.173007 | 0.346013 | 2329 raster files / 68 episodes |
| `nf_edge03_prec_10_20` | 0.35238 | 0.48072 | +0.128341 | [0.1096, 0.1472] | yes | 0.128341 | 0.256682 | 2329 raster files / 68 episodes |
| `nf_edge03_prec_20_40` | 0.30483 | 0.35049 | +0.045663 | [0.025, 0.0644] | yes | 0.045663 | 0.091326 | 2329 raster files / 68 episodes |
| `nf_edge03_prec_5_10` | 0.35748 | 0.46004 | +0.102559 | [0.0796, 0.1254] | yes | 0.102559 | 0.205118 | 2329 raster files / 68 episodes |
| `nf_edge03_prec_5_20` | 0.35391 | 0.47298 | +0.119070 | [0.1009, 0.1376] | yes | 0.119070 | 0.238140 | 2329 raster files / 68 episodes |
| `nf_edge03_rec_0_5` | 0.13486 | 0.36870 | +0.233840 | [0.1909, 0.2798] | yes | 0.233840 | 0.467681 | 2329 raster files / 68 episodes |
| `nf_edge03_rec_10_20` | 0.54484 | 0.45029 | -0.094556 | [-0.1181, -0.0708] | yes | 0.094556 | 0.189113 | 2329 raster files / 68 episodes |
| `nf_edge03_rec_20_40` | 0.31928 | 0.37855 | +0.059270 | [0.043, 0.0746] | yes | 0.059270 | 0.118541 | 2329 raster files / 68 episodes |
| `nf_edge03_rec_5_10` | 0.53891 | 0.51033 | -0.028584 | [-0.0541, -0.0021] | yes | 0.028584 | 0.057169 | 2329 raster files / 68 episodes |
| `nf_edge03_rec_5_20` | 0.54290 | 0.46991 | -0.072995 | [-0.096, -0.0494] | yes | 0.072995 | 0.145990 | 2329 raster files / 68 episodes |

### corridor sidewalk precision, within 0.5 m of a GT sidewalk cell

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2329 raster files / 68 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `nf_prec05_0_5` | 0.22215 | 0.70975 | +0.487605 | [0.4071, 0.5672] | yes | 0.487605 | 0.975210 | 2329 raster files / 68 episodes |
| `nf_prec05_10_20` | 0.74817 | 0.72864 | -0.019529 | [-0.0426, 0.0073] | no | 0.019529 | 0.039059 | 2329 raster files / 68 episodes |
| `nf_prec05_20_40` | 0.71673 | 0.75108 | +0.034353 | [-0.0034, 0.0676] | no | 0.034353 | 0.068707 | 2329 raster files / 68 episodes |
| `nf_prec05_5_10` | 0.72111 | 0.74874 | +0.027629 | [-0.0192, 0.0789] | no | 0.027629 | 0.055259 | 2329 raster files / 68 episodes |

### corridor sidewalk precision, exact GT sidewalk cell

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2329 raster files / 68 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `nf_prec_0_10` | 0.23691 | 0.55196 | +0.315053 | [0.2449, 0.3851] | yes | 0.315053 | 0.630105 | 2329 raster files / 68 episodes |
| `nf_prec_0_5` | 0.14533 | 0.52848 | +0.383151 | [0.3085, 0.4598] | yes | 0.383151 | 0.766301 | 2329 raster files / 68 episodes |
| `nf_prec_10_20` | 0.61789 | 0.59310 | -0.024793 | [-0.0468, 0.0032] | no | 0.024793 | 0.049585 | 2329 raster files / 68 episodes |
| `nf_prec_20_40` | 0.62534 | 0.65292 | +0.027585 | [-0.0102, 0.0607] | no | 0.027585 | 0.055170 | 2329 raster files / 68 episodes |
| `nf_prec_5_10` | 0.57141 | 0.57459 | +0.003183 | [-0.044, 0.0575] | no | 0.003183 | 0.006365 | 2329 raster files / 68 episodes |

### corridor sidewalk+edge precision, GT {edge, sidewalk}, GT-seen cells (incl. the camera blind zone)

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2329 raster files / 68 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `nf_swedge_prec_0_5` | 0.16048 | 0.42485 | +0.264370 | [0.2084, 0.3229] | yes | 0.264370 | 0.528739 | 2329 raster files / 68 episodes |
| `nf_swedge_prec_10_20` | 0.48992 | 0.50339 | +0.013471 | [-0.0085, 0.0391] | no | 0.013471 | 0.026941 | 2329 raster files / 68 episodes |
| `nf_swedge_prec_20_40` | 0.55781 | 0.56935 | +0.011540 | [-0.0179, 0.0385] | no | 0.011540 | 0.023080 | 2329 raster files / 68 episodes |
| `nf_swedge_prec_5_10` | 0.41246 | 0.44258 | +0.030117 | [0.0007, 0.0653] | yes | 0.030117 | 0.060234 | 2329 raster files / 68 episodes |
| `nf_swedge_prec_5_20` | 0.46648 | 0.48495 | +0.018466 | [-0.0029, 0.0443] | no | 0.018466 | 0.036932 | 2329 raster files / 68 episodes |

### corridor sidewalk+edge precision, GT {edge, sidewalk}, lift-valid cells (the M1 map primary)

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2329 raster files / 68 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `nf_swedge_prec_lv_0_5` | 0.34085 | 0.40617 | +0.065318 | [0.0159, 0.1196] | yes | 0.065318 | 0.130637 | 2329 raster files / 68 episodes |
| `nf_swedge_prec_lv_10_20` | 0.48992 | 0.50339 | +0.013471 | [-0.0085, 0.0391] | no | 0.013471 | 0.026941 | 2329 raster files / 68 episodes |
| `nf_swedge_prec_lv_20_40` | 0.55781 | 0.56935 | +0.011540 | [-0.0179, 0.0385] | no | 0.011540 | 0.023080 | 2329 raster files / 68 episodes |
| `nf_swedge_prec_lv_5_10` | 0.41246 | 0.44258 | +0.030117 | [0.0007, 0.0653] | yes | 0.030117 | 0.060234 | 2329 raster files / 68 episodes |
| `nf_swedge_prec_lv_5_20` | 0.46648 | 0.48495 | +0.018466 | [-0.0029, 0.0443] | no | 0.018466 | 0.036932 | 2329 raster files / 68 episodes |

## Reading notes

* 25 of 57 defined metrics have C1 vs C0 separated by the episode bootstrap. A separated C1 - C0 is NOT a failure of the rig: it is the proof (H-ESTIM-SEED-1) that a separated interval alone cannot certify a lever -- which is why the keep rule adds the 2 x floor clause.
* 0 defined metrics have floor exactly 0 (degenerate).
* Map rows are what `r9_e1_eval.py` emits: `nf_prec_*` = sidewalk-flag (`psw`) precision on GT-seen cells of the corridor (the historical figure); `nf_swedge_prec_*` = the sidewalk+edge union vs GT {edge, sidewalk}; `nf_swedge_prec_lv_*` (the M1 map primary) = the same on lift-valid cells. Both exist only when `--lift-valid` gave the shared mask.
* The census rows (`prec_gate_*`, `rec_gate_*`, `boxes_per_object_*`, `frac_objects_ge2_*`) are read at the fixed gate in the header for both arms, not at each arm's own P=R point.
* Inputs: eval_c0 sha256 6d367a34c38f; eval_c1 sha256 c5e2197fd34d; paired sha256 38f28aade79c.
