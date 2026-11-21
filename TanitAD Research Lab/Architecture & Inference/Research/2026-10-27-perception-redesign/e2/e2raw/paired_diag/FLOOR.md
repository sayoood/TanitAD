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
| `ap1m_0_20` | 0.34836 | 0.36533 | +0.016971 | [-0.0004, 0.036] | no | 0.016971 | 0.033943 | 2308 windows / 70 episodes |
| `ap1m_20_40` | 0.11036 | 0.10003 | -0.010331 | [-0.0211, -0.0002] | yes | 0.010331 | 0.020662 | 2308 windows / 70 episodes |
| `ap1m_20_60` | 0.07923 | 0.07346 | -0.005770 | [-0.0135, 0.0012] | no | 0.005770 | 0.011540 | 2308 windows / 70 episodes |
| `ap1m_40_60` | 0.02639 | 0.02723 | +0.000839 | [-0.0047, 0.0051] | no | 0.000839 | 0.001678 | 2308 windows / 70 episodes |
| `ap1m_all` | 0.19282 | 0.19771 | +0.004887 | [-0.0042, 0.0148] | no | 0.004887 | 0.009773 | 2308 windows / 70 episodes |

### AP@2 m, all and by range (head output, no NMS)

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2308 windows / 70 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `ap2m_0_20` | 0.51062 | 0.52385 | +0.013229 | [-0.0045, 0.0303] | no | 0.013229 | 0.026458 | 2308 windows / 70 episodes |
| `ap2m_20_40` | 0.21208 | 0.21954 | +0.007467 | [-0.0047, 0.022] | no | 0.007467 | 0.014934 | 2308 windows / 70 episodes |
| `ap2m_20_60` | 0.16774 | 0.17124 | +0.003501 | [-0.0051, 0.0139] | no | 0.003501 | 0.007002 | 2308 windows / 70 episodes |
| `ap2m_40_60` | 0.09030 | 0.08534 | -0.004961 | [-0.0133, 0.002] | no | 0.004961 | 0.009921 | 2308 windows / 70 episodes |
| `ap2m_all` | 0.31362 | 0.32062 | +0.007006 | [-0.0021, 0.0163] | no | 0.007006 | 0.014013 | 2308 windows / 70 episodes |

### P=R census at the FIXED gate (same gate for both arms)

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2308 windows / 70 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `boxes_per_object_noNMS` | 2.03053 | 1.96922 | -0.061310 | [-0.1066, -0.0172] | yes | 0.061310 | 0.122619 | 2308 windows / 70 episodes |
| `boxes_per_object_noNMS_0_20` | 2.12655 | 2.08499 | -0.041556 | [-0.0849, 0.0006] | no | 0.041556 | 0.083113 | 2308 windows / 70 episodes |
| `boxes_per_object_noNMS_20_40` | 1.79573 | 1.67023 | -0.125503 | [-0.2468, -0.02] | yes | 0.125503 | 0.251005 | 2308 windows / 70 episodes |
| `boxes_per_object_noNMS_40_60` | 1.45238 | 1.00000 | -0.452381 | [-0.6667, -0.1106] | yes | 0.452381 | 0.904762 | 2308 windows / 70 episodes |
| `frac_objects_ge2_noNMS` | 0.66654 | 0.64088 | -0.025652 | [-0.0511, 0.001] | no | 0.025652 | 0.051304 | 2308 windows / 70 episodes |
| `prec_gate_noNMS` | 0.36358 | 0.37528 | +0.011696 | [-0.002, 0.0254] | no | 0.011696 | 0.023391 | 2308 windows / 70 episodes |
| `rec_gate_noNMS` | 0.33954 | 0.33968 | +0.000135 | [-0.0127, 0.0134] | no | 0.000135 | 0.000270 | 2308 windows / 70 episodes |

### localisation / class of the 2 m greedy TPs

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2308 windows / 70 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `dx_0_40_mean` | 0.78914 | 0.78577 | -0.003373 | [-0.0212, 0.0141] | no | 0.003373 | 0.006747 | 2308 windows / 70 episodes |
| `dx_0_40_median` | 0.71639 | 0.70995 | -0.006444 | [-0.0416, 0.0228] | no | 0.006444 | 0.012888 | 2308 windows / 70 episodes |
| `tp_class_acc` | 0.81420 | 0.81864 | +0.004439 | [-0.0047, 0.013] | no | 0.004439 | 0.008878 | 2308 windows / 70 episodes |

### other

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2308 windows / 70 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `n_det_noNMS_0_20` | 1857.00000 | 1859.00000 | +2.000000 | [-52, 57] | no | 2.000000 | 4.000000 | 2308 windows / 70 episodes |
| `n_det_noNMS_20_40` | 656.00000 | 655.00000 | -1.000000 | [-72, 62] | no | 1.000000 | 2.000000 | 2308 windows / 70 episodes |
| `n_det_noNMS_40_60` | 42.00000 | 20.00000 | -22.000000 | [-46, -5] | yes | 22.000000 | 44.000000 | 2308 windows / 70 episodes |
| `n_pos` | 7410.00000 | 7410.00000 | +0.000000 | [0, 0] | no | 0.000000 (DEGENERATE: not a floor) | 0.000000 | 2308 windows / 70 episodes |

## Perception -- map raster, corridor |y| < 3 m (T0)

Map flags cut at each arm's OWN FIT-split thresholds (not 128): C0 `522b263d5ec88ef41ac494694e5f3c10dc24b1d98290074ce12a992d73cde856`, C1 `8f96d2c6b386f2264b49c67e5115e808a06a026b234ffc7a71093318b8081eed` (sha256 of the files).

### edge-class P / R / F @ 0.3 m of {pedge >= the arm's level} vs GT edge (E0d thin-class protocol; BESIDE, never a primary)

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2329 raster files / 68 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `nf_edge03_f_0_5` | 0.16933 | 0.39973 | +0.230400 | [0.1954, 0.2671] | yes | 0.230400 | 0.460801 | 2329 raster files / 68 episodes |
| `nf_edge03_f_10_20` | 0.43428 | 0.46739 | +0.033115 | [0.0156, 0.05] | yes | 0.033115 | 0.066230 | 2329 raster files / 68 episodes |
| `nf_edge03_f_20_40` | 0.31096 | 0.36532 | +0.054357 | [0.0411, 0.0674] | yes | 0.054357 | 0.108715 | 2329 raster files / 68 episodes |
| `nf_edge03_f_5_10` | 0.43160 | 0.48583 | +0.054231 | [0.0337, 0.0753] | yes | 0.054231 | 0.108462 | 2329 raster files / 68 episodes |
| `nf_edge03_f_5_20` | 0.43348 | 0.47368 | +0.040197 | [0.0232, 0.0568] | yes | 0.040197 | 0.080395 | 2329 raster files / 68 episodes |
| `nf_edge03_prec_0_5` | 0.26788 | 0.43178 | +0.163904 | [0.1244, 0.2049] | yes | 0.163904 | 0.327809 | 2329 raster files / 68 episodes |
| `nf_edge03_prec_10_20` | 0.36926 | 0.48879 | +0.119526 | [0.1019, 0.1368] | yes | 0.119526 | 0.239053 | 2329 raster files / 68 episodes |
| `nf_edge03_prec_20_40` | 0.31359 | 0.35326 | +0.039673 | [0.0186, 0.0587] | yes | 0.039673 | 0.079345 | 2329 raster files / 68 episodes |
| `nf_edge03_prec_5_10` | 0.36176 | 0.46346 | +0.101699 | [0.0795, 0.1236] | yes | 0.101699 | 0.203397 | 2329 raster files / 68 episodes |
| `nf_edge03_prec_5_20` | 0.36689 | 0.47922 | +0.112327 | [0.095, 0.1295] | yes | 0.112327 | 0.224654 | 2329 raster files / 68 episodes |
| `nf_edge03_rec_0_5` | 0.12379 | 0.37211 | +0.248318 | [0.2044, 0.2957] | yes | 0.248318 | 0.496635 | 2329 raster files / 68 episodes |
| `nf_edge03_rec_10_20` | 0.52708 | 0.44779 | -0.079287 | [-0.1016, -0.0561] | yes | 0.079287 | 0.158574 | 2329 raster files / 68 episodes |
| `nf_edge03_rec_20_40` | 0.30838 | 0.37823 | +0.069851 | [0.0526, 0.087] | yes | 0.069851 | 0.139701 | 2329 raster files / 68 episodes |
| `nf_edge03_rec_5_10` | 0.53485 | 0.51047 | -0.024384 | [-0.0505, 0.0027] | no | 0.024384 | 0.048767 | 2329 raster files / 68 episodes |
| `nf_edge03_rec_5_20` | 0.52962 | 0.46827 | -0.061343 | [-0.0836, -0.0382] | yes | 0.061343 | 0.122686 | 2329 raster files / 68 episodes |

### corridor sidewalk precision, within 0.5 m of a GT sidewalk cell

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2329 raster files / 68 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `nf_prec05_0_5` | 0.22670 | 0.71468 | +0.487986 | [0.4075, 0.5671] | yes | 0.487986 | 0.975971 | 2329 raster files / 68 episodes |
| `nf_prec05_10_20` | 0.75169 | 0.72374 | -0.027950 | [-0.0531, 0.0028] | no | 0.027950 | 0.055899 | 2329 raster files / 68 episodes |
| `nf_prec05_20_40` | 0.72516 | 0.74434 | +0.019176 | [-0.0204, 0.053] | no | 0.019176 | 0.038351 | 2329 raster files / 68 episodes |
| `nf_prec05_5_10` | 0.71528 | 0.74981 | +0.034525 | [-0.0126, 0.089] | no | 0.034525 | 0.069051 | 2329 raster files / 68 episodes |

### corridor sidewalk precision, exact GT sidewalk cell

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2329 raster files / 68 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `nf_prec_0_10` | 0.24113 | 0.56011 | +0.318979 | [0.2477, 0.3904] | yes | 0.318979 | 0.637959 | 2329 raster files / 68 episodes |
| `nf_prec_0_5` | 0.14845 | 0.53768 | +0.389231 | [0.313, 0.4684] | yes | 0.389231 | 0.778463 | 2329 raster files / 68 episodes |
| `nf_prec_10_20` | 0.62322 | 0.59453 | -0.028693 | [-0.0538, 0.003] | no | 0.028693 | 0.057385 | 2329 raster files / 68 episodes |
| `nf_prec_20_40` | 0.63231 | 0.64890 | +0.016589 | [-0.0212, 0.0492] | no | 0.016589 | 0.033179 | 2329 raster files / 68 episodes |
| `nf_prec_5_10` | 0.56777 | 0.58144 | +0.013665 | [-0.0345, 0.069] | no | 0.013665 | 0.027330 | 2329 raster files / 68 episodes |

### corridor sidewalk+edge precision, GT {edge, sidewalk}, GT-seen cells (incl. the camera blind zone)

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2329 raster files / 68 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `nf_swedge_prec_0_5` | 0.16332 | 0.42942 | +0.266104 | [0.2087, 0.3253] | yes | 0.266104 | 0.532209 | 2329 raster files / 68 episodes |
| `nf_swedge_prec_10_20` | 0.49799 | 0.50841 | +0.010418 | [-0.0136, 0.0382] | no | 0.010418 | 0.020837 | 2329 raster files / 68 episodes |
| `nf_swedge_prec_20_40` | 0.56358 | 0.57219 | +0.008613 | [-0.0217, 0.0359] | no | 0.008613 | 0.017225 | 2329 raster files / 68 episodes |
| `nf_swedge_prec_5_10` | 0.41600 | 0.45077 | +0.034777 | [0.0049, 0.0707] | yes | 0.034777 | 0.069553 | 2329 raster files / 68 episodes |
| `nf_swedge_prec_5_20` | 0.47293 | 0.49117 | +0.018243 | [-0.0044, 0.045] | no | 0.018243 | 0.036486 | 2329 raster files / 68 episodes |

### corridor sidewalk+edge precision, GT {edge, sidewalk}, lift-valid cells (the M1 map primary)

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2329 raster files / 68 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `nf_swedge_prec_lv_0_5` | 0.34015 | 0.41288 | +0.072725 | [0.0218, 0.1258] | yes | 0.072725 | 0.145450 | 2329 raster files / 68 episodes |
| `nf_swedge_prec_lv_10_20` | 0.49799 | 0.50841 | +0.010418 | [-0.0136, 0.0382] | no | 0.010418 | 0.020837 | 2329 raster files / 68 episodes |
| `nf_swedge_prec_lv_20_40` | 0.56358 | 0.57219 | +0.008613 | [-0.0217, 0.0359] | no | 0.008613 | 0.017225 | 2329 raster files / 68 episodes |
| `nf_swedge_prec_lv_5_10` | 0.41600 | 0.45077 | +0.034777 | [0.0049, 0.0707] | yes | 0.034777 | 0.069553 | 2329 raster files / 68 episodes |
| `nf_swedge_prec_lv_5_20` | 0.47293 | 0.49117 | +0.018243 | [-0.0044, 0.045] | no | 0.018243 | 0.036486 | 2329 raster files / 68 episodes |

## Reading notes

* 26 of 57 defined metrics have C1 vs C0 separated by the episode bootstrap. A separated C1 - C0 is NOT a failure of the rig: it is the proof (H-ESTIM-SEED-1) that a separated interval alone cannot certify a lever -- which is why the keep rule adds the 2 x floor clause.
* 0 defined metrics have floor exactly 0 (degenerate).
* Map rows are what `r9_e1_eval.py` emits: `nf_prec_*` = sidewalk-flag (`psw`) precision on GT-seen cells of the corridor (the historical figure); `nf_swedge_prec_*` = the sidewalk+edge union vs GT {edge, sidewalk}; `nf_swedge_prec_lv_*` (the M1 map primary) = the same on lift-valid cells. Both exist only when `--lift-valid` gave the shared mask.
* The census rows (`prec_gate_*`, `rec_gate_*`, `boxes_per_object_*`, `frac_objects_ge2_*`) are read at the fixed gate in the header for both arms, not at each arm's own P=R point.
* Inputs: eval_c0 sha256 50ac4365eef6; eval_c1 sha256 eced5167c5e4; paired sha256 bfec1b4fdf81.
