# L2 headroom: how much of refcv8's cruise false-brake does the SHIPPED re-timing (RETIME_DEPLOY8) already absorb?

**LABEL (primary, S6 compliant windows): PARTIAL**  |  S6 all windows (the surface H-R8-S6-LON-1 was measured on): PARTIAL

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; EMA-30k, `step30000ema_unc`; evidence class MEASURED (ours), CPU only.
* estimator: paired episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci), B 2000, seed 0, over the episodes present in the masked windows: answers 'another draw of EPISODES' only; NOT another training run (single training seed per arm, H-ESTIM-SEED-1); inference seeds 0 and 1 are shown separately, never pooled.
* pre-registered rule: `raw/refcv9_headroom/PREREG_L2_HEADROOM.md` (sha256 sealed before any re-timed number was computed): ABSORBS = cruise ADE delta vs refcv7 not separated worse at either seed AND false-decel(DEPLOY8) <= 0.5 x false-decel(os) AND missed-decel(DEPLOY8) not separated worse than os; PARTIAL = cruise ADE delta shrinks >= 25 % at both seeds but ABSORBS fails; NONE otherwise; guard: if os shows no separated cruise regression on a surface the label there is NOT-APPLICABLE.
* artifacts: DEPLOY8_POS_step30000ema_unc_s0.npz (sha b85a7bd25d00), DEPLOY8_POS_step30000ema_unc_s1.npz (sha ff7321fbed02); prereg sha f779c3a98713; wall 91.0 s.

## PRIMARY: S6 COMPLIANT windows

n **3251 windows / 138 episodes**; GT-cruise (dvclass, ADE and along-track) **1654 windows / 119 ep**; GT-cruise (plan class, false decel) **1856**; GT-decel (plan class, missed decel) **639**.
Rule label **PARTIAL** (effective label PARTIAL; precondition os regression present: True). Clause a (cruise ADE vs refcv7 not separated worse, per seed): {0: False, 1: False}; clause b (false decel <= 0.5 x os): {0: False, 1: False}; clause c (missed decel not separated worse than os): {0: True, 1: True}; ADE shrink (point): {0: 0.5253631817099309, 1: 0.5377002827521207}.

| seed | arm | cruise ADE delta vs refcv7 [CI] m | cruise signed along-track @6 s (m) | false decel (rate) | missed decel (rate) | false decel on dvclass-cruise (supplement) |
|---|---|---|---|---|---|---|
| 0 | os (refcv8) | +0.420 [+0.205, +0.646] SEP | -2.04 | 0.266 | 0.387 | 0.281 |
| 0 | DEPLOY8 | +0.199 [+0.034, +0.362] SEP | -1.59 | 0.253 | 0.358 | 0.269 |
| 0 | refcv7-50,400 | 0 (reference) | +0.16 | 0.091 | 0.559 | 0.097 |
| 0 | *DEPLOY8 - os (paired)* | -0.221 [-0.307, -0.145] SEP | +0.45 [+0.20, +0.73] SEP | -0.012 [-0.027, +0.001] | -0.028 [-0.052, -0.005] SEP | -0.012 [-0.027, +0.003] |
| 1 | os (refcv8) | +0.424 [+0.203, +0.666] SEP | -2.05 | 0.258 | 0.380 | 0.272 |
| 1 | DEPLOY8 | +0.196 [+0.028, +0.374] SEP | -1.59 | 0.254 | 0.357 | 0.271 |
| 1 | refcv7-50,400 | 0 (reference) | +0.16 | 0.094 | 0.568 | 0.100 |
| 1 | *DEPLOY8 - os (paired)* | -0.228 [-0.319, -0.152] SEP | +0.46 [+0.20, +0.73] SEP | -0.003 [-0.014, +0.007] | -0.024 [-0.046, -0.002] SEP | -0.001 [-0.013, +0.012] |

Absorbed shares (descriptive, not in the rule) and the L2 sizing residual:

| seed | cruise ADE: os D / DEPLOY8 D (m, vs refcv7) | ADE shrink [joint-draw CI] | false-decel excess absorbed | along-track excess absorbed |
|---|---|---|---|---|
| 0 | +0.420 / +0.199 | 53 % [40 %, 81 %] (draws with os D>0: 2000/2000) | 7 % (0.266 -> 0.253, refcv7 0.091) | 21 % (-2.04 -> -1.59, refcv7 +0.16) |
| 1 | +0.424 / +0.196 | 54 % [40 %, 84 %] (draws with os D>0: 2000/2000) | 2 % (0.258 -> 0.254, refcv7 0.094) | 21 % (-2.05 -> -1.59, refcv7 +0.16) |

Descriptive extras (added after the first run; the rule cells are identical between the runs). Cruise windows' contribution on this surface's all-window scale (sum of the cruise windows' ADE delta vs refcv7 / n surface windows) and the cruise mean signed along-track error by instant:

| seed | cruise contribution os / DEPLOY8 (m) | surface total delta vs refcv7 os / DEPLOY8 (m) | along-track by instant 1..6 s: os | DEPLOY8 | refcv7 (m) |
|---|---|---|---|---|---|
| 0 | +0.214 / +0.101 | +0.119 / -0.076 | +0.00 -0.07 -0.27 -0.60 -1.22 -2.04 | -0.03 -0.11 -0.26 -0.51 -0.97 -1.59 | +0.03 +0.04 +0.07 +0.13 +0.14 +0.16 |
| 1 | +0.216 / +0.100 | +0.137 / -0.064 | +0.00 -0.07 -0.27 -0.60 -1.22 -2.05 | -0.03 -0.11 -0.26 -0.52 -0.97 -1.59 | +0.03 +0.03 +0.06 +0.13 +0.13 +0.16 |

## SECONDARY: S6 ALL windows (H-R8-S6-LON-1's surface)

n **3668 windows / 139 episodes**; GT-cruise (dvclass, ADE and along-track) **1739 windows / 128 ep**; GT-cruise (plan class, false decel) **1949**; GT-decel (plan class, missed decel) **832**.
Rule label **PARTIAL** (effective label PARTIAL; precondition os regression present: True). Clause a (cruise ADE vs refcv7 not separated worse, per seed): {0: False, 1: False}; clause b (false decel <= 0.5 x os): {0: False, 1: False}; clause c (missed decel not separated worse than os): {0: True, 1: True}; ADE shrink (point): {0: 0.5254122430539869, 1: 0.5338345864661654}.

| seed | arm | cruise ADE delta vs refcv7 [CI] m | cruise signed along-track @6 s (m) | false decel (rate) | missed decel (rate) | false decel on dvclass-cruise (supplement) |
|---|---|---|---|---|---|---|
| 0 | os (refcv8) | +0.443 [+0.231, +0.671] SEP | -2.21 | 0.274 | 0.345 | 0.288 |
| 0 | DEPLOY8 | +0.210 [+0.047, +0.378] SEP | -1.68 | 0.259 | 0.326 | 0.273 |
| 0 | refcv7-50,400 | 0 (reference) | +0.06 | 0.096 | 0.548 | 0.101 |
| 0 | *DEPLOY8 - os (paired)* | -0.233 [-0.312, -0.161] SEP | +0.53 [+0.27, +0.79] SEP | -0.015 [-0.028, -0.002] SEP | -0.019 [-0.040, -0.001] SEP | -0.015 [-0.030, -0.001] SEP |
| 1 | os (refcv8) | +0.452 [+0.239, +0.682] SEP | -2.19 | 0.266 | 0.341 | 0.279 |
| 1 | DEPLOY8 | +0.211 [+0.043, +0.385] SEP | -1.67 | 0.260 | 0.325 | 0.275 |
| 1 | refcv7-50,400 | 0 (reference) | +0.06 | 0.101 | 0.558 | 0.106 |
| 1 | *DEPLOY8 - os (paired)* | -0.241 [-0.325, -0.168] SEP | +0.52 [+0.27, +0.79] SEP | -0.006 [-0.017, +0.004] | -0.017 [-0.037, +0.000] | -0.004 [-0.016, +0.008] |

Absorbed shares (descriptive, not in the rule) and the L2 sizing residual:

| seed | cruise ADE: os D / DEPLOY8 D (m, vs refcv7) | ADE shrink [joint-draw CI] | false-decel excess absorbed | along-track excess absorbed |
|---|---|---|---|---|
| 0 | +0.443 / +0.210 | 53 % [41 %, 78 %] (draws with os D>0: 2000/2000) | 9 % (0.274 -> 0.259, refcv7 0.096) | 23 % (-2.21 -> -1.68, refcv7 +0.06) |
| 1 | +0.452 / +0.211 | 53 % [42 %, 80 %] (draws with os D>0: 2000/2000) | 4 % (0.266 -> 0.260, refcv7 0.101) | 23 % (-2.19 -> -1.67, refcv7 +0.06) |

Descriptive extras (added after the first run; the rule cells are identical between the runs). Cruise windows' contribution on this surface's all-window scale (sum of the cruise windows' ADE delta vs refcv7 / n surface windows) and the cruise mean signed along-track error by instant:

| seed | cruise contribution os / DEPLOY8 (m) | surface total delta vs refcv7 os / DEPLOY8 (m) | along-track by instant 1..6 s: os | DEPLOY8 | refcv7 (m) |
|---|---|---|---|---|---|
| 0 | +0.210 / +0.100 | +0.166 / -0.052 | +0.00 -0.08 -0.31 -0.67 -1.33 -2.21 | -0.03 -0.12 -0.28 -0.55 -1.03 -1.68 | +0.03 +0.03 +0.04 +0.09 +0.07 +0.06 |
| 1 | +0.214 / +0.100 | +0.189 / -0.037 | -0.00 -0.08 -0.31 -0.66 -1.32 -2.19 | -0.03 -0.12 -0.28 -0.55 -1.03 -1.67 | +0.03 +0.03 +0.04 +0.09 +0.07 +0.06 |

## Controls

* **ALL_PASS: True** (C1 True, C2 True, C3 True, C4 True, C5 True, C6 True, C7 True).
* C1 os reproduces the banked H-R8-S6-LON-1 literals on ALL windows (seed 0: ADE total +0.1658 [+0.0268, +0.3153]; cruise 1,739 / 128 ep, contribution +0.2099, share 1.2657, bin-mean +0.4427; signed along-track -2.2078 vs +0.0602; false decel 0.2745 vs 0.0965, +0.1780 [+0.1228, +0.2404], n 1,949 / 138; missed decel 0.3450 vs 0.5481, -0.2031 [-0.2707, -0.1435], n 832 / 80; seed 1 share 1.1316, total +0.1895): **True** (27 checks).
* C2 alignment (npz ws / eid / os bit-equal the panels, compliant 3,251 / 138, GT equal across seeds, twins == imported S6 definitions): True.
* C3 numpy ADE / along-track equal `components` (< 1e-4): True (max 3.8e-06).
* C4 DEPLOY8 - refcv7 S6 ADE recomputed equals the banked `FORWARD_DEPLOY8` RETIME_DEPLOY8 `NR_minus_refcv7.ade_m` cell field for field (both seeds, compliant and all): True.
* C5 identity arm (DEPLOY8 := os): every cell exactly 0 with a zero-width CI, label NONE: True.
* C6 decision-function arms (answer fixed by construction): identity expected NONE -> compliant: NONE, all: NONE; overbrake expected NONE -> compliant: NONE, all: NONE; oracle expected ABSORBS -> compliant: ABSORBS, all: ABSORBS; oracle_never_brakes expected PARTIAL -> compliant: PARTIAL, all: PARTIAL.
* C7 mutations of the C1 instrument, each must go RED: m1_cruise_thr_1p0: RED (7 checks failed); m2_decel_thr_1p0: RED (14 checks failed); m3_os_shift_x_plus_0p01m: RED (7 checks failed).

## Caveats

* single TRAINING seed per arm: the interval is an episode draw only and is blind to training variance (`H-ESTIM-SEED-1`); the two inference seeds are the replicate for the planner's sampling floor and are not pooled.
* T1 open loop on logged eval139 frames: a statement about plans scored against the logged future, not about driving.
* the speed-label input differs between refcv7 (per-clip continuous `v_max_ms`) and refcv8 (4-class road-type label): every refcv8-vs-refcv7 row crosses that input change.
* the DEPLOY8 table was fit on TRAIN-side data (`retime_trainfit`), not on these eval episodes; this read is a property of THIS stage on THIS checkpoint.
* the instrument defines GT-cruise two ways (dvclass for ADE / along-track, plan class for false decel); both are kept as defined and the false-decel rate on the dvclass mask is shown as a supplement.

