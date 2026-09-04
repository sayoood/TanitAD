# refcv3 @ 40,284 — STRATIFIED open-loop re-analysis: is the hold-action loss a straight-road artefact?

**tag** `refcv3-40284-stratified` · **tier** T1 · **loop class** ⛔ **OPEN LOOP** (PI ruling 2026-09-02, every arm
without exception) · **generated** 2026-09-04T05:13:58Z · **GPU used: NONE** — 35.6 s
of CPU re-analysis of the banked dump · **tool** `taniteval/tools/stratified_openloop.py`

---

## THE ANSWER — in one line

> ⛔ **NO. The loss is NOT a straight-road artefact. It is the OPPOSITE: the pooled
> +0.1423 m UNDERSTATES how badly refcv3 loses to hold-action, because the corpus is
> 67.2% straight-and-constant-speed windows where the model loses LEAST.**
>
> On the 1,583 manoeuvre windows the loss is
> +0.2530 m [+0.2007, +0.3058] —
> **2.87×** the
> +0.0882 m loss on the 3,240 straight/constant windows.
> Both are separated, and both stay separated at the Bonferroni-adjusted 97.5 % level.

**This is pre-registered outcome [B]: the published headline is CONFIRMED and STRENGTHENED, not
weakened.** The mirror check was committed in advance and this is the branch it landed on. The
BEV-Planner mechanism's *precondition* holds on our corpus — it really is dominated by straight,
constant-speed driving — but its *consequence* does not: our straight-policy control does not win
by being flattered by the mix; it wins everywhere, and by MORE where the driving is hard.

⚠️ **The headline needs no correction of direction — it needs an ADDED sentence, because as written
it is the most flattering number in the table.** The exact replacement text is in §9.

---

## 0. What is being re-analysed, and the evidence class of every number here

| | |
|---|---|
| **source** | the banked 141-episode window dump `taniteval/results/refcv3-40284-openloop-dump.tar.gz`, md5 `aff9bad5aff3869c261903d8ff1768e9` |
| **published record it must reproduce** | `taniteval/results/refcv3-40284-openloop.ARM.json` (md5 `0de8e8a4ece162332bc3a387fd5d679c`) / `…-openloop.json` (md5 `5cfe3258c18871218bd85d691904eb20`) |
| **second step, used as a control** | `refcv3-30k-openloop-20260903-2004-dump.tar.gz` (md5 `00dd5985257220be094d437d382319a7`) |
| **evidence class** | **MEASURED (ours)** for every number in this document, from the artifacts above. The two BEV-Planner figures in §1 are **PUBLISHED (primary, banked)** — library key `2312.03031`, sha256 `c289b890…7899` — and are never used as ours. |
| **n** | 4,823 windows / 141 episodes, grid `2s` (K = 4, dt = 0.5 s) |
| **estimator** | `taniteval.ci.paired_episode_cluster_bootstrap`, cluster = **episode**, n_boot 2000, seed 0, point = **full_set pooled mean**. ⛔ `overlapping_holdout_se` is not used anywhere. |
| **parity** | ⛔ **NON-PARITY** — the run's own `config.json` carries `v2_parity.parity false`, `checked false`, `corpus_key null`. |
| **nav** | ⛔ **ORACLE-derived** — the v7.2 `nav_command` token, provenance ego-future. It will not exist at deployment. |
| **selection** | `out["traj"]` / `sel_score_v3`. ⛔ never `a_star`. |

### The stratifier, declared before any number was computed

`four_families.maneuver_kinematics(GT, dt)` → `refc_tactical.factor_from_kinematics(…, kappa=None)`
— the **v1 gate**, applied to the **ground-truth path `g`** in the dump. This is the identical call
`refav1_arm._components` already makes for the published `TAC_traj_*_correct` metrics; nothing
statistical is re-derived here. This module contributes exactly one thing: a **partition** of the
windows, plus the arithmetic that decomposes a pooled mean over it.

**Why it is admissible** — three independent reasons, each of which alone would be enough:

1. It is **label material**, computed from the ego's own future path. Admissible by the PI ruling of
   2026-08-03 (*labels may use ego; **inference** is vision-only*). It never reaches a model.
2. It is **not computed from any arm's output**, so the arm being graded does not choose the strata
   that grade it — the admissibility test `taniteval/p7_strata.py` states for a stratifier.
3. Its thresholds are **horizon-matched**: `LABEL_HORIZON = 20` steps @ 10 Hz = **2.0 s**, which is
   *exactly* this grid's horizon. A threshold quoted outside its horizon is the `step_s` scope error
   in another costume, and this one is not.

Thresholds, from `stack/tanitad/refs/refc_tactical.py`:
`YAW_TURN_RAD 0.15` rad ·
`DV_ACCEL_MS +1.0` /
`DV_BRAKE_MS -1.0` m/s ·
`STOP_V_MS 0.3` · `MOVING_V_MS 1.0`.

---

## 1. The corpus mix — MEASURED, ours, first and independently

⛔ **BEV-Planner's numbers are about nuScenes and are NOT imported as ours.** Verified from the
banked primary PDF, the paper carries **two** figures, at two different scopes, and only the first
is the one the v7 stream quoted: *"our analysis shows that 73.9 % of the nuScenes data involve
scenarios of driving straightforwardly"* (§3, dataset analysis) and, separately, *"straight-driving
scenarios (87 % of all evaluation samples)"* (§4.3, the camera-ablation discussion). Ours follows.

**Lateral (GT, v1 gate) — n = 4,823 windows / 141 episodes**

| class | n windows | fraction | n episodes |
|---|---:|---:|---:|
| `lane_keep` | 4,176 | **86.59%** | 141 |
| `turn_left` | 251 | **5.20%** | 35 |
| `turn_right` | 396 | **8.21%** | 41 |

**Longitudinal (GT, v1 gate)**

| class | n windows | fraction | n episodes |
|---|---:|---:|---:|
| `brake_stop` | 631 | **13.08%** | 78 |
| `steady` | 3,654 | **75.76%** | 141 |
| `accelerate` | 538 | **11.15%** | 72 |

**The cross — the registered primary contrast**

| stratum | n windows | fraction | n episodes |
|---|---:|---:|---:|
| **`lane_keep` AND `steady`** (straight + constant speed) | 3,240 | **67.18%** | 139 |
| **everything else** (a manoeuvre on at least one axis) | 1,583 | **32.82%** | 103 |

**The full 3 × 3**

| | `brake_stop` | `steady` | `accelerate` |
|---|---:|---:|---:|
| **`lane_keep`** | 511 (10.60%) | 3,240 (67.18%) | 425 (8.81%) |
| **`turn_left`** | 27 (0.56%) | 184 (3.82%) | 40 (0.83%) |
| **`turn_right`** | 93 (1.93%) | 230 (4.77%) | 73 (1.51%) |

⇒ **The precondition of the BEV-Planner mechanism HOLDS on our corpus.** Two thirds of the scored
windows (67.18%) are straight *and* constant-speed;
86.59% are laterally `lane_keep` and
75.76% are longitudinally `steady`. A control that drives
straight at the current speed is therefore doing the *right thing* on most of the corpus by
construction. That is precisely why the pooled mean had to be split — and why the split does not
say what the mechanism predicts.

**Sensitivity — the v2 curvature gate** (a gentle highway curve stays `lane_keep` instead of being
called a turn): lateral becomes `lane_keep` 89.09% /
`turn_left` 3.96% / `turn_right` 6.95%,
and straight+constant rises to **69.46%**. The mix conclusion is
gate-insensitive; the corpus is straight-dominated under either rule.

---

## 2. `os − ha` per stratum — the table the headline was missing

**ADE (mean L2 over the 4 grid slots), metres. Positive = `os` WORSE than the hold-action control.**
Direction `os - ha` · paired episode-cluster bootstrap · 95 % percentile intervals.

| stratum | n win | n ep | `os` | `ha` | `os − ha` | verdict |
|---|---:|---:|---:|---:|---|---|
| **CROSS — straight + constant speed** | 3,240 | 139 | 0.3093 | 0.2211 | **+0.0882** [+0.0701, +0.1062] | os WORSE |
| **CROSS — manoeuvre (everything else)** | 1,583 | 103 | 0.7134 | 0.4604 | **+0.2530** [+0.2007, +0.3058] | os WORSE |
| | | | | | | |
| LATERAL — `lane_keep` | 4,176 | 141 | 0.4107 | 0.2577 | **+0.1530** [+0.1296, +0.1766] | os WORSE |
| LATERAL — `turn_left` | 251 | 35 | 0.6520 | 0.6373 | **+0.0147** [-0.0824, +0.1134] | NOT SEPARATED |
| LATERAL — `turn_right` | 396 | 41 | 0.6381 | 0.5280 | **+0.1101** [+0.0150, +0.2102] | os WORSE |
| | | | | | | |
| LONGITUDINAL — `steady` | 3,654 | 141 | 0.3330 | 0.2537 | **+0.0793** [+0.0611, +0.0979] | os WORSE |
| LONGITUDINAL — `brake_stop` | 631 | 78 | 0.8699 | 0.3769 | **+0.4930** [+0.4237, +0.5575] | os WORSE |
| LONGITUDINAL — `accelerate` | 538 | 72 | 0.6799 | 0.5210 | **+0.1589** [+0.0869, +0.2339] | os WORSE |

**And the full 3 × 3, where the two axes interact**

| cell | n win | n ep | `os` | `ha` | `os − ha` | verdict |
|---|---:|---:|---:|---:|---|---|
| `lane_keep` + `brake_stop` | 511 | 75 | 0.8406 | 0.3427 | **+0.4979** [+0.4253, +0.5743] | os WORSE |
| `lane_keep` + `steady` | 3,240 | 139 | 0.3093 | 0.2211 | **+0.0882** [+0.0701, +0.1062] | os WORSE |
| `lane_keep` + `accelerate` | 425 | 64 | 0.6668 | 0.4344 | **+0.2325** [+0.1565, +0.3085] | os WORSE |
| `turn_left` + `brake_stop` | 27 | 8 | 0.9941 | 0.5096 | **+0.4846** [+0.1268, +0.6347] | os WORSE ⚠️ |
| `turn_left` + `steady` | 184 | 31 | 0.5536 | 0.6065 | **-0.0528** [-0.1472, +0.0332] | NOT SEPARATED |
| `turn_left` + `accelerate` | 40 | 12 | 0.8736 | 0.8653 | **+0.0083** [-0.1176, +0.0943] | UNDERPOWERED ⚠️ |
| `turn_right` + `brake_stop` | 93 | 28 | 0.9950 | 0.5261 | **+0.4689** [+0.2596, +0.6412] | os WORSE ⚠️ |
| `turn_right` + `steady` | 230 | 36 | 0.4901 | 0.4308 | **+0.0593** [-0.0214, +0.1333] | NOT SEPARATED |
| `turn_right` + `accelerate` | 73 | 19 | 0.6498 | 0.8370 | **-0.1872** [-0.3604, +0.0010] | UNDERPOWERED ⚠️ |

**The registered primary contrast at the Bonferroni-adjusted level** (2 cells ⇒ α = 0.05/2 = 0.025,
i.e. a 97.5 % interval):

| stratum | n win | n ep | `os − ha` @ 97.5 % | separated |
|---|---:|---:|---|---|
| straight + constant | 3,240 | 139 | **+0.0882** [+0.0672, +0.1088] | **True** |
| manoeuvre | 1,583 | 103 | **+0.2530** [+0.1932, +0.3142] | **True** |

**Both survive the adjustment.** The primary contrast is not a multiple-comparisons artefact.

### What the table says

1. ⛔ **The single worst stratum is `brake_stop`**: `os` **0.8699 m** against
   `ha` **0.3769 m** — the model is
   **2.31×** worse than
   holding the closing action, **+0.4930** [+0.4237, +0.5575]. On 13.08%
   of the windows. This is the programme's known longitudinal blindness, localised.
2. **The loss is real but small on straight/constant road** (**+0.0882** [+0.0701, +0.1062]), and
   **2.87× larger** where the
   ego actually does something.
3. **The sign never reverses WITH A SEPARATED INTERVAL, in any cell.** Three cells are *not
   separated* and all three are lateral: `turn_left` (**+0.0147** [-0.0824, +0.1134], n = 251),
   `turn_left`+`steady` (**-0.0528** [-0.1472, +0.0332], n = 184),
   `turn_right`+`steady` (**+0.0593** [-0.0214, +0.1333], n = 230).
   **Not separated is not a win** — see §10.
4. ⚠️ **One cell hints the other way and is UNDERPOWERED**: `turn_right`+`accelerate`
   **-0.1872** [-0.3604, +0.0010] on n = 73 windows /
   19 episodes. The upper bound is
   +0.0010 — it *nearly* excludes zero in the model's favour.
   ⛔ **This is a POWER LIMIT, not a finding** (RETRACTION_LOG #17). It is the one cell worth
   powering deliberately, and it is a candidate, not a claim.

---

## 3. The decomposition of the pooled +0.1423 m

The mean is linear, so `Σ_s (n_s/N) · δ_s` reconstructs the pooled delta **exactly** — measured
residual 5.55e-17 (tolerance 1e-9, C3 PASS on all four
groupings). Every share below is therefore an identity, not a model.

**By the primary cross** (pooled +0.1423 m):

| stratum | n | weight | δ | contribution (m) | share of pooled |
|---|---:|---:|---:|---:|---:|
| straight_const | 3,240 | 0.6718 | +0.0882 | **+0.0593** | **+41.65 %** |
| manoeuvre | 1,583 | 0.3282 | +0.2530 | **+0.0830** | **+58.35 %** |

**By longitudinal class:**

| stratum | n | weight | δ | contribution (m) | share of pooled |
|---|---:|---:|---:|---:|---:|
| `brake_stop` | 631 | 0.1308 | +0.4930 | **+0.0645** | **+45.32 %** |
| `steady` | 3,654 | 0.7576 | +0.0793 | **+0.0601** | **+42.22 %** |
| `accelerate` | 538 | 0.1115 | +0.1589 | **+0.0177** | **+12.45 %** |

**By lateral class:**

| stratum | n | weight | δ | contribution (m) | share of pooled |
|---|---:|---:|---:|---:|---:|
| `lane_keep` | 4,176 | 0.8659 | +0.1530 | **+0.1325** | **+93.11 %** |
| `turn_left` | 251 | 0.0520 | +0.0147 | **+0.0008** | **+0.54 %** |
| `turn_right` | 396 | 0.0821 | +0.1101 | **+0.0090** | **+6.35 %** |

### The answer to "how much of the pooled +0.1423 is the straight/constant stratum?"

**41.65 %** (+0.0593 m of +0.1423 m)
— from **67.18%** of the windows. The manoeuvre stratum supplies the other
**58.35 %** from only **32.82%** of the windows.

⇒ **The straight stratum DOES NOT dominate the mean, and the sign does not flip anywhere it is
resolvable. Outcome [A] is refuted; outcome [B] is what the data say.**

The sharper decomposition is longitudinal: `brake_stop` + `accelerate` together are
**24.24%** of the windows and supply
**57.77 %** of the pooled loss;
`steady` is **75.76%** of the windows and supplies **42.22 %**.
On a per-window basis the non-steady classes are
**4.28×**
as damaging as `steady`.

---

## 4. The mirror check, stated as plainly as the alternative would have been

The pre-registration committed both readings before anything was computed. **The data landed on
[B].** For completeness, here is what [B] does and does not license:

* ✅ **The published direction is correct in every stratum where it is resolvable.** In **no** cell,
  at any n, does `os` beat `ha` with a separated interval.
* ✅ **The published magnitude is CONSERVATIVE.** The pooled +0.1423 m is a *mix-weighted average of
  a small loss on easy road and a large loss on hard road*. Quoting it alone understates the
  manoeuvre deficit by **1.78×**.
* ❌ **It does not license "uniform".** Three lateral cells are not separated and two are
  underpowered, so *"refcv3 loses to hold-action on every kind of window"* is **not** what was
  measured. The measured statement is: *the loss is present and separated on every longitudinal
  stratum and on the pooled lateral majority, and it is unresolved on turns.*

---

## 5. The four families, per stratum (binding — Sayed 2026-08-02; ADE alone is incomplete)

Direction `os − ha` throughout. `*` = the 95 % paired interval excludes zero. Positive = `os` worse,
**except** the two `TAC_traj_*_correct` agreement rates, where positive = `os` **better**.

| stratum | n | LON speed MAE | LON along MAE | LON accel MAE |
|---|---:|---|---|---|
| `CROSS/straight_const` | 3,240 | +0.1255\* [+0.1057, +0.1450] | +0.0964\* [+0.0776, +0.1155] | +0.3135\* [+0.2831, +0.3428] |
| `CROSS/manoeuvre` | 1,583 | +0.3452\* [+0.3000, +0.3916] | +0.3153\* [+0.2675, +0.3649] | +0.4674\* [+0.4192, +0.5138] |
| `LON/steady` | 3,654 | +0.1240\* [+0.1051, +0.1427] | +0.0968\* [+0.0781, +0.1160] | +0.3108\* [+0.2831, +0.3370] |
| `LON/brake_stop` | 631 | +0.5325\* [+0.4578, +0.6027] | +0.5180\* [+0.4472, +0.5856] | +0.5969\* [+0.5281, +0.6651] |
| `LON/accelerate` | 538 | +0.3048\* [+0.2353, +0.3755] | +0.2429\* [+0.1776, +0.3097] | +0.4521\* [+0.3848, +0.5192] |
| `LAT/lane_keep` | 4,176 | +0.1943\* [+0.1693, +0.2186] | +0.1681\* [+0.1430, +0.1933] | +0.3608\* [+0.3331, +0.3864] |
| `LAT/turn_left` | 251 | +0.1869\* [+0.1096, +0.2700] | +0.1051 [-0.0025, +0.2103] | +0.2796\* [+0.1972, +0.3596] |
| `LAT/turn_right` | 396 | +0.2388\* [+0.1651, +0.3261] | +0.2092\* [+0.1419, +0.2863] | +0.4515\* [+0.3529, +0.5501] |

| stratum | n | LAT cross MAE | LAT heading MAE (deg) | LAT yaw-rate MAE |
|---|---:|---|---|---|
| `CROSS/straight_const` | 3,240 | +0.0028 [-0.0074, +0.0137] | +0.0354 [-0.0323, +0.1058] | +0.2410\* [+0.1394, +0.3717] |
| `CROSS/manoeuvre` | 1,583 | -0.0488\* [-0.0779, -0.0174] | -0.4608\* [-0.7323, -0.1911] | +0.0701\* [+0.0355, +0.1101] |
| `LON/steady` | 3,654 | -0.0064 [-0.0194, +0.0069] | -0.0724 [-0.1643, +0.0249] | +0.2150\* [+0.1255, +0.3263] |
| `LON/brake_stop` | 631 | +0.0130 [-0.0148, +0.0390] | +0.1928 [-0.0120, +0.3785] | +0.0451\* [+0.0282, +0.0645] |
| `LON/accelerate` | 538 | -0.0986\* [-0.1466, -0.0564] | -0.9425\* [-1.4645, -0.4804] | +0.1448\* [+0.0564, +0.2416] |
| `LAT/lane_keep` | 4,176 | -0.0039 [-0.0140, +0.0062] | +0.0349 [-0.0343, +0.1101] | +0.2122\* [+0.1326, +0.3115] |
| `LAT/turn_left` | 251 | -0.0838\* [-0.1481, -0.0329] | -1.3793\* [-2.2168, -0.7092] | +0.0317 [-0.0181, +0.0836] |
| `LAT/turn_right` | 396 | -0.0779 [-0.1580, +0.0048] | -1.0281\* [-1.7276, -0.3725] | -0.0062 [-0.0257, +0.0165] |

| stratum | n | TAC lat agreement | TAC lon agreement |
|---|---:|---|---|
| `CROSS/straight_const` | 3,240 | +0.0028 [-0.0118, +0.0144] | -0.0373\* [-0.0617, -0.0132] |
| `CROSS/manoeuvre` | 1,583 | +0.0423\* [+0.0198, +0.0640] | -0.2179\* [-0.2685, -0.1695] |
| `LON/steady` | 3,654 | +0.0123 [-0.0033, +0.0257] | -0.0285\* [-0.0521, -0.0041] |
| `LON/brake_stop` | 631 | +0.0095 [-0.0259, +0.0447] | -0.3138\* [-0.3886, -0.2392] |
| `LON/accelerate` | 538 | +0.0465\* [+0.0190, +0.0743] | -0.3048\* [-0.3893, -0.2206] |
| `LAT/lane_keep` | 4,176 | +0.0029 [-0.0103, +0.0135] | -0.1020\* [-0.1251, -0.0786] |
| `LAT/turn_left` | 251 | +0.0757\* [+0.0284, +0.1259] | -0.0757 [-0.1852, +0.0268] |
| `LAT/turn_right` | 396 | +0.1136\* [+0.0596, +0.1682] | -0.0530 [-0.1320, +0.0274] |

### ⭐ The finding ADE was hiding, in the opposite direction

**refcv3 is LATERALLY BETTER than hold-action, and LONGITUDINALLY much worse — and ADE reports the
longitudinal loss because along-track error dominates it.**

* On manoeuvre windows the model **beats** `ha` on cross-track
  (-0.0488\* [-0.0779, -0.0174] m) and on heading
  (-0.4608\* [-0.7323, -0.1911] °), and on turns it beats it harder:
  `turn_left` heading -1.3793\* [-2.2168, -0.7092] °,
  `turn_right` heading -1.0281\* [-1.7276, -0.3725] °.
* On the **tactical lateral** decision it is right more often than `ha` on turns:
  `turn_right` +0.1136\* [+0.0596, +0.1682],
  `turn_left` +0.0757\* [+0.0284, +0.1259].
* On the **tactical longitudinal** decision it is much worse exactly where the ADE loss lives:
  `brake_stop` -0.3138\* [-0.3886, -0.2392],
  `accelerate` -0.3048\* [-0.3893, -0.2206].
* ⚠️ One consistent lateral loss: **yaw-rate MAE**, worse in almost every stratum
  (straight+constant +0.2410\* [+0.1394, +0.3717]).
  A discrete anchor fan produces a jerkier curvature profile than a held steering angle does; that
  is a property of the *representation*, and it is on the same axis as the anchor-fan finding the
  registry already carries.

⇒ **This is the four-families rule paying for itself.** A reader given only the ADE row would
conclude the model is worse at everything. It is not: it is a better *steerer* and a much worse
*speed controller* than doing nothing.

---

## 6. The other headline, `os − ha0`, stratified — because it needs the same qualifier in reverse

Direction `os - ha0`. Negative = `os` **better** than the constant-velocity floor.

| stratum | n win | n ep | `os` | `ha0` | `os − ha0` | verdict |
|---|---:|---:|---:|---:|---|---|
| **CROSS — straight + constant speed** | 3,240 | 139 | 0.3093 | 0.3335 | **-0.0242** [-0.0488, -0.0016] | os BETTER |
| **CROSS — manoeuvre (everything else)** | 1,583 | 103 | 0.7134 | 1.3656 | **-0.6522** [-0.7628, -0.5515] | os BETTER |
| | | | | | | |
| LATERAL — `lane_keep` | 4,176 | 141 | 0.4107 | 0.5181 | **-0.1074** [-0.1339, -0.0802] | os BETTER |
| LATERAL — `turn_left` | 251 | 35 | 0.6520 | 1.6665 | **-1.0145** [-1.2277, -0.8041] | os BETTER |
| LATERAL — `turn_right` | 396 | 41 | 0.6381 | 1.6683 | **-1.0302** [-1.2833, -0.7633] | os BETTER |
| | | | | | | |
| LONGITUDINAL — `steady` | 3,654 | 141 | 0.3330 | 0.4790 | **-0.1460** [-0.2124, -0.0902] | os BETTER |
| LONGITUDINAL — `brake_stop` | 631 | 78 | 0.8699 | 1.2614 | **-0.3914** [-0.4786, -0.3091] | os BETTER |
| LONGITUDINAL — `accelerate` | 538 | 72 | 0.6799 | 1.2940 | **-0.6141** [-0.7119, -0.5282] | os BETTER |

⚠️ **THE `os − ha0` WIN IS ALSO A MANOEUVRE PHENOMENON, AND ON THE MAJORITY STRATUM IT IS FRAGILE.**
On straight+constant windows — **67.18% of the corpus** — the model's
advantage over constant velocity is **-0.0242** [-0.0488, -0.0016] at 95 %, and at the
Bonferroni-adjusted 97.5 % level it is **-0.0242** [-0.0529, +0.0010] — **not separated**.
The manoeuvre stratum supplies **92.93 %** of the pooled
-0.2304 m from 32.82% of the windows;
straight+constant supplies **7.07 %**.

⇒ **refcv3's entire measured advantage over the trivial floor is earned on the third of the corpus
where something happens; on two thirds of it, it is 2.4 cm and does not survive a two-cell
adjustment.** That is the more useful reading of the `WON` row than the pooled −0.2304 m gives, and
it should travel with it.

---

## 7. Cross-checks — the conclusion is not an artefact of my stratifier

### 7.1 A completely different stratifier: the banked v7.2 tactical labels

⛔ **SUBSET ONLY, and NOT a random one.** These labels cover **1,157 of
4,823 windows (24.0%)**: a clip carries ONE v7.2 record and
its band admits `|t_now − t0| ≤ 2.0 s` (`v7_labels.window_in_band`), which **enriches manoeuvres**.
⇒ **the v7 mix is NOT the corpus mix and must never be quoted as one.** This block tests only
whether the *sign and ordering* of the contrast survive a different cut.

| stratum (v7.2 labels → kin3) | n win | n ep | `os − ha` | verdict |
|---|---:|---:|---|---|
| `V7:CROSS/straight_const` | 443 | 54 | **+0.1057** [+0.0614, +0.1535] | os WORSE |
| `V7:CROSS/manoeuvre` | 714 | 87 | **+0.1988** [+0.1384, +0.2587] | os WORSE |
| `V7:LAT/lane_keep` | 1,050 | 128 | **+0.1616** [+0.1218, +0.2048] | os WORSE |
| `V7:LAT/turn_left` | 40 | 5 | **-0.0130** [-0.3347, +0.2752] | UNDERPOWERED ⚠️ |
| `V7:LAT/turn_right` | 67 | 8 | **+0.2927** [+0.1149, +0.4508] | os WORSE ⚠️ |
| `V7:LON/steady` | 443 | 54 | **+0.1057** [+0.0614, +0.1535] | os WORSE |
| `V7:LON/brake_stop` | 408 | 50 | **+0.2639** [+0.1724, +0.3606] | os WORSE |
| `V7:LON/accelerate` | 306 | 37 | **+0.1120** [+0.0669, +0.1560] | os WORSE |

✅ **Same answer.** `os − ha` is positive and separated in every adequately-powered v7 cell, and the
manoeuvre cell (**+0.1988** [+0.1384, +0.2587]) again exceeds the straight+constant cell
(**+0.1057** [+0.0614, +0.1535]). Two stratifiers that share no LABEL-DERIVATION path and disagree about
40.0 % of the longitudinal labels give the same
verdict. That is the two-probe standard, satisfied.

**How much the two stratifiers actually agree, on the 1,157 windows they both cover:**
lateral **87.21%**, longitudinal **59.98%**.
The longitudinal disagreement is expected and is not a defect of either: the v7 vocabulary is
**semantic intent** (`ADAPT_SPEED_FOR_CURVE` n = 182,
`CREEP` n = 49 both project to `brake_stop`) while the kinematic gate is a
threshold on **realised Δv**. They answer different questions and the projection
`V7_TO_KIN3_*` is many-to-one by design.

### 7.2 The v2 curvature gate (§1) — the mix conclusion is gate-insensitive.

---

## 8. Controls — all five pass, and one standing claim needed correcting

| control | what it must read | result |
|---|---|---|
| **C1 pooled reproduction** | the banked `os` 0.4419, `ha` 0.2996, `ha0` 0.6723 and both paired deltas, to 4 dp | ✅ **PASS** — all five exact |
| **C2 partition** | each grouping sums to 4,823 with every window in exactly one cell | ✅ **PASS** (4 groupings) |
| **C3 decomposition identity** | `Σ (n_s/N)·δ_s == pooled δ` | ✅ **PASS**, residual ≤ 5.6e-17 |
| **C4 constant-zero control** | a zero-path predictor's ADE inside each cell **==** that cell's mean ‖GT‖, recomputed in float64 numpy | ✅ **PASS** (every cell) |
| **C5 floor invariance across steps** | the same cells must give the same `ha`/`ha0` means on the step-30,000 dump | ✅ **PASS** — all 16 cell means agree to < 1e-9 |

**C1 in full** — the reproduction is exact, so everything downstream is on the published surface:

| quantity | published | measured here |
|---|---|---|
| `arm_os_ade_m` | 0.4419 | 0.4419 |
| `arm_ha_ade_m` | 0.2996 | 0.2996 |
| `arm_ha0_ade_m` | 0.6723 | 0.6723 |
| `paired_os_minus_ha` | +0.1423 [+0.1187, +0.1658] | +0.1423 [+0.1187, +0.1658] |
| `paired_os_minus_ha0` | -0.2304 [-0.2881, -0.1781] | -0.2304 [-0.2881, -0.1781] |

### ⚠️ A standing claim that is *nearly* true and should be stated precisely

The brief and the surrounding notes carry: *"`ha` and `ha0` are **bit-identical** at 30k and 40,284,
which is a control proving one measurement surface."* **MEASURED here, that is true for `ha0`, `g`,
`ws` and `v0` — and NOT literally true for `ha`.**

| dump key | bit-identical across the two steps | differing elements | max abs Δ |
|---|---|---:|---:|
| `g` | ✅ yes | 0 / 38,584 | 0.000e+00 |
| `ha` | ⛔ **no** | 6 / 38,584 | 2.384e-07 |
| `ha0` | ✅ yes | 0 / 38,584 | 0.000e+00 |
| `ws` | ✅ yes | 0 / 4,823 | 0.000e+00 |
| `v0` | ✅ yes | 0 / 4,823 | 0.000e+00 |

The deviation is **float32-ulp scale (2.38e-07 m on
6 of 38,584 elements)** and moves **no**
per-cell mean at 1e-9, so the control's *conclusion* — one measurement surface, the strata partition
the same windows — **stands unchanged**. The word *"bit-identical"* does not, for that one arm.
⇒ the accurate form is **"`ha0` is bit-identical; `ha` agrees to float32 resolution
(≤ 2.4e-07 m on 6/38,584 elements)"**.
Root-cause class: *a true claim quoted one notch stronger than it was measured* — the same family as
quoting an exponent without its window.

---

## 9. ⇒ WHAT THE PUBLISHED SENTENCES SHOULD READ

The published headline is **not wrong** and needs **no retraction**. It is **incomplete in the
direction that flatters the model**, so it needs one added sentence in both places.

### 9.1 `Project Steering/MODEL_REGISTRY.md` §4.5 — after the `⛔⛔ THE HEADLINE IS UNCHANGED…` block

> ⭐ **AND THE POOLED NUMBER IS THE MODEL'S BEST CASE, NOT ITS TYPICAL ONE — STRATIFIED 2026-09-04.**
> The eval corpus is **67.2% straight-AND-constant-speed windows**
> (3,240 of 4,823; laterally 86.6% `lane_keep`,
> longitudinally 75.8% `steady`), so the BEV-Planner mix artefact
> (arXiv `2312.03031`) was a live hypothesis. **It is REFUTED, in the direction that makes the row
> worse:** `os − ha` is +0.0882 [+0.0701, +0.1062]
> on straight/constant road and +0.2530 [+0.2007, +0.3058]
> on manoeuvre windows — **2.9× worse where it counts**, both separated,
> both surviving a Bonferroni-adjusted 97.5 % interval. The worst single stratum is **`brake_stop`**:
> `os` 0.8699 m vs `ha` 0.3769 m,
> +0.4930 [+0.4237, +0.5575] —
> **2.3×** the control's error on
> 13.1% of the windows. ⇒ `brake_stop` + `accelerate` are
> 24.2% of the corpus and **58 %** of the pooled
> deficit. ⚠️ **The sign is not uniform, though:** three lateral cells are not separated
> (`turn_left` n = 251, `turn_left`+`steady` n = 184,
> `turn_right`+`steady` n = 230), and `turn_right`+`accelerate`
> (-0.1872 [-0.3604, +0.0010],
> n = 73) hints the model's way and is **UNDERPOWERED, not a win**.
> ⭐ **And ADE hides the reverse finding:** on manoeuvre windows `os` **beats** `ha` on cross-track
> (-0.0488\* [-0.0779, -0.0174] m) and heading
> (-0.4608\* [-0.7323, -0.1911] °) — refcv3 is a **better steerer and a
> much worse speed controller** than doing nothing. Record:
> `taniteval/results/refcv3-40284-stratified.json`, 0 GPU.

Also add, beside the `os − ha0` **WON** row:

> ⚠️ **the `os − ha0` win is a MANOEUVRE win.** 93 % of the pooled
> -0.2304 m comes from the 33% of windows that
> are not straight+constant; on the 67% that are, it is
> -0.0242 [-0.0488, -0.0016]
> and **not separated** at the two-cell-adjusted 97.5 % level.

### 9.2 `Project Steering/HF_CARD_tanitad-refc-v3.md` — beside the `os` − `ha` line (L266 / L308)

> ⛔ **and the stratified reading is worse, not better.** 67% of the
> 4,823 scored windows are straight at constant speed; on those `os − ha` is
> +0.0882 m, and on the 33% that
> involve a manoeuvre it is +0.2530 m
> (2.9×). The pooled +0.1423 m is a
> mix-weighted average, not the typical case. Worst stratum: braking/stopping,
> +0.4930 m. ⭐ On the same manoeuvre windows the model **beats** the
> control laterally (cross-track -0.0488\* [-0.0779, -0.0174] m,
> heading -0.4608\* [-0.7323, -0.1911] °): the deficit is
> **longitudinal**, not general.

⚠️ **Neither edit changes a published number.** Both are additive, so no HF version needs
retracting — only extending.

---

## 10. Power, multiple comparisons, and what this does NOT say

* **Power floors declared in advance:** a cell below **100 windows** or
  **10 episodes** whose interval straddles zero is stamped
  `UNDERPOWERED` by the tool and **may not be read as "no effect"** (RETRACTION_LOG #17). Two cells
  carry that stamp: `turn_left`+`accelerate` (n = 40 / 12 eps)
  and `turn_right`+`accelerate` (n = 73 / 19 eps).
* **Multiple comparisons:** the **only** contrast promoted to a claim is the registered 2-cell
  primary, reported at both 95 % and Bonferroni 97.5 %; both cells are separated at both levels.
  Every other cell is **descriptive**, with its n printed. No cell was selected after seeing the
  result and none is quoted as a significance test.
* ⚠️ **A structural caveat that belongs with any `ha` comparison.** `ha` holds the action *closing
  at t0* — it is a function of the **past**, while the strata are defined by the **future**. On
  windows where the future continues the present (a sustained brake, a sustained turn) `ha` is close
  to an oracle, and that is exactly the corpus's temporal autocorrelation, not a defect of the cut.
  It means `brake_stop` is the stratum where `ha` is *strongest*, which makes the model's
  2.3× loss there a statement about
  the model, not about the baseline being unfair.
* ⛔ **Everything here is OPEN LOOP.** No arm's trajectory affects the data it is next fed. The word
  "closed" appears nowhere in this record.
* ⛔ **NON-PARITY corpus, ORACLE nav.** Both statements travel with every number above; the only
  admissible cross-model statistic remains each arm's margin over the shared `ha0` floor.
* **What this does not test:** whether a different *checkpoint* or a different *grid* would
  stratify the same way. C5 shows the floors and strata are stable across the two banked steps; it
  does not show the model's per-stratum deficit is.

---

## 11. What this implies for refcv4 — the reason the distinction mattered

1. ⛔ **The next lever is LONGITUDINAL and it is specifically DECELERATION.** `brake_stop` is
   13.1% of windows and **45 %** of the deficit,
   at 2.3× the control's error, with the
   tactical `lon` agreement -0.3138\* [-0.3886, -0.2392] against it. This is
   the 5-way-softmax longitudinal blindness (`CLAUDE.md`) reappearing in the *trajectory*, not just
   the head, on a factored-head model — i.e. **factorising the head did not fix the fan**.
2. ⭐ **Do not spend the next run on lateral.** On the windows that matter the model already beats
   hold-action on cross-track and heading and gets the lateral tactical class right more often. A
   lateral improvement cannot move the headline; a braking improvement can move most of it.
3. ⚠️ **The anchor fan remains the named suspect and this sharpens it.** The registry's finding —
   a perfect selector plus oracle nav (0.0751 + 0.0239 = 0.099 m) still does not reach hold-action's
   0.1423 m — now has a stratum: the fan does not *contain* the decelerating trajectories.
   A fan-coverage probe on `brake_stop` windows (is the GT path within ε of ANY of the 128 anchors?)
   is the cheapest discriminating experiment and needs no training.
4. **Power the two accelerate-while-turning cells deliberately** if the hint in
   `turn_right`+`accelerate` is to be resolved; at n = 73 it cannot be.

---

## 12. Deliverable manifest

| artifact | where it lives | in more than one place? |
|---|---|---|
| `taniteval/tools/stratified_openloop.py` — the instrument (0 GPU, `--dump` + `--expect`) | `repo:taniteval/tools/stratified_openloop.py` | yes (repo + scratchpad) |
| `taniteval/results/refcv3-40284-stratified.json` — the raw record | `repo:taniteval/results/refcv3-40284-stratified.json` | yes |
| `taniteval/results/RESULT-refcv3-40284-stratified.md` — this document | `repo:taniteval/results/RESULT-refcv3-40284-stratified.md` | yes |
| the input dump | `repo:taniteval/results/refcv3-40284-openloop-dump.tar.gz` (unchanged) | already banked |
| the pre-registration | embedded in the JSON at `.prereg` and in the tool docstring | yes |

**Reproduce with zero GPU:**

```
python taniteval/tools/stratified_openloop.py \
  --dump     <extracted refcv3_40284_dump> \
  --dump-30k <extracted refcv3 30k dump> \
  --expect   taniteval/results/refcv3-40284-openloop.ARM.json \
  --out      taniteval/results/refcv3-40284-stratified.json \
  --tag      refcv3-40284-stratified
```

**Escalation to the Master Mind:** the two additive edits in §9 are needed on the **HF model card**
and **`MODEL_REGISTRY.md` §4.5**, which are outside this stream's ownership. Neither retracts a
number. The `ha` bit-identity precision in §8 should also be corrected wherever that phrase is
quoted.
