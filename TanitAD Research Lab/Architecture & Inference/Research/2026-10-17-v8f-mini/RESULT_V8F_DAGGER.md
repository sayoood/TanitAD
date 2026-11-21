# PREREG_V8F_DAGGER: result, both seeds (DAG vs DAG-0; banked and trained on box 2, T1-kin read on the dev-box CPU; read 2026-10-27)

**Outcome, as committed: ROBUST.** The pre-registered reader (`dag_read.py` at Amendment 2, sha256 `1cf56c7c30195ac4…`) matched exactly one
row of the outcome table, `ROBUST`: *"R-TLAT and R-TLON both clear (§5) at both seeds; R-ADE not separated-worse at either seed; R-T0 ≤ +F/2
at both"*. **All 176 controls PASS.**

* **Committed next, verbatim from the ROBUST row:** *"adopt DAG as the base. Re-read T1 dominance over P10 and FLAT's committed repeat on it.
  If the crawl beside rows (M1, L2) still fail, next is DAgger iteration 2 (DAG's own states), under its own read"*. Registered as
  `Project Steering/PREREG_V8F_DAG_REREAD.md` (commit `bdf8d561`) before any of its rows.
* **Evidence class:** MEASURED (ours). Every number in §§ 1–2 is copied from `raw/2026-10-27-dagger-run/read/DAG_READ.json` (sha256
  `3393fdd5…`, written 06:37:54 by the re-read in `read/reread.log`). § 3 cites `raw/2026-10-27-dagger-families/`.
* **Tier, as the reader states it:** T1 (T1-kin: a self-action OPEN loop, plain causal bucket, nav dead-reckoned; ⛔ never "closed loop") for
  R-TLAT / R-TLON / R-ADE; T0 (teacher-forced own-code, eval-141) for R-T0. No post-map is applied: the runs and the bank are PLAIN.
* **Estimator of every interval:** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, over the 141 episodes (2,399
  windows). DAG − DAG-0 compares two networks per seed on the same windows. It answers *"would another draw of EPISODES say this?"*. The
  training-seed question is answered by the two seeds and by the replicate floor RF = |F1(DAG-0) − F1(ORC-0)|, which raises the bar to
  max(2F, 2·RF). The planner is deterministic, so no inference-seed question arises.
* **Pre-registration:** `Project Steering/PREREG_V8F_DAGGER.md`.
  * The draft was banked at `ac139f71` before O-RECOVER's read. Amendment 1 (`A1-2026-10-26`) was landed at `2a78f059` before any data.
  * The reader's PREREG-TEXT control pins that text (sha256 `a4df4b8f…`).
  * Amendment 2 (`A2-2026-10-27`, landed `7d29172b`) is post-data and control-only. It made BANK-CONFIG read `limit_clips` as the generator
    writes it (0 = no limit). The first read REFUSED on that clause alone, at the bank stage, before any outcome row
    (`raw/2026-10-27-dagger-run/dag_read_refused.out`).
  * The reader was then re-proved on its final bytes (120 passed, 160/160 mutants) and run once.

---

## 1. The outcome as committed

| outcome | condition | committed next |
|---|---|---|
| **ROBUST** | **R-TLAT and R-TLON both clear (§5) at both seeds; R-ADE not separated-worse at either seed; R-T0 ≤ +F/2 at both** | **adopt DAG as the base. Re-read T1 dominance over P10 and FLAT's committed repeat on it. If the crawl beside rows (M1, L2) still fail, next is DAgger iteration 2 (DAG's own states), under its own read** |
| PARTIAL | exactly one head clears at both seeds; guards held | adopt for the head that moved; iteration 2 for the other head |
| NO EFFECT | no head clears at both seeds; guards held | T-EGOCUT, under its own PREREG |
| HARMS | R-ADE separated-worse at ≥ 1 seed, or R-T0 above +F/2 at ≥ 1 seed | DAG is not adopted (p = 0.1 ⇒ T-EGOCUT) |
| BLOCKED | COVERAGE < 0.5 | a rejoin-target form, under its own PREREG |
| REFUSED | any control fails | fix and re-run |

A head **clears** at seed s iff it is separated-better AND its point is ≥ bar(h, s) = max(2F_h, 2·RF_h,s), inclusive. 2F = 0.0248 (lat) /
0.0146 (lon), the T0 fresh seed floors.

### 1.1 The four rows, per seed (DAG − DAG-0)

F1 rows: T's macro-F1 at T1-kin, ticks ≥ 1, on matched cells (scored in both runs). ADE rows: per-window mean.

| row | seed | tier | DAG | DAG-0 | DAG − DAG-0 [95 % interval] | separated | bar | clears / guard |
|---|---|---|---|---|---|---|---|---|
| R-TLAT: lat macro-F1 | s0 | T1-kin | 0.6029 | 0.4947 | **+0.1082 [+0.0491, +0.1572]** | better | 0.0248 (2F) | **clears** |
| R-TLON: lon macro-F1 | s0 | T1-kin | 0.3213 | 0.2659 | **+0.0554 [+0.0300, +0.0847]** | better | 0.0502 (2·RF) | **clears** |
| R-ADE: T1-kin ADE6, m | s0 | T1-kin | 2.2777 | 2.8212 | **−0.5435 [−0.7564, −0.3690]** | better | guard: not separated-worse | held |
| R-T0: T0 own-code ADE6, m | s0 | T0 | 1.9736 | 1.9717 | **+0.0019 [−0.0230, +0.0266]** | no | guard: point ≤ 0.0604 | held |
| R-TLAT: lat macro-F1 | s1 | T1-kin | 0.6385 | 0.5201 | **+0.1184 [+0.0755, +0.1585]** | better | 0.0248 (2F) | **clears** |
| R-TLON: lon macro-F1 | s1 | T1-kin | 0.3311 | 0.2505 | **+0.0806 [+0.0560, +0.1043]** | better | 0.0278 (2·RF) | **clears** |
| R-ADE: T1-kin ADE6, m | s1 | T1-kin | 2.4565 | 2.8611 | **−0.4046 [−0.5496, −0.2782]** | better | guard: not separated-worse | held |
| R-T0: T0 own-code ADE6, m | s1 | T0 | 2.0984 | 2.0501 | **+0.0483 [+0.0178, +0.0807]** | worse | guard: point ≤ 0.0604 | held |

* Matched cells: lat 3,863 (s0) / 3,849 (s1); lon 3,770 / 3,758. All rows are over 141 episodes.
* **The replicate floor:**
  * RF_lat = 0.0067 (s0) / 0.0043 (s1), from DAG-0 against ORC-0. ORC-0 is a same-argv replicate of DAG-0, trained by another build (control
    ORC0-REPLICATE).
  * RF_lon = 0.0251 / 0.0139.
  * The lon bar is therefore set by 2·RF, at 0.0502 / 0.0278. The lat bar stays at 2F.
  * The smallest margin over a bar is R-TLON s0: 0.0554 against 0.0502.
* **Own-cells version beside** (each run's own cells, not the matched ones): lat +0.1079 [+0.0499, +0.1605] / +0.1167 [+0.0751, +0.1582];
  lon +0.0558 [+0.0276, +0.0839] / +0.0819 [+0.0576, +0.1061]. All are separated-better.
* ⚠ **R-T0 at s1 is separated-WORSE**, by +0.0483 m: 0.40 F, under the registered guard of F/2. DAgger did not improve logged-state
  (teacher-forced) ADE at either seed. It improved T on the states the model itself produces, which is what it was designed to do.

### 1.2 Size of the effect, read against the rig's noise

* **R-TLAT:** the points are 4.4× / 4.8× 2F and 16× / 28× RF_lat.
* **R-TLON:** 1.10× / 2.90× its bar (2·RF), and 2.2× / 5.8× RF_lon.
* **R-ADE:** the T1-kin ADE6 falls by 0.54 / 0.40 m. That is 4.5 F / 3.4 F, with F = 0.1207 m (`MODEL_REGISTRY.md` §17, M2b's two seeds). It
  was a guard, not a target, and it moved further than any head row.

---

## 2. The controls: all 176 PASS

| group | what it checks | n |
|---|---|---|
| PREREG-TEXT, M2B-CKPT | the PREREG text and the init checkpoint are the registered bytes | 1 + 2 |
| BANK-FILES, BANK-CONFIG, BANK-ACCOUNTING, K-B0…K-B5 | the bank's shards are the published ones and its config is the registered rollout (PLAIN, causal, dead-reckoned, m = 10). K-B0: tick-0 identity bit for bit. K-B1: seeded replay. K-B4 / K-B5: re-anchoring. The records add up | 2 × 11 |
| SUMMARY, INIT-CKPT, ARGV, GATE-TOKEN, BANK-DECLARED, DAGGER-REACHED, TRAIN-LIST | each run is the registered argv, passed its launch gate, and reached the lever: swap share 0.1013 (s0, 14,688 of 145,037 eligible) / 0.1008 (s1, 14,839 of 147,199); DAG-0 exactly 0 | per run |
| T0-CKPT, T0F-CKPT, T0-AGES | the T0 evals read the trained checkpoints, at the registered ages | per run |
| K1…K7, GRID, T1-CONFIG, DUMP-WINDOWS, GRID-COUNTS, R2-ADE6 (and the logged-bucket K2 / K3 / GRID / T1-CONFIG) | the T1-HEADS reader's controls on every run (ORC A2 / A4): K4's comparator is the arm's own fresh-age T0 eval, and K7 requires the rebuilt track to equal the dumped one | 6 runs × 16 |
| PAIR, ORC0-REPLICATE, SAME-WINDOWS, K-B3.2 | the pairs are scored on the same windows; ORC-0 is a same-argv replicate | 2 + 2 + 4 + 2 |
| K1-DAG-ROWS | the reader on (a, a) reads +0 [0, 0] on its own rows (12 rows) | 1 |

**The bank** (per seed, 4,512 TRAIN clips, 306,566 windows, 613,132 own states before cuts):

| seed | COVERAGE (eligible fraction; bar 0.5) | states kept | cuts: off-lat / off-head / end-edge / non-finite |
|---|---|---|---|
| s0 | **0.7556** (231,650 / 306,566 windows) | 407,450 | 38,212 / 57,718 / 109,752 / 0 |
| s1 | **0.7687** (235,648 / 306,566) | 414,765 | 44,804 / 39,169 / 114,394 / 0 |

---

## 3. The four metric families (binding), per family, never pooled

Computed by the families package `raw/2026-10-27-dagger-families/` (`FAMILIES_DAG.md`, `DAG_FAMILIES_ROWS.json`). Those are O-RECOVER's
instruments, unchanged: `v8f_paired_families.py`, `family_windows` + `paired_row`, and `read_dk`. Its 14 controls PASS:
* each instrument's (a, a) identity reads +0 [0, 0];
* the plan-derived T0 agreement is equal across the two tools;
* R-T0, R-ADE, R-TLAT and R-TLON are reproduced to 4 dp;
* an independent verifier passes 107 / 107, including 30 cells recomputed from the raw npz.
Estimator as above. DAG − DAG-0, s0 / s1. Levels are DAG / DAG-0.

### 3.1 LONGITUDINAL

| row | tier | s0 | s1 |
|---|---|---|---|
| speed MAE (m/s) | T1-kin | **−0.466 [−0.613, −0.331]** better (1.31 / 1.78) | **−0.282 [−0.390, −0.181]** better (1.59 / 1.87) |
| along-track MAE (m) | T1-kin | **−0.402 [−0.615, −0.230]** better | **−0.249 [−0.376, −0.131]** better |
| accel MAE (m/s²) | T1-kin | −0.019 [−0.293, +0.233] | **+0.449 [+0.134, +0.810] WORSE** |
| LEAD-VIOL, at-risk share | T1-kin | **+0.042 [+0.013, +0.079] WORSE** (0.082 / 0.040; 595 windows / 70 episodes) | **+0.030 [+0.012, +0.055] WORSE** (0.052 / 0.022) |
| \|min-TTC error\| vs the human (s) | T1-kin | **+0.868 [+0.057, +1.598] WORSE** | **+0.597 [+0.002, +1.240] WORSE** |
| \|time-gap error\| vs the human (s) | T1-kin | **+0.084 [+0.015, +0.170] WORSE** | +0.013 [−0.015, +0.040] |
| \|headway error\| vs the human (m) | T1-kin | +0.228 [−0.072, +0.506] | +0.072 [−0.114, +0.253] |
| min headway (m), no direction | T1-kin | −1.02 [−1.45, −0.58] (24.74 / 25.88) | −0.56 [−0.96, −0.22] |
| speed / accel / along-track MAE | T0 | not separated | **+0.020 / +0.034 / +0.046 WORSE** (the R-T0 cost, inside its guard) |

* ⚠ **A distance-keeping cost at T1-kin, at both seeds.** On its own motion DAG closes on the lead. Behind a lead, its at-risk violation share
  doubles (8.2 % vs 4.0 %, and 5.2 % vs 2.2 %), and its min-TTC error grows. Its headway error against the human is not separated.
* DAG is faster than DAG-0 and closer to the human's speed (speed MAE −0.47 / −0.28 m/s). It crawls less (crawl fraction 0.087 / 0.140
  against 0.137 / 0.161, `DAG_READ.json` beside).
* ⚠ In T1-kin the lead follows its LOGGED future and does not react to the ego.
* The programme's anti-echo block admits no longitudinal CAPABILITY claim for any run (`FAMILIES_DAG.md` §1.1). These are contrasts.

### 3.2 LATERAL

| row | tier | s0 | s1 |
|---|---|---|---|
| heading MAE (deg) | T1-kin | **−3.51 [−4.71, −2.47]** better (6.19 / 9.71) | **−1.76 [−2.28, −1.20]** better |
| cross-track MAE (m) | T1-kin | **−0.296 [−0.398, −0.211]** better | **−0.307 [−0.419, −0.213]** better |
| yaw-rate MAE (deg/s) | T1-kin | **−6.46 [−9.70, −3.71]** better | −1.70 [−4.34, +0.40] |
| curvature MAE (1/m) | T1-kin | −0.0071 [−0.0180, +0.0041] | **+0.0168 [+0.0052, +0.0303] WORSE** |
| all four rows | T0 | not separated | not separated |

### 3.3 TACTICAL

| row | tier | s0 | s1 |
|---|---|---|---|
| R-TLAT / R-TLON (decisive, §1.1) | T1-kin | +0.108 / +0.055 | +0.118 / +0.081 |
| lateral decision agreement, plan-derived | T1-kin | **+0.100 [+0.070, +0.133]** (0.786 / 0.686) | **+0.192 [+0.151, +0.233]** (0.778 / 0.587) |
| longitudinal decision agreement, plan-derived | T1-kin | **+0.054 [+0.024, +0.085]** | **+0.072 [+0.038, +0.106]** |
| TURN_R F1 (lat head, ticks ≥ 1) | T1-kin | **+0.171 [+0.107, +0.230]** (292 cells / 27 episodes) | **+0.190 [+0.117, +0.260]** |
| TURN_L F1 | T1-kin | +0.133 [−0.026, +0.263] (92 cells / 11 episodes) | **+0.148 [+0.053, +0.238]** |
| LANE_KEEP F1 | T1-kin | **+0.021 [+0.012, +0.030]** | **+0.017 [+0.009, +0.026]** |
| worse at T0 | T0 | lon-head accuracy −0.017 | lateral plan agreement −0.011; goal-point error +0.101 m |

* The turn classes, where M2b lost half its F1 on its own motion (§7.15 of the report), gain +0.13 to +0.19.
* No LANE_CHANGE cell is labelled in eval-141, so those classes are not computed (n = 0).

### 3.4 STRATEGIC (not paused for v8F)

**Paired at T1-kin**, built from the T1-HEADS reader's own arrays, since no paired reader existed. S's checkpoint error, ticks ≥ 1:

| arc | position s0 | position s1 | heading s0 | heading s1 |
|---|---|---|---|---|
| 30 m | **−2.49 [−2.98, −1.97]** m (3.39 / 5.88) | **−3.44 [−4.02, −2.89]** m (3.38 / 6.81) | **−6.57°** | **−9.66°** |
| 50 m | **−4.72 [−5.65, −3.77]** m (6.34 / 11.05) | **−6.49 [−7.52, −5.46]** m (6.22 / 12.70) | **−6.70°** | **−9.71°** |
| 80 m | **−7.60 [−9.22, −5.99]** m (12.62 / 20.22) | **−10.36 [−12.13, −8.63]** m (12.38 / 22.74) | **−6.19°** | **−8.73°** |

* All 12 cells are separated-better. S's route checkpoints on the model's own motion are about halved.
* At T0, 0 of 12 cells are separated: S's logged-state skill is unchanged.
* ⚠ The pairing is new in this package. It was verified against the reader's unpaired per-run values (§3.7 below), not by a mutation
  proof.

### 3.5 Beside: DAG − ORC and DAG − M2b

* **DAG − ORC** (T1-kin): R-TLAT +0.073 / +0.088 and R-TLON +0.045 / +0.036, all separated-better. On-policy states beat the perturbation law
  on both heads, but the two arms differ in more than the lever.
* **DAG − M2b** (T0, the previous base): ADE6 −0.202 (s0, separated) / +0.044 (s1, n.s.). LEAD-VIOL at s0 is +0.022, worse. S-POS@80 at
  s0 is +0.49 m, worse. Speed, accel and along-track change sign across seeds.

### 3.6 Not computed, with the reason and n

* Paired T1-kin head accuracy and goal-head F1: the paired-families CLI reads eval dumps, not tick dumps. The unpaired goal-head F1 is in
  §3.7.
* LANE_CHANGE classes: no labelled cell (n = 0).
* The crawl rows M1 / L2 against P10, and the off-route rows: these belong to `PREREG_V8F_DAG_REREAD` Read A, which owns every row against
  P10.

### 3.7 STRATEGIC beside the read (from `DAG_READ.json`, unpaired, within each run; decides nothing)

STRATEGIC is not paused for v8F (PI 2026-10-17, R73). S-POS is S's checkpoint position error against CTRV at 30, 50 and 80 m of arc, at T1-kin
ticks ≥ 1, run − CTRV (m, negative = S better than CTRV):

| run | S-POS@30 | S-POS@50 | S-POS@80 |
|---|---|---|---|
| DAG s0 | **−1.72 [−2.27, −1.28]** | **−4.48 [−5.63, −3.50]** | **−9.69 [−11.84, −7.57]** |
| DAG-0 s0 | +0.12 [−0.27, +0.49] | −0.91 [−1.97, +0.07] | **−3.89 [−6.09, −1.85]** |
| ORC-0 s0 | −0.07 [−0.45, +0.29] | **−1.35 [−2.36, −0.44]** | **−4.90 [−7.04, −2.89]** |
| DAG s1 | **−1.87 [−2.36, −1.44]** | **−4.90 [−5.96, −3.92]** | **−10.33 [−12.44, −8.35]** |
| DAG-0 s1 | **+0.81 [+0.35, +1.29]** (worse) | +0.33 [−0.81, +1.46] | −1.95 [−4.37, +0.49] |
| ORC-0 s1 | **+0.82 [+0.37, +1.25]** (worse) | +0.28 [−0.78, +1.28] | **−2.24 [−4.52, −0.04]** |

* DAG's S beats CTRV, separated, at every horizon and at both seeds. DAG-0's S beats CTRV only at 80 m at s0. At s1 it is separated-WORSE
  than CTRV at 30 m and not separated at 50 or 80 m.
* The off-route count of T at tick 2 is 180 (DAG) against 205 (DAG-0) at s0, and 181 against 232 at s1, of 2,399 windows.
* ⚠ These are unpaired, within-run contrasts. A paired DAG − DAG-0 S-POS at T1-kin needs a tick-dump instrument (a work item, as in
  O-RECOVER § 3.4).

**Tick-0 head F1** (did DAG lose logged-state skill?): lat 0.847 / 0.850 (DAG / DAG-0, s0) and 0.857 / 0.862 (s1); lon 0.466 / 0.471 and
0.483 / 0.511. These are unpaired. DAG is at or slightly below DAG-0 at tick 0, which is consistent with R-T0.

**Goal-head macro-F1, ticks ≥ 1:** 0.187 / 0.162 (DAG / DAG-0, s0) and 0.207 / 0.165 (s1).

---

## 4. What this does and does not establish

* ✅ **Established (MEASURED, decisive):** at T1-kin, supervising T on the model's own rolled-out states at p = 0.1 improves T's lateral and
  longitudinal decisions on its own motion, at both seeds, beyond 2F and beyond the run-to-run replicate floor. The executed T1-kin ADE6
  falls by 0.40–0.54 m. It is the first v8F training lever to clear a pre-registered T1 bar on both heads.
* ✅ **The lever, not the extra steps.** DAG-0 has DAG's argv with `--dagger-p 0` (M2b + the same 3,000 steps). Every row is DAG − DAG-0, so
  the fine-tune itself is controlled for.
* ⛔ **Not established:**
  * that v8F beats the kinematic floor on every family;
  * that the hierarchy beats a flat planner.
  Both are the committed next (`PREREG_V8F_DAG_REREAD.md`).
* ⛔ Not a closed-loop claim. T1-kin replays the logged lead and re-perception is not provisioned (T2).
* ⚠ Logged-state (T0) ADE did not improve and is slightly worse at s1 (+0.048 m, under the guard). DAgger moves the model on its own state
  distribution, not on the human's.
* ⚠ **Distance keeping costs at T1-kin** (§3.1). Behind a lead, DAG's at-risk violation share doubles and its min-TTC error grows, at both
  seeds; its accel error is worse at s1. ROBUST is the pre-registered outcome and stands. This cost is a beside finding, and it is a
  row of the dominance read (DK-VIOL against P10), which decides whether it blocks dominance.

## 5. Next (Rule Zero)

* **Running now:** `PREREG_V8F_DAG_REREAD.md` (dominance over P10 with a registered bar; FLAT's repeat at its tier and at T1-kin), zero
  training, dev-box CPU.
* **Its outcome decides** between DAgger iteration 2 (the crawl rows) and FLAT-DAG (attribution). Both are box-2 arms and need the PI's go.
* **T-EGOCUT** (built, mutation proof running) is parked by this outcome, per its draft's mapping. Its build is banked, not launched.

## 6. Where every number comes from

| what | where |
|---|---|
| the read | `raw/2026-10-27-dagger-run/read/DAG_READ.json` (sha256 `3393fdd5…`), `read/reread.log`, `read/reread_cmd.txt` |
| the refused first read | `raw/2026-10-27-dagger-run/dag_read_refused.out` |
| the reader at A2 + its proof | `raw/2026-10-26-dagger-read/` (`dag_read.py` sha256 `1cf56c7c…`) |
| banks, runs, gate tokens | `raw/2026-10-27-dagger-run/bank_s{0,1}/`, `M2b_s{0,1}_dag{,0}/`; checkpoints and shards on E: (`E:/Projects/TanitAD-artifacts/v8f-dagger/`), named by sha256 in each `SHA256SUMS.txt` |
| T1-kin manifest and pins | `raw/2026-10-27-dagger-run/t1/runs.json`, `t1/PINS.txt`; dumps and ticks on E: (`E:/Projects/TanitAD-artifacts/v8f-dagger-t1/`) |
| the four families | `raw/2026-10-27-dagger-families/` |
