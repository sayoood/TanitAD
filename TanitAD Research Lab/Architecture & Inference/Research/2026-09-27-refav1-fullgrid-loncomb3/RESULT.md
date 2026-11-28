# RESULT — refav1 `loncomb3` on the FULL 141-episode grid: **FAIL-WORSE** (pre-registered), and every zero-training planner lever on this checkpoint is eliminated

**2026-09-27, TrainingFlyWheel. Staged, never committed.** SPEC: `SPEC.md` (§4–§5, Amendments A1 §8, A2 §9,
A3 §10 — all staged before any `loncomb3` full-grid data existed). Checkpoint `refav1-b1-v72-ep3-speed` step 21,109
(md5 `1189bc020018c2c67ce03d566c390285`); code `stack/` + `taniteval/` at `b3f7ea6f` (the refav1 files are unchanged at
origin `c36b6ddd`); both arms run on the dev-box RTX 4060, 09:33–13:48 Berlin (`raw/run/provenance.txt`).
**Estimator:** paired episode-cluster bootstrap over 141 episodes (282 windows), n_boot 2,000,
`taniteval/tools/refav1_paired_delta.py`. **Tier:** T1 (self-action open loop; NOT closed loop). Every number below is
**MEASURED** from `raw/analysis/pd_fullgrid*.json` unless marked EXPLORATORY.

## 0. The verdicts, as written

| bar (SPEC) | measured (ADE, m) | verdict |
|---|---|---|
| **BAR-L1** (§4, primary) | `A1 − ha0_ext` **+0.0896 [+0.0117, +0.1756]**; `A2 − ha0_ext` **+0.1515 [+0.0710, +0.2424]** — both separated **WORSE** | **FAIL-WORSE** |
| **BAR-B1** (§8, the blend) | `b50(A1) − ha0_ext` **−0.1072 [−0.1731, −0.0444]**; `b50(A2) − ha0_ext` **−0.0797 [−0.1467, −0.0160]**; mean \|−0.0935\| > 2 × \|`b50(A2) − b50(A1)`\| = 2 × 0.0275 = 0.0550 | **PASS** |
| **BAR-B2** (§9/§10, the attribution bar) | stronger damped floor = **`damp50`** (`damp50 − dampha` −0.0201 [−0.0296, −0.0116], separated). `b50(A1) − damp50` **+0.0505 [+0.0120, +0.0918]**; `b50(A2) − damp50` **+0.0779 [+0.0343, +0.1226]** — separated **WORSE** at both seeds | **FAIL** |
| §9's clause on the floor | `damp50 − ha0_ext` **−0.1576 [−0.2297, −0.0990]**, separated | **`ha0_ext` is no longer refav1's strongest do-nothing floor**; every earlier refav1 reading "vs `ha0_ext`" must be restated against `damp50` |
| VOID checks (§5/§6) | `g, v0, ha, ha0, ha0_ext, ol, ws, eid, clip_index` **bit-identical** A1 vs A2 on 141/141 episodes; known-value control `w0 − ha0_ext` = **+0.0000 [+0.0000, +0.0000]** on every metric; the seeds really differ (`cl` differs in **122/282** windows) | **valid, not VOID** |

⇒ **The planner configuration `loncomb3` does not beat doing nothing on the full grid — it is worse, separated, at both
inference seeds.** Blending it 50/50 with `ha0_ext` does beat `ha0_ext` (BAR-B1), but a blend that contains NO
planner at all (`damp50 = 0.5·ha0 + 0.5·ha0_ext`) beats that blend, separated, at both seeds (BAR-B2). The planner
contributes nothing that a damped kinematic hold does not already give.

## 1. The committed predictions, scored
| SPEC prediction | measured | |
|---|---|---|
| §4: ADE ≈ −0.10 vs `ha0_ext` | +0.0896 / +0.1515 | **WRONG** |
| §4: LON speed still WORSE than `ha0_ext`, separated | +0.1262 [+0.0938, +0.1610] / +0.1492 [+0.1128, +0.1896] | right |
| §4: heading and yaw-rate BETTER | heading −0.39° [−1.05, +0.25] / +0.08° [−0.62, +0.77]; yaw-rate +0.0018 / +0.0127, all n.s. | **WRONG** (no separated gain) |
| §4: `turn_left` recall ≈ 0 | **1/15 = 0.0667** (A1), 2/15 = 0.1333 (A2); `ha0_ext` 14/15 = 0.9333 | right |
| §9: BAR-B1 likely passes, driven by damping | passes; `damp50` alone is better still | right |
| §9: the planner's separated contribution, if any, is LONGITUDINAL | `b50(A1) − damp50` LON speed −0.0412 [−0.0714, −0.0144], along −0.0326 [−0.0568, −0.0119] (separated); **at A2 NOT separated** (speed −0.0285 [−0.0604, +0.0013]) | holds at one seed only — not a lever effect |

## 2. Four families, T1, 141 clusters (strategic UNAVAILABLE: no route label on this surface, n = 0 — a work item)
| arm − floor | ADE | FDE | LON speed | LON along | LON accel | LAT cross | LAT heading ° | LAT yaw-rate | TAC lat ok | TAC lon ok |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 − `ha0_ext` | **+0.0896** | +0.1180 | **+0.1262** | +0.0251 | **+0.1006** | +0.0568 | −0.3908 | +0.0018 | +0.0142 | **−0.0780** |
| A2 − `ha0_ext` | **+0.1515** | **+0.2988** | **+0.1492** | **+0.0540** | **+0.1202** | **+0.1189** | +0.0835 | +0.0127 | −0.0532 | **−0.1241** |
| A1 − `damp50` | **+0.2472** | **+0.5877** | +0.0044 | −0.0064 | +0.0049 | **+0.3109** | **+1.8747** | **+0.0460** | **−0.0780** | −0.0319 |
| A2 − `damp50` | **+0.3092** | **+0.7685** | +0.0274 | +0.0224 | +0.0244 | **+0.3730** | **+2.3490** | **+0.0570** | **−0.1454** | **−0.0780** |
| `damp50` − `ha0_ext` | **−0.1576** | **−0.4697** | **+0.1218** | **+0.0315** | **+0.0957** | **−0.2542** | **−2.2574** | **−0.0443** | **+0.0922** | −0.0461 |
Bold = CI excludes 0 (full intervals in `raw/analysis/pd_fullgrid.md`, `pd_fullgrid_extra.md`).
Per-class decision recall (A1 / A2 / `ha0_ext`): `turn_left` 0.0667 / 0.1333 / 0.9333 (n 15); `turn_right` 0.10 /
0.20 / 0.90 (n 20); `lane_keep` 0.9352 / 0.8462 / 0.8016 (n 247); `brake_stop` 0.6667 / 0.6667 / 0.5778 (n 45);
`accelerate` 0.3111 / 0.3111 / 0.6222 (n 45).
⇒ **Against `damp50` the arm is TIED on every longitudinal metric and loses the whole difference LATERALLY.**

## 3. The inference-seed replicate is itself separated — and large
`A2 − A1` ADE **+0.0619 [+0.0094, +0.1169]**, LON speed +0.0230, heading +0.47°, `turn_left` recall 0.13 vs 0.07 —
all from the planner's sampling alone (122/282 windows change plan). On this config the inference-seed floor is
**0.062 m ADE**, vs `lonshift`'s −0.0076 on p4: the floor is strongly config-dependent (as `RESULT_HARVEST.md` §2
warned). ⇒ any single-seed refav1 "gain" under ~0.12 m (2×) on this config is not readable as a lever effect.

## 4. EXPLORATORY — zero GPU, post-verdict, every candidate built is reported (`raw/analysis/explore/`)
Not pre-registered; none of these may be quoted as a lever effect without its own pre-registration.
1. **The planner's LONGITUDINAL profile loses to holding the measured acceleration.** `kd_<src>` = the `damp50`
   path re-timed to `<src>`'s travelled distance (lateral geometry identical across arms; control `kd_self − damp50`
   = 0.0000 exactly). `kd_A1 − kd_ha0_ext` ADE **+0.0783 [+0.0576, +0.1004]**, `kd_A2 − kd_ha0_ext` **+0.0945
   [+0.0695, +0.1221]**, LON speed +0.1262 / +0.1492 — separated worse. It beats constant speed (`kd_A − kd_ha0`
   −0.0912 / −0.0751) but not a trivial acceleration hold. **The p4 finding "the planner's contribution over the damped
   hold is longitudinal" does not survive the full grid.**
2. **The p4-tuned lateral is worse than planning straight.** The 2026-09-04 shipped-cost dump (Thor, older stack) sits
   on the identical windows (`ha`, `ha0`, `v0`, `ws` bit-identical; `g` within 3.8e-6 m) and its lateral equals
   `ha0`'s EXACTLY (`shipped − ha0` = 0.0000 on cross, heading, yaw-rate: it plans κ = 0 on every window).
   `A1 − shipped` cross **+0.1977**, heading **+1.15°**, yaw-rate **+0.0434**; `A2 − shipped` **+0.2598 / +1.63° /
   +0.0544** (all separated) — while `loncomb3`'s LONGITUDINAL beats the shipped cost (speed −0.1444 / −0.1214,
   separated). ADE `A1 − shipped` +0.0629 [−0.0098, +0.1493], `A2 − shipped` **+0.1248 [+0.0448, +0.2124]**.
   ⇒ Tuning 30 arms on the 8-episode, turn-dense p4 panel bought a lateral behaviour that p4 rewarded (heading
   −3.36° there) and the lane-keep-dominated grid punishes (247/282 windows are `lane_keep`). **p4 is not an
   admissible tuning surface for lateral levers.**
3. **A stronger planner-free floor.** `kd_ha0_ext` (the damped path at `ha0_ext`'s timing: half the measured
   curvature, the measured acceleration held) − `ha0_ext` ADE **−0.2282 [−0.2946, −0.1702]**, − `damp50` **−0.0705
   [−0.0905, −0.0519]**. Replicated on the 2,399-window stride-4 grid of the same episodes: **−0.2638
   [−0.3374, −0.2005]** vs `ha0_ext` (`../2026-09-27-refav1-trunk-probe/`). A floor only makes tests stricter, so it
   may be adopted as the refav1 do-nothing floor now; a claim that anything BEATS it needs its own pre-registration.
4. **Seed ensemble** `0.5·A1 + 0.5·A2`: − A1 ADE +0.0126 [−0.0208, +0.0433] — no gain.
4b. **No stratum where the planner beats a damped floor** (strata from the GT path only, as later pre-registered for
   the trunk probe's A2′; ADE, A1 / A2): **straight** (228 windows / 132 clusters) vs `damp50` +0.1949 / +0.2682,
   vs `ha0_ext` +0.1529 / +0.2262 (all separated worse); **turn** (21 / 20) vs `ha0_ext` −0.4514 / −0.4447 (better,
   separated) but vs `damp50` +0.4190 / +0.4257 and vs `kd_x` +0.5257 / +0.5324 (worse, separated); **accelerate**
   (32 / 31) vs `damp50` +0.0556 / −0.0065 (n.s.); **stop** (v0 > 3 m/s, GT speed < 1 m/s at 2 s): 0 windows at
   stride 40 — absent, not read.
5. **Why, from source (not a measurement of this run):** `W_KAPPA` is 99.5 % of all cost variation and the world
   model's lateral contribution is **1.63e-10** (`refa_v1.py:403`, `D-REFAV1-COST-SURFACE`); the winning plan's
   curvature is the decoded token's canonical value, exactly 0 or ±0.08 1/m (R 12.5 m) on 31/40 p4 windows
   (`refa_v1.py:2755`), while the corpus curves at R 100–1000 m. The proposal head was never trained
   (`w_aux_head 0.0` in the checkpoint's `config.json`). **Nothing in this checkpoint can steer from vision.**

## 5. What this eliminates, and the next lever (Rule Zero)
**Eliminated on checkpoint 21,109:** every zero-training PLANNER lever — cost weights (30 p4 arms), the longitudinal
sustain / jerk-seam levers, inference-seed choice, seed ensembling, and the planner blend. On the full grid the planner
adds nothing over t0 kinematics on either axis.
**The next lever is a TRAINED trajectory readout** (PI requirement R5: one combined 6 s trajectory). Its cheapest
discriminating test — does the frozen trunk carry trajectory information beyond kinematics at all? — was
pre-registered and run the same afternoon: `../2026-09-27-refav1-trunk-probe/` (SPEC + RESULT).
**Named blockers for anything beyond that:** arming `W_VEND` (PI-reserved; it is LONGITUDINAL, and the planner is
already tied with `damp50` on LON); the refav1 TRAIN fp8 features no longer exist on Thor (~303 GB of fp8 features — 4,572 × ~66.2 MB (`T_c × 655,360 + 1,908` B per episode, the encoder's documented size); ⛔ CORRECTED 2026-09-27: I first quoted ~155 GB, which is the v2ep SOURCE-frame cache (34 MB/ep) the encoder reads, not the fp8 cache the trainer reads (class C82: price the file the CONSUMER opens); to rebuild —
a storage decision, PI); a launch-gate profile for `refa_v1_train.py` (the binding gate at `c36b6ddd` has only the
refcv7 profile; the 40-flag inventory is in `products/P4-training-pipelines/2026-09-27-v7f-refav1-state/raw/`).

## 6. Provenance and files
- `raw/run/`: `run_fullgrid.sh` (the exact invocation), `provenance.txt` (md5 of tool, model code, checkpoint, labels,
  cache index), `loncomb3_s{0,1}.log`, `run_fullgrid.out`, `rec_loncomb3_s{0,1}.json` (the arm records), and the two
  dumps `dump_loncomb3_s{0,1}/ep*.npz` (141 each; integer episode ids only).
- `raw/analysis/`: `pd_fullgrid.{md,json,log}` (the pre-registered pairs), `pd_fullgrid_extra.{md,json}`
  (`loncomb3 − damp50`, `b50 − loncomb3`, `damp50 − dampha`), `void_check.py` (+ its output in §0), the blend/floor
  dump builders, and `explore/` (`build_explore.py`, `pd_explore.{md,json}`). The synthetic dumps are NOT banked: each
  is a deterministic function of the two banked dumps and the banked shipped dump (rebuild in seconds; the `kd_self`
  and `w0` controls re-verify exactness).
- Clip ids: none in any banked file (dumps carry integer indices; text files scanned; `manifest.json` lists every file
  with md5 at source and banked).
