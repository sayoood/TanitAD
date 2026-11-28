# D-REFCV8-FANCEIL: refcv8's fan ceiling on PhysicalAI eval139

**Verdict: the fan is good and the selector is weak.** refcv8 EMA-48,852 builds a 149-candidate fan. Its best candidate is
**0.084 m** from GT at 2 s and **0.68 m** at 1 to 6 s. The model's own pick is **0.297 to 0.299 m** at 2 s and **2.78 to 2.80 m**
at 1 to 6 s. Paired against refcv7 @ 50,400 on the identical windows, with GT bit-equal:

- refcv8's fan oracle is **0.030 m better** (separated, both inference seeds).
- refcv8's own pick is **0.020 m worse** (separated, both seeds).

The selector therefore lost ground while the fan gained it. About 78 % of the selection loss is longitudinal.

* **Evidence class.** MEASURED. Every number comes from the artifacts listed at the end.
* **Tier.** Oracle-in-fan, random pick and every difference that contains them use GT, so they are **T0 DIAGNOSTIC**
  and never driving performance. `selected` alone is the os pick (T1 stamp, ruling open).
* **Estimator.** FULL-SET mean. Single quantities use the episode-cluster bootstrap; differences use the PAIRED
  episode-cluster bootstrap. Both come from `taniteval/taniteval/ci.py` with n_boot 2000 and seed 0, and
  `fan_control_x8.fan_panel` is imported unchanged.
* **Which variance the CI answers.** It answers only "another draw of EPISODES" (139 clips, 4,754 windows). There is
  **one training seed** (refcv8-r101-s0). Inference-run variance is read from the two fresh DDIM draws (s0 vs s1),
  not from the CI.
* **Code path.** These are the battery's own rolls: `roll_seed_r8_pf.py` → `roll_seed_r7` → patched `refcv3_arm`,
  with `--nav-feed perframe_full`, `R8_NO_POSE_SYNC=1`, `--os-only`, and the same windows JSON (md5 a3a78dba…).
  The rolls ran on Thor (`thor6`, torch 2.13.0+cu130). The checkpoint md5 is b4364029….

## 2 s grid (the E1 grid): 4,754 windows, 139 episodes

| | refcv8 EMA s1 | refcv8 EMA s0 (fresh) | refcv7 50,400 s0 | refcv7 50,400 s1 |
|---|---|---|---|---|
| candidates N | 149 (117 base + 32 extra) | 149 | 117 | 117 |
| **selected** (own pick) | **0.2990** [0.2747, 0.3266] | **0.2973** [0.2728, 0.3248] | 0.2772 | 0.2790 |
| **oracle-in-fan** (T0) | **0.0841** [0.0755, 0.0960] | **0.0847** [0.0761, 0.0965] | 0.1143 | 0.1145 |
| oracle-in-REACH-KEPT (T0) | 0.0841 (148–149 of 149 kept; the oracle is always kept) | 0.0847 | – | – |
| random pick, analytic (T0) | 0.8318 [0.8104, 0.8530] | 0.8318 [0.8105, 0.8533] | 1.5501 | 1.5501 |
| random pick, 12 seeded draws (T0) | 0.8214–0.8399 | 0.8241–0.8406 | 1.5557 (mean) | 1.5557 (mean) |
| oracle − random (T0) | −0.7477 [−0.7658, −0.7291] | −0.7471 [−0.7651, −0.7287] | −1.4358 | −1.4356 |
| random − selected (T0) | +0.5328 [0.5076, 0.5560] | +0.5345 [0.5094, 0.5586] | +1.2728 | +1.2711 |
| selected − oracle (T0) | +0.2150 [0.1965, 0.2339] | +0.2126 [0.1939, 0.2323] | +0.1630 | +0.1645 |
| skill fraction (T0, point) | 0.7125 | 0.7155 | 0.8865 | 0.8854 |
| selected ADE-rank, median / mean | 13 / 22.5 of 149 | 12 / 22.2 | 2 / 7.6 of 117 | 2 / 7.8 |
| selected IS the oracle | 8.9 % | 9.2 % | 30.0 % | 28.8 % |

How to read the skill fraction: refcv8's fan is much **tighter** (random pick 0.83 vs 1.55), so the skill fraction
penalises it for a better fan. The paired gap and the rank below are the clean comparators.

**Paired refcv8 − refcv7, same inference seed, identical windows, GT max diff 0.0 m (T0 except `selected`):**

| | s1 | s0 |
|---|---|---|
| selected | **+0.0200** [0.0082, 0.0318] sep | **+0.0200** [0.0089, 0.0322] sep |
| oracle-in-fan | **−0.0305** [−0.0333, −0.0280] sep | **−0.0296** [−0.0324, −0.0270] sep |
| selected − oracle | **+0.0505** [0.0381, 0.0630] sep | **+0.0496** [0.0377, 0.0626] sep |

These differences are between two trained arms, each with one training seed (H-ESTIM-SEED-1). They replicate across two
inference seeds. They are not replicated across training seeds.

**LON/LAT split of the 2 s error (T0):** `along` is the error along the GT tangent and `cross` is the error along the normal.

| | along (s1/s0) | cross (s1/s0) |
|---|---|---|
| refcv8 selected | 0.254 / 0.253 | 0.0995 / 0.099 |
| refcv8 oracle | 0.060 / 0.061 | 0.044 / 0.044 |
| selected − oracle | **+0.194 / +0.192** | +0.055 / +0.055 |
| refcv7 selected / oracle (matched script) | 0.229 / 0.078 | 0.101 / 0.065 |

So 78 % of refcv8's selection loss is longitudinal.

**TACTICAL proxy (T0) on the 3,668 S6 windows:** each plan gets a heading class at 6 s (left, straight or right, with a
±0.15 rad dead band). This is a proxy for the manoeuvre, not the tactical head's label. Results are s1/s0, from
`TACSPLIT_R8_s*.json`:

| | all S6 windows | GT turns only (n = 1,038) |
|---|---|---|
| some fan candidate has GT's class | **99.92 %** | 99.7 % |
| the selected candidate matches GT's class | 87.9 % / 87.9 % [0.854, 0.903] | **73.4 / 73.0 %** |
| the S6-oracle candidate matches GT's class | 90.2 / 89.8 % | 83.9 / 83.0 % |

The fan almost always contains the right manoeuvre, and the pick misses it on about 27 % of turns. **STRATEGIC:** this
instrument has no route or goal readout, so the family does not apply here. The battery's banked strategic rows stand
as they are.

**Base vs extra candidates (T0):**

- Oracle within the 117 base candidates: 0.1006 / 0.1013. Oracle within the 32 extras: 0.1253 / 0.1256.
- The oracle candidate is an extra on **45.5 %** of windows, but the selector picks an extra on only **11.8–12.0 %**.
- Random pick within the extras: 0.47. Within the base: 0.93.

## S6 grid (1–6 s, model slots 1,3,4,5,6,7): the battery's own 3,668 S6 windows

| | s1 | s0 |
|---|---|---|
| selected | 2.7994 [2.5651, 3.0705] | 2.7757 [2.5427, 3.0470] |
| oracle-in-fan (T0) | **0.6816** [0.6280, 0.7395] | **0.6848** [0.6311, 0.7433] |
| random, analytic (T0) | 6.5017 | 6.5008 |
| selected − oracle (T0) | +2.1178 [1.9133, 2.3475] | +2.0909 [1.8904, 2.3166] |
| skill fraction (T0) | 0.636 | 0.641 |

## Controls

Each control must read a known value.

| control | s1 (alignment seed) | s0 |
|---|---|---|
| C1 sel_idx agreement with the banked `sel_idx_nav_true` (bar ≥ 0.99) | **0.9958** (20 / 4,754 disagree) **PASS** | N/A by construction*, reads 0.751 |
| C2 selected 2 s ADE vs the banked own-pick (tolerance 1e-3) | 0.2990 vs 0.2985, diff 0.00053, **PASS** | N/A*, 0.2973 vs 0.2956 |
| C3 GT appended as candidate 150, so the oracle must be exactly 0 | **0.0 PASS** | **0.0 PASS** |
| C4 fan[sel_idx] == the model's own traj | **0.0 m PASS** | **0.0 m PASS** |
| C5 whole fan vs the bank: terminal heading of all 149 candidates | 99.43 % within 1e-2 rad; reach_keep 100 % equal | 33.8 % (a different draw)* |
| C6 roll GT == bank GT, bit-for-bit | **FAIL as coded**: max 3.8e-6 m (float32, cross-platform) | same |
| per-frame feed census (args_known / token≠clip) | 2802 / 1157, identical to the battery manifest | identical |

Notes on the controls:

- **C6.** The analysis scores against the banked GT, never the roll's GT, so no headline number rides on that 3.8 µm.
- **Cross-hardware drift.** On the 4,734 windows where the selection agrees, the plan differs from the bank by a median
  of 1.0 mm (p99 6.8 mm, max 2.5 cm). This is Thor against the dev-box RTX 4060. The 20 windows where the selection
  disagrees account for the 0.0005 m difference in C2.
- **\*Why s0 cannot be aligned.** The banked s0 was a **full-arm** roll: nav_true, shuffled and flipped are batched as
  3 rows. The base noise is `eps_b = randn((b, 117, …))` (`refc.py:3036`), so b = 3 and b = 1 draw different noise.
  An os-only s0 is therefore a fresh draw by construction. The C5 result discriminates this: s1's fan matches its bank
  on 99.4 % of candidate cells, while s0 matches its bank on only 33.8 %. Aligning s0 would need a full-arm s0 roll,
  which costs about 80 min on Thor and was **not run**. s0 serves here as the inference-seed replicate.

**Input verification** (`INPUT_VERIFY.json`):

- All 15 single-file inputs are md5-equal between the dev box and Thor: ckpt, config, anchors, labels, sidecars,
  v9 eval npz, extrinsics, tau, vis1, pose-sync and defect masks.
- Eval cache: all 141 file sizes are equal; 16 of 141 files were compared by md5 and all 16 are equal. The md5 run was
  stopped to avoid E: IO contention with the locked SHIP-R8 battery. C1 and C5 cover the cache end-to-end by content.
- SAM3 GT: 137 of 137 files are md5-equal.
- The code tarballs are md5-equal on both ends.

## Next lever (Rule Zero): GT-free pickers over the same fan

This section is **post-hoc and exploratory, not a registered claim**. The question it tests: if the selector is the
weak part, can an inference-admissible rule recover some of the headroom?

These pickers use no GT and no extra input. Their ADE uses GT only for scoring. Differences are paired vs the model's
own pick on the full set, with the half-B read in brackets for s1. "Cap-proxy over" is a slot-level proxy for the
PI 10-18 speed cap; the own pick reads about 5 % on it.

| picker | s1 2 s Δ | s0 2 s Δ (**frozen cell**, fresh draw) | S6 Δ s1 / s0 | cap-proxy over |
|---|---|---|---|---|
| `base` = the model's own **pre-E9-graft** core pick (`sel_idx_base`) | −0.0156 [−0.0219, −0.0100] (half-B −0.0139) | **−0.0161** [−0.0226, −0.0106] | −0.188 / −0.169 | 5.0 % / 5.0 % (same as own) |
| argmax core `sel_score` | −0.0236 | −0.0224 | −0.161 / −0.126 | **11.0 / 11.4 %**, so it breaks the cap and is inadmissible as is |
| **MBR in core top-32, T=1, cap-proxy-filtered** | −0.0347 [−0.0430, −0.0268] (half-B −0.0337) | **−0.0308** [−0.0411, −0.0216] | **−0.288 / −0.255** | 4.2 / 4.1 % |

- **The frozen replicate PASSED.** The MBR cell was chosen on s1 episode-half A and frozen in `FROZEN_PICKER_CELL.md`
  (19:51:51 Berlin) before the s0 fan existed. It is separated and below zero on the fresh s0 draw. `base` also passes
  on both seeds.
- **Absolute values.** MBR-capproxy reaches **0.2643 / 0.2664 m** at 2 s; ha0_ext (the E1 reference) is 0.2886. This is
  **not** an E1 read: the E1 bar is defined on compliant windows and on the four families.
- **What stays open before adoption:**
  1. capture the model's **own** cap mask at roll time instead of the slot proxy;
  2. pre-register the picker;
  3. report all four metric families, because MBR may move speed behaviour, and 78 % of the gap is longitudinal.

**Remaining levers and their blockers:**

1. **Drop or retrain the E9 goal-distance graft and the drivable-critic stage.** `base` beats `own` on both seeds.
   This is a PI or Master Mind decision because it touches the hierarchy thesis.
2. **A refcv9 selector trained to rank the extras.** The oracle is an extra 45 % of the time, but the selector picks
   an extra only 12 % of the time. This needs compute and a spec.
3. **A full-arm s0 roll** to close the s0 alignment. About 80 min of Thor time.
4. θ_DEC fan: measured below. It is worse on both the fan and the pick, so it is not a fan-side lever on PhysicalAI.

## θ_DEC (WiSE mod2 DEC, md5 dad4cb5b…), inference seed 1, os-only

This roll uses the same code path, feed and windows (the per-frame census reads 2802 / 1157, identical). There is no
per-frame battery bank for this checkpoint, so C1, C2 and C5 are informative only. C3 and C4 PASS.

| 2 s (T0 except selected) | θ_DEC s1 | paired θ_DEC − EMA s1 |
|---|---|---|
| selected | **0.4513** [0.4235, 0.4777] | **+0.1522** [0.1308, 0.1723] sep |
| oracle-in-fan | 0.1329 [0.1238, 0.1446] | **+0.0488** [0.0444, 0.0531] sep |
| random, analytic | 1.3177 [1.2836, 1.3498] | +0.4859 [0.4658, 0.5054] sep |
| selected − oracle | +0.3184 [0.2970, 0.3383] | +0.1034 [0.0849, 0.1204] sep |
| skill fraction | 0.7313 | – |
| S6 selected / oracle | 3.0717 / 1.0312 | – |
| selected along / cross | 0.423 / 0.091 (oracle 0.105 / 0.059) | – |

On PhysicalAI, θ_DEC degrades **both** the fan (its oracle is 0.049 m worse) **and** the pick (0.152 m worse). It loses
to ha0_ext (0.2886) by 0.163 m. This is the decoder-drift reading again, and it is all longitudinal: along 0.423 vs
EMA 0.254. The frozen MBR-capproxy cell does **not** transfer to θ_DEC: +0.0036 [−0.0081, +0.0156], not separated.
`base` still helps, at −0.0104 [−0.0168, −0.0046]. Artifacts: `FANCEIL_R8_dec_s1.json`, `PICKERS_R8_dec_s1.json`,
`DEC_VS_EMA_PAIRED_s1.json` and `TACSPLIT_R8_dec_s1.json`.

## Artifacts

All paths are under `C:/Users/Admin/refcv8_pf/fanceil/`.

- `FANCEIL_R8.json` combines everything below.
- Per-roll controls and panels: `FANCEIL_R8_{s1,s0,dec_s1}.json`.
- GT-free pickers: `PICKERS_R8_{s1,s0,dec_s1}.json`.
- Paired comparison against refcv7: `R7R8_PAIRED_{s1,s0}.json`.
- θ_DEC paired against EMA: `DEC_VS_EMA_PAIRED_s1.json`. Tactical proxy: `TACSPLIT_R8_{s1,s0}.json`.
- Input checks: `INPUT_VERIFY.json`. The frozen picker cell: `FROZEN_PICKER_CELL.md`.
- Pulled rolls: `runs/<tag>/` holds `dump/` (fan_extras.npz plus the full battery-format dump and decisions),
  `fan_roll_s*.json`, `roll_s*.json` and the log.
- Code: `code/`. Thor originals are under `/home/nvidia/r8ns/fanceil/{code,out}`.
- The refcv7 reference values come from
  `D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/raw/fanctl_x8/ctl_refcv7_50400_fan_s{0,1}.json`.
  The matched rank and LON/LAT numbers for refcv7 were recomputed from its banked fan (`D:/refcv7_eval_kit/fanctl/s{0,1}/dump/fan_extras.npz`).
  That recomputation reproduces its JSON exactly: 0.2790 / 0.1145 / 1.5501 / 0.8854.
