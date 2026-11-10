# OTBAND read, both seeds and the replicate: the card's outcome

2026-11-10, Master Mind. Card `PREREG_V8F_OTBAND.md` (frozen, prefix sha256 `fd81c66e`), read as written by `read_otband.py` (reading choices r1-r6 in `README_READ.md`, committed before any OTBAND number was read).

**Tier T1-kin** (self-action open loop; m = 10, causal bucket, dead-reckoned nav, logged scene; the lead follows its logged future), never a closed-scene driving result. **Evidence class MEASURED** (dumps of the banked checkpoints at commit `202849da`, zero training). **Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap` (same windows) and `episode_cluster_bootstrap`, B = 2,000, seed 0, 141 episodes / 2,399 windows per seed. It answers "would another draw of EPISODES say this?" only. "Would another TRAINING run say this?" is the replicate OB-BAND_rep (OB-BAND s0's flags and init, training seed 2) and the two seeds side by side. The planner is deterministic (top-K control: repeated `forward_infer` differs by 0.0 m), so there is no inference-variance question.

Numbers: `out/READ_FULL.json` (bars, rows, ladder), `out/FLOOR_TABLE.json` (every row against the replicate floor), `out/TOPK_FULL.json` (top-3), `out/HEADS_FULL.json` (T's own decisions), `out/DAG_FLOOR.json` (DAG-OACC's own run-to-run floor). Dumps banked privately on HF (`HF_BANK_<tag>.json`, 16 tars, each verified by sha256 after re-download).

## Runs (box 2, commit `202849da`, each with self-check PASS and freeze identity 43 of 43 frozen tensors equal to the init)
| run | init | seed | loss | steps |
|---|---|---|---|---|
| OB-CTL_s0 / s1 | M2b s0 / s1 | 0 / 1 | 30-knot term (`--w-traj 1`) | 9,582 |
| OB-BAND_s0 / s1 | M2b s0 / s1 | 0 / 1 | `--w-op 1/3 --w-tac-traj 2/3 --w-traj 0` | 9,582 |
| OB-BAND_rep | M2b s0 | 2 | as OB-BAND | 9,582 |
All five: `--detach-upper --freeze S adapter cond`, `--steps-mult 2`, no DAgger rows, no acceleration term. OB-BAND-TOK is not trained (its named blocker, stream C's `stok` dump, still stands).

## Controls
* **Parity:** P0 FAIL under its rule (thread drift; rolled track within 3.9 mm of the banked DAG-OACC dump), P0b PASS-EXACT (the baseline's own tool tree equals the commit tree bit for bit). Every compared pair is dumped in this read's environment (amendment 1).
* Every dump's windows and model-free arrays equal the banked DAG-OACC dump's (asserted per load). Checkpoint sha256 of every arm matched its bank manifest before and after its dump.
* Top-3: `traj30_top1` equals the dumped plan exactly; repeated `forward_infer` differs by 0.0 m on all five arms.

## The card's bars: OB-BAND minus OB-CTL, paired, per seed
| family | row | operative s0 | operative s1 | tactical s0 | tactical s1 |
|---|---|---|---|---|---|
| PRIMARY | ADE (m) | **-0.0168 [-0.0275, -0.0098]** | **-0.0149 [-0.0202, -0.0105]** | -0.0235 [-0.1147, +0.0672] | -0.0600 [-0.1539, +0.0306] |
| PRIMARY | FDE (m) | **-0.0393 [-0.0678, -0.0214]** | **-0.0267 [-0.0420, -0.0140]** | -0.0030 [-0.3080, +0.3190] | **-0.5278 [-0.8690, -0.2178]** |
| LONGITUDINAL | speed MAE (m/s) | **-0.0184 [-0.0362, -0.0056]** | **-0.0258 [-0.0335, -0.0184]** | -0.0139 [-0.1126, +0.0842] | **-0.2527 [-0.3612, -0.1435]** |
| LONGITUDINAL | accel MAE (m/s^2) | -0.0450 [-0.1123, +0.0074] | **-0.1057 [-0.1379, -0.0746]** | +0.0864 [-0.1110, +0.3000] | **-0.4279 [-0.6768, -0.1972]** |
| LONGITUDINAL | along-track MAE (m) | **-0.0173 [-0.0283, -0.0102]** | **-0.0157 [-0.0210, -0.0112]** | +0.0081 [-0.0813, +0.0985] | -0.0391 [-0.1331, +0.0554] |
| LONGITUDINAL | DK headway abs error (m) | **-0.0427 [-0.0783, -0.0053]** | +0.0073 [-0.0468, +0.0939] | -0.1449 [-0.2941, +0.0125] | +0.0983 [-0.0644, +0.2458] |
| LONGITUDINAL | DK time-gap abs error (s) | **-0.0245 [-0.0602, -0.0030]** | -0.0012 [-0.0096, +0.0107] | -0.0420 [-0.1016, +0.0048] | +0.0122 [-0.0200, +0.0425] |
| LATERAL | cross-track MAE (m) | -0.0006 [-0.0023, +0.0009] | -0.0006 [-0.0020, +0.0007] | -0.0353 [-0.0877, +0.0152] | -0.0273 [-0.0984, +0.0366] |
| LATERAL | heading MAE (deg) | +0.0311 [+0.0059, +0.0620] | +0.0313 [+0.0003, +0.0681] | +0.3938 [-0.1904, +0.9773] | -0.2212 [-0.9609, +0.4723] |
| LATERAL | yaw-rate MAE (deg/s) | +0.1339 [+0.0172, +0.2912] | +0.0705 [-0.0104, +0.1609] | -1.9620 [-4.5450, +0.2681] | **-4.3512 [-7.6631, -1.4572]** |
| LATERAL | curvature MAE (1/m) | +0.0035 [+0.0001, +0.0078] | +0.0025 [-0.0000, +0.0054] | -0.0088 [-0.0305, +0.0112] | -0.0234 [-0.0621, +0.0109] |

Bold: separated better. LONGITUDINAL DK-VIOL (counts of 743 lead windows, beside): tactical OB-CTL 19 / 16, OB-BAND 12 / 14, DAG-OACC 23 / 22; operative 5-7 in every arm. TACTICAL, plan-derived: TAC-LAT accuracy BAND - CTL **+0.0229 [+0.0017, +0.0455]** / **+0.0371 [+0.0141, +0.0630]**; TAC-LON -0.0050 [-0.0250, +0.0142] / +0.0208 [-0.0029, +0.0421]. STRATEGIC: **PAUSED** (PI item 25), not a pass and not omitted.

| bar | s0 | s1 |
|---|---|---|
| B1 operative non-regression | holds | holds |
| B2 tactical ADE and FDE upper bounds < 0 | fails (ADE +0.067, FDE +0.319) | fails (ADE +0.031; FDE holds) |
| B3 tactical speed and along-track upper bounds <= 0, DK not worse | fails (speed +0.084, along +0.099) | fails (along +0.055, DK headway +0.246 > +0.2, DK time gap +0.043 > +0.03) |
| B4 tactical yaw rate and curvature upper bounds < 0 | fails (yaw +0.268, curvature +0.011) | fails (curvature +0.011; yaw holds) |
| B5 TAC-LAT and TAC-LON lower bounds >= -0.02 | fails (TAC-LON -0.025) | holds |
| B6 B2's rows exceed the replicate floor | no | no |

**Outcome, by the card as written: BAND-COST at both seeds; the seeds agree. B4 fails at both seeds.** TOKEN: BLOCKED (not trained). In words: the band loss improves the operative band at both seeds, by more than the replicate floor on ADE, FDE and along-track, at a cost of 0.03 deg operative heading; on the tactical band no row moves at both seeds, and the bars that fail are upper bounds that do not reach zero (no improvement), never a tactical row separated worse.

## TACTICAL: T's own decisions (`out/HEADS_FULL.json`, no bar; macro-F1 against the labels at the matched logged row, ticks 0, 1, 2)
| pair | T-LAT-F1 all ticks | T-LON-F1 all ticks | goal macro-F1 all ticks |
|---|---|---|---|
| OB-BAND - OB-CTL, s0 | +0.0058 [-0.0281, +0.0356] | +0.0151 [-0.0067, +0.0427] | +0.0110 [-0.0131, +0.0321] |
| OB-BAND - OB-CTL, s1 | +0.0026 [-0.0297, +0.0373] | **+0.0195 [+0.0010, +0.0356]** | +0.0101 [-0.0292, +0.0501] |
| OB-BAND_rep - OB-BAND_s0 (replicate) | **+0.0866 [+0.0413, +0.1255]** | -0.0142 [-0.0439, +0.0231] | **+0.0453 [+0.0039, +0.0737]** |
| OB-CTL - DAG-OACC, s0 | -0.1521 [-0.1925, -0.1129] | -0.0336 [-0.0650, -0.0061] | -0.0251 [-0.0542, +0.0047] |
| OB-CTL - DAG-OACC, s1 | -0.0961 [-0.1390, -0.0554] | -0.0478 [-0.0744, -0.0198] | -0.0040 [-0.0407, +0.0270] |

The band loss leaves T's decisions where they were (one separated row, T-LON at s1, +0.020, far inside the replicate's spread). The replicate pair differs by +0.087 in T-LAT-F1: training-run noise on T's decisions is larger than any band effect on them. Against DAG-OACC, T-LAT-F1 is level at tick 0 (-0.0056 / +0.0088, not separated) and lower at ticks 1-2 (-0.194 / -0.119, separated) in every arm: the gap opens once the model drives on its own actions, which is what the DAgger rows train. Identity control (an arm minus itself): 0 [0, 0] on all 12 rows.

## The replicate floor (r4) decides what any tactical bar can see
OB-BAND_rep minus OB-BAND_s0 (same flags, same init, training seed 2 against 0): operative ADE -0.0074, tactical ADE **-0.2869 [-0.4073, -0.1665]**, tactical FDE -0.8925, tactical yaw-rate -3.18 deg/s, all separated. Two runs of the same recipe differ on tactical ADE by 5 to 12 times the band loss's tactical ADE effect (0.287 against 0.060 at s1 and 0.024 at s0). So the episode bootstrap's separated tactical rows of this card are within one training run's noise: at s1, FDE -0.53 is inside its 0.89 floor, yaw rate -4.35 and speed -0.25 clear their floors only at that seed. Against the floor, the band loss's effect is operative only: ADE, FDE and along-track clear it at both seeds (`out/FLOOR_TABLE.json`). Same mechanism as `H-ESTIM-SEED-1`: the paired bootstrap holds the models fixed.

## Beside, post-hoc, decides nothing: where the gap to DAG-OACC comes from (amendments 2-3)
Both arms trail DAG-OACC (OB-CTL - DAG-OACC tactical ADE +0.734 [+0.505, +1.029] / +0.685 [+0.472, +0.965] m). DAG-OACC differs from M2b in three ways at once; the banked ladder, re-dumped here, splits them (all share M2b's init, 3,000 steps, lr 1e-4, p_gt 0.2, a fully trainable model):

| tactical band, s0 / s1 | step 1: schedule (DAG0 - INIT) | step 2: DAgger rows (DAG - DAG0) | step 3: acceleration term (DAG-OACC - DAG) |
|---|---|---|---|
| ADE (m) | -0.187 / +0.032 | **-0.819 / -0.615** | -0.022 / -0.165 |
| FDE (m) | -0.43 / +0.39 | **-2.92 / -2.01** | +0.24 / -0.57 |
| speed MAE (m/s) | -0.08 / +0.11 | **-0.70 / -0.43** | **-0.21 / -0.50** |
| accel MAE (m/s^2) | -0.17 / -0.05 | -0.05 / +0.64 | **-2.01 / -2.56** |
| cross-track MAE (m) | -0.02 / -0.11 | **-0.44 / -0.46** | +0.03 / -0.07 |
| heading MAE (deg) | -0.49 / +0.31 | **-4.63 / -2.57** | +0.97 / -0.74 |
| yaw-rate MAE (deg/s) | -2.7 / +5.7 | -7.7 / -1.4 | +5.3 / +0.1 |
| TAC-LAT accuracy | +0.011 / -0.045 | **+0.100 / +0.192** | -0.002 / +0.012 |

Bold: separated at both seeds with |delta| above the replicate floor at both (for TAC-LAT: separated at both seeds; no replicate floor is computed for the accuracy rows). Operative band: the acceleration term carries DAG-OACC's operative gain (ADE -0.062 / -0.079, speed -0.170 / -0.207, accel -0.97 / -1.14, all above the floor), and the DAgger rows cost a little there (operative ADE +0.007 / +0.016). DK-VIOL (tactical, of 743): DAG0 20 / 10, DAG 43 / 28, DAG-OACC 23 / 22: the DAgger rows raise distance-keeping violations and the acceleration term lowers them again. **So: the DAgger rows are the tactical lever (about 3 times the replicate floor at s0, 2 times at s1); the acceleration term is the operative and acceleration lever; the band loss adds an operative gain a fifth to a quarter the size of the acceleration term's (ADE -0.017 / -0.015 against -0.062 / -0.079).** The seed-0 note that put the whole gap on DAgger holds for the tactical position and heading rows, not for the operative and acceleration rows.

The floor used in the bold marks above is the band recipe's (OB-BAND_rep), applied to the DAG-recipe steps as a stated assumption. DAG-OACC's own pair is read in the next section.

## Beside: DAG-OACC's own run-to-run spread (amendment 4; `out/DAG_FLOOR.json`, `dag_floor.py`)
Pair per init: the DAG-ORC card's REP (DAG-OACC's argv at training seed 10 / 11) minus DAG-OACC, both dumped here. **The pair differs in two ways, not one:** the training seed, and the trainer file (`v8f_mini_train.py` sha256 `30819a6f` for DAG-OACC, `1c888280` for REP; the later file adds the ego-perturbation-with-DAgger levers, which REP's summary reads back as off: `ego_perturb_with_dagger` False, `ego_perturb_stream` 0). Bank, init and argv otherwise equal. So this is an upper estimate of the seed floor, not the seed floor itself.

| REP - DAG-OACC | operative ADE | tactical ADE | tactical FDE | tactical heading (deg) | tactical yaw-rate (deg/s) | tactical curvature (1/m) | TAC-LAT | TAC-LON |
|---|---|---|---|---|---|---|---|---|
| init s0 (seed 10 vs 0) | +0.0031 [-0.0014, +0.0079] | **+0.6021 [+0.4134, +0.7978]** | **+1.68** | **+3.69** | **+24.45 [+18.25, +31.03]** | **+0.207** | **-0.042** | **-0.095** |
| init s1 (seed 11 vs 1) | -0.0083 (separated) | +0.0215 [-0.0756, +0.1188] | +0.09 | +0.35 | +1.48 | +0.004 | -0.026 (separated) | +0.020 |

Bold: separated, worse for REP. At init s0 the second draw of DAG-OACC's recipe plans a much worse tactical band, above all laterally (yaw-rate MAE about 55 deg/s against 30.4); at init s1 the two draws agree on the tactical band. MEASURED, T1-kin, one pair per init. Consequences: (1) the ladder's DAgger-rows step still exceeds this pair on tactical ADE at both inits (-0.819 against 0.602; -0.615 against 0.022), and the acceleration step exceeds it on acceleration MAE at both; (2) any later card that reads a lever against DAG-OACC needs a same-commit, seed-only replicate of the base, not this pair; (3) the adopted base's tactical lateral plan is fragile across training runs at init s0, which is a finding about the base, not about the band loss.

## Beside: trajectories, plural (R-OUT), one-shot plan of each window's first tick
| arm | best-of-3 minus top-1, operative | best-of-3 minus top-1, tactical | GT pair in top-1 | GT pair in top-3 |
|---|---|---|---|---|
| OB-CTL_s0 | -0.0153 [-0.0171, -0.0136] | -0.3645 [-0.3983, -0.3326] | 0.572 | 0.896 |
| OB-BAND_s0 | -0.0133 | -0.3537 [-0.3852, -0.3224] | 0.580 | 0.901 |
| OB-CTL_s1 | -0.0155 | -0.3904 [-0.4400, -0.3448] | 0.583 | 0.895 |
| OB-BAND_s1 | -0.0151 | -0.3609 [-0.4056, -0.3238] | 0.585 | 0.897 |
| OB-BAND_rep | -0.0120 | -0.3763 [-0.4186, -0.3362] | 0.575 | 0.895 |

Best-of-3 is an oracle pick, not a deployable plan. **The card's top-K trigger fires:** best-of-3 beats top-1 on tactical ADE by 0.35-0.39 m at both seeds, above the replicate floor whether that floor is read on the rolled track (0.287 m) or on this one-shot top-1 (|2.908 - 3.026| = 0.118 m). Committed consequence: a trained multi-trajectory form (a mixture head over the K codes) is proposed to the Master Mind; no head is added in this card.

## What the outcome triggers (Rule Zero; the card's committed lists, read against what was measured)
* BAND-COST's committed list names an operative failure, which did not happen (B1 holds at both seeds): (1) `--w-op` x3, (2) unfreeze `cond`, (3) report the seed spread against the replicate before reading it. Item 3 is this document. Items 1 and 2 aim at a band that already improved; they are not the largest measured lever.
* The measured levers, ranked by effect on the tactical band, where the error sits (DAG-OACC tactical ADE 3.254 / 3.368 m against operative 0.219 / 0.225 m): DAgger rows (-0.82 / -0.62 m, already in DAG-OACC), T's manoeuvre selection (oracle best-of-3 -0.36 m one-shot; the GT pair is in T's top 3 in 90 % of windows and its top 1 in 57 %), the band loss (none measurable on the tactical band).
* Next, in order: (a) the top-K proposal: a selection or mixture head over T's top-K codes, trained on the DAG-OACC recipe, under its own card; (b) the PI-ruled detach (`--detach-upper`) carried into DAG-OACC's recipe with and without the band loss, so the recipe of record satisfies R-T2O and keeps DAG-OACC's gains, under its own card (draft `theory/obd/PREREG_V8F_OBD_DRAFT.md`), read against a same-commit, seed-only replicate of the base, because DAG-OACC's banked pair carries a trainer change and a large spread at init s0. Both need a box-2 slot after the measured-speed runs.
