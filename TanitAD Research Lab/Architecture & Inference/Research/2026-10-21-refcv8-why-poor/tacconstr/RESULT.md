# refcv8 EMA-48,852: do the tactical constraint heads carry the headroom? (T0, CPU only, 2026-10-21)

**Verdict.** UNDECIDED FOR ADE, EVIDENCE LEANS CONSTRAINTS. The trained selector rewards candidates that satisfy the predicted heading and progress constraints (gamma 0.456 / 0.122) and does not reward its own decision posterior (beta -0.043 / -0.021); the constraint heads leave a large measured gap to GT, and in lonspeed (the largest single v8F constraint group) the head is at the ego-scalar floor in aggregate; the lateral decision head is already 0.9415 against a 0.9039 constant. But no refcv8-planner oracle has been run, and the planner never reads the 26-dim v9 vectors, so the headroom in metres is NOT measured for refcv8.

**Evidence classes.** Head outputs, labels, controls: MEASURED (ours, artifacts named in section 9). The v8F oracle shares (0.88-0.90 / 0.07-0.12 of 1.05-1.30 m): INHERITED from `RESULT_V8F_TAC_A1.md` (read at commit aa8366d; not re-run, and measured on v8F-mini, a different architecture). Trainer-log numbers: INHERITED, used only as a cross-check.

**Tier.** T0: every head is read on the GT-active class's query (the trainer's own rule). It says how accurate the head is, never how well the car drives.

**Estimator (what the intervals answer).** Episode-cluster bootstrap over the 139 eval episodes, B 2,000, seed 0 (same resample stream as `taniteval.ci._draws`), pooled entry-mean |error| (ratio of sums), paired where two arms share windows. It answers *another draw of episodes at this checkpoint*, with ONE training seed. It does not answer *another training run* (H-ESTIM-SEED-1); a separated row here is necessary, not sufficient. The decision heads read identically at inference seeds 0 and 1 (banked TAC_CONTROLS), so I treat the tactical decoder as deterministic; the constraint heads share that decoder but were captured once (seed 0), which I did not re-verify.

## 0. What ran

| step | status |
|---|---|
| 1. Are the constraints in the banked battery dumps? | **NO.** 1112 npz read (4 trees x 139 episodes x {ep, decisions}), 0 read errors, a same-breath control key present in all 1112, 0 constraint-like keys (`STEP1_DUMP_AUDIT.json`). |
| 1b. Are they anywhere per window? | **YES, in a different dump.** `E:/refcv8_eval_kit/videos/heads_48852_pf/heads_ep000..138.npz` (the post-battery `dump_heads` pass: EMA-48,852, REAL_CHECKPOINT_ON_REAL_CORPUS, perframe_full feed, 139 episodes, 4,754 windows, wall 1,995 s). It carries all five constraint outputs. Its decision heads equal the battery's `lat/lon_pred_nav_true` on **100.0 %** of 4,754 windows, so it is the same checkpoint and feed. **No GPU capture is needed for the heads.** |
| 2. Amendment 2(C) dose-response | **Registered, NOT RUN, NOT CPU-RUNNABLE here** (section 7). Not invented, not approximated. |
| 2b. Constraint MAE vs GT per group, EMA-48,852 | **Done, per window**, against the four controls (sections 2-4). |
| 3. Estimator | as above; n and d printed in every row. |
| 4. Controls | GT-as-prediction, constant = TRAIN mean, lateral agreement re-read: all read their known values (section 6). |

## 1. What refcv8 emits and who reads it (source-literal, launch commit fb47a58; the bundle's blobs equal it)

| output | shape | trained at | read by the planner? |
|---|---|---|---|
| `out_r8_v9_lat_c` / `_lon_c` / `_speed` | [B,8,12] / [B,8,10] / [B,8,4] = **26 dims, the same 26 as v8F's `constraints`** (latcons 12, lonspeed 8, stoplead 6) | `--w-r8-v9-cons 0.05` (Huber beta 0.1, normalised v9 units, GT-active class query only) | **NO** - "they feed the loss, never the planner" (`refcv8_conditioning.py:129-132`) |
| `out_r8_cons_lat` / `_lon` | [B,8,2] each = psi_term/pi, t_onset/6 \| log1p(prog6)/log1p(120), v_end/30 | `--w-r8-cons 0.05` (Huber beta 0.05, GT-active PATH-TAGGED hypothesis) | **YES** - phi of allocated candidates, `FactorizedSelection` (`refc.py:2500-2551, 3834`), `set_r8_force(cons=...)` |

So for refcv8 the v8F finding ("O reads the 26-dim code") has no direct counterpart: the planner reads **four** constraint numbers per hypothesis, and the 26 are auxiliary supervision.

### 1b. What the TRAINED selector weights (MEASURED from the EMA-48,852 checkpoint scalars, `CKPT_SELECTOR_WEIGHTS.json`)

`FactorizedSelection` scores a candidate `beta_lat log p(lat) + beta_lon log p(lon) + gamma_prog sat_prog + gamma_head sat_head`. All four scalars start at exactly 0, so a value away from 0 was moved by training.

| scalar | trained value | acts on | max swing on the score |
|---|---|---|---|
| `gamma_head` | **+0.4562** | -\|terminal heading - predicted psi_term\|/pi, in [-1, 0] | 0.456 |
| `gamma_prog` | **+0.1219** | -\|log1p(path length) - log1p(predicted prog6)\|, down to -4.80 | 0.585 |
| `beta_lat` | -0.0430 | log p(lateral hypothesis) | 0.396 (upper bound; log p floored at -9.21) |
| `beta_lon` | -0.0210 | log p(longitudinal hypothesis) | 0.193 (upper bound) |

The constraint-satisfaction weights are positive (the selector favours candidates that satisfy the predicted heading and progress); the decision-posterior weights are small and **negative** (a higher score for the less probable hypothesis, the opposite of rewarding the decision). The route-checkpoint gate is exactly 0 (`--r8-no-rc`). This says what training moved, **not** how many metres each term is worth; the beta bounds are reached only by near-zero-probability hypotheses. It does say that the consumed channel is the constraint one, which is the necessary condition for constraint accuracy to matter to refcv8's plan.

## 2. v9 constraint heads, per window (139 episodes, 4,754 windows)

Pooled entry-mean |error| in the heads' normalised units, 95 % CI in brackets. `gap closed` = 1 - head / control. `paired head - control` < 0 means the head is better.

| group (d) | entries / windows | **head** | const (TRAIN mean) | class-cond const | ridge (v0, a0) | ridge poly-2 |
|---|---|---|---|---|---|---|
| lat_c (12) | 7,680 / 4,035 | **0.1061 [0.0924, 0.1190]** | 0.2506 [0.2137, 0.2809] | 0.1590 [0.1397, 0.1768] | 0.2339 [0.2013, 0.2605] | 0.2337 [0.2009, 0.2609] |
| lon_c (10) | 16,108 / 3,734 | **0.1999 [0.1711, 0.2354]** | 0.3921 [0.3570, 0.4301] | 0.3380 [0.3034, 0.3765] | 0.2896 [0.2615, 0.3196] | 0.2812 [0.2528, 0.3115] |
| speed (4) | 14,880 / 3,734 | **0.1230 [0.1101, 0.1378]** | 0.3860 [0.3336, 0.4467] | 0.3377 [0.2911, 0.3911] | 0.1142 [0.1037, 0.1250] | 0.1085 [0.0997, 0.1178] |
| latcons (12) | 7,680 / 4,035 | **0.1061 [0.0924, 0.1190]** | 0.2506 [0.2137, 0.2809] | 0.1590 [0.1397, 0.1768] | 0.2339 [0.2013, 0.2605] | 0.2337 [0.2009, 0.2609] |
| lonspeed (8) | 23,624 / 3,734 | **0.1346 [0.1226, 0.1475]** | 0.3740 [0.3284, 0.4265] | 0.3334 [0.2920, 0.3804] | 0.1421 [0.1298, 0.1540] | 0.1359 [0.1255, 0.1460] |
| stoplead (6) | 7,364 / 1,785 | **0.2538 [0.1929, 0.3302]** | 0.4377 [0.3818, 0.5038] | 0.3522 [0.2913, 0.4241] | 0.4084 [0.3508, 0.4728] | 0.3983 [0.3401, 0.4634] |
| ALL26 (26) | 38,668 / 4,431 | **0.1517 [0.1362, 0.1689]** | 0.3616 [0.3318, 0.3947] | 0.3023 [0.2737, 0.3329] | 0.2111 [0.1939, 0.2283] | 0.2053 [0.1884, 0.2220] |

Paired difference, head minus control (the CI is over the same resampled episodes in both arms):

| group | - const | - class-cond const | - ridge (v0, a0) | - ridge poly-2 |
|---|---|---|---|---|
| lat_c | -0.1445 [-0.1701, -0.1135] sep | -0.0529 [-0.0646, -0.0409] sep | -0.1277 [-0.1509, -0.1003] sep | -0.1276 [-0.1516, -0.1002] sep |
| lon_c | -0.1922 [-0.2188, -0.1691] sep | -0.1381 [-0.1640, -0.1150] sep | -0.0897 [-0.1066, -0.0723] sep | -0.0813 [-0.0984, -0.0633] sep |
| speed | -0.2630 [-0.3150, -0.2159] sep | -0.2147 [-0.2600, -0.1737] sep | +0.0087 [-0.0057, +0.0229] ns | +0.0145 [+0.0005, +0.0289] sep |
| latcons | -0.1445 [-0.1701, -0.1135] sep | -0.0529 [-0.0646, -0.0409] sep | -0.1277 [-0.1509, -0.1003] sep | -0.1276 [-0.1516, -0.1002] sep |
| lonspeed | -0.2394 [-0.2876, -0.1983] sep | -0.1987 [-0.2405, -0.1623] sep | -0.0075 [-0.0200, +0.0059] ns | -0.0013 [-0.0140, +0.0119] ns |
| stoplead | -0.1838 [-0.2114, -0.1559] sep | -0.0984 [-0.1285, -0.0708] sep | -0.1546 [-0.1854, -0.1215] sep | -0.1445 [-0.1763, -0.1123] sep |
| ALL26 | -0.2100 [-0.2396, -0.1848] sep | -0.1507 [-0.1778, -0.1277] sep | -0.0594 [-0.0722, -0.0464] sep | -0.0537 [-0.0668, -0.0401] sep |

**Reading.**
* Every group beats the constant, separated (gap closed 42-68 %). That alone is the weakest bar and is met.
* **`speed` (the four speed fields) and `lonspeed` (the eight, the largest single group in the v8F oracle, 0.20-0.26 m) do NOT beat an ego-scalar floor:** `speed` head - ridge +0.0087 [-0.0057, +0.0229] ns; head - poly-2 +0.0145 [+0.0005, +0.0289] sep (the head is slightly WORSE there); `lonspeed` head - ridge -0.0075 [-0.0200, +0.0059] ns, head - poly-2 -0.0013 [-0.0140, +0.0119] ns. A head that does not beat the floor has added nothing beyond the ego state it could read at t0.
* `stoplead` (lead gap / time gap, stop point) and `latcons` beat every control, separated, including the class-conditional constant and both ridges - these are the groups where the head carries scene information.
* The class-conditional constant is the like-for-like floor for `lat_c`: the head is read on the GT class's query, so the lateral-class identity (which dominates the turn fields: const 0.81 vs class-cond 0.24 on `turn_dyaw_deg`) is given for free.

### Exploratory strata (descriptive, NOT a committed bar): where the lonspeed head loses to the floor

Stratified by the GT speed change ahead, |`lon_v_target` - v_now| (labels may use the future for stratification). Pooled |error|, head minus ridge, paired.

| stratum | windows | head | ridge (v0, a0) | head - ridge | head - poly-2 |
|---|---|---|---|---|---|
| speed_change_ge2ms (lonspeed) | 1,704 | 0.1424 [0.1276, 0.1581] | 0.1980 [0.1843, 0.2114] | -0.0556 [-0.0680, -0.0428] sep | -0.0376 [-0.0503, -0.0245] sep |
| steady_lt2ms (lonspeed) | 2,030 | 0.1258 [0.1090, 0.1449] | 0.0780 [0.0725, 0.0844] | +0.0478 [+0.0318, +0.0650] sep | +0.0404 [+0.0230, +0.0595] sep |

The aggregate tie hides two opposite effects. **On windows where the target speed really differs from the current speed (1,704 windows) the head beats the ego floor, separated**; **on steady windows (2,030) it is WORSE than reading v0** (it is noisy exactly where the answer is the current speed). The per-field view agrees: `v_a_ms` (the speed at the band start) head 1.70 m/s vs a linear v0 readout 0.58 m/s.

## 3. The probe-level constraints (the four numbers the planner reads)

Same estimator; TRAIN controls from the 4,855 windows of the 139 TRAIN-episode dump (`trainfit`), ridge floor on ego_t0 = (v0, a_long, yaw rate).

| group | entries | **head** | const (TRAIN mean) | class-cond const | ridge ego3 | ridge ego3 poly-2 |
|---|---|---|---|---|---|---|
| lat(psi_term,t_onset) | 4,153 | **0.0326 [0.0267, 0.0388]** | 0.0761 [0.0610, 0.0920] | 0.0518 [0.0431, 0.0608] | 0.0675 [0.0548, 0.0814] | 0.0660 [0.0531, 0.0798] |
| lon(prog6,v_end) | 7,286 | **0.0490 [0.0445, 0.0536]** | 0.1667 [0.1480, 0.1878] | 0.1389 [0.1233, 0.1558] | 0.0678 [0.0618, 0.0741] | 0.0557 [0.0504, 0.0612] |
| ALL4 | 11,439 | **0.0431 [0.0395, 0.0468]** | 0.1338 [0.1225, 0.1465] | 0.1073 [0.0975, 0.1178] | 0.0677 [0.0605, 0.0753] | 0.0594 [0.0528, 0.0664] |
| psi_term | 3,618 | **0.0260 [0.0215, 0.0312]** | 0.0626 [0.0494, 0.0779] | 0.0360 [0.0300, 0.0425] | 0.0527 [0.0425, 0.0643] | 0.0523 [0.0424, 0.0639] |
| t_onset | 535 | **0.0769 [0.0626, 0.0944]** | 0.1671 [0.1474, 0.1873] | 0.1590 [0.1429, 0.1767] | 0.1675 [0.1483, 0.1875] | 0.1584 [0.1409, 0.1758] |
| prog6 | 3,618 | **0.0404 [0.0357, 0.0451]** | 0.1300 [0.1159, 0.1458] | 0.1064 [0.0957, 0.1173] | 0.0615 [0.0540, 0.0697] | 0.0424 [0.0369, 0.0483] |
| v_end | 3,668 | **0.0575 [0.0509, 0.0644]** | 0.2030 [0.1777, 0.2310] | 0.1710 [0.1487, 0.1949] | 0.0741 [0.0665, 0.0823] | 0.0689 [0.0621, 0.0756] |

Paired head minus control:

| group | - const | - class-cond | - ridge ego3 | - ridge ego3 poly-2 |
|---|---|---|---|---|
| lat(psi_term,t_onset) | -0.0435 [-0.0558, -0.0322] sep | -0.0192 [-0.0242, -0.0143] sep | -0.0349 [-0.0445, -0.0261] sep | -0.0334 [-0.0430, -0.0244] sep |
| lon(prog6,v_end) | -0.1177 [-0.1368, -0.1012] sep | -0.0899 [-0.1050, -0.0761] sep | -0.0188 [-0.0256, -0.0124] sep | -0.0067 [-0.0127, -0.0009] sep |
| ALL4 | -0.0908 [-0.1026, -0.0809] sep | -0.0642 [-0.0738, -0.0558] sep | -0.0246 [-0.0311, -0.0184] sep | -0.0164 [-0.0224, -0.0107] sep |
| psi_term | -0.0366 [-0.0482, -0.0265] sep | -0.0099 [-0.0132, -0.0068] sep | -0.0266 [-0.0345, -0.0198] sep | -0.0263 [-0.0338, -0.0196] sep |
| t_onset | -0.0902 [-0.1121, -0.0676] sep | -0.0821 [-0.0975, -0.0673] sep | -0.0906 [-0.1126, -0.0682] sep | -0.0815 [-0.1068, -0.0555] sep |
| prog6 | -0.0896 [-0.1037, -0.0769] sep | -0.0660 [-0.0765, -0.0566] sep | -0.0211 [-0.0291, -0.0141] sep | -0.0020 [-0.0067, +0.0024] ns |
| v_end | -0.1454 [-0.1705, -0.1230] sep | -0.1134 [-0.1342, -0.0941] sep | -0.0166 [-0.0257, -0.0074] sep | -0.0113 [-0.0198, -0.0028] sep |

At the planner-facing level the head **does** beat the ego floors (ALL4, both ridges, separated). The two the selector actually weights differ: `psi_term` (gamma 0.456) beats every floor, including the class-conditional constant (-0.0099 [-0.0132, -0.0068] sep); `prog6` (gamma 0.122) is at the poly-2 ego floor (-0.0020 [-0.0067, +0.0024] ns), i.e. the progress constraint adds nothing a (v0, a, yaw-rate) readout does not. Its residual error is 0.0431 normalised (v_end: 1.73 m/s).

## 4. Decisions, beside the constraints

| head | agreement | constant | head - constant (paired, 136 / 133 episodes) |
|---|---|---|---|
| lateral (3 exact classes) | 0.9415 (n 3,985) | 0.9039 | +0.0376 [+0.0123, +0.0631] |
| longitudinal (exact classes) | 0.6103 (n 3,672) | 0.3352 | +0.2751 [+0.2165, +0.3354] |

The lateral decision is already 94.2 % against a 90.4 % constant: 5.8 % of windows are wrong, and the head removes 39 % of the constant's errors. The v8F oracle says a perfect decision layer buys 7-12 % of the headroom; nothing here contradicts that being small for refcv8 too, but **no refcv8 decision oracle exists**.

## 5. Cross-check against the trainer's own log (independent channel)

The trainer logs the EMA head's mean |error| every 500 steps on 8 fixed eval batches (128 windows, 80 episodes; `metrics.jsonl`). I rebuilt the targets and masks from source on the cached batches: my supervised-entry counts equal the trainer's logged counts exactly (v9: 25.25 / 57.625 / 53.375 per batch; probe: 14.0 / 25.5), through two different code paths (numpy re-implementation and the source torch function agree on all 4,754 windows too).

| group | trainer-logged head (128 windows) | const on those 128 | per-window head (4,754) | const on 4,754 |
|---|---|---|---|---|
| lat_c | 0.1196 | 0.2087 | 0.1061 | 0.2506 |
| lon_c | 0.1942 | 0.3622 | 0.1999 | 0.3921 |
| speed | 0.1086 | 0.3556 | 0.1230 | 0.3860 |
| probe Huber (trainer loss) | 0.02416 | 0.10076 (the real `constraint_head_loss`) | - | - |

The two windows sets differ (the 128 are a random draw), but the head/constant ratios are of the same order (0.57 / 0.54 / 0.31 vs 0.42 / 0.51 / 0.32; the 128-window set has 202 / 461 / 427 entries), so the heads dump behaves as the head the trainer evaluated.

## 6. Controls that had to read known values

| control | result |
|---|---|
| GT-as-prediction MAE, every group, v9 and probe | **exactly 0** (True, True) |
| constant control = the TRAIN mean | True (asserted against an independently computed per-field mean over 909,651 release rows) |
| lateral agreement re-read from the heads dump | **0.9415** vs banked 0.9415 (n 3,985); constant 0.9039 vs banked 0.9039; also re-read from the battery dumps at seeds 0 and 1: 0.9415 and +0.0376 [+0.0123, +0.0631], identical to the banked JSON |
| longitudinal agreement re-read | 0.6103 vs banked 0.6103 |
| heads dump == battery model | `out_lat/lon_logits_tac` argmax equals `lat/lon_pred_nav_true` on 100.0 % / 100.0 % of 4,754 windows |
| target derivation | numpy targets/masks equal the SOURCE torch `v9_constraint_targets` on all 4,754 windows: True |
| join | window -> v9 row k = ws + 2; |dump v0 - release v_now_ms| mean 0.0061 m/s; class labels agree except where the battery masked a window (all 50 + 62 disagreements are battery -100 vs a release class) |
| mutants | const + 1 reads worse than const: True; GT + N(0, 0.05) reads 0.0397 (~0.04 expected for sigma 0.05) |
| mispaired control (head vs the GT of a window 1,000 positions away) | ALL26 head reads **0.4614** vs 0.1517 true-pair (3.04x), and not better than the constant on the same mispairing (0.3616): the instrument is sensitive to the pairing, so a silent join defect would not pass |

## 7. Amendment 2(C), the registered dose-response: why it did not run

PREREG_V8F_TAC Amendment 2 (C): `--tac-groups constraints --tac-alpha {0.25, 0.5, 0.75}`, class read on M2b at alpha 0.5 (PAYS if r >= 0.4, THRESHOLD if r < 0.2). It is registered and stamped **NOT RUN, box 2 awaits the PI's go**. It needs O's forward at the intermediate codes; the four checkpoints and the eval feature cache are on box 2 only (`D:/tanitad/runs/v8f_mini_full/*/ckpt_last.pt`; none on this machine - checked), and the banked outputs hold only alpha 0 and alpha 1, so an interpolation of banked outputs would be an invented arm. It also reads a **different model** (v8F-mini's O consumes the whole 60-dim code), so even when run it speaks to whether *partial constraint accuracy pays through that O*, not to refcv8's planner.

## 8. What this does and does not say, and the next levers (Rule Zero: the verdict is a waypoint)

**Says (MEASURED).** (a) refcv8's constraint heads are far from GT in the dimensions that matter, and the largest single v8F constraint group (lonspeed) is at the ego-scalar floor in aggregate, with a clean split: informative when speed must change, noisy when it should not. (b) `stoplead` and `latcons` carry scene information. (c) The lateral decision head is near its ceiling. (d) The 26 v9 vectors are never read by the planner; the four probe constraints are, they beat the ego floors (heading clearly, progress only to the poly-2 floor), and the trained selector weights the constraint terms (+0.456 / +0.122) while giving the decision posteriors small negative weights (-0.043 / -0.021).

**Does not say.** How many metres of ADE a better constraint head buys refcv8. v8F's 0.9 m is for another architecture whose O reads all 26 dims, at T0, with one-group mixes off-distribution. No second training seed exists for these heads, so a separated paired CI here is necessary, not sufficient. The 'floor' comparison is a linear (and a poly-2) function class, and yaw rate is not in the v9 release so the 26-dim ridge uses (v0, a0) only.

**Next levers, ranked by what they would settle:**

1. **The refcv8-native oracle (settles the question).** `RefCV3Model.set_r8_force` already exists as an EVAL seam: `cons=` forces the constraint of the (own or forced) hypothesis, `lat3/lon6=` forces the decision. Arms on the 4,754 windows: own / GT-4-dim-constraints-at-own-hypothesis / GT-hypothesis (grouped by class, the seam takes one class per batch) / both. Cost: one forward pass per arm; the heads pass measured **1,995 s for 4,754 windows** (`heads.log`), so about 2.2 h GPU for four arms as an upper bound (ESTIMATED from that one pass, which also wrote 20 MB of perception tensors per episode). **Blocked on a GPU**: the dev-box GPU runs the locked battery and Thor is reserved. Unblock: a free GPU slot, or the PI's go on one.
2. **Amendment 2(C) as registered.** 40 evals, ~15 min on box 2. Blocked on the PI's box-2 go; or ship the four checkpoints + feature cache here and run it on CPU.
3. **A v0-residual speed head (hypothesis, not a result).** The steady-window loss and the `v_a` readout error say the speed head would gain from predicting target - v0 instead of target. It is an architecture change: PI go, own prereg, launch gate. Nothing here trains it.
4. **Wire the 26 vectors to the planner, or drop their weight claim.** They currently cost 0.05 of the loss and influence the plan only through the shared trunk. PI decision.

## 9. Artifacts and provenance

| item | path |
|---|---|
| headline JSON (merged) | `C:/Users/Admin/refcv8_pf/tacconstr/TACCONSTR_R8.json` |
| this report | `C:/Users/Admin/refcv8_pf/tacconstr/RESULT.md` |
| step 1 audit | `C:/Users/Admin/refcv8_pf/tacconstr/STEP1_DUMP_AUDIT.json` |
| per-window v9 + probe + strata | `C:/Users/Admin/refcv8_pf/tacconstr/TACCONSTR_R8_heads_perwindow.json` |
| 128-window trainer-log cross-check (v9) | `C:/Users/Admin/refcv8_pf/tacconstr/TACCONSTR_R8_v9heads.json` |
| 128-window trainer-log cross-check (probe) | `C:/Users/Admin/refcv8_pf/tacconstr/TACCONSTR_R8_probe.json` |
| decision-head control re-read | `C:/Users/Admin/refcv8_pf/tacconstr/CONTROL_lat_lon_agreement_reread.json` |
| code | `C:/Users/Admin/refcv8_pf/tacconstr/code/` |

Inputs (all read-only): heads dump `E:/refcv8_eval_kit/videos/heads_48852_pf/`; battery dumps `C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/dump_s{0,1}`; v9 eval release `C:/Users/Admin/qland/work/refcv8_hf_v9/release/v9_labels_eval139.npz` (md5 6b5c7f20...); v9 train release `C:/Users/Admin/qland/hold_v9train4572/work/release_train4572/v9_labels_train.npz` (md5 83e30bd3..., the `--r8-v9-md5` of the launch); G0 batch cache `C:/Users/Admin/refcv8_eval_archive/g0cache/step40000/`; trainer log `C:/Users/Admin/qland/work/watch8/metrics.jsonl`; TRAIN-139 dumps `C:/Users/Admin/refcv8_pf/trainfit/step30000ema_train139_pf`; source `C:/Users/Admin/refcv8_pf/navsim_ckpt/thor_bundle/stage/tree/stack` (blobs of `refcv8_conditioning.py` and `refcv8_train.py` equal fb47a58).

**Paths in the brief that do not exist:** `RESULT_V8F_TAC_A1.md` and commit aa8366d are not in the D:/E: working tree (HEAD 37645fcc); they are in the clone `C:/Users/Admin/lab019` (also objects in tanitad-push, wt-be, tanitad-wt-join3d, tanitad-wt-percep-v6, scratch_hjunits). `D:/Projects/TanitAD/FlyWheels/.../2026-10-20-refcv8-perframe-final/` is also absent; the TAC_CONTROLS file read here is `C:/Users/Admin/refcv8_pf/post48852/TAC_CONTROLS_step48852ema_pf.json`.
