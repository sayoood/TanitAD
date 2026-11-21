# PREREG_V8F_ACTRSP (FROZEN 2026-11-16 by the MM, NOT RUN): train the ladder world model's action response directly, then read the T0 planner rows again

**Status.** DRAFT. When this was written, no ACTRSP code, run or scored window existed. Box-2 training needs: the Master Mind's freeze line in the Log, the code of `REACHABILITY_ACTRSP.md` landed on ONE commit (as a diff under `raw/`, the ACTSENS rule F2), a PASS token per run, build control B-CAL read (section 6), and the box-2 go. Nothing here was launched.

**Tier.** Every planner row is **T0**: the goal is the encoded true-future latent and the planner is frozen with zero planner training. T0 rows are a world-model diagnostic and never measure driving performance. HOLD is the T1-kin reference.

**Code base (MEASURED, 2026-11-16).** The lever attaches to the ACTSENS commit `81cee6c1` (tip of `origin/agent/v8f-actsens-20261115`, `git ls-remote`), whose trainer `stack/scripts/v8f_mini_train.py` (sha256 `f20e345f…`, 3,589 lines) carries `--wm-only` and the ladder levers. It does NOT attach to the tip of `agent/arch-inf-20260803` (`8a716bb9`): that trainer has 0 hits for `wm-only` (control: 92 hits for `add_argument`), so the R35 recipe cannot be built there. `81cee6c1` is not an ancestor of `8a716bb9` (`git merge-base --is-ancestor`: no). All six ACTSENS arms trained on `81cee6c1` (`RUN.done` `commit` field).

## 1 Why (measured; each number carries its artifact)

* **ACTSENS read NULL for IDM and HARM for DIV** (T0, oracle goal; `F:/TanitAD-scratch/actsens/read/verdict/verdict_actsens.md`, MEASURED). DIV on chassis 1, LONGITUDINAL 0-2 s `acc_mae` +0.1325 [+0.070, +0.197] against R35_s1, beyond its training floor 0.0638 (`VERDICT_ACTSENS_FAMILIES.md`).
* **The training floor is large and the levers did not replicate** (`VERDICT_ACTSENS_FAMILIES.md`, MEASURED). NULL_s3 − R35_s1 at 0-2 s `acc_mae` +0.0638 [+0.004, +0.126], separated with zero levers moved. IDM_s0 has 4 cells beyond the floor and IDM_s1 has 0; DIV_s0 is better on 3 LONGITUDINAL cells and DIV_s1 worse on 7 LATERAL cells.
* **No trained ladder arm has a positive level-1 lateral gain** (IA-3, L1 2.0 s, gain = decoded effect / physical effect, 1.0 = right size; MEASURED: R35 from `ladder_read/RESULT_LADDER_READ.json` `runs/R35_s{0,1}/beside/ia3/1|10`, the rest from `F:/TanitAD-scratch/actsens/read/out/<arm>/ia_read.json` `beside.ia3`):

| arm | chassis | L1 2 s lat | L1 2 s lon | L2 6 s lat | L3 12 s lat |
|---|---|---|---|---|---|
| R35_s0 | 0 | −0.0385 [−0.049, −0.028] | 0.384 | 0.142 | 0.317 |
| IDM_s0 | 0 | −0.042 [−0.051, −0.033] | 0.333 | 0.182 | 0.260 |
| DIV_s0 | 0 | −0.047 [−0.059, −0.036] | 0.418 | 0.127 | 0.329 |
| NULL_s2 | 0 | −0.039 [−0.048, −0.031] | 0.318 | 0.179 | 0.275 |
| R35_s1 | 1 | −0.0359 [−0.045, −0.027] | 0.301 | (not pulled) | (not pulled) |
| IDM_s1 | 1 | −0.030 [−0.040, −0.020] | 0.312 | 0.168 | 0.294 |
| DIV_s1 | 1 | −0.061 [−0.070, −0.053] | 0.314 | 0.177 | 0.408 |
| NULL_s3 | 1 | −0.046 [−0.055, −0.037] | 0.321 | 0.126 | 0.282 |

  The ladder read's other arms read the same sign (C0_s0 −0.074, C0_s1 −0.046, R35H_s0 −0.010, R35H_s1 −0.027; `RESULT_LADDER_READ.json`). Neither lever moved this cell toward a positive gain. Against R35 the shifts are IDM −0.0038 / +0.0060 and DIV −0.0086 / −0.0254, with seed spreads |NULL − R35| of 0.0010 / 0.0104 (chassis 0 / 1). The only shifts beyond the spread go further negative.
* **The single-state probe that IA-3 reads at L1 cannot see yaw rate or acceleration** (IA-1, 1 − NMSE of a cross-fitted linear probe on one L1 state; MEASURED, `RESULT_LADDER_READ.json` and `ia_read.json` `beside.ia1`): R35_s0 / R35_s1 yaw rate −0.085 [−0.286, +0.060] / −0.063 [−0.276, +0.088]; acceleration −0.317 / −0.364; speed 0.675 [0.590, 0.743]. NULL_s2 / NULL_s3 yaw rate −0.177 / −0.059.
* **A state PAIR does carry them.** The level-1 IDM head (an MLP on `[z_t, z_{t+1}]`) explains R² 0.489 / 0.514 of acceleration and 0.637 / 0.629 of yaw rate on R35_s0 / R35_s1 (2,397 TRAIN windows, 31,575 transitions; `actsens/PROBE_ACTSENS_RESULT.json` `partA.*.idm_L1`, MEASURED).
* **The level-1 forward response is a tenth of the real step** (same file, `fwd_sens_L1_ctx1`, MEASURED): a +1 sd action step moves the predicted next latent by 8.7 % / 8.8 % (acceleration) and 10.3 % / 9.8 % (yaw rate) of the real one-step latent change.

**Reading (HYPOTHESIS, with its arithmetic).** For a linear readout whose real-data skill is R², a WM with the TRUE action response is read at a gain of about R², not 1.0 (the readout's slope on its own target equals R² for a calibrated regressor). At L1 the single-state yaw-rate probe has R² ≤ 0, so IA-3 would read a lateral gain near 0 even on a WM whose lateral response is right. The level-1 lateral "never ANSWERS" is therefore partly an instrument limit, and partly a response defect that the forward-response numbers show directly. Both halves shape this card: the training readout is a PAIR readout, and the mechanism read adds a calibrated pair probe (IA-3P, section 4.1).

## 2 Choice of objective (the two candidates of ACTSENS section 6, ACTSENS-NULL row)

**(a) DIVRS counterfactual target as written: pairs of windows with opposite lateral actions, a loss on the difference of their predicted states.** For two windows A and B, `||(ẑ_A − ẑ_B) − (z_A − z_B)||² = ||e_A − e_B||²` with `e` the prediction residual. Its minimiser is the per-window prediction loss's own minimiser, so on two DIFFERENT states it only re-weights residuals that the WM already sees; it gives no information about how one state answers two actions. Its sampling-form neighbour DIV did not move the L1 lateral gain toward the right sign (table above: −0.047 / −0.061 against NULL −0.039 / −0.046) and read HARM on chassis 1. On the ladder, two windows of one clip share at least 77 of 91 frames (ACTSENS card section 2, from R35's `summary.json`), so "same scene type" is a clip-level match.

**(b) Gain matching toward IA-3 gain 1.0.** This carries the information (a) lacks: a physical prior on how ONE state answers TWO actions. On logged driving the action is close to a function of the state history, so the factual loss leaves dF/da weakly constrained; a prior is the only source of counterfactual supervision without a simulator. But the readout matters: IA-3's single-state probes have no L1 skill for the two action channels (section 1), so a target defined through them would be unreachable for yaw rate.

**Chosen: (b), in the counterfactual form of (a), read through a pair readout.** The DIVRS "difference of predicted states under opposite actions" is kept, but on the SAME state with synthetic opposite actions `a ± δ`, and the difference is read by the level-1 IDM head (the pair readout with R² 0.49-0.64) and matched to gain 1.0 relative to that readout's slope on real transitions. Name: the **action-response term**, `rsp/L1`.

### 2.1 What the term computes (level 1 only; levels 2 and 3 have no term)
Per batch (B = 64 windows of 91 frames: `rollout_valid/L1` 11,200 = 64 × 175 at step 19,164, `ladder_R35_s0/train_log.jsonl`, MEASURED), on `enc = HJEPAStack.encode(z1, a)`:
1. `z = enc.z[0]` [B, 91, 128] and `a = enc.idm_target[0]` [B, 90, 2], the level-1 actions in WM units (acceleration / 2.0, yaw rate / 0.4; `v8f_mini.py:164-166`, `:761`).
2. **Anchors** i ∈ {0, 10, …, 80} (9 per window), **K = 10** steps (2.0 s, the deciding band). Every step's pair lies in the window, so no step is masked.
3. **Anchor state detached:** `z0 = z[:, i].detach()`. The term trains the forward response, never the latent geometry.
4. **Four contrasts**, IA-3's structure (`lonp, lonm, latp, latm`, `ladder_read/lr_lib.py:50`) at a smaller size: channel c ∈ {acc, yaw}, `a_c ± δ_c` over all K steps, **δ = (0.42, 0.24) WM units = 1 sd** of each channel on R35's TRAIN windows (`PROBE_ACTSENS_RESULT.json` `action_sd_wm_units` 0.4203 / 0.2369, MEASURED) = 0.84 m/s² and 0.096 rad/s. One sd keeps the contrasted transitions inside the readout's training range; IA-3 reads at 1.0 m/s² and 0.25 rad/s (2.64 sd on yaw), so train and read contrasts differ.
5. **Rollouts in the planning form:** `ẑ^{c±} = stack.rollout(1, z0, a[:, i:i+K] ± δ_c e_c, context=1)` [B, 9, K, 128], all four contrasts in one batched call. Context 1 is the planner's (`plan-ctx 1`, phase 1).
6. **Readout with frozen parameters:** `Î(x, y)` = `idm_heads[0]` evaluated with DETACHED parameters (`torch.func.functional_call`) on the pairs `(ẑ_{h−1}, ẑ_h)`, h = 1..K, `ẑ_0 = z0`. The IDM head keeps training on real pairs only, through its unchanged `idm/L1` term.
7. **Readout slope on real transitions** (detached, float32): `s_c = max(Cov(Î_c(z_t, z_{t+1}), a_{t,c}) / Var(a_{t,c}), 0.1)` over the batch's B × 90 real level-1 transitions. This is the readout's own attenuation; the floor 0.1 keeps the term finite and live while the IDM is untrained (G-LIVE runs at step 0).
8. **Gain matrix** per (sample, anchor, step): `G[c′, c] = (Î_{c′}(ẑ^{c+}) − Î_{c′}(ẑ^{c−})) / (2 δ_c s_{c′})`. Target: the identity (diagonal 1: the predicted transition carries the commanded action exactly as strongly as a real one, through the same readout; off-diagonal 0: a yaw command does not change the read acceleration, and the reverse).
9. **Loss:** `rsp/L1 = mean over B × 9 × K × 4 entries of (G − I)²`; `level/L1 += λ · rsp/L1`.
10. **Logged, detached:** `rsp/L1`, the four mean gains `rsp_g_aa`, `rsp_g_ww`, `rsp_g_aw`, `rsp_g_wa` and the two slopes `rsp_s_a`, `rsp_s_w` (all `/L1`), through the existing `wm/<term>` path into `train_log.jsonl`.

* **Gradient path.** The term reaches `stack.predictors.0` and `stack.act_embed` and nothing else: the readout's parameters and the anchor state are detached. `act_embed` also feeds the macro-action encoders of levels 2 and 3 (`hjepa.py:684`), so the term reaches the upper levels' inputs indirectly; that is the WM's own action entry, the one ACTSENS control C8 zeroes.
* **No new parameter and no random draw.** The term adds no module (the readout is the existing IDM head) and uses no RNG (fixed anchors and contrasts). So a flag-on build at seed k has the SAME WM initialisation and the SAME data order as the flag-off build at seed k (control C4). That makes every lever arm exactly paired with a banked flag-off arm.
* **Admissibility.** The term is a training loss on the world model. At inference the WM reads exactly what R35 reads (frozen adapter, vision only); the IDM readout runs only inside the loss. No situation-classifier output enters anything.

### 2.2 Dose: λ = 1.0 (fixed here, never tuned on scored windows)
* At R35's end (`ladder_R35_s0/train_log.jsonl`, step 19,164, MEASURED) the level-1 factual prediction terms are `pred/L1` 0.0503 + `rollout/L1` 0.1180 = 0.168; the IDM term is 100 × 0.0460 = 4.60; `level/L1` 5.1024.
* On a WM whose response is a tenth of the real step (section 1), the diagonal gains sit near 0.1 and `rsp/L1` near (0.81 + 0.81 + 0 + 0) / 4 ≈ 0.41 (ESTIMATED). At λ 1 that is 2.4× the factual L1 prediction terms and 0.09× the L1 IDM term: strong enough to move the response, an order below the term that dominates level 1.
* Brackets, named now: **HARM ⇒ λ 0.3. NULL with the term not trained down** (end `rsp/L1` > 0.5 × its first logged value) **⇒ λ 3.0.**
* The flag: `--wm-rsp-w W` (float, default **0.0 = off = the ACTSENS / ladder build, bit for bit**). Every ACTRSP lever argv NAMES `--wm-rsp-w 1.0`; every NULL argv leaves it absent.

## 3 Arms (WM-only on box 2; adapter, readers and planner frozen at M2b s{k}, exactly as R35)

Base argv: `C:/Users/Admin/t2v8f_pulls/actsens_NULL_s2/argv.txt` (chassis 0) and `actsens_NULL_s3/argv.txt` (chassis 1), each = R35_s{k}'s recipe with only the seed changed: `--wm-k 1 --wm-strides 3 5 --wm-rollout-k 10 10 6 --wm-rollout-stride 5 5 1 --wm-ctx 3 3 3 --wm-rollout-w 1.0 --wm-only --wm-windows wmonly --wm-win-stride 1 --wm-pooled on --batch 64 --steps 19164 --steps-mult 4 --kin-mode yaw --exclude 540:600 --hmax 91 --arm M2b`, with `--init-ckpt` and `--wm-zcache` of chassis M2b_s{k}, and `--out` under `D:/tanitad/runs/v8f_actrsp/<tag>`. Every new arm runs on ONE commit, the ACTRSP commit, with the flag default-off.

| order | tag | chassis | seed | argv change from the base | pairs with (same init, same data order) | purpose |
|---|---|---|---|---|---|---|
| 0 | PILOT | M2b_s0 | 0 | RSP_s0's argv + `--steps 200` | — | s/step, GPU memory, `rsp/L1` trajectory |
| 1 | RSP_s0 | M2b_s0 | 0 | seed 0, `+ --wm-rsp-w 1.0` | R35_s0 (banked) | lever, chassis 0, seed A |
| 2 | RSP_s2 | M2b_s0 | 2 | seed 2, `+ --wm-rsp-w 1.0` | NULL_s2 (banked) | lever, chassis 0, seed B |
| 3 | NULL_s4 | M2b_s0 | 4 | seed 4 only | — | third flag-off arm, chassis 0 |
| 4 | RSP_s1 | M2b_s1 | 1 | seed 1, `+ --wm-rsp-w 1.0` | R35_s1 (banked) | lever, chassis 1, seed A |
| 5 | RSP_s3 | M2b_s1 | 3 | seed 3, `+ --wm-rsp-w 1.0` | NULL_s3 (banked) | lever, chassis 1, seed B |
| 6 | NULL_s5 | M2b_s1 | 5 | seed 5 only | — | third flag-off arm, chassis 1 |

* **Two training seeds per chassis for the lever, each exactly paired.** ACTSENS had one lever seed per chassis, so a seed effect and a chassis effect could not be told apart; here each chassis carries two lever seeds and each lever seed is paired with the flag-off arm of the same seed. Two of the four baselines (NULL_s2, NULL_s3) already exist and are not retrained.
* **Three flag-off arms per chassis for the training floor** (R35_s{k}, NULL_s{k+2}, NULL_s{k+4}): three pairwise differences per row instead of ACTSENS's one. Same-seed replicates read a floor of exactly 0 (box-2 WM-only training is bit-identical at a fixed seed: R35b_s0's weights equal R35_s0's, `ladder_read/VERDICT_LADDER.md`, INHERITED from the ACTSENS card), so the NULL arms change the seed.
* **Run order.** If the chain dies after order 3, chassis 0 holds a complete contrast (two lever seeds, three floor arms).

## 4 Read

### 4.1 Mechanism read (per lever arm against its paired baseline)
* **IA-3, code of record** (`ladder_read/run_ladder_read.py` through `actsens/read/ia_read.py`): gain lon, gain lat and cosine at L1 2.0 s, L2 6.0 s, L3 12.0 s. Reported on every arm; at L1 lateral it reads through a probe with no skill (section 1).
* **IA-3P, the pair probe (new; `READ_PLAN.md` item N1):** IA-3's contrasts and rollouts, decoded by a cross-fitted LINEAR probe on `[z_{h−1}, z_h]` → (acceleration, yaw rate), fitted on the real level-1 pairs of the 141 eval clips with IA-3's folds, at context 1 and at the built context. **Calibrated gain** = the raw gain ÷ the probe's own slope on real pairs (the attenuation of section 1). Cells: L1 h = 1.0 s and 2.0 s.
* **MECH-MOVED** = at both chassis and for both lever seeds, the arm's IA-3P calibrated L1 2.0 s lateral gain (context 1) or longitudinal gain lies above its paired baseline's, the CIs are disjoint, and the point gap exceeds F_T,k on that cell.
* **Beside, no bar:**
  * **TARGET-READ:** the arm's own cycle gain G (section 2.1, its own IDM, its own slope) on the 141 eval clips, and the `rsp/L1` trajectory from `train_log.jsonl`. This is the trained quantity, exposed to its own readout, and it never decides.
  * `probe_actsens.py` Part A on each arm: L1 forward response per channel, L1 IDM R² per channel.
  * The off-image diagnostic (`hwm_plan/decode_check/decode_check.py` through `actsens/read/decode_check_arm.py`): median nearest-neighbour distance of planned macros to the TRAIN macro cloud over the held-out in-image distance (R35: 6.77 / 6.79), share of decodes at the bounds (R35: 94 %).
* **Health (feeds HARM):** BAND-MET from the ladder read (IA-0 skill against Copy-Last at the L1 2 s, L2 6 s, L3 12 s band ends) on every arm.

### 4.2 Deciding read (HWM-PLAN phase-1 rows on each arm)
* **Planning, unchanged from ACTSENS section 4.2:** `hwm_plan/run_hwm.py` `windows` / `plan` with phase 1's levers (`--seed-mode prior --lam 0`, T3 2, plan-ctx 1, `stitch decode`, vocab off, constraint weight 0), arms `hwm_oracle` and `flat_oracle`, inference seeds 0 and 1, the arm's own checkpoint and own TRAIN latents, torch intra-op threads 1.
* **Contrast:** `F:/TanitAD-scratch/hwm_plan/phase2_l1full/contrast_l1full.py` with `--b` / `--rep-b` = the PAIRED baseline's plans: R35_s{k}'s from the HF phase-1 tar for seeds 0 and 1, NULL_s2's / NULL_s3's banked plans (`actsens/read/out/NULL_s{2,3}/win/plans_*_s{0,1}.npz`) for seeds 2 and 3. `--windows` = R35's phase-1 windows dir after control C6.
* **Deciding quantities:** band **0-2 s** (level 1's own plan, no decode): LONGITUDINAL `acc_mae` and LATERAL `yaw_mae`.
  * Δ_{k,j} = HWM-ORACLE(RSP at chassis k, seed j) − HWM-ORACLE(its pair), paired, inference seed 0; j runs over the chassis' two lever seeds.
  * Δ̄_k = the mean of the two, bootstrapped as one paired statistic over the same windows (`READ_PLAN.md` item N3).
  * **BETTER** on a quantity = at both chassis: the CI of Δ̄_k lies wholly below 0, |Δ̄_k| exceeds Floor_k, and both Δ_{k,j} are below 0.
  * **WORSE** = at either chassis: the CI of Δ̄_k lies wholly above 0 and Δ̄_k exceeds Floor_k.
* **Absolute rows:** HWM-ORACLE(arm) − HOLD, per lever arm, 0-2 s and full 6 s.
* **Same rules, reported and not deciding:** FLAT-ORACLE rows, the 2-6 s band and the full 6 s band (2-6 s passes through the off-image decode, DECODE-WEAK).

### 4.3 Families (per band: 0-2 s, 2-6 s, full 6 s; each family separately, never pooled)
* **LONGITUDINAL:** `acc_mae`, `speed_err`, and distance-keeping `hw_err` / `tg_err` / `ttc_err` on windows with a lead (n printed per cell).
* **LATERAL:** `yaw_mae`, `head_err`, `curv_err`, `xtrack`, `ade`.
* **TACTICAL:** plan-derived (`hwm_eval.TACTICAL_SCOPE`): `tac_man_dis`, `tac_lat_dis`, `tac_lon_dis` per band. Goal and anchor selection: n = 0, NOT COMPUTABLE (the planner has no decision output).
* **STRATEGIC** (reported in full; the item 25 pause ended by PI ruling 2026-11-16). Cells from the NVIDIA navigation text (`strategic_cells.py`, `strategic_text.py`, `hwm_eval.py` in `raw/2026-11-16-v8f-strategic-audit/reader/src/`; the text row is joined at the window's NOW, offset 0, MEASURED in `reader/out/STRATEGIC_TEXT_OFFSET_CHECK.json`), full 6 s band only, the 0-2 s and 2-6 s bands empty by construction:
  * `s_adh` route adherence (announced L/R turn inside the plan's arc; signed-degree twin beside it) and `s_false` false turn, each with the logged-path rate beside it (S_FALSE is read only as plan minus logged). n on the 994 SCORE windows (910 carry a text row): S_ADH 8-16 scorable windows in 3-6 episodes per arm and planner, S_FALSE about 850 windows in 69-70 episodes, logged-path S_FALSE rate 0.131 on 854 windows / 69 episodes (MEASURED on the ACTSENS arms, `reader/out/STRATEGIC_ACTSENS_DEMO.json`): S_ADH is NOT decision-grade here and is reported with its n.
  * HWM-ORACLE plans: the goal is the encoded true future, so these cells measure the world model's use of an oracle (T0 diagnostic), not a strategic decision.
  * `S_CKPT`, strategic intent 6-30 s, derived goals, detected info: NOT COMPUTABLE on these arms, each with its reason (no S head in the HWM planner; no output beyond 6 s; no derived-goal or context output) and the would-have n. A structural n/a, never a pass.
  * COMPLETE_M 20 m, theta 25 deg (SLIGHT 10 deg) and the 30 m margin are HYPOTHESIS values, fixed here before any ACTRSP window is scored; one sensitivity row at COMPLETE_M 10 and 30. No ACTRSP outcome in section 7 is decided on a STRATEGIC cell.

### 4.4 Estimator and floors (every reported interval names the question it answers)
* **Interval:** paired episode-cluster bootstrap (`taniteval/ci.py` through `hwm_eval.contrast`), B 2,000, seed 0, over the 994 SCORE windows in 71 episodes. It answers *another draw of episodes*, nothing else.
* **Training floor** F_T,k(q) = max − min over the three flag-off arms of chassis k of their point means on the row (exact from the points against R35, because C6 makes the windows identical). It answers *another training run*, from three runs per chassis. For scale, ACTSENS's one-pair floors at 0-2 s: `acc_mae` 0.0502 / 0.0638, `yaw_mae` 0.0101 / 0.0159 (`verdict_actsens.md`, MEASURED).
* **Lever-seed spread** |Δ_{k,A} − Δ_{k,B}|: reported per row; the sign rule in BETTER is its decision use.
* **Inference floor** F_I(q) = max over every lever arm and every baseline of |m(inference seed 1) − m(seed 0)| on the row. It answers *another inference run* (the planner samples).
* **Floor_k** = max(F_T,k, F_I).

## 5 Cost on box 2 (RTX 4060 8 GB; ESTIMATED from MEASURED ACTSENS rates)
* **MEASURED, ACTSENS (`C:/Users/Admin/t2v8f_pulls/actsens_*/summary.json` `seconds`, `RUN.done` `at`):** trainer 9,275.6-9,673.8 s per arm at 19,164 steps = 0.484-0.505 s/step. Chain intervals between consecutive `RUN.done` (2026-11-15 08:14:44 → 11:33:21 → 14:56:52 → 18:21:46 → 21:49:32 → 2026-11-16 02:09:08): 3 h 19 min, 3 h 24 min, 3 h 25 min, 3 h 28 min, 4 h 20 min. Gate, selfcheck, post-check and bank add a median 49 min per arm. Bank 44 MB per run (`du`).
* **The term's work, ESTIMATED by counting predictor positions per window:** the R35 step evaluates about 864 (L1 one-step 90, L1 free run 18 anchors × 27, L2 30 + 162, L3 6 + 90); the term adds 4 × 9 × 10 = 360 at context 1 (+42 %), plus small IDM MLP passes and their backward. ⇒ 0.55-0.70 s/step, 2.9-3.7 h trainer, **3.7-4.5 h chain per RSP arm**; NULL arms 3.4 h as measured.
* **Total:** pilot 0.8 h + 4 RSP × 3.7-4.5 h + 2 NULL × 3.4 h ≈ **22-26 h ESTIMATED**, inside the package's 72 h rule. Without the two new NULL arms: 15.6-18.8 h (the floor then falls back to one pair per chassis, as ACTSENS).
* **Stop rules:** PILOT s/step above 1.0, or `torch.cuda.max_memory_allocated` above 6.5 GB, stops the chain; the Master Mind decides. No automatic change to the dose, the anchors or the contrasts.
* **Disk:** about 0.3 GB of banks for six runs; box 2's per-drive budget (≤ 14.9 GB free) is not at risk.
* **Dev-box CPU for the read:** about 2.0-2.2 h per checkpoint for HWM + FLAT at two inference seeds (ACTSENS card section 2, from `hwm_plan/phase1_R35/phase1_R35.log`, INHERITED) plus IA-3 (341 s) and IA-3P (ESTIMATED 10 min): about 14 h for six checkpoints. NULL_s2 / NULL_s3 are read already.

## 6 Controls (if one fails, the read is INVALID)
| # | control | known value |
|---|---|---|
| B-CAL | Build time, before the first run (dev-box CPU, R35_s0 TRAIN latents, no scored window): how much of the IDM's skill comes from `z_t` alone. A linear probe on `z_t` vs one on `[z_t, z_{t+1}]` for each channel | if the `z_t`-only R² exceeds 0.25 × the pair R² on a channel, `s_c` is replaced by the pair-only slope by amendment BEFORE the first run (IA-1 single-state skill ≤ 0 on the eval clips predicts a pass) |
| C1 | PARITY: the ACTRSP commit with the flag absent runs `--selfcheck-only` on R35_s0's argv: `live.terms` equal R35_s0's banked `selfcheck.json` `live.terms`; the flag-absent WM state_dict equals the ACTSENS-commit build at seed 0 (dev-box CPU, two trees) | bit-equal; a tree whose default is 1.0 goes red |
| C2 | RSP-LOG identity (`rsp_w_from_log.py`, the `gamma_from_log.py` pattern) on every `train_log.jsonl`: (level/L1 − pred/L1 − 0.08 sigreg/L1 − 100 idm/L1 − 1.0 rollout/L1) / rsp/L1 | RSP arms: 1.0 within 1e-4 relative at every logged step; NULL arms: no `rsp/L1` key and the identity without it closes to 1e-4 |
| C3 | G-LIVE identity at the gate (step 0): the trainer compares step 0 against a deep copy of the SAME weights and batch with `rsp_w = 0` and writes `live.c3` (re-specified at build: R35's banked selfcheck holds only the last step's terms, so a banked step-0 comparison does not exist) | every unweighted pred / sigreg / idm / rollout term equal; level/L1 difference = 1.0 × rsp/L1 within 1e-5 relative to the term or 8 float32 ulps of level/L1 (level/L1 about 100 at step 0) |
| C4 | PAIR: the flag-on and flag-off builds at the same seed have equal WM init sha256 and equal (ci, t) lists over the first 50 batches | exactly equal |
| C5 | Post-training reachability: `probe_zerograd3.py` (Adam state) on every checkpoint | every `wm.*` tensor has state (ACTSENS `RUN.done`: `n_wm_tensors` 145) |
| C6 | WINDOWS-EQUAL: each arm's 994 SCORE windows (`eid`, `now`, logged `A`, `v0`) against R35's phase-1 windows | exactly equal |
| C7 | REPRO + SELF: `contrast_l1full.py --check` on the HF tar reproduces R35's 75 phase-1 rows; NULL_s2 against R35_s0 re-contrasted reproduces `arm_NULL_s2.json`'s rows; an arm against itself reads 0.0 [0, 0] | exact |
| C8 | ZERO-ACT deliberate regression (zero training, dev box): R35_s0 with `stack.act_embed` zeroed | IA-3 and IA-3P L1 gains, TARGET-READ cycle gain and Part A forward response exactly 0.0 |
| C9 | ANALYTIC IDENTITY of the term (unit test, float64): linear predictor `F(z, a) = Wz + Ba`, linear readout `I(x, y) = K₁x + K₂y`, slopes s fixed | with W = 0 and K₁ = 0, G = K₂B ÷ s at every step, for any δ; with K₂B = diag(s) the loss and its gradient are exactly 0.0; with B = 0 the loss is exactly 0.5; for general (W, K₁), G equals the recursion Δẑ_h = WΔẑ_{h−1} + 2δBe_c read through (K₁, K₂), to 1e-10; δ → 0 or one contrast's sign flipped goes RED |
| C10 | IA-3P instrument: the pair probe's calibration slope on real pairs is > 0.3 on both channels at L1 for R35_s0 and R35_s1 (else IA-3P is NOT COMPUTABLE and MECH-MOVED falls back to IA-3, said so) | measured once on R35, before any ACTRSP checkpoint is read |

## 7 Committed outcomes (both chassis and both lever seeds per chassis are required; each names its next step, run in the same work package under Rule Zero)
| outcome | condition | next step |
|---|---|---|
| **ACTRSP-CEILING** | HWM-ORACLE(arm) − HOLD at 0-2 s: CI wholly below 0 on `acc_mae` or `yaw_mae` for all four lever arms, and not WORSE on the other | (1) This WM replaces R35 as the planning WM of record. (2) The full 6 s and 2-6 s rows are read at once; any band that also beats HOLD is named. (3) **The T1 read that follows:** HWM-PLAN phase 2's predicted-goal arms at tier T1-kin on this WM, HWM-PLAN and FLAT-PLAN with the goal-head (G4 / G5) goal against HOLD and PRIOR (`hwm_plan/PREREG_V8F_HWMPLAN_DRAFT.md`, arm table and outcome GOAL-HEAD-ADDS / HIER-ADDS). (4) The term at levels 2 and 3, carded by amendment, for the 2-6 s band. |
| **ACTRSP-NARROWS** | BETTER on at least one deciding quantity, not WORSE on the other, not CEILING | λ 3.0, four runs (two seeds × two chassis), the same pairs and floors, carded by amendment before any of its windows is scored |
| **ACTRSP-MECH-ONLY** | MECH-MOVED, no deciding quantity BETTER | The response is right and the plan did not improve, so the response is not the binding limit at T0. Zero training, dev-box CPU: the prior-regularised planner (`run_hwm.py --lam > 0`, amended dose) on this WM, then the level-2 image constraint |
| **ACTRSP-NULL** | not MECH-MOVED and no deciding quantity BETTER, with C2 / C3 holding | Two cases, read from TARGET-READ. (i) The term trained down (end `rsp/L1` ≤ 0.5 × first logged, eval cycle gain ≥ 0.8 on both diagonals) and IA-3P did not move: the WM satisfied its own readout without a response a second readout sees. Next: a predictor-only fine-tune of R35 (latent space frozen) with a FIXED linear pair probe pre-fitted on R35's TRAIN pairs as the readout. (ii) The term did not train down: λ 3.0 |
| **ACTRSP-HARM** | WORSE on a deciding quantity at either chassis, or BAND-MET lost at a level | The term stays off. λ 0.3, the bracket below, four runs |
| **SEEDS-DISAGREE** | a label holds at one chassis only, or the two lever seeds of one chassis disagree in sign on a deciding quantity | Reported as such; the next step of the less favourable label applies |

* An arm that misses its bar **FAILED** and is reported that way; its next step runs (Rule Zero).
* **Done** means (a) CEILING with its T1 read started, or (b) every next step above blocked, each blocker named with what would unblock it.
* No dose, anchor, contrast or readout changes after data is seen. Every next dose is named above.

## 8 Box-2 package and stop rules
* `theory/actrsp/box2/`, copied from `actsens/box2/` (`F:/TanitAD-scratch/actsens/box2_build`): the pinned argvs; the per-run token from `actrsp_check.py gate` (the `actsens_check.py` pattern) plus the trainer's `--selfcheck-only`; the pilot and the 72 h rule; the GPU lock `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock` (dev-box stages only); a WMI launch with `CREATE_NO_WINDOW`; the bank to `C:/Users/Admin/t2v8f_pulls/actrsp_<tag>/` with sha256 against box 2's copy.
* A refused arm writes `ARM_REFUSED_<tag>.txt` and the chain continues with the next arm.
* Deletions on box 2 are limited to the run's own gate and pilot dirs, by literal path, after `BANK_VERIFIED`.
* Box 2 is a person's desktop. The chain touches only `D:\tanitad\actrsp\` and `D:\tanitad\runs\v8f_actrsp\`, plus the read-only inputs the argvs name.

## 9 Unchanged and out of scope
* R35's recipe in every other flag, and the data (`exclude 540:600`). `--wm-idm-gamma` and `--wm-sampler` stay at their defaults (ACTSENS levers off).
* The planner, its readers and the phase-1 levers. HWM-PLAN's card and amendments.
* The 2-6 s decode question (AMEND_HWMPLAN_2). No T1 or driving claim is made from this card.

## Log
* 2026-11-16, agent draft. No ACTRSP code, run or scored window existed. Every number above comes from banked artifacts named beside it; the dose and the contrasts come from TRAIN windows and from R35's training log, never from scored windows. Freeze line: Master Mind.
* 2026-11-16, MM: section 4.3 STRATEGIC filled after the PI ruling of 2026-11-16 (pause ended); cells and their HYPOTHESIS constants fixed before any ACTRSP window exists.
* 2026-11-16, MM: section 4.3 n updated to the reader's measured join (offset 0; the audit prototype had used now + 3, which the offset check does not support).
* 2026-11-16, MM: FROZEN before any ACTRSP run, token or scored window. Code of record: branch `agent/v8f-actrsp-20261116` = `a13f6f16` (base `81cee6c1`; 3 trainer files + `test_v8f_actrsp.py`; build report `raw/2026-11-16-v8f-actrsp-prereg/build/BUILD_REPORT.md`). Folded in from the build: C3 re-specified as above; a third hazard (`built_rollout` counted the term's own rollout call as a ladder free run, k 14 instead of 4) fixed and pinned by mutation MR10. PI go for box 2: 2026-11-16 ("1 yes"). Still required before the first run: B-CAL read (section 6; if it triggers, an amendment is frozen before the run), G-SUITE 0 regressions against `81cee6c1`, the real-batch selfcheck on box 2, and a PASS token per run from `actrsp_check.py gate`.
