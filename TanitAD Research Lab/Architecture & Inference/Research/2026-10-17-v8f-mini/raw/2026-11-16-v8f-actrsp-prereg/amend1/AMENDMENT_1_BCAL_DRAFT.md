# PREREG_V8F_ACTRSP, Amendment 1 (DRAFT for the Master Mind's freeze): B-CAL triggered; the term's readout is degenerate; what runs instead

**Status.** DRAFT, 2026-11-16, Arch+Inference agent. Frozen card: `raw/2026-11-16-v8f-actrsp-prereg/PREREG_V8F_ACTRSP.md`, sha256 `ce2d02df8deb44b5e8fc1fd012fe709117fd101335501bb9c358d2099053b480` (checked against the file on `0fc8a103`). Code of record: `a13f6f16`. **No ACTRSP run, token or scored window exists.** Every number below was measured on the 141 TRAIN clips of `latents_train_R35_s{0,1}.pt` (the B-CAL files). No eval latent was read, and nothing was fit on a scored window. Compute: LoRA pod CPU, about 4 min per script per chassis.

**Estimator, for every interval below:** a clip-cluster percentile bootstrap (B 1,000, seed 0, 141 TRAIN clips). It answers *another draw of clips*, with R35's weights and any fitted probe held fixed. It does not answer *another training run* or *another inference run*. Probe skill is cross-fitted with 5 folds **grouped by clip** (one fold map for every probe, so the comparisons are paired).

**Controls that read known values (MEASURED):**
* The trained head's R² on the 31,575 window transitions reproduces `PROBE_ACTSENS_RESULT.json` exactly: 0.4894 / 0.6368 (s0) and 0.5139 / 0.6285 (s1).
* De-duplication: 31,575 window transitions are 9,575 unique (clip, frame) transitions. The latents of one frame are identical in every window that holds it (max |Δz| = 0.0, max |Δa| = 0.0).

## 1 The measurements this amendment rests on

### 1.1 The trained readout Î (`idm_heads[0]`): its skill sits on z_t (MEASURED, `out/amend1_R35_s{0,1}.json` part A)

Î is evaluated on the unique real transitions with z_{t+1} replaced by a counterfactual y, while z_t is held fixed.
* **The slope as written** (card 2.1 item 7, pooled): `s_full` = 0.453 [0.410, 0.496] / 0.663 [0.588, 0.723] (s0, acc / yaw) and 0.480 [0.438, 0.524] / 0.673 [0.613, 0.723] (s1).
* **The part of `s_full` that runs through the transition**: `s_tr = Cov(Î(z_t, z_{t+1}) − Î(z_t, y), a) / Var(a)`, so that `s_full = s_state + s_tr` exactly.

| chassis | counterfactual y | s_tr acc | s_tr yaw | s_tr / s_full acc | s_tr / s_full yaw | share of Var(Î) moved |
|---|---|---|---|---|---|---|
| s0 | STAY (y = z_t) | 0.040 [0.035, 0.046] | 0.097 [0.064, 0.139] | 0.088 | 0.146 | 0.091 / 0.085 |
| s0 | WM_MEAN (y = F(z_t, ā), context 1) | 0.047 [0.041, 0.052] | 0.099 [0.066, 0.140] | 0.103 | 0.150 | 0.091 / 0.087 |
| s0 | TIME (same clip, another step grafted, ≥ 5 frames away) | 0.044 [0.033, 0.057] | 0.113 [0.068, 0.157] | 0.098 | 0.170 | 0.173 / 0.137 |
| s0 | DONOR (nearest z_t of another clip, step grafted) | 0.062 [0.043, 0.079] | 0.101 [0.060, 0.139] | 0.138 | 0.152 | 0.200 / 0.158 |
| s1 | STAY | 0.041 [0.035, 0.049] | 0.081 [0.054, 0.115] | 0.086 | 0.120 | 0.086 / 0.068 |
| s1 | WM_MEAN | 0.053 [0.045, 0.062] | 0.080 [0.052, 0.112] | 0.110 | 0.120 | 0.089 / 0.068 |
| s1 | TIME | 0.046 [0.035, 0.059] | 0.104 [0.069, 0.141] | 0.096 | 0.155 | 0.162 / 0.121 |
| s1 | DONOR | 0.048 [0.029, 0.069] | 0.094 [0.064, 0.124] | 0.100 | 0.139 | 0.171 / 0.127 |

* **The matched step-swap slope** `s_match = Cov(ΔÎ, a − a′) / Var(a − a′)`, with z_t exactly fixed: TIME 0.033 [0.026, 0.042] / 0.057 [0.032, 0.083] (s0) and 0.033 / 0.051 (s1); DONOR 0.056 / 0.085 (s0) and 0.037 / 0.091 (s1).
* **Reading (MEASURED).** 83-91 % of Î's slope runs through z_t. Replacing the next state by z_t itself moves Î by 7-9 % of its variance. The readout's transition-only slope is **0.03-0.11, at or below the card's floor of 0.1**.
* **Î's skill is in-sample.** Held out by clip, a network of Î's own shape (MLP 256-256 on z_t) reads acceleration R² −0.400 [−0.721, −0.195] (s0) and −0.276 [−0.496, −0.112] (s1). Its yaw R² is 0.104 [−0.148, 0.276] and 0.195 [0.008, 0.338] (`D_readouts_clip_grouped.h1`). The card's "R² 0.49-0.64" is therefore a training-set fit to clips the head has seen.
* **The donor geometry explains why matched counterfactuals are hard.** The nearest z_t in another clip lies 3.2-3.3 real steps away (`geometry`). The level-1 states are strongly clip-specific.

### 1.2 What the frozen term would do on R35 at its end (MEASURED, part C, 837 anchors, K 10, context 1)

* **Numerator per unit command** (Î read change ÷ 2δ), mean over K steps: acc 0.089 [0.074, 0.104] / 0.081 [0.069, 0.094]; yaw 0.030 [0.017, 0.045] / 0.020 [0.009, 0.033].
* **With the card's slope:** G_aa 0.196 / 0.168, G_ww 0.045 / 0.029, `rsp/L1` 0.476 / 0.478 (the card ESTIMATED about 0.41).
* **The card's target is mis-scaled.** Its slope is 6.7-11.3× the readout's transition-only slope (ratio of the points above). So G = 1 asks the WM's predicted transitions to move Î by roughly 7-11× as much as a real action of the same size moves it through the transition.
* **With the pair-only slope that B-CAL's rule names** (floored at 0.1, which binds on acc on both chassis): G_aa 0.889 and G_ww 0.301 on s0. Here the calibration is the floor constant, not a measurement.

### 1.3 The one-step linear pair probe (the card's IA-3P and NULL (i) readout) is degenerate too (MEASURED, `D_readouts_clip_grouped.h1`)

* **Held-out by clip, the pair adds almost nothing to z_t:** ΔR² +0.008 [−0.006, +0.021] (acc) and +0.012 [−0.003, +0.027] (yaw) on s0; +0.007 / +0.013 on s1.
* **Its calibration slope is far below C10's bar:** 0.049 [0.032, 0.072] / 0.077 [0.061, 0.096] (s0) and 0.050 / 0.069 (s1), against the bar of > 0.3. Its transition-only part (WM_MEAN) is 0.003-0.004.
* ⇒ **C10 is predicted to FAIL.** This is a TRAIN-clip analogue; C10 itself reads eval clips, and those were not read here. IA-3P would then be NOT COMPUTABLE, and MECH-MOVED would fall back to IA-3, whose L1 probes have no skill (card section 1).

### 1.4 The readout that does see the transition: 2 s (K = 10) displacement (MEASURED, parts D and E)

Clip-grouped held-out R² by horizon h, target = the mean action over h steps (`D_readouts_clip_grouped`). Linear probe on `z_{t+h} − z_t` (D), and the best state-only model:

| h (s) | acc: D | acc: z_t | yaw: MLP pair | yaw: z_t (ridge) |
|---|---|---|---|---|
| 1 (0.2 s) | 0.018 / 0.017 | 0.030 / 0.024 | 0.123 / 0.208 | 0.079 / 0.062 |
| 5 (1.0 s) | 0.114 / 0.110 | 0.038 / 0.036 | 0.164 / 0.245 | 0.070 / 0.054 |
| 10 (2.0 s) | **0.211 [0.148, 0.274] / 0.206 [0.142, 0.269]** | 0.048 / 0.047 | 0.212 / 0.174 | 0.053 / 0.046 |

Two fixed readouts were fitted for K = 10 (`out/kstep_R35_s{0,1}.json`; frozen weights in `out/readout_K10_R35_s{0,1}.pt`):

* **RA, the acceleration readout** (ridge on `z_{t+10} − z_t`):
  * Held-out R² 0.211 / 0.205.
  * Beyond the best state-only model: **+0.163 [0.111, 0.214] / +0.158 [0.103, 0.215]**.
  * Transition-only slope with z_t exactly fixed (in-clip displacement swap): **s_swap 0.201 [0.140, 0.265] / 0.203 [0.140, 0.268]**. The WM_MEAN form gives 0.166 / 0.164.
  * ⇒ A calibratable acceleration readout exists at 2 s, on both chassis.
* **RY, the yaw-rate readout** (MLP on `[z_t, z_{t+10}]`):
  * Held-out R² 0.200 [0.042, 0.341] / 0.241 [0.044, 0.414].
  * Beyond the best state-only model: +0.147 [−0.018, 0.276] / +0.139 [0.010, 0.223].
  * Transition-only slope s_swap: **0.038 [−0.027, 0.097] / 0.013 [−0.042, 0.067], CI includes 0 on both chassis**. The WM_MEAN form gives 0.131 / 0.087, so the two forms disagree.
  * ⇒ **No calibratable yaw-rate readout exists at level 1, at 0.2 s or at 2 s.**

### 1.5 R35's own response through RA: not deficient (MEASURED, part E `term_preview_R35_end`)

* **The term's G at R35's end (the fine-tune's step 0), read at the 2 s endpoint through RA** and calibrated by s_swap: **G_aa 1.244 [1.116, 1.376] (s0) and 1.202 [1.066, 1.333] (s1).**
* **Regression form agrees:** `Cov(RA(F¹⁰(z, a)) − RA(F¹⁰(z, ā)), ā) / Var(ā)` = 0.287 / 0.258, against the real 0.166 / 0.164.
* **RA reads R35's rollout under the logged actions at R² 0.410 / 0.373**, against 0.211 / 0.205 on the real transitions.
* **Reading.** Through a readout with transition skill, R35's 2 s acceleration response is 1.2× the real one, not a tenth.
  * The card's "8.7-10.3 % of the real step" (section 1) is a ratio of the norms of the WHOLE one-step latent change. Most of that change is not action-driven. It is a correct measurement read as the wrong quantity (root-cause class for `RETRACTION_LOG.md`: wrong-scope probe).
  * **ESTIMATED:** the frozen term, whose target sits 7-11× above the transition response, would have pushed the acceleration response well past the real one.
* **Yaw:** the response cannot be sized at level 1 (1.4). G_ww reads 0.680 / 1.587 through RY, uncalibrated, and is not quotable.

### 1.6 Why yaw is invisible at level 1 (MEASURED structure; HYPOTHESIS mechanism)

* **The structure (MEASURED).** R35 is built with `--grid pool4x10` (argv). The checkpoint `cfg` has `n_tok` 40, a 4 × 10 spatial token grid. The level-1 state is `z1 = proj(mean over the 40 tokens)` (`v8f_mini.py` `z1_from_mean`; the zcache holds the token mean, `--wm-pooled on`, `v8f_mini_train.py:355-357`).
* **The mechanism (HYPOTHESIS).** A mean over a spatial grid is invariant to which column a feature sits in. A heading change of 11° in 2 s (1 sd of yaw rate) moves features sideways and leaves their mean nearly unchanged. Forward motion changes the features themselves. That matches the measured asymmetry: linear D reads acc 0.21 and yaw −0.048 / −0.050 at 2 s.

## 2 What changes (the amendment)

### 2.1 B-CAL's rule, applied as written
B-CAL triggered (`launch/bcal/`, `0fc8a103`). The rule replaces `s_c` by "the pair-only slope" before the first run. That slope is now defined so that it can be measured (`s_tr` and `s_match` above). It reads 0.03-0.11, at or below the 0.1 floor, so the term would be calibrated by its floor constant, not by data. ⇒ **The term as frozen (readout Î, item 7 slope) is DEGENERATE.** It is not run in any form that keeps Î as its readout.

### 2.2 Recommended: Option A, the ACTRSP arms are HELD and the two levers that the measurements name run instead
* **The arms are held.** Arms 0-6 (section 3) are not run. The box-2 package stays unshipped. No `actrsp_check.py gate` token is issued.
* **No trainer change.** `a13f6f16`'s `--wm-rsp-w` stays default-off and is named in no argv.
* **Why it is held (Rule Zero rule 5: the largest measured effect).**
  * On acceleration, R35's response is already ≥ 1 through an informative readout (1.5). The term has no calibratable deficit to train on that channel.
  * On yaw rate, no level-1 readout can calibrate the response (1.4).
  * C10 is predicted to fail (1.3), so MECH-MOVED could not be read either.
  * Running the arms would spend the card's 22-26 box-2 hours on a target that the build controls show cannot be met as specified.
* **Outcome label: ACTRSP-HELD-AT-B-CAL.** This is a build-control outcome, not one of section 7's run outcomes. It is reported as a FAILED build control with its two next levers:
  1. **LAT-TOK (lateral, zero training, CPU).** The pre-run check of a level-1 state that keeps the column layout. Pre-registered here, with both outcomes committed:
     * **Inputs.** The same 141 TRAIN clips and 2 s targets. The adapter's 40 spatial tokens per frame, recomputed from the clips' `feats` with the frozen M2b_s{k} adapter (the weights are inside the R35 checkpoints). Feature sets:
       * (a) the 40-token mean, the present z1 input;
       * (b) the column means (10 × 256);
       * (c) all 40 tokens.
     * **Readouts.** Linear D and MLP pair on each feature set, clip-grouped 5-fold, plus the swap slope.
     * **Pass.** (b) or (c) beats (a) on 2 s yaw-rate R² by a paired clip-cluster CI wholly above 0 on both chassis, AND its s_swap CI is wholly above 0.
     * **Pass ⇒** the lever is the level-1 state's construction: z1 built from the column means instead of the grid mean. That is an architecture change, carded separately, and needs the PI's go.
     * **Fail ⇒** 2 s heading change is not in the frozen adapter's tokens either. The lever moves to the adapter itself (adapter training), a PI decision. Named, not run.
     * **Blocker.** The token-level adapter output is not cached; the zcache holds the token mean. Computing it needs the 141 clips' `feats` from box 2's `cache_full/train/clips` (the `collect_latents_train_R35.py` pull), plus the adapter forward on dev-box CPU (after the 16:40Z memory window) or on pod CPU. Not run here: this brief forbade box 2 and dev-box memory.
  2. **LON-PLAN (longitudinal, zero training, dev-box CPU).** The card's own ACTRSP-MECH-ONLY next step, entered now from the measured response (1.5): the prior-regularised planner (`run_hwm.py --lam > 0`) on R35_s{0,1}, then the level-2 image constraint.
     * **Blocker.** It needs its λ dose fixed in an HWM-PLAN amendment (MM) before any window is scored.
* **Cost delta (Option A):** −22 to −26 h box-2 (card section 5, ESTIMATED) and −14 h dev-box CPU read. LAT-TOK is ESTIMATED at about 0.5-1 h CPU (141 pulls at 3-10 s each, measured in `collect_R35.log`, plus 40 × 2,397 × 16 adapter tokens; probes of minutes, as here). LON-PLAN costs about 2.2 h dev-box CPU per checkpoint and inference seed pair (card section 5, INHERITED).

### 2.3 Specified, not recommended: Option B, a calibrated 2 s acceleration term in a frozen-latent fine-tune
This is for the MM, in case the MM prefers a run. **Predicted (ESTIMATED from 1.5): the term starts at mean G_aa 1.2.** Its step-0 loss of 0.94 / 0.99 is almost all spread across anchors, not bias, so the term would mainly flatten the gain across states and lower its mean by about 20 %. The sign of its effect on `acc_mae` is not predictable from these data.

* **The term.** Acceleration channel only, read at the K = 10 endpoint by the FROZEN RA of chassis k.
  * `G = w·(ẑ_K^{+} − ẑ_K^{−}) / (2 δ_a s_a)`, with `s_a` = the s_swap constant (0.2008 for s0, 0.2032 for s1, from `readout_K10_R35_s{k}.pt`).
  * Two contrasts instead of four. `rsp/L1 = mean (G − 1)²`. Anchors and context are unchanged. There is no floor: s is a pre-registered constant with a CI wholly above 0.
  * Yaw is dropped (1.4), and so are the off-diagonals: the yaw read is not calibratable.
* **The arms become fine-tunes**, because the readout is only valid in R35's latent space.
  * Every arm starts from R35_s{k}'s `ckpt_last.pt` with ALL `wm.*` tensors loaded, and keeps `wm.proj` frozen. Today's WM-only mode re-initialises the WM (`v8f_mini_train.py:119`, `:325`, `:831`, `WM_ONLY_NOT_COMPARED`).
  * Per chassis: FT-RSP seeds A and B, and FT-NULL seeds A, B and C (the same flags with the term off). Each FT-RSP arm pairs with the FT-NULL arm of its seed: same start, same data order, and the term draws no random number, so C4 holds.
  * 4,000 steps; λ 1.0, with brackets 0.3 and 3.0 kept.
  * Bars, the deciding quantities (0-2 s `acc_mae`, `yaw_mae`), the floors and section 7's labels are unchanged. MECH-MOVED is read through IA-3P-K (below) on the acceleration channel only.
* **The trainer change.** Minimal and default-off-safe: every new flag's default is the `a13f6f16` build, bit for bit.
  * **Flags.**
    * `--wm-rsp-readout {idm, fixed_k}`, default `idm`.
    * `--wm-rsp-probe <path>` and `--wm-rsp-probe-sha256 <hex>`. Required when `fixed_k`; the trainer refuses on a sha mismatch.
    * `--wm-init-from-ckpt-wm {off, on}`, default `off`.
    * `--wm-freeze-proj {off, on}`, default `off`.
  * **Hops.**
    * Each flag: argparse → `FLAG_CHECKS` lever entry → `make_config` → `V8FMiniConfig` field with `__post_init__` refusals (`fixed_k` needs `wm_rsp_w > 0`, the probe file and `--wm-freeze-proj on`).
    * The readout goes to `WMAux.__init__` → `LevelConfig(rsp_readout, rsp_w_vec, rsp_s_cal)`, with the probe as non-persistent buffers built last, so the WM init does not move. Then `HJEPAStack.response_loss` takes the `fixed_k` branch.
    * `--wm-freeze-proj` goes to `freeze_all_but_wm` (`proj.requires_grad_(False)` before the optimiser). `--wm-init-from-ckpt-wm` goes to `load_init_ckpt` (load the `wm.*` keys instead of re-initialising).
    * G-DVB: `built_rsp` gains `readout`, `n_contrasts` (2) and `s_cal`. `bad_wm_rsp` compares them with the argv. A new `bad_wm_freeze` reads the frozen set back from `requires_grad` and checks `wm.proj.*` against the checkpoint after one step.
  * **New mutations that must go RED:**
    * **MR11:** the `fixed_k` branch computes its slope from the batch (`Cov/Var` of the readout on real transitions) instead of the constant. It goes RED on the new C9' identity.
    * **MR12:** `--wm-freeze-proj on` is parsed but `proj` keeps `requires_grad`. It goes RED on `bad_wm_freeze` and on a test that `wm.proj` is bit-equal to the checkpoint after 2 steps.
    * **MR13:** the probe sha check is skipped. It goes RED on a test that loads a probe with one changed byte.
    * **MR14:** `fixed_k` reads the per-step pairs instead of the endpoint. It goes RED on the C9' geometry assertion (1 read per anchor and contrast).
* **The controls under Option B.**
  * **C3:** unchanged in form (`level/L1` difference = λ × `rsp/L1`).
  * **C9 becomes C9′ (analytic identity, float64).**
    * The world: `z_{t+1} = z_t + B* a_t`, with autocorrelated actions, so that z_t carries the action history, and a readout `R(x, y) = K₁x + K₂y` with K₁ ≠ 0.
    * With the WM's `B = B*`, `W = I` and the transition-only constant `s = diag(K₂B*)`: G = 1 exactly for any K₁ (loss and gradient exactly 0.0).
    * With `B = 0`: loss exactly 1.0.
    * The card's pair slope on the same data gives G ≠ 1 whenever K₁ ≠ 0. This is the state leak that 1.1 measures, and MR11 must turn it RED.
  * **C10 becomes C10′.** C10 is predicted to fail (1.3). IA-3P is replaced by IA-3P-K: RA's form, cross-fitted on the eval clips with IA-3's folds and calibrated by its s_swap on real pairs. Bar: the s_swap CI is wholly above 0 on acceleration, at both chassis, measured once on R35 before any FT checkpoint is read.
  * **C1, C2, C4-C8:** unchanged. C4 now pairs FT-RSP with FT-NULL, not with a from-scratch arm.
* **Cost (Option B, ESTIMATED from the card's MEASURED 0.48-0.51 s/step and 49 min of overhead per arm):** 10 arms × (4,000 × 0.55-0.70 s + 49 min) ≈ 14-16 h box-2, plus a 0.8 h pilot. The read is 10 checkpoints × about 2.2 h of dev-box CPU ≈ 22 h. One build agent for the four flags, four mutations, C9′ and C10′.

## 3 Unchanged
The card's sections 4.3 (families, STRATEGIC reported in full), 4.4 (estimators and floors), 8 (package rules) and 9. Every number in sections 1-3 of the card stays as measured. Only the reading "the level-1 forward response is a tenth of the real step" changes: through an informative 2 s readout it is 1.2×, and at the 0.2 s step no readout separates the action from the state.

## 4 Register rows for the MM (to land with the freeze)
* **`H-ACTRSP-BCAL`**, FAILED build control, MEASURED. The level-1 IDM readout's skill is 83-91 % state, its transition-only slope is 0.03-0.11, and held out by clip its shape reads acceleration R² −0.28 to −0.40. Next: LAT-TOK and LON-PLAN.
* **`H-ACTRSP-RESP-2S`**, MEASURED. R35's 2 s acceleration response reads 1.20-1.24× the real one through a readout with clip-held-out transition skill (+0.16 R² over the state). No level-1 yaw readout is calibratable (s_swap CI includes 0, both chassis).

## Log
* 2026-11-16, agent draft. Measurement scripts: `amend1_readout.py`, `amend1_kstep.py`. Raw outputs: `out/`. Model code from `a13f6f16` (hjepa blob `0cfd7081`, v8f_mini blob `f15c28c9`). Freeze line: Master Mind.
* Deletions (own temporary data, literal paths). Before the deletes, the sources were checked intact and the pulled outputs were checked sha256-equal to the pod's copies:
  * pod `/root/actrsp_amend/`, 132,234,531 B (copies of the two latents files and two R35 checkpoints, a code export, outputs);
  * `F:/TanitAD-scratch/actrsp/amend1/ref/`, 627,866 B (git exports);
  * session scratchpad `podship/`, 192,877 B (code export).
