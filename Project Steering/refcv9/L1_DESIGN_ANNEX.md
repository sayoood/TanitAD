# refcv9 L1 — along-track residual calibration: in-model design annex

*Architecture & Inference stream, 2026-10-19 (~20:30 Berlin). Annex to `Project Steering/PLAN_REFCV9.md` §2 L1 (DRAFT v0). Nothing
here is authorised compute and no GPU job was run. Every number carries its evidence class. MEASURED numbers come from the three
CPU probes in `refcv9/l1_probe/` (scripts, logs and JSON next to this file), or from the battery hold
`hold_r8batt/.../2026-10-15-refcv8-milestone-battery/raw/`. All planner numbers are T1 open loop on logged eval139 frames,
refcv8 EMA-30k, ceiling-compliant S2 windows (4,159 windows / 138 episodes), with the paired episode-cluster bootstrap over EPISODES
(B 2000). They come from ONE training run (H-ESTIM-SEED-1); inference seeds 0 / 1 are reported as "s0 / s1".
Source lines are cited at `origin/agent/arch-inf-20260803` @ `66eb571`.*

> ⚠️ **Path note.** The brief cites `stack/tanitad/models/refc.py` and `models/refc_v3.py`. Those paths do not exist on the branch.
> The files are `stack/tanitad/refs/refc.py` and `stack/tanitad/refs/refc_v3.py`, and every citation below uses those paths.
> `kinematic_prior.py` is in `stack/tanitad/models/`.

---

## 0. Headline (read this if nothing else)

1. **Localisation: SELECTION.** The along-track excess is not in the anchor bank's final effect and not in the refinement. MEASURED:
   * The **a_star candidate** is the object `loss_traj` supervises: the GT-nearest bank anchor, refined. Its progress is
     calibrated, with flat re-timing gain **g\* 0.909 [0.784, 1.004]** against 0.455 for the emitted plan.
   * **89 %** of the emitted plan's S2 ADE gap to that candidate is *progress along the path*, not path shape. Re-timing refcv8's
     own path to the a_star candidate's progress gives E1 **−0.0807** against **−0.0892** for the candidate itself, and −0.0136
     for os.
   * The **refinement shrinks** the selected anchor's along-track residual (ρ 1.60 → 0.86). It does not inflate it.
   * The **train/eval gap** (g\* 0.69 vs 0.46) is entirely in the emitted plan. The a_star candidate reads the same on train and
     eval clips (Δg\* +0.001 [−0.126, +0.136]).
2. **The plan's framing "progress runs ~2x too far ahead" is wrong as a mechanism and should be corrected** (`PLAN_REFCV9.md` §1
   D1; register `H-R8-RS-1`'s *"residual is too large"*).
   * The emitted residual over the prior is **smaller** than the human's: ρ 0.84 [0.77, 0.92].
   * Its mean is unbiased: +0.142 m vs the human's +0.143 m at 2 s.
   * The real problem is that it is **mis-aligned**: correlation with the human's residual r = **0.38** [0.34, 0.43].
   * g\* = r/ρ. The 0.46 is the least-squares shrink of a noisy speed choice, not a 2x over-scale.
   * The defect is **inherited unchanged from refcv7-50,400**: g\* 0.428, r 0.390; refcv8 − refcv7 Δr −0.007 [−0.051, +0.034].
3. **Recommended mechanism: (c4) a deterministic PROGRESS HEAD plus an in-forward re-timing layer.**
   * The head regresses the human's arc-length residual over the prior at the 8 slots.
   * The layer re-times the *selected* path to the head's progress.
   * This decouples "how fast" (regression, which generalises) from "which path" (selection, which does not).
   * Lateral is untouched by construction: the path shape is kept, as with RETIME, whose ACC-LAT held.
   * The same layer with a fixed table is mechanism (a), and with a fitted gain it is (b). So the floor (= refcv8 + stage) is
     reproduced at initialisation.
4. **A model-free control that refcv9 must carry: E1 against `ha0_ext_pose` can be passed WITHOUT VISION.**
   * MEASURED: a ridge regressor of human progress on the t0 ego scalars plus the fed ceiling, fit on TRAIN clips only, applied
     along the **prior's own path**, gives E1 **−0.0212 [−0.0299, −0.0130]**.
   * Applied along refcv8's path, it **ties the shipped stage**: −0.0328 vs −0.0318, Δ −0.0010 [−0.0071, +0.0051].
   * ⇒ An L1 lever only shows vision if it beats this scalar floor, separated.
5. **Cheapest ranking experiment: bank `z_tac` once, then fit heads many times on CPU.**
   * ~3.5–4 dev-box GPU-h (ESTIMATED), no Thor, after refcv8-final is banked.
   * Selection-loss arms (c3) only if that fails: ~13 Thor-h.
6. **Escalations.** The PI must rule on four things, and nothing in this annex is merged or wired:
   * (i) whether a fixed in-forward table (a) satisfies B9-1;
   * (ii) the warm start, which decides whether a clean validation split exists (§3);
   * (iii) compute for §4 E1/E2;
   * (iv) the PLAN §1 D1 wording.

---

## 1. Where the over-run is produced

### 1.1 How `traj` is composed (source)

| step | what happens | source |
|---|---|---|
| prior | `(a0, κ0)` from the observed pose window: a0 = (v[t0] − v[t0−1])/dt, κ0 = ω0/max(v0, 2) clamped. Withheld rows get zero controls. In training, κ0 → 0 on 30 % of rows (lateral prior dropout, `--r8-lat-prior-dropout 0.3`). | `refs/refc.py:2339-2402` `_residual_prior`; `models/kinematic_prior.py:366-470`; `refs/refc.py:4570` `_residual_prior_inputs` (called at `:5188`) |
| bank | The 117 v0-conditioned anchors are **residual controls Δ on P**, rolled with the prior. | `refs/refc.py:3488-3491`, `:2405-2421` `_roll_residual_bank` |
| composition | `a = a0 + Δa`, `κ = κ0 + alat_to_curvature(Δalat, v)`, each slot's residual held over its span, unicycle roll at 0.1 s. **The residual lives in CONTROL space**, not position space. | `kinematic_prior.py:677-715` (`:710`, `:713`), `:718-748` `roll_plan` |
| refinement | DDIM from `t = 8`, 2 steps. Each pass predicts `du`, and `x0_hat = x_n + du` (residual). In training: one pass at a random t (`f1_random_t`). The fan is `roll_plan(u0_hat, P)`. | `refs/refc.py:2961-3160` (`:3079`, `:3114`, `:3141`, `:3158`) |
| selection 1 | The ranked score is the emitting pass's confidence (F5) + grafts + the refcv8 factorised term `β·log p(h_k)` (**the tactical heads' posterior**) + sub-score critics + speed-ceiling mask. `idx = argmax`. | `refs/refc.py:3880-3905`, `:3940-3957`, `traj = x[·, idx]` at `:3990` |
| selection 2 (E9) | Re-ranks the fan by distance at the 2 s slot to the predicted tactical goal point `g2`, plus the drivable critic, then argmax. | `refs/refc_v3.py:2570-2655` |
| cap | `plan_vmax_clamp` (inference only) caps the emitted plan to the fed ceiling. | `refs/refc_v3.py:2656-2671`; `out["traj"]` at `:2673` |
| refcv7 scorer | **Not active in refcv8.** `refcv7_select` needs `--refcv7`, which the argv does not carry. | `refs/refc_v3.py:586`, `:2826-2828`; `stack/ops/runs.d/refcv8-r101-s0.argv.json` |
| prior export | `out["residual_prior_path"] = prior_path(a0, κ0, v0, horizons)`. This is the prior the shipped stage and SHIP-R8 clause (7) use. | `refs/refc.py:3995-4006` |

**Trajectory losses** (`stack/scripts/refc_v3_train.py`):

* `a_star = argmin_n ||GT − bank_n||²` over the 8 valid slots (`:5543-5545`).
* `loss_traj = L1(anchor_traj[a_star], GT)` on x + y in the t0 ego frame, divided by 2·#valid slots (`:5558-5560`). Its weight
  is TRAJ_WEIGHT 1.0 (`refc_train.py:76`).
* Selection is trained only as classification toward `a_star`: focal CE, ANCHOR_CLS_WEIGHT 1.0 (`:5553-5557`;
  `refc_train.py:77`). The cascade adds the same pair per stage (`:5952-5969`).
* `--w-u0` is 0 through `--f6-w-u0-zero` (`:538-542`).
* refcv8 adds the r8 terms (`stack/tanitad/train/refcv8_train.py:625-727`). The listwise term, whose cost includes the arc-length
  speed profile, is **OFF** (`--w-r8-listwise 0.0`, PI-R8-NOLW: it cost turns). The sub-score critics are on (`--w-r8-subscore 0.5`)
  and include a 6 s path-length `progress_10pct` target (`refs/refcv8_conditioning.py:537-556`).

⇒ **No loss ever sees the geometry of the EMITTED plan.** The regression trains one candidate per window, `a_star`. The emitted
plan is whatever the selector picks, and selection is trained only to name `a_star`.

### 1.2 The decomposition (MEASURED, `l1_probe/ALONG_TRACK_LOCALISE.json`)

**Definitions.**
* For each arm X: `d_X = s_X − s_P`, the arm's arc length minus the prior's at each S2 instant. `h = s_H − s_P`, the human's.
* g\* = ⟨d,h⟩/⟨d,d⟩ is GAINTRAJ's flat re-timing gain. ρ = ‖d‖/‖h‖ is the residual's size. r = corr(d, h) is its alignment.
  A = ⟨d,h⟩/⟨h,h⟩ is exactly additive across arms.
* Arms:
  * **os** = the emitted plan;
  * **bank** = the selected candidate's UNREFINED bank path (`sel_bank_nav_true`; only where sel_idx < 117, which is 91.8 % of
    compliant windows);
  * **orc** = the a_star candidate, refined (`oracle_sel`). It is a T0 ceiling, because a_star uses GT.

**Controls.** All pass:
* os g\* reproduces `GAINTRAJ.json` 0.4552 / 0.4616 and `GAINTRAJ_TRAIN.json` 0.6863 / 0.6765 exactly.
* The prior/os-swap mutation reads 0.545 (RED).
* Known values: d := h gives 1 / 1 / 1 / 1; d := 2h gives g\* 0.5, ρ 2; d := h + e with e ⟂ h gives g\* 0.5, ρ √2.
* `plan_full[:, :4]` is bit-identical to os.

**EMA-30k eval139, compliant, in-bank windows (n 3,819 / 138 ep; s1 within ±0.01):**

| arm | g\* | ρ (size) | r (alignment) | A |
|---|---|---|---|---|
| bank (selected, unrefined) | 0.195 [0.157, 0.234] | **1.60** [1.47, 1.73] | 0.311 [0.259, 0.362] | 0.497 |
| os (emitted) | 0.470 [0.392, 0.551] | **0.86** [0.78, 0.93] | **0.402** [0.353, 0.450] | 0.344 |
| orc (a_star refined, T0) | **0.919** [0.803, 1.006] | 0.83 [0.78, 0.89] | **0.760** [0.715, 0.792] | 0.628 |
| os − bank (refinement) | +0.275 SEP | **−0.743 SEP** | +0.091 SEP | −0.153 SEP |
| os − orc (selection) | −0.449 SEP | +0.028 n.s. | **−0.358 SEP** | −0.285 SEP |

On all compliant windows (n 4,159), os reads g\* 0.455 / ρ 0.842 / r 0.383, and orc reads 0.909 / 0.828 / 0.753.

**What the table says.**
* The refinement shrinks an over-sized bank residual to below the human's size and improves alignment slightly. It is not where
  the excess comes from.
* Between os and the a_star candidate the residual size is the same (ρ n.s.) and the **alignment** differs.
* The emitted plan picks a candidate whose speed profile is the wrong one for this window. **That is selection.**

**Mean signed residual per instant** (os / human, m, at 0.5 / 1 / 1.5 / 2 s): 0.021 / 0.074 / 0.110 / **0.142** against 0.034 / 0.070
/ 0.106 / **0.143**. There is no systematic "ahead". Human RMS deviation from the prior is 0.05 / 0.15 / 0.35 / 0.68 m.

### 1.3 Train vs eval and over training (MEASURED, same file)

| | os g\* / ρ / r | orc g\* / ρ / r |
|---|---|---|
| eval139, EMA-30k | 0.455 / 0.842 / 0.383 | 0.909 / 0.828 / 0.753 |
| train139 (A6), EMA-30k | 0.686 / 0.737 / 0.506 | 0.906 / 0.864 / 0.783 |
| train − eval (unpaired, s0) | **+0.219 SEP** / −0.115 SEP / **+0.107 SEP** | +0.001 [−0.126, +0.136] / +0.028 n.s. / +0.027 n.s. |
| eval139, EMA-5k | 0.503 / 0.804 / 0.405 | 0.910 / 0.821 / 0.747 |

**The generalisation gap is the selector's, not the regression's.** The candidate trained by L1 regression reads the same on clips
the model trained on and on clips it never saw; the selection does not. Between 5k and 30k, os alignment does not improve
(0.405 → 0.383) while the bank's over-size grows (ρ 1.37 → 1.60). This is consistent with GAINTRAJ's "structural, does not fade".

**Inherited, not new** (MEASURED, `l1_probe/REFCV7_ALIGNMENT.json`, refcv7-50,400 plans banked in the same panels, same windows):

| | g\* | ρ | r |
|---|---|---|---|
| refcv7-50,400 | 0.428 [0.354, 0.515] | 0.911 | 0.390 |
| refcv8 − refcv7 | +0.027 n.s. | −0.068 SEP | −0.007 [−0.051, +0.034] n.s. |

refcv8's sub-score critics (with their `progress_10pct` target) and its factorised selection did **not** change along-track alignment.

### 1.4 How big is the prize, and how much does the stage already take (MEASURED, `l1_probe/PROGRESS_HEADROOM.json`)

Every arm keeps refcv8's emitted path and changes only the arc length. The geometry is the shipped stage's
(`a0k0_headroom_retime_r8` = `retime_stage.py` port).

**Controls.** `RETIME_TRAIN` reproduces `FORWARD_TRAINFIT_step30000ema_unc.json` exactly: −0.0318 / −0.0313, mean ADE 0.254019.
CONST_G05 reproduces −0.0308 / −0.0303.

| arm (S2 ADE − ha0_ext_pose) | s0 | s1 | vs RETIME_TRAIN (s0) |
|---|---|---|---|
| os | −0.0136 [−0.0286, +0.0018] | −0.0127 | +0.0182 SEP |
| RETIME_TRAIN (= shipped stage on S2) | −0.0318 [−0.0433, −0.0201] | −0.0313 | — |
| CONST_G05 (no fit) | −0.0308 | −0.0303 | +0.0010 SEP |
| GATED_RETIME_TRAINFIT (v0-bin train gains; battery `FORWARD_TRAINFIT`) | −0.0348 | −0.0340 | point −0.003; paired CI not computed |
| **PROG_ORC**: refcv8's path at the a_star candidate's progress (T0) | **−0.0807** [−0.0955, −0.0668] | −0.0793 | **−0.0489 SEP** |
| orc itself: path + progress (T0) | −0.0892 [−0.1039, −0.0762] | −0.0880 | — |
| PROG_HUMAN: refcv8's path at the human's progress (T0 ceiling of any re-timing) | −0.1780 | −0.1775 | −0.1462 SEP |

⇒ **(−0.0807 + 0.0136) / (−0.0892 + 0.0136) = 89 %** of the os → a_star-candidate gap on S2 is progress.

The shipped stage takes **0.0182 of the 0.0671** that a progress choice as good as the trained candidate's would take (27 %).
Everything beyond it needs better progress *information*, not better shrinkage.

### 1.5 The model-free floor (MEASURED, `l1_probe/PROGRESS_SCALAR.json`)

**Setup.**
* Ridge regression of `h` (4 S2 instants) on [v0, a0, ω0, v0·ω0, v0², a0·v0, min(v_lim, 40), min(v_lim, 40) − v0, v_lim valid].
* These are the prior's own t0 inputs plus the fed ceiling label.
* It is fit on the 139 TRAIN clips only (4,211 windows / 137 ep). λ = 10 was chosen by 5-fold episode CV on train.
* Train R² by instant: 0.51 / 0.10 / 0.07 / 0.10. Eval R²: 0.53 / 0.12 / 0.06 / 0.09.
* **Controls:** RETIME reproduces; zero correction == prior exactly.

**Results.**
* **PROG_SCALAR_PRIOR**, fully model-free (the prior's path, re-timed by the scalar regressor): E1 **−0.0212 [−0.0299, −0.0130]**
  at both seeds. **It passes the E1 form against `ha0_ext_pose` without any model.**
* **PROG_SCALAR_OS** (refcv8's path, re-timed by the scalar regressor): **−0.0328 / −0.0323**. Versus RETIME: −0.0010
  [−0.0071, +0.0051], not separated.

⇒ **What the shipped stage's value can and cannot be credited to.**
* Its value is reproducible from ego scalars. Beyond 0.5 s the scalars explain ≤ 12 % of human progress variance, so the remaining
  ~88 % needs the scene.
* An L1 claim of *vision* therefore needs: refcv9-arm − PROG_SCALAR_OS, separated < 0.
* E1 as registered compares the planner against a prior that a 10-coefficient regression already beats. Flagged for the bar owner
  (§6).

### 1.6 Not measured (named, so nobody quotes around it)

* **S6.** `oracle_sel` is dumped on S2 only, and S6 decisions carry `sel_bank` rows for a different row set. The S6 localisation
  needs the a_star candidate at 3–6 s: one os roll with a `--dump-orc-s6` key.
* **Which half of selection.** Is it the base score (F5 confidence + `β·log p(h_k)`) or E9's goal distance to `g2` that picks the
  wrong speed? `goal_dist_sel` is dumped, but `g2` and the pre-E9 index are not.
* **Steps 40,000 / 48,852.** The 40k tree is incomplete on disk (`dump_s0/refcv6_extras.npz` missing; skipped by the probe).
  48,852 does not exist yet. The probe re-runs on any banked tree in ~10 s per seed.
* **The allocator extras.** The 8.2 % of compliant windows where sel_idx ≥ 117 have no bank row and are outside the in-bank
  decomposition.

---

## 2. Three mechanisms, plus the recommended fourth

### 2.0 The common component: `ProgressRetime`, an in-forward layer

**Where.**
* `RefCV3Model` E9 block, immediately after `idx = rank.argmax(dim=1)` / `traj = fan[·, idx]` (`refs/refc_v3.py:2654-2655`) and
  **before** the `plan_vmax_clamp` cap (`:2656-2671`), so the hard ceiling stays the last word.
* ⚠️ The shipped stage re-times the post-cap `out["traj"]` (`retime_stage.py:233-248`). So at initialisation the in-forward
  function equals RETIME_DEPLOY8 only on windows where the cap does not bind. The pre-registered equivalence check must say so.
  The cap-binding fraction is UNMEASURED.

**What.**
* A torch port of `retime_stage.polyline_geometry` / `eval_polyline` / `retime_progress`, applied as one call on all 8 slots
  (SHIP-R8 Amendment 2's deployed geometry).
* The prior is `residual_prior_path` from the same forward (`refs/refc.py:4003`), so clause (7) prior parity holds by construction.
* Output `out["traj"]` = the re-timed plan. Keep `out["traj_preretime"]` for audit and for the B9 paired reads.
* It is differentiable in the target progress `s'`. The path vertices are detached, so no gradient reaches selection.

**Why the existing training is unaffected.** `loss_traj` and the selection CE read `anchor_traj[a_star]` and the logits, never
`out["traj"]` (`refc_v3_train.py:5545-5560`). The layer therefore leaves every existing gradient bit-identical, and its effect is
attributable.

**Lateral.** Every re-timed point lies on the selected polyline or its end ray. RETIME's pre-registered ACC-LAT held: S6 heading
vs os −0.037 [−0.105, +0.027], no lateral row worse (`H-R8-RETIME-1`, `H-R8-RETIME-TRAINFIT-1`). TAC_traj_lat/lon of the re-timed
plan were not worse than refcv7 at both seeds (S2 lon **+0.030 SEP better**, `raw/retime_deploy/TAC_step30000ema_unc.md`).

### 2.1 (a) Fixed along-track gain table in the forward = the frozen stage moved in-model

| | |
|---|---|
| code | `ProgressRetime(mode="fixed")`. Buffers = `configs/refcv8_retime_deploy_gains.json` (sha-checked as `retime_stage.load_deploy_table` does), `s' = s_P + g(t)(s_sel − s_P)` |
| params | 8 buffers, 0 trainable |
| training cost | none |
| measured | E1 −0.0318 / −0.0313; S6 vs pose −0.976 / −0.973; ACC-LAT HOLDS; missed decel −0.028 / −0.024 SEP *better* than os (S6 compliant, `raw/refcv9_headroom/L2_HEADROOM.md`). MEASURED as a post-hoc stage on the identical function. |
| can fix | E1 form at refcv8's level, deterministically, at 1x inference |
| cannot fix | anything it was not fitted on. It cannot learn and cannot improve alignment (r). It under-corrects on eval because its gains are train-fit (0.69 vs 0.46). If eval g\* keeps falling (GAINTRAJ), it may stop clearing E1. **It cannot beat refcv8 + stage: it IS refcv8 + stage.** |
| admissibility | reads only the selected path and the causal prior (vision-only plus measured t0 ego state). Admissible. |
| ⚠️ B9-1 | it meets the letter ("no post-hoc stage") and not the intent ("the gain must be in the model"). **PI ruling needed.** |

### 2.2 (b) Learned per-horizon gain, conditioned on admissible t0 scalars, calibrated on a held-out split

| | |
|---|---|
| code | `ProgressRetime(mode="gain")`: `g(t_k) = g_deploy(t_k) + tanh(W_k·[v0, a0, ω0, 1])` (zero-init, so it starts at (a)) or a 3→16→8 MLP. **Do not** feed κ0 (curvature is not admissible as an input, PI 10-18). **Do not** feed the selection score, its margin or its entropy: refcv8's ranked score contains `β·log p(h_k)` from the tactical heads (`refs/refc.py:3885-3886`), which is a classifier-output back door. |
| params | flat per-horizon 8; linear 32; MLP ≈ 200 |
| training cost | ~0 GPU. Every input is a t0 scalar already in the dumps, so it can be **fit offline on banked rolls** of the calibration split, with everything else frozen, and written into the checkpoint. Or fit in a calibration fine-tune stage (minutes). |
| measured headroom | the v0-bin version exists: GATED_RETIME_TRAINFIT S2 −0.0348 / −0.0340 vs RETIME −0.0318 / −0.0313 (point −0.003), S6 −0.961 / −0.956 vs −0.976 / −0.973 (point +0.016 worse). Paired GATED − RETIME CI not computed, so **no claim** that conditioning helps. The scalar regressor (§1.5) bounds what ego-scalar conditioning can add: it ties RETIME. |
| can fix | the window-dependence of the optimal shrink (e.g. g\* ~0.1 at v0 < 2 m/s per H-R8-RS-1) |
| cannot fix | alignment (r). It only hedges a noisy choice. Fitted in-sample it inherits the selector's train/eval gap, hence §3. |
| admissibility | admissible with the inputs above |

### 2.3 (c) Loss-side fixes: the three the brief names, ranked by the localisation

| variant | where | prediction from §1 | verdict |
|---|---|---|---|
| **c1** along/cross reweighting of `loss_traj` | `refc_v3_train.py:5558-5560`: weight the along-track component of `recon − traj_tgt` in the GT path's local frame | acts on the a_star candidate, which is **already calibrated** (orc g\* 0.91, r 0.76, train = eval). Wrong locus: predicted ≈ 0 on E1, with a risk to the candidate's lateral fit. | **Not recommended** |
| **c2** along-track residual penalty on the emitted plan | needs the **inference-mode** emitted plan in training (training runs one random-t pass, `refs/refc.py:3079`, `:3114`), so +1 decoder pass per step under no-grad selection | a magnitude penalty shrinks ρ, but ρ is already 0.84 < 1. The defect is r. It would learn the stage, calibrated to train (0.69). | **Not recommended** |
| **c3** selection-side: soft full-geometry selection target (both axes) or a progress critic | selection CE (`:5553-5557`) / r8 critics | the only (c) that attacks r through the selector. **Precedent against it:** the X1 listwise target (arc-length speed profile + terminal heading) cost turns (PI-R8-NOLW: GT-turn cross 14.98 → 10.42 m when removed). refcv8's `progress_10pct` critic did not move r (Δr vs refcv7 −0.007 n.s.). | ladder-only, second line |

### 2.4 (c4) RECOMMENDED: a deterministic progress head that drives the re-timing layer

| | |
|---|---|
| what | `ProgressHead`: input `z_tac` (the shared tactical latent, `refs/refc_v3.py:1629`, d_tac 512 by default `:365`; **UNVERIFIED** for refcv8's build, not in `config.json`) + [v0, a0, ω0, fed ceiling] + `s_P` at the 8 slots → `Δŝ` [8]. Then `ProgressRetime(mode="head")`: `s' = s_P + g_cal(t) · Δŝ`, where g_cal is (b)'s held-out calibration, initialised at 1. |
| loss | `L_prog = Σ_k valid_k · |(s_P + Δŝ)_k − s_H,k|`. The human arc length along the human's own path, from `traj_tgt` (`refb_labels.waypoint_targets`, already built at `refc_v3_train.py:5529`). Weight ~0.1–0.5, a pre-registration parameter. |
| params | MLP 523→128→8 ≈ **68k** (ESTIMATED) = 0.07 % of refcv8's 96.4 M (`config.json` `ema.n_params` 96,435,990, MEASURED) |
| training cost | compute negligible. Two arms: **stop-grad on `z_tac`** (every other head trains bit-identically, so the effect is attributable) and **full-grad** (lets the trunk learn speed cues; changes the training dynamics, so it needs its own replicate). |
| why it fits §1 | it moves "how fast" from the selector (generalises badly: train − eval Δr +0.107 SEP) to an L1 regression (generalises: orc train = eval). It targets the 0.049-m PROG_ORC − RETIME gap. It is lateral-safe through 2.0. It is deterministic, so it removes the progress part of D4's inference variance at 1x cost. The same object (progress) carries D1, D3 and D2's along-track part: cruise along-track −2.04 m at 6 s vs refcv7 +0.16 (`L2_HEADROOM.md`). |
| cannot fix | path shape (D5 turns); a wrong *path* choice; anything the latent does not carry |
| floors it must beat | RETIME (−0.0318), PROG_SCALAR_OS (−0.0328, the ego-scalar head), and a deliberate-regression arm with `z_tac` replaced by a constant (must collapse onto PROG_SCALAR_OS) |
| admissibility | inference inputs are vision (through `z_tac`) plus measured t0 ego state plus the fed ceiling label, all already admissible. **Shared-trunk disclosure** (BINDING rule of 2026-08-03): `z_tac` also feeds `lat_head_tac` / `lon_head_tac` (`refs/refc_v3.py:1319-1320`). The head reads the latent **upstream** of those heads. It reads no posterior, argmax or embedding of them, and it is not a goal input, so no classifier output enters it. The stricter variant reads the pooled planner features instead of `z_tac`. |

---

## 3. A non-eval validation split for (b)'s and (c4)'s strength

**The constraint, MEASURED.** Anything the selector has trained on reads optimistic: A6 train g\* 0.686 vs eval 0.455, separated, and
the gap is in selection (§1.3). So the calibration split must be clips that **neither refcv9 nor its warm start** has trained on.

| option | clips | contamination | parity consequence | size | verdict |
|---|---|---|---|---|---|
| **V-A: the 203 refcv8-only clips** (in refcv8's 4,572 view, not in refcv6/refcv7's 4,369 cache: `a6/select_record.json` n_train_cache_clips 4,369 ⊂ 4,572 label records; INHERITED from `FEASIBILITY.md` §2 that both caches link into one pool) | 203 | **clean iff refcv9 warm-starts from refcv7-50,400** (`--v2-cache refcv6-b1-416x1024-train`, `refcv7-r101-s0.argv.json`) and excludes them. Seen once (≤ 1 pass) if it warm-starts from refcv8-final. | refcv9 trains on exactly **refcv7's corpus (4,369)**: same as refcv7, −203 clips (−4.4 %) vs refcv8. eval139 untouched. Say so in every refcv9 vs refcv8 row. | ≈ 7.1k S2 windows (ESTIMATED from the A6 rate 4,855 / 139 clips) | **recommended** if the PI picks the refcv7 warm start |
| V-B: A6-style content-hash holdout from the 4,572 (smallest sha12, K = 139–278) | 139–278 | **contaminated under any warm start**: refcv7 and refcv8 both trained on them | −3 to −6 % of clips vs refcv8 | 4.2k–8.4k | only for a from-scratch refcv9 (not on the table) |
| V-C: fresh PhysicalAI-AV clips outside the 4,711 used | any | clean | **none**: the training corpus is unchanged | as built | cleanest. Blocked on a corpus build (cache build cost per CLAUDE.md: PNG encode is 70 % of it). Availability of further clips UNVERIFIED. |
| **A6 139** (the train-fit sample) | 139 | in-sample for refcv7 and refcv8 | none | 4,211 compliant S2 / 137 ep | **cannot choose the strength** (MEASURED optimism). It can serve as the in-sample reference and the contamination control. |

**A cheap check that de-risks V-A before anyone relies on it** (GPU, dev box, after refcv8 frees nothing; Thor untouched):
* Roll refcv7-50,400 os-only on the 203 clips and on A6. Price: ~1.1 h + 0.75 h per inference seed (ESTIMATED from the measured
  2,730 s / 4,754 windows).
* Compare g\* / r. Committed outcomes:
  * **203 ≈ eval139 < A6** ⇒ the gap is clip memorisation, and V-A is a valid calibration split.
  * **203 ≈ A6 > eval139** ⇒ eval139 is distribution-shifted. No train-derived split can calibrate for it, and the honest
    calibration is the train one, scored on eval and reported as under-correcting.
* The 203 clips' frames must be pulled from Thor (~16.6 GB, ESTIMATED at 82 MB/clip from A6's 11.37 GB / 139) **after** refcv8
  finishes. Their representativeness (why they were added) is UNVERIFIED.

---

## 4. The cheapest pre-registrable experiment that ranks (a) vs (b) vs (c)

**E0, 0 GPU, done in this turn (§1).** Localisation, headroom, scalar floor, and refcv7 inheritance. (c1) and (c2) are already
ranked below the re-timing family on mechanism.

**E0-next, 0 GPU, ~1 h, before any GPU.**
1. Port `ProgressRetime` to torch.
2. Known-value control: the fixed mode on banked `raw/retime_deploy/DEPLOY8_POS_step30000ema_unc_s{0,1}.npz` inputs must reproduce
   those positions to ≤ 1e-5 m on uncapped windows.
3. Mutations: g = 1 must give os bit-for-bit; g = 0 must give the prior's arc length on the path.
4. Fit (b)-linear on train139 dumps and score it paired against RETIME_DEPLOY on eval139.

**E1, the ranking run (GPU, dev box RTX 4060 only; after refcv8-final is banked):**
1. **Bank `z_tac` once**: a deterministic trunk + tactical forward with no sampler, for ~20k train windows, the V-A clips if pulled,
   and eval139 (4,754).
2. Fit every head on CPU, many times.
3. **Price: ~3.5–4 GPU-h** (ESTIMATED: ≤ 0.57 s/window, the MEASURED os-only roll rate, which includes the sampler).
4. Engineering: a `z_tac` export key in the roll (`r7_roll` patch), ~½ day.

| arm | what | role |
|---|---|---|
| H0 | (a) RETIME_DEPLOY table, in-forward | the floor = refcv8 + stage |
| H1 | (b) linear-conditioned gain, fit on the calibration split | (b) |
| H2s | (c4) head with **scalar inputs only** | model-free floor; must reproduce PROG_SCALAR_OS within CI |
| H2 | (c4) head, `z_tac` + scalars, stop-grad (fit on frozen latents) | the lever |
| H2m | H2 with `z_tac` := its train mean (deliberate regression) | must collapse onto H2s (ValidateAIDesign) |
| H2r | H2 refit with another init seed and another 80 % subsample | head-level replicate (H-ESTIM-SEED-1 analogue) |

Every arm is scored at inference seeds 0 and 1 on refcv8-final's banked os rolls (the battery at 48,852).

**Bars (to be sealed in a PREREG before any E1 number exists):**
* **P1** H2 − ha0_ext_pose E1 separated < 0 at both inference seeds, with no stage outside the layer.
* **P2** H2 − H0 separated < 0 at both seeds.
* **P3** H2 − H2s separated < 0 at both seeds (vision adds).
* **P4** |H2 − H2r| < |H2 − H0|.
* **B9-0 guard:** no ADE / LONGITUDINAL / LATERAL / TACTICAL row separated worse than refcv8-final + RETIME_DEPLOY8 at both seeds;
  S6 ACC-LAT holds; missed decel not separated worse than os; false decel ≤ os. STRATEGIC n = 0 (`--no-strategic`), reported as such.

**Committed outcomes:**
* **O1, all pass** → refcv9 L1 = (c4). Train the head **jointly** in refcv9 (stop-grad arm first; full-grad only with a second
  training seed per B9-2).
* **O2, P3 fails** (the latent adds nothing over the scalars) → L1 = (a)/(b) in-forward. **refcv9 cannot beat refcv8 + stage on E1
  through L1**, and that is reported as such. The next levers are (c3) on the ladder (~4.4 Thor-h per arm, MEASURED ladder
  precedent; arm + control + replicate ≈ 13 h) and L3 K-draw distillation.
* **O3, P3 passes and P2 fails** → compose: `s' = s_P + g_cal·Δŝ` with g_cal fit on the calibration split; re-score on CPU.
* **O4, B9-0 fails** → the family named in the failure is the next diagnosis; nothing ships.

**Total for the ranking: 0 Thor-h, ~4 dev-box GPU-h, plus 2–4 h if the V-A check runs** (all ESTIMATED). The full refcv9 training
run is separate and PI-gated (§5 of the plan).

---

## 5. Risks

| risk | evidence | assessment |
|---|---|---|
| **Missed deceleration** (refcv8's advantage over refcv7) | Shrinking toward the prior **improves** missed decel: os 0.387 → DEPLOY8 0.358, −0.028 [−0.052, −0.005] SEP (s1 −0.024 SEP). refcv7 0.559 (S6 compliant). False decel 0.266 → 0.253 n.s. MEASURED, `L2_HEADROOM.md`. (The plan's 0.345 / 0.548 is a different window set.) | (a)/(b): **low**, measured. (c4): **UNMEASURED**. A conditional-median progress head can under-brake in rare sharp-decel windows, so E1's guard carries missed decel explicitly. |
| **Turns** | Re-timing keeps the path shape. ACC-LAT held for RETIME and TRAINFIT (heading −0.037 n.s.). Shrinking the forward *position* component (RS_LON) cost turns (heading +0.235° SEP), because it changed the path. The layer must re-time, never re-scale. | low for every re-timing form; **high for c3** (listwise precedent) |
| **Cruise false-brake (D2)** | DEPLOY8 absorbs 53 % of cruise ADE delta, 21 % of along-track excess, 2–7 % of false decel (`L2_HEADROOM.md`, label PARTIAL) | (c4) is the only L1 form that could take the rest, because D2's excess is a progress error |
| **Ceiling compliance** | the layer sits before the cap, so the cap still binds; re-timing toward a faster prior is re-capped | flagged equivalence caveat (§2.0) |
| **E1 is beatable without vision** | PROG_SCALAR_PRIOR −0.0212 SEP (§1.5) | carry H2s / PROG_SCALAR_OS as a mandatory control; report to the bar owner |
| **Single training seed** | every number above is one run's checkpoints | B9-2 replicate stays required for any refcv9 training-side claim |

---

## 6. Escalations, open items and corrections (for the orchestrator; not written into any other file)

1. **Correct `PLAN_REFCV9.md` §1 D1 and §2 L1.**
   * "runs ~2x too far ahead" should read "the emitted speed profile is mis-aligned with the human's (r 0.38, size ρ 0.84); its least-squares
     shrink is 0.46".
   * Localisation: selection.
   * Register `H-R8-RS-1`'s "the residual is too large" needs the same refinement.
   * New register rows are warranted for §1.2–§1.5. Suggested: **H-R9-L1-LOC-1** (selection, MEASURED), **H-R9-L1-INHERIT-1**
     (refcv7 same), **H-R9-L1-SCALARFLOOR-1** (E1 passed model-free, −0.0212).
2. **PI rulings:**
   * (i) does the fixed in-forward table (a) satisfy B9-1?
   * (ii) the warm start (refcv7 makes V-A clean; refcv8 leaves no clean split without V-C);
   * (iii) compute for the V-A check and E1.
3. **Bar owner:** E1 vs `ha0_ext_pose` is passed by a 10-coefficient ego-scalar regressor applied to the prior. Consider adding
   "beats PROG_SCALAR_PRIOR" beside E1 for refcv9. This is a *new* control, not a moved goalpost: it would be registered before
   any refcv9 number exists.
4. **Integration:** nothing here is wired. `ProgressRetime` / `ProgressHead` are designs. The probes live only in the qland hold.

---

## 7. Deliverable manifest

| artifact | where | copies |
|---|---|---|
| this annex | `qland:C:/Users/Admin/qland/hold_refcv9plan/stage/Project Steering/refcv9/L1_DESIGN_ANNEX.md` | **ONE place** (hold; not staged in any repo; no git writes per brief) |
| localisation probe (script, log, JSON) | `…/refcv9/l1_probe/along_track_localise.py`, `run.log`, `ALONG_TRACK_LOCALISE.json` | ONE place (hold) |
| headroom probe | `…/refcv9/l1_probe/progress_headroom.py`, `run_headroom.log`, `PROGRESS_HEADROOM.json` | ONE place (hold) |
| model-free scalar floor | `…/refcv9/l1_probe/progress_scalar.py`, `run_scalar.log`, `PROGRESS_SCALAR.json` | ONE place (hold) |
| refcv7 inheritance read | `…/refcv9/l1_probe/refcv7_alignment.py`, `run_r7.log`, `REFCV7_ALIGNMENT.json` | ONE place (hold) |
| inputs (read-only, not modified) | `C:/Users/Admin/refcv8_eval_archive/battery/step{30000,5000}ema_unc`, `E:/refcv8_eval_kit/trainfit/step30000ema_train139`, battery code `hold_r8batt/.../code/` (imported, not edited), launch tree `C:/lgs/Lfb47a58376/{stack,taniteval}` | n/a |

Reproduce any probe (CPU, ~10–50 s):
```
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code' C:/Users/Admin/venvs/tanitad/Scripts/python.exe -X utf8 <script>.py
```
