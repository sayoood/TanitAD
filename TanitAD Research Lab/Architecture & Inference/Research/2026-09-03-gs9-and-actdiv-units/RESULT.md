<title>RESULT — GS-9 read at last, and the actdiv units-invariance argument put to the test (2026-09-03)</title>

# RESULT — E-AI-GS9UNITS-0903

`TanitAD Research Lab · Architecture & Inference FlyWheel · 2026-09-03`
`Tier: T0-DIAGNOSTIC on EVERY number below — world-model probes, never driving performance.`
`Device: dev-box CPU for every reading. ⛔ Thor (refav1, until ≈ 2026-09-04 01:00Z) and the A40 pod`
`(refcv3, until ≈ 22:15Z) were NOT contacted. The RTX 4060 was probed`
`(nvidia-smi --query-compute-apps: desktop/browser only, no python) and left unused.`
`Evidence class: MEASURED (ours) unless marked. Every number has a raw path in §7.`
`Speed scale on every reading — Task A: v_last / SPEED_SCALE = v/10 (flagship_v15.SPEED_SCALE,`
`the v7 arms' own trained scale, applied by the trainer's own _lift3). Task B: refav1`
`SPEED_SCALE_MPS = 30.0 and INERT (cfg.speed_channel = False on both checkpoints, a_in_dim = 2).`
`SPEC.md was written before any pass was launched; §0 records exactly what was and was not`
`pre-registrable.`

MARKER: `E-AI-GS9UNITS-0903-RESULT`

---

## ⛔ HEADLINE FOR THE COORDINATOR — INTEGRATION, three items

1. **`taniteval/tools/transition_probe.py` IS STILL WIRED INTO NOTHING, and it has now produced a
   load-bearing negative.** It is the only instrument in the programme that answers L3's question
   on an admissible estimator (§3). It must enter the standard read-out path (`mm_e19_read.py`
   currently still calls the banked `actdiv_local.py`), or the next arm will again be judged on
   `envpred.loeo`, whose t-statistics `H-LEAK-2` withdrew.
2. **`run_panel()` iterates a FIXED `FEATURE_ORDER` tuple and silently skips any cell not in it.**
   That is a false-negative generator for anyone extending the panel — my `v_t` control would have
   vanished without a word. It needs an explicit "unknown feature" refusal. I did not patch it
   (not my file); the fix is a 2-line guard.
3. **Two proposed register rows in §5** (`H-GS9-1`, `D-ACTDIV-UNITS-INVARIANCE`) plus **one
   correction to a banked row** await adoption. `MODEL_REGISTRY.md` still has **no row** for
   `k8clip05p30k` (0 hits, two probes) — reported again, not papered over.

---

## 0. The answers, one line each

| question | answer | class |
|---|---|---|
| **GS-9 / R1** Does the encoder displacement `Δz` carry the ego transition? | **NO — ENC-TRANSITION-ABSENT, and stronger than the pre-registration asked.** `dz_enc` does not merely fail to beat the pixel floor: it reads the **constant's own value** (skill −0.0000 … +0.0005 on 4 targets, both arms). Under the RFF non-linear map, **raw pixels carry 5–200× more transition-specific structure than the learned `Δz`** (point estimates, no interval — §2.4). | MEASURED |
| **GS-9 / R2** Does the predictor's transition add anything over `z_t`? | **NO — L3-TRANSITION-FAIL on both arms (1 of 4 targets, bar was ≥ 2).** And ⛔ **the one target that passed is `dx_fwd`, the column contaminated by `v` at the model boundary** — so the single beat is inadmissible as a capability claim on top of failing the count. | MEASURED |
| **GS-9 / R3** Does the action reach the transition? | **PARTLY: `dzhat_true` beats `dzhat_zero` on `dv` only** (+0.0433 / +0.1119, CIs exclude 0) — and on `dx_fwd` the true action makes the prediction **WORSE** (−0.0076 / −0.0515, CIs exclude 0). The anchored response `dzhat_anch` **never exceeds the fed action `act2` on any target** (it carries 18–93 % of it). ⇒ **a lossy echo, not dynamics.** | MEASURED |
| **GS-9 / R4** Is the read about transitions? | **SPLIT, and this is the sharpest structural fact.** The action-carried cells are genuinely transition-specific (`dzhat_anch`→`dv` 93 %, →`dyaw` 91 %; `act2`→`dv` 90 %). The **prediction-carried** cells are **not**: `dzhat_true`→`dx_fwd` retains only **17 %** after a within-clip shuffle — 83 % of its headline 0.5703 is a **clip-level cue**. | MEASURED |
| **Is GS-9 admissible where L3 was not (`H-LEAK-2`)?** | **YES — and that is the headline.** L3's t's were pseudo-replicated (24 scores at ~92 % overlap treated as independent) and were withdrawn wholesale. GS-9 scores **every clip exactly once out-of-fold** under a 5-fold **clip-grouped** split, intervals are a **clip-cluster bootstrap over the 129 clips**, marginals are **paired on the same draws**, λ is chosen by clip-grouped inner CV **on the fit split only**, and both required controls read their known values. ⇒ **GS-9 is the first admissible read of L3's question on ego-transition targets, and it reads FAIL.** | MEASURED + analysis |
| **Task B** Does the *"separation is invariant under a monotone reparametrisation, so no number changes"* argument hold? | **PARTLY — `INVARIANCE-PARTIAL`. It is RIGHT about the separation test and WRONG about "no number changes".** The verdict (`LAT-INSENSITIVE-REFUTED`, 3 spaces × 2 arms × 2 conventions), the separation ratio (×0.965 / ×1.003), Spearman ρ = +1.000, the material-magnitude reading and every control are invariant — and the raw pass reproduces the banked run on **51/51 statistics at 6 s.f.** But **‖m(κ)‖/‖m(a)‖ moves ×2.84 / ×2.77 (0.00329 → 0.00935; 0.02514 → 0.06966)** — the very ratio the register calls *"the ratio that IS comparable"* — the three-significant-figure linearity claim moves (5.005 → 4.911), the *"compared at matched σ"* sentence is **false** under the curvature reading (0.1 rad/m = **6.09 σ_κ**, not 2.1σ), and the top-level antisymmetry weakens −0.9956 → −0.9615. | MEASURED |
| **Task B, unasked and only the converted pass could see it** | **The predictor's lateral path is a LINEAR map of the fed steer angle out to 0.2823 rad (16.2°)** — a 4.87× extension of the banked range — to within **+0.7 % / −1.9 %** of a pure linear response, **with no saturation**. That calibrates the queued h = 10 cost-surface probe: one measured gain per checkpoint, not a per-level grid. | MEASURED |

---

## 1. Task A — the admissibility audit (SPEC §2), item by item

Checked from the **raw JSON and the tool source**, never from the sibling's prose.

| # | rule | verdict | evidence |
|---|---|---|---|
| **A1** | λ on the FIT split only | ✅ **PASS** | `transition_probe.py:208 select_lambda` runs a clip-grouped inner CV over `X_fit` only; `crossfit:266` calls it with `X_all[tr]`; the shuffled variants **reuse the real-target λ and the real-target basis** (`:270-276`), so no control can be tuned into a favourable value |
| **A2** | PCA basis on the FIT split only | ✅ **NOT APPLICABLE, verified not assumed** | `panel.pca = 0` on both arms; `pca_fit` is never called, so the basis-selection trap cannot fire |
| **A3** | Constant-only control reads the no-information value EXACTLY | ✅ **PASS** | `cells.const.real.skill = [0.0, 0.0, 0.0, 0.0]` on both arms; `controls.constant_reads_exactly_zero = true`. It is exact **by construction**, not by luck: the all-ones column has `trace(XᵀX) = 0`, the guard at `_RidgeBasis:172` keeps λ > 0, β is exactly 0 and `pred == const` |
| **A4** | A raw-input floor exists **and is not degenerate** | ⚠️ **PARTIAL FAIL — the LINEAR floor is DEGENERATE** | `cells.pixdelta` reads **−0.0000 / +0.0000 / −0.0000 / −0.0000**, i.e. *the same value as A3*, and its λ sat at the **grid ceiling 1e4 on 3 of 5 folds** (grid = `logspace(-4, 4, 17)`). A floor that has shrunk to the constant **discriminates nothing**. ⇒ every R1/R3 statement of the form "beats the floor" is re-stated below as "**beats the CONSTANT**", which is the stronger and honest form. The **RFF floor is not degenerate** (0.0625 on `dx_fwd`) and becomes the operative floor in §2.4 |
| **A5** | Time-shuffled controls | ✅ **PASS** | `within_shuffle` (within-clip target permutation) and `global_shuffle` on **every** cell; max \|global\| over both arms and all 40 cell-target pairs = **0.0293**, inside the pre-registered [−0.05, +0.05] |
| **A6** | `n` and `d` printed | ✅ **PASS** | `n_score = 11,868` rows (129 clips × 92, no padded stacks), `n_fit_per_fold = 9,476…9,568`, `d ∈ {1, 2, 160, 1024, 2048, 4096}`. Worst case `n_fit/d = 9,476/4,096 = **2.31**` — thin but **not** the `n ≪ d` regime that manufactured the 2026-08-22 zeros |
| **A7** | ⛔ **`v` is NEVER an input** | ❌ **FAIL — at the model boundary, not in the feature list** | The feature list is clean. But `clip_features:521` calls `_lift3(a2, v0, cond_param)` with `cond_param = "steer_accel_v"`, and `_lift3` (`train_v6_staged.py:4692`) appends `v0 / SPEED_SCALE` as the predictor's **third channel**, with `v0 = poses[rows, 3]` — the pose speed at `t`. And `a3z[:, -1, :2] = 0` zeroes **only steer/accel**, so `dzhat_zero` keeps `v` too. ⇒ **`dzhat_true` and `dzhat_zero` are outputs of a predictor that was handed `v_t`.** Quantified in §2.3: this is not a theoretical concern, it is the entire `dx_fwd` column |
| **A8** | Estimator named on every interval | ✅ **PASS** | `clip_cluster_bootstrap(skill)` / `paired_clip_cluster_bootstrap(skill)`, `n_boot = 1000`, `n_clips = 129`, on every CI |
| **A9** | Speed scale = the arm's own trained scale | ✅ **PASS** | the probe does **not** roll its own lift: it imports the trainer's `_lift3`, whose divisor is `flagship_v15.SPEED_SCALE = **10.0**` — the same constant the v7 arms trained under. This is the defect (`v/30` fed to models trained on `v/10`) that invalidated the banked `actdiv` family this week, and **it is absent here** |
| **A10** | Linear-probe rule | ✅ **PASS, with the function class stated** | the `--mlp` RFF column (d = 1024, fixed seed, its own within- and global-shuffle) is the second function class and it **agrees** with the linear negative — see §2.4. Every negative below is written as *not decodable by these two map families*, never as *unlearnable* |

**Rescue.** A4 and A7 are real failures. **A4 is rescuable from the banked raw** (re-state against
the constant; use the RFF floor) and is rescued below. **A7 is NOT rescuable from the banked raw** —
the run contains no `v`-zeroed predictor arm and the JSON stores **aggregates only, no per-row
predictions** — so it was rescued by **measuring the leak's ceiling directly** on the identical rows
through the identical estimator (§2.3), which needs no model and no image decode.
⛔ **What genuinely cannot be rescued:** R4's own fallback ("re-read R1–R3 on the transition-specific
column") asks for a **CI on `real − within`**. That needs row-level predictions, which the raw does
not hold. Its point estimates are reported in §2.5 **without intervals and without a verdict** —
elevating a point estimate to a verdict is precisely probe-trap failure #2.

---

## 2. Task A — the read

Population, both arms, identical: `physicalai-val130-heldout`, **129 clips**, first 100 frames,
dataset-consistent stacked rows, W = 6, one tick (k = 1 = 0.1 s), **n = 11,868**, split **by clip**,
5-fold outer × 5-fold inner. Targets `dx_fwd, dy_left, dyaw, dv` from `poses`.
Arms: `postrain30k` (md5 `a58585883c27…`, registry §13.9) and `k8clip05p30k`
(md5 `2d744d6d2faa…`, ⚠️ **no registry row** — its model facts come from the k8 package's raw JSON).

### 2.1 The panel (skill = 1 − SSE(pred)/SSE(fit-split mean), pooled out-of-fold)

`postrain30k` | `k8clip05p30k` — **real** skill, then the **transition-specific** column
(real − within-clip-shuffle) in brackets:

| cell | d | `dx_fwd` | `dy_left` | `dyaw` | `dv` |
|---|---|---|---|---|---|
| `const` | 1 | **0.0000** \| **0.0000** | 0.0000 \| 0.0000 | 0.0000 \| 0.0000 | 0.0000 \| 0.0000 |
| `pixdelta` (linear, ⚠️ λ at ceiling) | 160 | −0.0000 \| −0.0000 | 0.0000 \| 0.0000 | −0.0000 \| −0.0000 | −0.0000 \| −0.0000 |
| `act2` (the fed action) | 2 | −0.0025 \| −0.0025 | 0.0839 [0.0595] | **0.3785** [0.2995] | **0.5856** [0.5256] |
| `z_t` | 2048 | −0.0226 \| +0.0400 | −0.0243 \| −0.0082 | 0.0025 \| 0.0574 | −0.0148 \| −0.0314 |
| `dz_enc` | 2048 | **−0.0000** \| **−0.0000** | 0.0000 \| −0.0001 | −0.0003 \| −0.0002 | 0.0005 \| 0.0003 |
| `dzhat_true` | 2048 | **0.5703** [0.0991] \| **0.3973** [0.0780] | −0.0458 \| −0.0476 | −0.0265 \| −0.0293 | 0.0087 \| 0.0628 |
| `dzhat_zero` | 2048 | **0.5779** \| **0.4488** | −0.0447 \| −0.0411 | −0.0333 \| −0.0227 | −0.0346 \| −0.0491 |
| `dzhat_anch` | 2048 | 0.0897 \| 0.0713 | 0.0154 \| 0.0262 | **0.2240** [0.2037] \| 0.1309 [0.1209] | **0.4826** [0.4499] \| **0.5305** [0.4896] |
| `z_t+dz_enc` | 4096 | −0.0308 \| 0.0424 | −0.0276 \| −0.0073 | 0.0012 \| 0.0616 | −0.0210 \| −0.0316 |
| `z_t+dzhat_true` | 4096 | 0.0522 \| 0.1151 | −0.0308 \| −0.0103 | −0.0201 \| 0.0752 | −0.0159 \| −0.0322 |

### 2.2 The four pre-registered readings, decided on the committed rule

"A beats B" = paired clip-bootstrap 95 % CI excludes 0 **AND** |Δskill| ≥ 0.02.

| read | `postrain30k` | `k8clip05p30k` | verdict |
|---|---|---|---|
| **R1** `dz_enc` − `pixdelta` | `dx_fwd` −0.0000 [−0.0001, +0.0001]; `dyaw` −0.0002 [−0.0004, −0.0000] | `dx_fwd` +0.0000 [−0.0000, +0.0001]; `dyaw` −0.0001 [−0.0002, +0.0000] | ⛔ **ENC-TRANSITION-ABSENT** on both arms. The two "separated" cells (`dyaw`, `dv`) miss the 0.02 effect floor by **40–100×** — exactly the case the floor was written to stop a CI firing on |
| **R2** `z_t+dzhat_true` − `z_t` | `dx_fwd` **+0.0748** [+0.018, +0.133] ✅; `dy_left` −0.0065; `dyaw` −0.0226 [−0.054, **+0.0002**] (CI includes 0); `dv` −0.0010 | `dx_fwd` **+0.0752** [+0.051, +0.102] ✅; `dyaw` +0.0178 [+0.005, +0.029] (CI excludes 0 but **|Δ| < 0.02**); others no | ⛔ **L3-TRANSITION-FAIL**, **1 of 4** on both arms (bar: ≥ 2). **And the single passing target is `dx_fwd` — see §2.3** |
| **R3** `dzhat_true` − `dzhat_zero` | `dv` **+0.0433** [+0.026, +0.062] ✅; `dx_fwd` **−0.0076** [−0.012, −0.003] (**wrong sign**, CI excludes 0) | `dv` **+0.1119** [+0.081, +0.141] ✅; `dx_fwd` **−0.0515** [−0.075, −0.027] (**wrong sign**, |Δ| ≥ 0.02) | **ACTION-REACHES-TRANSITION — on `dv` only.** On `dx_fwd` the true action **degrades** the prediction against a zero-action one, significantly so on `k8clip05p30k` |
| **R3 echo test** `dzhat_anch` vs `act2` | gaps −0.069 / −0.155 / **−0.103**; ratios 0.18 / 0.59 / **0.82** | gaps −0.058 / −0.248 / **−0.055**; ratios 0.31 / 0.35 / **0.91** | **NOT an exact echo (no gap ≤ 0.02) — a LOSSY one.** ⭐ The load-bearing fact is the **sign**: `dzhat_anch` is **below `act2` on every target of both arms**. The predictor's anchored action response carries **strictly less** about the transition than the 2-number action it was handed. It adds nothing |
| **R4** transition-specific ≥ 0.5 × real | `dzhat_true`→`dx_fwd` **0.174**; `dzhat_zero`→`dx_fwd` 0.171; `dzhat_anch`→`dx_fwd` 0.188 — **FAIL**. `dzhat_anch`→`dv` 0.932, →`dyaw` 0.909; `act2`→`dv` 0.898 — **HOLD** | `dzhat_true`→`dx_fwd` **0.196**; `dzhat_zero` 0.179; `z_t+dzhat_true`→`dx_fwd` 0.324 — **FAIL**. `dzhat_anch`→`dv` 0.923, →`dyaw` 0.924 — **HOLD** | **SPLIT.** Everything the *action* carries is transition-specific; everything the *prediction* carries on `dx_fwd` is **83 %** a clip-level cue |

### 2.3 ⛔ The A7 failure, measured — `dx_fwd` is a `v`-echo, and it is the only column that passed

`_lift3` hands the predictor `v_t / 10` as its third channel. Over one tick `dx_fwd ≈ v_t · dt`. So
the question is not whether `v` leaks but **how much of the banked 0.5703 it accounts for**. Measured
on the **identical 11,868 rows / 129 clips**, through `transition_probe`'s **own** `crossfit` /
`skill_score` / `clip_bootstrap_skill` (imported by file, never copied), with **no model and no
image decode** — `raw/gs9_vleak.json`:

| cell | d | `dx_fwd` | `dy_left` | `dyaw` | `dv` |
|---|---|---|---|---|---|
| `const` (control) | 1 | **0.0000** | 0.0000 | 0.0000 | 0.0000 |
| **`v_t` alone** | **1** | **+0.9986** [+0.9970, +0.9994] | +0.0082 | −0.0025 | −0.0022 |
| `act2` | 2 | −0.0025 | +0.0839 | +0.3785 | +0.5856 |
| `act2 + v_t` (the FULL 3-channel tuple `_lift3` builds) | 3 | +0.9967 | +0.0875 | +0.3654 | +0.5831 |

raw Pearson `r(v_t, target)` = **`dx_fwd` +0.9997**, `dy_left` −0.1035, `dyaw` −0.0579, `dv` −0.0586.
*(The brief's r 0.9988 is the same phenomenon measured on a different population; here it is 0.9997
on 11,868 rows.)* Controls: `const` exactly 0.0000; every global-shuffle ≤ |0.0006|; no λ at a grid
edge. ⭐ **`act2` reproduces the banked panel to four decimals on all four targets
(−0.0025 / +0.0839 / +0.3785 / +0.5856)** — an independent reproduction of the row construction,
which is what makes this control trustworthy rather than merely convenient.

**What it settles.**

* **A single scalar the predictor is handed scores 0.9986 on `dx_fwd`.** The banked `dzhat_true`
  scores **0.5703** on the same column from **2,048 dimensions**. In residual terms that is
  `√(1−0.5703) = 0.655` against `√(1−0.9986) = 0.037` — **≈ 18× the residual error of a 1-d readout
  of the raw input.** The predictor is not transporting the transition on `dx_fwd`; it is
  **degrading a number it was given**, and the probe was reading that degraded copy.
* **The clip-level cue is the same cue.** `v_t`'s own within-clip-shuffle reads **0.9010** (90 %) —
  clips have characteristic speeds. `dzhat_true`'s within-clip-shuffle reads **0.4711** of 0.5703
  (83 %). The R4 failure and the `v` leak are **one phenomenon**, not two.
* **The leak is surgical.** `r(v_t, ·)` is ≤ 0.104 on the other three targets, and `act2 + v_t` does
  not beat `act2` on `dy_left`/`dyaw`/`dv` (paired Δ +0.0035 / −0.0131 / −0.0025, all CIs include 0).
  ⇒ **only `dx_fwd` is contaminated.** R3's `dv` beat and `dzhat_anch`'s `dyaw`/`dv` scores are
  **clean of `v`** and stand.
* ⇒ **R2's single passing cell is inadmissible**, on top of R2 already failing 1-of-4. The
  L3-TRANSITION-FAIL verdict does not depend on this — but the *only* number that could have been
  quoted the other way is now excluded on its own evidence.

⚠️ **`dv` has its own provenance caveat, and it is not the same defect.** `act2`'s accel channel is
the dataset's **measured longitudinal `ax`** (`physicalai.py:625`, explicitly *not* a finite
difference of `v`), while the target `dv = v_{t+1} − v_t` is the difference of the interpolated speed
column. They are the same physical quantity read through two instruments, so `act2 → dv` at 0.5856
is **near-tautological** — the same class as `H-LEAK-6`'s "the positive action sequence is the
target's realised motion". ⇒ R3's `dv` beat says the predictor's latent displacement **retains a
readable copy of the accel input**, which is what "echo" means. It is not evidence of dynamics.

### 2.4 A4's rescue — the operative floor, and the finding it exposes

The linear pixel floor is degenerate (λ at the ceiling, reads the constant's value). The **RFF**
floor is not, and it is **model-independent** — it reads **identically on both arms**
(+0.0625 / −0.0041 / +0.0233 / −0.0002), which is a hidden control reading its known value.

Transition-specific column (real − within-clip-shuffle), RFF, d = 1024:

| cell | `dx_fwd` | `dyaw` |
|---|---|---|
| **`pixdelta` (raw 8×20 gray difference)** | **+0.0193** | **+0.0204** |
| `dz_enc` — `postrain30k` | +0.0001 | +0.0017 |
| `dz_enc` — `k8clip05p30k` | +0.0041 | +0.0015 |
| `dzhat_true` — `postrain30k` | +0.0260 | +0.0047 |
| `dzhat_true` — `k8clip05p30k` | +0.0097 | +0.0061 |

⛔ **The learned encoder displacement is beaten by raw pixels on the transition-specific column, by
5–200×.** CLAUDE.md's floor rule states the consequence directly: *a learned representation that does
not beat raw input has added nothing.*

⚠️ **How far this may be pushed, and no further.** The `mlp_rff` cells carry **no CI** in the banked
JSON, so these are **point estimates only** and are **not** a pre-registered "beat". They are quoted
as the second function class required by the linear-probe rule, and they *agree* with the linear
negative rather than overturning it — which is exactly what that rule asks for. The admissible
sentence is: **`Δz` carries no ego transition decodable by a ridge-linear map, and the RFF read
gives no indication that a modest non-linearity recovers one, while the same non-linearity does find
transition-specific structure in raw pixels.**

### 2.5 R4's fallback, and where the raw runs out

R4 fails on the winning `dx_fwd` cells, so its own rule asks for R1–R3 re-read on the
transition-specific column. Point estimates for R2's marginal (`z_t+dzhat_true` − `z_t`):

| arm | real marginal (with CI) | transition-specific marginal (point only) |
|---|---|---|
| `postrain30k` | +0.0748 [+0.018, +0.133] | +0.0356 − 0.0140 = **+0.0216** |
| `k8clip05p30k` | +0.0752 [+0.051, +0.102] | +0.0373 − 0.0267 = **+0.0106** |

⛔ **These have no interval and the raw cannot produce one** (aggregates only, no per-row
predictions), and they **straddle the 0.02 floor from both sides**. ⇒ the transition-specific re-read
**cannot decide R2 in either direction**, and R2 stands on the pre-registered rule alone
(1 of 4 ⇒ FAIL). This is stated rather than resolved on purpose: reporting +0.0216 as a pass would
be probe-trap failure #2 (a point estimate promoted over a missing interval).

---

## 3. GS-9 against `H-LEAK-2` — the admissibility comparison, which is the headline

`D-P2-LEAK-AUDIT`'s `H-LEAK-2` withdrew **every L3 / E-DEC-28b / E-DEC-29 t-statistic**:
`envpred.loeo` (`envpred.py:211-217`) scored a **pooled 12-clip second half per iteration**
(`v7tiny_probe.py:180-186`) and **24 overlapping scores (~92 % shared rows) were treated as
independent** (`:238-241`). Its verdict: *"L3 has never been passed or failed on an admissible
read."* In the banked form a **constant** beat `z_t` at "t 4.7" on `rdw8p30k`.

| defect in L3 | what GS-9 does instead | verified at |
|---|---|---|
| 24 scores with ~92 % row overlap treated as independent | **5-fold outer split BY CLIP; every clip scored exactly ONCE out-of-fold.** There is no score reuse to pseudo-replicate | `crossfit:250-278`, `grouped_folds:196` |
| a t-statistic whose replication unit was undefined | **clip-cluster bootstrap over the 129 clips** — the cluster IS the dependence unit, resampled with replacement | `clip_bootstrap_skill:314`; `estimator` stamped on every CI |
| cross-arm deltas assembled from separate scores | **paired** bootstrap on the **same clip draws**, never combined in quadrature | `clip_bootstrap_skill(..., pred_b=...)` |
| a constant beating the baseline unnoticed (negative cross-clip R² baseline) | the metric is **skill against the FIT-SPLIT MEAN**, so a constant reads **exactly 0.0000 by construction** and cannot beat anything | `skill_score:302`; `const` reads 0.0000 on both arms |
| hyper-parameters not separated from scoring | λ by **clip-grouped inner CV on the fit split only**; shuffled controls scored **through the identical fits** | `select_lambda:208`, `crossfit:270-276` |

⇒ **GS-9 is admissible where L3 was not.** It is the first admissible read of the question L3 was
built to answer — *does the predictor's output add information beyond the current state?* —
**on EGO-TRANSITION targets**, and on both local 30k arms the answer is:

> ### **L3-TRANSITION-FAIL.** The predictor's one-tick latent displacement does not add decodable information about the ego transition beyond `z_t` — **1 of 4 targets**, against a bar of 2, and that one target is the `v`-echo column (§2.3).

⭐ **AND IT IS NOT ALONE — `H-LEAK-2`'s OWN CORRECTED READ AGREES, ON A DIFFERENT TARGET AND A
DIFFERENT ESTIMATOR.** The audit did not only withdraw the t's; it re-ran L3 properly and reported
that *under true leave-one-clip-out with within-clip r, **no column separates from `z_t`** on
`n_agents` for `rdw8p30k` / `postrain30k` / `postrain30k_freeze` (|t| ≤ 1.5, every K)* — and, in the
same sentence, that **raw pixels DO separate** where the `ẑ` columns do not. GS-9 reaches both
conclusions independently: no `ẑ` column beats `z_t` (§2.2 R2), and **raw pixels carry 5–200× more
transition-specific structure than the learned `Δz`** (§2.4). Two instruments, different targets
(`n_agents` vs ego `Δ`), different estimators (LOO + within-clip r vs clip-grouped K-fold +
clip-cluster bootstrap), different corpora — **same two answers.** That agreement is what makes
this quotable; a single probe agreeing with itself is the `ls-tree` trap.

Two boundaries travel with it, and neither is optional:

1. **Function class.** This is a negative about a ridge-linear map, checked against one RFF
   non-linearity that agrees. It is **not** a claim that the information is absent from `Δẑ`.
2. **Scope.** Two arms, one tick, one corpus, `T0`. `o11p30k`, `splitp30k` and `postrain30k_freeze`
   are Thor-only and were **not** run — the tool takes one CLI invocation each when Thor is free.

**What it does to `P5` and `P2`.** The launch gate's phrasing — *"the predictor is not transporting
the scene, it is restating it"* — now has an admissible transition-level measurement behind it, and
a sharper form: on `dx_fwd` the predictor is **not even restating** its input faithfully, it is
**attenuating a handed scalar to 57 %**. `P2` narrows in the same motion: the action **does** reach
the latent transition (`dzhat_anch` scores 0.22–0.53 on `dyaw`/`dv`, 91–93 % of it
transition-specific), but it arrives **strictly degraded relative to the two numbers that were fed**
— never above `act2` on any target of either arm. So `P2` at the transition level is a
**transmission-loss** problem, not an absence-of-wiring problem, which is the same conclusion GS-8's
anchored read reached from the geometry side (`SEPARATED-NONMONOTONE`, gain ~10⁻³ of the scene).

---

## 4. Task B — the units-invariance argument, TESTED

`D-ACTDIV-ANCHORED-REFAV1` recorded, after its run, that the swept channel is a road-wheel **STEER
ANGLE** (`physicalai.py:621` writes `arctan(wheelbase·curvature)`), and argued that **"no number
changes"** because separation is invariant under a monotone reparametrisation. That argument was
inherited into the register (the `D-ACTDIV-ANCHORED-REFAV1` row; and `D-UNITS-CHECKPOINT-TEST-VOID`
queued *"the admissible instrument is `actdiv_anchored`'s `d(a)` run in BOTH conventions"*). **It is
now measured, not inherited.**

**What was run.** `actdiv_anchored.py --family refav1` on **both** banked step-1,000 checkpoints,
**twice**, differing in **exactly one thing** — the number placed on channel 1:

| convention | flag | what the model receives at nominal level `L` |
|---|---|---|
| **raw** (the banked path) | `--action-units kappa` | `L` unchanged |
| **converted** | `--action-units steer` | `arctan(2.9 · L)` via `tanitad.models.kinematic.as_command` |

`L_enc = STEER_WHEELBASE_M = 2.9`, the **ENCODING** constant (⛔ never cross-applied to ZOD / l2d /
alpasim; the tool refuses the flag on `--family v7`). Fed values, printed by the tool and stamped in
both JSONs: `0.02 → 0.0579351`, `0.05 → 0.1439964`, `0.1 → 0.2822574` rad. Same 140 windows
(20 eval-slice episodes, `k_loader = cfg.op_steps = 30`, stride 10), `--n-perm 200 --seed 0`,
`h = 1`, verdict space `tac`, **CPU**, ~22 min per arm per convention.

**Speed channel, printed as required.** Both checkpoints carry `cfg.speed_channel = False`
(verified from the primary artefacts: `ckpt_ep2/config.json`; the incumbent's `ckpt['cfg']` as
loaded), so `a_dim 2 → a_in_dim 2` and `RefAV1.augment_actions` (`refa_v1.py:1263`) returns the
`(a, κ)` controls unchanged. The model's own constant is `SPEED_SCALE_MPS = **30.0**` and it is
**INERT on this read** — all four passes printed
`SPEED_SCALE_MPS=30 (UNUSED: speed_channel=False, predictor input is (a,kappa) only)` and banked
`speed_scale_effective: null`. ⇒ the `v/30`-vs-`v/10` defect that invalidated a banked instrument
family this week **cannot occur here**, structurally: the tool never widens the action tensor, it
calls the model's own `augment_actions`. *(The v7 arms' 10.0 does not enter Task B — no v7 arm was
run. It does bind Task A, where `_lift3` correctly uses it — audit item A9.)*

### 4.1 B0 — the reproduction gate

| arm | statistics compared | mismatch at 6 s.f. | verdict |
|---|---|---|---|
| `incumbent_fp32` (md5 `45b9f4d82a3a…`) | **51** | **0** | `LAT-INSENSITIVE-REFUTED` (banked: same) |
| `clean_epoch_ema_bf16` (md5 `c26c7ad00e5b…`) | **51** | **0** | `LAT-INSENSITIVE-REFUTED` (banked: same) |

The `--action-units kappa` pass reproduces the banked
`…/2026-09-03-anchored-actdiv-refav1/raw/actdiv_anchored_refav1_step1000.json` **exactly** across
all three spaces (`F_sep`, `F_over_null_p95`, `scene_spread`, `rel_units`,
`rel_mag_at_material_level`, `max_abs_displacement`, and each axis's `F_sep` / `F/p95` /
`null.p95`), the full-field `F_sep`, every `norm_by_level` and `sign_cos_by_level`, both controls
and both action σ. ⇒ **the flag is a true pass-through, the read is deterministic, and the banked
refav1 result is REPRODUCIBLE.** B0 HOLDS, so nothing below is VOID.

### 4.2 The argument's four axes, per arm

Verdict space `tac`, n = 140, T0-DIAGNOSTIC, CPU, speed scale 30.0 (inert).

| reading | `incumbent_fp32` raw → converted | `clean_epoch_ema_bf16` raw → converted | committed band | outcome |
|---|---|---|---|---|
| **B1 verdict**, 3 spaces | REFUTED → REFUTED (×3) | REFUTED → REFUTED (×3) | no flip | ✅ **INVARIANT** |
| **B2 κ-axis `F_sep`/null p95** | **2297.75 → 2217.60** (×0.965) | **11.755 → 11.787** (×1.003) | \|log(c/r)\| ≤ log 1.25 | ✅ **INVARIANT** |
| **B2 control — accel axis** (untouched) | 1036.481 → **1036.481** (×1.00000) | 31.617 → **31.617** | must be bit-identical | ✅ **bit-identical**, as it must be |
| **B3 rank monotonicity** Spearman ρ(\|level\|, ‖m‖) | +1.000 → +1.000 | +1.000 → +1.000 | +1.000 | ✅ **INVARIANT** |
| **B3 linearity digits** ‖m‖ ratios to level 0.02 | **1 : 2.5004 : 5.0046** → **1 : 2.4899 : 4.9113** | **1 : 2.4981 : 4.9799** → **1 : 2.4668 : 4.7682** | steer top 4.872 ± 0.05 | ⛔ **MOVES** — see §4.3 |
| **B4 antisymmetry** cos(m(+L), m(−L)), κ axis | −0.9998 / −0.9989 / −0.9956 → −0.9985 / −0.9907 / **−0.9615** | −0.9999 / −0.9995 / −0.9980 → −0.9993 / −0.9960 / **−0.9856** | HOLD ≤ −0.99 every level; FAIL > −0.95 | ⚠️ **LANDS BETWEEN THE TWO COMMITTED THRESHOLDS** — reported as such, not rounded to "holds" |
| **B5 ‖m(κ=0.1)‖ / ‖m(a=1.5)‖**, exact full field | **0.003287 → 0.009348** (**×2.843**) | **0.025144 → 0.069661** (**×2.770**) | ≤ ×1.25 to be invariant | ⛔ **MOVES, ×2.8** |
| **B5** same ratio, `tac` / `pooled` | 0.002244 → 0.006360 (×2.834) / 0.003287 → 0.009347 (×2.843) | 0.026194 → 0.072188 (×2.756) / 0.025028 → 0.069302 (×2.769) | — | ⛔ **MOVES in every space** |
| **material magnitude** (rel-mag at the material level; bar 5.95 %) | 0.375 % → **0.377 %** | 1.251 % → **1.307 %** | — | ✅ **effectively INVARIANT** — the aggregate is dominated by the **untouched accel axis** (`max_abs_displacement` 0.2317 / 0.0413, identical in both conventions), so "structured but NOT MATERIAL" survives |
| **controls** | C0 `0.0`, zero-model `0.0`, zero anchor `[0.0, 0.0]`, scene spread 4.249215 — identical across conventions | same; scene spread 0.089146 | must read known values | ✅ **all at their known values in all four passes** |

### 4.3 ⭐ What the converted pass could measure and the raw one could not

The conversion multiplies the stimulus by the **arctan gain**, which is **not a constant** —
×2.8968 / ×2.8799 / ×2.8226 at the three levels. Measured gain (converted ‖m‖ ÷ raw ‖m‖, exact full
field over all 655,360 coordinates):

| level | arctan gain | `incumbent_fp32` | `clean_epoch_ema_bf16` |
|---|---|---|---|
| 0.02 | 2.8968 | **2.8975** (+0.03 %) | **2.8935** (−0.11 %) |
| 0.05 | 2.8799 | **2.8853** (+0.19 %) | **2.8572** (−0.79 %) |
| 0.10 | 2.8226 | **2.8435** (+0.74 %) | **2.7705** (−1.85 %) |

⇒ **The predictor's lateral path is essentially a LINEAR MAP OF THE FED STEER ANGLE out to
0.2823 rad (16.2°) — a 4.87× extension of the range the banked grid probed — to within +0.7 %
(incumbent, mildly super-linear) and −1.9 % (clean epoch, mildly sub-linear) of a pure linear
response. There is NO saturation.** That is a new measurement; the banked read stopped at 0.1 rad
and could not have produced it. It is also the fact that decides B3: because the model is linear in
the **fed** value, it cannot also be linear in the **nominal** level once the two differ by a
non-constant gain — which is exactly why 5.005 becomes 4.911.

⚠️ **And one thing does degrade with the larger stimulus: SIGN CONSISTENCY.** On the incumbent the
top-level antisymmetry falls **−0.9956 → −0.9615** — `m(+L)` and `m(−L)` acquire a ~16° common
component at a 5.93 σ steer input. It is far from the committed FAIL threshold (> −0.95) and far
from the accel axis's known rotation (−0.7972 at a = 1.5), but it is a **real, stimulus-monotone
weakening that a "nothing changes" argument asserts cannot exist.** It is the same even-order
response GS-8 found on the accel axis, appearing on the lateral axis once the lateral axis is driven
as hard.

### 4.4 The σ question — where the argument is exactly right, and where it is exactly wrong

σ of the windows' own channel-1 actions, **measured in the channel as fed** (identical on both arms
— the same 140 windows): **σ_steer = 0.047565 rad**, curvature equivalent
`tan(0.047565)/2.9 =` **σ_κ = 0.0164140 rad/m**. σ of channel 0 = **0.742635 m/s²**.

| stimulus | under the STEER reading (what the channel IS) | under the CURVATURE reading (what the level was CALLED) |
|---|---|---|
| `a = 1.5` | **2.020 σ_a** | 2.020 σ_a |
| level `0.1`, raw pass (fed 0.100000 rad) | **2.102 σ_steer** — matched to the accel axis | — |
| level `0.1`, converted pass (fed 0.282257 rad) | **5.934 σ_steer** | `0.1 rad/m` = **6.092 σ_κ** |

⇒ **`D-ACTDIV-ANCHORED-REFAV1` §4.2's sentence — *"the two axes are compared at matched σ, which is
why this ratio is meaningful"* — is TRUE as a STEER statement and FALSE as a CURVATURE statement.**
Read the level as a curvature and the ratio compares a **6.1 σ** lateral input against a **2.0 σ**
longitudinal one; `0.00329` is then not a matched-σ ratio at all.

⭐ **But the argument's core intuition survives in a precise form, and it deserves to be stated
because it is where the reparametrisation genuinely does nothing:** at ~2 σ the arctan map is within
**0.23 %** of the identity. The level that is exactly `2 σ_κ` (0.0328279 rad/m) is fed as
**0.094915 rad = 1.9955 σ_steer**, against `2 σ_steer = 0.09513`. ⇒ **a σ-MATCHED comparison IS
convention-invariant. Only comparisons pinned to a fixed NOMINAL level are not** — and the banked
ratio is pinned to `κ = 0.1`, which is 2.1 σ in one reading and 6.1 σ in the other.

### 4.5 VERDICT — `D-ACTDIV-UNITS-INVARIANCE` = **INVARIANCE-PARTIAL**

Decided on SPEC §4's committed three-way table; no clause was added after the run.

> **The argument is RIGHT about what a separation test measures and WRONG about "no number
> changes".** The verdict, the rank-monotonicity, the separation ratio, the material-magnitude
> reading and every control are invariant. **Two families of banked statement are not: the linearity
> digits, and every magnitude quoted at a fixed nominal κ level — including the ratio the register
> calls "the ratio that IS comparable".**

**The banked numbers that MOVE, named exactly so they can be re-quoted rather than re-derived:**

| where | banked statement (raw = steer convention) | under the curvature convention | move |
|---|---|---|---|
| `GOALS_AND_CLAIMS.md` `D-ACTDIV-ANCHORED-REFAV1` point (4); package `RESULT.md` §4.2 | ‖m(κ)‖/‖m(a)‖ = **0.00329 → 0.02514** across the two checkpoints | **0.00935 → 0.06966** | **×2.84 / ×2.77** |
| same row, the "three-factor" paragraph | *"the lateral channel moves the imagination only **1/300th** (incumbent) to **1/40th** (clean epoch) as far as the longitudinal one"* | **1/107th** to **1/14th** | ×2.8 |
| package `RESULT.md` §4.1 | *"norms scale as **1 : 2.500 : 5.005** and **1 : 2.498 : 4.980** … proportional to κ **to three significant figures**"* | **1 : 2.490 : 4.911** and **1 : 2.467 : 4.768** | the three-significant-figure claim is **convention-dependent** |
| package `RESULT.md` §2.5 clause (2) and §4.2 | *"σ is measured in the same channel, so 0.1 ≈ 2.1σ is exact regardless of what the channel denotes"*; *"the two axes are compared at matched σ"* | 0.1 rad/m = **6.09 σ_κ**; fed as a curvature it is a **5.93 σ_steer** stimulus | the σ-matching claim is **true only under the steer reading** |
| package `RESULT.md` §4.1 | cos(m(+L), m(−L)) = **−0.9956** at the top level (incumbent) | **−0.9615** | sign consistency **weakens with stimulus** |

**The banked statements that STAND — unchanged, and now measured rather than argued:**
`LAT-INSENSITIVE-REFUTED` on both checkpoints in all three spaces under both conventions; the κ-axis
separation ratio (2297.75 → 2217.60, ×0.965; 11.755 → 11.787, ×1.003); Spearman ρ = +1.000; the
"structured but NOT MATERIAL" reading against the 5.95 % bar (0.375 → 0.377 %, 1.251 → 1.307 %); and
every control at its known value in all four passes.

⚠️ **What this does to §2.5's *"nothing measured here is invalidated"*: it is HALF RIGHT.** No
*verdict* is invalidated, and its reason (1) — separation is invariant under a monotone
reparametrisation — is **confirmed, to ×0.965 / ×1.003**. Reasons (2) and (3) are the ones that
fail. (2) The σ normalisation is convention-safe only near ~2 σ, and the banked grid's top level is
**not** at 2 σ under the curvature reading. (3) `tan δ ≈ δ` to 0.34 % holds over **0…0.1 rad**,
which is the range of the *fed* values in the raw pass — it does **not** license carrying the
linearity claim into the curvature convention, where the fed range is 0…0.2823 rad and the
level-to-fed gain varies **2.6 %** across the grid. The measured deviation (−1.9 % … +0.7 %) is the
same order as that 2.6 %, which is precisely why the third digit moves.

⭐ **What it settles for the next probe.** `D-UNITS-CHECKPOINT-TEST-VOID` and
`D-ACTDIV-ANCHORED-REFAV1` both queue the **cost-surface probe at h = 10 in both conventions**. This
read supplies its calibration for free: over 0…0.2823 rad the predictor's lateral response is
**linear in the fed steer to within ±2 %**, so a cost-surface sweep may use a **single measured
lateral gain per checkpoint** instead of a per-level grid, and `C-STEER-CURVATURE-INTERFACE`'s ×2.9
boundary defect propagates into the goal term as a **clean ×2.8 factor on the field displacement**,
with the `0.05·κ²` penalty still charged at the full proposed κ. The compounding argued in the
banked row is therefore **correct in direction, and its multiplier is now measured (×2.77–2.84)
rather than assumed from the wheelbase.**

---

## 5. Proposed register rows

Two NEW rows and one correction, drafted verbatim-ready in
`PROPOSED_REGISTER_ROWS.md` beside this file. **Status PROPOSED, not applied** — the brief scopes me
to read-only outside this package, and that conflict with CLAUDE.md's "update the register in the
same turn" is escalated rather than resolved unilaterally.

* **`H-GS9-1`** (NEW; `GOALS_AND_CLAIMS.md` has **0 hits**, two differently-bound probes) —
  L3-TRANSITION-FAIL + ENC-TRANSITION-ABSENT + ACTION-ECHO-ONLY, with the `v`-boundary audit finding
  and the `H-LEAK-2` admissibility comparison.
* **`D-ACTDIV-UNITS-INVARIANCE`** (NEW; **0 hits**, two probes) — **INVARIANCE-PARTIAL**, naming the
  five banked statements that move and the five that stand.
* **A CORRECTION to the standing `D-ACTDIV-ANCHORED-REFAV1` row** — its point (4) quotes
  ‖m(κ)‖/‖m(a)‖ = 0.00329 → 0.02514 as *"the ratio that IS comparable"*, and the "three-factor"
  paragraph quotes *1/300th to 1/40th*. Both are correct **as steer-convention numbers** and must
  carry that stamp; under the curvature convention they are **0.00935 → 0.06966** and
  **1/107th to 1/14th**. This should not wait — the row is being quoted in the v7f design evidence
  table.

## 6. Raw artifacts

| artifact | what it is |
|---|---|
| `raw/transition_probe_v7.json` *(in the SIBLING package `…/2026-09-03-anchored-actdiv-and-transition-probe/raw/`)* | the banked GS-9 panel this task read; **not copied**, cited in place |
| `raw/gs9_vleak.json` + `.log` | **NEW** — the `v`-leak ceiling control (audit A7): `const`, `v_t`, `act2`, `act2+v_t` on the identical 11,868 rows through `transition_probe`'s own estimator |
| `raw/actdiv_units_kappa.json` + `.log` | **NEW** — `--action-units kappa`, both refav1 checkpoints (the B0 reproduction pass) |
| `raw/actdiv_units_steer.json` + `.log` | **NEW** — `--action-units steer`, both refav1 checkpoints (the converted pass) |
| `raw/units_tables.txt` | **NEW** — the B0–B5 decision output verbatim, as produced by `analyse_units.py` |
| `gs9_vleak.py`, `analyse_units.py`, `run_units.ps1` | the three scripts, so every number above is re-derivable |
| checkpoints (md5) | `incumbent_fp32 45b9f4d82a3a7f15bda6eecf11e4fb71`, `clean_epoch_ema_bf16 c26c7ad00e5b408b54570cb2bd1bcfae` (`C:\Users\Admin\refav1_eval_slice\ckpt{,_ep2}\ckpt.pt`); `postrain30k a58585883c27…`, `k8clip05p30k 2d744d6d2faa…` (`C:\Users\Admin\tanitad-caches\mm-e19-assets-20260901\`) — bulk, not repo |

## 7. Deliverable manifest

⛔ Nothing was committed and nothing was pushed. Every row was `git add`-ed and verified by **blob
comparison** (`git ls-files --stage` vs `git hash-object`) — `git add` exit codes are not evidence —
and re-verified at the end of the turn because the shared index moves under a running agent (it did:
`RESULT.md` was caught stale once and re-staged).

| artifact | location | only one place? |
|---|---|---|
| `SPEC.md` | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-gs9-and-actdiv-units/` | **YES — repo only** |
| `RESULT.md` | same directory | **YES — repo only** |
| `PROPOSED_REGISTER_ROWS.md` | same directory | **YES — repo only** |
| `raw/gs9_vleak.json`, `raw/gs9_vleak.log` | `repo:…/2026-09-03-gs9-and-actdiv-units/raw/` | no — an identical copy remains in the session scratchpad (md5-verified) |
| `raw/actdiv_units_{kappa,steer}.json` | `repo:…/raw/` | no — identical scratchpad copies (md5-verified) |
| `raw/actdiv_units_{kappa,steer}.log` | `repo:…/raw/` | no — scratchpad copies (UTF-16 originals, transcoded to UTF-8 on the way in) |
| `raw/units_tables.txt` | `repo:…/raw/` | no — scratchpad copy (md5-verified) |
| `gs9_vleak.py`, `analyse_units.py`, `run_units.ps1` | `repo:…/2026-09-03-gs9-and-actdiv-units/` | no — scratchpad copies |
| **`taniteval/tools/actdiv_anchored.py`** (`--action-units` pass-through + `apply_action_units`) | **`repo:` staged** — and a run-copy in the mirror `C:\Users\Admin\tanitad-wt\` (md5-identical) | no — but the repo is the source; the mirror is re-synced FROM it and its copy is disposable |
| **`stack/tests/test_actdiv_anchored.py`** (+8 tests, 42 total) | **`repo:` staged**; md5-identical run-copy in the mirror | no — same |
| the two banked GS-9 / actdiv JSONs this task READ | `repo:…/2026-09-03-anchored-actdiv-and-transition-probe/raw/`, `…/2026-09-03-anchored-actdiv-refav1/raw/` | already in the repo; **not** duplicated here |
| checkpoints | dev box only (`C:\Users\Admin\refav1_eval_slice\`, `…\tanitad-caches\…`) and Thor | **not repo (bulk)** — unchanged by this task |

**Nothing lives only in a worktree, a pod, or the scratchpad.** Every scratchpad file above has an
md5-verified twin inside the repo.

## 8. Test status

`stack/tests/test_actdiv_anchored.py`: **42 passed** (34 pre-existing + **8 new** covering the
pass-through: `kappa` returns the candidate list **object-identical**; `steer` maps 0.1 →
0.28225742 rad; the accel channel and the zero anchor are untouched; the conversion goes through
`kinematic.as_command` **to the bit** and inverts with `kappa_of_steer` to 1e−12; an explicit
wheelbase is honoured; an unknown convention raises; `--action-units steer` is **refused on
`--family v7`**; and the per-level gain is non-constant, so 4.872 ≠ 5.000).
Broader selection `-k "kinematic or actdiv or steer or units or transition_probe"`: **145 passed**.

⚠️ **One unrelated collection error, in the MIRROR only:** `stack/tests/test_dinov3_seed.py` there
imports `assert_trunk_anchor_unwired`, which the mirror's `train_v6_staged.py` does not define. **The
REPO is self-consistent** (`assert_trunk_anchor_preflight` in both, two probes) — it is another
agent's in-flight state in a disposable run-copy, not a repo break and not caused by this task.
