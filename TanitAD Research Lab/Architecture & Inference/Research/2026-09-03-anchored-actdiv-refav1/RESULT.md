<title>RESULT — anchored action-divergence on refav1: H-REFAV1-LAT-INSENSITIVE is REFUTED</title>

# RESULT — E-AI-ACTDIV-REFAV1-0903

`TanitAD Research Lab · Architecture & Inference FlyWheel · 2026-09-03`
`Tier: T0-DIAGNOSTIC on every number below — a world-model probe, never driving performance.`
`Device: dev-box CPU throughout. The RTX 4060 was never used; no pod and no Thor contact.`
`Evidence class: MEASURED (ours). Every number below has a raw JSON path in §8.`
`Pre-registration: SPEC.md §4 (this package), verbatim from the 04:10 Berlin sibling SPEC §10 —`
`committed in writing AND in code before any refav1 forward pass existed.`

---

## ⛔ HEADLINE FOR THE COORDINATOR — three things need a decision

1. **H-REFAV1-LAT-INSENSITIVE is REFUTED, on BOTH checkpoints, in all three read spaces.**
   refav1's predictor **does** respond to the lateral action — consistently (F_sep **11.8×** and
   **2,298×** its own measured permutation p95), **perfectly linearly** in κ (norms in the exact
   ratio 1 : 2.50 : 5.00 against levels 0.02 : 0.05 : 0.1), and **perfectly antisymmetrically**
   (cos(m(+L), m(−L)) = −0.998 … −1.000). ⇒ **The flat-κ plan is a COST property, not a world-model
   insensitivity.** The register row that says the world model is the cause must be corrected.
   ⭐ **And the defect at that boundary was named independently 2 minutes after this run finished:**
   `C-STEER-CURVATURE-INTERFACE` (commit `87d9d68`) measures the planner proposing **curvature**
   into a channel the predictor consumes as a **steer angle** — a ×2.9 mis-scaling. That mechanism
   *requires* a responsive predictor to work, and this package is the measurement that it is
   responsive. **The two results are the two halves of one explanation** (§2.5).
2. **RESCUED FROM A SINGLE DISK.** The previous agent's completed GS-8/GS-9 runs — including
   **both refav1 reads this task was asked to produce** — existed ONLY in the worktree at
   `C:\Users\Admin\tanitad-wt\_gs89_pkg\`. 17 files (4 checkpoint reads, the transition probe, the
   dry run, the launchers) are now copied into the repo and staged. Its `RESULT.md` §4 and §5 are
   still marked *pending* — **§5 is answered by this document; §4 (GS-9) is unwritten and its raw
   JSON is banked and unread.**
3. **`taniteval/tools/actdiv_anchored.py` and `transition_probe.py` are NEW and wired into nothing.**
   `mm_e19_read.py`'s actdiv stage still calls the banked `actdiv_local.py`. This is an integration
   request, not a note in a README.

---

## 0. The answer, one line

| question | answer | class |
|---|---|---|
| Is refav1's step-1,000 predictor insensitive to the lateral action κ but not to a (**H-REFAV1-LAT-INSENSITIVE**)? | **NO — REFUTED on both checkpoints.** Both axes separate far above their measured nulls; the κ response is linear and sign-consistent. The κ response is however **40–300× SMALLER than the accel response** (‖m(κ=0.1)‖ / ‖m(a=1.5)‖ = **0.0033** incumbent, **0.0251** clean epoch), which is the quantitative fact the cost-surface probe must now weigh against the planner's explicit `0.05·κ²` penalty. | MEASURED |
| Can the imagination discriminate two recipes whose deployed trajectories are bit-identical? | **YES, and enormously.** The two checkpoints of `D-REFAV1-PAIRED-READ-VOID` — identical to 0.0 m on 140/140 windows — differ by **195×** in κ-axis F_sep and **48×** in latent scene spread. | MEASURED |

---

## 1. What was run

| item | value |
|---|---|
| tool | `taniteval/tools/actdiv_anchored.py --family refav1` (NEW; 34 unit tests) |
| population | the T1 read's own **140 windows**: 20 eval-slice episodes, `k_loader = cfg.op_steps = 30`, stride 10 on `t − (W−1)` |
| loader | **reused, not copied** — `refav1_arm.py:483` `build_loader`, `:477` `episode_names`, `:271` `load_model` (strict, with the config cross-check), imported by file at `actdiv_anchored.py::_load_refav1_arm` |
| corpus | `C:\Users\Admin\refav1_eval_slice\fp8` (20 clips) + `...\eps`; labels/nav `s2_labels_v7.2_eval.jsonl.gz`, md5 `aa12c948f062181c3297265b51526ec5`, 20/20 episodes joined, 0 missing |
| candidates | lateral channel ∈ ±{0.02, 0.05, 0.1} at a = 0; a ∈ ±{0.5, 1.5} m/s² at lateral = 0; **10 + the zero anchor**. ⛔ **the lateral channel is a STEER ANGLE in rad, not a curvature in rad/m** — see §2.5 |
| horizon | h = 1 (one operative step, 0.2 s) |
| spaces | `tac` (the planner's cost space, **verdict space**, 65,536-d), `pooled` (1,024-d), `full_subsample` (8,192 of 655,360 coords; the exact full-field statistics come from running sums over **all** 655,360) |
| null | **measured**, 200 within-window label permutations per axis |
| device | CPU, `OMP_NUM_THREADS=6`; ~26 min per checkpoint |

**Both checkpoints loaded STRICTLY** — `missing_keys: []`, `unexpected_keys: []` — each against its
own config source.

| arm | md5 | config source | parameters | `a_dim → a_in_dim` |
|---|---|---|---|---|
| `incumbent_fp32` (retired fp32) | `45b9f4d82a3a7f15bda6eecf11e4fb71` | `ckpt['cfg']` (no `config.json`) | **175,164,468** | 2 → 2 |
| `clean_epoch_ema_bf16` (EMA + bf16 + TF32) | `c26c7ad00e5b408b54570cb2bd1bcfae` | `config.json` beside it | **182,455,604** | 2 → 2 |

⭐ **The parameter gap is NOT an architecture difference — it is the frozen EMA teacher, and it
takes no part in this read.** Measured by rebuilding both configs:

```
ema_targets=True   total 182,455,604  =  student 175,164,468  +  EMA path 7,291,136
ema_targets=False  total 175,164,468                                    (== the incumbent exactly)
```

`_EmaTargetPath` (`refa_v1.py:1052`, class at `:855`) is a frozen deepcopy of `adapter`,
`tac_queries`, `tac_pool` and `strategic.read`; every parameter is `requires_grad=False`, it is
moved only by `ema_update()` during training, and **`refav1_read`'s forward path never touches it**
(it calls the LIVE `encode` / `_tac_field` / `_run_brains` / `operative.step`). ⇒ **The two arms'
student networks are architecturally identical**, and the whole-config diff between them is
**exactly one field: `ema_targets: False → True`.**

⚠️ Still not strictly one-variable as a *recipe*: the clean epoch also trained under **bf16 +
TF32** where the incumbent was fp32 (`config.json`'s `args`/`precision`/`tf32` blocks; the
incumbent carries no `args` block, so that half is INHERITED from the register's description of it
as the "retired fp32 incumbent", not verified from its own artefact). So: identical architecture,
three recipe knobs (EMA targets, bf16, TF32).

⚠️ **`refav1_arm.load_model`'s `trainable_parameters` reads 0 for every checkpoint** — it calls
`requires_grad_(False)` on every parameter *before* counting. The counts above are totals, printed
by this tool instead. A log that quotes the provenance field claims an empty model.

## 2. ⛔ The speed scale (D-P2-LEAK-AUDIT / H-LEAK-1) — checked, and inert here

Both step-1,000 configs carry **`speed_channel: false`**, verified from the primary artefacts
(`ckpt_ep2\config.json`; the incumbent's `ckpt['cfg']` as loaded). So `RefAV1.augment_actions`
returns the `(a, κ)` controls unchanged, **`a_dim 2 → a_in_dim 2`, and no speed value enters the
predictor at all.** The `v/30`-vs-`v/10` leak that invalidated a banked instrument family this week
**cannot occur on this read.**

The scale is nonetheless printed beside every reading and recorded in the JSON
(`speed_scale_mps: 30.0`, `speed_scale_effective: null`), because the guarantee is structural, not
incidental: **this tool never builds the speed channel.** It calls the model's own
`augment_actions`, which normalises by the model file's own `SPEED_SCALE_MPS`
(`refa_v1.py:90` = 30.0) and **refuses a pre-widened 3-wide action tensor**. Two tests pin it,
including a source-level guard that fails if any hand-rolled `/ 30` ever appears inside
`refav1_read`.

## 2.5 ⛔ UNITS CORRECTION — the "κ" axis is a STEER ANGLE, and this read supplies the missing half of `C-STEER-CURVATURE-INTERFACE`

**Recorded AFTER the run, on a primary-source check, and it changes every label on the lateral
axis in this document — but not one number and not the verdict.**

While this read was executing, another FlyWheel committed **`C-STEER-CURVATURE-INTERFACE`**
(`GOALS_AND_CLAIMS.md`, commit `87d9d68`, 06:35 Berlin). Verified here at the source rather than
from the row: `stack/tanitad/data/physicalai.py:621` is

```python
steer = np.arctan(float(wheelbase) * curv)      # a ROAD-WHEEL ANGLE, in rad
```

and that value — **not** curvature — is what the `v2ep` action channel carries and what
`refav1_loader.py:264` hands to the model **labelled `kappa`**. The planner, meanwhile, proposes,
clamps and integrates in **true curvature** (`refa_v1.py:115-116`, `refa_v1_plan.py:163`).

⇒ **The levels this instrument swept are steer angles in rad, not curvatures in rad/m.** With the
row's (explicitly UNVERIFIED) L ≈ 2.9 m, `κ_true = tan(steer)/L`:

| this document's level | what it actually is | equivalent true curvature | turn radius |
|---|---|---|---|
| 0.02 | steer 0.02 rad (1.15°) | ≈ 0.0069 rad/m | ≈ 145 m |
| 0.05 | steer 0.05 rad (2.86°) | ≈ 0.0173 rad/m | ≈ 58 m |
| 0.10 | steer 0.10 rad (5.73°) | ≈ 0.0346 rad/m | ≈ 29 m |

**Nothing measured here is invalidated, for three reasons.** (1) The verdict is a *separation*
test, and separation is invariant under any monotone reparametrisation of the axis. (2) The σ
normalisation is measured **in the same channel** the candidates were fed (σ = 0.047565 of the
windows' own lateral actions), so "0.1 ≈ 2.1σ" is exact regardless of what the channel denotes.
(3) The **linearity** finding transfers: over this range `tan(δ) ≈ δ` to better than 0.34 %
(tan 0.1 = 0.10033), so linear-in-steer is linear-in-curvature to three significant figures — the
same three figures §4.1 quotes.

⭐ **And the two results close a loop.** `C-STEER-CURVATURE-INTERFACE` argues that a planner
proposal of a real 12.5 m turn is imagined by the predictor as a 36 m one, *"so turning looks
nearly free of consequence, the cost surface goes flat in κ, and the search returns the straight
line."* That mechanism **requires** the predictor to be responsive to its lateral input — a deaf
predictor would make the ×2.9 mis-scaling irrelevant. **This read is the measurement that the
predictor is responsive** (linear, sign-consistent, 2,298× / 11.8× above its own null), and it was
produced independently and pre-registered before either result existed. The two halves are:

| half | who measured it | what it says |
|---|---|---|
| the world model **hears** the lateral action | **this package** | linear, antisymmetric, far above a measured null, on both checkpoints |
| the planner **mis-speaks** it by ×2.9 | `C-STEER-CURVATURE-INTERFACE` | κ proposed, steer consumed; 0.7156 → 0.0600 m lateral error on conversion |

⇒ The verdict's committed consequence — *"the fix is in the cost / the planner boundary, not the
world model"* — now has a **named, measured defect at exactly that boundary.**

⚠️ Two things this does NOT license. The wheelbase 2.9 is **UNVERIFIED** (per-clip fits cluster
near 2.85 and 3.09, none at 2.9), so the "equivalent curvature" column above is indicative, not
quotable. And this package did not re-run its grid in converted units; doing so would rescale the
lateral axis by ~1/2.9 and is the obvious next refinement, together with the h = 10 read.

## 3. Controls — every one reads its known value

| control | known value | `incumbent_fp32` | `clean_epoch_ema_bf16` |
|---|---|---|---|
| **C0 identity** (same inputs twice) | **exactly 0.0** | `0.0` ✓ | `0.0` ✓ |
| **zero model** (real predictor fed the zero action under a non-zero label) | **exactly 0.0** | `0.0` ✓ | `0.0` ✓ |
| **zero anchor** max\|d\| | **exactly 0.0** | `0.0` ✓ | `0.0` ✓ |
| **C1 scene spread** (`tac`) | **> 1e-6** | `4.2492` ✓ | `0.089146` ✓ |
| **shuffled labels** (200 perms) — κ axis | band ≈ 1 | median `0.994`, p95 `2.595` ✓ | median `1.078`, p95 `2.545` ✓ |
| **shuffled labels** — a axis | band ≈ 1 | median `1.005`, p95 `3.696` ✓ | median `1.064`, p95 `3.235` ✓ |
| **shuffled ACTIONS** (realised κ bins) | falls to the floor | `27.90` → **`0.360` / `1.825`** ✓ | `16.21` → **`0.465` / `1.861`** ✓ |

**n = 140 windows** on both arms; `d`: `tac` 65,536, `pooled` 1,024, full field 655,360.
σ of the windows' own first actions = **[a 0.7426, κ 0.047565]** (identical across arms — the same
windows, as required).

The shuffled-action row is the one that matters most: the realised κ separation of **27.9** and
**16.2** collapses to **≤ 1.9** when window *i* is fed another window's action and kept under its
own label. The signal is the action, not the window.

## 4. THE READ (verdict space `tac`, h = 1, n = 140)

| | `incumbent_fp32` | `clean_epoch_ema_bf16` |
|---|---|---|
| **κ axis** F_sep | **5,962.49** | **29.91** |
| κ null p95 (measured) | 2.595 | 2.545 |
| **κ F / null p95** | **2,297.75** | **11.75** |
| κ separates (bar 5×)? | **YES** | **YES** |
| **a axis** F_sep | 3,831.33 | 102.29 |
| a null p95 | 3.696 | 3.235 |
| **a F / null p95** | **1,036.48** | **31.62** |
| a separates? | YES | YES |
| whole-grid F / null p95 | 1,584.94 | 45.33 |
| **VERDICT** | **LAT-INSENSITIVE-REFUTED** | **LAT-INSENSITIVE-REFUTED** |

**All three spaces agree on both arms** — `tac`, `pooled` and `full_subsample` all read
LAT-INSENSITIVE-REFUTED. No SPACE-DEPENDENT stamp is needed.

### 4.1 The κ response is not merely present — it is LINEAR

Exact full-field mean-displacement norms (all 655,360 coordinates, from running sums):

| κ level | `incumbent_fp32` ‖m‖ | ratio to 0.02 | `clean_epoch` ‖m‖ | ratio to 0.02 |
|---|---|---|---|---|
| 0.02 | 0.002959 | 1.000 | 0.013674 | 1.000 |
| 0.05 | 0.007399 | **2.500** | 0.034159 | **2.498** |
| 0.10 | 0.014809 | **5.005** | 0.068094 | **4.980** |

The level ratios are 2.5 and 5.0. **The response is proportional to κ to three significant
figures on both checkpoints.** Spearman ρ(|level|, ‖m‖) = **+1.000** on both, and the sign
consistency is near-perfect: cos(m(+L), m(−L)) = **−0.9998 / −0.9989 / −0.9956** (incumbent) and
**−0.9999 / −0.9995 / −0.9980** (clean epoch).

A predictor that "does not use κ" cannot produce a perfectly linear, perfectly antisymmetric
displacement 2,298× above its own permutation null. **The hypothesis is refuted on its own terms.**

⚠️ The accel axis carries only **two** levels, so its Spearman ρ is reported as `nan` by design
(`spearman()` returns nan for n < 3 — a 2-point rank correlation is always ±1 and carries no
information). Its monotonicity is read directly: ‖m‖ rises 1.2815 → 4.5047 (incumbent) and
1.1455 → 2.7081 (clean epoch) from a = 0.5 to 1.5. Both increase.

### 4.2 What IS small is the κ response *relative to accel* — and that is the real finding

| ‖m(κ = 0.1)‖ / ‖m(a = 1.5)‖ | `incumbent_fp32` | `clean_epoch_ema_bf16` |
|---|---|---|
| `tac` (cost space) | **0.00224** | **0.02619** |
| `pooled` | 0.00329 | 0.02503 |
| full field (exact) | **0.00329** | **0.02514** |

The lateral channel moves the imagination **1/300th** (incumbent) to **1/40th** (clean epoch) as
far as the longitudinal channel does, at grid levels that are each ~2σ of the corpus' own action
spread (σ_κ = 0.0476, so κ = 0.1 ≈ 2.1σ; σ_a = 0.7426, so a = 1.5 ≈ 2.0σ — the two axes are
compared at matched σ, which is why this ratio is meaningful).

**This is the number the cost-surface probe must weigh.** Read from the source
(`stack/tanitad/refs/refa_v1.py:1795` `_cost_chunk`; goal term `:1812-1813`, comfort `:1815`,
**curvature `:1816`**, target speed `:1819`), the cost is exactly:

```
c = (1 - cosine_similarity(tac_field(z_H).flatten, goal.flatten))   # goal term
  + 0.02 * mean(jerk^2)                                            # comfort
  + 0.05 * mean(controls[..., 1]^2)                                # CURVATURE - channel 1 is kappa
  + 0.10 * (v_end - target_speed)^2                                # target speed
```

⭐ **The decisive asymmetry is not only magnitude — the two competing terms have different
homogeneity.** The goal term is a **cosine**, invariant to the overall scale of the field: a
displacement changes it only insofar as it **rotates** `z_H`. The curvature term is an **absolute**
`0.05·κ²`. The trade the planner actually makes is therefore *"radians of rotation bought per unit
κ"* against a fixed quadratic penalty — and at κ = 0.1 that penalty is only `0.05 × 0.01 = 5e-4`,
so it does not need to buy much to win. A lateral displacement 1/300th the size of the longitudinal
one plausibly buys correspondingly little rotation.

⭐ **And the ×2.9 interface defect (§2.5) COMPOUNDS with this, in the same direction.** The
planner's `controls[..., 1]` is a **curvature** it proposes and penalises as `0.05·κ²`, but the
predictor reads that same number as a **steer angle** — so a candidate scored as a 12.5 m turn is
imagined as a ~36 m one. The goal term therefore sees ~1/2.9 of the field rotation the planner
believes it is buying, **while the κ² penalty is charged at the full proposed κ.** The two effects
multiply: a lateral channel that is intrinsically 1/300th as potent as the longitudinal one (this
read), further under-actuated ×2.9 at the boundary (`C-STEER-CURVATURE-INTERFACE`), against a
penalty charged in full.

⚠️ **The cost surface itself is still NOT measured here, and the numbers above do not compute it.**
Rotation per unit steer needs ‖z_H‖ and the goal direction at the planner's own **h = 10**, none of
which this h = 1 read produces. This result establishes only that **the world model is not deaf.**
The cost-surface probe is thereby specified concretely: evaluate `_cost_chunk`'s four terms
**separately** over a lateral sweep on these same 140 windows at h = 10, **in both conventions
(raw, and with `κ → steer = arctan(L·κ)` applied at the model boundary)**, and report the level at
which the curvature term overtakes the goal term in each. If it overtakes at ≈ 0 raw and much later
converted, the flat plan is explained *and* the repair is quantified in one panel.

### 4.3 Magnitude against the programme's material bar

The per-dim RMS of the mean displacement over the per-dim scene std (`rel_units` in the JSON):
**0.00375** (incumbent, `tac`) and **0.01251** (clean epoch, `tac`) — i.e. **0.4 %** and **1.3 %**
of the scene spread, against the programme's committed material bar of **5.95 %**. So on the
STRUCTURE/MAGNITUDE split the picture matches the v7 arms exactly: **structured, sign-consistent,
monotone — and not material in magnitude.** Maximum absolute displacement over the whole grid:
0.2317 (incumbent) and 0.0413 (clean epoch).

### 4.4 ⭐ The imagination discriminates the two recipes; the trajectory could not

`D-REFAV1-PAIRED-READ-VOID` measured these same two checkpoints producing **bit-identical
trajectories on 140/140 windows** (max |Δ| exactly 0.0 m). On the **same 140 windows**, their
imaginations are not close:

| | incumbent | clean epoch | ratio |
|---|---|---|---|
| κ-axis F_sep (`tac`) | 5,962.49 | 29.91 | **199×** |
| a-axis F_sep (`tac`) | 3,831.33 | 102.29 | **37×** |
| scene spread (`tac`) | 4.2492 | 0.089146 | **48×** |
| max abs displacement | 0.23172 | 0.041338 | **5.6×** |
| κ/a asymmetry (full field) | 0.00329 | 0.02514 | **0.13×** |

F_sep is **scale-invariant** (asserted, not assumed — the output-scale control multiplies every
prediction by c and requires F, ρ and cos to be unchanged), so the 199× is a structural difference
and not a units artefact. ⇒ **The claim in `D-REFAV1-PAIRED-READ-VOID` (ii) — that the
discriminating instrument at low steps is the imagination, not the deployed trajectory — is now
MEASURED, not argued.**

Two readings sit inside that, and this package does **not** adjudicate between them:
(a) the clean epoch's latent field is 48× more compressed and its action response 199× less
consistent — a *worse* imagination; or (b) the incumbent's enormous F reflects a near-deterministic,
low-window-variance response (F is between/within) that may itself be a degenerate regime. The
open packages `2026-09-02-refav1-target-space-collapse` and `2026-09-02-refav1-ema-inflation` are
where that belongs, and the scene-spread collapse measured here is direct evidence for them.

## 5. Verdict against the pre-registered table

> **LAT-INSENSITIVE-REFUTED** — `F_sep(κ-axis) ≥ 5 × its null p95` — the predictor DOES respond to κ
> consistently; the flat-κ plan is then a COST property, not a WM insensitivity.

The bar is `F_sep / null_p95 ≥ 5`. κ reads **2,297.75** (incumbent) and **11.75** (clean epoch)
against that 5 — i.e. **460×** and **2.4×** clear of the threshold. The verdict is **REFUTED on
both checkpoints, in all three spaces, with every control at its known value.**

⚠️ **The clean epoch is the closer call and is reported as such:** 11.75 against a bar of 5 is a
factor 2.4, not a factor 460. It is still an unambiguous pass — the same arm's realised κ
separation (16.21) collapses to ≤ 1.86 under the shuffled-action control, and its κ norms are
linear in κ to three significant figures with cos(+L, −L) ≤ −0.998, none of which a noise response
produces. But a reader who wants the margin should take 2.4×, not 460×.

**Committed consequence, now due:** *"the fix is in the cost (the κ² penalty / the goal cosine's
blindness to lateral change), and the register row must say the world model is not the cause. The
next probe is the cost surface itself."*

## 6. Instrument validation (H-GS8-1 half)

34 tests pass on CPU with no checkpoint and no corpus (`stack/tests/test_actdiv_anchored.py`).
The seven added by this session read the instrument end-to-end on a **random-init tiny `RefAV1`**
(85k params, fitted standardizer) through the real predictor path:

| test | known value it reads |
|---|---|
| action-blind model (`operative.act[0]` zeroed) | every displacement **exactly 0.0**; `ss_between = ss_within = 0` |
| planted **κ-only** response | κ separates, accel dead → **LAT-INSENSITIVE-REFUTED** |
| planted **accel-only** response | accel separates, κ dead → **LAT-INSENSITIVE-CONFIRMED** |
| C0 / zero-model / C1 end-to-end | 0.0, 0.0, > 1e-6 in all three spaces |
| speed-scale contract | model's own 30.0; integration formula; pre-widened tensor REFUSED; `speed_channel=False → 2` channels |
| source guard | no hand-rolled `/ 30` inside `refav1_read` |
| full-field exact statistics | match the direct computation |

The accel-only test is the one that earns the verdict its credibility: **without it the instrument
could only ever refute H-REFAV1-LAT-INSENSITIVE, never confirm it.** It confirms on demand.

⚠️ **A known instrument property, now pinned:** an *exactly* dead axis reads **`nan`, not 0** — its
F is 0/0. `verdict_refav1` guards with `np.isfinite(f) and f >= bar`, so nan is correctly *not
separated*. Two of the new tests initially failed by expecting 0; the instrument was right and the
tests were wrong.

## 7. Limits — what this read does NOT establish

1. **h = 1 only** (0.2 s, one operative step). The planner rolls **h = 10** at 2.0 s. A response
   that is linear at one step can still vanish, saturate or rotate over ten. The h = 10 read is a
   GPU follow-up and is **not** done.
2. **The cost surface is not measured.** §4.2's mechanism is a hypothesis with a quantitative
   motive, not a result.
3. **Step 1,000 only.** Both checkpoints are at step 1,000 of a run whose epoch is 21,109 steps.
   Nothing here forecasts the trained model.
4. **Not one-variable as a recipe** (§1) — though the architectures ARE identical: EMA targets,
   bf16 and TF32 all changed together, so a per-knob attribution is not available here.
5. **The lateral axis is swept in STEER-ANGLE units, not curvature (§2.5)** — every "κ" label on
   that axis in this document means the model's lateral input channel. The verdict, the σ
   normalisation and the linearity all survive; the *physical* curvature equivalents depend on a
   wheelbase the register marks UNVERIFIED, so they are indicative only. The grid was **not**
   re-run in converted units.
6. **`C-REFAV1-KIN-CONTRACT-LAT` is RESOLVED — by someone else, during this run.** It is the same
   unit defect: `κ = tan(steer)/2.9` moves the curved-window lateral error 0.7156 → 0.0600 m
   against a 0.0527 m floor and the yaw RMSE 19.81° → 0.63°
   (`C-STEER-CURVATURE-INTERFACE`, commit `87d9d68`). This package did not contribute to that and
   quotes it as INHERITED.
7. **`ep2` is retired** (stopped at step 6,850, kept as H-EPOCH-2 evidence) and the epoch was
   restarted as `ep3` **with `speed_channel: true`** on PI directive at ~06:30 Berlin. The read
   above is therefore a reading of two **superseded** checkpoints. It remains valid as banked
   evidence about them — and the D-REFAV1-EP3-SPEED row's independent proof that both banked
   checkpoints have `operative.act.0.weight` of shape `[1024, 2]` **corroborates §2's finding**
   that no speed entered this read.
8. **The four metric families are REFUSED on the record**, in the JSON, with the reason: a T0
   latent-space diagnostic has no trajectory to score.

## 8. Raw artifacts

| what | path |
|---|---|
| **first read, incumbent fp32** (primary) | `…/2026-09-03-anchored-actdiv-and-transition-probe/raw/actdiv_anchored_refav1_step1000_fp32.json` |
| **first read, clean epoch** (primary) | `…/2026-09-03-anchored-actdiv-and-transition-probe/raw/actdiv_anchored_refav1_step1000_ep2.json` |
| **independent replication, both arms** (this session) | `…/2026-09-03-anchored-actdiv-refav1/raw/actdiv_anchored_refav1_step1000.json` |
| replication console log | `…/2026-09-03-anchored-actdiv-refav1/raw/actdiv_anchored.log` |
| tool | `taniteval/tools/actdiv_anchored.py` |
| tests | `stack/tests/test_actdiv_anchored.py` (34 pass) |
| pre-registration | `…/2026-09-03-anchored-actdiv-refav1/SPEC.md` §4; original `…-and-transition-probe/SPEC.md` §10 |
| **GS-9 transition probe, run but UNREAD** | `…/2026-09-03-anchored-actdiv-and-transition-probe/raw/transition_probe_v7.json` |

## 9. Replication — the read reproduces BIT-FOR-BIT across independent launches

A second, independently launched pass over both checkpoints (05:56 → 06:33 Berlin, separate
process, both arms in one invocation instead of two, plus the speed-scale provenance fields the
first pass lacked) reproduces the first read **exactly** — not "within tolerance", but to every
digit printed:

| quantity (`tac`, h = 1) | first read | replication | Δ |
|---|---|---|---|
| incumbent κ F/p95 | 2,297.751 | 2,297.751 | **0** |
| incumbent a F/p95 | 1,036.481 | 1,036.481 | **0** |
| incumbent full-field F_sep | 2,789.998 | 2,789.998 | **0** |
| incumbent scene spread | 4.249215 | 4.249215 | **0** |
| clean epoch κ F/p95 | 11.755 | 11.755 | **0** |
| clean epoch a F/p95 | 31.617 | 31.617 | **0** |
| clean epoch full-field F_sep | 82.130 | 82.130 | **0** |
| clean epoch scene spread | 0.089146 | 0.089146 | **0** |
| every κ/a norm, every cos(+L,−L), every ρ | — | — | **0** |
| C0 identity / zero model, both arms | 0.0 / 0.0 | 0.0 / 0.0 | **0** |

Same 140 windows, same σ = [a 0.7426, κ 0.047565], same seed 0, same 200 permutations, same
strict load (`missing_keys: []`, `unexpected_keys: []`). Runtime 1,182.5 s + 1,151.8 s on CPU.

⇒ **The instrument is deterministic and the verdict is not a seed artefact.** Combined with the
C0 identity control (the same inputs forwarded twice differ by *exactly* 0.0), the read carries no
run-to-run noise at all — which is what makes the 199× incumbent-vs-clean-epoch gap in §4.4 a
statement about the checkpoints rather than about the harness.

⚠️ Both passes ran the SAME code on the SAME box, so this establishes **determinism and
reproducibility**, not independent implementation. It is not a cross-implementation check and is
not claimed as one.
