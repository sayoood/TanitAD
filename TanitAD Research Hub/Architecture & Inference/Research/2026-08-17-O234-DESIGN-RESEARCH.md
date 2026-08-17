# O2 / O3 / O4 — DESIGN AND WEIGHTING RESEARCH

**Date:** 2026-08-17 · **Branch:** `agent/arch-inf-20260803` · **Agent:** o234-design-research
**PI decision this serves (2026-08-17, verbatim):** *"We will stick to the unsupervised path of the
wm, we need a research about design and weighting of O2/O3/O4."*
**Eval tier:** ⛔ nothing here is a driving number. Every measurement re-quoted is **T0-DIAGNOSTIC**
or a **source-level derivation**, and is stamped as such.

**Evidence classes used throughout:** `DERIVED (algebra on our source, file:line)` ·
`MEASURED (ours + artifact path)` · `PUBLISHED (cited)` · `INHERITED (not re-verified)` ·
`UNVERIFIED`. ⛔ A literature claim never appears without a fetched URL.

---

## ⭐ THE ANSWER IN ONE PAGE

**The headline is a DERIVATION, not an opinion, and it does not need the literature to stand.**

1. ⛔ **O2 is not an independent objective. It is O5's own residual, re-weighted across cells.**
   `V6Stack.cells()` (`stack/tanitad/models/v6.py:3710`) is a **pure `reshape`** — no projection, no
   learned map. O2 and O5 are therefore computed from the **literally identical tensor**
   `zhat_steps[j] − z_true[j]`. The algebra closes exactly:

   > **O2 = (O5's step-`j` term) + Cov_c(w, err)**

   where `w` is O2's time-to-reach weight profile, **mean-1 normalised over cells by construction**
   (`time_to_reach_weights(..., normalize=True)`, `v6.py:343-344`). ⇒ **O2's entire distinct
   content is the covariance between a fixed weight profile and the per-cell error.** The measured
   **`cos(g_O2, g_O5) = +0.870`** is not a finding to be explained — it is **forced by the code**.

2. ⛔ **That distinct content is a 4-value tilt across image ROWS, and it barely touches the cells
   the lead vehicle is in.** The readout grid is **4×4**, so `w` has **four distinct values,
   identical across all four columns** (`readout_grid_ranges` gives every column in a row the same
   range, `v6.py:383`, and its own docstring stamps the table **ESTIMATED — a declared monotone
   image-row prior, NOT calibrated depth**). At 15 m/s the profile is
   **[0.131, 0.771, 1.396, 1.703]** for rows at **[80.0, 26.8, 9.0, 3.0] m**. The MEASURED median
   GT lead sits at **`cx` 15.05 m** — i.e. **between rows 1 and 2, in the band weighted 0.77–1.40,
   essentially unweighted.** The row O2 actually promotes (3.0 m, weight 1.70) is road surface
   directly under the ego's nose. ⇒ **O2 spends its one degree of freedom somewhere other than
   where the agents are.**

3. ⛔ **O4 contains no perception signal of any kind and cannot.** `kinematic_saliency`
   (`v6.py:442`) is `|jerk| + |decel| + steering-reversal-rate` **from the recorded ego actions
   alone** — the code says so and is right to. It has **no gradient** (it is a sampler,
   `build_o4_weights` → `InteractionSampler`, `train_v6_staged.py:746`/`:2495`). ⇒ **it cannot
   distinguish braking for a lead vehicle from braking for a stop line, a curve, or a speed bump.**
   It is an **ego-kinematic re-sampler wearing the name `InteractionSampler`**, and
   `DIAGRAM_CONFORMANCE.md:59` already says so: *"O4 is the ego-kinematic version only"*, with the
   multi-agent T3 curriculum ⬜ **NOT BUILT**.

4. ⭐⭐ **THE ARCHITECTURAL FINDING THAT SUBSUMES ALL THREE — every latent objective sits
   downstream of a 40:1 AVERAGE POOL.** `SpatialGridReadout` (`stack/tanitad/models/readout.py:88`)
   is `AvgPool2d((16//4, 40//4)) = AvgPool2d((4, 10))`: each of the 16 readout cells is the
   **arithmetic mean of 40 ViT patch tokens** (64 × 160 px of a 256 × 640 field), followed by one
   shared `Linear`. **O1, O2, O3 and O5 all act on the post-pool cells; no term in the objective
   ever sees the pre-pool token grid.** The MEASURED median GT lead is **37.8 px wide ≈ 2.4
   patches** — so the lead vehicle enters the loss as a **single-digit percentage of one cell's
   average**, summed with ~36 background tokens **before any gradient is computed**.

   ⇒ ⭐ **We do have an architectural bottleneck — but it is a POOLING bottleneck, and pooling is
   the one operation that provably destroys individuation.** It is the opposite of the
   competition-for-explanation bottleneck the object-centric line uses. `cells()`'s own docstring
   quotes *"pooling is where geometry goes to die"* and then exposes **the pooled tensor** — the
   4×4 grid is the *output* of the pooling, not a repair for it.

5. ⭐ **AND THE LITERATURE SAYS THE SAME THING, INCLUDING THE FIX.** Meta's own V-JEPA 2.1
   ([arXiv:2603.14482](https://arxiv.org/abs/2603.14482)) diagnoses V-JEPA 2's *"fragmented local
   spatial structure"* as caused by supervising **only masked tokens**, which lets context patches
   become *"global aggregators"* — **exactly what a 40:1 average pool forces our cells to be.**
   Their fix is a **dense loss over ALL tokens**: ADE20K **→ 47.9 mIoU**, NYUv2 **0.642 → 0.307**,
   ⭐ **DAVIS J&F 52.5 → 69.0**, **and the global scores rose too** (IN1K 82.2 → 85.5). ⚠️ preprint.

⇒ **VERDICT ON THE PI'S QUESTION.** O2/O3/O4 are **right in ambition, wrong in KIND for the job of
individuating agents, and not fixable by re-weighting.** Two of the three are structurally weaker
than their nominal `1.0` suggests (O2 is ~1 effective degree of freedom bolted onto O5; O4 is a
data-order change with no gradient), and all of them are asked to create agent structure **behind an
average pool that has already averaged the agent away.**
⛔ **"They are basically right and merely need patience" was an outcome the brief explicitly invited,
and the evidence REFUTES it three times over** — Didolkar (object-centric FG-ARI plateaus at ~8 k
samples, *"no evidence of favorable data scaling laws"*), DINOv3 (dense locality **degrades** with
longer training), and [arXiv:2606.07687](https://arxiv.org/abs/2606.07687) verbatim: *"capacity
scaling cannot recover the missing structure when the pretraining objective itself fails to encode
it."*
⚠️ **The single highest-value change is not a loss weight. It is to give at least one objective
access to the pre-pool token grid** — and `encode_window(..., return_tokens=True)` already exists.

⛔⛔ **AND THE FINDING I LIKE LEAST, STATED UP FRONT: the one published case of object structure
emerging from our objective class FAILED on a narrow corpus** (SSv2 alone ⇒ *"almost chance-level"*;
it needs HowTo100M-scale **diversity**, not volume). **Our 2 376 driving episodes are narrow in that
same way.** ⚠️ HYPOTHESIS-strength for us, but it is the only finding here that could invalidate the
whole objective class, and **no weight, term or architecture change addresses it** (§5-S6, §8-3).

⚠️ **What is NOT claimed:** that O2/O3 are useless, that the run should be stopped, or that any
weight should change on the live 30 k. §6.3 gives five discriminating experiments, **two of them
literally zero-GPU**, with both outcomes pre-committed.

---

## 0. ⛔ THE STAMPS

1. **Nothing was executed on GPU.** Thor's live v6F 30 k run was not touched; the dev box's ladder
   was not touched. The only computation run for this document is a **~20-line pure-Python
   arithmetic reproduction of `time_to_reach_weights`** (§2.2), which imports nothing.
2. **Source is quoted at `file:line` against the working tree at the time of writing.**
3. **Re-quoted measurements carry their original artifact path** and are marked `INHERITED` where I
   did not re-run them.
4. ⚠️ **The `2026-08-17-latent-linear-ladder` stream had NOT landed a report** when this was
   written — `…/incoming/2026-08-17-latent-linear-ladder/` contained `code/` and `raw/` but **no
   `.md`**. Its results are therefore **absent from this document** and its arrival should be
   checked against §6.
5. ⛔ **`MODEL_REGISTRY.md` was not touched** (held by another agent).

---

## 1. WHAT O2 / O3 / O4 ACTUALLY ARE — established from source, not inherited

| lever | symbol | `file:line` | kind | weight | gradient? |
|---|---|---|---|---|---|
| **O1** | response-form / factual / scene | `train_v6_staged.py:1240` | loss (3 sub-terms) | `w_o1_*` | ✅ |
| **O2** | `o2_near_field_loss` | `train_v6_staged.py:609`, called `:1287` | **loss** | `--w-o2` **1.0** | ✅ |
| **O3** | `o3_masked_cell_loss` | `train_v6_staged.py:646`, called `:1297` | **loss** | `--w-o3` **1.0** | ✅ |
| **O4** | `build_o4_weights` | `train_v6_staged.py:746`, consumed `:2495` | ⛔ **SAMPLER** | `--o4-alpha` **1.0** | ⛔ **NONE** |
| **O5** | `o5_rollout_consistency_loss` | `train_v6_staged.py:700`, called `:1279` | loss | `--w-o5` **1.0** | ✅ |
| **O6** | `o6_sigreg_loss` (LeJEPA/SIGReg) | `train_v6_staged.py:730`, called `:1317` | loss | `--w-o6` **0.1** | ✅ |

⛔ **"O4's weight" is not a loss coefficient and `cos(g_O4, ·)` does not exist.** `--o4-alpha` is the
exponent in `saliency_weights`, `w = (floor + s)**alpha` (`v6.py:503`), turning a per-window scalar
into a sampling distribution. `alpha = 0` reproduces uniform sampling exactly — the attributability
control. This correction was already made by the probe-positive-control run
(`PROBE_POSITIVE_CONTROL.md` TASK 3) and is re-established here from source.

### 1.1 O2 — what it computes

```
w   = time_to_reach_weights(cell_ranges_m, v_ego, tau_s=2.0)   # [B, C], MEAN-1 over C
err = |cells(zhat_j) − cells(z_true_j)|.mean(dim=-1)           # [B, C], mean over d_readout=128
O2  = (w * err).mean()
```

* the target `z_true` is **detached** (`train_v6_staged.py:2594`), so this is a stop-grad
  regression, correctly;
* it fires at **exactly one rollout step** `j = min(o1_k, k_roll) − 1` (`:1288`), not across the
  rollout;
* `cell_ranges_m` is `readout_grid_ranges(4, 4, near_m=3.0, far_m=80.0)`, a **geometric row ladder**
  whose own docstring stamps it **ESTIMATED … NOT calibrated depth** and states
  *"Columns share a row's range"* (`v6.py:359-368`).

### 1.2 O3 — what it computes

`MaskedCellPredictor` (`v6.py:3245`) replaces masked cells with a learned mask token **after**
input projection (`h = torch.where(mask, mask_token, h)`, `:3284`) and adds a per-cell positional
embedding, then runs a **2-layer, 256-wide, 4-head transformer encoder** over the 16 cells. Context
is `cells(zhat_j)` under `--o3-mode action` (the default in the chain), target is `cells(z_true_j)`,
and **only masked cells are scored** (`:671`). ✅ **The masking is genuine — there is no copy path.**
Default mask: `--o3-blocks 2` contiguous blocks, `--o3-band-rows 0`. MEASURED mask rate in the dry
ladder: **0.3125** (`…/2026-08-16-v6-stage-chain/raw/dry_ladder_default.json`).

⚠️ **O3 is the ONE term in the objective with an independent gradient direction and its own
parameters** — and `masked_cells` receives gradient from O3 and nothing else
(`O6_ABLATION_AND_MASK_PROBE.md` §2.5: turn O3 off and it is a frozen random head).

### 1.3 O4 — what it computes

```
s = w_jerk·mean|Δaccel|/dt /5.0  +  w_decel·mean(relu(−accel))/2.0  +  frac(steering sign changes)
w = (0.25 + s)**alpha,  normalised to sum 1     # per-window sampling weight
```
(`v6.py:442-506`). **Steer and accel are the only inputs.** The docstring is explicit and correct:
*"⛔ NO PERCEPTION LABEL ENTERS."* — which is what makes it admissible under the JEPA thesis, and
also what caps what it can possibly teach.

---

## 2. ⭐ THE DERIVATIONS — what the code forces, before any experiment

### 2.1 `DERIVED` — O2 is O5's residual plus a covariance term

`cells()` is a pure reshape (`v6.py:3717-3718`), so with `err[b,c]` the mean absolute residual of
cell `c` and `w[b,·]` mean-1 over the `C` cells:

| quantity | expression |
|---|---|
| O5's step-`j` term | `per_j = E_b[ (1/C) Σ_c err_bc ]` |
| O2 | `E_b[ (1/C) Σ_c w_bc · err_bc ]` |
| **⇒ O2 − per_j** | `E_b[ Cov_c(w_b, err_b) ]` **(exact, since mean_c(w−1) = 0)** |

and `O5 = (1/k) Σ_j per_j` for `--o5-mode uniform` (weights normalised to mean 1, `v6.py:697`).

⇒ **the shared direction between the two gradients is `∇per_j`, which enters O2 with coefficient 1
and O5 with coefficient 1/k.** ⭐ **This is a sufficient mechanistic explanation of the measured
`cos = +0.870` and removes any need to hypothesise about it.**

### 2.1a ⭐⭐ `MEASURED (ours)` — **the trainer already logs BOTH halves of that decomposition**

`o2_near_field_loss` returns `o2_loss = (w*err).mean()` **and** `o2_unweighted = err.mean()`
(`train_v6_staged.py:640-643`). ⭐ **`o2_unweighted` IS O5's step-`j` term, exactly** — both are the
mean absolute value of every element of `zhat_j − z_true_j`. ⇒ **O2's unique content is the
difference of two numbers the trainer prints on every log row, and it costs ZERO GPU to read.**

MEASURED over all 7 banked rows carrying both fields
(`…/2026-08-16-v6-stage-chain/raw/dry_ladder_default.json`, `dry_ladder_arms.json`,
`dry_ladder_default.log`; extraction reproduced at `raw/2026-08-17-O234/o2_cov_from_logs.py`):

| stage | step | `o2_loss` | `o2_unweighted` (= O5 step-`j`) | `Cov` = diff | \|Cov\|/unwt |
|---|---|---|---|---|---|
| S-W | 1 | 3.6532 | 3.7398 | −0.0865 | 2.31 % |
| S-W | 2 | 3.8276 | 3.8609 | −0.0333 | 0.86 % |
| S-J | 1 | 4.0415 | 4.1808 | −0.1393 | 3.33 % |
| S-J | 2 | 4.0608 | 3.9884 | **+0.0724** | 1.81 % |
| S-W | 1 | 4.3546 | 4.3743 | −0.0197 | 0.45 % |
| S-W | 2 | 4.7429 | 4.6010 | **+0.1418** | 3.08 % |
| S-J | 2 | 4.0606 | 3.9884 | **+0.0722** | 1.81 % |

⇒ ⭐ **n = 7 · \|Cov\|/unweighted = 0.45 %–3.33 %, median 1.81 % · SIGN FLIPS (4 −, 3 +).**
**~98 % of O2's value is literally O5's step-`j` term, and the remaining ~2 % does not even have a
stable sign.**

⚠️⚠️ **THE CAVEAT THAT MUST TRAVEL WITH THIS NUMBER, AND IT IS NOT SMALL.** These are **dry-ladder
rows at steps 1–2, at or near initialisation**, where the per-cell error profile is close to uniform
and `Cov_c(w, err) ≈ 0` **almost by construction**. At convergence the error profile could become
strongly non-uniform and `Cov` could grow by an order of magnitude. ⛔ **This measurement therefore
establishes that the decomposition is exact and cheap to read — NOT that O2 is a 2 % term for all
time.** The live 30 k run prints both fields on every row, so settling it at convergence is a
**one-line read of an existing log**. That is experiment **E-O2-A** (§6.3), and it is the single
cheapest decision-relevant measurement this document produces.

### 2.2 `MEASURED (ours — pure-python reproduction of `time_to_reach_weights`)` — the profile

Reproducing `readout_grid_ranges(4,4, near_m=3, far_m=80)` and
`time_to_reach_weights(·, tau_s=2.0, horizon_s=6.0, v_floor=1.0)` in ~20 lines of arithmetic
(banked at `raw/2026-08-17-O234/o2_weight_profile.py` / `.json`):

**Row ranges (row 0 = TOP = far → row 3 = BOTTOM = near): `[80.00, 26.78, 8.96, 3.00] m`**

| `v_ego` | per-row weight (all 4 columns identical) | max/min | half-weight distance |
|---|---|---|---|
| 5 m/s | `[0.157, 0.217, 1.288, 2.338]` | 14.9× | 6.9 m |
| 10 m/s | `[0.110, 0.579, 1.411, 1.901]` | 17.3× | 13.9 m |
| **15 m/s** | **`[0.131, 0.771, 1.396, 1.703]`** | 13.0× | 20.8 m |
| 20 m/s | `[0.228, 0.863, 1.346, 1.563]` | 6.9× | 27.7 m |
| 30 m/s | `[0.388, 0.943, 1.268, 1.401]` | **3.6×** | 41.6 m |

✅ **Cross-check against the live trainer's own log:** the dry ladder recorded
`o2_w_min 0.1455 / o2_w_max 2.4681` and `0.1390 / 1.6884`
(`…/2026-08-16-v6-stage-chain/raw/dry_ladder_default.json`), inside the `[0.11, 2.34]` envelope
computed above. ⇒ the reproduction is faithful.

**Three consequences, all `DERIVED`:**

1. **O2 has ~1 effective degree of freedom, not 16.** Four values, monotone, column-constant.
2. **At motorway speed it nearly vanishes** — `max/min = 3.6×` at 30 m/s, so `Cov_c(w, err) → 0` and
   **O2 → O5's step-`j` term exactly.**
3. ⛔ **It does not target the lead vehicle.** MEASURED lead `cx`: mean 15.53 m, median 15.05 m, p10
   8.15 m, p90 24.33 m (`PROBE_POSITIVE_CONTROL.md` §3.1, n = 2 721 windows). Against the row
   ladder `[80.0, 26.8, 9.0, 3.0]`, **the entire p10–p90 lead range lives in rows 1–2**, weighted
   **0.77–1.40 at 15 m/s** — a ±30 % reallocation. The 1.70× row is the 3.0 m row.

### 2.3 ⭐⭐ `DERIVED` — the 40:1 average pool, and why it is the real bottleneck

| fact | value | source |
|---|---|---|
| encoder field | 256 × 640 px, patch 16 | `PROBE_POSITIVE_CONTROL.md` §3 (cache meta) |
| token grid | **16 × 40 = 640 tokens** | ″ |
| readout | `AvgPool2d((4, 10))` → 4 × 4 = 16 cells, then `Linear(d_model, 128)` | `readout.py:88, 111` |
| **tokens averaged per cell** | **40** | `DERIVED` |
| image area per cell | **64 × 160 px** | `DERIVED` |
| **median GT lead apparent width** | **37.8 px ≈ 2.4 patches** | `PROBE_POSITIVE_CONTROL.md` §3.1 (MEASURED, n = 2 721) |
| lead share of a cell's columns | **≈ 2.4 / 10 = 24 %** | `DERIVED` |
| lead share of a cell's 40 tokens | **≈ 5–12 %** | ⚠️ **ESTIMATED** — needs vehicle height, which the join does not carry (`PROBE_POSITIVE_CONTROL.md` §3: *"HEIGHT IS ABSENT FROM THE JOIN"*); at a nominal 1.5 m height and 15 m range, `f_ref·1.5/15 ≈ 30.6 px ≈ 1.9 patches` ⇒ ~4.6 of 40 tokens |

⇒ ⭐ **The nearest, largest, easiest agent in the corpus contributes on the order of one part in ten
to the average that every loss term sees.** And `AvgPool2d` is a **fixed linear operator with no
competition, no assignment and no capacity to represent "which"** — it cannot separate a vehicle
from the road behind it, because they are summed into the same scalar before the loss exists.

⚠️ **This is a statement about where the objectives ACT, not a claim that the encoder discards
agents.** The encoder's 640 tokens may well carry the agent; the objectives simply never look at
them. §5 is where the literature decides whether that distinction matters.

---

## 3. ⭐ WHAT OUR OWN MEASUREMENTS ALREADY SAY ABOUT WEIGHTING

### 3.1 `INHERITED — MEASURED` gradient geometry (5 seeds, live grid)

Pairwise `cos(g_i, g_j)` on the trunk, mean over 5 seeds; chance = `1/√257995` = **0.00197**
(`…/2026-08-16-o6-ablation/O6_ABLATION_AND_MASK_PROBE.md` §2.4):

| | o1 | o2 | o3 | o5 | o6 |
|---|---:|---:|---:|---:|---:|
| **o1** | — | **+0.321** ✅ | +0.127 ❌ | **+0.322** ✅ | −0.011 ❌ |
| **o2** | | — | +0.028 ❌ | ⭐ **+0.870** ✅ | +0.005 ❌ |
| **o3** | | | — | +0.031 ❌ | +0.042 ❌ |
| **o5** | | | | — | −0.019 ❌ |

✅ = sign-consistent across all 5 seeds; ❌ = sign flips ⇒ not separated from orthogonal.

⭐ **Read as a rank statement, this is the weighting finding.** The objective advertises **five**
gradient-carrying terms. It has, at most, **three independent directions**:

| effective direction | terms | evidence |
|---|---|---|
| **1. "fit the rolled latent to the encoded future"** | **O1 + O2 + O5** | O2↔O5 +0.870; O1↔O2 +0.321; O1↔O5 +0.322 — **all three sign-stable** |
| **2. masked spatial inpainting** | **O3** | +0.028/+0.031/+0.042, **all sign-unstable** ⇒ independent |
| **3. anti-collapse** | **O6** | \|cos\| ≤ 0.042, all sign-unstable ⇒ orthogonal to everything |

⇒ ⛔ **Three of the five nominal `1.0` weights are buying one direction.** Raising or lowering
`--w-o2` moves the model along **almost exactly the axis `--w-o5` already moves it along**; the only
part of `--w-o2` that is not `--w-o5` is the ±30 % row tilt of §2.2.

### 3.2 `INHERITED — MEASURED` relative pull

`O6_ABLATION_AND_MASK_PROBE.md` §2.4-2.5: O6's trunk pull is **5.3 %** of the rest of the
objective; O3's is **7.7 %**. Per-module: O6 exerts **exactly 0.000** on `predictor_op` (it is
applied to the *encoded* window, which never passes through the predictor) ⇒ **SIGReg shapes
encoder + readout only and cannot prevent a collapse living in the predictor.**

### 3.3 ⚠️ `INHERITED — MEASURED` — the collapse O6 defends against **does not develop**

`O6_ABLATION_AND_MASK_PROBE.md` §3, `raw/collapse_trajectory.json`: with **`w_o6 = 0`, SIGReg fully
OFF, maximum collapse pressure**, pooled effective rank went **446.49 → 453.10 (+1.5 %) over 1 200
steps at the live lr**, while the loss fell 8.6× at 10× lr with the rank moving −3.1 %.

⚠️ **This is a fact about the weighting question and it cuts BOTH ways.** It does not argue for
removing O6 — the PI has ruled SIGReg stays, the term costs 5.3 % of the pull, and *"no collapse
under a term that prevents collapse"* is not evidence the term is idle. It **does** mean **O6 is not
where the missing agent structure is**, and that raising `--w-o6` is not a candidate lever.

### 3.4 ⭐ `INHERITED — MEASURED` — what the latent demonstrably does and does not carry

⛔ **D1 is WITHDRAWN** (`…/2026-08-17-probe-positive-control/PROBE_POSITIVE_CONTROL.md`). Nothing
below may be built on *"the latent does not carry agents"*.

**What stands, stated at full precision — and it is weaker than the shorthand:**

| instrument | v6F@11250 | v6F@9000 | random-latent null | C-CONST |
|---|---|---|---|---|
| **ridge (linear) `lead_gap_abs_err_m`** | **6.713** [6.010, 7.482] | **6.944** [6.194, 7.816] | **8.534** [8.148, 8.936] | **5.133** |
| K1 vs constant | +1.580 **fail** | +1.811 **fail** | +3.401 **fail** | — |
| corr(pred, GT) | **+0.159** | **+0.108** | **−0.018** | — |

⭐ **The real reading:** the v6 latent beats a random-vector null by **1.6–1.8 m with positive
correlation**, ⛔ **and still loses to a constant predictor.** `r = +0.159` is **weak signal, not
latent agent geometry**. ⚠️ Single fit each, no seed replicate.

And at the **repaired** slot operating point (`n_queries 16`, the first configuration that passes on
the answer and fails on noise), the real arm scored **8.331 m, K1 +3.217 [+2.310, +4.246] FAILS,
with BOTH anti-echo controls UNSEPARATED — like noise, unlike the oracle** (which separates by 3.1
and 6.1 m). ⚠️ One seed, early-read at 37.5 %.

⇒ **The honest state: the information is present, weak, and linearly accessible in trace amounts.
That is exactly the signature of a quantity that survived an average pool as a residue.** §2.3 is
the mechanism that predicts precisely this.

---

## 4. THE LITERATURE

⚠️ **PROVENANCE, STATED ONCE.** Every source below was **fetched and read** during this run by one
of three parallel literature streams; each carries its arXiv id and URL. ⛔ **Nothing here is
written from memory.** I **independently re-fetched** the single most load-bearing citation
(V-JEPA 2.1, §4.2) because the whole of recommendation **R1** turns on it. The rest I did not
personally re-fetch — that is an **INHERITED-PUBLISHED** step and is flagged as such. Each
subsection ends with what the streams **could not verify**.

### 4.1 Q1 — does agent/object structure emerge for free from prediction? **CONFIRMED, with one boundary condition**

**The hypothesis:** *object individuation does not emerge from reconstruction/prediction alone; it
needs either an architectural bottleneck that forces competition, or a signal that rewards
individuation (motion/flow, depth).*

**VERDICT: CONFIRMED for individuation. TOO STRONG if "object structure" means "object-related
information is present in the latent" — those two are different things, and our measurement sits
exactly on that fork.**

**FOR the hypothesis:**

| finding | source |
|---|---|
| ⭐ **Pixel reconstruction does not individuate on real images.** On COCO, Slot Attention and SLATE (both pixel-reconstruction) score **FG-ARI ≈ 0**; DINOSAUR, which changes **only the reconstruction TARGET to DINO features**, reaches **FG-ARI 34.1 / mBO_i 31.6**. Their diagnosis: pixel reconstruction *"produces too weak of a signal"*. Robust across encoders (DINO 40.9, MoCo-v3 40.4, MSN 40.7, **MAE 37.7**) ⇒ *"features not pixels"*, not *"DINO specifically"* | Seitzer et al., ICLR 2023, [arXiv:2209.14860](https://arxiv.org/abs/2209.14860) |
| **Flow is what binds slots.** SAVi predicts **optical flow** and is conditioned on a **first-frame hint**; without flow it *"fails to learn meaningful object segmentations"* on the harder MOVi++ — and flow is **unnecessary** on the easy MOVi. ⇒ **the requirement appears exactly as scene realism rises** | Kipf et al., ICLR 2022, [arXiv:2111.12594](https://arxiv.org/abs/2111.12594) |
| **Depth is the next rung.** SAVi++ adds **depth prediction** (incl. sparse LiDAR) and is what gets emergent segmentation/tracking on **real Waymo Open** | Elsayed et al., NeurIPS 2022, [arXiv:2206.07764](https://arxiv.org/abs/2206.07764) |
| **Motion individuates even when appearance cannot.** Slot attention on **flow only** (no RGB): DAVIS16 **68.3 J** (prior SSL SOTA 59.2); on camouflage (MoCA) **63.4 J vs supervised MATNet's 64.2**, with zero annotation | Yang et al., ICCV 2021, [arXiv:2104.07658](https://arxiv.org/abs/2104.07658) |
| **And it transfers to driving data.** 2D motion cues drive 2D/3D multi-object discovery on **KITTI and Waymo**, **+8.7 to +15.1 F1@50** over 2D object-discovery baselines | Lahlali et al., CVPR 2025, [arXiv:2503.15022](https://arxiv.org/abs/2503.15022) |
| **Remove the bottleneck and you get collapse, not emergence.** Ablating the object-centric term (λ_oc = 0) collapses training so that *"all pixels are assigned to one slot"* | Đukić et al., ICLR 2025, [arXiv:2503.15141](https://arxiv.org/abs/2503.15141) |
| ⭐ **"Just train it longer" is refuted for this branch.** Object-centric FG-ARI **plateaus around 8 192 samples**; authors: *"we do not find evidence of favorable data scaling laws"*, and **complexity/realism matters more than diversity** | Didolkar et al. 2024, [arXiv:2408.09162](https://arxiv.org/abs/2408.09162) |
| **The theoretical floor.** Unsupervised disentanglement is *"fundamentally impossible without inductive biases"*; 12 000+ models across 7 datasets | Locatello et al., ICML 2019, [arXiv:1811.12359](https://arxiv.org/abs/1811.12359) |
| **Texture is the documented breaking point**, and increasing latent size alone does not recover segmentation | Papa, Winther, Dittadi 2022, [arXiv:2204.08479](https://arxiv.org/abs/2204.08479) ⚠️ *workshop venue — weight accordingly* |

**AGAINST / the boundary condition — reported at equal weight:**

| finding | source |
|---|---|
| ⭐⭐ **THE STRONGEST COUNTER.** V-JEPA — **masked latent prediction, no object-centric architecture, no motion or depth target** — shows **object permanence and shape consistency** under violation-of-expectation: **IntPhys 98 %** (CI 95–99), GRASP 66 %, InfLevel-lab 62 %, vs ~50 % for random init. **VideoMAEv2 (pixel prediction) is only marginally above random init**; Qwen2-VL-7B and Gemini 1.5 Pro near chance. **128 h of unique video suffices** for >70 % | Garrido et al. 2025, [arXiv:2502.11831](https://arxiv.org/abs/2502.11831) |
| ⚠️ **…but read its readout before importing the conclusion.** The measurement is **prediction error in representation space used as a surprise signal** — **not** a segmentation, not an individuation, and not a decoder that must say *which* object. ⇒ **latent prediction installs object-level REGULARITIES; it is not shown to make entities SEPARABLE** | *ibid.* |
| **Emergent segmentation in DINO is real but modest — and from a different objective class.** VOC12 Jaccard **45.9 (DINO ViT-S/16) vs 22.0 random, 27.3 supervised**. But the signal is **self-distillation + multi-crop + momentum** — an invariance/competition objective, **not** reconstruction or prediction — and what emerges is **semantic/saliency, not instance-level** | Caron et al., ICCV 2021, [arXiv:2104.14294](https://arxiv.org/abs/2104.14294) |
| **…which is why dedicated extraction machinery exists on top** (LOST, TokenCut), both targeting a **single** most-salient object per image | Wang et al., CVPR 2022, [arXiv:2202.11539](https://arxiv.org/abs/2202.11539) |
| ⭐ **Emergence can DEGRADE with more training.** DINOv3's Gram anchoring exists because CLS↔patch similarity rises through training so that *"the locality of the patch features diminishes"* ⇒ **directly counter to "our levers are merely under-trained"** | Siméoni et al. 2025, [arXiv:2508.10104](https://arxiv.org/abs/2508.10104) |
| **Emergent attention has documented artifacts** — high-norm tokens in low-information background regions of DINOv2 and others; register tokens fix it and *"enable object discovery methods with larger models"* | Darcet et al. 2023/24, [arXiv:2309.16588](https://arxiv.org/abs/2309.16588) |
| **Pure generation yields emergent tracking** in video diffusion models with no task training — but motion is isolated in the **early high-noise denoising stages**, i.e. a *temporal* factor, and the right stage had to be found | Zhang et al., NeurIPS 2025, [arXiv:2512.02339](https://arxiv.org/abs/2512.02339) |
| ⛔ **The one driving-domain slot success has a large asterisk.** CarFormer's slots are learned self-supervised — **on GROUND-TRUTH BEV maps** (*"We currently assume access to the ground truth BEV maps"*). Slot count is brittle: 7 slots misses vehicles, 30 binds several slots to one. ⇒ **individuation came from the input representation — discrete blobs on an empty raster — not from the slot learner** | Hamdan & Güney, ECCV 2024, [arXiv:2407.15843](https://arxiv.org/abs/2407.15843) |

⛔ **COULD NOT VERIFY (Q1):** SAVi's exact ablation table numbers (two fetches disagreed: 72.0 vs
71.2 mIoU on the same row; the qualitative finding was consistent across both and is what is quoted).
DINOSAUR has **no clean pixel-vs-feature ablation with everything else held fixed** — the comparison
is against *published* pixel-recon methods. Motion Grouping has **no RGB-vs-flow input swap**; its
case rests on the camouflage result. DINOv2's own dense numbers are quoted **as reported in
DINOv3's Table 3**, not from the DINOv2 paper. ⭐ **And the most important absence: NO published work
was found that tests whether masked BEV-cell / occupancy prediction individuates agents** —
adjacent work exists (Occupancy-MAE, BEV-MAE, OccFeat) but nothing ablates individuation.
⚠️ Per rule 2 of the operating standard this is **absence found at one search strategy, not
established absence** — but if it holds it is a gap we are positioned to fill.

### 4.2 Q2 — what frontier self-supervised video models actually use

⭐⭐ **THE MOST DIRECTLY APPLICABLE RESULT IN THE ENTIRE REVIEW, AND I RE-FETCHED IT MYSELF.**

**V-JEPA 2.1: *Unlocking Dense Features in Video Self-Supervised Learning*, Mur-Labadia et al.,
submitted 2026-03-15, [arXiv:2603.14482](https://arxiv.org/abs/2603.14482).**

| what the paper says | value |
|---|---|
| the problem it names in V-JEPA 2 | feature maps *"noisy and fragmented"* local spatial structure |
| the **stated cause** | the loss is applied to **masked tokens only**; the model *"receives no supervision on how it encodes visible context tokens"* ⇒ it takes shortcuts and misses object boundaries |
| the **fix** | a **dense predictive loss in which BOTH visible and masked tokens contribute to the training signal**, plus deep hierarchical supervision |
| ADE20K linear-probe semantic segmentation | **22.2 → 47.9 mIoU** (**+23.4**, gains of +23.4 to +27.6 across segmentation datasets) |
| NYUv2 depth, linear probe | **0.682 → 0.307 RMSE** |
| ⭐ **and the global scores ALSO rose** | IN1K **82.2 → 85.5**; SSv2 **72.8 → 77.7** |

⇒ ⭐⭐ **This is not a trade-off between dense and global quality — supervising every token improved
BOTH.** ⚠️ **Verification note:** my own fetch of the abstract page confirmed the title, date,
the NYUv2 0.307 figure and the dense-loss formulation verbatim, but did **not** surface the ADE20K
pair; those came from the literature stream's fetch of the full text and a corroborating search
result. **Treat 22.2 → 47.9 as PUBLISHED-but-single-path and re-check before it enters a registry
row.**

**The rest of the frontier picture:**

* **JEPA line.** V-JEPA (Bardes et al. 2024, [arXiv:2404.08471](https://arxiv.org/abs/2404.08471)) —
  masked **latent** prediction; frozen-backbone 81.9 K400 / 72.2 SSv2 / 77.9 IN1K. **It makes no
  object-structure claim.** Its frozen protocol uses an **attentive probe**, not a linear head
  (§4.4).
* **DINO family.** Self-distillation at scale; DINOv3 adds **Gram anchoring** specifically to stop
  dense locality degrading. Frozen-backbone ADE20K **55.9 mIoU** (DINOv2 g/14 **49.5**), NYUv2
  **0.309** (0.372) — i.e. **DINO-family frozen dense features are ~2.2× better than V-JEPA 2's on
  the same benchmark**, which is the empirical statement of "latent prediction alone does not buy
  dense structure".
* **Masked pixel modelling.** MAE (He et al. 2021,
  [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)) — its **features are still a usable
  reconstruction target** (37.7 FG-ARI in DINOSAUR's table) but its **linear probe is weak**
  (§4.4). VideoMAEv2's pixel prediction is **near random-init** on intuitive physics (§4.1).
* **Object-centric on video.** VideoSAUR (Zadaianchuk et al., NeurIPS 2023,
  [arXiv:2306.04829](https://arxiv.org/abs/2306.04829)) — feature reconstruction suffices for real
  *images*, but for **video** they add a **temporal feature-similarity loss explicitly described as
  introducing a motion bias**; that combination is what first scales to unconstrained video.

#### 4.2a ⭐ Four frontier facts that change design decisions at OUR scale

**(i) MASKING IS A FIRST-CLASS HYPERPARAMETER, NOT A DETAIL — 21 POINTS.** V-JEPA's own ablation
(ViT-L/16, frozen): random-tube [0.9] **51.5** K400 → causal multi-block[6] 61.3 → causal
multi-block[12] 71.9 → **multi-block 72.9**. Average mask ratio **~90 %**, 3D multi-block with
**tubes through the entire temporal extent**. ⭐ **And causal masking is WORSE than non-causal**
([arXiv:2404.08471](https://arxiv.org/abs/2404.08471)). I-JEPA: **4** target blocks at scale
(0.15, 0.2); predictor at **fixed width 384** regardless of encoder width, and depth 6 → 12 moves
IN-1 % 64.0 → 66.9 ([arXiv:2301.08243](https://arxiv.org/abs/2301.08243)). VideoMAE: **tube 90 %
beats random 90 % (69.6 vs 68.3) and frame masking (61.5)**
([arXiv:2203.12602](https://arxiv.org/abs/2203.12602)).
⇒ ⭐⭐ **Our O3 masks 31.25 % of 16 cells (2 blocks). Frontier practice is ~90 % of 640–1500
tokens.** ⚠️ **The transfer is NOT direct** — 90 % of 16 cells leaves 1.6 cells of context, which is
not the same task. **The right reading is that masking granularity and rate are an unexplored,
weight-free lever, and that our granularity is the binding constraint** — which is the same finding
as §2.3.

**(ii) ⭐⭐ WHAT SUPERVISES THE PREDICTED LATENT MAY DOMINATE EVERYTHING ELSE.** LVDrive
([arXiv:2605.22089](https://arxiv.org/abs/2605.22089), 2026) ablates only the **target encoder**:
**VQGAN-ImageNet 82.39 DS vs DINOv3-Large 71.72 DS vs MoVQGAN 59.91 DS** — a **>10-point driving-score
swing** from the choice of what the predicted latent is regressed onto. ⚠️ **This is a bigger
measured lever than any loss weight in this document**, and our target is our own stop-grad encoder
output. ⚠️ **PREPRINT, peer-review status not established.**

**(iii) ⛔ CAPACITY DOES NOT RESCUE A WEAK OBJECTIVE — stated verbatim in the literature.**
[arXiv:2606.07687](https://arxiv.org/abs/2606.07687) (frozen action-recovery R² across backbones):
*"Scaling Dreamer 4 from 64M to 276M produces effectively no aggregate improvement… Capacity scaling
therefore cannot recover the missing structure when the pretraining objective itself fails to encode
it."* ⭐ **And a sub-100M positive: V-JEPA 2.1 ViT-B (87M) outperforms a 91M pixel-diffusion model at
matched scale.** ⚠️ **Report the inconvenient half too: FROZEN, VideoMAE beats V-JEPA (R² 0.46 vs
0.40); the latent-prediction advantage appears only after in-domain adaptation (0.85 vs 0.75).**
Also: **PSNR/FVD/LPIPS are orthogonal to action recoverability** — generation quality does not
predict whether the latent carries the action. ⚠️ PREPRINT.

**(iv) ⭐⭐ THE CLOSEST PUBLISHED ANALOGUE TO O3 EXISTS, AND IT CARRIES A FREE WEIGHTING RECIPE AND A
DECODER-FREE PROBE.** **AD-L-JEPA**, [arXiv:2501.04969](https://arxiv.org/abs/2501.04969) — pure
latent prediction on **LiDAR BEV cells**, no reconstruction, no labels:
* loss = cosine-similarity on masked BEV embeddings **weighted α₀ = 0.25 (empty) / α₁ = 0.75
  (non-empty)** + a VICReg-style variance term (β = 1, γ = 1/16); EMA 0.996 → 1; **50 % masking**;
* ⭐ **the emergence instrument needs NO DECODER: cosine similarity between a predicted masked
  embedding and a learnable "empty" token** recovers occupancy **with occupancy labels never used**
  (similarity ~0.7, not 1, read as scene uncertainty rather than collapse);
* KITTI moderate: scratch 66.36 → Occupancy-MAE 67.08 → **AD-L-JEPA 67.92**; Waymo→KITTI transfer
  **67.71 vs 66.01**;
* ⛔ **but: "no agent-identity visualization — the demonstrated structure is geometric occupancy,
  not agent individuation."** ⇒ **the single closest thing to O3 in the literature produced
  occupancy, not agents** — which is §5's S2 in someone else's lab.

**(v) And the tiny-model existence proof for a geometric individuation signal.** UnO
([arXiv:2406.08691](https://arxiv.org/abs/2406.08691), CVPR 2024): **17.4 M parameters**, supervised
only by free-space along LiDAR rays and occupancy behind returns — **no boxes, no categories** — and
at precision 0.7 it recalls **stroller ~85 % vs ~35 %**, bicyclist ~75 % vs ~40 % against
box-supervised detectors. ⛔ **BUT LiDAR ray supervision IS a depth/geometry signal**, so UnO does
**not** qualify as "no depth signal", and **we have no LiDAR.** It is cited as proof that the *class*
of geometric individuation signal works at trivial scale, not as something we can copy.

#### 4.2b ⭐ THE ANSWER TO THE SUB-QUESTION — three parts, and the third is the one that binds us

> **Is there any case where agent/object-level structure emerged from a purely predictive latent
> objective with NO object-centric architecture and NO motion/depth signal?**

**(a) YES for coarse OBJECT-LEVEL PHYSICAL structure — one strong case, at a scale we can reach.**
Garrido et al. ([arXiv:2502.11831](https://arxiv.org/abs/2502.11831)): object permanence **85.7 %
vs 51.4 % untrained, effect size g = 9.0 [6.3, 11.7]**, at **ViT-B 115 M** and **128 hours** of
unique video, read out by the pretraining loss itself with no probe.

**(b) NO for INDIVIDUATION — and the field's own strongest lab says so.** V-JEPA 2 measures
**ADE20k 24.4 mIoU and DAVIS J&F 52.5** against DINOv2's 49.5 and DINOv3's 71.1. Recovering
object-level structure required **adding a second, dense loss term**. Independently, the registers
paper shows **LOST object-discovery corloc moving 35.3 → 55.4 on DINOv2 while ADE20k moves only
46.6 → 47.9** ⇒ ⭐ **object structure is far more fragile than aggregate dense metrics reveal.** In
driving specifically, **Latent-WAM's attention lands on lanes and drivable area and explicitly NOT
on objects** ([arXiv:2603.24581](https://arxiv.org/abs/2603.24581)).

**(c) ⛔⛔ AND THE CAVEAT THAT BINDS TANITAD HARDEST — CORPUS DIVERSITY, NOT SCALE.** Garrido et
al.'s data ablation: training on **SSv2 alone** — narrow, motion-centric, ~3 months of video —
gives ***"almost chance-level performance"***; Kinetics-710 is above chance; **HowTo100M
(~15 years, uncurated, diverse) is best.** Subsampling by **video** beats subsampling by **frame**.
⚠️⚠️ **Our corpus is narrow in exactly the way SSv2 is narrow: 2 376 driving episodes, one domain,
one sensor rig family.** ⇒ **The one published case of object-level structure emerging from our
objective class may not transfer to us — not because we are too small, but because we are too
narrow.** This is the most important single caveat in the review and it is **not** fixable by any
weight, any mask rate, or any new term.

**And the counterweight to (a), so it is not over-read:** IntPhys 2
([arXiv:2506.09849](https://arxiv.org/abs/2506.09849)) — photorealistic, dynamic lighting, moving
cameras: **V-JEPA 2 57.51 % (best model) vs human 96.44 %**, with ~6 of 8 models **at chance**, and
models trained on IntPhys 1 **do not transfer**. **The 98 % does not survive scene complexity.**

⭐ **AND THE READOUT-DEPENDENCE IS ITSELF A FINDING — OUR D1 STORY, THREE TIMES OVER.** The same
models give opposite verdicts under different readouts:

| readout | verdict |
|---|---|
| zero-shot surprise, IntPhys 1 | V-JEPA **98 %**; VideoMAEv2 at chance |
| trained **linear** probe, IntPhys 2 | V-JEPA **50.98 % (chance)**; VideoMAE **58.33**; ⭐ **pixel-diffusion LTX-Video 61.76 — the best** |
| **attentive** probe, IntPhys 2 | V-JEPA 56.86; VideoMAE 58.82; V-JEPA 2.1 **66.67** |
| frozen action-recovery R² | VideoMAE **0.46 > V-JEPA 0.40**; after in-domain adaptation V-JEPA **0.85 > 0.75** |

⭐ Best layers are **intermediate, not final** — V-JEPA 2 peaks at 75 % depth (56.86) and **drops to
47.06 at the final layer** ([arXiv:2606.09646](https://arxiv.org/abs/2606.09646), PREPRINT).
⚠️ **We probe the readout output, which is the very last thing in the stack.**

#### 4.2c O6 / SIGReg — what the source paper actually says

**LeJEPA, [arXiv:2511.08544](https://arxiv.org/abs/2511.08544)** — VERIFIED to exist. SIGReg projects
embeddings on **M random unit directions** (512–2048, **1024 recommended**) and applies a univariate
**Epps–Pulley** characteristic-function test per direction, averaged. Bounded gradients
`|∂EP/∂zᵢ| ≤ 4σ²/N` (Thm 4); Jarque-Bera was rejected for **gradient explosion**. ⭐ **Its default
is `λ = 0.05`; our `--w-o6` is `0.1` — 2× the paper's default.** The paper's sweep stays in
**~71–75 %** and *"none of the [hyperparameter] choices lead to a catastrophic collapse."*
⭐ **In-domain small-model result, the most relevant table for us:** Galaxy10 frozen probe, LeJEPA
**ConvNeXt-V2-Nano 82.72 / ResNet-34 83.28** vs **DINOv3 ViT-S/16 81.60 / DINOv2 ViT-S/16 78.34** —
*"domain-specific SSL beats generic transfer learning, even against massive-scale frontier models."*
⛔⛔ **HARD LIMITATION: there are NO video or temporal experiments in the paper. Images only.**
⚠️ Also note SIGReg's stated purpose is to **remove** stop-gradient, the EMA teacher and the
predictor; we run SIGReg **alongside** all three. Not wrong — but it means the paper's guarantees are
not the regime we are in.

⛔ **COULD NOT VERIFY (Q2):** **DINOv3's numeric refinement weights (w_D, w_DK, w_Gram) are NOT in
the paper** — exactly the number one would want for weighting an analogous term. V-JEPA 2.1's own
V-JEPA 2 baseline is **internally inconsistent** (diagnosis text ADE20k **22.2** / NYUv2 0.682;
comparison table **24.4** / 0.642) — **both are quoted above with their source and the discrepancy
is unresolved.** V-JEPA 2.1's distilled ViT-B/ViT-L results are **truncated in the HTML**; ⛔ a
figure circulating as "~41–43 mIoU for ViT-L" was flagged by the stream as **model interpolation,
NOT published — it is not used here.** VideoMAE V2 verified from **abstract only**, and note it uses
a **LABELLED post-pre-training stage**, so its headline numbers are not label-free. GAIA-2's corpus
hours are **arithmetically inconsistent** across the source and are not quoted. ⚠️
[arXiv:2603.14482](https://arxiv.org/abs/2603.14482), 2606.07687, 2605.15618, 2606.09646, 2605.22089
and 2603.24581 are **arXiv preprints whose peer-review status was not established.**

### 4.3 Q3 — weighting, and what `cos = +0.870` licenses

**⭐ THE LOAD-BEARING CITATION.** Du, Czarnecki, Jayakumar, Farajtabar, Pascanu, Lakshminarayanan,
*Adapting Auxiliary Losses Using Gradient Similarity*,
[arXiv:1812.02224](https://arxiv.org/abs/1812.02224). Their gating rule is
`∇L_main + max(0, cos(∇L_main, ∇L_aux))·∇L_aux`. The sentence that decides our case is their reading
of the **high** end: *a cosine of 1 **"is not informative"***, because in that case **using the
auxiliary loss is equivalent to increasing the learning rate**; positive transfer comes mostly from
gradients being **not** perfectly aligned.

⇒ ⭐ **The admissible conclusion for O2↔O5 at +0.870: to first order O2 is acting as a STEP-SIZE
MULTIPLIER on O5, not as an independent constraint.** The **ratio** `w_o2 : w_o5` is therefore
**weakly identifiable** — trading weight between them barely rotates the update — while their
**combined scale relative to O3 and O6** is the parameter that does something. ⭐⭐ **And our §2.1
algebra and §2.1a measurement say the same thing from the other end, independently of any paper.**

**⛔ Gradient surgery is a NO-OP on our geometry — this is a measured argument, not a preference.**
PCGrad (Yu et al., NeurIPS 2020, [arXiv:2001.06782](https://arxiv.org/abs/2001.06782)) defines
conflict as **strictly `cos < 0`**, and additionally requires the *"tragic triad"* — conflicting
gradients **co-occurring with** high positive curvature **and** large gradient-magnitude difference.
Our measured pairs are `+0.870`, `+0.321`, `+0.322`, `+0.127`, `+0.028`, `+0.031`, `+0.042`,
`−0.011`, `−0.019`. ⇒ **PCGrad / CAGrad
([arXiv:2110.14048](https://arxiv.org/abs/2110.14048)) / Nash-MTL
([arXiv:2202.01017](https://arxiv.org/abs/2202.01017)) would fire on essentially nothing.**

**⛔ And four independent papers say sophisticated loss balancing does not beat tuned scalarisation:**

| finding | source |
|---|---|
| Unitary scalarisation *"matches or improves upon"* MGDA/IMTL/PCGrad/GradDrop/RLW, once you add ordinary hygiene (early stopping, ℓ₂ 1e-4/1e-3, dropout). **PCGrad measured 35× slower on CelebA** | Kurin et al., NeurIPS 2022, [arXiv:2201.04122](https://arxiv.org/abs/2201.04122) |
| *"MTO methods do not yield any performance improvements beyond what is achievable"* by a weighted average. They land **on the same Pareto frontier**; learned weights **barely move**; ⭐ **sparse LR-grid sampling produced 6–7× larger variance than random seeds** | Xin et al., NeurIPS 2022, [arXiv:2209.11379](https://arxiv.org/abs/2209.11379) |
| **Random** per-step loss weights are comparable to **twelve** SOTA methods across seven problems | Lin et al., TMLR 2022, [arXiv:2111.10603](https://arxiv.org/abs/2111.10603) |
| *"no evidence that [gradient conflict] is a unique problem in MTL"* for angular alignment; **gradient MAGNITUDE differences are the distinguishing factor** | Elich et al., GCPR 2024, [arXiv:2311.04698](https://arxiv.org/abs/2311.04698) |

**⚠️ THE COUNTERWEIGHTS — do not let the skeptics become a theorem:**

* Kendall, Gal & Cipolla (CVPR 2018, [arXiv:1705.07115](https://arxiv.org/abs/1705.07115)):
  **equal weights lost 9.3 IoU points** vs tuned on the same model (50.1 vs 62.8 %), and the learned
  optimum was **43 : 1 : 0.16** — *"far from uniform, as is often assumed"*. Uniform is not
  automatically safe.
* Hu et al. (NeurIPS 2023, [arXiv:2308.13985](https://arxiv.org/abs/2308.13985)): for linear MTL,
  *"scalarization is in general incapable of tracing out the Pareto front"* under
  **under-parameterisation** — so "tuned scalars are enough" is an *empirical* result at
  over-parameterised scale, not a proof. **We are a sub-300 M model on 2 376 episodes; we are not
  obviously in the over-parameterised regime.**
* **HarmonyDream** (Ma et al., ICML 2024, [arXiv:2310.00344](https://arxiv.org/abs/2310.00344)) is
  the closest published analogue to us: fixed loss coefficients **inside a world model**
  (observation- vs reward-modelling) being imbalanced; automatic coefficient adjustment reports
  **10 %–69 % absolute gains** on visual robotic tasks. ⚠️ Different term structure from ours; cited
  as the existence proof that intra-world-model coefficient balance can matter a lot.

**⭐ THE SSL-SPECIFIC SENSITIVITY NUMBER, and it is the most useful single table for us.** VICReg
(Bardes, Ponce, LeCun, ICLR 2022, [arXiv:2105.04906](https://arxiv.org/abs/2105.04906)) Table 7:

| λ, μ, ν | ImageNet top-1 |
|---|---|
| 1, 1, 1 | **collapse** |
| 5, 5, 1 | 68.1 |
| 10, 10, 1 | 68.2 |
| **25, 25, 1** (default) | **68.6** |
| 50, 50, 1 | 68.3 |
| any setting with μ = 0 | **collapse** |

⇒ ⭐ **Two lessons, both sharp. (i) SCALE is not neutral even at fixed ratio** — λ=μ=1 collapses
while λ=μ=5 works. **(ii) Once inside the basin, a 10× range of weights moves the result by 0.5
points.** ⇒ **fine-tuning ratios is low-value; being in the right basin is high-value.**

**⛔ AND THE HONEST LIMIT ON ALL OF THIS.** The stream found **NO paper measuring gradient cosine
between the loss terms of a single self-supervised world model** — the entire literature is about
**task** gradients in multi-task learning. **Transferring the MTL interpretation to intra-objective
terms is an ASSUMPTION.** ⚠️ **And there is no published threshold for "cosine high enough ⇒ merge
or drop a term"; anyone quoting one is extrapolating.** ⭐ **This is exactly why §2.1's algebra
matters more than the citation: for O2↔O5 we do not need the analogy at all — the identity
`O2 = O5_j + Cov` is exact.**

⛔ **COULD NOT VERIFY (Q3):** GradNorm/CAGrad/Nash-MTL/IMTL/FAMO verified at abstract/venue level
only, **no experimental numbers**. Xin et al.'s specific figures came from the ar5iv rendering only.
Barlow Twins' λ ablation is **figure-only** — no numeric sensitivity values obtainable. Standley et
al.'s r-values are single-source via a PDF-to-text proxy.

### 4.4 Q4 — probing: "present" vs "accessible"

**⭐ THE PROBE HEAD ALONE MOVES THE SCORE BY UP TO 24 POINTS ON IDENTICAL FROZEN WEIGHTS.**

| evidence | numbers |
|---|---|
| **MAE** linear probe vs fine-tune, ImageNet-1K | ViT-B **68.0 → 83.6** (gap 15.6); ViT-L **73.5 → 84.9** (11.4); ViT-H 76.6 → 86.9 (10.3). Tuning **one** block → **81.0**; only the last MLP sub-block → 79.1. Authors: *"linear separability is not the sole metric"* — [arXiv:2111.06377](https://arxiv.org/abs/2111.06377) |
| **Attentive vs linear probing**, same frozen weights | DiT **+24.3**, SimMIM **+13.6**, MAE **+7.9**, BEiTv2 +2.7, DINO +0.5, DINOv2 +0.8 — Psomas et al., [arXiv:2506.10178](https://arxiv.org/abs/2506.10178) |
| **V-JEPA**, same frozen encoder, pooling choice only | average pooling 56.7 (K400) / 50.1 (SSv2) → attentive **73.7 / 66.2** = **+17.3 / +16.1** — [arXiv:2404.08471](https://arxiv.org/abs/2404.08471) |

⇒ ⭐⭐ **A probe disagreement of the size we observed is INSIDE the range that probe architecture
alone explains.** Note the pattern: the gap is **largest exactly for models trained on local /
masked objectives** (DiT, SimMIM, MAE) and **smallest for DINO-family** — i.e. our objective class
is the one for which linear probing most understates the representation.

**The asymmetry, and it favours the linear probe:**

* **Pimentel et al., ACL 2020, [arXiv:2004.03061](https://arxiv.org/abs/2004.03061):** by the
  data-processing inequality a representation cannot contain more about a target than its input
  does, so probing is really about **ease of extraction**; probe performance is a **LOWER BOUND** on
  the information present, and *"one should always select the highest performing probe one can"*.
* **Belinkov, *Computational Linguistics* 48(1), [arXiv:2102.12452](https://arxiv.org/abs/2102.12452):**
  a **low probing score does not establish absence** — "not encoded usefully" and "the probe is
  limited" are indistinguishable.

⇒ ⭐ **The methodological verdict on our own discrepancy:**
> **A probe that SUCCEEDS raises the lower bound. A probe that FAILS leaves it where it was.**
> ⇒ the ridge's positive result is the informative one; the slot decoder's null **cannot overturn
> it** and was never able to. ⭐ **Our positive-control run reached exactly this conclusion
> empirically, in advance of the literature** — which is a strong sign the instrument discipline is
> working.

**⚠️ AND THE TWO CAUTIONS ON OUR *POSITIVE* RESULT — these matter more than the null:**

1. ⛔ **Kumar, Tan & Sharma, NeurIPS 2022, [arXiv:2207.04153](https://arxiv.org/abs/2207.04153):**
   even where concept features alone would give 100 % accuracy, *"a probing classifier is likely to
   use non-concept features"* — **a positive probe can be carried by a correlate.** ⭐⭐ **Applied
   to us: lead gap correlates with EGO SPEED (you follow further when faster), and our latent
   certainly encodes ego speed — the speed channel is a trained input.** ⚠️ **The ridge's
   1.6–1.8 m advantage and `r = +0.159` may be entirely a speed correlate and NOT agent
   information.** This is the same family as the nav-echo defect and the sitclf leak, and it is
   **unchecked**. It is experiment **E-PROBE-A** (§6.3) and costs **zero GPU**.
2. **Elazar et al. (amnesic probing, [arXiv:2006.00995](https://arxiv.org/abs/2006.00995)) and
   Ravichander et al. ([arXiv:2005.00719](https://arxiv.org/abs/2005.00719)):** *"conventional
   probing performance is not correlated to task importance"*, and properties distributed as
   **random noise** were still decoded above chance. ⇒ **even a clean positive says nothing about
   whether the world model USES the information.**

**And the probe-capacity instrument we do not have:** Voita & Titov, EMNLP 2020,
[arXiv:2003.12298](https://arxiv.org/abs/2003.12298) — MDL/codelength probing. Their headline is
directly relevant: on POS tagging, **accuracy 93.7 % (real) vs 96.3 % (random-label control)** —
accuracy says the *control* is better learned — while **codelength 163 vs 267 kbits** inverts it.
Across 10 probe configurations **accuracy flipped its verdict while MDL was stable**. Control tasks
needed probes **3–4× larger**. Also: Hewitt & Liang, EMNLP 2019,
[arXiv:1909.03368](https://arxiv.org/abs/1909.03368) — **selectivity** (task accuracy minus
control-task accuracy) as the probe-design criterion.

⛔ **COULD NOT VERIFY (Q4):** AIM/CAE/CAPI as origins of attentive probing were verified only
**second-hand** via Psomas et al.'s attribution. The ICLR 2026 venue for
[arXiv:2506.10178](https://arxiv.org/abs/2506.10178) is as stated on the arXiv page, not
independently confirmed. ⭐ **And no paper was found that studies slot-structured decoder probes vs
linear probes on world-model latents** — the transfer of the MAE/V-JEPA attentive-probe argument to
slot decoders is **reasoned analogy, not a published finding**, and is labelled as such.

---

## 5. ⭐ SYNTHESIS — the literature and our source read the same way

**Five statements, each traced to both sides.**

**S1. We are on the RIGHT side of the pixel/latent divide, and that half of the design is
vindicated.** Every result that separates them says predicting in a learned feature space is the
productive choice: DINOSAUR (pixel FG-ARI ≈ 0 vs feature 34.1), Garrido (latent 98 % IntPhys vs
pixel near-chance). **O2/O3/O5 predict latents. Keep that.**

**S2. But the literature is unanimous that latent prediction installs REGULARITIES, not
INDIVIDUATION** — and individuation is what O2/O3/O4 were commissioned to produce. No case was
found of separable entities emerging from a purely predictive latent objective. **⇒ O2/O3/O4 are
underpowered IN KIND for the job of making agents individually readable, and no value of `--w-o2` or
`--w-o3` changes that.** ⚠️ **This is NOT the same as saying they are useless** — see S5.

**S3. Our specific mechanism is now named, and it has a published twin.** V-JEPA 2's diagnosis —
*"no supervision on how it encodes visible context tokens"* ⇒ noisy, fragmented local structure,
ADE20K 22.2 vs DINOv2's 49.5 — is **the same failure mode as ours, arrived at by a different
route**. Theirs: the loss touches only masked tokens. **Ours: the loss touches only the 16 cells
that remain after a 40:1 average pool (§2.3).** ⭐ **In both cases the fix that WORKED was giving
the objective access to more of the token field — and in theirs it improved dense AND global scores
simultaneously (+23.4 ADE20K, +3.3 IN1K).**

**S4. "Be patient" is refuted THREE times, from three directions.** Didolkar et al.: object-centric
performance **plateaus at ~8 k samples**, *"no evidence of favorable data scaling laws"*. DINOv3:
dense **locality DEGRADES with longer training** unless a term is added to stop it.
[arXiv:2606.07687](https://arxiv.org/abs/2606.07687), verbatim: *"capacity scaling cannot recover
the missing structure when the pretraining objective itself fails to encode it"* (Dreamer 4 flat
from 64 M to 276 M). ⇒ ⛔ **"O2/O3/O4 are right and merely need patience" is NOT supported.**
⚠️ **The brief explicitly invited that answer as a legitimate outcome. The evidence does not support
it, and I am saying so rather than taking the safe option.**

**S5. …but our own numbers say the levers are not idle, either.** O3 is the **one term with an
independent gradient direction and its own parameters** (§3.1), and O6 is orthogonal to everything.
The latent does carry *something* linearly (§3.4). And Garrido et al. is direct published evidence
that the O2/O3/O5 class buys real physical understanding. ⇒ **The correct verdict is
"under-specified", not "wrong".** The objective has three effective directions where it advertises
five, and none of the three rewards individuation.

**S6. ⛔⛔ AND THE FINDING I LIKE LEAST, WHICH IS WHY IT LEADS THE ESCALATIONS: OUR CORPUS MAY BE
TOO NARROW FOR THIS OBJECTIVE CLASS TO WORK AT ALL.** The single published case of object-level
structure emerging from masked latent prediction (Garrido et al.) **fails on a narrow,
motion-centric corpus** — SSv2 alone gives *"almost chance-level performance"* — and needs
HowTo100M-scale **diversity**, not volume. **Our 2 376 driving episodes are narrow in exactly that
way.** ⚠️ **This is not a weighting problem, an architecture problem, or a patience problem, and
none of R1–R4 addresses it.** ⚠️ It is **HYPOTHESIS-strength for us** — one paper's data ablation,
on a different domain, and driving video is not obviously "narrow" in the same sense that a
gesture-recognition corpus is. **But it is testable and nobody has tested it here.**

**S7. ⭐ AND THE LARGEST MEASURED LEVER IN THE WHOLE REVIEW IS ONE WE HAVE NEVER VARIED.** LVDrive's
target-encoder ablation swings driving score **>10 points** (82.39 / 71.72 / 59.91 DS) purely by
changing **what the predicted latent is regressed onto**. Our target for O2/O3/O5 is our own
stop-grad encoder output and **has never been an arm.** ⚠️ Preprint, one paper, different stack —
but it is a bigger measured effect than any loss weight in this document.

⭐ **THE ONE-SENTENCE SYNTHESIS:**
> **We built three levers to make agent structure emerge, then placed all of them behind an average
> pool that removes the spatial resolution agent structure lives in — and two of the three are not
> independent levers at all. The weighting question is real but second-order; the ACCESS question is
> first-order, and the frontier has already published its fix.**

---

## 6. ⭐ RECOMMENDATION

⛔ **RECOMMENDATION ONLY. Nothing here was implemented, no weight was changed, the live v6F 30 k was
not touched.** Every item is the PI's call. ⚠️ **R1 and R2 add parameters and therefore CANNOT enter
the live run** — a tensor-strict resume would refuse them. They belong at a **stage boundary under
`STAGE_MAY_INTRODUCE`** (the mechanism `tac_goal_cond` already uses at S-T) or in a **fresh arm**.

### 6.1 The verdict per lever

| lever | verdict | why |
|---|---|---|
| **O2** | ⛔ **DEMOTE or MERGE — it is not an independent lever** | `O2 = O5_j + Cov` exactly (§2.1); Cov measured at **0.45–3.33 %, median 1.81 %, sign-unstable** at init (§2.1a); `cos +0.870` (§3.1); Du et al.: at high cosine an aux loss *is* a learning-rate increase. Its one degree of freedom is a 4-value row tilt that **does not cover where the lead is** (§2.2) and **nearly vanishes at motorway speed** (3.6× at 30 m/s) |
| **O3** | ✅ **KEEP at 1.0 — the only independently-directed structural term we have.** ⭐ **And it has a free, published upgrade that is not a weight** | sign-unstable cosines to everything (+0.028/+0.031/+0.042), own parameters, 7.7 % of trunk pull (§3.1-3.2). ⭐ **AD-L-JEPA — the closest published analogue — weights its masked-BEV-cell loss α₀ = 0.25 (empty) / α₁ = 0.75 (non-empty)**, i.e. it up-weights *occupied* cells. That is a directly transferable, published, **intra-term** weighting we do not have. ⚠️ but O3 too sees only the 16 pooled cells, and its 31 % mask rate is far off frontier practice (§4.2a-i) |
| **O4** | ✅ **KEEP at `alpha` 1.0 — and ⛔ STOP CALLING IT AN INTERACTION LEVER** | zero gradient, zero perception content, ego-kinematics only (§1.3). Didolkar: *complexity/realism of data matters more than diversity* ⇒ a saliency re-sampler is a cheap, parity-safe way to raise sample complexity, and **Garrido et al.'s "subsample by VIDEO, not by FRAME" is the same shape of finding**. But it **cannot teach individuation** and **no citation anywhere in this review suggests a sampler ever could** |
| **O5** | ✅ **KEEP at 1.0** | it is the actual rollout-consistency objective; O2 is its shadow |
| **O6** | ✅ **KEEP — PI already ruled.** ⚠️ **but note `0.1` is 2× the source paper's default** | orthogonal to all (§3.1); VICReg Table 7: **μ = 0 collapses at every setting of the others**. ⭐ **LeJEPA's own default is `λ = 0.05`; ours is `0.1`** — and its sweep is flat (~71–75 %) with *"none… catastrophic"*, so this is a **note, not a defect**. ⚠️ our own measurement says the collapse *"does not develop"* (§3.3) — that bounds the term's *urgency*, not its correctness. ⛔⛔ **And LeJEPA has ZERO video/temporal experiments; we also run it alongside the stop-grad, EMA and predictor it was designed to REMOVE — so we are outside the paper's regime** |

### 6.2 What to ADD — in priority order

**R1 (HIGHEST VALUE — frontier-verified, and the plumbing already exists).**
**Give at least one objective access to the PRE-POOL token grid.** Concretely: an O3-style masked
predictive term computed on the encoder's **640 patch tokens**, not on the 16 pooled cells, with
**both visible and masked tokens supervised** (V-JEPA 2.1's exact prescription).
* ⭐ **The exact published mechanism, not just the idea:** `L_dense = L_predict + L_context`, where
  the context term supervises **visible** patches with **distance-weighted coefficients
  `λᵢ = λ / √(d_min(i, M))`** — visible tokens weighted inversely by distance to the nearest masked
  region. Their stated root cause for V-JEPA 2's failure: masked-only supervision lets context
  patches act as *"global aggregators"* rather than encoding localised information. ⭐⭐ **That is
  precisely what a 40:1 average pool forces our cells to be, by construction.**
* **Published warrant:** ADE20K linear probe **→ 47.9 mIoU** (from **24.4** per the comparison table
  / **22.2** per the diagnosis text — ⚠️ the paper is internally inconsistent, §4.2c), NYUv2
  **0.642 → 0.307**, ⭐ **DAVIS J&F 52.5 → 69.0** (a *tracking* metric, the most agent-relevant one
  available), **and global scores rose too** (IN1K 82.2 → 85.5, SSv2 72.8 → 77.7) —
  [arXiv:2603.14482](https://arxiv.org/abs/2603.14482).
* **Our warrant:** §2.3 — the objective currently never sees the agent, only its ~1/10 contribution
  to a 40-token average.
* ⭐ **And it makes O3's mask rate meaningful for the first time.** At 640 tokens a frontier-style
  high mask ratio is a real task; at 16 cells it is not (§4.2a-i). **Granularity and masking are one
  problem, not two.**
* ⭐ **Cost is far lower than it looks: `encode_window(..., return_tokens=True)` ALREADY EXISTS
  and is tested** (`v6.py:3691`, the F-18 `slot_src="tokens"` arm). The docstring's own cost note is
  `B × W × 640 × 768` floats ≈ **63 MB at B=8, W=4** — affordable on Thor.
* ⚠️ **It does not by itself produce individuation** (S2). It removes the bottleneck that makes
  individuation *impossible*, and it is the change with the strongest published effect size.

**R2 (THE ONLY LEVER THE LITERATURE SAYS ACTUALLY INDIVIDUATES — and it is label-free for us).**
**Add an ego-motion-compensated residual-motion target.** Optical flow between consecutive frames,
minus the flow predicted by the ego's own motion ⇒ **"what moves that my own motion does not
explain"** = other agents.
* **Published warrant:** SAVi (flow is what binds slots; without it, failure on realistic scenes),
  SAVi++ (depth on real Waymo), Motion Grouping (**flow alone**: 68.3 J DAVIS16; **63.4 J on
  camouflage vs supervised 64.2**), VideoSAUR (temporal-similarity motion bias is what scales OC to
  unconstrained video), xMOD (**+8.7–15.1 F1@50 on KITTI and Waymo**).
* ⭐ **Admissibility, checked explicitly against the two binding rules:**
  – **Label-free** ⇒ the unsupervised path is preserved. Flow is *computed from the video*, not
  annotated. This is why it is the right answer to the PI's constraint rather than a detour around
  it.
  – **Vision-only at inference** ⇒ ✅ satisfied. Flow is a **training target**; nothing changes at
  inference. Ego-motion compensation uses ego state, which the binding rule permits **explicitly**
  for *label derivation* (*"ego state, other agents, maps, future poses — anything"*).
  – **Parity** ⇒ ✅ untouched; this adds a target, it does not re-select episodes.
* ⚠️ **Depth (SAVi++'s signal) is NOT available to us** — PhysicalAI-AV ships no depth channel
  (settled at five probes in `CLAUDE.md`). **Motion is the one individuation cue we can actually
  get.** That is why R2 is flow and not depth.
* ⚠️ **UNVERIFIED for our corpus:** flow quality on 10 Hz 120°-FOV cylindrical frames at 256×640,
  and the compute cost of precomputing it over 2 376 episodes. **Both must be measured before this
  is scheduled** (E-FLOW-A, §6.3).

**R3 (WEIGHTING — the honest answer is "don't build a weighter").**
* ⛔ **Do NOT adopt PCGrad / CAGrad / Nash-MTL.** Three independent reasons, each with a number:
  they are **no-ops on our measured geometry** (PCGrad fires only at `cos < 0`; our most negative
  pair is −0.019); **four papers** find no gain over scalarisation; **PCGrad measured 35× slower**
  on CelebA (Kurin et al.).
* ⛔ **Do NOT grid-search the six weights.** Xin et al.: MTO weights trace the **same Pareto
  frontier** as static scalars, and **LR-grid variance was 6–7× seed variance** — an under-tuned LR
  would manufacture a fake "weight win".
* ✅ **DO check the SCALE, not the ratios.** VICReg collapses at λ=μ=1 and is flat to 0.5 points
  across a 10× range — the question is which basin we are in, not which point inside it.
* ✅ **DO free the weight O2 is spending.** If **E-O2-A** returns Outcome A, `--w-o2` is buying
  ~2 % of a direction `--w-o5` already covers, and the objective has one fewer real knob than the
  CLI advertises. **Say so in the config, whatever is decided.**
* ⚠️ **If an automatic weighter is ever wanted**, the literature's cheap picks are Kendall
  uncertainty weighting (one run, beat approximate grid search 63.4 vs 62.8) or FAMO
  ([arXiv:2306.03792](https://arxiv.org/abs/2306.03792), O(1) space/time, **no per-task
  gradients** — the only one that does not multiply our backward passes by six).

**R4 (PROBING — fix the instrument before trusting either result).**
* ⛔ **Run the confound check on the RIDGE before its +1.6–1.8 m is quoted again** (E-PROBE-A).
* ✅ **Re-read the latent with an ATTENTIVE probe, not only linear.** Measured gap on frozen weights
  is **+7.9 to +24.3 points**, and it is **largest precisely for masked/local objectives like
  ours**. A linear probe is the *weakest* admissible instrument for our objective class.
* ⭐ **Steal AD-L-JEPA's DECODER-FREE readout.** Cosine similarity between a predicted masked
  embedding and a **learnable "empty"/"occupied" token** recovers occupancy **with labels never
  used** — no slot decoder, no Hungarian matching, no 74-vs-16-query operating point to get wrong.
  ⭐⭐ **Given that our entire F-18 line was destroyed by a decoder's readout rule, a probe with no
  decoder is worth more to this programme than to the paper it comes from.**
* ⚠️ **Probe the INTERMEDIATE representation too, not only the readout output.**
  [arXiv:2606.09646](https://arxiv.org/abs/2606.09646) measured V-JEPA 2 peaking at **75 % depth
  (56.86)** and **falling to 47.06 at the final layer**. **We probe the readout — the very last
  thing in the stack, and the one behind the 40:1 pool.**
* ✅ **Report probe **codelength**, not only accuracy** (Voita & Titov) — accuracy flipped its
  verdict across 10 probe configurations while MDL did not.

**R5 (⭐ THE UNVARIED LEVER — raise it with the PI even though it is out of scope here).**
**What supervises the predicted latent has never been an arm in this programme.** LVDrive measured
a **>10-point driving-score swing** (82.39 / 71.72 / 59.91 DS) from the target encoder alone —
larger than any loss weight discussed in this document. Our O2/O3/O5 all regress onto our **own
stop-grad encoder output**. ⚠️ **This is a pre-registration-shaped question, one preprint deep, and
I am flagging it rather than recommending it** — but it should not stay invisible simply because
nobody has ever varied it.

### 6.3 ⭐ THE CHEAPEST DISCRIMINATING EXPERIMENTS — both outcomes pre-committed

| id | question | method | cost | **outcome A** | **outcome B** |
|---|---|---|---|---|---|
| ⭐⭐ **E-O2-A** | Is O2 an independent lever at CONVERGENCE, or still O5's shadow? | Read `o2_loss` and `o2_unweighted` from the **live v6F 30 k log** at ≥3 well-separated steps; compute `\|o2_loss − o2_unweighted\| / o2_unweighted` and its sign. Script banked: `raw/2026-08-17-O234/o2_cov_from_logs.py` | ⭐ **ZERO GPU. Reading a log.** | **< 10 % and/or sign-unstable ⇒ O2 is a re-parameterisation of O5's step-`j` term.** Demote/merge; record that the objective has 3 effective directions, not 5 | **≥ 10 % and sign-stable ⇒ the spatial reallocation IS doing work at convergence**; +0.870 understates its distinctness; O2 stays at 1.0 and §2.1a is superseded |
| ⭐⭐ **E-PROBE-A** | Is the ridge's lead-gap signal real, or an EGO-SPEED correlate? | On the **banked** @11250 / @9000 / null caches: (i) ridge on `v_ego` alone → lead gap; (ii) ridge on the latent → **residual** after removing the speed prediction; same 70 clusters, same paired episode-cluster bootstrap | ⭐ **ZERO GPU** — closed-form solves on banked tensors | **Latent beats the speed-only baseline on the residual ⇒ genuine (weak) agent information; §3.4 stands** | **It does not ⇒ the +1.6–1.8 m and `r = +0.159` are a SPEED CORRELATE, `PROBE_POSITIVE_CONTROL.md` §2.3's "genuinely new fact" must be retracted, and the programme has NO evidence the latent carries lead geometry** |
| **E-PROBE-B** | Is the linear/slot gap a probe-format artefact? | Re-read the same banked caches with an **attentive** probe (learnable query + cross-attention) beside the ridge and the repaired 16-query slot probe; report all three with the same estimator | low — one head, banked caches | **Attentive ≫ linear ⇒ our latent stores it non-linearly; the linear number was a floor** | **Attentive ≈ linear ⇒ the information really is thin, and the format hypothesis is not what was limiting** |
| **E-TERM-A** | Does O2 earn its place *at all*? | TAG-style **lookahead** (Fifty et al., [arXiv:2109.04617](https://arxiv.org/abs/2109.04617)): take one optimiser step on O2 alone, read the change in O5's loss, and vice versa; average over a few hundred steps. ⚠️ **Not a gradient cosine** — the affinity literature says cosine is a hypothesis generator, not a verdict | one extra forward per probe step; **no retrain** | **Lookahead affinity ≈ that of doubling `w_o5` ⇒ confirms the LR-multiplier reading (Du et al.)** | **Distinguishable ⇒ O2's row tilt is a real second direction and the collinearity reading is wrong** |
| **E-FLOW-A** | Is R2 buildable on OUR corpus? | Feasibility only: run an off-the-shelf flow estimator on ~200 sampled windows at 256×640/10 Hz; ego-compensate with `egomotion` + `camera_intrinsics`; measure (i) flow quality, (ii) whether residual flow **separates the known `obstacle.offline` agents** from background, (iii) precompute cost over 2 376 episodes | small, and **it is a perception measurement, not a training run** | **Residual flow separates agents ⇒ R2 is buildable and gets a pre-registration** | **It does not (rolling shutter, 10 Hz too sparse, cylindrical warp) ⇒ R2 is dead on this corpus and the individuation lever must come from AlpaSim/NuRec instead** |

⚠️ **Priority if only one is run: `E-O2-A`.** It is a log read, it settles the PI's weighting
question directly, and it can only confirm or refute the sharpest claim in this document.

---

## 7. ⛔ WHAT THIS DOCUMENT DOES NOT COVER — read before acting

1. ⚠️ **The `2026-08-17-latent-linear-ladder` stream had not landed a report** (`code/` and `raw/`
   present, **no `.md`**). If it contradicts §3.4, **the ladder wins** — it is the newer instrument.
2. ⚠️ **§2.1a's 0.45–3.33 % is at INITIALISATION, n = 7 dry-ladder rows.** It does not establish
   O2's size at convergence. **E-O2-A is the measurement that does**, and it is a log read.
3. ⚠️ **The cell→metre table is `ESTIMATED`, by its own docstring.** Every statement in §2.2 about
   *where* O2 puts its weight inherits that. A calibrated table would change the profile but **not**
   the §2.1 identity, which is geometry-independent.
4. ⚠️ **The MTL→intra-objective transfer in §4.3 is an ASSUMPTION** — no paper measures gradient
   cosine between the loss terms of one SSL world model. §2.1's algebra does not depend on it.
5. ⚠️ **PROVENANCE: the literature was fetched by three parallel streams; only V-JEPA 2.1 was
   re-fetched by me** (and my fetch confirmed the title, date, dense-loss formulation and NYUv2
   0.307 but **did not surface the ADE20K pair**). **Anything destined for `MODEL_REGISTRY.md` must
   be re-verified at source first.**
6. ⚠️ **Several load-bearing citations are arXiv PREPRINTS of unestablished peer-review status:**
   2603.14482 (V-JEPA 2.1 — carries **R1**), 2605.22089 (LVDrive — carries **R5**), 2606.07687,
   2605.15618, 2606.09646, 2603.24581. ⛔ **R1 rests on a preprint.** The *derivation* in §2.3 does
   not, which is why R1 is defensible without it — but the effect size is preprint-grade.
7. ⛔ **V-JEPA 2-AC's action-conditioned recipe is summarised but NOT used here.** For the record,
   because it bears on our S-T stage: frozen encoder, ~300 M new block-causal predictor, **two L1
   terms summed with EQUAL weight** (`L_teacher-forcing` at T=15 + `L_rollout` at T=2), trained on
   **<62 h / ~23 000 trajectories**, planning by CEM in representation space. ⚠️ "Unlabelled" there
   means no reward/task/success flag — **actions ARE used**, from proprioception.
8. ⛔ **NOT ANSWERED: whether masked BEV-cell / occupancy prediction individuates AGENTS.** Two
   independent streams searched and found **no paper that ablates it**; the closest (AD-L-JEPA)
   demonstrates **occupancy, explicitly not agent identity**. ⚠️ Per operating-standard rule 2 this
   is **absence found at two search strategies, which is still not established absence** — but if it
   holds, it is a genuine gap and we are positioned to fill it.

---

## 8. ⛔ ESCALATIONS — these do not belong in a README

1. ⭐⭐ **`E-O2-A` should be run by whoever next touches the live log. It is a LOG READ, zero GPU,
   and it settles the PI's weighting question directly.** Script banked and tested:
   `raw/2026-08-17-O234/o2_cov_from_logs.py`. If it returns Outcome A, **`--w-o2` is not an
   independent knob and the config should say so.**
2. ⛔ **`PROBE_POSITIVE_CONTROL.md` §2.3's "genuinely new fact about the v6 latent" (+1.6–1.8 m,
   r = +0.159) is UNCHECKED AGAINST AN EGO-SPEED CONFOUND** ([arXiv:2207.04153](https://arxiv.org/abs/2207.04153):
   a positive probe is likely to ride a correlate). **Lead gap correlates with ego speed and our
   latent is trained with a speed channel.** `E-PROBE-A` is zero-GPU on banked caches. ⚠️ **Until it
   runs, that sentence should not be re-quoted as evidence the latent carries agents.**
3. ⚠️ **S6 — the corpus-narrowness risk — belongs to the PI, not to this stream.** If Garrido et
   al.'s data ablation transfers, **no change to O2/O3/O4/O5/O6 fixes it**, and the answer is corpus
   composition (AlpaSim, NuRec, or external video). It is HYPOTHESIS-strength and it is the only
   finding here that could invalidate the whole objective class.
4. ⚠️ **R1 and R2 add parameters ⇒ they CANNOT enter the live v6F 30 k**, which resumes
   tensor-strict. They belong at a stage boundary under `STAGE_MAY_INTRODUCE` or in a fresh arm.
   **This is a scheduling constraint, not a design objection.**
5. ⚠️ **`DIAGRAM_CONFORMANCE.md` fixes F-7/F-8/F-9 (T2 contrastives, T5 temporal consistency, T3
   multi-agent curriculum) are all still ⬜ NOT BUILT.** ⭐ **T5 (momentum-aware temporal
   consistency) is the closest existing catalog entry to R2's motion signal, and F-9's own note
   says T3 is gated on P8 occupancy maturity** — i.e. **the programme's own catalog already points
   where this review points.**

