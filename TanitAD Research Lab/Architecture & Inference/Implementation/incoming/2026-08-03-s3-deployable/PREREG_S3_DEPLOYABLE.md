# PRE-REGISTRATION — E-SEL-1D: the DEPLOYABLE consequence score

**Date:** 2026-08-03 (Europe/Berlin) · **Stream:** arch-inf · **Status:** written and STAGED
**BEFORE any statistic was computed.** No number below was seen first.

**Parent:** `Project Steering/PREREG_D-SEL_REFC_SELECTION_SURFACE.md` §6.2 / §6.3
**Trigger:** `…/incoming/2026-08-03-esel-verdict/ESEL_VERDICT.md` §4.1 and escalation §9.2 —
*"measure the DEPLOYABLE consequence score before S3 is costed."*

**Estimator, declared before any number.**
`taniteval/taniteval/ci.py::episode_cluster_bootstrap` (point + interval) and
`::paired_episode_cluster_bootstrap` (two statistics on the same windows), resampling unit =
**episode**, `n_boot = 2000`, canonical val `physicalai-val-0c5f7dac3b11`, **881 windows / 40
episodes**. ⛔ **`overlapping_holdout_se` is never called.** It is not a jackknife, it is not a valid
SE, and it biases the POINT estimate (mean-of-split-means vs `full_set`, measured −6.67 % to
+11.69 % over 27 arms, bidirectional, up to a sign flip on paired deltas).

---

## 1. The defect in the existing number

E-SEL-1 measured Spearman **ρ = 0.6657** [0.6183, 0.7157] (`refc-base-30k`) and **0.6212**
[0.5650, 0.6791] (`refc-xl-30k`) and returned verdict **S3 LIVE**.

The statistic was

```
s_oracle[i,j] = − mean_sq( law_head([pooled_i, fan_ij]) − z_{t+5,i} )      z = encode_pooled(frame_{t+5})
```

⚠️ **`z_{t+5}` is the FUTURE FRAME.** The thing S3 deploys — `refc_select.consequence_scores`,
called at `refc.py:1246-1251` with `cons_ctx = pooled` (`refc.py:1836`) — never sees it. From source
(`stack/tanitad/refs/refc_select.py:308-321`) the deployed score is

```
s_deploy[i,j] = conf_head( layer_norm( feat_proj( law_head([pooled_i, fan_ij]) ) ) )
```

⇒ **ρ_oracle is an UPPER BOUND on the information present, not S3's deliverable gain.** Quoting
0.6657 as "S3's effect size" is the same shape as the C6 confound (a decoder scored on its own
marginal) and the REF-A I-JEPA leak (~80 % of val inside train): **a measurement-time input the
deployed path does not have.** This pre-registration fixes, in advance, what the deployable number
has to be for S3 to be funded.

---

## 2. What is measured

Per window `i`, per candidate `j`, on the **same 881 windows**, all statistics paired:

| symbol | score | sees the future? |
|---|---|---|
| **ρ_oracle** | `s_oracle` above — E-SEL's statistic, recomputed here | ✅ yes — upper bound only |
| **ρ_deploy** | `s_deploy` above — **the headline of this experiment** | ⛔ **no** |
| **ρ_ctxswap** | `s_deploy` with `pooled` globally permuted across the 881 windows (derangement, seed 20260803) | ⛔ no |
| **ρ_cv** | `s_cv[i,j] = −mean‖fan_ij − cv_i‖`, the banked constant-velocity baseline. **ZERO parameters, zero compute** | ⛔ no |
| **ρ_shuffled** | `s_deploy` permuted along the candidate axis | ⛔ no (**declared vacuous**, §3.6) |

ρ_X = per-window Spearman(`s_X[i,:]`, `−ADE[i,:]`), aggregated by episode-cluster bootstrap.
`ADE[i,j]` is `taniteval.refc_rerank._score_row`'s definition verbatim, via
`refc_sel_probe.candidate_ade`.

Both arms: **`refc-base-30k`** (128 anchors) and **`refc-xl-30k`** (256), step 29,999.

---

## 3. NEGATIVE CONTROLS — run and reported BEFORE the headline

E-SEL passed four and then found one of them was vacuous by construction. That finding is carried
here: a control that cannot fail is not a control.

| # | control | what it establishes | **can it fire?** |
|---|---|---|---|
| **3.1 C-raster** | fed raster asserted against the arm's own `grid_shape` (256×256 / 8×8 = 64 tokens) before a single window is scored | R-2026-08-02-a: base returns numbers **silently** at the wrong raster | ✅ yes (it fired historically) |
| **3.2 C-identity** | `argmax(logits) == sel_idx` on 1.0000; selected ADE reproduces published **0.4728** / **0.4714** (tol 1.5e-4) | the harness changed nothing else | ✅ yes |
| **3.3 C-reproduce-esel** ⭐ NEW | the re-decode must reproduce E-SEL's banked `fan`, `logits`, `refined_logits`, `cons_score` **BIT-FOR-BIT**. Eval decode is deterministic — `refc.py:1217-1218` puts noise behind `if self.training` | if it deviates, ρ_deploy and ρ_oracle are not on the same fan and the paired contrast is **void** | ✅ **yes** |
| **3.4 C-oracle-reproduce** ⭐ NEW | ρ_oracle recomputed here must land inside E-SEL's CI — [0.6183, 0.7157] base, [0.5650, 0.6791] XL | if it does not, this pipeline is not the one that produced the upper bound and the gap is not attributable | ✅ **yes** |
| **3.5a C-ctxswap** ⭐ NEW, load-bearing | `pooled` globally permuted, `fan` unchanged. Asks: **is the score reading the scene at all, or only trajectory shape?** | ✅ **yes, in both directions** |
| **3.5b C-cv** ⭐ NEW, load-bearing | a **zero-parameter** score: negative distance to the constant-velocity baseline already in the bank | if a free heuristic matches S3, S3's parameter and its per-candidate `law_head` evaluation are unjustified | ✅ **yes** |
| **3.5c C-degenerate** ⭐ NEW | median per-window std of `s_deploy` across candidates. `< 1e-6` ⇒ `assert_candidate_axis` would raise and ρ is noise | the exact silent failure `consequence_scores`' docstring names | ✅ **yes** |
| **3.6 C-shuffled** | kept for continuity with E-SEL — and **DECLARED VACUOUS**: for Spearman on a permuted candidate axis `E[ρ] = 0` analytically, for **any** score. It establishes only `ρ ≠ 0`, which is a weak bar | ⛔ **NO — reported, never load-bearing** |

⛔ **The load-bearing controls are 3.5a and 3.5b.** The verdict in §4 is written against them, not
against C-shuffled.

---

## 4. ⛔ BOTH OUTCOMES, COMMITTED IN ADVANCE — thresholds fixed HERE

The bar **`|ρ| ≥ 0.10`** is the parent pre-registration's own registered S3 threshold
(`PREREG_D-SEL…` §6.3). It is **transcribed unchanged** — neither weakened for the harder
statistic nor tightened after the fact.

| branch | trigger (fixed now) | what it means | **what happens next** |
|---|---|---|---|
| **D-FUND** | ρ_deploy **separated from C-ctxswap** (paired) **AND separated from C-cv** (paired) **AND \|ρ_deploy\| ≥ 0.10** | the world model's consequence carries context-conditioned candidate-discriminating information the deployed path can actually reach | §6.3's *"include S3 in the retrain arm"* stands **on the deployable path**. **ρ_deploy — not 0.6657 — sizes it.** |
| **D-MARGINAL** | \|ρ_deploy\| ≥ 0.10 and separated from shuffled, but **NOT** separated from C-ctxswap **or** not from C-cv | information present but **not attributable to the world model**: the same ranking is available from scene-blind geometry or from a zero-parameter heuristic | **S3 is NOT funded as `cond_imagination`.** Report as a branch the parent prereg does not enumerate — exactly as E-SEL reported the adverse-separation gap rather than forcing it into a row. |
| **D-NULL** | ρ_deploy CI includes the shuffled control, **OR** \|ρ_deploy\| < 0.10 | REF-C's consequence is candidate-blind *along the path that ships* | §6.3's **S3 DEAD** applies: **drop S3 from the arm and say so.** ⛔ Do **not** reframe as "needs training to emerge". |
| **RED-FLAG — AUDIT** | ρ_deploy separated **better** than ρ_oracle | a strictly-fewer-inputs path beating the future-seeing path. Not impossible (the readouts differ, `s_oracle` is a fixed function and not an optimal readout) but a tripwire | **audit before publishing**; do not headline it. |

### 4.1 Sizing, also fixed now (P2)

Realized reranking on the banked fan: `rank = anchor_logits + α · zscore(s_deploy)` (per-window
z-score so α is scale-free), α over the grid **{0, ±0.05, ±0.1, ±0.2, ±0.5, ±1, ±2, ±5}**, fixed here.

* **α\*-on-test** — best α chosen on all 881 windows. **Optimistic by construction; labelled as an
  upper bound, never headlined.**
* **LOEO** — α chosen on 39 episodes, applied to the held-out one, all 40 folds. **This is the
  honest realized number.**

| branch | trigger | consequence |
|---|---|---|
| **S3 PAYS** | LOEO paired ΔADE@2s **separated** in the improving direction **AND ≥ 0.02 m** (the parent prereg's own `free_win_m`) | S3 clears the free-win bar at frozen weights |
| **S3 LOWER-BOUND ONLY** | otherwise | report the realized effect **as a LOWER BOUND on S3** — `cons_gate` is zero-init and untrained here, exactly the asymmetry §6.1 registered for S1 — and **do not upgrade it** to an effect size |

**Fraction of the SELECTION gap closed** = ΔADE ÷ (shipped − oracle-in-fan) = ÷ **0.2813** (base),
÷ **0.3075** (XL).

⚠️ **The axis test is binding.** MEASURED (E-SEL §5.1): the selection gap is **89.28 %** (base) /
**87.60 %** (XL) **LONGITUDINAL**. Even a *perfect* reranker of this fan buys **0.0334 m**
cross-track and moves heading, curvature and yaw-rate **not at all** (wrong sign, not separated).
⇒ **A consequence score that helps only laterally cannot pay for itself.** The four-family
decomposition of the realized rerank is reported per family, and if the effect is lateral-only that
is stated as a reason S3 does not pay, not buried.

⚠️ The graft seam (`refc_select.graft_with_seam`) can only **shrink** a graft relative to the base
norm, so the unclamped α-sweep is an **upper bound** on what the learned `cons_gate` can express.

### 4.2 Registered personal prediction — so I can be wrong on the record

**I predict ρ_deploy = 0.10–0.35** — positive, but far below the 0.6657 upper bound — landing most
likely in **D-MARGINAL**, and I predict **C-cv is the control that bites**.

Reasoning, stated in advance so it is falsifiable: (a) `conf_head` was trained to score
**post-attention decoder queries**, not layer-normed `law_head` outputs, so `s_deploy` is a trained
readout evaluated **off its training distribution** — the identical mechanism E-SEL measured for
`refined` (ESEL_VERDICT §3.2), which cost 0.84–0.92 m; (b) 72–74 % of the fan is outside a
bounded-acceleration band around `v0` and deleting it is **exactly inert**, so the fan's spread is
dominated by speed geometry, which `s_cv` reads directly and for free.

**If ρ_deploy comes back ≥ 0.50, my §1 framing was overstated and I will say so in the results file
rather than reframe it.** If it comes back at the null, that is D-NULL and S3 gets dropped — a
refuted lever is a result this programme funds.

⛔ **I commit now that no threshold in §4 or §4.1 moves after a result is seen.** ρ_deploy = 0.09 is
**D-NULL**, not "essentially 0.10".

---

## 5. Adverse priors carried, not hidden

1. **The oracle gap is ~92 % irreducible** and REF-C v1.2's learned re-scorer recovered at most
   **8.4 %** of it across 47 trained arms, **not separated** (+0.00893 [−0.0062, +0.0250]).
   ⚠️ Evidence class **INHERITED** — it resolves to a prose note in `MODEL_REGISTRY.md` §4.1, not to
   a results JSON. ⚠️ It is **NOT** the *other* 8.4 % in `MODEL_REGISTRY.md` §1.4b (a relative change
   in the flagship's fan under unfreezing).
2. **The C6 confound is inherited on purpose**: `nav_mode = follow_constant`, because that is the
   condition the published 0.4728 / 0.4714 and E-SEL's banked fans were collected in, and changing
   it would move the baseline this experiment is paired against. A real limitation of these rows,
   restated rather than inherited silently.
3. **R-2026-08-03-c**: `dt` must come from `four_families.infer_dt` (0.5 s on the 4-waypoint grid,
   not 0.1 s). Paired deltas survive that defect; absolute rates do not.

---

## 6. What this experiment does NOT do

* It does **not** train anything, launch any arm, or touch `tanitad-new` (v5f) or `tanitad-pod4`
  (v1arch). Cost: one deterministic forward per arm on an **idle** Jetson Thor (~27 s each, MEASURED
  from E-SEL's `wall_s`).
* It does **not** measure S3 *after training*. `cons_gate` is zero-init; `feat_proj` and `conf_head`
  do receive gradient in S3, so a trained readout may differ. **ρ_deploy is what the deployed path
  reaches at these weights** — which is the quantity that should replace 0.6657, and the α-sweep
  bound is explicitly the frozen-weight lower bound on S3.
* It does **not** re-open E-SEL-0. That verdict stands.

---

## 7. Verifiability of "fixed in advance"

The falsifiable object is this file's **git blob id at staging time**, recorded by the runner as
`prereg_s3_deployable.staged_blob` alongside the working-tree hash. If they differ, the thresholds
moved after staging and "committed in advance" is **void**. *(R11 refuted D-TAC1's mtime-based
version of this claim; an mtime is not evidence.)*

```
git ls-files -s -- "TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-03-s3-deployable/PREREG_S3_DEPLOYABLE.md"
git hash-object    "TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-03-s3-deployable/PREREG_S3_DEPLOYABLE.md"
```
