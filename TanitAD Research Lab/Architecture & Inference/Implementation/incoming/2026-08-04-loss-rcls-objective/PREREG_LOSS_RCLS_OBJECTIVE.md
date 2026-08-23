# PRE-REGISTRATION — E-OBJ-1: is `loss_rcls` itself the liability, and if so **which half of it**?

**Date:** 2026-08-04 (Europe/Berlin) · **Stream:** arch-inf (D-SEL) · **Status:** written and
**staged BEFORE any statistic below was computed** · **GPU cost registered: 0. No training
launched, no training pod touched.**

**"Fixed in advance" is made checkable the same way its antecedents make it:** by the **git blob id
of this file at staging time**. The runner re-reads it on every arm
(`prereg.thresholds_unmoved_since_staging`) and the results JSON carries both the staged blob and
the worktree blob. A mismatch means the thresholds moved and the run is inadmissible. No self-hash
is printed — a document cannot contain its own digest (R11).

**Estimator, declared before any number.** `taniteval/taniteval/ci.py::
paired_episode_cluster_bootstrap` / `episode_cluster_bootstrap`, resampling unit = **episode**,
`n_boot = 2000`, seed fixed, on the canonical **881 windows / 40 episodes** of
`physicalai-val-0c5f7dac3b11`. Two rankers on the same windows use the **paired** form.
⛔ `overlapping_holdout_se` is never called — it biases the POINT ESTIMATE bidirectionally, up to a
sign flip on paired deltas.

**Antecedents, all MEASURED, all cited not re-derived from prose:**
`…/incoming/2026-08-03-esel-verdict/ESEL_VERDICT.md` ·
`…/incoming/2026-08-03-s3-deployable/S3_DEPLOYABLE.md` ·
`…/incoming/2026-08-03-s1-climbout/S1_CLIMBOUT.md` (§4, §3.4 — **both explicitly POST-HOC**).

---

## 0. WHAT IS BEING DECIDED, AND WHY IT OUTRANKS THE LEVERS

E-S1-0 registered a listwise softmax CE against the one-hot oracle index — **exactly the objective
`refc_train.loss_rcls` optimises** (`stack/scripts/refc_train.py:408-415`,
`loss_rcls = F.cross_entropy(ce_score, r_star.detach())` with
`r_star = ce_err.argmin(dim=1)`). Under it, **every** fitted ranker came back separated **worse**
than the incumbent selector — *including feature sets that contain the incumbent's own score* — with
a C-leak gap of −0.001 to −0.003 m, i.e. **not overfitting**.

A POST-HOC probe in that same run then swapped the objective for expected-ADE under the score's own
softmax and recovered **−0.0974 m** (base) / **−0.1670 m** (XL) of a **+0.0940 / +0.1689 m** deficit,
separated, with the recovery concentrated on the **LONGITUDINAL** family (`speed_abs` −0.1102 /
−0.1816). That is a claim about the objective that **S1, S3, S5 and S6 all depend on**, and it was
made by a probe that adjudicated no branch.

> **THE QUESTION, in three parts, all registered before measuring:**
> **(Q1) Does the objective swap survive a proper pre-registration** — three-sided, with direction
> predicates, on both arms, and with a control that can void it?
> **(Q2) WHICH HALF of `loss_rcls` is the liability** — the **CE form**, or the **one-hot target**?
> They are different one-line changes with different gradient consequences, and only one of them is
> the right recommendation.
> **(Q3) Does it COMPOSE with the timestep-token finding** (`emitted_t0`), or are the two findings
> the same effect measured twice?

⚠️ **Q2 is the part with no antecedent at all.** Nothing in the programme has separated
"CE is the wrong shape of loss" from "one winner among ~128 near-duplicates is the wrong target".
The distinction decides which line the PI is asked to change:

| candidate change to `loss_rcls` | params | keeps the gradient path `cons_gate` / `route_to_anchor` depend on (`refc_train.py:377-387`)? |
|---|---|---|
| replace one-hot target with `softmax(−fan_err/τ)` (**target shape**) | **0** | ✅ yes — still a CE on `sel_score` |
| replace CE with `E_{softmax(s)}[fan_err]` (**objective form**) | **0** | ✅ yes — still differentiates `sel_score` |
| leave it and add a head | > 0 | — ⛔ excluded, C34 forbids a capacity change before attribution |

---

## 1. ⛔ WHAT THIS EXPERIMENT CANNOT DECIDE — stated BEFORE measuring, not after

This is the same firewall E-S1-0 §2 put up, and it is put up for the same reason: so a result here
cannot later be mis-reported as something stronger.

1. **It cannot show that TRAINING under a different objective helps.** It measures an objective's
   effect on **COMBINING frozen banked scores** with 1–7 parameters. A retrain optimises 104 M
   parameters and can change what the score *reads*, not only how readouts are mixed.
   ⇒ **A win here is NECESSARY-NOT-SUFFICIENT for funding a retrain. A null here does NOT show a
   retrain would fail.** Both directions registered.
2. **It is structurally blind on a single feature.** `argmax_i (w·φ_i)` is invariant to any positive
   rescaling of `w`, so for a 1-D feature set **every objective has the identical argmax**. The
   degenerate rows therefore cannot move, and are used as controls rather than treatments (§4).
   ⚠️ In training, `sel_score` is also a 1-D score — but it is produced by 104 M parameters, so the
   analogy is imperfect. **This limitation is registered, and any conclusion must be phrased as a
   statement about the objective's effect on a LEARNABLE score, bounded by a linear probe.**
3. **It does not re-open S1, S3, S5 or S6.** `S1_IS_DEAD_AFTER_ALL` fired; S3's deployable ρ
   collapsed and flipped sign on XL. Those verdicts **stand** and are not amended by this document.
   Re-measuring any of them is a NEW pre-registration.
4. **It changes no baseline.** The C6 confound (`nav_mode='follow_constant'`) is inherited on
   purpose — it is the condition the published 0.4728 / 0.4714 were collected in.

---

## 2. THE DATA — banked, 0 GPU, 0 inference

| bank | carries | source run |
|---|---|---|
| `…/2026-08-03-esel-verdict/raw/fan_refined_refc-{base,xl}-30k.pt` | `fan, logits, refined_logits, cons_score, sel, gt, cv, eid, v0` | E-SEL |
| `…/2026-08-03-s1-climbout/raw/fan_emitted_t0_refc-{base,xl}-30k.pt` | `+ emitted_logits, emitted_t0_logits, prefinal_logits` | E-S1 |

⛔ `cons_score` / `cons_oracle` are **excluded from every feature set**: they read `z_{t+5}`, the
future frame. Including them is the C6 / REF-A-I-JEPA class — a measurement-time input the deployed
path does not have.

**Bit-identity of the two banks is asserted before anything is fit** (`fan`, `gt`, `eid`, `v0`,
`logits`), and `prefinal_logits == refined_logits` — the same control E-S1 ran. A changed fan would
silently re-baseline every D-SEL number.

---

## 3. THE PROTOCOL — held FIXED across objectives, so the objective is the only difference

* **Support:** the **S2-reachable survivor set**, via the re-export of `flagship_v15.reachability_mask`
  through `refc_select` (the same function object the 73.76 % / 72.08 % measurement used).
  A window with no survivor keeps its whole fan (the decoder's own empty-set fallback).
* **Features:** per-window z-scored **over the survivors only**. For a single feature this is a
  strictly-monotone within-window transform ⇒ the degenerate controls still reproduce their
  incumbents **exactly**, which is what makes C-monotone a control that can fire.
* **Fit:** `s_i = w·φ_i`, no bias (a per-window constant cancels in a softmax over candidates), Adam,
  `iters = 400`, `lr = 0.25`, `w init = 0`, float64. **Identical for all three objectives.**
* **Eval:** **LEAVE-ONE-EPISODE-OUT**, folds asserted disjoint per fold. No window is scored by a
  model that saw its own episode.
* **Headline read:** **selection ADE@2s** of `argmax_i s_i` over the survivors, **paired** against
  the named comparator on the same windows.
* ⛔ **ρ is a SECONDARY diagnostic, computed on the reachable subset only, and NEVER reported without
  the selection ADE beside it.** S3_DEPLOYABLE §3 measured that ρ over the full candidate axis is
  disconnected from selection for *every* score (ρ = 0.6657 selects at **6.49 m**; ρ = 0.9951
  selects at **0.8149**), and **73.76 % / 72.08 % of the fan is unreachable and measured never
  selected**, so a rank statistic over the whole fan is dominated by candidates that can never win.

### 3.1 The three objectives

| id | objective | what it is |
|---|---|---|
| **O-ce** | `CE(s, argmin_{survivors} fan_err)` | ⚠️ **INCUMBENT — bit-for-bit the shape of `refc_train.loss_rcls`.** Reproduces E-S1-0's registered rows, which is itself a control (§4 C-reproduce) |
| **O-softade** | `E_{i∼softmax(s)}[fan_err_i]` over survivors | the escalation's candidate: optimise the quantity actually wanted |
| ⭐ **O-softce(τ)** | `CE(s, softmax(−fan_err/τ))` over survivors | **NEW — the mechanism discriminator.** Keeps the CE *form*, replaces only the *target shape* |

**τ is registered here and is not tuned.** Headline **τ = 0.25 m**, chosen before measuring because
it is the scale of the published selection gap (`sel_gap` = **0.2813 m** base / oracle-in-fan
**0.1914 m**) — i.e. "candidates about as good as the winner". A **registered sensitivity strip**
τ ∈ {0.01, 0.05, 0.10, 0.25, 0.50, 1.00} is printed in full, and its role is declared now:
⛔ **if the three-sided branch of the Q2 contrast is not the same at τ = 0.10, 0.25 and 0.50, the
conclusion is τ-DEPENDENT and must be reported as such rather than as a finding.**

### 3.2 The feature sets

| id | φ | role |
|---|---|---|
| **A-shipped** | `logits` | ⚠️ DEGENERATE control — 1-D ⇒ identical under all three objectives, must reproduce **0.4728 / 0.4714** |
| **A-refined** | `refined_logits` | ⚠️ DEGENERATE control — must reproduce **1.3100 / 1.3861** |
| **A-t0** | `emitted_t0_logits` | ⚠️ DEGENERATE control — must reproduce E-S1 §3.4's **0.6253 / 0.5194** |
| **E-cv** | `−‖cand − cv‖` | ⚠️ DEGENERATE control that has already fired twice — **0.8149 / 0.8158** |
| **B-both** | `logits, refined_logits` | ⭐ **the escalation's headline** — the row where soft-ADE recovered −0.0974 / −0.1670 |
| **C-lon** | `along_end_m, dv_mps, abs_dv_mps, outside_band_mps` | the longitudinal-geometry lever; **registered as a clean null** by E-S1 (+0.0008 / +0.0030 under the swap, not separated) — a replication target |
| **D-lon+scores** | `B-both ∪ C-lon` | the second row where the swap separated better |
| ⭐ **F-t0** | `logits, emitted_t0_logits` | **Q3 — the composition test.** Token finding × objective finding |
| ⭐ **G-all-t0** | `D-lon+scores ∪ emitted_t0_logits` | Q3, widest deployable set |

Every feature is **deployable at inference**: computed from the emitted fan, `v0`, and readouts of
the decoder's own `conf_head`. `emitted_t0_logits` is `conf(X₂, t=0)` — one extra conf-only pass,
**0 parameters**, offset discarded.

---

## 4. THE CONTROLS — every one must be able to fire, and each carries a DIRECTION predicate

⛔ **`C-shuffled` is NOT used.** Permuting a score and then taking argmax is a uniform random pick
for *any* score, so `E[ρ] = 0` and `E[ADE]` = the uniform floor **regardless of the score**. It is
declared vacuous in code (`can_fire: False`), reported, and **never load-bearing.**

| control | predicate — **direction, not just separation** | can it fire? |
|---|---|---|
| **C-reproduce** | the bank's `fan/gt/eid/logits` are bit-identical across the two banks; `argmax(logits) == sel` on 1.0000; `prefinal_logits == refined_logits` | ✅ |
| **C-monotone** | all four DEGENERATE rows reproduce their incumbents to ≤ 1e-6 **under all three objectives**, and their argmax is bit-identical across objectives | ✅ — a wrong LOEO split, a wrong survivor mask or a wrong argmax breaks this |
| ⭐ **C-optimiser** ⚠️ **THE ONE THAT CAN VOID THE HEADLINE** | the cross-objective loss matrix, in-sample: `L_ce(w_ce) < L_ce(w_softade)` **AND** `L_softade(w_softade) < L_softade(w_ce)`. ⛔ **If the CE fit is beaten under its OWN objective, the difference is an OPTIMISER artifact and the finding is VOID, not a result.** | ✅ — and it is the first hypothesis if the effect is large |
| ⭐ **C-continuity** | `O-softce(τ = 0.01)` must land within **1e-3 m** of `O-ce`, because `softmax(−e/τ) → onehot(argmin e)` as `τ → 0` | ✅ — a τ knob that does not converge to the incumbent target is not the knob I claim it is |
| ⭐ **C-scale-invariance** | `argmax(w·φ) == argmax(cw·φ)` for `c = 100`, every arm; and `cos(w_ce, w_softade)` reported | ✅ — catches a non-linear or mis-stacked score |
| **C-permuted-features** | features permuted independently per window, **`keep` NOT permuted**, refit LOEO under each objective; must be **NOT separated** from the survivor-restricted uniform floor. ⚠️ The mask-permuting version of this control **failed spuriously once** (E-S1 §3.5) by comparing against the full-fan floor — the fixed version is used | ✅ |
| **C-leak** | in-sample vs LOEO, per objective. A gap ≥ 50 % of any claimed effect voids that effect | ✅ |
| **C-incumbent** | ⭐ every treatment is scored against the **shipped selector**, never against a shuffled or random comparator. This is the control S3 registered against a shuffled comparator and thereby missed | ✅ |
| ~~C-shuffled~~ | ⛔ VACUOUS BY CONSTRUCTION | ❌ **no** — named, never counted |

---

## 5. ⛔ BOTH OUTCOMES COMMITTED IN ADVANCE — three-sided on every two-sided quantity

**Separated** = the paired episode-cluster bootstrap 95 % CI excludes 0.
**Material** = |Δ ADE@2s| ≥ **0.02 m** (the parent's `free_win_m`, inherited, not re-invented).
**Every branch reads BOTH bits** — separated **and** the sign of the delta — via a `_sep_dir()`
predicate applied mechanically in `adjudicate()`. No row is chosen by hand.

**Multiplicity is controlled by naming the primaries now.** There are three PRIMARY quantities, all
on **`refc-base-30k`**; **`refc-xl-30k` is a pre-registered REPLICATION** and a conclusion requires
**the same three-sided branch on both arms**. Everything else in this document is SECONDARY and is
labelled SECONDARY in the results JSON.

### 5.1 PRIMARY-1 (Q1) — does the objective swap survive? `B-both`, `O-softade − O-ce`, paired

| branch | trigger | meaning | next |
|---|---|---|---|
| **OBJ-LIABILITY-CONFIRMED** | separated **AND ≤ −0.02 m** (better) **on BOTH arms** **AND** C-optimiser passes **AND** C-leak gap < 50 % of the effect | the registered CE objective was discarding information the same features carry. `loss_rcls`'s shape is a measured liability for combining readouts | escalate a **pre-registered training arm**; ⛔ do not patch |
| **OBJ-MARGINAL** | separated and better but **> −0.02 m**, or better on only ONE arm | real but not worth a GPU-day alone | fold into an arm already running; do not fund alone |
| **OBJ-NOT-SEPARATED** | CI includes 0 | the POST-HOC −0.0974 / −0.1670 does not survive pre-registration | ⛔ **report the post-hoc claim as NOT REPLICATED**, and say so in the escalation that raised it |
| **OBJ-ADVERSE** ⚠️ | separated **AND worse** | the swap actively harms | report it; the CE stands |
| **VOID** ⚠️ | C-optimiser fails | an optimiser artifact, not an objective finding | ⛔ **not a result in either direction** |

### 5.2 PRIMARY-2 (deployment) — does ANY objective/feature combination beat the SHIPPED selector?

Read on the best arm by point estimate among {B-both, D-lon+scores, F-t0, G-all-t0} × {O-softade,
O-softce(0.25)}, paired vs shipped, on both arms.

| branch | trigger | meaning | next |
|---|---|---|---|
| **DEPLOYABLE-WIN** | separated **AND ≤ −0.02 m** better than shipped **on BOTH arms** | a free, 0-parameter reranker beats the incumbent selector out-of-episode | ⭐ this outranks every D-SEL lever — but see the RED FLAG below |
| **DEPLOYABLE-MARGINAL** | separated and better, **> −0.02 m**, or on one arm only | too small to ship on its own | report; do not fund |
| **DEPLOYABLE-NULL** | CI includes 0 | ⭐ **the expected outcome.** The objective swap repairs a *self-inflicted* deficit and returns the fitted ranker to parity with the incumbent — **it does not create a win.** This is the honest reading of the post-hoc −0.0034 / +0.0019 | ⛔ **report as "no free win"; the levers stay dead** |
| **DEPLOYABLE-ADVERSE** ⚠️ | separated worse | as today | report |

⚠️ **RED FLAG — stop and audit, do not publish:** any arm separated better than shipped by
**> 0.10 m**. That is ~1/3 of the entire selection gap from a LOEO linear model on ≤ 7 features, and
the first hypothesis must be a **protocol leak** (a non-disjoint episode split, or a feature computed
from `gt` / `z_{t+5}`), **not a win**.

### 5.3 PRIMARY-3 (Q2) — WHICH HALF of `loss_rcls`? `B-both`, `O-softce(0.25)` vs the two poles

Two paired contrasts, each three-sided, adjudicated **jointly** into one attribution:

| attribution | trigger | what the PI is asked to change |
|---|---|---|
| **TARGET-SHAPE** | `O-softce − O-ce` separated better **AND** `O-softce − O-softade` **NOT separated** | ⭐ **the one-hot target is the liability**, not the CE. One line: `r_star` → `softmax(−fan_err/τ)`, 0 params, CE form and its gradient path preserved |
| **OBJECTIVE-FORM** | `O-softce − O-ce` **NOT separated** **AND** `O-softade − O-ce` separated better | the CE *form* is the liability; the soft target does not rescue it | one line: swap the CE for expected-ADE, 0 params |
| **BOTH-PARTIAL** | `O-softce − O-ce` separated better **AND** `O-softce − O-softade` separated **worse** | the soft target recovers *part*; the rest is the CE form | report both magnitudes; prefer the target change (cheaper, keeps the gradient path) unless the residual is material |
| **NEITHER** | both contrasts not separated | the Q2 question has no answer on this probe | report as such; **do not** pick a change |
| **τ-DEPENDENT** ⚠️ | the branch differs at τ = 0.10 / 0.25 / 0.50 | the attribution is an artefact of a tuning knob | ⛔ report the strip, claim nothing |

### 5.4 SECONDARY (Q3) — does the token finding COMPOSE with the objective finding?

`F-t0` and `G-all-t0`, paired vs shipped and vs the same features under `O-ce`. Three-sided.
Registered readings: **COMPOSES** (F-t0 under a swapped objective is separated better than both
B-both-under-swap and A-t0) · **REDUNDANT** (not separated from B-both-under-swap) · **INTERFERES**
(separated worse than either). ⚠️ `A-t0` is the DEGENERATE control that bounds this row from below.

### 5.5 SECONDARY — `C-lon` replication

E-S1 MEASURED `C-lon` as **unmoved** by the objective (+0.0008 / +0.0030, not separated) and read it
as a clean null: purely longitudinal candidate *geometry* carries no selection signal on this fan.
Registered branches: **NULL-REPLICATES** / **NULL-FAILS-TO-REPLICATE** / **ADVERSE**.

---

## 6. THE FOUR METRIC FAMILIES — per family, never pooled (binding)

Grid **derived, never assumed**: `wp_steps [5,10,15,20] × 0.1 s → dt = 0.5 s`
(`four_families.infer_dt`). A hard-coded 0.1 s inflates speed ×5 and accel ×25 (R-2026-08-03-c).
⛔ **An ADE horizon sweep is ONE ROW OF FIVE**, never "the result".

| family | metric | pre-committed direction |
|---|---|---|
| **ADE** | `ade_0_2s` | headline, **never alone** |
| **LONGITUDINAL** | `speed_abs_err_mps`, `speed_signed_err_mps`, `along_abs_err_m`, `along_signed_err_m` — **signed AND absolute together**, because every fitted ranker so far bought a *bias* reduction at the cost of *absolute* error and reporting the signed row alone would read as a gain | ⭐ **improve — this is the family the lever must move (87.6–89.9 % of the selection gap)** |
| **LONGITUDINAL — distance-keeping** | headway / time-gap / TTC via `taniteval/lead_source.py::lead_block` + `lead_metrics.distance_keeping`, over the **~270 of 881** windows in state `LEAD` | improve; ⛔ **NO_LABEL is never counted as free flow** — three states, reported with their own denominators |
| **LATERAL** | `cross_abs_err_m`, `heading_abs_err_deg`, `curvature_abs_err_1pm`, `yaw_rate_abs_err_degps` | ⛔ **GUARD-RAIL: must be NOT separated worse.** Tight by construction and MEASURED: a *perfect* reranker of this fan buys **0.033 m** of cross-track and moves heading/curvature/yaw **not at all** |
| **TACTICAL** | `rank_acc`, `sel_gap`, `frac_sel_2x_worse` (goal/anchor selection) | improve |
| **TACTICAL — manoeuvre confusion** | — | reported **UNAVAILABLE with reason and n** if a fan bank stores no decoded manoeuvre logits |
| **STRATEGIC** | route/goal quality with its majority-class baseline beside it | reported **UNAVAILABLE with reason and n** if no route label is in the bank |

Where a family cannot be computed it is reported **per family, with the reason and the n** — never
silently dropped, and never as a pass.

---

## 7. IF a training arm is warranted — the CHEAPEST DISCRIMINATING ARM, not launched here

Identical data, parity key `physicalai-train-e438721ae894`, skip-hash `f09e44db`, identical
optimizer / schedule / seed. **The ONLY difference is the named flag.** ⛔ **Not launched. A GPU-day
is the PI's call.**

| arm | change | isolates | params |
|---|---|---|---|
| `refc-base-30k` | — (the published control) | — | 0 |
| **`obj-softce`** | `loss_rcls` target `r_star` → `softmax(−fan_err/τ)`, τ = 0.25 | **the TARGET SHAPE** | **0** |
| `obj-softade` | `loss_rcls` → `E_{softmax(sel_score)}[fan_err]` | **the OBJECTIVE FORM** | **0** |

**Cheapest sufficient arm:** whichever of the two §5.3 attributes the recovery to, **vs the published
control**, 30 k steps, one A40-class GPU. If §5.3 returns **NEITHER** or **τ-DEPENDENT**, ⛔ **no arm
is recommended.**

**Success:** LONGITUDINAL (`speed_abs`, `along_abs`) separated **better** **AND** LATERAL not
separated worse **AND** ADE not separated worse **AND** `frac_sel_2x_worse` separated below control.
**Failure:** LATERAL separated worse (guard-rail), **or** LONGITUDINAL CI includes 0.

---

## 8. REGISTERED PERSONAL PREDICTION — pinned, and scored in the results file

* **PRIMARY-1: `OBJ-LIABILITY-CONFIRMED`,** ~80 % confidence. The post-hoc effect is large
  (−0.0974 / −0.1670), replicated on two scales and two anchor counts, and it has a clean mechanism:
  a one-hot CE over ~128 near-duplicate candidates spends its gradient distinguishing trajectories
  that differ by centimetres. What I am *least* sure of is C-optimiser.
* **PRIMARY-2: `DEPLOYABLE-NULL`,** ~85 % confidence. I expect the swap to **repair a self-inflicted
  deficit back to parity and stop there.** The post-hoc numbers (−0.0034 / +0.0019, neither
  separated) say exactly that, and S1 is dead on the deployment question either way.
* **PRIMARY-3: `TARGET-SHAPE`,** ~55 % confidence — genuinely uncertain, and it is the row I most
  want the measurement for. Reason to expect it: soft-ADE and a soft-target CE differ only in how
  they weight the tail, and the near-duplicate structure of the fan is a *target* problem.
  Reason it might be `OBJECTIVE-FORM`: expected-ADE is metric-aware in a way no CE is, and the
  measured recovery was LONGITUDINAL — a *metric* axis, not a *label* axis.
* **Q3: `REDUNDANT`,** ~60 %. `emitted_t0` and `logits` are two readouts of the same head at the same
  token; I expect a linear blend of them to add little to `B-both`.
* **C-lon: `NULL-REPLICATES`,** ~85 %.

⚠️ **E-S1-0's author got the conclusion right and the DIRECTION of every component wrong — three for
three — and wrote that the lesson is "I under-predicted how often a lever is actively harmful rather
than merely useless."** That correction is inherited here: every table above has an `ADVERSE` row,
and if a row comes back adverse it is reported as adverse, not reframed.

---

## 9. WHAT THIS IS NOT

* **Not a test of a retrain** (§1.1, registered in advance).
* **Not a revival of S1, S3, S5 or S6** (§1.3). Their verdicts stand.
* **Not a capacity change.** Every candidate change is **0 parameters** — stated against the accepted
  ledger (+897 / +385 / +128 / 0) and against the one lever that cost **+272,001** before its own
  control caught it.
* **Not a headroom claim.** *"The oracle gap is ~92 % irreducible"* is **INHERITED** from a prose
  note in `MODEL_REGISTRY.md` §4.1, not a results JSON, and is flagged load-bearing wherever used.
* **Not launched.** No training started, no training pod touched, `OMP_NUM_THREADS=6` set before any
  multi-arm panel.
