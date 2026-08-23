# PRE-REGISTRATION — does a PER-SITUATION `(window, lead_s)` beat the single frozen setting?

**Date** 2026-08-03 · **Stream** sitclf per-situation horizon · **Substrate** dev box (RTX 4060 /
CPU), **0 pod GPU-h** — `tanitad-new` (v5f) and `tanitad-pod4` (v1arch) are training and are not
touched.
**Written and staged BEFORE any held-out number from this run was read.**

---

## 0. Why a new pre-registration is required, and exactly what it unfreezes

`…/2026-07-26-situation-classifier/PRE_REGISTRATION.md` §2.4 fixes `lead_s = 3.0`, and its §7 is
binding: *"no re-sweep of any §2 constant after a held-out number is seen … **Any follow-up is a new
pre-registration.**"* The sibling temporal study honoured that by **characterising** the horizon
without selecting one (`…/2026-08-03-sitclf-temporal/PRE_REGISTRATION.md` §5b).

That characterisation produced a finding nobody was looking for, and it is the reason this document
exists (MEASURED, `…/2026-08-03-sitclf-temporal/results_horizon.json`):

| situation | skill vs the permuted-feature null, by anticipation horizon |
|---|---|
| `intersection` | **DECAYS monotonically**: +0.982 (1 s) → +0.378 (5 s); precision-lift 2.713 → 1.794 |
| `lane_change` | **RISES**: separates **only** at 5 s (+0.436), null or negative at 1 s and 3 s |

Two differently-constructed metrics agree on the `intersection` decay, which is what rules out the
base-rate artefact. **The programme forces one window and one horizon onto two phenomena with
opposite timescales.**

⭐ **This document unfreezes exactly one thing and nothing else:** whether `(window, lead_s)` may be
chosen **per situation** rather than shared. It does **not** touch a single detector, a single
detector threshold, `MIN_USEFUL_LEAD_S`, the parity cache, or the deployed default — `lead_s = 3.0`
and `WIN = 8` remain the deployed setting unless *this* study's pre-registered predicate fires.

⛔ **Nothing here re-selects episodes.** Same 500-clip B4 substrate, same clip order, same folds.

---

## 1. The two questions, because they are genuinely different

A per-situation `lead_s` changes **what is being predicted**, so a naive "AP at lead 1 s beats AP at
lead 3 s" compares two different tasks and is inadmissible. The finding decomposes into two
questions with two different — and both legitimate — evaluation protocols:

| id | question | what moves | what is scored |
|---|---|---|---|
| **Q-A · TRAIN-HORIZON** | does tuning `(W, L)` **per situation** improve the **deployed** 3 s anticipation? | `L` is only the head's **training** target; the evaluation label stays the frozen `lead_s = 3.0` | frame-level AP-lift + precision/recall at a 5 % alarm budget |
| **Q-B · DEPLOY-HORIZON** | if each situation may **declare its own warning horizon**, does the system warn more onsets, earlier, at the **same alarm budget**? | `L` is the training target **and** the declared horizon | ⭐ a **horizon-agnostic EVENT-level** yardstick (§3), identical in definition for every arm |

Q-A touches no frozen constant at all. Q-B is the one that acts on the finding, and it is why this
pre-registration is needed.

⭐ **Both questions are answered from the SAME score columns.** One fit pass produces a score column
per `(situation, W, L)` cell; Q-A and Q-B differ only in the metric applied to it. There is no arm
that exists for one question and not the other.

---

## 2. Substrate, rows, and the constants that do not move

* **Substrate** `C:/Users/Admin/tanitad-data/eval/sitclf_b4_substrate.npz` — 99,477 × 2048 fp16
  frozen-v1 latents over 500 clips, built by
  `…/2026-08-03-sitclf-matched-capacity/build_substrate.py`. **NOT the parity cache**; absolute APs
  are not comparable to the banked table and every claim here is a **within-substrate paired
  contrast on identical rows**.
* **Representation** frozen at the reference recipe: appearance PCA **rank 16**, fitted per fold on
  that fold's FIT rows only, **never on the labels** — so one basis serves every `(W, L)` cell and
  the only things moving are the window and the lead.
* **Folds** 2 outer folds over whole clip CLUSTERS (`sitclf.cluster_folds`, seed 0); 20 % of each
  train half's clusters is the inner **SEL** split used for **all** hyper-parameter choice.
* ⭐ **ONE evaluation row set for every arm and both questions:**
  `EVAL_ROWS(s) = valid(lead = 5.0)[:, s] ∧ hist_ok(WIN_MAX = 32)`.
  `valid(5.0) ⊆ valid(3.0)` (the ongoing-event mask is identical; only the end-of-clip tail differs),
  so the deployed 3 s label is well defined on exactly these rows. Scoring a `W = 32` arm on rows a
  `W = 1` arm cannot reach would confound the window with the row set and invalidate the paired
  estimator.
* ⭐ **ONE training row rule for every arm:**
  `TRAIN_ROWS(f, L, s) = FIT_f ∧ hist_ok(32) ∧ valid(L)[:, s]`. `hist_ok` is held at **32** for every
  `W`, so a short-window arm does not silently get more training rows than a long-window one. Only
  the label's lead moves the training positives.
* **Estimator** paired **episode-cluster bootstrap** over clip clusters (`tanitad.eval.ap_ci`,
  which delegates its draws to `taniteval.ci._draws`), `B = 2000`, `seed 0`; `separated` ⇔ the 95 %
  percentile interval excludes 0. ⛔ **`overlapping_holdout_se` is never used** — it biases the point
  estimate (mean-of-split-means, bidirectional, up to a sign flip on paired deltas).
* **⛔ Frozen, untouched:** every detector, every detector threshold, `MIN_USEFUL_LEAD_S = 1.0`, the
  `intersection` `cross=None` convention, the clip set, the clip order, the fold seed.

### 2.1 The grid

`W ∈ {1, 4, 8, 16, 32}` (0.0 s … 3.1 s of latent history) × `L ∈ {1.0, 2.0, 3.0, 4.0, 5.0}` s
= **25 cells per situation**. The frozen setting `(W = 8, L = 3.0)` is a cell in the grid.

**Ridge `λ ∈ {1, 10, 100, 1000, 10000}` is selected per situation on the SEL split for EVERY arm,
frozen included** — so λ is not a moving part and cannot be the source of a per-situation gain.

---

## 3. ⭐ The EVENT-LEVEL yardstick (Q-B) — defined before it is computed

A declared horizon changes the label, so Q-B needs a metric that does not depend on it. Onsets do
not depend on it: an onset is an onset whatever lead the training label used.

* **Onset universe.** Every detected onset `a` of situation `s` that has at least one row of
  `EVAL_ROWS(s)` in `[a − 50, a)` (5 s at 10 Hz). `n_onsets` is **FIXED across arms** and is the
  recall denominator.
* **Alarm budget.** The top **5 %** of `EVAL_ROWS(s)` **by rank** (`sitclf_deploy._top_frac_alarm`,
  rank not threshold, because ties otherwise turn a 5 % operating point into a 100 % one).
  `n_alarm` is **identical for every arm by construction** and is the precision denominator.
* **`warned(a)`** ⇔ some alarm row `t ∈ [a − 50, a)`.
* **`event_recall` = |warned| / n_onsets**.
* **`median_lead_s`** = median over **warned** onsets of `(a − t_first_alarm)/10`, bounded by 5 s. An
  onset with no alarm contributes **no lead** and is counted in `n_onsets_no_alarm` — never scored as
  0 s, which would reward an arm that never fires with perfect punctuality.
* **`alarm_precision_H`** for `H ∈ {5.0, 3.0}` s = fraction of alarm rows `t` with an onset in
  `(t, t + 10H]`. Reported at both the yardstick horizon and the **deployed** one.
* **Bootstrap.** The alarm mask is computed **once on the full row set** and then held fixed; the
  cluster bootstrap resamples clusters and recomputes each statistic over the resampled onsets /
  alarm rows. That evaluates a **fixed decision rule** rather than re-deriving a threshold inside
  every draw, which would confound threshold instability with arm quality.

⚠️ **Registered in advance as the honest reading:** a longer training lead trains the head to fire
earlier, so it may raise `event_recall` for *both* situations. **If both powered situations select
the same `L`, the per-situation claim FAILS on Q-B** — the finding would then characterise the task
without yielding a per-situation configuration lever. That outcome is committed here, in advance, as
a legitimate result.

---

## 4. Arms and controls — every control able to FAIL

| id | what it is | how it can FAIL |
|---|---|---|
| **FROZEN** | `(W = 8, L = 3.0)` for all three situations, λ per situation on SEL | the baseline; it is what everything is paired against |
| **PS-SEL** | ⭐ the arm under test: **per situation**, pick `(W_s, L_s)` **out-of-fit** on that fold's SEL split, then score the held-out fold | it is a *procedure*, not a cherry-pick; it fails by not beating FROZEN |
| **C-GLOBAL** | the identical selection procedure constrained to ONE `(W, L)` **shared** by all three situations | ⭐ **fires if `PS-SEL ≈ C-GLOBAL`** ⇒ any gain is from *selection*, not from *per-situation-ness* |
| **C-SEL-NULL** | the identical per-situation selection procedure run on the **clip-permuted** feature substrate (the parent's `NEG_FEAT` permutation, within-clip temporal order preserved), against a FROZEN arm fitted on the same permuted features | ⭐ **fires if the procedure "gains" where there is no clip-specific signal** ⇒ the gain is a selection artefact and the table is VOID. Not vacuous: the argmax runs over real, noisy SEL-fold APs on features that carry real motion — another drive's |
| **C-ORACLE-PS** ⛔ not deployable | `(W_s, L_s)` chosen on the **TEST** fold, i.e. cheating, then reporting that same test-fold number — the **upper bound** on any per-situation gain | ⭐ **the decisive control.** If even the oracle does not separate above FROZEN, there is **no per-situation gain to find** and the null is MECHANISTIC, not a selection-power problem |
| **C-POW** | positive **CLUSTERS** per situation (Q-A) and **onset-bearing clusters** (Q-B), counted and written to disk **before any score is read**; bar = **40** | `< 40` ⇒ `UNDERPOWERED_C_POW`, **no verdict**, reported with its `n` |
| **C-FID-PARENT** | one cell re-run under the parent `run_horizon.py` row rule and λ rule, asserted against `results_horizon.json` at `lead 3.0` | a mismatch means this pipeline is not the one that produced the banked table and nothing here may be quoted |
| **C-FID-RIDGE** | the λ-swept ridge used here asserted **bit-identical** to `stack`'s `sitclf.ridge_scores` at one cell per window | a mismatch means the fast path is not the banked estimator |

⚠️ **Lesson carried from the E-SEL stream:** its `C-shuffled` leg was **vacuous by construction** —
permuting a score and then taking an argmax is a uniform random pick for *any* score, so the control
could not fire. Every control above is checked against that failure mode: each one has a state of
the world in which it fires and a state in which it does not.

⛔ **`roundabout` is expected to be `UNDERPOWERED_C_POW`** (37–39 positive clusters against a bar of
40 at B4). It is reported **with its n** and decides nothing. ⛔ **The bar is not lowered.**

---

## 5. ⭐ OUTCOMES — all committed in advance

Evaluated **per question** and **per situation**, never pooled.

| verdict | predicate | what the programme should then do |
|---|---|---|
| **PER_SITUATION_WINS** | on a POWERED situation, `PS-SEL − FROZEN` has a paired 95 % CI **excluding zero** and delta **> 0**, **and** C-SEL-NULL does not show the same, **and** `PS-SEL − C-GLOBAL` also separates upward | build the per-situation head. §7 states the exact build and its parameter cost in advance |
| **GAIN_IS_SELECTION_NOT_PER_SITUATION** | `PS-SEL` separates above FROZEN but `PS-SEL − C-GLOBAL` does **not** | re-tune the **shared** `(W, L)` once; do **not** build per-situation machinery |
| **NO_PER_SITUATION_GAIN_EXISTS** | `PS-SEL` does not separate above FROZEN **and** **C-ORACLE-PS** does not either | ⭐ the strongest null available: the ceiling itself is flat. Stop; the horizon finding is a **characterisation of the task**, not a configuration lever. ⛔ Do **not** build the trainer |
| **NO_EFFECT_ABOVE_MDE** | `PS-SEL` does not separate but **C-ORACLE-PS** does | a gain exists but is **not selectable out-of-fit** at this n ⇒ the work item is power (more clips), not architecture |
| **SELECTION_ARTEFACT** | C-SEL-NULL shows a positive separated gain | the table is VOID; the instrument is the finding |
| **UNDERPOWERED_C_POW** | `< 40` positive / onset-bearing clusters | no verdict for that situation; more clips, not more model |

⚠️ **Null language is constrained** (parent §7): a `separated: false` is reported as **no effect
above this study's MDE**, with the MDE stated as the widest paired-vs-FROZEN CI half-width actually
achieved — never as an unbounded refutation.

⛔ **No threshold in this document may be weakened after a number is seen, and no outcome may be
re-scoped.** A refuted lever is a result this programme funds.

---

## 6. ⭐ MY REGISTERED PREDICTION — so the study can embarrass me

**I predict `NO_PER_SITUATION_GAIN_EXISTS` on `intersection`, and I put ~65 % on it.**

Reasoning, stated now so it cannot be retrofitted: the parent measured `ridge_app16_w1` — **one
latent, 17 parameters, zero history** — as statistically indistinguishable from the deployed 0.7 s
window on `intersection` (+0.009 [−0.049, +0.054], an interval ~5× tighter than that study's MDE).
If eight latents buy nothing over one, a per-situation *window* has almost nothing left to trade,
and the horizon decay looks to me like a property of **the label's difficulty**, not of a
recoverable configuration.

The ~35 % I hold back is on `lane_change`: its skill *rises* with the lead and it is the situation
where a longer horizon has a mechanism (a 4 s ego manoeuvre whose precursors build up over seconds).

**On Q-B I predict both powered situations select `L = 5.0`** — the longest lead tested — because a
head trained to fire earlier wins a budget-matched event-recall race regardless of the task's
signature. If that happens, the per-situation claim fails on Q-B **by my own §3 rule**, and I will
report it as such rather than re-reading the metric.

---

## 7. ⭐ WHAT A POSITIVE RESULT WOULD COST — registered in advance, so the price cannot move

The programme's capacity discipline is strict: an input lever once cost **+272,001 params** before
its own control caught it; the accepted ones were **+897**, **+385**, **+128**.

The deployed floor is a per-situation ridge at `16·W + 1` parameters per head; at `W = 8` that is
**129/head, 387 total** over three situations. A per-situation window changes only the flat dim:

| configuration | params/head (lane_change / roundabout / intersection) | total | Δ vs frozen 387 |
|---|---|---:|---:|
| FROZEN `W = 8` everywhere | 129 / 129 / 129 | **387** | — |
| all-`W = 1` | 17 / 17 / 17 | 51 | **−336** |
| `lane_change` `W = 32`, others `W = 8` | 513 / 129 / 129 | 771 | **+384** |
| all-`W = 32` | 513 / 513 / 513 | 1539 | **+1152** |

⚠️ **A per-situation `lead_s` costs ZERO parameters** — it is a label definition, not a weight. Its
cost is entirely in the **pipeline**: three label sets instead of one, three training targets, and a
downstream contract change (the tactical layer must be told which horizon each channel carries).
§7 of the report states that build concretely against whichever outcome fires.

---

## 8. What would make me wrong in a way I could not detect

* The frozen v1 trunk is the shared ceiling of every arm here. If it saturates the usable evidence,
  this study measures **the trunk**, not the horizon. Only a video-pretrained trunk (BACKLOG **B5**)
  separates those.
* 500 local clips, 2 folds. `lane_change` sits at ~55–64 positive clusters and `roundabout` below the
  bar; a per-situation effect smaller than this study's MDE is invisible here and is reported as
  such rather than as absence.
* The event-level yardstick is **new**. It is defined in §3 before it is computed and its
  denominators are reported on every row, but it has no external replication.
