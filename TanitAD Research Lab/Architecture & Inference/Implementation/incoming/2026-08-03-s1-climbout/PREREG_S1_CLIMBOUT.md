# PRE-REGISTRATION — E-S1: THE S1 CLIMB-OUT. Is there a target to climb TO?

**Date:** 2026-08-03 (Europe/Berlin) · **Stream:** arch-inf (D-SEL, S1) · **Status:** written and
staged **BEFORE** any statistic below was computed · **GPU cost registered:** 0 training.

**Parent pre-registration:** `Project Steering/PREREG_D-SEL_REFC_SELECTION_SURFACE.md`
(§6.3 fifth branch, POST-HOC, is what sends this experiment).
**Antecedents, both MEASURED:** `…/incoming/2026-08-03-esel-verdict/ESEL_VERDICT.md` ·
`…/incoming/2026-08-03-s3-deployable/S3_DEPLOYABLE.md`.

**Estimator, declared before any number.** `taniteval/taniteval/ci.py::
paired_episode_cluster_bootstrap` / `episode_cluster_bootstrap`, resampling unit = **episode**,
`n_boot = 2000`, on the canonical **881 windows / 40 episodes** of
`physicalai-val-0c5f7dac3b11`. Two rankers on the same windows use the **paired** form.
⛔ `overlapping_holdout_se` is never called.

**"Fixed in advance" is made verifiable the same way the parent makes it:** by the **git blob id of
this file at staging time**, re-checked by the runner on every arm
(`prereg_s1.thresholds_unmoved_since_staging`). No self-hash is printed — a document cannot contain
its own digest, and R11 refuted the mtime version of this claim.

---

## 0. WHAT IS BEING DECIDED

E-SEL-0 MEASURED that REF-C's discarded refined confidence is a **WORSE** ranker than the shipped
t=0 classifier score — `+0.8372 m` [+0.6915, +0.9939] (base), `+0.9187 m` [+0.7778, +1.0669] (XL),
separated on both arms — while still scoring **8.7× / 16.6× chance**. The parent's post-hoc fifth
branch reads that as *"off-distribution, not uninformative ⇒ S1 must CLIMB OUT (supervise the
refined readout), not HARVEST"*.

⚠️ **`rank_acc` above chance is NOT the decision quantity, and this pre-registration refuses to
treat it as one.** S3_DEPLOYABLE §3 MEASURED that a score can carry large, separated rank
information and still select catastrophically: ρ = 0.6657 selects at **6.49 m** against a shipped
**0.4728**, and ρ = 0.9951 (`cv`) selects at **0.8149**. The quantity that decides a GPU-day is
**selection ADE out-of-episode on the reachable candidate set**, and every branch below is written
on it.

> **THE QUESTION:** *before spending a GPU-day supervising the refined readout, is there any
> supervised ranker over REF-C's own deployable, per-candidate information that beats the incumbent
> selector out-of-episode?* If yes, S1's climb-out has a target. If no, S1 is dead **whatever
> `rank_acc` says**, and the ~92 %-irreducible caveat wins.

---

## 1. WHAT WAS IMPLEMENTED FIRST (and why it is not what this experiment tests)

Two **zero-parameter** flags, default OFF, staged before this file. They remove the two places where
the object that is **SCORED**, the object that is **SUPERVISED** and the object that is **EMITTED**
are three different things.

| id | flag | mechanism | params |
|---|---|---|---|
| **S1b** | `sel_score_emitted` | the ranked confidence is read from the **EMITTED** fan. MEASURED from source: `_decode(kv, cond, x_in, t)` returns the confidence **of `x_in`** and the offset that improves it, and the loop emits `x = x_in + off` — so `refined` scores the estimate the last pass *consumed*, and the trajectories that actually leave the decoder are scored by **no head at all**. One extra conf-only pass; the offset is discarded so `anchor_traj` (and the published oracle-in-fan) is bit-unchanged. | **0** |
| **S1c** | `sel_ce_reach` | the ranked-score CE normalises over **exactly the survivor set the argmax ranks over**, and its target is the best candidate **in that set**. Today it is a full-fan softmax while the selector solves a 26–28 %-sized problem (73.76 % / 72.08 % of the fan is unreachable and never selected, and deleting it moves ADE by **exactly 0.0**). | **0** |

**Capacity delta: exactly 0, and it is proven, not asserted** — `param_breakdown` is unchanged with
either flag on, pinned by test, and an all-off build is byte-identical (state_dict keys **and**
values) and bit-identical in the forward to HEAD.

⚠️ **S1b is HYGIENE and is registered as such.** The shipped ranker is **two** passes stale and the
refined ranker is **one** pass stale, and the *two*-passes-stale one is 0.84 m **better**. ⇒
staleness is not the mechanism; supervision is. A large S1b effect would therefore be a **surprise**,
and §4 registers it as one.

⚠️ **S1c cannot be measured without training** — it is a change to a loss, and no frozen checkpoint
carries its effect. It is registered here so the arm is defined; it is **not** what §3 measures.

---

## 2. ⛔ WHAT IS *NOT* TESTABLE WITHOUT TRAINING — stated before measuring, not after

**A supervised climb-out cannot be simulated by re-weighting the banked score.** A ranker built as
any strictly-monotone function of one banked scalar has the **identical argmax**, so:

* refitting on `refined_logits` alone reproduces **1.3100 / 1.3901** exactly, and
* refitting on `logits` alone reproduces **0.4728 / 0.4714** exactly.

Both are *controls*, not treatments (§3.2). The climb-out changes what the readout **reads**, so
bounding it needs at minimum the refined pass's **query features `q`** — a GPU *inference* dump, 0
training — and measuring it needs a retrain. **This is registered here so that a null in §3 is not
later mis-reported as "the climb-out was tested and failed".**

What §3 *does* decide is strictly weaker and still decision-grade: **whether a target exists at
all.**

---

## 3. THE EXPERIMENT — E-S1-0, 0 GPU, on the banked augmented fans

`…/2026-08-03-esel-verdict/raw/fan_refined_refc-{base,xl}-30k.pt` (carry `refined_logits`) and
`…/2026-08-03-s3-deployable/raw/fan_deploy_refc-{base,xl}-30k.pt`. No checkpoint, no cache, no pod.

### 3.1 The protocol

A per-candidate scorer `s_i = w·φ_i` fit by **listwise softmax cross-entropy against the oracle
index**, **restricted to the S2-reachable survivor set**, evaluated **leave-one-episode-out**: no
window is ever scored by a model that saw its own episode. Headline read = **selection ADE@2s** of
`argmax_i s_i` over the survivors, paired against the shipped selector on the same windows.

⛔ **ρ is reported only as a secondary diagnostic and only on the reachable subset.** The trap that
caught two streams in a row is treating ρ over the full candidate axis as a proxy for a selector; it
is not, and §0 quotes the measurement that shows it.

### 3.2 The feature sets — nested, so every marginal is attributable

| id | φ | what it isolates |
|---|---|---|
| **A-shipped** | `logits` | ⚠️ **CONTROL, degenerate by construction** — monotone ⇒ must reproduce the incumbent exactly |
| **A-refined** | `refined_logits` | ⚠️ **CONTROL, degenerate by construction** — must reproduce E-SEL-0's refined selector exactly |
| **B-both** | `logits, refined_logits` | **does the refined readout carry ranking information the shipped one does not already have?** This is S1's marginal, in its cheapest honest form |
| **C-lon** | along-track candidate geometry only (terminal along-track displacement, its deviation from a bounded-accel band around `v0`, implied mean speed) | ⭐ **the LONGITUDINAL lever the gap says must exist** — 87.60 % / 89.28 % / 89.88 % of the selection gap is longitudinal at three scales |
| **D-lon+scores** | `C-lon ∪ B-both` | does longitudinal geometry add to the readouts, or replace them? |
| **E-cv** | `−‖cand − constant_velocity‖` alone | ⚠️ **CONTROL that has already fired once** — zero parameters, fully deployable, ρ = 0.995, and it selects at **0.8149 / 0.8158**, i.e. WORSE than shipped |

Every feature is **deployable**: computed from the emitted fan, the banked readouts and `v0`.
⛔ `cons_score` / `cons_oracle` are **excluded** — they read `z_{t+5}`, the future frame.

### 3.3 The controls — and the one that is vacuous is labelled, not counted

| control | what it establishes | **can it fire?** |
|---|---|---|
| **C-monotone** | A-shipped reproduces **0.4728 / 0.4714** and A-refined reproduces **1.3100 / 1.3901**, to ≤ 1e-6 | ✅ **yes** — if the LOEO plumbing, the reachability restriction or the argmax is wrong, a single-feature fit will *not* reproduce the incumbent |
| **C-incumbent** | ⭐ the **DIRECTION PREDICATE**: every treatment is scored against the **shipped selector**, not against a random or shuffled comparator | ✅ **yes, and it is the control S3 registered against a shuffled comparator and thereby missed** |
| **C-leak** | the same fit evaluated **in-episode** (leaky) beside the LOEO number | ✅ **yes** — a large leaky-vs-LOEO gap means any "win" is fitting noise |
| **C-permuted-target** | the fit is re-run against a **within-episode permutation of the oracle index**; its LOEO selection ADE must land at the uniform-random floor (**14.5426 / 13.9564**) | ✅ **yes** — a protocol that scores above the floor on a destroyed target is broken |
| **C-cv** | a zero-parameter deployable score (§3.2 E) | ✅ **yes — it has already fired once**, in S3_DEPLOYABLE §2.3 |
| **C-reproduce** | the bank's `logits`, `refined_logits`, `fan`, `gt` are bit-identical to E-SEL's and `argmax(logits) == sel` on 1.0000 | ✅ yes |
| ~~C-shuffled~~ | ⛔ **VACUOUS BY CONSTRUCTION — reported, never load-bearing.** Permuting a score then taking argmax is a uniform pick for *any* score, so `E[ρ] = 0` and `E[ADE]` = the uniform floor regardless of the score. It establishes only `≠ 0`. | ❌ **no** |

---

## 4. ⛔ BOTH OUTCOMES COMMITTED IN ADVANCE — every two-sided quantity gets a THREE-SIDED row

Thresholds fixed **here**. "Separated" = the paired episode-cluster bootstrap 95 % CI excludes 0.
"Material" = |ΔADE@2s| ≥ **0.02 m** (the parent's `free_win_m`, inherited deliberately rather than
re-invented). **Every trigger carries a DIRECTION predicate** — *separated **and** the delta favours
the treatment* — because a trigger satisfied literally while its controls beat the score is what
happened to S3 one day ago.

### 4.1 B-both — S1's marginal (the headline)

| branch | trigger (fixed now) | meaning | next |
|---|---|---|---|
| **S1-TARGET-EXISTS** | `ADE(B-both) − ADE(shipped)` separated **AND ≤ −0.02 m** (better) **AND** LATERAL not separated worse **AND** C-leak gap < 50 % of the effect | the refined readout carries ranking information the shipped one lacks, and it converts into **selection**, not just into ρ | fund `dsel-s1only` (§5); the climb-out has a target |
| **S1-TARGET-MARGINAL** | separated and better but **> −0.02 m** | real but too small to buy a GPU-day on its own | do **not** fund alone; fold into an arm that is already running for another reason |
| **S1-NOT-SEPARATED** | CI includes 0 | the refined readout adds nothing over the shipped one that survives out-of-episode | ⛔ **S1 IS DEAD AFTER ALL** — see §4.4 |
| **S1-ADVERSE** ⚠️ | separated **AND worse** | combining the two readouts is worse than the shipped one alone — the refined channel is actively misleading where they differ, which E-SEL's `corr = 0.9239` already hints at | ⛔ **S1 IS DEAD AFTER ALL**, and more strongly: report that the refined channel must not be *added*, only *replaced* |

### 4.2 C-lon — is the longitudinal lever reachable from deployable geometry?

| branch | trigger | meaning | next |
|---|---|---|---|
| **LON-LEVER-LIVE** | `ADE(C-lon) − ADE(shipped)` separated **AND ≤ −0.02 m** **AND** the LONGITUDINAL family (`speed_abs_err_mps`, `along_abs_err_m`) separated **better** | the 87.6–89.9 % longitudinal gap is at least partly rerankable from information REF-C already has at inference | ⭐ **this outranks S1**: it says the ranker should be given longitudinal geometry, and it is free |
| **LON-LEVER-NULL** | CI includes 0 | the longitudinal share of the gap is *not* reachable by reranking this fan | the gap is a **proposal** problem on the longitudinal axis, not a selection one — which reverses D-SEL's premise for that axis and must be reported as such |
| **LON-LEVER-ADVERSE** ⚠️ | separated **AND worse** | longitudinal geometry actively mis-ranks | report it; it would mean the fan's longitudinal spread is anti-correlated with correctness |

### 4.3 S1b — the emitted-fan readout (needs one inference pass, 0 training)

| branch | trigger | meaning |
|---|---|---|
| **S1b-HYGIENE (predicted)** | `ADE(conf(X₂)) − ADE(conf(X₁))` **not separated**, or separated by < 0.02 m | staleness is not the mechanism, as the 2-passes-stale shipped ranker being *better* already implies. Keep the flag: it makes the scored object the emitted object, which is correct regardless of effect size |
| **S1b-MATERIAL** ⚠️ | separated **AND ≤ −0.02 m better** | I was wrong about the mechanism; part of the refined deficit is that the score never saw the emitted fan | say so; raise S1b above S1c in the arm |
| **S1b-ADVERSE** ⚠️ | separated **AND worse** | scoring the emitted fan is *worse* than scoring its predecessor — which would mean the readout is keyed to the pre-offset geometry | say so; it would make S1b inadmissible as a default and is a real finding about what the conf head reads |
| **S1b-UNMEASURED** | no free GPU host reachable | report **UNMEASURED with the reason**, never as a pass |

### 4.4 ⛔ THE "S1 IS DEAD AFTER ALL" BRANCH, stated explicitly

**S1 is dead** if `B-both` is **not separated better than the shipped selector** (4.1 rows 3–4)
**AND** `C-lon` is not separated better (4.2 row 2–3). In that case:

* the 8.7× / 16.6×-chance `rank_acc` that kept S1's premise alive is **not a capability**, it is a
  rank statistic that does not convert into selection — the same reading S3_DEPLOYABLE forced on
  ρ = 0.6657;
* D-SEL reduces to **S2 + S4 as hygiene**, exactly as the parent's `S1 DEAD` row says;
* ⛔ **it is reported as a death, not as "needs training to emerge".** That reframing is the exact
  move S3_DEPLOYABLE's escalation 2 forbids.

⚠️ **What a null here does NOT license:** *"supervising the refined readout cannot work."* §2 is
registered in advance precisely so this cannot be claimed. A null bounds the *free* version; the
supervised version changes what the readout reads and is out of this experiment's reach.

### 4.5 RED FLAG — stop and audit, do not publish

Any treatment separated better by **> 0.10 m**. That is ~1/3 of the whole selection gap from a
LOEO-fit linear model on 2–5 features, and the first hypothesis must be a **leak in the protocol**
(an episode split that is not disjoint; a feature computed from `gt` or `z_{t+5}`), not a win.

---

## 5. IF a retrain is warranted — the arm, and how it is judged

Identical data, parity key `physicalai-train-e438721ae894`, skip-hash `f09e44db`, identical
optimizer/schedule/seed. **The ONLY differences are the named flags.**

| arm | flags | isolates |
|---|---|---|
| `refc-base-30k` | — (the published control) | — |
| `dsel-s1only` | `--sel-refined --labels v21` | S1 as the parent registered it |
| **`dsel-s1climb`** | `--sel-refined --sel-score-emitted --sel-reach-clamp --sel-ce-reach --labels v21` | **the climb-out**: score the emitted fan, supervise it on the set the argmax ranks over |
| `dsel-s1c-only` | `--sel-refined --sel-reach-clamp --sel-ce-reach --labels v21` | `dsel-s1climb − dsel-s1c-only` = S1b's marginal |

**Cheapest sufficient arm:** `dsel-s1climb` vs `dsel-s1only`, both 30 k steps, one A40-class GPU
each. ⛔ **Not launched here. A GPU-day is the PI's call.**

**Success:** TACTICAL `frac_sel_2x_worse` separated **below** control **AND** LONGITUDINAL separated
**better** **AND** LATERAL not separated worse **AND** ADE not separated worse.
**Failure:** LATERAL separated worse (guard-rail), **or** `frac_sel_2x_worse` CI includes 0.

⚠️ **The LATERAL guard-rail is tight by construction and that is MEASURED, not assumed:** a
*perfect* reranker of this fan buys only **0.033 m** (base) / **0.0425 m** (XL) of cross-track and
moves heading, curvature and yaw-rate **not at all**. Any separated lateral regression is therefore
a real fault.

---

## 6. THE FOUR METRIC FAMILIES — per family, never pooled (binding)

Grid **derived**, never assumed: `wp_steps [5,10,15,20] × 0.1 s → dt = 0.5 s`
(`four_families.infer_dt`). A hard-coded 0.1 s inflates speed ×5 and accel ×25 (R-2026-08-03-c).

| family | metric | pre-committed direction |
|---|---|---|
| **ADE** | `ade_0_2s` | headline, **never alone** |
| **LONGITUDINAL** | target-speed accuracy, `speed_bias` (**signed**), `along_abs/signed_err_m`, **distance-keeping: headway / time-gap / TTC** via `taniteval/lead_source.py::lead_block` + `lead_metrics.distance_keeping` | improve — **this is the family the lever must move** |
| **LATERAL** | `cross_abs_err_m`, `heading_abs_err_deg`, `curvature_abs_err_1pm`, `yaw_rate_abs_err_degps` | ⛔ **GUARD-RAIL: must be NOT separated worse** |
| **TACTICAL** | `rank_acc`, `sel_gap`, `frac_sel_2x_worse` (goal/anchor selection); manoeuvre confusion where available | improve |
| **STRATEGIC** | route/goal quality **with its majority-class baseline printed beside it** | reported with `n` and reason if not computable |

Where a family cannot be computed it is reported **per family, with the reason and the n** — never
silently dropped. ⛔ An ADE horizon sweep is **one row of five**, never "the result".

---

## 7. ⚠️ INSTRUMENT HAZARDS THAT BIND THIS EXPERIMENT

1. **ρ over the full candidate axis is not a proxy for a selector** (S3_DEPLOYABLE §3). Restrict to
   S2-reachable **before** any rank statistic, and report selection ADE alongside.
2. **R-2026-08-03-c**: every published four-family ABSOLUTE rate before the `dt` fix is wrong by
   5×–25×. Paired deltas on the same windows survive; absolute rates are re-stated only post-fix.
3. **R-2026-08-02-a**: REF-C at the wrong raster returned numbers *silently* on base and crashed on
   XL. Any pass that touches a checkpoint asserts 256×256 → 8×8 = 64 tokens first.
4. **The C6 confound is inherited on purpose**: the banks decoded `nav_mode='follow_constant'`,
   which is the condition the published 0.4728 / 0.4714 were collected in. Changing it would move
   the baseline every contrast is paired against.
5. **`dist_to_gt_traj_m` ≡ `cross_track_abs_m`** — four lateral metrics, not five.
6. **0.1640 / 45.4 % are XL's; base is 0.1914 / 41.09 %.** The arm travels with the number.

---

## 8. REGISTERED PERSONAL PREDICTION

* **B-both: `S1-NOT-SEPARATED`.** I expect the refined readout to add nothing over the shipped one
  that survives leave-one-episode-out. Reason: `corr(anchor, refined) = 0.9239 / 0.8998` — it is
  nearly the same score, and where it differs E-SEL measured it decisively worse. A 2-feature linear
  blend has no mechanism to exploit "same but worse".
* **C-lon: `LON-LEVER-NULL`, low confidence (≈55 %).** The longitudinal share is a statement about
  *where the gap is*, not about whether reranking can close it — and `cv`, the strongest purely
  geometric deployable score in the programme, already selects **worse** than shipped. But this is
  the row I most expect to be wrong, and it is the one I would most like to be wrong about.
* **S1b: `S1b-HYGIENE`.** |Δ| < 0.02 m.
* **Overall: I expect §4.4 to fire and S1 to be reported DEAD.** If it does, this document and the
  two zero-parameter flags are the cheap way the programme found that out — and the flags stay,
  because "the scored object is the emitted object" and "the CE normalises over the set the argmax
  ranks over" are correct independent of whether they pay.

**If B-both comes back separated and better, I was wrong that the refined channel is redundant, and
I will say so in the results file rather than reframe it.**

---

## 9. WHAT THIS IS NOT

* **Not a test of the supervised climb-out** (§2, registered in advance).
* **Not a headroom claim** — the oracle gap is ~92 % irreducible by the registry's standing caveat,
  which is itself **INHERITED** from a prose note and flagged as load-bearing.
* **Not a capacity change** — both flags are 0 parameters, proven by `param_breakdown`.
* **Not launched.** No training started, no training pod touched.
