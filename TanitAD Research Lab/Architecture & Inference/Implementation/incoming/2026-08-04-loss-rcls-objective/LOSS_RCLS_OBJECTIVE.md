# E-OBJ-1 — the objective swap SURVIVES pre-registration, it costs **0 parameters**, and the half everyone would have changed is the WRONG half

**Date:** 2026-08-04 (Europe/Berlin) · **Stream:** arch-inf (D-SEL) · **GPU cost: ZERO.**
No training launched, no training pod touched (`tanitad-new` / `tanitad-pod4` untouched), nothing
pushed. Every number below is CPU work on **banked fans** already in the repo.

**Pre-registration:** `PREREG_LOSS_RCLS_OBJECTIVE.md` (this directory), **staged and
content-pinned BEFORE the first statistic** — git blob **`603f9a9793c18d4c329fdbe3d105a8ef80fe108a`**,
`git ls-files -s` and `git hash-object` **MATCH**, and the runner re-verifies it on every arm
(`prereg.thresholds_unmoved_since_staging`). That is the falsifiable object and anyone can check it.

**n = the canonical 881 windows / 40 episodes** of `physicalai-val-0c5f7dac3b11`, both arms.
**Estimator: paired episode-cluster bootstrap**, unit = episode, `n_boot = 2000` (`taniteval/ci.py`).
⛔ `overlapping_holdout_se` never called.

---

## 0. LEAD — what the brief asked, answered first

| the question | the answer |
|---|---|
| **Does the objective swap survive a proper pre-registration?** | **YES — `OBJ-LIABILITY-CONFIRMED` fires on BOTH arms**, three-sided, with direction predicates, and with the control that could have voided it (**C-optimiser**) passing. The post-hoc −0.0974 / −0.1670 replicates exactly. |
| **What does it cost?** | **EXACTLY ZERO PARAMETERS**, proven not asserted: `param_breakdown` is *identical* — not merely equal in total — across `softade`, `softce`, `softce+weight` and the full D-SEL preset; `refc_config()` stays at **104,191,577**. Ledger: accepted levers cost **+897 / +385 / +128**; one rejected one cost **+272,001**. This is **0**. |
| **Does it beat the shipped selector?** | ⛔ **NO. `DEPLOYABLE-NULL`.** The swap **repairs a self-inflicted deficit back to parity and stops there.** It buys no free win, and **it does not revive S1, S3, S5 or S6.** |
| ⭐ **WHICH HALF of `loss_rcls` is the liability?** | **The OBJECTIVE FORM — metric-awareness — NOT the one-hot target.** Softening the target (`softmax(−fan_err/τ)`) while keeping the CE is **separated WORSE than the incumbent** at every τ tested. ⚠️ This is the *opposite* of my registered prediction (`TARGET-SHAPE`, ~55 %), and it inverts the one-line change a reader of the escalation would have made. |
| **Is the recovery in the family that matters?** | **YES, and it is now measured with the distance-keeping family that E-S1-0 could not compute** — 270 of 881 windows carry a lead, from a lead block that **two independent builds reproduce to 2.3e-13 m**. |

⛔ **And the plainest statement of what this does NOT license:** the escalation said *"we may have
spent a session testing four levers built on a defective objective."* That is **half right and the
half it gets wrong is the expensive half.** The objective IS a measured liability for *combining*
readouts — but the levers' verdicts were **not** decided by it. §7 says which stand and why.

---

## 1. WHAT WAS MEASURED, AND WHY IT IS A DIFFERENT EXPERIMENT FROM E-S1-0

E-S1-0 fit every candidate ranker with a **listwise softmax CE against the one-hot oracle index** —
bit-for-bit the shape of `refc_train.loss_rcls` (`stack/scripts/refc_train.py`,
`loss_rcls = F.cross_entropy(ce_score, r_star.detach())`, `r_star = ce_err.argmin(dim=1)`). Under it
**every** fit was separated **worse** than the incumbent, *including feature sets containing the
incumbent's own score*, with a C-leak gap of −0.001 to −0.003 m — so **not** overfitting. Its
post-hoc probe swapped the objective for expected-ADE and recovered most of the deficit.

This experiment holds **features, z-scoring, survivor mask, LOEO folds, optimiser, iterations,
learning rate, initialisation and dtype fixed** and changes **only the objective**, across three
forms:

| id | objective | role |
|---|---|---|
| **O-ce** | `CE(s, argmin_survivors fan_err)` | ⚠️ the INCUMBENT — reproduces E-S1-0's registered rows, which is itself a control |
| **O-softade** | `E_{i∼softmax(s)}[fan_err_i]` | the escalation's candidate — **METRIC-AWARE** |
| ⭐ **O-softce(τ)** | `CE(s, softmax(−fan_err/τ))` | **NEW.** Keeps the CE *form*; replaces only the *target shape* |

**Why the third arm is the whole point.** Nothing in the programme had separated *"the CE is the
wrong shape of loss"* from *"one winner among ~128 near-duplicates is the wrong target"*. They are
**different one-line changes at 0 parameters**, and only one of them is the right recommendation.

---

## 2. THE RESULT — three primaries, adjudicated mechanically

<!-- VERDICT_TABLE -->

### 2.1 What each primary means, in words

**PRIMARY-1 (Q1) — `OBJ-LIABILITY-CONFIRMED`, both arms.** Swapping the objective on identical
features recovers essentially the whole deficit the CE created. C-leak is a rounding error, so this
is not fitting noise; **C-optimiser passes**, so it is not an optimiser artifact; C-monotone,
C-scale-invariance and the survivor-restricted C-permuted-features all pass, so it is not a plumbing
error. **The escalation's core claim replicates under pre-registration.**

**PRIMARY-2 — `DEPLOYABLE-NULL`, both arms.** Nothing in the declared-in-advance pool beats the
shipped selector. ⭐ **The honest reading: the objective swap does not create information, it stops
destroying it.** A fitted ranker returns from *separated worse* to *statistically indistinguishable*
from the incumbent, and stops. My registered prediction (~85 %) was right, and the RED FLAG for a
>0.10 m "win" never had to fire.

**PRIMARY-3 (Q2) — the registered table returned `UNCLASSIFIED`, and I am not rewriting the trigger.**
Observed: `softce − ce` **ADVERSE**, `softade − ce` **BETTER**. My registered `OBJECTIVE-FORM` branch
required `softce − ce` to be *not separated*; it is separated **in the adverse direction**, which is
the same conclusion arrived at **more strongly**. The runner names that substance in a field that
explicitly **never overrides the attribution** (`substance_POSTHOC_never_overrides_attribution`), and
the JSON carries both. ⚠️ **This is the third pre-registration in a row whose joint table could not
express what happened** — see §8.1; the difference here is that the classifier *has an explicit
`UNCLASSIFIED` outlet that fires*, rather than mis-binning into the nearest branch.

⇒ **The recommendation is `softade`, and it is NOT the change most readers of the escalation would
have made.** "Soften the one-hot target over near-duplicate candidates" is the intuitive fix, it is
the cheaper-looking fix, it preserves the CE form — and **it is measured to make things worse.**

---

<!-- PANEL_TABLE -->

---

## 3. THE CONTROLS — including the two that FIRED

<!-- CONTROLS_TABLE -->

### 3.1 ⭐ C-optimiser — the control that could have voided the headline, and did not

Registered as *"if the CE fit is beaten under its OWN objective, the difference is an OPTIMISER
artifact and the finding is VOID — not a result in either direction."* Every non-degenerate feature
set has `each_fit_wins_its_own_objective = True`: the three fits are three genuinely different
optima of three genuinely different objectives, reached by the same optimiser at the same budget.
`cos(w_ce, w_softade) < 1` confirms the difference is a **direction** difference, and
**C-scale-invariance** confirms a linear score's magnitude cannot change its argmax — so the effect
cannot be a scale artifact.

### 3.2 ⚠️ C-continuity FIRED — and the reason is a property of the fan, not a bug

Registered predicate: `O-softce(τ = 0.01)` must land within **1e-3 m** of `O-ce`, because
`softmax(−e/τ) → onehot(argmin e)`. It did not. The POST-HOC extension (labelled, **moves no
threshold, adjudicates no branch**) shows why: the fan is made of **near-duplicates**, and the ADE
gap between the best and second-best survivor is *finer than τ = 0.01 m* in the median window — so
at τ = 0.01 the target is still spread over several near-ties and the one-hot limit has not been
reached. At τ = 1e-3 / 1e-4 it converges. **The registered tolerance was set at the wrong τ**, and
the loss-level continuity is pinned exactly by test
(`test_softce_converges_to_the_incumbent_ce_as_tau_goes_to_zero`, gap < 1e-6 at τ = 1e-4).
⛔ C-continuity is **not** in the VOID set — it was registered as a diagnostic, and it stayed one.

### 3.3 ⭐ AN INSTRUMENT CORRECTION — E-S1-0's `C-permuted-features` is not a valid control as written

`C-permuted-features` destroys the feature↔candidate association and must land at the
survivor-restricted uniform floor. E-S1-0 permuted the **whole row**. But `_zrow` fills every
**non-survivor entry with the constant 0**, and ~72–74 % of the fan is non-survivor — so a whole-row
permutation drags those zeros into survivor slots and **changes the tie structure the argmax
resolves**. That is an artefact of the standardisation, not of the data.

Both versions are run here and both are printed. The **survivor-restricted** version — survivors
permuted among themselves, non-survivors left in place — is the load-bearing one, and it is what
`passes` reads.

⚠️ **This is the same class as E-S1-0 §3.5's own correction** (*"a control judged against the wrong
floor is not a control"*), one level deeper: there it was the mask that must not be permuted; here
it is the **fill value** that must not be moved. E-S1-0's row passed under the CE and would have
been read as a clean control.

---

## 4. THE FOUR METRIC FAMILIES — per family, never pooled (binding)

Grid **derived, never assumed**: `wp_steps [5,10,15,20] × 0.1 s → dt = 0.5 s`
(`four_families.infer_dt`). A hard-coded 0.1 s inflates speed ×5 and accel ×25 (R-2026-08-03-c).
⛔ **An ADE horizon sweep is one row of five and is never "the result".**

<!-- FAMILIES_TABLES -->

### 4.1 ⭐ LONGITUDINAL — distance-keeping, the family E-S1-0 reported at **n = 0**

E-S1-0's §8 said plainly: *"Did not compute LONGITUDINAL distance-keeping — the one family the brief
singled out."* **It is computed here, at 0 re-inference**, because `taniteval/lead_source.py`
reproduces the canonical 881-window grid exactly, so a banked lead block is **row-aligned** with a
banked fan dump.

**Alignment is proven by CONTENT, not assumed** (`C-lead-alignment`): the lead block's `speeds` come
from the raw **egomotion parquet** via registration; the fan bank's `v0` comes from the **episode
cache**. Two independent paths to the same physical quantity agree on all 881 rows. A window-count
match alone would prove nothing — two different orderings can have identical counts.

⭐ **And the instrument itself is REPLICATED, not single-source** (`raw/C_lead_cross_build.json`):
the Thor-built block (`thor:/home/nvidia/leadwork/val40_lead_blocks.pt`, aarch64) and the
Benchmarks & Eval stream's independently-built `val40_lead_block.npz` (x86_64, a different runner)
agree on **every one of the 881 window states** and to **2.3e-13 m** on lead positions, **exactly 0**
on `lead_lens` and `gap0_m`. E-S1-0 escalated that this instrument lived on one Drive-backed
checkout; it now exists in three places and two of them were checked against each other.

⛔ **Three window states, never two.** `NO_LABEL` is never counted as free flow — that would
manufacture empty road and flatter every arm. Each mean carries its own denominator, and every
paired contrast is on the **intersection** of the two arms' finite windows with the asymmetry
printed, because an arm that steers its predicted path out of the corridor **silently drops its own
lead** and would otherwise look free.

<!-- DK_TABLES -->

### 4.2 TACTICAL — the goal/anchor-selection half

<!-- TACTICAL_TABLE -->

### 4.3 Per-family status, with the reason and the n where not computable

| family | status | n |
|---|---|---|
| **ADE** | ✅ MEASURED, every arm, both scales | 881 |
| **LONGITUDINAL** — speed / along-track, **signed AND absolute** | ✅ MEASURED, every arm | 881 |
| ⭐ **LONGITUDINAL** — distance-keeping (headway / time-gap / min-TTC) | ✅ **MEASURED** — the family E-S1-0 could not compute | **270 LEAD** of 881 (551 NO_LEAD, 60 NO_LABEL) |
| **LATERAL** — cross-track, heading, curvature, yaw-rate | ✅ MEASURED, all four, every arm | 881 |
| **TACTICAL** — goal/anchor selection (`rank_acc`, `sel_gap`, `frac_sel_2x_worse`) | ✅ MEASURED — the half D-SEL exists to move | 881 |
| **TACTICAL** — manoeuvre decision + confusion | ⛔ **UNAVAILABLE**: a fan bank stores no decoded manoeuvre logits. **A WORK ITEM, not a pass** — it needs a dump that carries `man_logits`, which is one inference pass and 0 training | **0** |
| **STRATEGIC** — route/goal quality | ⛔ **UNAVAILABLE**: no route/goal label in a fan bank, and the decode used `nav_mode='follow_constant'` so the route input was never exercised (the C6 confound, **inherited on purpose** — changing it would move the baseline every contrast is paired against) | **0** |

⚠️ **The LATERAL guard-rail is tight by construction and that is MEASURED, not assumed**: the
CEILING row (a *perfect* reranker of this fan minus shipped) shows how little lateral movement any
reranking can buy. Any separated lateral regression is therefore a real fault, and the table above
says which arms have one.

---

## 5. WHAT IS FREE, WHAT NEEDS TRAINING, AND THE EXACT PARAMETER COST (P2)

### 5.1 The split, stated cleanly

| question | measurable on the banked fans? | why |
|---|---|---|
| does the objective change how banked readouts should be **COMBINED**? | ✅ **YES — measured here** | a 1–7-parameter linear scorer over banked features, LOEO, needs no checkpoint |
| does the objective change what the score **READS**? | ⛔ **NO — needs a retrain** | that is a property of 104 M parameters, not of a linear blend |
| would a `softade`-trained REF-C select better? | ⛔ **NOT MEASURED, and registered as unmeasurable here in advance** (prereg §1.1) | a win here is **NECESSARY-NOT-SUFFICIENT**; a null here would not have shown a retrain fails |

⚠️ **And the structural blindness, registered before measuring:** `argmax(w·φ)` is invariant to a
positive rescaling of `w`, so on a **single feature every objective has the identical argmax**. The
probe is therefore blind to any objective effect on a 1-D score — which is why the 1-D rows are
CONTROLS. In training `sel_score` is *also* 1-D, but it is produced by 104 M parameters, so the
analogy is imperfect and the conclusion is phrased as a bound, not a transfer.

### 5.2 The parameter cost — **exactly 0**, and proven

```
refc_config()                                   104,191,577      (the registry's number)
+ sel_ce_objective="softade"                    104,191,577      Δ = 0
+ sel_ce_objective="softce", tau=0.25           104,191,577      Δ = 0
+ sel_ce_objective="softade", weight=0.1        104,191,577      Δ = 0
+ the full D-SEL preset with softade            104,191,577      Δ = 0
param_breakdown IDENTICAL (not merely equal in total) in all four
```

Pinned by `stack/tests/test_refc_ce_objective.py::test_every_objective_costs_exactly_zero_parameters`.
For scale: the accepted levers cost **+897** (`factored_maneuver`), **+385** (the D-SEL preset:
S3's 1 + S5's 384), **+128** (`nav_known_channel`); the rejected first implementation of
`factored_maneuver` cost **+272,001** and only its own capacity test caught it.

### 5.3 What was IMPLEMENTED (0 GPU), and its fail-louds

Three flags on `RefCConfig`, **all default OFF, all 0 parameters**, with the incumbent path kept
**bit-identical** (pinned by test):

| flag | what it does | guard |
|---|---|---|
| `--sel-ce-objective {ce,softade,softce}` | which objective trains the ranked score | refused unless one of `--sel-refined` / `--graft-cons` / `--graft-route` is on — with all three off `loss_rcls` is **never constructed** and the flag would be **silently inert while `config.json` recorded it ON** |
| `--sel-ce-soft-tau` | the softened-target temperature | refused with any objective but `softce`; refused if ≤ 0 under `softce`; refused if negative (a negative temperature puts the CE's mass on the **worst** candidate and still produces a falling loss curve) |
| `--sel-ce-weight` | multiplier applied **only off the `ce` path** | refused on the `ce` path, so the published control arm can never be silently rescaled |

⚠️ **`softce` prints a loud banner saying it is the CONTROL and carrying the measured numbers that
refute it** — the same discipline `--sel-score-emitted-t -1` already uses.

⭐ **THE SCALE TRAP, which the frozen probe structurally cannot see and which would have silently
confounded the arm.** `softade` is in **METRES**; the CE is in **NATS**; `REFINED_CLS_WEIGHT = 1.0`
was calibrated for the latter. Swapping the objective therefore **silently re-weights the whole
selection term against LAW / manoeuvre / trajectory.** `--sel-ce-weight` exists so that weight is a
**recorded decision** rather than an accident, and the trainer says so at launch. A `softade` arm run
without deciding it would be a lever confounded with a loss-weight change — the `--v2` conflation
failure in a new costume.

⛔ **`ce_err` is DETACHED in every branch, and that is an attribution bound, not hygiene.** `fan_err`
is differentiable w.r.t. `anchor_traj`; without the detach `softade` would also push the **fan**, and
the arm would be optimising the TRAJECTORY as well as the SELECTION with no way to assign the
result. The incumbent detaches its own target for exactly this reason. Pinned by
`test_softade_never_pushes_the_fan`.

---

## 6. THE CHEAPEST DISCRIMINATING ARM — specified, ⛔ NOT LAUNCHED

A GPU-day is the PI's call. Identical data, parity key `physicalai-train-e438721ae894`, skip-hash
`f09e44db`, identical optimizer / schedule / seed. **The ONLY difference is the named flag.**

| arm | flags | isolates | params |
|---|---|---|---|
| `refc-base-30k` | — (the published control) | — | 0 |
| ⭐ **`obj-softade`** | `--sel-refined --sel-reach-clamp --sel-ce-reach --sel-ce-objective softade --sel-ce-weight <decided> --labels v21` | **the OBJECTIVE FORM** — the arm the measurement supports | **0** |
| `obj-softce` | same, `--sel-ce-objective softce --sel-ce-soft-tau 0.25` | **the CONTROL** that separates metric-awareness from target-softness *in training* | **0** |

**Cheapest sufficient pair:** `obj-softade` vs the published control, 30 k steps, one A40-class GPU.
`obj-softce` is worth its GPU-day **only** if the pair separates — it is the control that says
whether the training-time effect is metric-awareness (as the frozen probe says) or merely a softer
target. **Pre-registered prediction, pinned now:** `obj-softce` will be **worse** than `obj-softade`,
and I will say so in the results file if it is not.

**Success:** LONGITUDINAL (`speed_abs`, `along_abs`) separated **better** **AND** LATERAL not
separated worse **AND** ADE not separated worse **AND** `frac_sel_2x_worse` separated below control.
**Failure:** LATERAL separated worse (guard-rail), **or** LONGITUDINAL CI includes 0.

⚠️ **Re-measuring any of this is a NEW pre-registration, not an amendment to this one.**

---

## 7. WHAT IT MEANS FOR S1 / S3 / S5 / S6 (P4) — and what does NOT change

### 7.1 The structural fact, from source

`refc_train.py` states it and asserts it at runtime
(`assert_selection_params_are_alive`, which raises rather than warns):

> *"The ranked score must be SUPERVISED for a graft on it to train — `argmax` has no gradient and
> nothing else in the loss differentiates w.r.t. `sel_score`."*

The parameters it guards are `cons_gate` (**S3**), `route_to_anchor` (**S5**) and
`goal_gate` / `goal_dist_gate` / `goal_head` (**S6**). ⇒ **`loss_rcls` is the SOLE trainer of S3's
1 parameter, S5's 384 and S6's 2,117.** That is why an objective defect here is not an S1 problem.

### 7.2 Which verdicts stand, and which are now conditional

| lever | how its verdict was obtained | does `loss_rcls` decide it? | status |
|---|---|---|---|
| **S1** `sel_refined` (0) | `S1_IS_DEAD_AFTER_ALL` fired on a **CE-fitted** linear ranker over banked readouts | ⚠️ **the REASON changes, the VERDICT does not** | ⛔ **STANDS.** PRIMARY-2 here is `DEPLOYABLE-NULL`: under the *corrected* objective the fitted ranker still does not beat the incumbent. The death is now attributable to *no free target*, not to *a bad fit* — which is a **stronger** result, not a reprieve |
| **S2** `sel_reach_clamp` (0) | argmax-only mask; paired ΔADE **exactly 0.0** | ⛔ **no** — it never touches a loss | ✅ **STANDS, untouched** |
| **S3** `graft_cons` (1) | ρ probe + an **α grid selected by held-out selection ADE**, not by a CE | ⛔ **no** for the frozen bound; ✅ **yes** for a trained graft | ⛔ **the DEPLOYABLE verdict STANDS** (ρ_deploy 0.2493 / **−0.2728**, sign flip on XL; realized ΔADE −0.0013 / −0.0024, not separated). ⚠️ **the TRAINED arm's prior changes**: `cons_gate` is trained *exclusively* by the objective this experiment measured as a liability |
| **S4** `seam_clamp` (0) | in-graph norm cap | ⛔ **no** | ✅ **STANDS** |
| **S5** `graft_route` (384) | registered as the **lowest-prior** lever on architectural grounds (`route_head` is nav-blind; junction accuracy 0.7613 **below** the 0.7806 majority baseline) — **not** on a CE-fitted measurement | ⛔ **no** — the argument is architectural | ✅ **STANDS.** A better objective cannot make a nav-blind readout informative |
| **S6** `graft_goal` (2,117) | **not yet measured** — sequenced behind the temporal-feature stream | ✅ **yes, when it runs** | ⚠️ **NO VERDICT TO REVISE.** But when it runs it should run under a *decided* objective, and its 2,117 parameters are trained by `loss_rcls` alone |

⇒ **Nothing is quietly revived.** S1 is dead, S3's deployable form is refuted, S5 is architecturally
low-prior. What changes is a **prior on future training arms** that graft onto the ranked score, and
the standing recommendation that any such arm names its objective explicitly.

### 7.3 ⭐ The sharpest consequence, stated plainly

**A `graft_*` lever's parameters see the world only through `loss_rcls`.** If that objective is
measurably the wrong shape for *combining* two banked readouts at 2 parameters, then every future
graft is being asked to learn through the same defect — and a null result from such a lever is
**confounded between "the lever has no information" and "the objective could not use it"**. That
confound is exactly what E-S1-0 ran into and it is now named. ⛔ **It is a prior, not a re-run
order:** none of the verdicts in §7.2 was decided by a CE fit.

---

## 8. WAS I RIGHT? — the registered prediction, scored

<!-- PREDICTION_TABLE -->

### 8.1 ⚠️ The pre-registration defect that has now recurred THREE TIMES

E-SEL-0 hit *"no branch for separated-adversely"* and escalated it. S3_DEPLOYABLE inherited the same
defect one day later. E-S1-0 fixed it for **single** contrasts — every two-sided quantity got an
`ADVERSE` row, and every row came back `ADVERSE`. **This pre-registration inherited that fix and it
worked.** What broke here is one level up: the **JOINT** table over three contrasts. Its four cases
were not exhaustive, and the observed combination (`softce−ce` ADVERSE **and** `softade−ce` BETTER)
fell through.

**What I did differently, and what I would do next time.** The classifier has an explicit
`UNCLASSIFIED` outlet that **fires and is reported**, and the substance is named in a field that
cannot override the attribution — so no branch was silently widened. **The generalisation:** an
ADVERSE row on each contrast is not enough; a joint table over *k* contrasts needs its cases
enumerated as a **product of the three-sided outcomes**, or an explicit catch-all, or it will have
holes. Filing this as an escalation rather than a footnote.

---

## 9. WHAT I DID NOT DO — plainly

* ⛔ **Did not train anything, launch any arm, or touch `tanitad-new` / `tanitad-pod4`.** GPU cost
  for this experiment: **zero**. The only remote work was a read-only survey of Thor and a 124 KB
  file pull.
* ⛔ **Did not measure whether a RETRAIN under `softade` helps.** Registered as out of reach in
  advance (prereg §1.1). The probe measures the objective's effect on **combining frozen scores**,
  not on **learning** a score.
* ⛔ **Did not re-open S1, S3, S5 or S6**, and did not re-run any of their experiments. §7 states
  which verdicts stand and why; re-measuring is a new pre-registration.
* ⛔ **Did not compute the TACTICAL manoeuvre-confusion or the STRATEGIC family** — UNAVAILABLE with
  reason and **n = 0** (§4.3). Both are work items, not passes.
* ⛔ **Did not run `refc-small-30k`** — no augmented bank exists for it (unchanged since E-SEL).
* ⛔ **Did not commit and did not push.** Everything is **staged**, verified with
  `git ls-files --cached` (an `add` exit code is not evidence).
* ⚠️ **Did not fix E-S1-0's published `C-permuted-features` row.** The instrument correction is
  implemented and reported here (§3.3); re-running E-S1-0's panel under it is a separate task, and
  its registered verdict does not depend on that row (it *passed* under the CE).
* ⚠️ **The index contains other streams' staged work.** I staged only my own paths and verified
  them individually; I did not commit, precisely because a pathspec-free commit would sweep them in.

---

## 10. 🔴 ESCALATIONS

1. **→ PI / arch-inf — the recommended one-line change is `softade`, NOT the intuitive one.**
   The soft-target fix that a reader of the escalation would naturally implement is **measured
   worse** than the incumbent at every τ. The arm to fund, if any, is `obj-softade`, and its
   `--sel-ce-weight` must be **decided, not defaulted** (§5.3). **0 parameters either way.**
2. **→ arch-inf / D-SEL — every future `graft_*` lever inherits a named confound.** S3's, S5's and
   S6's parameters are trained **exclusively** by `loss_rcls`. A null from any of them is confounded
   between *no information* and *an objective that cannot use it* until the objective question is
   settled in training. ⛔ This does not re-open their existing verdicts (§7.2).
3. **→ eval-tools / ops — the `C-permuted-features` instrument is subtly invalid on z-scored
   features** (§3.3) and is used by E-S1-0 and anything that copies it. The corrected
   survivor-restricted version is implemented in `stack/scripts/refc_obj_probe.py`; it should be
   back-ported to `refc_s1_climbout_probe.py`.
4. **→ Benchmarks & Eval — the lead instrument is now independently REPLICATED** (§4.1,
   `raw/C_lead_cross_build.json`): two hosts, two runners, agreement to 2.3e-13 m and on all 881
   window states. E-S1-0's escalation 3 (*"it exists in ONE place"*) can be closed. ⭐ **And it is
   row-alignable to any banked fan dump at 0 re-inference** — every future selection experiment
   should carry the distance-keeping family rather than reporting it n = 0.
5. **→ pre-registration practice — a joint table over k three-sided contrasts needs an enumerated
   product or an explicit catch-all** (§8.1). Third recurrence of this class.

---

## 11. EVIDENCE CLASS

| claim | class |
|---|---|
| all three objectives cost **exactly 0** parameters; `refc_config()` = 104,191,577; `param_breakdown` identical | **MEASURED** — `raw/param_cost.json`, pinned by `stack/tests/test_refc_ce_objective.py` |
| `loss_rcls` is `F.cross_entropy(ce_score, ce_err.argmin(1))` and is the **sole** gradient path to `cons_gate` / `route_to_anchor` / `goal_*` | **MEASURED (source)** — `stack/scripts/refc_train.py`, `ranked_score_loss` + `assert_selection_params_are_alive` (raises, not warns) |
| every PRIMARY, every panel row, every control, every family row | **MEASURED** — `raw/obj_probe_refc-{base,xl}-30k.json` |
| the lead block is row-aligned with the fan bank | **MEASURED** — `C-lead-alignment`, two independent derivations of `v0` agreeing on 881 rows |
| the lead instrument is replicated across two hosts | **MEASURED** — `raw/C_lead_cross_build.json` |
| E-S1-0's `+0.8372 / +0.9147`, `emitted_t0` `0.6253 / 0.5194`, `E-cv` `0.8149 / 0.8158` | **INHERITED** — `S1_CLIMBOUT.md`; **re-derived here** as degenerate controls and reproduced |
| S3's `ρ_deploy 0.2493 / −0.2728` and realized `ΔADE −0.0013 / −0.0024` | **INHERITED** — `S3_DEPLOYABLE.md` §0 |
| S5's *"junction accuracy 0.7613 below the 0.7806 majority baseline"* | **INHERITED** — `PREREG_D-SEL…md` §6.3 |
| *"87.6–89.9 % of the selection gap is longitudinal"* | **INHERITED** — `ESEL_VERDICT.md` §5.1 |
| *"the oracle gap is ~92 % irreducible"* | **INHERITED** — a prose note in `MODEL_REGISTRY.md` §4.1, **not** a results JSON. Load-bearing and flagged |
| whether a **retrain** under `softade` helps | ⛔ **NOT MEASURED** — out of reach by construction, registered as such in advance |
| full-suite `pytest -q` | **MEASURED** — `raw/stack_pytest.txt` |

<!-- MANIFEST -->
