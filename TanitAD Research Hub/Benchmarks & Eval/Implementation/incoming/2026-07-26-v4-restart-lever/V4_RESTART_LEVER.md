# flagship-v4 — the RESTART LEVER diagnosis

**Written:** 2026-07-26 (Europe/Berlin; pods and logs are UTC) · **Host:** eval pod (`tanitad-eval`).
**Question:** the 30 k gate returned NOT-CONTINUE → RESTART (budget 0/2). *A restart is only worth a
GPU-week if it changes a lever.* Which lever, with evidence?

Evidence class stamped on every number:
`MEASURED` (ours + artifact) · `PUBLISHED` (cited) · `INHERITED` (not re-verified) · `ESTIMATED` · `HYPOTHESIS`.

---

## HEADLINE

**λ_plan is EXONERATED. The lever is the trajectory RANKING objective. And a selector-only restart
would still fail this gate card.**

1. **λ_plan did not starve the world model** (Outcome B on the pre-registered rule, §2). The
   hypothesis is also **mis-specified at the mechanism level**: λ_plan is a *gradient scale on the
   trunk↔planner seam*, not a loss weight — `total = wm + planner + fac + sm + strat`, unweighted —
   and λ_plan = 1.0 is documented as *"a strict no-op"*. Over Phase C the WM term fell **−33.4 %** and
   the planner's share of the objective rose **+2.9 %** against a pre-registered ≥ 10 % bar.
2. **The world model was the healthiest thing in the run.** 15 k → 30 k, EVAL-GRADE, same 881
   windows: WM canary **−45.0 %**, fan quality **−21.9 %**, `oracle_in_fan` **−16.7 %** — all better.
   Only **selection** degraded: `sel_gap` **+43.6 %**, `miss_at_2m` **+25.5 %**.
3. **The regression is real and 100 % longitudinal.** The gate said it was "not tested for
   separation"; it is now — paired episode-cluster bootstrap, 881 win / 40 ep: `ade_0_2s`
   **+0.0584 [+0.0043, +0.1179]** SEPARATED, and **larger on the deployable produced-goal surface**
   (+0.0985 [+0.0374, +0.1631]). Along-track **+0.0581 SEPARATED WORSE**, cross-track
   **−0.0257 SEPARATED BETTER**.
4. **Onset:** a monotone `sel_gap` climb from **~step 11 000**, and a persistent level shift at
   **step 26 000** — which coincides with **exactly one** schedule event, the cosine LR falling to
   **4.95 % of peak**. It coincides with **nothing** at λ_plan's saturation (step 8 000).
5. **The selector throws away 0.4093 m — 63.7 % of v4's total 2 s error.** The fan's best is **0.2330 m**; the
   pick is **0.6423 m**; v1 is 0.4271 m. **v4's proposals already beat v1 by ~0.19 m.**
6. **⚠️ I hypothesised the learned longitudinal gate `sel_gate` was the cause and REFUTED IT
   MYSELF.** Direct counterfactual (`sel_gate := 0`, frozen fan): paired Δ **−0.0100 [−0.0191,
   −0.0020]** — the gate **helps**, separated. Logged as **C8**. It narrows the target to
   `refined_logits` and its objective.
7. **RECOMMENDED LEVER (§5.2):** replace the selector's error-blind 1-of-256 CE with a
   **cost-sensitive expected-regret listwise loss** (`TAU = 1.0`, `REFINED_CLS_WEIGHT = 1.0`,
   everything else unchanged). **Falsification test pre-registered in §6: 1–2 GPU-hours, head-only,
   frozen fan — run it before spending a GPU-week.**
8. **⚠️ NECESSARY BUT NOT SUFFICIENT (§5.4):** `wm_canary_ade_2s` = 1.1409 vs bar 0.55 is a
   *world-model* secondary and **nothing in the recommendation touches it**. A selector-only restart
   fails the same card again. This needs a PI decision.
9. **The biggest measured saving is not a lever:** ~**29.5 GPU-h — half the MEASURED 59.0 h run** (ESTIMATED: half the steps at near-constant pace) — went
   into training past the best checkpoint, invisible to every in-loop signal (**C11**). The trainer
   computes `rank_acc` and `frac_sel_2x_worse_than_oracle` every step and **discards both**.

⛔ **Nothing was launched. This is a decision package; the restart is Sayed's call.**

---

## 0. PRE-REGISTRATION — written BEFORE the loss trace was read

> Registered **2026-07-26T12:36:10Z**, before any loss-component value from
> `flagship-v4-fromscratch/train_log.jsonl` was inspected. What *had* been read at registration time:
> the log's **key names only** (schema), `metrics.json`, and the already-published 30 k gate results.
> No `wm` / `planner` / `total` / `lambda_plan` **value** had been looked at.

### 0.1 The hypothesis under test

**H-λ (INHERITED from the brief, sourced to a trainer-log line at step 27,450: "wm 1.911 still
descending under λ_plan = 1.0"):** λ_plan = 1.0 lets the planner loss **starve the world model**,
which would explain both a failing WM canary (1.1409 vs bar ≤ 0.55) and a within-run ADE regression
(15 k 0.5839 → 30 k 0.6423 oracle) as the planner increasingly dominates.

### 0.2 A structural constraint stated in advance (from the launch command, not from the data)

The run was launched with `--lambda-plan sched --phase-a-steps 2000 --phase-b-steps 8000`
(MEASURED, `…/2026-07-23-v4-fromscratch-launch/LAUNCH_CONFIRMED.md` §1). So **λ_plan is a schedule
that saturates**: Phase A `[0, 2000)` λ_plan = 0; Phase B `[2000, 8000)` ramps 0 → 1; Phase C
`[8000, 30000)` λ_plan = **1.0, constant for the final 22,000 steps**.

This forces the hypothesis to split, and I commit to the split now:

- **A-progressive** — λ_plan is a *time-varying* driver: the planner's share of the objective keeps
  growing through Phase C and the WM term stalls as it does. *For this to be true the growth must
  come from the loss MAGNITUDES, since the weight itself is pinned at 1.0.*
- **A-level** — λ_plan's *constant value* is simply too high: the WM term flattens at (or within a
  short window of) the Phase B → C transition and never recovers.
- **B — exonerated** — the WM term is **not** starved: it keeps descending through Phase C and the
  planner share does not progressively grow. The regression then has another cause.

**Note in advance:** the quoted line "still descending under λ_plan = 1.0" is, on its face, already
*evidence against* starvation. Registering that observation here so it cannot be quietly dropped.

### 0.3 Decision rule — committed before measurement

| | criterion (all on the v4 `train_log.jsonl`, Phase C = steps 8000–29999) |
|---|---|
| **A-progressive** | planner share of the total objective **rises** by ≥ 10 relative % from the 8–12 k window to the 26–30 k window, **AND** the `wm` term's descent stalls (26–30 k mean ≥ 8–12 k mean, i.e. no net improvement) |
| **A-level** | `wm` flattens **within 2,000 steps of step 8000** and its 26–30 k mean is ≥ 0.9 × its 8–10 k mean (net descent < 10 % over 22 k steps) |
| **B** | `wm` **continues descending** through Phase C (26–30 k mean < 0.9 × 8–10 k mean) **AND** the planner share does not rise ≥ 10 relative % |

**If B, I do not rescue H-λ.** I name the next-most-likely lever and the cheapest discriminating
experiment, and I say plainly that the brief's leading hypothesis was wrong.

### 0.4 Regression-onset protocol — committed before measurement

The gate regression is `ade_0_2s` **0.5839 @15 k → 0.6423 @30 k** (oracle, MEASURED, gate JSON).
Trainer-log val is *not* quotable as a result (≈10 % optimistic, RETRACTION_LOG C1) — it is used
**only** to locate the *mechanism and its timing*. Onset is declared where the trainer's own
`canary_ade@2s` / `val` series changes sign of trend on a 5-point rolling mean, and I then check
whether that step coincides with a schedule change (LR, λ_plan phase edge, `lam_mult` controller
action, unfreeze, sampler).

### 0.5 What would falsify whatever lever I end up recommending

Written into §6 as a named, cheap, pre-registered experiment with both outcomes committed.

---

*(Sections 1–12 below were written after the measurements. §0 above is unmodified from registration —
including its §0.2 note that the trainer line seeding the hypothesis was already evidence against it,
which is the part it would have been convenient to drop.)*

---

## 1. PROVENANCE — what was measured, and that it is the real run

| artifact | md5 | source |
|---|---|---|
| `v4fs_train_log.jsonl` (661 rows, steps 0→29999) | `7d8bdeb458e064ab36216f9d03214c61` | `tanitad-eval:/workspace/_v4gate/v4fs_train_log.jsonl` |
| same file, as archived on HF | `7d8bdeb458e064ab36216f9d03214c61` | `Sayood/flagship-v4-fromscratch` (gate report §1, PUBLISHED-by-us) |
| local copy used for the analysis | `7d8bdeb458e064ab36216f9d03214c61` | `raw/v4fs_train_log.jsonl` in this folder |

**Three-way md5 identity — MEASURED.** The eval-pod copy, the HF archive copy and the copy analysed
here are the same bytes. This matters because the eval pod has twice been found stale
(62 % stale + missing `corridor.py`, 2026-07-26): *presence is not completeness*, so the file was
hash-matched against an off-pod archive rather than trusted for being there.

**pod2 was NOT touched.** The v4 run trained on pod2, which currently hosts the K-sweep agent. The
training log was obtained from the eval pod's gate-time copy and hash-verified against HF, so no
read, no login and no load was placed on pod2, pod1 or pod3.

---

## 2. THE λ_plan VERDICT — **OUTCOME B. λ_plan IS EXONERATED.**

### 2.1 First: the hypothesis is mis-specified at the mechanism level (code, not data)

**MEASURED by code inspection** (`stack/scripts/train_flagship_v4.py:140`):

```python
total = wm_total + plan_l["loss"] + fac_loss + sm_loss + strat_loss
```

**λ_plan does not appear in the loss at all.** It is not a loss weight. It is a *gradient scale on
the trunk → planner activation seam* (`stack/tanitad/models/flagship_v4.py:211`,
`states_p = grad_scale(states, lambda_plan)`): the forward pass is bit-exact, and only the gradient
flowing **from the planner back into the trunk** is multiplied by λ_plan. The module docstring is
explicit that **`lambda_plan == 1` short-circuits to a strict no-op**.

Two consequences that matter for the restart decision:

1. The stated mechanism — *"λ_plan = 1.0 lets the planner **loss** starve the world model"* — cannot
   occur as stated. The five loss terms are summed **unweighted**; there is no `w_plan` knob in this
   trainer. Turning λ_plan down does not shrink the planner term by one unit.
2. λ_plan = 1.0 is not an aggressive setting; it is **the absence of an intervention**. It is the
   joint-training thesis itself (full planner gradient into the trunk). The knob only exists in the
   *protective* direction, and lowering it moves the arm **toward** the decoupled v1.5 regime.

The trainer line that seeded the hypothesis — *"wm 1.911 still descending under λ_plan = 1.0"* at
step 27,450 — says, on its own terms, that the WM was **not** stalled. That was registered in §0.2 in
advance and it survives.

### 2.2 The decision rule, applied

All values **MEASURED**, `raw/v4fs_train_log.jsonl` (md5 `7d8bdeb4…`), reduced by
`raw/lambda_verdict.json` · `raw/loss_windows.json`.

| pre-registered criterion | required | MEASURED | |
|---|---|---|---|
| **A-progressive**: planner share rises ≥ 10 rel % (8–12 k → 26–30 k) | ≥ +10 % | **+2.88 %** (0.6364 → 0.6547) | ❌ |
| **A-progressive**: WM train term stalls (26–30 k ≥ 8–12 k) | ≥ 2.687 | **1.834** (−31.72 %) | ❌ |
| **A-level**: WM 26–30 k ≥ 0.9 × WM 8–10 k | ≥ 2.514 | **1.834** | ❌ |
| **B**: WM 26–30 k < 0.9 × WM 8–10 k **and** share rise < 10 % | both | **both hold** | ✅ |

> **A-progressive REJECTED · A-level REJECTED · B ACCEPTED.**

### 2.3 The loss-component trace (why B is not a close call)

Per-window means, `total = wm + planner + other`; `other` = factorised CE + plan-smoothness + strategic-scalar.

| steps | total | wm | planner | other | planner share | wm share | plan_ade | oracle_ade |
|---|---|---|---|---|---|---|---|---|
| 0–500 | 246.588 | 17.723 | 15.988 | 212.876 | 6.5 % | 7.2 % | 24.307 | 5.440 |
| 500–2 000 | 13.845 | 6.477 | 7.118 | 0.250 | 51.4 % | 46.8 % | 3.824 | 0.824 |
| 2 000–4 000 | 10.335 | 4.040 | 6.118 | 0.177 | 59.2 % | 39.1 % | 2.150 | 0.518 |
| 4 000–6 000 | 9.115 | 3.312 | 5.657 | 0.146 | 62.1 % | 36.3 % | 1.589 | 0.368 |
| 6 000–8 000 | 8.531 | 3.040 | 5.361 | 0.129 | 62.8 % | 35.6 % | 1.411 | 0.328 |
| **8 000–10 000** *(λ_plan hits 1.0)* | 7.907 | 2.793 | 4.997 | 0.117 | 63.2 % | 35.3 % | 1.110 | 0.289 |
| 12 000–14 000 | 7.408 | 2.522 | 4.777 | 0.109 | 64.5 % | 34.1 % | 0.953 | 0.247 |
| 16 000–18 000 | 6.800 | 2.253 | 4.444 | 0.102 | 65.4 % | 33.1 % | 0.861 | 0.219 |
| 20 000–22 000 | 6.113 | 1.995 | 4.033 | 0.086 | 66.0 % | 32.6 % | 0.666 | 0.182 |
| 24 000–26 000 | 5.777 | 1.927 | 3.760 | 0.090 | 65.1 % | 33.4 % | 0.511 | 0.168 |
| **28 000–30 000** | 5.599 | 1.861 | 3.651 | 0.088 | 65.2 % | 33.2 % | 0.469 | 0.158 |

Read it plainly: **every term goes down and keeps going down.** Over Phase C (8 k → 30 k) `total`
−29.2 %, `wm` −33.4 %, `planner` −26.9 %, `plan_ade` −57.8 %, `oracle_ade` −45.3 %. The planner term
**shrinks faster in absolute terms than the WM term** (−1.35 vs −0.93); its *share* creeps up 2.9
points only because it started as the larger of the two. There is no starvation signature anywhere in
this trace.

**The controller never fired.** `lam_mult` = **1.0 at every one of the 601 logged train steps**, and
all **59** validation rows record `controller_action: "ok"`. So the `--lam-mult-floor 0.25`-vs-`0.15`
drift flagged in the launch note was, as predicted, **inert** — MEASURED, not assumed.

### 2.4 The real signature: the WM was the *best*-improving component, not the starved one

**EVAL-GRADE** (`eval_flagship_v4.py`, same 881 windows / 40 episodes at both steps —
`…/2026-07-25-flagship-v4-midtrain-eval/flagship-v4-fromscratch-15k_v4_diagnostics.json` and
`…/2026-07-26-v4-30k-gate/raw/flagship-v4-fromscratch-30k-oracle_v4_diagnostics.json`). Not the
trainer log.

| quantity | what it measures | 15 k | 30 k | Δ |
|---|---|---|---|---|
| `wm_canary_ade_2s` | **world-model integrity** | 2.0739 | **1.1409** | **−45.0 % BETTER** |
| `dense_headhorizons_oracle_ade` | **best trajectory in the fan** | 0.2401 | **0.1876** | **−21.9 % BETTER** |
| `oracle_in_fan` (4-wp) | fan coverage | 0.2797 | **0.2330** | **−16.7 % BETTER** |
| `seam_norm_ratio_max` | graft clamp health | 0.1244 | 0.1208 | flat, PASS both |
| `dense_headhorizons_sel_gap` | **which one it picks** | 0.2195 | **0.3151** | **+43.6 % WORSE** |
| `miss_at_2m` | selected traj > 2 m off | 0.1691 | **0.2123** | **+25.5 % WORSE** |
| `gate_primary_ade_0_2s` (4-wp, v1-comparable) | net | 0.5839 | **0.6423** | **+10.0 % WORSE** |

> **Every world-model and fan-quality metric improved. Every selection metric degraded. The net
> regression is the second overwhelming the first.**
>
> Arithmetic, on the dense head-horizon surface where `ade = oracle_ade + sel_gap`:
> the fan gave **−0.0525**, the selector took back **+0.0956** — the selector's degradation is
> **1.82×** larger than the fan's improvement, and that difference *is* the regression.

The same decomposition appears independently in the trainer log (`sel_gap@2s` 0.2219 → 0.3158,
+42.3 %; `oracle_ade@2s` 0.2403 → 0.1891, −21.3 %; `canary_ade@2s` 2.0739 → 1.1446, −44.8 %). Two
instruments, one conclusion — but only the eval-grade row above is quotable.

### 2.5 Stated plainly

**The brief's leading hypothesis is wrong, and I am not rescuing it.** λ_plan did not starve the
world model. The world model was the healthiest thing in the run. Restarting with a lower λ_plan
would (a) act on a gradient path that is not the failing one, (b) *slow* the WM's own improvement by
decoupling it from the planner, and (c) leave `miss_at_2m` — one of the two hard FAILs — completely
untouched, because **λ_plan does not scale the head's own learning at all** (at λ_plan = 0 "the
planner heads train at full strength on a trunk they cannot move",
`flagship_v4.py:32-35`). **A λ_plan restart would burn 1 of 2 restarts on the wrong subsystem.**

---

## 3. WHERE THE REGRESSION STARTS — and it is a *real* regression, now with an interval

### 3.1 First, the interval the gate did not compute

The 30 k gate report says of the 15 k → 30 k move: *"Not tested for separation, and stated as such
rather than implied."* It is tested now. **MEASURED**, `raw/v4_paired_15k_30k.json`.

Harness check first, because a delta off unverified dumps is worthless: both window dumps reproduce
their published point estimates **exactly** — 15 k **0.5839** (published 0.5839), 30 k **0.6423**
(published 0.6423) — on **881 windows / 40 episodes** whose `eid` sequence, `gt` and `cv` tensors are
**bit-identical** between the two dumps. Only then were deltas computed.

Estimator: **paired episode-cluster bootstrap** (`taniteval/ci.py`), n_boot 2000, resampling unit =
**episode**. Never `overlapping_holdout_se`. Oriented **30 k − 15 k**, so positive = 30 k is worse.

| paired Δ (30 k − 15 k), goal-ORACLE | Δ | 95 % CI | p(Δ>0) | |
|---|---|---|---|---|
| `ade_0_2s` (4-wp) | **+0.0584** | [+0.0043, +0.1179] | 0.982 | **SEPARATED — real regression** |
| `miss_at_2m` | **+0.0431** | [+0.0045, +0.0864] | 0.986 | **SEPARATED — real regression** |
| **along-track (LONGITUDINAL)** | **+0.0581** | [+0.0157, +0.1053] | **0.999** | **SEPARATED WORSE** |
| **cross-track (LATERAL)** | **−0.0257** | [−0.0405, −0.0088] | 0.003 | **SEPARATED BETTER** |

> **The regression is entirely longitudinal.** Along-track error rose 0.3496 → 0.4077 m (+16.6 %,
> separated). Cross-track error *fell* 0.2065 → 0.1808 m (−12.4 %, separated). The longitudinal share
> of squared error moved **64.34 % → 73.15 %**. v4 got **better at staying in its lane** and **worse
> at getting the speed right**, and the second is bigger than the first.

**Goal-mode control (guards C6).** If the regression only existed on the goal-oracle surface it would
be a goal-head artefact, not a model fact. It is not: on the **PRODUCED** (deployable) surface, same
windows, the regression is **larger** — `ade_0_2s` Δ **+0.0985 [+0.0374, +0.1631]**, p = 1.000, and
`miss_at_2m` Δ **+0.0897 [+0.0510, +0.1339]**, p = 1.000, both SEPARATED. The regression is real on
both goal surfaces and worse on the one that matters.

### 3.2 Onset: a slow trend from ~11 k and a persistent level shift at 26 k

From the trainer's own held-out series — **MECHANISM AND TIMING ONLY, not quotable as a result**
(≈10 % optimistic vs `eval_*.py`, RETRACTION_LOG C1):

- **`sel_gap@2s` begins a monotone climb at ≈ step 11 000** and never reverses: rolling-5 mean
  0.2223 (@11 k) → 0.2399 (@14 k) → 0.2592 (@20 k) → 0.2782 (@24.5 k) → **0.3123 (@27.5 k)**.
- **Raw val-ADE minimum is step 15 500** (0.4564); the 5-point rolling minimum is step 23 000.
- **A persistent level shift lands at step 26 000** and never recovers:

| trainer val | 21 000–25 500 | 26 000–29 500 | Δ |
|---|---|---|---|
| `ade@2s` | 0.4718 | 0.5072 | **+7.5 %** |
| `sel_gap@2s` | 0.2732 | 0.3144 | **+15.1 %** |
| `miss@2m` | 0.1872 | 0.2144 | **+14.5 %** |
| `oracle_ade@2s` | 0.1985 | 0.1928 | −2.9 % *(still improving)* |
| `canary_ade@2s` | 1.3362 | 1.1689 | −12.5 % *(still improving)* |

Even inside the regression the fan and the world model keep getting better. Only the pick degrades.

### 3.3 What schedule change coincides — and what does not

**All MEASURED from `raw/v4fs_train_log.jsonl` / `raw/v4fs_config.json`.**

| candidate | when it happens | coincides with onset? |
|---|---|---|
| **λ_plan phase edge** | saturates at **8 000**, then **1.0 for 22 000 steps** (single unique value) | **NO** — at 8 k val-ADE was still improving hard (0.5359 → 0.4622 by 15 k) |
| **scheduled sampling** (`produced_goal_frac`) | shares λ_plan's boundaries by design (O-17) → also saturates at **8 000** | **NO** — same edge |
| **canary controller** (`lam_mult`) | `lam_mult` = **1.0 at every logged step**; all **59** val rows `controller_action: "ok"` | **NO** — it never fired |
| **encoder unfreeze** | none — encoder 149/149 and predictor 159/159 required grad **from step 0** (`config.json/not_frozen_proof`) | **NO** — nothing to unfreeze |
| **batch / sampler** | `eff_batch` = **64** at every logged step, single unique value | **NO** |
| **LR cosine decay** | peak 1e-4 @ step 2 000; at **step 26 000 lr = 4.95e-6 = 4.95 % of peak**, → 0 by 29 950 | **YES — this is the only schedule event at 26 000** |

> **The 26 k level shift coincides with exactly one thing: the cosine LR entering its final ~5 %.**
> The last 4 000 steps are an annealing tail, and the selector's generalization degrades precisely as
> the model commits hardest to its training-set statistics. The 11 k trend onset coincides with
> **nothing** on the schedule.

### 3.4 The finding this really is: **C11**

Every term of the *training* objective improves monotonically to step 30 000 — `total` −29.2 %,
`wm` −33.4 %, `planner` −26.9 %, `plan_ade` −57.8 %, `oracle_ade` −45.3 % over Phase C — while the
**held-out selected trajectory gets worse, separated**. `total` and `wm` each rise in exactly one
2 000-step window out of eleven, and that window is **26 000–28 000**: the training loss barely
registers the event that costs the gate.

**This is `training-loss-is-not-a-generalization-guard` (C11) in its clean form**, and it is the
reason the run was allowed to spend 15 000 steps and ~29 GPU-hours travelling in the wrong direction:
the only in-loop signals a human was watching (`total`, `wm`, `plan_ade`) all looked healthy
throughout.

---

## 4. LOCALISING THE FAILING SUBSYSTEM — and one hypothesis of mine REFUTED

### 4.1 The selector, sized

**MEASURED**, `raw/v4_selgate_ablation.json` (`diag.as_trained`) and the gate diagnostics, same
881 windows / 40 episodes, goal-ORACLE, 4-waypoint surface:

| | value |
|---|---|
| best trajectory available **in the fan** (`oracle_in_fan`, 4-wp) | **0.2330 m** |
| trajectory the selector actually **picks** (`ade_0_2s`) | **0.6423 m** |
| **thrown away by ranking** | **0.4093 m — 63.7 % of v4's total 2 s error** |
| v1 reference (the bar) | 0.4271 m |

> **v4's fan already contains trajectories that beat deployed v1 by ~0.19 m. The selector converts
> that into a 0.22 m loss.** The proposals are not the problem. The pick is.
>
> Combined with §2.4 (WM canary −45 %, fan −21.9 %, both improving) and §3.1 (regression 100 %
> longitudinal), the failing subsystem is **trajectory ranking**, and specifically its longitudinal
> discrimination.

### 4.2 A code-level fact worth having on the record

`FlagshipV15Head.select()` (`flagship_v15.py:451-466`) scores the refined fan as

```
score  = refined_logits + sel_gate * ( -|v_term(i) - v_goal| )
v_goal = clamp(vt_speed, v0 ± sel_accel_max * 2.0 s)
```

and `_goal_inputs` sets **`vt_speed = v0`** (`train_flagship_v4.py:172`) — in the trainer *and* in
the eval harness, which imports the same function. With `sel_accel_max = 2.5` (MEASURED,
`raw/v4fs_config.json`) the reachable clamp is ±5.0 m/s, so the clamp is a strict no-op and
**`v_goal == v0` exactly, always.**

So the term documented as *"target-speed-aware… exactly the signal that would rank the decelerating
proposal first"* is, in every v4 run to date, a **pure constant-velocity preference** — it penalises
any proposal whose terminal speed differs from the current speed. **The `vt_speed` channel is never
connected to a target speed.** That is a real design/wiring gap and it is stated here as a fact
about the code, MEASURED, independent of whether it explains anything.

Its learned scale grew from its `sel_gate_init = 0.0`: **0.1101 @ 15 k → 0.1580 @ 30 k**
(**MEASURED**, `raw/v4_sel_gate.json`, read directly out of both checkpoints — the trainer computes
`sel_gate` into `out["telemetry"]` every step but the row-writer's fixed key list drops it, so the
checkpoints are the only surviving record).

### 4.3 ⚠️ I hypothesised that this was the lever. **A direct counterfactual REFUTES it.**

The growth of `sel_gate` (+43.5 %) and the growth of `sel_gap` (+43.6 %) over the same interval is a
striking coincidence, and the constant-velocity mechanism would neatly explain a purely longitudinal
regression. **So I tested it instead of writing it up.**

**Experiment** (`sel_gate_ablation.py`, `raw/v4_selgate_ablation.json`): reload the 30 k checkpoint,
set `head.sel_gate := 0`, and re-run the planner path on the identical 881 windows. `sel_gate` enters
*after* `decoder()`, so the trunk, the fan and `refined_logits` are untouched — this isolates the
term exactly. Harness check: the as-trained arm reproduces **0.6423** against the published
**0.6423** ✅.

| paired Δ (as-trained − sel_gate 0), oriented so **positive = the gate HURTS** | Δ | 95 % CI | p(Δ>0) | |
|---|---|---|---|---|
| `ade_0_2s` | **−0.0100** | [−0.0191, −0.0020] | 0.009 | **SEPARATED — the gate HELPS** |
| along-track (longitudinal) | **−0.0086** | [−0.0162, −0.0019] | 0.009 | **SEPARATED — the gate HELPS** |
| cross-track (lateral) | −0.0007 | [−0.0024, +0.0008] | 0.186 | not separated (as expected) |
| `miss_at_2m` | −0.0068 | [−0.0159, +0.0011] | 0.044 | not separated |

Point estimates: as-trained **0.6423**, `sel_gate = 0` **0.6523**. The term changes the pick on only
**6.6 %** of windows (`frac_windows_selection_changed`), and where it acts it is **net beneficial**.

> **VERDICT: `sel_gate` is NOT the lever. Removing it makes the arm slightly WORSE, separated.**
> My mechanism is refuted by the cheapest experiment that could refute it. Logged to
> `RETRACTION_LOG` as **C8 (premature root cause)** — see §7.
>
> **What the refutation buys.** Zeroing the *only explicit longitudinal term* in the score moves
> ADE by 0.0100 m out of a 0.4093 m ranking deficit — **2.4 %**. So the selection failure is **not**
> in the auxiliary gate; it is in the **base score `refined_logits` itself** and in the objective
> that trains it. That is a genuinely narrowed target, and it is narrowed by measurement.

---

## 5. THE LEVER RECOMMENDATION

### 5.1 What the evidence forces, and what it does not

| subsystem | state at 30 k | evidence | is it the lever? |
|---|---|---|---|
| world model (trunk) | canary **improving** 15.674 → 2.074 @15 k → **1.1409** @30 k; train `wm` −33.4 % over Phase C; never starved | MEASURED §2 | **No — but see 5.4, it still FAILS its bar** |
| λ_plan coupling | constant 1.0 for 22 000 steps; controller never fired; not a loss weight | MEASURED §2 | **No — exonerated** |
| proposal fan | `oracle_in_fan` **0.2330 m**, improving −16.7 % over 15 k→30 k | MEASURED §4.1 | **No — the fan is better than v1** |
| `sel_gate` (longitudinal gate) | net **beneficial**, −0.0100 m, touches 6.6 % of windows | MEASURED §4.3 | **No — refuted by counterfactual** |
| **trajectory RANKING (`refined_logits` + its objective)** | **gives away 0.4093 m = 63.7 % of total 2 s error**; `sel_gap` +43.6 %, monotone from ~11 k; regression 100 % longitudinal | MEASURED §3, §4 | **YES** |

### 5.2 The recommended change — the RANKING OBJECTIVE, with a specific setting

`v15_losses` (`flagship_v15.py:563-618`) trains the selector with

```python
r_star    = fan_err.argmin(dim=1)                             # 1-of-256 hard label
loss_rcls = F.cross_entropy(out["sel_score"].float(), r_star.detach())
loss      = 1.0*loss_traj + 1.0*loss_cls + 1.0*loss_rcls      # all three weights 1.0
```

Two structural properties of this objective (**MEASURED by reading the code**; the causal
attribution to the regression is **HYPOTHESIS**, and §6 is how it dies if wrong):

1. **The label is the argmin over 256 near-tied candidates.** Among near-ties it is effectively
   label noise, and CE spends capacity separating candidates that are equally good.
2. **CE is error-blind.** Picking the 2nd-best (about 0.24 m) and picking a catastrophic one
   (about 5 m) incur the *same* loss for the same probability mass on `r_star`. The loss has **no
   notion of how bad a wrong pick is** — while the gate metric is exactly the metres given up. That
   is a plain objective/metric mismatch, and it has the shape of what we measure: the `planner` term
   falls 26.9 % while the *selected* trajectory gets separated-worse.

**RECOMMENDED SETTING — replace the hard CE with a cost-sensitive (expected-regret) listwise loss,
keeping `REFINED_CLS_WEIGHT = 1.0` and changing nothing else:**

```python
regret    = fan_err - fan_err.min(dim=1, keepdim=True).values      # [B, N] metres, >= 0
p         = torch.softmax(out["sel_score"].float() / TAU, dim=1)   # TAU = 1.0
loss_rcls = (p * regret.detach()).sum(dim=1).mean()                # EXPECTED METRES GIVEN UP
```

- `TAU = 1.0` (plain softmax over `sel_score`); `REFINED_CLS_WEIGHT = 1.0` unchanged.
- `loss_traj` and `loss_cls` **unchanged** — they train the fan, and the fan is healthy. Do not touch
  what is working.
- This minimises the quantity the gate scores, is smooth under near-ties, needs no noisy-argmin
  label, and penalises catastrophic picks in proportion to their cost.

**Attributability: this must be the ONLY planner-side change in the restart.** In particular do
**not** also rewire `vt_speed` in the same arm (5.3). Two levers in one restart, from a budget of
two restarts, is how a program loses both.

### 5.3 A second, SEPARATE finding — do not bundle it

`vt_speed` is hard-wired to `v0` (§4.2), so the "target-speed-aware" selection channel has never been
connected to a target speed in any v4 run. That is a real wiring gap and it should be fixed — but the
counterfactual says the term *as wired* is currently net beneficial, so rewiring it is an
**unmeasured** change of unknown sign. It belongs in its own arm, after the ranking lever is scored.
**Escalated, not bundled.**

### 5.4 THE DECISIVE CAVEAT: a selector-only restart would STILL FAIL this gate

`wm_canary_ade_2s` = **1.1409** vs bar **0.55** is a **world-model** secondary. **Nothing in §5.2
touches it.** The ranking lever attacks `ade_0_2s` and `miss_at_2m` — where the fan's 0.2330 m best
means a >2 m miss is almost always a ranking failure — but the canary is a different subsystem.

The canary's own trajectory (MEASURED, `raw/lambda_verdict.json`): 8–10 k mean **1.4900** → 26–30 k
mean **1.1689**, i.e. **−21.6 % over 20 000 steps**, while the *train* `wm` term fell 34.3 % over the
same span. It is descending, but slowly, and it must fall **2.07x** to clear the bar. Per the
program's own rule I will **not** extrapolate that. The honest statement is:

> **UNRESOLVED.** At the Phase-C rate the canary does not reach 0.55 inside a 30 k budget, and
> whether the fix is more steps, a different WM recipe, or a λ_plan < 1 protection is **not
> determined by any measurement I made**. It needs its own cheapest-discriminating experiment — the
> natural one being the design's own `--lambda-plan 0` control, which is documented to reproduce the
> frozen-trunk v1.5 regime byte-identically and therefore isolates the planner gradient's effect on
> the WM.

**So the decision package is: the ranking lever is NECESSARY but NOT SUFFICIENT for this gate card.**
Either the restart carries a WM-side answer as well, or the card's `wm_canary` bar is re-registered
on evidence — and re-registering a bar after seeing the number is exactly what GATE_PROTOCOL §0.3
forbids, so that route needs Sayed's explicit decision, not an agent's.

### 5.5 The largest MEASURED saving is not a lever at all — it is stopping

The run's best primary was at **15 k** (0.5839) and it then spent **15 000 more steps** becoming
separated-worse, over a MEASURED **212 544.6 s = 59.0 h** total wallclock — so roughly **29.5
GPU-hours, half the run, went into travelling in the wrong direction**, invisible to every in-loop
signal (§3.4, C11). A held-out **selection-quality** early-stop / checkpoint-selection rule
(`sel_gap` or `miss_at_2m` — *not* `total`/`wm`) is worth more wall-clock than any throughput lever
in §6.

**The instrumentation gap that made this invisible, and that the restart must close:** `v15_losses`
already computes `rank_acc` and `frac_sel_2x_worse_than_oracle` — the exact diagnostics for this
failure — and `select()` computes `sel_gate` and `sel_pen_span`, **every single step**. The trainer's
row-writer (`train_flagship_v4.py:693-703`) filters each row down to a fixed key list and **throws
all four away**; none of them ever reached `train_log.jsonl`. This is a **log-only, zero-parity-effect**
fix and it should land *before* the restart, not after.

---

## 6. WHAT WOULD FALSIFY THIS — pre-registered, both outcomes committed

**The cheapest discriminating experiment does not need a 30 k run.** Fine-tune **only the selection
head** off the existing 30 k checkpoint — trunk, decoder and fan frozen, so the fan is provably
identical and only the ranking changes — for **2 000 steps** under the §5.2 regret loss. Score on the
standard 881 windows / 40 episodes, goal-ORACLE, **paired episode-cluster bootstrap** against the
as-trained 30 k selector on the same windows. **ESTIMATED cost 1–2 GPU-hours, against a GPU-week for
a restart.**

| pre-registered outcome | reading |
|---|---|
| **CONFIRM** — paired Δ `ade_0_2s` (regret − CE) at most **−0.10 m**, CI excluding 0, on the same fan | the ranking objective is the lever; the restart is justified, and its landing zone is bounded below by the fan's 0.2330 m and above by today's 0.6423 m |
| **PARTIAL** — Δ separated but weaker than −0.10 m | ranking is a real but insufficient lever; do **not** spend a restart on it alone |
| **REFUTE** — Δ not separated from 0, or positive | **the ranking objective is NOT the lever.** The deficit would then not be recoverable by re-scoring a fixed fan, which means `refined_logits` lacks the *information* to rank — a conditioning/architecture problem, not a loss problem — and the next probe is what the score is conditioned on, not how it is trained. **I commit to that reading now.** |

**Why this test is honest:** it runs on a **frozen fan**, so it cannot be rescued by the fan
improving; it measures exactly the quantity §4.1 says is being thrown away; and it can come back
negative. Two of my hypotheses have already died this session (§2.5, §4.3) and this one is set up to
die the same way if it is wrong.

**A falsifier for the diagnosis as a whole:** if a re-scored frozen fan cannot get below about
0.43 m (v1's 0.4271), then "the fan already contains v1-beating trajectories" (§4.1) is true only in
an oracle sense no realisable ranker can reach, and the selection framing is worth materially less
than §4.1 implies.

---

## 7. GATE_PROTOCOL §0.7 — THE VOID SECONDARY, PRINTED

> A suppressed criterion that is not printed is indistinguishable from one that passed. So it is
> printed here too, in a document that reopens the gate's conclusions.

```
[VOID]  nonav_route_beats_majority          original bar: >= 1
        STATUS       : VOID_BY_CONSTRUCTION
        ADJUDICATION : INSTRUMENT-FAIL -- NEVER MODEL-FAIL
        AUTHORITY    : GATE_PROTOCOL 0.7 (card `secondary_void`)
        IN KILL SET  : NO -- structurally excluded; it did NOT contribute to the
                       30k verdict and it does NOT contribute to this diagnosis
        MEASURED     : value null on this checkpoint
        WHY          : the strategic route TARGET is a lookup of the route INPUT
                       (refb_labels.route_target = _NAV_TO_ROUTE[nav_cmd]), true on
                       100.00% of CE-eligible windows under v1, v2 AND v2.1 -- no
                       labeler swap can break it. It measures the label bug, not
                       the model. Second, independent reason on THIS checkpoint:
                       v4's goal_head is a GoalScalarHead regressing only
                       continuous scalars -- no route classifier exists to score.
        RE-ARMS WHEN : an arm trains with real route supervision
                       (--labels-v21 --v2-route-from-vision)
```

**Nothing in this document is adjudicated on it, in either direction.**

---

## 8. "Any chance to accelerate the training of v2 corpus?" — analysis for a FUTURE run

⛔ **pod1 was not touched at all for this section** — not even read-only. Its micro-batch setting is
the arm's own instrument, and the brief forbids modifying it. Everything below is either the brief's
own figures (INHERITED, marked) or re-read by me from the **raw JSON** of the existing benchmark
(`…/Production & Optimization/Implementation/incoming/2026-07-26-v2-throughput/results_interleaved.json`,
`results_memory.json`) — not from that report's prose.

### 8.1 Reconciling the two readings of pod1's pace

| source | step | `step_s`/50 | `data_s`/50 | data share |
|---|---|---|---|---|
| throughput benchmark (raw log read, 4 samples) | 7 250–7 700 | 10.62–10.71 | 1.23–1.28 | 11.6–12.1 % |
| the brief (INHERITED, not re-verified by me) | ~11 150 | **10.88** | **1.34** | **12.3 %** |

Consistent, with a slow drift: **+2.2 % on step time and +0.6 pt on data share over ~3 450 steps.**
Both figures are `/50` reductions of the accumulated `--log-every` counters — the trap that has
produced false "430 s/step" alarms is correctly handled in both.

**ETA on the brief's numbers (ESTIMATED, straight-line):** 18 850 steps remaining × 10.88 s ≈
**205 000 s ≈ 57 h**.

### 8.2 The contention caveat, quantified from the primary artifact

`results_interleaved.json` `meta.host` = **`1e0bac0df88a`** — the **eval pod**, which is *shared*.
That is why one lever flipped sign: `gc_off_8x8` read **1.16× (faster)** while co-tenants ran at
85–90 % util, and **0.95× (slower)** once load settled — and the first reading implied gradient
checkpointing was *faster* than no checkpointing at matched micro-batch, which is physically
impossible. **That is the tell to look for.** Pooled over the stable session the same arm lands at
`speedup_vs_gc_on_16x4` = **0.8865** (MEASURED, `summary.gc_off_8x8`).

The consequence for the one lever that might win:

| arm | pooled median s/step | pooled speedup | per-repeat spread | peak alloc |
|---|---|---|---|---|
| `gc_on_16x4` (production) | 15.8323 | 1.0000 | — | 13.028 GiB |
| **`gc_on_32x2`** | 12.6916 | **1.2475** | **1.18× / 1.40× / 0.84×** | 24.118 GiB |
| `gc_off_8x8` | 17.8599 | 0.8865 | no win in 3/3 | 23.429 GiB |

> **A pooled 1.2475× whose repeats span 0.84–1.40 is not a certified lever.** On a 57 h remainder
> that range is "saves 11 h" to "costs 11 h". **Treat single-run lever readings on a shared pod as
> uncertified — that is the whole lesson of the sign flip.**

### 8.3 What would accelerate a FUTURE run — and how the levers stack

The useful correction to naive lever-stacking is that these three attack **different** parts of the
wall clock, so their ceilings do not simply multiply and must not be summed either:

| lever | what it attacks | ceiling | status |
|---|---|---|---|
| `--workers` / `--v2-lru` | the **loader stall** — `data_s`, 12.3 % of wall clock, *outside* compute | **1.14×** hard Amdahl cap (10.88 → 9.54 s/step) | parity-clean; safe; cannot pay for a restart alone |
| `pin_memory=True` + `non_blocking=True` | the **H2D copy**, which sits *inside* the 9.54 s of compute — `aten::copy_` is the single largest CUDA-time op at **23 %** at ~66.1 MB/sample | unknown share is recoverable; **2-line change** | **the best code-level lead; UNCERTIFIED — must be A/B'd, never assumed** |
| `--batch-size 32 --accum 2` | the **compute shape** itself | pooled 1.2475×, repeats 0.84–1.40 | **needs ~30 min on an EXCLUSIVE GPU before anyone spends a GPU-day** |
| `torch.compile` | compute | unmeasured on this stack | plausible on a ViT; Linux-only (dev box has no Triton); next run, not mid-flight |
| `--grad-checkpoint off` | compute | **DEAD** | 42.35 GiB at micro-batch 16 vs pod1's 47.99 GiB — ~5.6 GiB margin, unattended, for ~57 h. And at the only micro-batch where it fits (8) it lost 3/3 repeats |

**ESTIMATED best case if all three land and are roughly multiplicative: ~1.14 × ~1.05 × ~1.25 ≈
1.50×** — i.e. a 90 h 30 k run becomes ~60 h. **Every factor in that product is ESTIMATED or
uncertified; the multiplicativity assumption is itself unverified.** It is a planning figure, not a
result, and it must not authorise anything on its own.

### 8.4 Two things that are NOT free, and one that dwarfs all of the above

- **`batch × accum` is not a free knob.** `ego_decorr_loss` (v2 LEVER B / H25) and SIGReg are both
  estimated **across the micro-batch**; for uncorrelated inputs the decorr penalty carries an
  `E[r²] ≈ 1/b` noise floor, so micro-batch 8 **doubles** it versus production 16. And `16 × 4` is a
  deliberate cross-arm constant — v4.2's registry entry says *"matching v1"* verbatim. Changing it
  for `v2corpus` alone would give the arm that exists to isolate the **v2 corpus** a second
  distinguishing lever. **Change it only at a fresh launch, where the new value becomes that arm's
  own constant.**
- **Restart timing.** `ckpt.pt` auto-resumes, but `--ckpt-every 1000` means an arbitrary restart
  discards up to 1 000 steps ≈ **3.0 h**. Any restart must fire in the minutes after a
  `[ckpt] saved at step N` line. (Also: the resume path restores model/optimizer/step but **not** the
  DataLoader iterator, and `torch.manual_seed` re-runs at startup, so the shuffle order restarts from
  the top.)
- **And the lever that dwarfs all of them: don't train past the best checkpoint.** §5.5 measures
  **~29.5 GPU-hours — half of v4's MEASURED 59.0 h (212 544.6 s) run — spent getting separated-worse** (the hours figure is ESTIMATED: half the steps at near-constant pace). No throughput lever
  in this table is worth 1.5×; a held-out selection-quality stopping rule was worth **2.0×** on
  time-to-best-checkpoint for v4, and it costs nothing to implement.

**Bottom line for Sayed's question:** for the *running* `v2corpus` arm, nothing safe is being left on
the table — let it finish (~57 h). For the *next* launch: take `--workers 16` and the `pin_memory` /
`non_blocking` pair, **certify `32 × 2` on an exclusive GPU first**, and spend the real effort on a
stopping rule rather than on s/step.

---

## 9. FOR `Project Steering/RETRACTION_LOG.md` — with root-cause classes

Three entries. The first two are mine, from this session.

### R-1 — **C8 (premature root cause)** — MINE, caught by my own experiment

> **Claimed:** the learned longitudinal selection gate `sel_gate` (grown 0.1101 @15 k → 0.1580 @30 k,
> +43.5 %) was driving v4's selection regression, because `_goal_inputs` hard-wires `vt_speed = v0`
> so the term reduces to a pure constant-velocity preference — and `sel_gap` grew +43.6 % over the
> same interval.
> **Retracted:** a direct counterfactual (`sel_gate := 0`, same checkpoint, same 881 windows, frozen
> fan) gives paired Δ `ade_0_2s` = **−0.0100 [−0.0191, −0.0020]**, p = 0.009 — **the gate HELPS,
> separated.** It changes the pick on only 6.6 % of windows.
> **Root cause of the error:** two percentages that matched to 0.1 pt (+43.5 % and +43.6 %) over a
> single interval, plus a mechanism that would have explained a purely longitudinal regression. Both
> facts were true; the causal link was not. **A coincidence of two growth rates over ONE interval is
> n = 1 evidence, and a mechanism that "would explain everything" is exactly when to run the
> counterfactual rather than write it up.**
> **Artifact:** `…/2026-07-26-v4-restart-lever/raw/v4_selgate_ablation.json`.

### R-2 — **C6 (confounded comparison)** — MINE, caught before it was written down

> **Nearly claimed:** that the eval harness feeds the selector a counterfactual `vt_speed` (the card
> records `vt_speed` being "overwritten with the observed v0" at `eval_flagship_v4.py:78`), so a
> growing `sel_gate` would do progressively more damage at eval than in training.
> **Falsified at source:** `train_flagship_v4.py:172` sets `kw["vt_speed"] = v0` in `_goal_inputs`,
> which the **trainer and the eval harness both call**. There is **no train/eval mismatch**. The real
> fact is broader and is retained as a finding (§4.2): `vt_speed` is wired to `v0` *everywhere*, so
> the "target-speed-aware" channel has never carried a target speed in any v4 run.
> **Root cause of the near-error:** reading a train/eval asymmetry out of a doc-comment that
> described only the eval side. **Absence of a mention on one side is not asymmetry — check the
> other caller (rule 2).**

### R-3 — **C11 (training-loss-is-not-a-generalization-guard)** — a program-level entry

> **The pattern:** every term of v4's training objective improved monotonically to step 30 000
> (`total` −29.2 %, `wm` −33.4 %, `planner` −26.9 %, `plan_ade` −57.8 %, `oracle_ade` −45.3 % over
> Phase C) while the held-out **selected** trajectory got separated-worse (paired Δ `ade_0_2s`
> **+0.0584 [+0.0043, +0.1179]**; `miss_at_2m` **+0.0431 [+0.0045, +0.0864]**). ~29.5 GPU-hours went
> into a regression that no in-loop signal showed.
> **Aggravating instrument defect:** `v15_losses` computes `rank_acc` and
> `frac_sel_2x_worse_than_oracle`, and `select()` computes `sel_gate` / `sel_pen_span`, **every
> step** — and `train_flagship_v4.py:693-703` filters all four out of the written row. **The exact
> diagnostics for the failure were computed 601 times and discarded 601 times.**
> **Rule this earns:** a trainer may not write a row that omits a held-out *selection-quality*
> statistic, and no long run is launched without a stopping rule on one.

*(Also worth a line in the log, though it is the 30 k gate's finding rather than mine: the gate's
own report states the 15 k → 30 k move was "not tested for separation". It is tested in §3.1 now and
it **is** separated — on both goal surfaces. The convention of publishing an untested direction-of-
travel alongside tested numbers invites it being read as though it had an interval.)*

---

## 10. HOST STATE — the eval pod, verified rather than assumed

The pod has burned this program twice (62 % stale tree; missing `corridor.py`; a second stale tree
found only by an import-order accident; a silent `scp` truncation against 3.0 GB free). So:

**Sync verification (MEASURED, md5, CRLF-normalised where the repo file is CRLF):**

| module | pod `/root/v4eval/stack` | repo | verdict |
|---|---|---|---|
| `tanitad/models/flagship_v15.py` | `88889b1b0203bf936e6083ccf10c68e4` | `88889b1b…` (LF-normalised) | **IDENTICAL** |
| `scripts/train_flagship_v4.py` | `d6ba03e4c8fefc79e3e39b5d60544d3e` | `d6ba03e4…` | **IDENTICAL** |
| `scripts/eval_flagship_v4.py` | `aafe3975817e27ef7714499643aae6ff` | `04cb7c44…` (LF-normalised) | **DIFFERS — diffed, not guessed** |
| `taniteval/taniteval/ci.py` | `ef925f06febd20a99f5901491fcf75cb` | `ef925f06…` | **IDENTICAL** |

The `eval_flagship_v4.py` difference is **17 diff lines, all cosmetic**: the repo version hoists two
f-string suffixes out of a replacement field so the file parses on Python 3.11 (PEP 701 is 3.12+).
It changes a `method` description string and **no numeric path**. The pod runs 3.12/3.13, so both
parse. **Verified by reading the diff — this is exactly the check that a bare `ls` or a bare md5
mismatch would have failed in opposite directions.**

**Absence-is-not-absence, again (rule 2).** `/workspace/TanitAD/stack` **does not exist** on this
pod. Probing a second path found **six** copies of `eval_flagship_v4.py` (sizes 33 292 / 33 954 /
34 695 / 44 695 B) and **two distinct sizes** of `flagship_v15.py` across
`/root/TanitAD/stack`, `/root/v4eval/stack`, `/root/v4eval/_sync/stack`, `/root/v2bench/stack`,
`/root/_bak_20260726_produced_goal/`, `/workspace/geocalib_work/stack`. **`/root/v4eval/stack` is the
verified-current tree** and is the one used here. A future agent that picks a tree by `find`-order
will get a stale one.

**Disk — `dd`, never `df`:** fresh 300 MiB `oflag=direct` write at **380 MB/s**, no error. Memory
503 GiB total, 387 GiB free. GPU: A40, **0 MiB used** before my work.

**Left running, untouched** (reported rather than acted on — they are other agents' processes and
killing them was not needed for this diagnosis):

| PID | age | RSS | what |
|---|---|---|---|
| 1487931 / 1487926 / 1487782 / 1487785 | ~3.36 d | ~0.48 GiB each | stale AlpaSim `multiprocessing-fork` workers |
| 1487279 | ~3.36 d | 9 MiB | their `multiprocessing.resource_tracker` |
| 75731 | ~7.6 d | 3.5 MiB | a `bash -c … until ! pgrep -f 'taniteval.generalization'` waiter loop that will never exit |

≈1.9 GiB RSS and 5 idle processes. **⚠️ Note for whoever clears them: PID 75731 is itself a
`pgrep -f` loop — kill by explicit PID only.** They hold **0 MiB of GPU**, so they did not affect
anything measured here.

**Not touched at all: pod1, pod2, pod3.** The v4 run trained on pod2 (K-sweep agent) and its
training log lives there; I obtained the log from the eval pod's gate-time copy and hash-verified it
against the HF archive instead of reading pod2. No `stack/` file was modified, so `pytest -q` is
unaffected.

---

## 11. DELIVERABLE MANIFEST

Repo root: `G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD`.
Folder: `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-26-v4-restart-lever/`.
**All STAGED (`git add`). Nothing committed. Nothing pushed. No branch switched.**

| artifact | where it lives | also exists elsewhere? |
|---|---|---|
| `V4_RESTART_LEVER.md` — this document | `repo:…/2026-07-26-v4-restart-lever/V4_RESTART_LEVER.md` | **repo only** |
| `raw/v4fs_train_log.jsonl` — the v4 run's 661-row training log, md5 `7d8bdeb4…` | `repo:…/raw/` | ✅ also `tanitad-eval:/workspace/_v4gate/v4fs_train_log.jsonl` and `HF Sayood/flagship-v4-fromscratch` — **three-way md5 identity verified** |
| `raw/v4fs_config.json`, `raw/v4fs_metrics.json` | `repo:…/raw/` | ✅ also `tanitad-eval:/workspace/_v4gate/flagship-v4-fromscratch-30k/` + HF |
| `raw/lambda_verdict.json` — the pre-registered decision rule, applied | `repo:…/raw/` | **repo only** |
| `raw/loss_windows.json` — per-window loss-component means | `repo:…/raw/` | **repo only** |
| `raw/v4_paired_15k_30k.json` — the 15 k→30 k paired regression interval + axis split + goal-mode control | `repo:…/raw/` | also `tanitad-eval:/root/v4_paired_15k_30k.json` |
| `raw/v4_sel_gate.json` — learned `sel_gate` read out of both checkpoints | `repo:…/raw/` | also `tanitad-eval:/root/v4_sel_gate.json` |
| `raw/v4_selgate_ablation.json` — **the counterfactual that refuted my hypothesis** | `repo:…/raw/` | also `tanitad-eval:/root/v4_selgate_ablation.json` |
| `code/paired_15k_30k.py` | `repo:…/code/` | also `tanitad-eval:/root/paired_15k_30k.py` |
| `code/sel_gate_probe.py` | `repo:…/code/` | also `tanitad-eval:/root/sel_gate_probe.py` |
| `code/sel_gate_ablation.py` | `repo:…/code/` | also `tanitad-eval:/root/sel_gate_ablation.py` |

**Exists in only ONE place — flagged per the operating standard:**

| artifact | sole location | worth rescuing? |
|---|---|---|
| `windows_flagship-v4-fromscratch-30k-oracle.pt` / `…-produced.pt` (382 KB each) — the per-window dumps every paired interval in §3.1 is computed from | `tanitad-eval:/workspace/_v4gate/results/` | **YES — small, and without them no 30 k paired number can be recomputed.** Not staged here because they belong with the 30 k gate deliverable, not this one. **Escalated in §12.** |
| `v4_selgate_ablation_windows.pt` — per-window dumps of both ablation arms | `tanitad-eval:/root/` | low priority; regenerable in ~4 min from the staged script |
| v4-fromscratch milestone checkpoints 5 k / 10 k / 15 k / 20 k | **pod2 only**, except 15 k (also `tanitad-eval:/workspace/models/…`) and 20 k (also HF) | 5 k and 10 k are **single-disk on pod2** |

---

## 12. ESCALATIONS — decisions and integrations that must not sit in a file

1. **⛔ I did not launch anything. The restart decision is Sayed's.** This is a decision package.
2. **A selector-only restart would still FAIL the 30 k card** on `wm_canary_ade_2s` (1.1409 vs 0.55)
   — §5.4. The restart needs a WM-side answer *as well*, or the card needs a PI-level decision.
   **This is the single most important line in the document.**
3. **Land the log-only row-writer fix BEFORE any restart** (`train_flagship_v4.py:693-703`): emit
   `rank_acc`, `frac_sel_2x_worse_than_oracle`, `sel_gate`, `sel_pen_span`, and a held-out
   `sel_gap`. Zero parity effect; without it the next run is as blind as this one (§5.5, R-3).
4. **Run the §6 head-only falsification test (1–2 GPU-h) before spending a GPU-week.** Both outcomes
   are pre-registered and useful.
5. **Rescue the 30 k per-window dumps off the eval pod** — 764 KB total, currently single-disk, and
   every paired interval in the 30 k gate and in §3.1 depends on them.
6. **Three entries for `Project Steering/RETRACTION_LOG.md`** (§9): C8 and C6 are mine; C11 is
   program-level. I did not edit the log myself — appending to an append-only steering file is the
   orchestrator's call.
7. **`vt_speed` is wired to `v0` in every v4 run** (§4.2). Real wiring gap, **unknown sign**, own
   arm, do not bundle.
