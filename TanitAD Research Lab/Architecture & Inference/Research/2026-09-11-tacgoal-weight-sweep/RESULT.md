# `--w-tac-goal` MEASURED, not guessed — the weight sweep for refcv6 arm C

**Package** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-11-tacgoal-weight-sweep/`
**Register rows** `D-TACGOAL-2` (the budget) · `E-REFCV6-C` (arm C's hypothesis — **untouched**)
**Pre-registration** `PREREG_TACGOAL_WEIGHT_SWEEP.md`, in this package, fixed while the control
arm was at step 180/400 and **no weighted arm had a `metrics.jsonl` at all**
**Base** `agent/arch-inf-20260803` @ `d221843`, worked in `D:\Projects\TanitAD`

⛔ **THE VALUE IS NOT SETTLED HERE.** PI queue item 10 asks for a `MANEUVER_WEIGHT` budget
decision. This package converts it from a guess into a measurement and hands over **a curve and
a recommendation**. Ratification remains the PI's.

⛔ **NO TIER STAMP, NO ADE, NO FOUR-FAMILY EVAL TABLE — deliberate, not an omission.** Nothing
here is an eval; every number is a training-loss diagnostic read from the trainer's own
`metrics.jsonl`. A tier stamp on a non-eval would be a false strength claim. ⛔ **Nothing in
this package licenses a statement about driving performance, traffic-light behaviour, or
tactical goal quality** — that is `H-TACGOAL-1`, pre-registered elsewhere and not touched.

---

## 1. ⛔ THE BRIEF NAMED A RIG THAT CANNOT RUN THIS EXPERIMENT — and the correction makes the answer STRONGER, not weaker

The brief specified *"a short sweep on the tiny rig (`v7-tiny` is the real trainer at ~19 M
params)"*. **`--w-tac-goal` does not exist in that trainer.**

| file | `--w-tac-goal` | `tac_goal_tok_head` | `tac_goal_loss` | CONTROL `add_argument` | CONTROL `def main` |
|---|---|---|---|---|---|
| `stack/scripts/train_v6_staged.py` (v7-tiny) | **0** | **0** | **0** | **232** | 1 |
| `stack/scripts/refc_v3_train.py` (refcv6 arm C) | **12** | **25** | **5** | **111** | 1 |

The controls are what make the zeros admissible: a bare `0` from a file that could not be read
is a claim about the read, not about the content. Banked: `raw/rig_census.json`.

⇒ A v7-tiny sweep would have measured **nothing**. The sweep therefore ran on **arm C's own
trainer**, short-stepped instead of small-modelled:

| | |
|---|---|
| params | **108,257,502** — `config.json:param_breakdown.total`, the real refcv6 scale |
| head | `tac_goal_tok_head` **11,286** — the live `d_tac 512` width, a literal match with `REFCV6_ARM_FACTS.md` §5 |
| corpus | `physicalai-b1-w120-256x640cyl` + `s2_labels_v7.2_train.jsonl.gz`, **4,572** records, md5 `0ff902130ce76886b8a925eceed9e3a5`, schema `s2-geom-v7`, vocab `v7` — **refcv5-v2's own corpus** |
| parity | ⭐ **the trainer verified it itself, in its own log**: `[parity] v3 v2-cache: physicalai-b1-w120-256x640cyl v2 VERIFIED — 4713 clips, clip sha256 e8bfb98e06eb… matches the committed manifest (skip-hash f09e44db)`. `config.json` agrees: `v2_parity.parity = True`, `checked = True`. ⛔ **No episode re-selected; the skip-hash is the canonical `f09e44db`.** |
| held-out eval | built and run once at step 400: `[v3] held-out eval: 6 episodes -> 1032 windows, 8 fixed batches every 400 steps`. ⚠️ **6 episodes is small** — reported as a secondary reading, never as the primary. |
| steps | **400**, `--warmup 20` + cosine to 0 (5 %, matching refcv5-v2's 2000/40284 **fraction**) |
| seed | **0** on every arm |

⭐ **This is why the usual "a tiny-rig result may not transfer to 108 M" caveat does not apply
in the form the brief expected.** The architecture, the parameter count, the head width, the
corpus, the label join and the loss combination are **identical to refcv6 arm C, because this is
arm C's trainer**. What does not transfer is the **step budget** — 400 is **1.0 %** of 40,284 —
so this measures *early-training displacement*, not final quality. §8 states that in full.

## 2. Thor was idle when the sweep launched — by a probe that cannot match itself

⛔ My first idleness probe **lied**, in exactly the documented way: `grep -c -E "$T"` on
`ps -eo args` matched **the grep process's own argv**, reporting `1 trainer, 1 supervisor` on an
empty box. The fix is to break the loop between the emitted and the searched token:

1. one ssh writes `ps -eo pid,etime,rss,args` to a **file** (443 lines);
2. a **second, independent** ssh greps that file with patterns assembled at runtime
   (`A=$(echo tr)$(echo ainer)`), so the searched string never appears in either command line.

Result immediately before launch: **`ZZ A=0 B=0 C=0 D=0 ZZ`** — zero trainers, zero supervisors,
zero python processes, zero `refc_v3_train` — with `nvidia-smi` reading **0 %**. The sweep was
launched into an empty box and nothing else was on it.

## 3. The instrument — additive, default-OFF, and it reads the gradient BEFORE the clip

`stack/scripts/refc_v3_train.py`: **+93 lines, 0 removed** (HEAD 5,829 → 5,922; measured by line
sets, **zero** lines present only in HEAD, so the new file is a strict **superset**, not a
revert).

| # | site | what |
|---|---|---|
| E1 | module level | `_grad_probe_row(model, names, log_every_hit)` — per-module `sum(|grad|)`, `n_grad_none`, `n_tensors`, `n_params`, plus `torch.cuda.max_memory_allocated()` |
| E2 | `build_parser` | `--grad-probe-modules` (default **`""`**) |
| E3 | `train()` | parse the name list once; print it so the record says what was probed |
| E4 | the loop | the call, **between `losses["loss"].backward()` and `clip_grad_norm_`** |
| E5 | the log row | `row.update(_gp_row)` **AFTER** the rounding comprehension |

Three choices are load-bearing and each has a mutation arm in §5:

* ⛔ **Before the clip.** `clip_grad_norm_` rescales every gradient by a **global** factor, so a
  post-clip reading confounds *this head's own gradient* with *how big everything else's was
  that step*.
* ⛔ **After the rounding.** The log row rounds scalars to 5 dp. `round(1e-8, 5)` **is `0.0`** —
  a real gradient at a small weight would be logged as the very defect being measured.
* ⛔ **`found` per module.** A mistyped path reads `found = 0.0` and emits no `grad_abs_sum`, so
  an unreadable module and an unreached one never look the same.
* ⛔ **Empty name list ⇒ empty dict, in the HELPER.** A run that does not pass the flag computes
  nothing and its `metrics.jsonl` schema is unchanged. *(This was originally guaranteed only by
  the call site; the guard test caught it and it was moved into the helper — a guarantee that
  lives only in a caller erodes.)*

⭐ On Thor `gp_cuda_max_mem_gb` uses `torch.cuda.max_memory_allocated()` and nothing else:
`mem_get_info`, `free`/`tegrastats` and `VmRSS` all lie there, in both directions.

## 4. The smoke — the probe reads the defect, and the control reads the no-information value exactly

Two 6-step arms before the sweep, identical but for the flag (`raw/smoke_*_metrics.jsonl`):

| arm | `tac_goal_tok_head` `n_params` | `grad_abs_sum` @ steps 2 / 4 / 6 | `n_grad_none` | `tac_goal` term |
|---|---|---|---|---|
| `w0_control` (flag absent) | **11,286** | **0.0 / 0.0 / 0.0** — exactly | **2 of 2** | absent |
| `w005` (`--w-tac-goal 0.05`) | **11,286** | 1.7399985510855913 / 2.3706056494265795 / 3.9256331585347652 | **0 of 2** | 0.64711 / 0.63368 / 1.4082 |

⭐ **Two extra no-information controls came free and both read exactly 0.0 in BOTH arms** —
`core.decoder.offset_head` (**6,160**) and `scorer.goal_point` (**1,026**), each
`n_grad_none 2 of 2`. That independently reproduces `REFCV6_ARM_FACTS.md` §5 at the live width,
on a running trainer rather than a static probe, and it is what makes the head's non-zero
reading mean something: the same probe, in the same step, reads zero where zero is the truth.

⭐ **An analytic cross-check, not a re-run of the producer's arithmetic.** At step 2 the control's
total loss is `60.0083` and the weighted arm's is `60.04051`. The reported term is `0.64711`, so
`60.0083 + 0.05 × 0.64711 = 60.04066` against a measured `60.04051` (Δ `1.5e-4`, and the two arms
had already taken one differing optimizer step). The ON total is the OFF total plus the weighted
term and nothing else moved.

## 5. The guard, and the SEVEN mutations that must go RED

`stack/tests/test_grad_probe_tacgoal.py` — **11 tests, all passing**. Every expectation is a
**literal**: `0.0`, `2`, `15`, `30.0`, `11286`, `1e-8`. None is an expression over the code
under test.

`raw/mutation_proof.py` reintroduces real defects one at a time, runs the named test, then
restores the trainer and **md5-verifies the restore** (`3fa2f4e2e06e51d7f4944a6f3e3f6ab3` before
and after every arm). Log: `raw/mutation_proof.log`.

| mutation | the defect it reintroduces | verdict |
|---|---|---|
| **M1** | the row merge disabled (`if _gp_row and False:`) | **RED** |
| **M1b** | the row merge deleted entirely | **RED** |
| **M2** | the probe moved AFTER `clip_grad_norm_` | **RED** |
| **M3** | the probe counts `p.numel()` instead of `|grad|` | **RED** |
| **M6** | `.abs()` dropped — a cancelling gradient reads as none | **RED** |
| **M4** | `--w-tac-goal` default raised off `0.0` | **RED** |
| **M5** | a missing module reads like an unreached one | **RED** |

⭐⭐ **TWO OF MY OWN GUARDS WERE MEASURED INERT ON THE FIRST RUN, AND I AM REPORTING THAT RATHER
THAN THE FIXED VERSION ONLY.** The first pass read **`INERT GUARDS = 2`**:

* **M1 was GREEN.** The test searched for the substring `row.update(_gp_row)` and checked its
  ORDER. Disabling the merge with `and False` leaves the substring present and the order right —
  the probe never reaches `metrics.jsonl` and the test says nothing. Fixed by pinning the exact
  two-line block, guard condition included.
* **M3 was GREEN**, and this one is the sharper lesson. `test_unreached_head_reads_exactly_zero`
  and its `> 0.0` partner **both** passed against a probe that counts parameters instead of
  gradients: the unreached head's grads are `None`, so the mutated line never runs and it still
  sums to `0.0`; the reached head still sums to something positive. **`> 0.0` is exactly the
  kind of check that shares the defect it checks for.** Fixed with an **analytic target**:
  `reached` is `Linear(4, 3)` = **15** parameters in **2** tensors, every gradient element set
  to exactly `-2.0`, so `sum(|grad|)` **is 30.0** — while a `numel` probe reads `15.0` and an
  `abs`-less probe reads `-30.0`. One literal separates all three.

## 6. Suite evidence — the causal set, and a baseline that was MEASURED

Only `refc_v3_train.py` changed, so a test that does not import it cannot be affected. **All 32
files that import it were run**: **575 passed, 10 failed** (plus the 11 new).

⛔ The 10 failures are **PRE-EXISTING AT HEAD, measured not argued**: HEAD's own trainer was
installed in the worktree, the two files re-run, and the trainer restored and md5-verified —
**`10 failed, 11 passed`, the identical count**. They live in
`test_refc_v3_agent_gt_head_drop.py` / `test_refc_v3_agent_gt_reaches_forward.py` and every one
says *"the mutation anchor … matched 0 times … the deliberate-regression arm is DISARMED"*: the
agent-GT source anchors have drifted, and nothing in that code was touched here. ⚠️ **Those
guards currently prove nothing and that is a real open item**, flagged separately. Two further
collection errors (`test_refa_v1_dk_hook.py`, `test_metric_decode_refusal.py`) contain **0**
references to the trainer or the probe. Detail: `raw/suite_baseline.txt`.

---

## 7. ⭐ THE CURVE — nine arms, and the zero control is the point no scaling can produce

Nine 400-step arms on one binary (md5 `3575e1f4ab2851a51d282ff7e4f44d73`), ~28 min each,
`seed 0`, one lever moved. `grad_abs_sum` is on `tac_goal_tok_head`, **unrounded**, read after
`backward()` and before the clip. Every arm reports `n_params` **11,286** — a literal match.

| arm | `--w-tac-goal` | `grad` first → last | `n_grad_none` | `traj` tail-5 | `cuda_max_mem_gb` |
|---|---|---|---|---|---|
| `A_w0` | **absent** | **0.0 → 0.0** | **2 of 2** | 1.400468 | 36.1223 |
| `A_w0_rep` | **absent** | **0.0 → 0.0** | **2 of 2** | 1.316024 | 36.1223 |
| `A_w0_rep2` | **absent** | **0.0 → 0.0** | **2 of 2** | 1.327934 | 36.1223 |
| `B_w0p005` | 0.005 | 0.4885 → 9.0499 | 0 of 2 | 1.430578 | 36.1224 |
| `C_w0p05` | 0.05 | 4.7535 → 86.994 | 0 of 2 | 1.422654 | 36.1224 |
| `F_w0p15` | 0.15 | 14.541 → 312.50 | 0 of 2 | 1.530936 | 36.1224 |
| `D_w0p5` | 0.5 | 42.262 → 803.67 | 0 of 2 | 1.385472 | 36.1224 |
| `E_w5p0` | 5.0 | 223.93 → 5450.6 | 0 of 2 | 1.429518 | 36.1224 |

⭐ **Does the zero control read exactly 0.0? YES — and not once but 60 times.** All three
zero-weight arms read `grad_abs_sum` **exactly `0.0` at every one of their 20 logged steps**
(`sorted(set(...)) == [0.0]`), with `n_grad_none` **2 of 2**. The term is *absent from the
graph*, not multiplied by zero — and no scaling of a non-zero gradient can produce an exact
zero, which is what makes it a control rather than a small number.

⭐ **Two further no-information controls read exactly 0.0 in ALL NINE arms** —
`core.decoder.offset_head` (6,160) and `scorer.goal_point` (1,026). They reproduce
`REFCV6_ARM_FACTS.md` §5 on a *running trainer at the live width*, and they are what stops
"non-zero" from being an artifact of the probe.

⭐ **THE CURVE IS LINEAR IN w, WHICH IS AN ANALYTIC PREDICTION AND NOT A FIT.** For a linearly
weighted term the gradient on the head scales as `w`, so a ×10 weight should give ×10 gradient.
Measured, on the last logged step: `9.05 → 86.99` = **×9.61**, `86.99 → 803.67` = **×9.24**,
`803.67 → 5450.6` = **×6.78** (the trunk co-adapts at the top end, which is the expected
direction). Across the full **1000×** span `0.005 → 5.0` the gradient moves **×602**.

⚠️ **Memory is flat.** `torch.cuda.max_memory_allocated()` reads **36.1223 GB** for every
zero-weight arm and **36.1224 GB** for every weighted one — a difference of **~0.1 MB**, which
is the 11,286 parameters' gradient buffer. ⛔ **The weight costs no memory; it is not a capacity
decision.**

## 8. ⛔ THE BUDGET, AND THE VERDICT ARM BY ARM AGAINST THE PRE-REGISTERED RULE

The weight is meaningless without the term it multiplies. `tac_goal` is a pos-weighted
multi-label BCE and `traj` is an L1 — what is comparable is the **contribution each makes to the
scalar that is differentiated**. The references are literals from `refc_train.py`:
`TRAJ_WEIGHT = 1.0` (:76), `MANEUVER_WEIGHT = 0.1` (:80), `LAT_WEIGHT = LON_WEIGHT = 0.05`
(:90-91), combined by `refc_v3_train.py:2495-2496` as `0.025 x (lat + lat_tac + lon + lon_tac)`
— the **total tactical-auxiliary budget, as actually spent over its four surfaces**.

| arm | `w` | `w x tac_goal` | ÷ tactical budget | ÷ primary | ÷ total loss | `traj` Δ vs `A_w0` | × floor | (a) | (b) | (c) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `B_w0p005` | 0.005 | 0.00286 | **0.022** | 0.20 % | 0.015 % | +0.03011 | 0.357 | PASS | PASS | PASS | ✅ **ADMISSIBLE** |
| `C_w0p05` | 0.05 | 0.02815 | **0.190** | 1.98 % | 0.147 % | +0.02219 | 0.263 | PASS | PASS | PASS | ✅ **ADMISSIBLE** |
| `F_w0p15` | 0.15 | 0.09131 | **0.713** | 5.96 % | 0.446 % | +0.13047 | **1.545** | PASS | ⛔ **FAIL** | PASS | ❌ |
| `D_w0p5` | 0.5 | 0.26134 | **1.708** | 18.9 % | 1.42 % | −0.01500 | 0.178 | PASS | PASS | ⛔ **FAIL** | ❌ |
| `E_w5p0` | 5.0 | 2.97846 | **20.20** | 208 % | 13.2 % | +0.02905 | 0.344 | PASS | PASS | ⛔ **FAIL** | ❌ |

⛔ **`F_w0p15` FAILED its pre-registered criterion (b)** — `|Δ traj|` = 0.13047 against a floor
of 0.084444, i.e. **1.545×**. That is reported as a FAIL, and the floor is **not** re-estimated
to rescue it: adding the third control left `traj`'s floor at **exactly 0.084444**, because
`A_w0_rep2`'s 1.327934 landed *between* the other two.

### 8.1 ⚠️ WHY CRITERION (b) DOES NOT DISCRIMINATE ON THIS RIG — an argument from dose, not from convenience

| `w` | 0.005 | 0.05 | **0.15** | 0.5 | 5.0 |
|---|---|---|---|---|---|
| `traj` Δ | +0.03011 | +0.02219 | **+0.13047** | **−0.01500** | +0.02905 |
| × floor | 0.357 | 0.263 | **1.545** | 0.178 | 0.344 |

⛔ **Over a 1000× dose range the perturbation is NOT MONOTONE, and the largest weight perturbs
LESS than the smallest.** `w = 5.0` spends **20× the entire tactical budget** and drives the
gradient to 5,450 — and moves `traj` by +0.029, *less* than `w = 0.005` does. If the weight were
causing the movement, a 1000× change in dose would show it. ⇒ criterion (b) is reading the rig's
run-to-run noise.

⭐ **AND THE PANEL MEASURED THAT NOISE DIRECTLY.** `A_w0` / `A_w0_rep` / `A_w0_rep2` are the
SAME flags, the SAME seed, run three times with **zero levers moved** — and they differ at
**every one of the 20 logged steps**. This rig is **not deterministic at a fixed seed**. Over
13 tracked terms the single-pair floor was too tight and the third control widened it by up to
**3.73×** (`lon` 0.4655 → 1.7344; `loss` 2.24×; `goal2s_err_m` 1.97×; `goal_tac` 1.92×). ⛔ With
a one-pair floor, **16 of 39 arm×term cells (41 %) crossed it**, and the crossings were
non-monotone in dose — the `A0b_replicate` result the programme already recorded, reproduced
here at 108 M.

⇒ ⭐ **Criterion (c) is what actually sets the weight, because it is ARITHMETIC rather than
statistical.** `w x tac_goal` against the trainer's own tactical contribution is computed from
logged quantities and needs no noise model at all.

### 8.2 The criterion-(c) ceiling — five independent estimates that agree

`tac_goal` is nearly independent of `w` (0.523 – 0.609 across a 1000× span), so `w x term` is
close to linear and every arm gives its own estimate of the crossing point `w* = w / ratio`:

| arm | `B_w0p005` | `C_w0p05` | `F_w0p15` | `D_w0p5` | `E_w5p0` | |
|---|---|---|---|---|---|---|
| `w*` | 0.2292 | 0.2633 | 0.2103 | 0.2927 | 0.2476 | **mean 0.2486, range [0.2103, 0.2927]** |

⚠️ **Evidence class: EXTRAPOLATED between tested weights, not MEASURED.** Five arms agreeing
within ±18 % is what makes it quotable at all; the sweep measures admissibility only at the
weights it actually ran.

## 9. ⛔⛔ THE FINDING THAT OUTRANKS THE WEIGHT — refcv6 arm C, AS SPECIFIED, WOULD HAVE DIED AT STEP 500 OF 40,284

Every weighted arm **exited 1**, and it was not the training:

```
400 of 400 steps completed, "ckpt step 400 -> ckpt.pt" written, THEN the step-400
held-out eval raised:
  [v3] --w-tac-goal > 0 but the batch carries no `tac_goal_y`/`tac_goal_w` ...
```

`tac_goal_targets` was set in **exactly one place** — the TRAIN dataset. The eval dataset `e_ds`
is a different object and never got it. The refusal is **correct** (a silently-skipped term with
the weight stamped in `config.json` is the `w_agent` defect), but `SystemExit` derives from
`BaseException`, so the eval block's `except Exception` — which exists precisely so *"an
in-training eval must never take the run down"* — **cannot catch it**.

⛔ **refcv6 arm C is refcv5-v2's argv plus `--w-tac-goal`, and that argv carries
`--eval-every 500`.** ⇒ arm C would have died at **step 500 of 40,284**, roughly **35 GPU-minutes
into a ~47 GPU-hour run**, every time, with the cause reading like a flag error.

⭐ **The precedent was already in the same function, ten lines away.** The `--bev-aux` block sets
`e_ds.bev_spec` for exactly this reason and its comment spells out the `SystemExit` mechanism in
full. The tac-goal term never got the same line.

**Fixed** (`stack/scripts/refc_v3_train.py`, +25 lines, 0 removed): `e_ds.tac_goal_targets` and
`e_ds.tac_goal_negatives`, gated on the same weight, inside the `if args.eval_labels:` branch.
⛔ `pos_weight` and `class_mask` are **not** refitted there — they live on the model and were fit
on the TRAIN split; fitting them on the split they are about to score would be the
tuning-on-the-scored-data defect.

⭐ **Proven end to end, not asserted.** `EVALFIX_smoke` — the *same* `--w-tac-goal 0.05`
configuration that exited 1 on every sweep arm — runs on the fixed trainer
(`2c5821aaa036388a85103ed64b8878a5`) and **exits 0 with an eval row**:

```
eval_tac_goal                0.55854
eval_tac_goal_n_supervised  41.625      <- non-zero: the term is really computed
eval_tac_goal_n_pos         11.375
```

⚠️ `n_supervised 41.625` matters: a term that returned `0.0` with `n_supervised == 0` would be a
vacuous pass. Guard: `stack/tests/test_tacgoal_eval_target_wiring.py` (8 tests) with four
mutation arms (§10).

⚠️ **What this costs the sweep, stated plainly:** the held-out eval row exists for the three
zero-weight controls and is **absent for every weighted arm**, because they ran on the pre-fix
binary. The secondary held-out comparison between control and weighted arms is therefore **not
available**, and the verdicts above rest on the pre-registered training-side statistic — which
is what criterion (b) was written against. Recovering it needs a re-run, not a re-analysis.

## 10. The guards — 11 mutations, all RED, and two of my own were caught INERT

`stack/tests/test_grad_probe_tacgoal.py` (11 tests) and
`stack/tests/test_tacgoal_eval_target_wiring.py` (8 tests) — **19 passing**. Every expectation is
a literal: `0.0`, `2`, `15`, `30.0`, `11286`, `22`, `1e-8`.

`raw/mutation_proof.py` reintroduces each defect, runs the named test, restores the trainer and
**md5-verifies the restore** (`3dcbfa555896422e81d61eb5fa3dde18` before and after every arm).

| | mutation | verdict |
|---|---|---|
| M1 / M1b | the probe's row merge disabled / deleted | **RED** |
| M2 | the probe moved AFTER `clip_grad_norm_` | **RED** |
| M3 | the probe counts `p.numel()` instead of the gradient | **RED** |
| M6 | `.abs()` dropped | **RED** |
| M4 | `--w-tac-goal` default raised off `0.0` | **RED** |
| M5 | a missing module reads like an unreached one | **RED** |
| **M7** | **the EVAL dataset loses its goal-set target — the arm that died on Thor** | **RED** |
| M8 | the eval wiring loses its weight gate | **RED** |
| M9 | the sibling `--bev-aux` eval wiring dropped | **RED** |
| M10 | the eval handler widened to `BaseException` | **RED** |

**`INERT GUARDS = 0`.**

⭐⭐ **TWO OF MY OWN GUARDS WERE MEASURED INERT ON THE FIRST RUN, and the second is the more
useful lesson.** M3 — a probe that counts *parameters* instead of gradients — left BOTH "the
unreached head reads exactly 0.0" and its `> 0.0` partner GREEN: the unreached head's grads are
`None`, so the mutated line never ran; the reached head still summed to something positive.
**`> 0.0` is precisely a check that shares the defect it checks for.** The fix was an **analytic
target**: `Linear(4, 3)` is 15 parameters in 2 tensors, every gradient element set to exactly
`-2.0`, so `sum(|grad|)` **is 30.0** — while a `numel` probe reads `15.0` and an `abs`-less probe
reads `-30.0`. One literal separates all three.

## 11. ⭐ THE RECOMMENDATION — and it is a recommendation, not a decision

> **Recommend `--w-tac-goal 0.05` for refcv6 arm C.**

It is the **largest weight admissible on all three pre-registered criteria**, and it is the only
recommendation that does not depend on which noise floor you believe: it passes under the
one-pair floor and under the three-control floor, and it sits **4.97× below** the extrapolated
criterion-(c) ceiling of ≈0.25.

What 0.05 buys and costs, in the arithmetic the PI asked for:

* the head **trains** — `grad_abs_sum` **4.75 → 86.99**, against **exactly 0.0** for all 40,284
  steps of refcv5-v2;
* it adds **0.0282** to a loss of **19.1** — **0.147 %** of the total, **1.98 %** of the primary;
* against `MANEUVER_WEIGHT = 0.1` as actually spent, it is **19.0 %** — ⛔ **nothing is taken
  from it.** The flag is structurally additive; no existing term is rebalanced. The `/3.0`
  re-split remains **a separate arm, not a tweak**, because it would change `lat`/`lon` pressure
  and break pairing with the banked arm;
* it costs **~0.1 MB** of GPU memory.

⚠️ **If the PI wants the most signal rather than the most caution**, the measured band runs to
the criterion-(c) ceiling ≈**0.25**; `0.15` was tested and clears (a) and (c) but **failed (b) as
pre-registered**, so it is not recommended on this evidence. ⚠️ **If the PI wants the term
strictly inside `MANEUVER_WEIGHT` rather than beside it**, that is the `/3.0` re-split and it is
a different arm.

### 11.1 What transfers to refcv6 arm C, and what does not

| transfers | does NOT transfer |
|---|---|
| **the architecture and scale** — 108,257,502 params; this *is* arm C's trainer | ⛔ **the step budget.** 400 steps is **1.0 %** of 40,284. This is early-training displacement, not final quality. |
| **the head width** — 11,286, the live `d_tac 512` | ⛔ **anything about whether the head HELPS.** That is `H-TACGOAL-1`, with its own bar, its own eval and its four families. |
| **the corpus and label join** — b1 + v7.2, parity `f09e44db` VERIFIED by the trainer | ⚠️ **training-seed variance.** All three controls share seed 0, so the floor is the rig's *nondeterminism* — a LOWER BOUND on the seed floor. A different-seed control is the named next lever. |
| **the budget arithmetic** — `w x term` vs the tactical contribution is a property of the loss, not of the step count | ⚠️ **the held-out reading for weighted arms** — lost to the eval defect, recoverable only by re-running. |

## 12. Manifest

| artifact | where it lives | in ≥2 places? |
|---|---|---|
| `stack/scripts/refc_v3_train.py` (+93 probe, +25 eval fix, **0 removed**) | `repo:` staged · `thor:/home/nvidia/tacgoal_sweep/trainer_FIXED.py` | yes |
| `stack/tests/test_grad_probe_tacgoal.py` (11 tests) | `repo:` staged | repo only |
| `stack/tests/test_tacgoal_eval_target_wiring.py` (8 tests) | `repo:` staged | repo only |
| `PREREG_TACGOAL_WEIGHT_SWEEP.md` | `repo:` staged | repo only |
| `RESULT.md` (this file) | `repo:` staged | repo only |
| 9 × `raw/<arm>_metrics.jsonl` + 2 smoke | `repo:` staged · `thor:/home/nvidia/experiments/tacgoal-wsweep/<arm>/` | yes |
| `raw/A_w0_config.json`, `raw/C_w0p05_config.json`, `raw/A_w0_summary.json`, `raw/A_w0_rep_summary.json`, `raw/A_w0_train.log`, `raw/C_w0p05_train.stderr.log` | `repo:` staged · `thor:` same dirs | yes |
| `raw/BUDGET_ANALYSIS.json`, `raw/analyze_budget.py`, `raw/analyze_sweep.py` | `repo:` staged | repo only |
| `raw/mutation_proof.py` + `.log` (11 arms, 0 inert) | `repo:` staged | repo only |
| `raw/rig_census.json`, `raw/suite_baseline.txt` | `repo:` staged | repo only |
| `raw/run_tacgoal_sweep.sh`, `raw/run_tacgoal_followup.sh` | `repo:` staged · `thor:/home/nvidia/tacgoal_sweep/` | yes |
| `raw/sweep_progress.log`, `raw/followup_progress.log` | `repo:` staged · `thor:` | yes |
| checkpoints (`ckpt.pt`, **1,299,555,361 B × 9**) | ⚠️ **`thor:` ONLY** — deliberately not banked; they are 400-step artifacts of a calibration, not results | **no** |

⚠️ **DISK, because a full quota has killed a flagship mid-checkpoint here before.** The run left **13.5 GB** on Thor — `/home/nvidia/experiments/tacgoal-wsweep` **11 G** (9 arms, each with a 1.3 GB `ckpt.pt`) and `/home/nvidia/experiments/tacgoal-smoke` **2.5 G**. The box is now at **90 % (94 G free)**, up from 88 % before the sweep. ⛔ Nothing was deleted: every metric, config and log is banked in this package, so the checkpoints are reclaimable at any time, but that is a shared-box call rather than mine to take unilaterally.

⚠️ **Trainer currency.** The nine panel arms ran on md5 `3575e1f4ab2851a51d282ff7e4f44d73`; the
repo file is `2c5821aaa036388a85103ed64b8878a5`. The delta is **the eval-wiring fix plus a 3-line
helper guard that only fires when the probe's name list is EMPTY** — the sweep always passed
three names, so no panel number is affected. Both md5s are written into
`raw/followup_progress.log` by the runner itself.

## 13. ⛔ Escalation — three things that need a decision or an owner, not a doc

1. **PI queue item 10 is ready to close.** The weight has a measured recommendation
   (`--w-tac-goal 0.05`) and a curve. ⛔ **It is not closed by this package** — ratification is
   the PI's.
2. ⛔ **`D-TACGOAL-EVAL-1` must reach any launch path for arm C before it spends a pod.** The fix
   is staged and tested, and Thor's sweep tree carries it — but a launch from a stale checkout
   would still die at step 500. **Grep-verify `e_ds.tac_goal_targets` is present on the
   launching box before arm C starts.**
3. ⚠️ **10 tests are RED at HEAD and their deliberate-regression arms are DISARMED**
   (`test_refc_v3_agent_gt_head_drop.py`, `test_refc_v3_agent_gt_reaches_forward.py`). MEASURED
   identical against HEAD's own trainer, so they predate this work — but every one reports *"the
   mutation anchor matched 0 times … this test would pass without proving anything"*. They
   currently guard nothing. Filed as a separate task.
