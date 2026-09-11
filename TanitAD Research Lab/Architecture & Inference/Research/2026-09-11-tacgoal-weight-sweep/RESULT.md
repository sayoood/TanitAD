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

*(§7 the grad-versus-weight curve · §8 the budget and the recommendation · §9 the manifest —
written from the completed sweep.)*
