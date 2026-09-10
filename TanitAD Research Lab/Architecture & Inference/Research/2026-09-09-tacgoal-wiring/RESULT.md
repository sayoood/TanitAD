# `tac_goal_loss` IS WIRED — defaulted OFF, bit-identical when off, and the guard now goes RED

**Package** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-09-tacgoal-wiring/`
**Register row** `D-TACGOAL-TRAINER-SEAM-OPEN` — **CLOSED at the code layer, NOT YET BOUGHT by any arm**
**Base** `agent/arch-inf-20260803` @ `ad68b3d853dd886051d4f0ac7a841116ff89119e`
**Tier** n/a — this is a wiring + gradient-reach result, not a driving-capability claim. No ADE number is
quoted here and none is implied; nothing in this package licenses a capability statement about the
22 tactical goal tokens.

---

## 0. The four answers, up front

| question | answer | evidence |
|---|---|---|
| Does `tac_goal_tok_head` receive gradient when the flag is on? | **YES.** `grad_abs_sum = 3.602122873067856`, `verdict GRADIENT_REACHES`, `n_grad_none 0 of 2` | MEASURED, `raw/probe_patched.json` |
| Is the OFF path bit-identical to HEAD? | **YES.** `0 of 167` parameter tensors differ after a real optimizer step; total loss bitwise equal at `19.020389556884766` | MEASURED, `raw/poststep_compare.txt` |
| Did a deliberately-corrupted control prove that comparison can FAIL? | **YES, twice.** OFF stayed at `19.020389556884766` under both corruptions; ON moved to `31.563495635986328` (C1) and `701.5768432617188` (C2) | MEASURED, `raw/probe_corrupt.json` |
| Does the head appear in the effective-weights stamp? | **YES — for the first time.** `n_terms 6 → 7`; the row reads `status TRAINS`, `builds_graph true` when on | MEASURED, `raw/stamp_compare.txt` |
| What went RED in the mutation proof? | Two independent mutations: the loss call removed → **2 tests RED**; the dataset emission removed → **3 tests RED** | MEASURED, `raw/pytest_guard_RED_mutation.log`, `raw/pytest_RED_dataset_mutation.log` |

---

## 1. The defect, re-established from source rather than inherited

⛔ Evidence class **MEASURED (mine)**, not INHERITED. The brief stated the mechanism; I re-read the
trainer at HEAD and confirmed it independently before touching anything.

```
CONTROL: def count in refc_v3_train.py     43     (non-zero ⇒ the file was served)
tac_goal_loss occurrences                   0
TacGoalEmitter occurrences                  0
tanitad.refs.tac_goal_head imported         no
tac_goal_tok_head occurrences              13     (flag, cfg pin, seam stamp, param breakdown)
```

The control matters: on the G: mount an empty grep is indistinguishable from a failed read, so a bare
`0` proves nothing. `43` alongside the zeros is what makes the zeros admissible.

⇒ `refc_v3.py:1320` wrote `cache["tac_goal_logits"]` into a dict nothing read. The head was built,
stamped, forward-run, and had **no gradient path**. Only the trainer call was missing.

---

## 2. What was built — 8 additive edits, 0 changed defaults

All in `stack/scripts/refc_v3_train.py` (5,380 → 5,583 lines, **purely additive**):

| # | site | what |
|---|---|---|
| E1 | `REFC_WEIGHT_GATES` | the `w_tac_goal` row — the row this head has never had |
| E2 | `V3Dataset` class attrs | `tac_goal_targets` (default `False`), `tac_goal_negatives` |
| E3 | `V3Dataset.__getitem__` | emits `tac_goal_y` / `tac_goal_w` `[22]` beside `lat_v7`/`lon_v7` |
| E4 | `compute_losses_v3` | **the call that was missing** — `tac_goal_loss` on `out["tac_goal_logits"]` |
| E5 | `train()` | `model._w_tac_goal`, `_tac_goal_pos_weight`, `_tac_goal_class_mask` |
| E6 | `train()` label join | fits `pos_weight` and `class_mask` **on the loaded split** |
| E7 | `build_parser` | `--w-tac-goal` (default **0.0**), `--tac-goal-negatives` (default `measured`) |
| E8 | imports | `tanitad.refs.tac_goal_head as _tac_goal_head` |

⛔ **Nothing was taken from `MANEUVER_WEIGHT`. No existing loss was rebalanced. No existing default
was changed. No training was launched.** Spending the budget is the PI's decision (queue item 10) and
was explicitly out of scope.

### 2.1 Why the OFF path is an ABSENCE, not a multiplication by zero

```python
_w_tg = float(getattr(model, "_w_tac_goal", 0.0) or 0.0)
if _w_tg > 0.0:
    ...                       # the term is constructed only inside this branch
```

At `--w-tac-goal 0` the term never enters the autograd graph. That is what makes the OFF arm
bit-identical to the pre-wiring trainer, which is the precondition for landing this beside a live
recipe.

### 2.2 The doctrinal tension, named rather than papered over

`tac_goal_head.py` carries a rule that reads, at first glance, as the opposite of what I built:

> *"THE LOSS NEVER GUARDS ITSELF OUT OF EXISTENCE … a guarded term makes `p.grad` None, which is
> indistinguishable from a head that was never wired."*

That rule is right, and the ambiguity it names is real: at `--w-tac-goal 0` the census does read
`NOT_WIRED`, exactly as it did before the fix. **The resolution is that the ambiguity is now settled
by an independent record instead of by the gradient.** The head has a weight, so
`effective_weights_stamp_v3` emits a row for it — `status: OFF_BY_DEFAULT`, `builds_graph: false`.
Before this change it emitted **no row at all**, and `config.json` could not distinguish *"switched
off"* from *"never wired"*. Now it can.

Inside the channel nothing is guarded: `tac_goal_loss` divides by a clamped denominator on purpose,
so an all-ignored batch still returns a real zero **with `n_supervised`** and the head still receives
a gradient tensor. The rule is honoured where it applies.

⚠️ I considered the alternative — refusing `--tac-goal-tok-head` with a zero weight, the way
`_check_goal_point_args` refuses `--goal-point-inject --goal-point-w 0`. **Rejected**, because the
recorded refcv5-v2 argv carries `--tac-goal-tok-head` and no weight, so a refusal would make that
exact record unlaunchable and would break the bit-identity requirement outright.

---

## 3. THE HEADLINE MEASUREMENT — does a gradient reach?

Rig: the trainer's **own** `compute_losses_v3`, one real backward on `losses["loss"]`, smoke-width
config, seed 0, CPU. The recorded live refcv5-v2 argv (copied from its `config.json` via the standing
guard's `LIVE_ARGV`), plus `--w-tac-goal 1.0` for the ON arm.

| arm | `n_grad_none` | `grad_abs_sum` | verdict | total loss | term | `n_supervised` |
|---|---|---|---|---|---|---|
| **OFF** (default) | **2 of 2** | **0** | `NOT_WIRED` | `19.020389556884766` | absent | — |
| **ON** `--w-tac-goal 1.0` | **0 of 2** | **3.602122873067856** | `GRADIENT_REACHES` | `19.702945709228516` | `0.6825564503669739` | `3.0` |

⭐ **An independent analytic cross-check, not a re-run of the producer's own arithmetic:**

```
19.020389556884766  (OFF total, = HEAD's total)
+ 1.0 × 0.6825564503669739  (the reported term at weight 1.0)
= 19.70294600725174
measured ON total    = 19.702945709228516      Δ = 2.98e-7  (float32 accumulation)
```

The ON total is the OFF total plus exactly the weighted term. Nothing else moved.

⭐ `n_supervised = 3.0` is a **literal** expectation: the injected target supervises cells (0,0),
(0,1) and (0,2) and no others. It is written as `3`, never as an expression over the target tensor —
re-deriving the producer's own arithmetic would measure determinism, not correctness.

⚠️ **HONEST SCOPE — the width.** `n_params = 726` here, not the 11,286 the live config records. This
is the **smoke** width, which is the rig the standing guard
`test_built_heads_receive_gradient.py` itself uses; 11,286 is the live `d_tac = 512` width. The
quantity being measured — *does a gradient land on this module at all* — is width-independent, but
the `grad_abs_sum` magnitude is **not** comparable across widths and must not be quoted as if it were
a live-run number.

---

## 4. IS THE OFF PATH BIT-IDENTICAL TO HEAD?

Two trees, both off-Drive, both built from HEAD `ad68b3d`; the only difference is the patched
trainer (`md5 6ba18a753653` → `313acd19c33a`).

⚠️ **The first form of this comparison was BLIND, and I am reporting that rather than hiding it.**
Comparing parameters straight after construction read **0 differing for BOTH arms** — no optimizer
step had run, so the values were seed-determined and the loss could not yet have moved them. That
comparison proves the model *construction* is identical and is structurally incapable of seeing the
loss change. Presenting it as proof of the ON case would have been the *check that shares the defect
it checks for*. The comparison below therefore takes **one real SGD step** first.

| comparison | parameter tensors differing (post-step) | total loss |
|---|---|---|
| patched **OFF** vs HEAD | **0 of 167** | bitwise equal, `19.020389556884766` |
| patched **ON** vs HEAD | **68 of 167** — incl. `tac_goal_tok_head.net.{weight,bias}` | differs |

The 68 include encoder tensors: the gradient propagates back through the shared trunk, which is what
"the head is trained" actually means.

---

## 5. THE DELIBERATELY-CORRUPTED CONTROL — can the comparison FAIL?

⛔ WP-B's removability proof was green **for a reason unrelated to WP-B**: a zero-init gate had
multiplied its whole branch away, so it passed with the head deliberately corrupted. Two independent
corruptions, applied to *this* term:

* **C1** — the head's parameters are perturbed (`+7.5` on every tensor).
* **C2** — `tac_goal_loss` itself is replaced by a `×1000` version.

| case | OFF total | ON total | ON term | ON `grad_abs_sum` |
|---|---|---|---|---|
| clean | `19.020389556884766` | `19.702945709228516` | `0.6825564503669739` | `3.602122873067856` |
| **C1** params +7.5 | `19.020389556884766` — **UNMOVED** | `31.563495635986328` — **MOVED** | `12.543106079101562` | `4.860019385814667` |
| **C2** loss ×1000 | `19.020389556884766` — **UNMOVED** | `701.5768432617188` — **MOVED** | `682.5564575195312` | `3602.122772216797` |

⇒ **The OFF comparison survives deliberate corruption of the head and of the loss; the ON comparison
fails under both.** The OFF identity is therefore a genuine *absence from the graph*, not an
insensitive probe. C2's two numbers are also exact `×1000` scalings of the clean ones
(`682.5564503669739` and `3602.122873067856` predicted, to float32) — a second independent check.

---

## 6. THE HEAD IS NOW VISIBLE TO THE STAMP

The 2026-09-07 census established *why* the stamp was blind: it enumerates **declared loss weights**,
and this head had none, so it produced **no row at all**. Giving it a weight is therefore not
packaging around the fix — it is the half of the fix that makes the head visible to the instrument
that is supposed to see it.

| tree / arm | `n_terms` | `n_builds_graph` | `--w-tac-goal` row |
|---|---|---|---|
| HEAD, default | **6** | 1 | ⛔ **ABSENT** |
| patched, default | **7** | 1 | `requested 0.0`, `effective 0.0`, `status OFF_BY_DEFAULT`, `builds_graph false` |
| patched, `--w-tac-goal 1.0` | **7** | 2 | `requested 1.0`, `effective 1.0`, `status TRAINS`, `builds_graph true` |

And the refusal fires at launch, in milliseconds — `--w-tac-goal 1.0` **without** `--tac-goal-tok-head`:

```
[v3] REFUSED (effective-weight audit): --w-tac-goal 1 in this arm: --w-tac-goal needs
`--tac-goal-tok-head` (no head => no `tac_goal_logits` in `out`) AND `--v7-labels`
(no join => no `tac_goal_y`/`tac_goal_w` target).
```

The legitimate combination is accepted (`ON_ACCEPTED: True`). This is the `w_agent` defect —
a weight stamped in `config.json` whose loss term is silently skipped — refused at the preflight
rather than discovered after a GPU day.

---

## 7. THE GUARD — and what went RED

### 7.1 `KNOWN_UNWIRED` shrank to EMPTY, and was replaced by something STRICTER

`tac_goal_tok_head` did **not** move to a wider exemption. `KNOWN_UNWIRED` meant *"no loss exists"*,
which is no longer true, so leaving it there would keep describing a closed seam as open. It moved to
a new literal map:

```python
KNOWN_UNWIRED: dict[str, str] = {}          # ⛔ empty, and must stay that way
OFF_BY_WEIGHT: dict[str, str] = {"tac_goal_tok_head": "--w-tac-goal"}
```

⭐ `OFF_BY_WEIGHT` carries an obligation `KNOWN_UNWIRED` never did:
`test_off_by_weight_heads_are_wired_when_their_flag_is_on` **turns the named flag on** and requires
the verdict to flip to `GRADIENT_REACHES` with `grad_abs_sum > 0` and `n_grad_none == 0`. An entry
that cannot be flipped **fails** — so unlike an allow-list, this map cannot be used to park a broken
head.

### 7.2 The two mutations — both go RED, and both are specific

⛔ A guard that cannot be made to fail proves nothing. Both mutations reintroduce a **real** defect at
the **real subprocess/source boundary**, not a stubbed one.

| mutation | what it reintroduces | result |
|---|---|---|
| **M1** — the `tac_goal_loss` call block deleted from `compute_losses_v3` | D-TACGOAL-TRAINER-SEAM-OPEN **verbatim**: head built, stamped, forward-run, nothing consumes the logits | **2 RED**, 10 pass |
| **M2** — the `item["tac_goal_y"]` emission deleted from `V3Dataset.__getitem__` | the target never reaches the batch | **3 RED**, 11 pass |

M1's RED message:

```
⛔ 'tac_goal_tok_head' reads NOT_WIRED with --w-tac-goal 1.0 -- the weight is declared
   and STILL no gradient lands.
   Census row: {'n_tensors': 2, 'n_params': 726, 'n_grad_none': 2, 'grad_abs_sum': 0,
                'verdict': 'NOT_WIRED'}
   This is D-TACGOAL-TRAINER-SEAM-OPEN re-opened.
```

Both mutations left the other tests passing — the detectors are **specific**, not blanket.

### 7.3 The knob list is derived from argparse, never hand-written

`--wp-index` shipped with **3 of 6 knobs parsed, stamped and inert**, caught only because a test
enumerated knobs from the parser. `test_every_tac_goal_knob_the_parser_accepts_is_actually_CONSUMED`
walks `build_parser()._actions`, takes every option containing `tac-goal`, and requires each dest to
be READ in the trainer source. And `--tac-goal-negatives` gets its own discriminating control:
under `all` the CoT-only traffic-light cell must become supervised, under `measured` it must not.
If both read the same, the knob is inert and the test fails.

### 7.4 The dataset path is covered — the first gap in my own draft

⚠️ Both gradient probes **inject** `tac_goal_y`/`tac_goal_w` by hand, so neither ever executes the new
`V3Dataset.__getitem__` block. A term proven to reach a gradient from a hand-made target, whose real
target path is untested, is the `--wp-index` shape. Four tests now drive the **real loader** with a
v7.2 join (`test_tac_goal_trainer_flag.py` §6), including a clip with **no record** that must still
carry both keys as an explicitly all-ignored row — never a missing key.

### 7.5 The split-fitting block is covered — the second gap in my own draft

⚠️ Same shape again, one level further in. Both probes set `model._tac_goal_pos_weight = None` by
hand, so nothing executed the block in `train()` that fits `pos_weight` and `class_mask` on the
loaded split. A wrong key there (`mask_report(...)["mask"]` vs `["masks"]`) would have surfaced only
at a real launch — after the corpus mounts and a GPU-day is already committed, which is the failure
this whole package exists to prevent. Two tests (§7) now drive the **real** functions and assert
every key as a **literal**:

* `mask_report` returns exactly `{mask, masked_why, n_total, n_trainable, trainable}` — all five read
  by `train()`; `goal_pos_weight` returns 22; both convert to `[22]` `float32`.
* ⭐ **An analytic target, not a recorded number:** with every logit 0 and every target 0,
  BCE-with-logits is exactly `-log(1 - sigmoid(0)) = log 2`. MEASURED `0.6931471824645996`.
  And `n_supervised == 2 rows × n_trainable` — which is what proves the `class_mask` is actually
  applied rather than silently dropped.
* The `config.json` stamp is asserted **JSON-serialisable**: a run record that cannot be written is a
  run record that does not exist.

### 7.6 A third gap in my own draft, caught by my own argparse-derived audit

⛔ `tac_goal_stats` — the census, the mask with its reasons, and the split-fitted `pos_weight` — was
**computed and never written to `config.json`**. That is `--wp-index`'s *parsed, stamped and inert*
defect reproduced **inside the fix that warns about it**, and it was additionally a latent
`NameError`: the variable is bound only inside `if args.v7_labels:` and would have been read
unconditionally at config-write time, so **any arm without `--v7-labels` would have crashed at the
config write**. Edits E9 (initialise the carrier) and E10 (stamp it) close both.

---

## 8. Split-derived constants, not literals

`pos_weight` and `class_mask` are fitted **on the loaded split** in `train()`, never hardcoded:

* `v7_labels.goal_pos_weight(labels)` — `n_neg/n_pos` capped at 50. Hardcoding these is the
  derived-constant trap that moved `HORIZON` 7 → 8 and turned a reproduction into a different
  experiment.
* `tac_goal_head.mask_report(census)["mask"]` — switches off every class with positives but **no
  supervised negative** (5 of 22 on the v7.2 train blob). An unmasked logit there can only be pushed
  towards 1, which degrades the shared trunk and inflates any pooled score.
* Both, plus the full per-token census, are stamped into `config.json`, and a channel where **zero**
  classes are trainable is **refused at launch**.

---

## 9. Register and test-surface consequences

The census's own detector, re-run on the patched tree, independently measures the change — this is
not my literal being edited to make a test pass:

```
tac_goal_bce   NO_CONSUMER  ->  CONSUMED
```

Two tests correctly **failed** on the patched tree because they pinned the defect as open, and both
were updated in this package:

* `test_v7_vocab_reach_census.py::test_D_TACGOAL_1_is_STILL_OPEN_no_trainer_calls_the_goal_loss`
  → renamed `..._SEAM_IS_CLOSED_the_trainer_calls_the_goal_loss` and **inverted**, still two-sided.
  ⭐ The inverted form is also *safer*: the old `assert sym not in trainer` would have **passed on an
  unreadable file**; the new positive form fails on one.
* `EXPECTED_CONSUMER_STATUS["tac_goal_bce"]` → `CONSUMED`, with the honest caveat attached:
  **CONSUMED means a consumer EXISTS, never that an ARM BOUGHT IT.**

⛔⛔ **THE CLAIM THAT MUST NOT BE OVERSTATED.** `--w-tac-goal` defaults to `0.0` and **no live recipe
passes it**. The 22 tactical goal tokens are now **REACHABLE**, not **TRAINED**. Reading the census
row as *"the tokens are a training signal in the live run"* is the D-TACGOAL claim one step too far.
The arm-level fact lives in `config.json`'s effective-weights block, which now carries a
`--w-tac-goal` row for exactly this reason.

`pytest -q` on the affected surface: **136 passed** (8 files).

### 9.1 The whole suite, both trees — zero regressions

⛔ A green affected-surface is not evidence about the rest of the suite. The full `stack/tests` run
was executed on **both** trees — the untouched HEAD baseline and the patched tree — under
`--continue-on-collection-errors`:

| tree | failed | passed | skipped | errors |
|---|---|---|---|---|
| baseline (HEAD `ad68b3d`, untouched) | **128** | 6568 | 178 | **61** |
| patched | **128** | **6579** | 178 | **61** |

⭐ **The counts matching is not the evidence — the SETS matching is.** Equal counts can coincide. The
189 unique `FAILED`/`ERROR` lines were extracted from each log and diffed both ways: the set
difference is **empty in both directions**, and the two files hash **identically**
(`md5 1d61219c722a2710919008467724f605`, `raw/failset_{base,patched}.txt`).

⇒ **passed 6568 → 6579 = +11, exactly the number of tests this package adds; failures and errors
unchanged, member for member.** The 128 failures and 61 errors are pre-existing artifacts of the
off-Drive tree (only `stack/` was copied, so `taniteval` and the repo-root modules are absent) and are
present identically on the untouched baseline. ⚠️ They are **not** a claim about the repo's real
suite health, only that **this change moves none of them**.

---

## 10. What the PI still has to decide

**One decision, and it is the one this package deliberately did not take: what weight `--w-tac-goal`
should carry in a real arm, and whether that budget comes from `MANEUVER_WEIGHT` or is added on top**
(PI decision queue item 10). The seam is closed, the loss is available, the guard makes re-opening it
impossible, and every launch path refuses a weight it cannot spend — but until a recipe passes a
non-zero `--w-tac-goal`, the 22 tokens still train nothing.

---

## 11. Deliverable manifest

| artifact | where it lives |
|---|---|
| `stack/scripts/refc_v3_train.py` | **repo**, staged (8 additive edits, 5,380 → 5,583 lines) |
| `stack/tests/test_built_heads_receive_gradient.py` | **repo**, staged (`KNOWN_UNWIRED` → empty; `OFF_BY_WEIGHT`; 7 new tests) |
| `stack/tests/test_tac_goal_trainer_flag.py` | **repo**, staged (4 new dataset-path tests) |
| `stack/tests/test_v7_vocab_reach_census.py` | **repo**, staged (status flipped; headline test inverted) |
| `RESULT.md` (this file) | **repo**, staged |
| `raw/probe_patched.json`, `raw/probe_base.json` | **repo**, staged — the gradient probe, both trees |
| `raw/probe_corrupt.json` | **repo**, staged — C1/C2 deliberate corruptions |
| `raw/poststep_compare.txt`, `raw/fingerprint_compare.txt` | **repo**, staged — bit-identity |
| `raw/stamp_compare.txt`, `raw/stamp_base.json`, `raw/stamp_patched.json` | **repo**, staged |
| `raw/pytest_guard_GREEN.log` | **repo**, staged — 12 passed |
| `raw/pytest_guard_RED_mutation.log` | **repo**, staged — **M1: 2 failed, 10 passed** |
| `raw/pytest_RED_dataset_mutation.log` | **repo**, staged — **M2: 3 failed, 11 passed** |
| `raw/pytest_affected_GREEN.log` | **repo**, staged — 134 passed |
| `raw/probe_*.py`, `raw/patch_*.py` | **repo**, staged — the probes and the patch scripts, so every number is re-derivable |
| off-Drive working trees | `C:\Users\Admin\_tacgoal_wp\tree_{base,patched,mutant,mutant2}` — **scratch, NOT deliverables**; everything they produced is banked above |

⚠️ No pod, no GPU, no training launched. The dev-box RTX 4060 was left alone for the refcv5-v2
inference-seed replicate; every measurement here is CPU.
