# The conditioning contract is asserted on EVERY batch — and the once-only optimisation was never paying for itself

**Date** 2026-09-07 · **Stream** Architecture & Inference · **Branch** `agent/arch-inf-20260803`
**Closes** the single hole the sibling package named
(`Research/2026-09-07-rl-channel-values/` §4: *"the guard fires on the **first batch only**
(`refc_adapter.py:505`, `checked["done"] = True`). A rollout whose later batches drop a
channel would not be caught"*), on top of
`Research/2026-09-07-rl-channel-admissibility/` (commits `3fd8db0` `1360fd9` `41bdd6e` `b971e52`).

⛔ **NO EVAL TIER AND NO FOUR-FAMILY TABLE.** No model produces a trajectory anywhere in
this package — the model in every proof below is a stub returning zeros. It is a
**wiring-contract audit**; stamping an eval tier or a longitudinal/lateral/tactical/
strategic table on it would be a category error.

⛔ **ZERO GPU.** refcv5-v2 (A40, PID 2560646) was not touched, nothing was shipped to any
pod, and every run below was CPU-only (`CUDA_VISIBLE_DEVICES=""`) in an off-Drive mirror
created for this work (`C:\Users\Admin\tanitad-perbatch`) — **not** `tanitad-wt`, which
re-syncs from the repo mid-run and silently drops edits.

⚠️ **Contended files were READ ONLY and are byte-unchanged.** `git diff HEAD --stat` over
`refc.py` / `refc_v3.py` / `refc_v3_train.py` is empty; `refc_v3.py` md5
`b5535b59bf91ce83f94b38a0cb932dc7`, equal to the value the sibling recorded at session
start. `refc_adapter.py` is the only source file changed.

---

## 1. ⭐ THE COST/BENEFIT HAD INVERTED — and the measurement says it never held anyway

Checking once is a defensible optimisation **while a miss costs "a softer policy"**. A
sibling MEASURED what a missing `v0` actually does, and it is not that — it **silently
replaces the action space**. Every link re-read from source this session:

```
refc.py:3097   v_ms = v0 if (cfg.sel_reach_clamp and v0 is not None) else None
refc.py:2128   bank = self.roll_bank(v_ms, ego_keep, ...)
refc.py:1722   if v_ms is None or anchor_withheld_bank == "none":  v = full(ref_speed)
```

⇒ every anchor rolled at the **10 m/s reference** instead of the window's measured speed
(vocabulary gap `refc.py:363-370`: **0.3773 m** oracle-in-vocabulary fixed-path vs
**0.2610 m** v0-conditioned), plus `refc.py:2360` skipping the S2 reachability band and
`refc.py:2970-2977` setting `keep = 0` on **100 %** of rows where training saw `keep = 1`
on `1 - ego_dropout` of them. ⇒ **the cost of one missed batch is a well-formed fan from a
policy that never existed.**

### 1.1 ⛔ THE MEASURED PER-BATCH COST — the number, not an assertion

CPU-only, this box, **two independent runs** reported as a range (min of 7 timed blocks;
20,000 reps for the per-batch path, 2,000 for the resolve paths, 5 for the forwards).
Raw output: `raw/cost_measure.txt`; probe: `code/cost_probe.py`.

| path | per call | share of ONE forward (smoke) | share of ONE forward (deployed) |
|---|---|---|---|
| `conditioning_requirements` — resolved **ONCE** | 2.514 – 2.535 µs | 0.063 – 0.078 % | 0.00058 % |
| **`ConditioningContract.check` — run on EVERY batch** | **0.181 – 0.184 µs** | **0.0046 – 0.0056 %** | **0.00004 %** |
| `assert_conditioning`, fully UNCACHED | 3.071 – 3.124 µs | 0.076 – 0.097 % | 0.00070 – 0.00072 % |
| `RefCV3Model.forward` **smoke** B=1 (64 px, base_width 8, 20 anchors) | 3.219 – 4.030 **ms** | — | — |
| `RefCV3Model.forward` **default** B=1 (256 px, 9 ch, W=8) | 435.296 – 436.692 **ms** | — | — |

⭐ **The conservative bound** — slowest observed check against the *fastest* observed
forward, which is also the *smallest* forward in the repo: one per-batch check is
**1/17,495 of a smoke forward** and **1/2,365,739 of a deployed-scale one**.

⚠️ **REPORTED HONESTLY, BECAUSE THE BRIEF ASKED FOR A NUMBER AND NOT A REASSURANCE: the
once-only optimisation was never paying for itself.** Even the **fully uncached**
`assert_conditioning` is at most **0.097 %** of the *smoke* forward and 0.0007 % of the
deployed one. ⇒ **Nothing here is material**, and the honest conclusion is that the
original cost/benefit call was mis-specified in both directions: the benefit was ~0 and
the cost of a miss is a replaced action space. No alternative design is proposed, because
none is needed — the cache is kept for two *non*-performance reasons stated in §2.

⚠️ **Why the smoke forward is the reference and not only the deployed one.** It is the
cheapest forward the repo can build, so measuring against it is the hardest test the
guard can be given. Both are reported so the reader can see the ratio is not an artefact
of picking a big model.

---

## 2. What was built

`stack/tanitad/rl/refc_adapter.py` — the **only** source file changed.

⭐ **The requirement MAP is cached; the CONTRACT is asserted every batch.** These are
different objects and conflating them is what made "check once" look reasonable:
`conditioning_requirements(model)` walks a dotted config path per predicate
(`cfg.core.anchors.v0_conditioned` and friends) and is the only part with any cost; the
assertion is a `dict.get` per **required** channel — 2 on refcv4b/refcv5, 3 on a fully
conditioned build (MEASURED: `required = ('ego_state', 'v0', 'lan')`).

```python
contract = ConditioningContract(model) if strict_conditioning else None

def sample_fn(batch, cfg_in):
    if contract is not None:
        contract.check(batch)            # EVERY batch, 0.184 us
```

⭐ **The two reasons the cache is kept, neither of them speed:**

1. **A declaration bug raises at ONE deterministic point.** The map is resolved once per
   rollout, so an unresolvable predicate (the defect class the sibling closed with
   `_read_predicate`) surfaces once rather than on every batch.
2. **The contract is fixed for the life of the `sample_fn`.** It is keyed on the identity
   of `model.cfg`, so a *swapped* config re-resolves (pinned by a test) while the rollout
   is otherwise checked against one stable contract.

⚠️ **SCOPE STATED RATHER THAN IMPLIED:** an **in-place mutation** of the same config
object is deliberately *not* covered. A config mutated mid-rollout changes the policy
under the optimiser and is a different defect class from a batch that forgot a key, which
is what this guard is for. Written into the class docstring, not left for rediscovery.

⭐ **The refusal now has ONE source** (`_conditioning_refusal`), shared by the per-batch
path and the single-shot `assert_conditioning`, so the two callers cannot drift into
saying different things — and it **names the batch**:

```
checkpoint was trained WITH ['v0'] (its own config says so) but the batch supplies None.
⛔ REFUSED ON BATCH #2 of this rollout — a LATER batch, which a once-only check would
have missed. ...
```

⭐ **`strict_conditioning=False` is now literally RECORDED, not merely documented.** The
docstring already claimed *"it must be typed, and it is recorded"*; nothing recorded it.
The returned callable now carries `.strict_conditioning` and `.conditioning_contract`
(`None` when off), so a preflight or run record banks the arm instead of trusting an argv
string.

---

## 3. ⛔ THE MUTATION PROOF — and it is TEMPORAL, because the defect is

`code/mutate.py`, run against the **landed bytes** (mirror md5 `0f117ee2…` == repo md5).
Full output: `raw/mutation_proof.txt`. Each mutation reintroduces one real defect in the
real file and requires a RED; the controls require a GREEN.

| # | mutation | expected | got | pytest |
|---|---|---|---|---|
| — | CONTROL, unmutated | GREEN | **GREEN** | 87 passed |
| **M1** | ⭐ **THE DEFECT RESTORED, FAITHFULLY** — the pre-change body verbatim (`checked = {"done": False}` **and** the two recording lines that reference `contract`) | RED | **RED** | 9 failed |
| **M1b** | ⭐ **ONCE-ONLY SEMANTICS IN ISOLATION** — every other line identical, the check simply stops after batch 1 | RED | **RED** | 7 failed |
| M2 | the guard becomes a latch in disguise: batches 2..N return early | RED | **RED** | 5 failed |
| **M3** | ⭐ **OVER**-assertion: every declared channel required regardless of config | RED | **RED** | 10 failed |
| M4 | the cache swallows a declaration bug: cfg recorded BEFORE the resolve | RED | **RED** | 1 failed |
| M5 | ⛔ the escape hatch is broken: `strict_conditioning=False` still asserts | RED | **RED** | 4 failed |
| M6 | VACUITY: the check refuses every batch | RED | **RED** | 10 failed |
| M7 | the VALUE is not checked, only the KEY's presence | RED | **RED** | 1 failed |
| — | CONTROL, after restore | GREEN | **GREEN** | 87 passed |

**10/10 expectations met.**

⭐ **WHY M1 AND M1b ARE BOTH THERE, AND WHY THE FIRST ATTEMPT WAS WRONG.** My first M1
reverted only the head of `make_refc_sample_fn` and left `sample_fn.conditioning_contract
= contract` referencing a name that no longer existed. It went RED — **for a
`NameError`**. A RED caused by a broken file proves nothing about a temporal defect, so
M1 was rewritten as a faithful revert of the whole function body, and **M1b** was added to
isolate the once-only *semantics* with every other line untouched. Both go RED, and the
second one cannot be explained by anything but the timing of the check.

⚠️ **No expectation in the new suite is "whatever the code does".** The shape the sibling
found in `test_rl_channel_guard.py` — `if missing: expect refusal; else: expect pass` —
has an expected value computed from the code under test and can never fail. Every
expectation here is a hard-coded literal: *this* call passes, *that* call raises, this
counter is 1 and that one is 50.

### 3.1 ⭐⭐ THE TEMPORAL PROBE — the same sequence under both worlds

Not a test assertion: `code/mutate.py` drives ONE `sample_fn` with a good batch and then a
`v0`-less one and **prints what happens**, under the live code and under M1/M1b.

**LIVE (per-batch):**
```
  BATCH 1 (v0 present): PASSED, fan (1, 2, 4, 4, 2), forward got v0=7.5
  BATCH 2 (v0 DROPPED): REFUSED -> checkpoint was trained WITH ['v0'] ... ⛔ REFUSED ON
                        BATCH #2 of this rollout — a LATER batch, which a once-onl...
  forward call count = 1 (the bad batch never reached it)
```

**M1 and M1b (once-only restored) — identical output from both:**
```
  BATCH 1 (v0 present): PASSED, fan (1, 2, 4, 4, 2), forward got v0=7.5
  BATCH 2 (v0 DROPPED): NO REFUSAL. fan (1, 2, 4, 4, 2) logp (1, 2, 4) all-finite=True
  the forward was handed v0=None -> refc.py:3097 v_ms=None -> refc.py:1722 every anchor
                        rolled at the 10 m/s reference
  forward call count = 2 (the defective batch WAS sampled from)
  => A WELL-FORMED FAN FROM A POLICY THAT NEVER EXISTED, SILENTLY.
```

⇒ **the good-batch-then-bad-batch sequence passes silently under the once-only guard and
is refused on the second call under the new one.** That is the defect, demonstrated
rather than argued, and it is the difference the test suite is pinned to.

### 3.2 ⭐ The over-assertion control is alive (M3)

A mutation that requires channels the model was **not** trained with goes RED against
three tests written for it: an un-conditioned build must accept a `v0`-less batch on
batch 1 **and on batch 20**; a `nav_known_channel` build must not demand `nav_cmd` on any
batch (REF-C's published arm decodes with `nav_cmd=None` — the C6 confound / `os_navzero`);
and 100 good batches must all pass and all reach the forward.

⚠️ This matters as much as M1: **refusing valid launches is how a guard gets deleted
rather than fixed**, and moving a check into the loop multiplies that risk by the number
of batches.

---

## 4. ⭐ `strict_conditioning=False` behaves exactly as documented

Three hard-coded tests, plus M5 which proves they can fail:

* a `v0_conditioned` build, **10** consecutive `v0`-less batches, **zero** refusals, **10**
  forwards, and the forward is handed `v0=None` each time — which is what an
  un-conditioned ablation arm asked for;
* it **resolves nothing at all**: with a config on which no predicate could resolve, the
  arm still samples, because `contract is None` and the declaration is never consulted;
* `strict` is the **default**, so the hatch cannot be taken by omission.

The pre-existing `test_strict_conditioning_can_be_turned_off_but_must_be_typed` is
unchanged and still green.

---

## 5. Regression sweep

| suite set | before | after |
|---|---|---|
| the siblings' 9 suites + `test_rl_per_batch_conditioning.py` | 196 | **212 passed** |
| every `tests/test_rl_*.py` (21 files) | — | **358 passed, 0 failed** |
| `rl_control_space_preflight.py` CHECK 0 | `=> COMPLETE` | `=> COMPLETE` |

⚠️ **A control I ran because the first sweep looked red.** An early `test_rl_*` sweep read
`6 failed`. Re-running the *identical* set against the **pre-patch** adapter produced the
**identical 6 failures** ⇒ not caused by this change. Their cause was then read from
source: `ModuleNotFoundError: taniteval` (2) and `SystemExit: ... tools/criteria_check.py
... is absent` / `products/P7-TanitEval/CRITERIA_REGISTRY.json ... is absent` (4) — all
**off-Drive-mirror completeness artifacts**, because I had copied only `stack/`. Mirroring
`taniteval/`, `tools/` and that one JSON took the sweep to **358 passed, 0 failed**. ⇒ a
red that is not your red still has to be explained, not waved at.

---

## 6. ⭐ IS THE RL VALUE FLOW CLOSED?

**For `refc_adapter` — the refcv4b/refcv5 binding — yes.** Which channels may be supplied
is a declared decision per seam (sibling 1); which are REQUIRED is read from the build's
own nested config and refuses an unresolvable predicate (sibling 2); and **the contract is
now asserted on every batch of a rollout, not the first** (this package). A required
channel can no longer arrive `None` unnoticed at any point in a rollout.

⛔ **WHAT IS LEFT — and it is one module over, not one level down.** MEASURED, with a
same-breath control (`refcv3_adapter` referenced 25×, so the search surface was live):

* **`refcv3_adapter.make_refcv3_sample_fn` has NO conditioning assertion at all — on any
  batch.** It is the sample_fn that `stack/scripts/rl_pilot_refc21.py:438` — **the only
  production RL script in the repo that builds one** — actually uses, and it passes
  **three of the twelve conditioning channels** (`nav_cmd`, `v0`, `lan`; plus the
  positional `frames` and explicit `steps`), which is the gap `refc_adapter` was created
  to close in the first place.
* **`make_refc_sample_fn` has no production caller yet** (`stack/scripts/` reference it
  zero times; only the two test files do). ⇒ this guard is **armed for the refcv4b/refcv5
  RL arms when they launch**; it is not today protecting a running rollout, and saying
  otherwise would be true-but-wrong for the reader.
* The pilot's own batch builder does supply `v0` on every batch
  (`rl_pilot_refc21.py:136`), so this is a **missing guard, not a known live miss** — but
  it supplies no `ego_state`, `lan` or `nav_known` at all, and nothing would say so.

⇒ **The value flow is closed on the adapter the next RL arm will use, and open on the one
today's pilot uses.** The fix is not to move this code: `refcv3_adapter` is kept verbatim
on purpose (two tests import it). The decision to escalate is whether
`rl_pilot_refc21.py` should be repointed at `refc_adapter`, which is a stream-owner call,
not mine.

---

## 7. Deliverable manifest

| artifact | where it lives | only one place? |
|---|---|---|
| `ConditioningContract` + per-batch assertion + recorded arm | `repo:stack/tanitad/rl/refc_adapter.py` | no — mirror + repo |
| 16 temporal / vacuity / escape-hatch tests | `repo:stack/tests/test_rl_per_batch_conditioning.py` | no |
| mutation harness (10 expectations, 2 temporal probes) | `repo:…/2026-09-07-rl-per-batch-conditioning/code/mutate.py` | no |
| cost probe | `repo:…/code/cost_probe.py` | no |
| mechanical patch (`apply_patch.py`, `new_block.py`, `fix_numbers.py`) | `repo:…/code/` | no |
| mutation proof output | `repo:…/raw/mutation_proof.txt` | no |
| cost measurement output | `repo:…/raw/cost_measure.txt` | no |
| this report | `repo:…/RESULT.md` | no |

**Escalation for the stream owner:** §6 — `rl_pilot_refc21.py` runs on the unguarded
`refcv3_adapter`. Repointing it at `refc_adapter` would put the whole contract behind the
one production RL script; leaving it is a decision that should be written down rather than
inherited.

---

*Evidence classes: **PUBLISHED-CODE** — every line cite re-read from source this session
(`refc_adapter.py`, `refcv3_adapter.py:119-160`, `rl_pilot_refc21.py:136,438`,
`refc.py` chain via the sibling's package, re-checked). **MEASURED** — the two-run cost
table; the 10/10 mutation table; the two temporal probes; the 358-test sweep; the
pre-patch control run that attributed 6 failures to mirror completeness. Every absence
claim was paired with a same-breath control that read non-zero (`make_refcv3_sample_fn` 7,
`assert_conditioning` 22, `refcv3_adapter` 25) and cross-checked on both the G: mount and
the local mirror, which agreed.*
