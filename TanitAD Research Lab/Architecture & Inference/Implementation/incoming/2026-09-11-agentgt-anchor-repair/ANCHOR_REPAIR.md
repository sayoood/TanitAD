# The agent-GT deliberate-regression arms were DISARMED BY THE PLATFORM, not by a drift

**Date:** 2026-09-11 · **Repo:** `D:\Projects\TanitAD` · branch `agent/arch-inf-20260803`
**Subject:** `stack/tests/test_refc_v3_agent_gt_head_drop.py`,
`stack/tests/test_refc_v3_agent_gt_reaches_forward.py`
**Prior record this repairs:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-09-wpc-oracle-gate/raw/APPLICATION_RECORD.md`

---

## 1. The symptom

Ten tests RED at HEAD, every one the same shape:

```
AssertionError: the mutation anchor for 'gate_removed_head' matched 0 times, not 1.
The deliberate-regression arm is DISARMED and this test would pass without proving
anything. Re-anchor it against refc_v3_train.py.
```

MEASURED baseline, this checkout: **10 failed, 11 passed**.

---

## 2. ⛔ ROOT CAUSE — the anchors fired on the PLATFORM, and the trainer had not changed at all

The anchors are byte literals written with `\n`. `refc_v3_train.py` in this working
tree is **CRLF**, so a multi-line anchor can never match.

| probe | value |
|---|---|
| trainer, working tree | **5,975 CRLF / 0 bare LF**, 366,178 B |
| trainer, git index blob | **360,203 B, 0 CR** |
| `366,178 − 5,975` | **360,203** — exact |

The index blob is **byte-for-byte the LF form of the working-tree file**. The trainer
source is unchanged; only the checkout's line endings differ. `core.autocrlf=true`
and the repo has **no `.gitattributes`**, so the same blob is CRLF on this Windows
dev box and LF on every Linux pod.

**The anchor-by-anchor evidence, and why it is conclusive:**

| anchor | lines | LF-form matches | CRLF-form matches | its tests |
|---|---|---|---|---|
| `CORRUPT_GATE_FROM` | 2 | **0** | 1 | were RED |
| `REVERT_FROM` | 2 | **0** | 1 | were RED |
| `CORRUPT_VALUE_FROM` | 1 | 1 | 1 | were **GREEN** |

⭐ **That split is the signature.** The only anchor that kept working is the only
single-line one — it contains no newline, so it has just one form. A real source
drift would not sort failures by whether the anchor spans a line break.

⚠️ Same root cause as `test_v5_trainer_v2_val.py`'s trainer pin, corrected 2026-08-04
(*"The trainer pin was firing on the PLATFORM, not on an edit"*). Second occurrence of
this class in this repo, in a different guard.

**Every anchor's target still exists in the trainer.** Nothing is unguarded, and no
test was deleted or weakened:

| anchor | target, current trainer |
|---|---|
| `CORRUPT_GATE_FROM` | `scripts/refc_v3_train.py:2341-2342` |
| `REVERT_FROM` | `scripts/refc_v3_train.py:2366-2367` |
| `CORRUPT_VALUE_FROM` | `scripts/refc_v3_train.py:2356` |

---

## 3. The fix

⛔ **Re-baselining the anchors to `\r\n` was rejected.** It is wrong twice over: it
moves the false negative onto the pods, and it is the "re-baseline on red" move the
2026-08-04 correction exists to forbid.

⛔ **LF-normalising the trainer's bytes before mutating was also rejected**, even
though that is what the 2026-08-04 hash pin does. A hash has no diff to read; these
arms do, and `_mutate`'s own docstring records why: normalising would turn a one-line
mutant into a ~5,975-line diff and *"make the arm impossible to audit"*.

**What was done instead:** the anchors stay in canonical LF form — the git blob's form
— and two new helpers resolve them to whatever ending the checkout actually has:

* `_eol(src)` — the trainer's own line ending.
* `_resolve_anchor(src, anchor)` — returns `(the form present in src, total matches
  across forms)`. **Both** forms are counted, so a mixed-ending file reports 2 and the
  arm fails LOUDLY rather than silently picking one. The `count == 1` assertion is
  unchanged.

The trainer's bytes are **not** normalised, so a mutant still differs from the shipped
file by exactly one hunk on every platform.

One assertion had to follow. `test_DELIBERATE_REGRESSION_the_mutation_anchor_is_armed`
compares `len(shipped) - len(mutant)` against the anchor's length; on a CRLF tree the
resolved anchor is one byte longer per line. It now takes the delta against the
**resolved** anchor, and additionally re-asserts `count == 1` inline — strictly
stronger than before, not weaker.

**Result: 21 passed, 0 failed.**

---

## 4. ⭐ THE PROOF THAT THE ARMS CAN FAIL — two directions, because one is not enough

A green suite is not evidence. *(`CLAUDE.md`: an AST census read 0 suspects on BOTH
the fixed and the broken trainer.)*

### 4a. Each mutation is a REAL behaviour change

Applied to the **real** `refc_v3_train.py` in place, both test files run, then restored
and md5-verified. Baseline md5 `96908240f2766533431f6e08ebf698fd` (366,178 B),
**restored exactly after all three**.

| mutation | anchor resolved | suite | BEHAVIOUR RED | trainer restored |
|---|---|---|---|---|
| GATE → `if True:` | 1 match, 118 B, CRLF | 10 failed / 11 passed | **5** | md5 matches |
| REVERT (drop `agent_gt=agent_gt`) | 1 match, 95 B, CRLF | 14 failed / 7 passed | **10** | md5 matches |
| VALUE (`* 2.0` in-branch) | 1 match, 57 B, LF | 4 failed / 17 passed | **2** | md5 matches |

*BEHAVIOUR RED* = a test asserting the shipped trainer's behaviour now fails. Counted
separately from *ANCHOR RED*, where an arm reports its anchor missing because the
in-place edit consumed the text that arm mutates on — expected bookkeeping, not
evidence. Raw logs in `raw/red_GATE.log`, `raw/red_REVERT.log`, `raw/red_VALUE.log`.

### 4b. Each ARM's green DEPENDS on the substitution

4a cannot speak for the arms themselves, because applying a mutation in place consumes
their anchor. So `_mutate`'s substitution was neutered to a no-op (`nb = ob`), making
every "mutant" byte-identical to the shipped trainer.

**Result: 8 failed, 13 passed — and the 8 are EXACTLY the 8 arms.** Zero controls
moved.

| arm | under no-op substitution |
|---|---|
| `test_a_head_build_REFUSES_a_supplied_agent_gt` | RED |
| `test_the_message_names_the_two_ways_out` | RED |
| `test_DELIBERATE_REGRESSION_the_mutation_anchor_is_armed` | RED |
| `test_case_1_still_refuses_agent_gt_with_no_seam` | RED |
| `test_the_HEAD_row_is_now_DISCRIMINATING` | RED |
| `test_REVERTING_THE_PATCH_REPRODUCES_THE_DEFECT` | RED |
| `test_the_OFF_parity_comparison_CAN_FAIL` | RED |
| `test_the_ORACLE_value_assertion_CAN_FAIL` | RED |

Both test files restored and md5-verified. Log: `raw/disarm_sim.log`.

---

## 5. ⚠️ The residual risk this repair does NOT remove

The repo still has no `.gitattributes`. Any future guard that pins trainer **bytes**
will hit this again. Two occurrences now, two separate fixes. A `.gitattributes`
normalising `*.py` to LF would remove the class outright — **not done here**, because
it rewrites every Python file's working-tree endings across every active worktree and
several trainings are live. Logged for the PI rather than taken unilaterally.

---

## 5b. ⚠ Suite status, stated honestly

The two repaired files: **21 passed, 0 failed** (reproduced 3×).

The WIDER suite is **NOT green at HEAD, and was not before this change**:
**7,720 tests collected, 2 collection ERRORS**, both missing implementation
symbols, both landed committed-broken in bb030da and untouched by this work:

| module | missing symbol |
|---|---|
| tests/test_metric_decode_refusal.py | UntrainedMetricReadout from tanitad.models.v6 |
| tests/test_refa_v1_dk_hook.py | DistanceKeepingSpec from tanitad.refs.refa_v1 |

Each symbol occurs ONLY in its own test file and nowhere in the implementation.
A full run past those errors was started and **stopped at 31 %** (~80 min
projected on this CPU box); it showed further failures in unrelated modules,
which are **NOT enumerated here** and are therefore not claimed either way.

⭐ That this change cannot reach them is MEASURED, not argued, three ways:
1. no module imports either repaired file (one comment string in
   tanitad/rl/refc_adapter.py:391 mentions a filename);
2. the only shared-global vector is _run(spy=True) patching
   RefCV3Model.forward — after both files run, that attribute is
   is-identical to its pre-run value, so the patch does not leak;
3. stack/scripts/refc_v3_train.py is byte-unchanged (md5 verified).

## 6. Deliverable manifest

| artifact | where it lives |
|---|---|
| re-anchored test (head drop) | repo · `stack/tests/test_refc_v3_agent_gt_head_drop.py` — **staged** |
| re-anchored test (reaches forward) | repo · `stack/tests/test_refc_v3_agent_gt_reaches_forward.py` — **staged** |
| this record | repo · `…/Implementation/incoming/2026-09-11-agentgt-anchor-repair/ANCHOR_REPAIR.md` — **staged** |
| in-place RED logs (3) | repo · `…/2026-09-11-agentgt-anchor-repair/raw/red_{GATE,REVERT,VALUE}.log` — **staged** |
| disarm-simulation log | repo · `…/2026-09-11-agentgt-anchor-repair/raw/disarm_sim.log` — **staged** |
| proof harnesses (2) | repo · `…/2026-09-11-agentgt-anchor-repair/raw/prove_red.py`, `prove_armed.py` — **staged** |

`stack/scripts/refc_v3_train.py` was **NOT modified** — md5
`96908240f2766533431f6e08ebf698fd` before and after, verified after each of the three
in-place mutations. Nothing committed, nothing pushed.
