# Two of the three missing pre-launch refusals, built and mutation-proven

**Date:** 2026-09-17 · **Evidence class: MEASURED** · **Tier: dev box, CPU**
**Closes:** two of the three ⛔ rows in `PREREG_REFCV6_V2.ERRATUM-1.md` §E4.

## What was missing

§12 of the pre-registration lists ten refusals under the sentence *"the launch checker exits
non-zero on any of these"*. E4 measured that claim and found **4 ✅ · 3 ⚠️ (wired to the
2026-09-10 arm set) · 3 ⛔ (did not exist)**. This package builds two of the three that did not
exist:

| # | refusal | before | now |
|---|---|---|---|
| 3 | an unregistered hypothesis id | ⛔ nothing read `GOALS_AND_CLAIMS.md` | ✅ `refuse_unregistered_hypotheses` |
| 4 | a hyper-parameter selected on the **scored** split | ⛔ nothing checked it | ✅ `refuse_scored_split_leak` + `refuse_overlapping_splits` |
| 10 | a capped `goal_pos_weight` quoted without its cap sentence | ⛔ | ⛔ **still missing, deliberately** — see below |

⭐ **Refusal 4 is the one worth having.** A hyper-parameter chosen on the split it is later scored
on does not crash, does not look wrong, and does not show up in any curve — it **manufactures a
positive**. Every other refusal in §12 catches something that would otherwise be visible.

## ⛔ What this can and cannot do, stated before the results

*"Was this threshold fitted on the scored split?"* is **not decidable from a panel**. What is
decidable, and is what is enforced:

1. every tuned quantity **declares** its split, and an **undeclared** one is REFUSED — not assumed
   innocent. **Silence is the defect**: it is exactly how a tuned threshold passes as a constant.
2. no declared split is the scored one;
3. the scored split is **disjoint** from fit and val — MEASURED on ids, not asserted.

⇒ a run can still **lie** in its declaration. It can no longer pass by **saying nothing**, which is
the failure mode the record actually contains.

## The guard-removal audit — 8 of 8 KILLED

⛔ An unmutated guard is a comment. Each refusal branch was deleted in turn and the suite required
to go RED:

| guard removed | verdict |
|---|---|
| `R3-unregistered` · `R3-empty-claim` · `R3-missing-registry` | **KILLED** |
| `R4-no-tuned-block` · `R4-fitted-on-scored` · `R4-undeclared` | **KILLED** |
| `SPLIT-empty-scored` · `SPLIT-overlap` | **KILLED** |

**8 killed / 0 escaped / baseline restored GREEN** (`raw/guard_removal_audit.json`). Tests: **14
passed**.

### ⚠️ Two defects the audit found — in the audit, and in the module

1. **The audit's first run reported 8/8 KILLED and it was WORTHLESS.** Every run returned pytest's
   exit code **2** — a *collection* error, not a failed test: a path bug meant the module never
   imported, so "the suite failed" for a reason that had nothing to do with the guard. ⛔ A
   guard-removal that merely breaks the import masquerades as a guard that bites. The audit now
   counts a kill **only** on returncode 1 and reports `BROKEN-RUN` otherwise.
2. **With that fixed, `R3-missing-registry` ESCAPED.** Deleting `if not p.is_file(): raise` changed
   nothing, because `read_text` raises `FileNotFoundError` by itself — so a type-only
   `pytest.raises(FileNotFoundError)` passed with the guard gone. The guard's real contribution is
   the **distinction it names** (*"'no ids are registered' and 'the registry could not be read' are
   different facts"*), so the test now pins that message. It then KILLED.

⭐ Both are the same lesson in different places: **a check that passes for the wrong reason is
indistinguishable from a check that works**, until something forces it to fail on purpose.

## ⛔ Refusal 10 is NOT built, and that is a decision, not an omission

Refusal 10 — *a capped `goal_pos_weight` quoted without the cap sentence* — is **a rule about
prose**. It cannot be enforced against a report that a human writes. It should either become a check
on the emitted panel's own fields (a `pos_weight` entry carrying `capped: true` and a `cap_note`),
or be **demoted from a refusal to a reporting convention**. ⛔ Leaving it listed as a refusal while
nothing implements it is what guarantees it is never built — which is the defect E4 was about.

## What remains before a GPU arm

Refusals 1, 2 and 9 still need **rewiring** from the 2026-09-10 arm set (`check_one_variable` takes
`w_agent` / `w_tac_goal`; the v2 variable is the **backbone**), and refusal 10 needs the decision
above. ⛔ §11's OWED list stands.

## Files

| path | what |
|---|---|
| `stack/tanitad/train/prelaunch_v2.py` | the two refusals |
| `stack/tests/test_refcv6_prelaunch_v2.py` | 14 tests, both directions |
| `code/guard_removal_audit.py` | the audit |
| `raw/guard_removal_audit.json`, `raw/guard_removal_audit.txt` | its output |
