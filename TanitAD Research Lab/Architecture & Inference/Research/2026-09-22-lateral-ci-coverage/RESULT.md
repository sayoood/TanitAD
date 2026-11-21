
<!-- LATERAL-CI-COVERAGE-CROSS-FAMILY-CONTEXT-2026-09-22 -->

### ⭐ 2026-09-22 — the third tip red closed: the LATERAL family's CI coverage read INCOMPLETE because a measurement bounded in ANOTHER family had nowhere to be declared

MEASURED by me, CPU only, no GPU
(`TanitAD Research Lab/Architecture & Inference/Research/2026-09-22-lateral-ci-coverage/`).
Continues `9373432` (5 reds enumerated) and `7de8d4e` (two closed). **Three down, two to go.**

`test_refcv3_arm::test_every_family_is_present_or_refused_with_a_reason` asserted
`fam["_ci_coverage"]["lateral"]["complete"] is True` and got False. Reproducing the test's own path
and printing the block localises it exactly:

| | |
|---|---|
| `missing` | **[]** — no declared component lacks an interval |
| `unavailable` | **[]** |
| `undeclared` | **`["_along_mae_m_for_context"]`** ⇐ the whole cause |

⇒ **NOT a missing interval.** `ci_coverage`'s third question — *"a numeric leaf that is neither a
declared component nor known provenance, i.e. a metric someone added without an interval"* — fired
on the ALONG-track error that the LATERAL block carries so a reader can compare the two axes.

⭐ **AND THE QUANTITY IS BOUNDED — PROVEN, NOT ASSUMED, BEFORE SILENCING ANYTHING.**
LONGITUDINAL declares `along_mae_m` and gives it a full interval: mean **9.0068**, CI
**[7.1367, 10.8779]**, `episode_cluster_bootstrap`, 42 windows / 3 episodes. The lateral copy is the
**IDENTICAL number** (9.0068 == 9.0068). So it is genuine cross-axis context, bounded in the family
that owns it.

⛔ **IT DOES NOT BELONG IN `_NON_METRIC_KEYS`, AND PUTTING IT THERE WOULD HAVE BEEN THE EASY WRONG
FIX.** That set is documented as *"PROVENANCE / DENOMINATORS, not measurements — they must NOT carry
an interval"*. This **is** a measurement; its interval simply lives one block over. Filing it as
provenance would blur the exact distinction the instrument was built to make, and would also
silence any FUTURE leaf that happened to share the name.

⇒ a third category: **`_CROSS_FAMILY_CONTEXT`**, which NAMES the family and component that bounds
each entry — and `ci_coverage` **VERIFIES that interval actually exists** rather than trusting the
table. A pointer at a component that was never bounded falls through to `undeclared`, so the table
cannot silence a real gap by pointing at nothing. Both call sites now pass the sibling families,
without which the verification could never succeed.

⛔ **Mutation-proven:** repointing `_along_mae_m_for_context` at a component that does not exist
turns the test **RED**. File restored byte-identical. ⭐ That is the arm that matters — it proves
the fix is a *declaration with evidence*, not a suppression.

**Suite:** `test_refcv3_arm.py` **32 passed** (was 31 + 1 failed).

**Remaining at the tip: 1** — `test_runbook_commands`, where the runbook still tells an operator to
launch the **RETIRED** v6F arm with a line the trainer's own preflight refuses (no `--nav-cond`,
PI directive 2026-08-30; and `--horizons 1 2 4` declaring heads no loss consumes). ⚠️ That one is a
**content** decision about a retired arm, not a mechanical fix, and is left stated rather than
guessed at.

⭐ **Three of three examined so far were defects in the CHECKING LAYER, not in the trainer** — two
tests and one coverage instrument. Worth carrying into the last one.
