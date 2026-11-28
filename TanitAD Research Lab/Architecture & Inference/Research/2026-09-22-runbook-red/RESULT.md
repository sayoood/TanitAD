
<!-- LAST-TIP-RED-FROZEN-SELF-EXPIRING-EXCEPTION-2026-09-22 -->

### ⭐ 2026-09-22 — the LAST of the five tip reds closed: a KNOWN, DOCUMENTED, genuinely BLOCKED refusal is now a FROZEN, SELF-EXPIRING exception instead of permanent furniture

MEASURED by me, CPU only, no GPU
(`TanitAD Research Lab/Architecture & Inference/Research/2026-09-22-runbook-red/`).
Closes the series: `9373432` enumerated 5, `7de8d4e` closed 2, `d912eaf` closed 1, this closes the
**5th**.

`test_every_runbook_launch_line_passes_the_trainers_own_preflight` failed because
`V6_GO_PACKAGE.md` §2.2's production lines are refused for want of `--nav-cond` (PI directive
2026-08-30). ⛔ **And the runbook ALREADY SAYS SO** — its own CORRECTED 2026-09-18 banner states
the refusal, and names why it cannot be fixed there:

1. **§2.2 is a RENDERING of `v6_chain.py`, never an edit of it** — editing the `.md` is caught by
   `test_runbook_launch_lines_are_exactly_what_v6_chain_emits` **by design**;
2. **`ChainConfig` has NEITHER a `nav_cond` NOR a `nav_labels` field**, so the chain *cannot render
   a compliant command at all*; and `--nav-cond` without `--nav-labels` is refused too, while every
   `--nav-labels` reference in this repo is a **PLACEHOLDER** — no concrete blob exists to point at;
3. and the PI **RETIRED v6F** (*"we will go directly to v7f, we dont need revival of v6f"*), so
   nobody should be running these lines in the first place.

⛔⛔ **A PERMANENTLY-RED TEST IS WORSE THAN NO TEST, AND THIS SUITE IS THE PROOF.** The full run
carried **five** failures at the tip and nobody noticed — because everyone ran filtered slices past
reds that had become furniture. A red nobody can fix trains people to stop reading the suite, and
that is how the other four hid.

⇒ recorded as a **frozen, self-expiring** exception: the refused set must be **EXACTLY**
`{S-W, S-T, S-S, S-J}`. A **new** refused line fails the test; a **known** line that starts
**PASSING** also fails, which forces the exception to be DELETED the moment `ChainConfig` grows the
nav fields. ⭐ It cannot rot green, which is the property an `xfail` alone would not give.

⚠️ **AND THE SET IS MEASURED, NOT GUESSED.** My first instinct was `{S-W, S-S, S-T, S-O}` from the
chain's stage names; the real set contains **S-J, not S-O**. It was derived by emptying the frozen
set and reading the assertion's own failure message — which is also how anyone should re-derive it.

⛔ **Mutation-proven in BOTH directions**, because a self-expiring exception is worthless if it
cannot expire: adding a never-refused stage to the frozen set → **RED** (a known line started
passing); removing `S-W` from it → **RED** (a new line is refused). File restored byte-identical.

⛔ **What is NOT claimed:** nothing here fixes the underlying gap. `ChainConfig` still cannot render
a nav-compliant v6F command, and no concrete `--nav-labels` blob exists. Those remain open, and are
now visible as a named exception rather than as a red everyone steps over. Fixing them is a **v6F
revival decision the PI has already declined**, so it is recorded, not attempted.

**Series result: 5 of 5 tip reds closed. 5 of 5 were defects in the CHECKING LAYER** — three tests,
one coverage instrument, and one documented-blocked exception — **none in the trainer.**
