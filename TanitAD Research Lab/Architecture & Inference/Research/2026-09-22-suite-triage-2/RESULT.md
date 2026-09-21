
<!-- SUITE-TRIAGE-2-BOTH-REDS-WERE-TEST-DEFECTS-2026-09-22 -->

### ⭐ 2026-09-22 — two more of the five tip reds closed, and BOTH were defects in the TESTS, not in the trainer — each mutation-proven to still catch the real thing

MEASURED by me, CPU only, no GPU
(`TanitAD Research Lab/Architecture & Inference/Research/2026-09-22-suite-triage-2/`).
Continues `9373432`, which enumerated 5 failures at the tip. Two down, three to go.

#### 1. ⛔ `test_the_guard_is_WIRED_into_the_config_write_not_merely_defined` — a TEXTUAL predicate guarding a claim about EXECUTION ORDER

The test compared `src.index("_assert_speed_max_stamp(...)")` against
`src.index('(out_dir / "config.json").write_text')` and reported `425435 < 190431` as *"the stamp
guard runs AFTER config.json is written"*.

**MEASURED: the trainer is CORRECT.** There are **FOUR** `config.json` writes — the PRIMARY one at
line **7257**, immediately after the guard at **7249**, and three RE-writes that append keys to the
same dict afterwards (`conflict_detector`, `trunk_bn_recalib`, and the BN-recalib finalisation).
The last of those lives in a helper **defined earlier in the file but called later**, at line 3302.
`src.index` takes the FIRST occurrence, so the assertion compared the guard against a write it was
never meant to precede.

⭐ **Byte offsets are a proxy for execution order only when there is exactly one write, and nothing
enforced that** — the same family as the programme's *"an EXISTENCE predicate cannot guard a claim
about PLACE"*. ⇒ the check is now scoped **via AST** to the function that CONTAINS the guard, and
asserts the guard precedes every `config.json` write **in that same function**.

⛔ **Mutation-proven, because a red test turned green is the worst outcome if it was weakened:**
moving the guard call to AFTER the primary write turns the corrected test RED with a precise
message (`min([7256, 7315, 7325])` — the AST correctly seeing `train()`'s three writes and not the
helper's). Trainer restored byte-identical.

#### 2. ⛔ `test_REFC_WEIGHT_GATES_covers_every_weight_flag_the_parser_accepts` — three flags the NAME PATTERN catches that are not term weights

The test enumerates weight flags **lexically** (`"-w-" in o`), so `--r7-w-ttc`, `--r7-w-ep` and
`--r7-w-comf` were reported ungated.

⛔⛔ **AND THE NAIVE FIX WOULD HAVE REFUSED EVERY EXISTING ARM.** MEASURED: those three default to
**5.0 / 5.0 / 2.0**, while `refc_weight_specs` states *"every refc weight defaults to 0.0 ON
PURPOSE, so that adding a seam to the code cannot change a run that does not ask for it"*.
`effective_weights.classify` sets `NO_GRAPH` on `effective > 0.0 and missing` **without consulting
`explicit`**, and `refusals()` refuses `NO_GRAPH` — so a registry entry gated on the scorer would
fire on the **defaults** of every run that does not pass `--refcv7`. *Adding the "obvious" entries
would have broken the whole trainer.*

⇒ they are excluded by name **with their reason**, in the shape this test's own docstring names
(`PATH_ARGS ∪ NOT_A_PATH`): they are RELATIVE weights INSIDE the scorer's single 7-sub-score BCE,
and the TERM they live in — `--w-r7-scorer` — **is** registered and carries the whole gate. A new
test **asserts that transitive gating rather than assuming it**, and pins the three defaults as
literals so the measured reason cannot quietly stop being true. The exclusion set is also asserted
to be a subset of the parser's real flags, so it cannot rot into a dumping ground.

⚠️ **The red was pointing at something real and it is recorded, not silenced:** three refcv7 flags
break the registry's zero-default invariant. Changing their defaults is a **refcv7 design
decision**, not a test fix.

⛔ **Mutation-proven:** removing `w_r7_scorer` from `REFC_WEIGHT_GATES` turns BOTH the exhaustiveness
test and the new transitive test RED. Trainer restored byte-identical.

#### Remaining at the tip: 3

`test_runbook_commands` (the runbook tells operators to run a **retired** v6F launch line the
preflight refuses — missing `--nav-cond`, and `--horizons 1 2 4` declaring heads no loss consumes),
and `test_refcv3_arm::test_every_family_is_present_or_refused_with_a_reason`. ⭐ Two of two examined
so far were TEST defects, not trainer defects — worth knowing before "fixing" the remaining ones.
