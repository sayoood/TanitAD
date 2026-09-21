# I added an unstampable knob one commit ago — config, stamp, plumbing, and a mutation-proven guard

**Evidence class:** MEASURED (ours). **Compute:** CPU only, no GPU. **Date:** 2026-09-21.

## The debt, and it is mine

`6a472d1` added `AgentSlotDecoder.occ_from_geometry` — a flag that changes what the model **emits at
inference** — as a bare attribute with **no config field and no stamp entry**. `AgentSeamConfig.as_dict`'s
own docstring names the cost:

> a run record that cannot rebuild its own model config is not a run record

A run that enabled it could not be rebuilt from `config.json`, and its `occ` numbers would be
**unattributable between the learned slice and the derived read** — the anchor-units failure in a new
costume, where a correct number is quoted outside the scope that gives it meaning.

⚠️ **The existing provenance test could not have caught it.**
`test_P2_every_knob_is_recoverable_from_the_stamp_BY_VALUE` walks **argparse** dests; this knob has no
CLI flag. **A guard is only as wide as the thing it enumerates**, and that is a property of the guard,
not a fact about the code.

## The fix — both heads, config + stamp + plumbing

| file | change |
|---|---|
| `refc_agents.py` | `AgentSeamConfig.occ_from_geometry` field, an `as_dict` entry, and `build_agent_head` **applying** it |
| `refcv6_perception_branch.py` | `PerceptionBranchConfig.occ_from_geometry` field, an `as_dict` entry, and `PerceptionBranch.__init__` **applying** it to the `Box3DSlotDecoder` |

⛔ **Declared is not plumbed.** A stamped field that never reaches the module it names is **worse**
than an unstamped one, because the record then asserts a behaviour the model does not have — the
refcv6 seam defect verbatim (six channels declared, two never wired). The tests pin the **plumbing
lines**, not the dataclasses.

⚠️ **Both branches were checked, and the first probe was wrong.** A grep over
`PerceptionBranchConfig`'s field block found no `as_dict` and I nearly recorded "the 3-D branch has no
stamp". A second probe — for how the trainer records it — found `as_dict` at
`refcv6_perception_branch.py:190` **and** `perception_stamp` at `refc_v3_train.py:6183`. *Absence at
one location is not absence*, caught by the rule rather than by luck.

## The guard — key sets pinned as LITERALS

`AGENT_SEAM_STAMP_KEYS` and `PERCEPTION_STAMP_KEYS` are frozen literals. Asserting "every dataclass
field appears in `as_dict`" would be an **expression over the code under test** and would pass for
any future knob added to both at once — including one added to neither. A literal forces a deliberate
edit, which is exactly the discipline the five-site gate value earned.

## Mutation-proven 5/5, with named catchers

| arm | defect | caught by |
|---|---|---|
| S1 | agent stamp entry dropped (**the original defect, restored**) | key-set pin + both value tests |
| S2 | perception stamp entry dropped | perception key-set pin + both plumbing tests |
| S3 | stamp **hardcodes the default** instead of the value | `test_the_stamp_reports_the_value_that_was_actually_set[True]` |
| S4 | agent builder stops applying the field | `..._ACTUALLY_REACHES_the_built_agent_head[True]` |
| S5 | perception branch stops applying the field | `..._ACTUALLY_REACHES_the_built_box3d_head[True]` |

Baseline green, all arms RED, both files restored **byte-identical** by md5, final clean run green.

⭐ **S5 is caught only because I fixed a test that was lying about itself.** The first version of
`test_the_config_field_ACTUALLY_REACHES_the_built_box3d_head` asserted only `as_dict` — it passed,
and it would have passed with the plumbing line deleted. **A test whose NAME claims more than its
body checks is worse than a missing test, because it reads as coverage.** It now builds the
`PerceptionBranch` and reads the attribute off the decoder that was actually constructed.

## ⛔⛔ Two defects in the prover itself, both of which produced a verdict about the TESTS

1. **CRLF anchors.** These files are CRLF; four of five anchors ended in `\n` and matched **nothing**.
   The prover skipped S1, S2, S4 and S5 in silence and printed **"ONLY 1/5 CAUGHT"** — a verdict
   about the guard manufactured entirely by a defect in the prover. ⇒ anchors now carry no line
   terminator, and **an arm that cannot be applied ABORTS the run as INVALID** rather than being
   folded into an `n/total` ratio. *An arm that never applied is not a failed arm; it is no arm at
   all* — the same family as scoring an empty output as "not caught".
2. **A silent no-op edit.** The heredoc patch that was supposed to replace the `MUTATIONS` block
   matched nothing and reported success; only the *second* replacement in the same script took
   effect, which is why the abort fired while the anchors stayed broken. Fixed with `Edit`, whose
   failure is loud.

Together with `mutate_occ_geometry.py`'s cp1252 defect, that is **three separate ways a mutation
prover reported a conclusion about the code when the fault was in the prover** — and all three read
as findings about the tests. ⇒ a prover needs its own controls exactly as much as an estimator does.

## Suite

`stack/tests/` filtered to `agent or slot or box3d or occ or perception or provenance`:
**537 passed, 6 skipped, 1 failed**. The failure is the same
`test_P2_every_knob_is_recoverable_from_the_stamp_BY_VALUE` (`--w-r7-wta` / `--refcv7`) already
**verified pre-existing** against the tip's own file in `6a472d1`. No new failures.

## Manifest

| file | what |
|---|---|
| `stack/tanitad/refs/refc_agents.py` | agent seam: field, stamp entry, plumbing (REWRITE) |
| `stack/tanitad/models/refcv6_perception_branch.py` | perception branch: field, stamp entry, plumbing (REWRITE) |
| `stack/tests/test_occ_knob_is_stamped.py` | 8 tests: two frozen key sets, value fidelity, both plumbing paths |
| `code/mutate_occ_stamp.py` | the prover, with its CRLF and invalid-arm defects fixed and documented |
| `raw/mutation_proof_occ_stamp.json` | 5/5 RED, named catchers, restore verified by md5 |
