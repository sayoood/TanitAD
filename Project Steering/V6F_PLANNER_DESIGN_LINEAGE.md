# Lineage note — `V6F_PLANNER_DESIGN.md` landed under an unrelated commit message

**What happened.** `V6F_PLANNER_DESIGN.md` (644 lines), `stack/scripts/sel_winners_curse_law.py`,
`stack/tests/test_v6_selector.py`, the `GoalDistanceScorer` change in `stack/tanitad/models/v6.py`,
the `train_v6_staged.py` wiring and the three `ewc_law_*.json` measurement artifacts were all
authored by the planner-design stream — and were swept into commit **`b2e37b8`**, whose message
describes the `DIR_YAW_RAD` gate re-read, the Alpamayo-2 registry row and `DATA_STRATEGY` v2.0.

That is the **documented whole-index hazard** (`CLAUDE.md` — "git commit commits the ENTIRE INDEX,
not the files you just added"): a concurrent stream staged while another stream's commit was being
prepared. **Nothing is lost and no history was rewritten**; the work is in `b2e37b8` and pushed. Only
the *attribution* is wrong, and a reader searching the log for the planner design would not find it.

⇒ **This file exists so the search works.** The planner design is `b2e37b8`, not a missing commit.

## Independent verification performed at the time of this note (not inherited)

| claim | check | result |
|---|---|---|
| all-off build is byte-identical to pre-change HEAD | rebuilt `V6Stack(V6Config())` from `git show 30d6d60:…/v6.py` and compared tensor-by-tensor under a fixed seed | **405 keys both sides, 0 tensors differing** |
| param count unmoved | `sum(p.numel())` on the default config | **87,893,449** |
| the live S-W resume cannot be perturbed | the gate defaults to `"none"`, which draws no RNG; `emission.` and the scorer both sit in the `planner` group, and S-W sets `lambda_plan = 0.0` by construction | unaffected |

⚠️ **One correction to the stream's own report.** It quoted a `state_dict` md5 of
`a012aad286309d3283f8e055b75bfb32`; recomputing here via `torch.save` into a buffer gives
`2bc472f21957ce0d864ab4af7545cf55`. **This is not a discrepancy in the artifact** — `torch.save`
archives are not byte-stable across runs, so an md5 of the serialised buffer is not a reproducible
identity. The tensor-by-tensor comparison above is the check that carries the claim, and it passes.
⇒ **RULE: prove state-dict identity by comparing TENSORS, never by hashing a `torch.save` buffer.**
