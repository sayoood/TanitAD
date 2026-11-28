# ⛔ `--init-from` would have killed the S-T launch on its first command

**2026-08-16 · branch `agent/arch-inf-20260803` · found while dry-running the S-T selector arms.**
⛔ Nothing was trained. Thor's v6F S-W run was not touched.

---

## 1. The defect, MEASURED

`load_stage_init` (`stack/scripts/train_v6_staged.py`) is the other half of X5: stage *N+1* starts
from stage *N*'s checkpoint, so the ladder is one lineage rather than four unrelated models. It
loaded with torch's `strict=True`. Running the real S-W → S-T transition:

```
S-W ckpt saved: 405 keys
  S-T selector=none  -> OK   missing=[] unexpected=[]
  S-T selector=goal  -> RuntimeError: Missing key(s) in state_dict:
                        "cand_score.cand_bias", "cand_score.log_tau",
                        "cand_score.goal_point.weight", ...
  S-T selector=mlp   -> RuntimeError: Missing key(s) in state_dict:
                        "cand_score.cand_bias", "cand_score.fc1.weight",
                        "cand_score.fc1.bias", ...
```

**Both selector arms — the entire point of S-T — were unlaunchable.** S-W never builds a selector
(`selector="none"` by construction, which is what makes the world stage attributable), so its
checkpoint cannot contain one; S-T is *where the selector is built*, and it is *supposed* to start
from a fresh initialisation.

*Evidence class: MEASURED (ours), reproduced on a real `V6Stack` state_dict round-trip.*

## 2. Why the guard was right in spirit and blind in practice

Its docstring names the failure it exists to prevent, and that failure is real:

> *A key mismatch means the two stages were built with DIFFERENT geometry — silently allowing that
> is how a stage ends up training on a randomly-initialised trunk while its log looks healthy.*

`strict=True` cannot separate two very different situations:

| situation | correct verdict |
|---|---|
| the **TRUNK** is missing (`encoder.*`, `readout.*`, `predictor_op.*`) | ⛔ **FATAL** — the stage would train on a random encoder and the loss curve would look fine |
| the **PLANNER'S NEW HEAD** is missing (`cand_score.*` at S-T) | ✅ **EXPECTED** — this is the designed introduction |

⚠️ **And `strict=False` is NOT the fix.** It would equally wave through a missing `emission.*` and
silently random-init the whole emission head — the same class of disaster one module over.

## 3. The fix: an explicit per-stage ALLOWLIST

```python
STAGE_MAY_INTRODUCE: dict[str, tuple[str, ...]] = {
    "S-W": (),                  # starts the ladder; nothing to inherit
    "S-T": ("cand_score.",),    # the selector is built HERE, by design
    "S-S": (),                  # trains layer_str, which S-T already carried
    "S-J": (),                  # joint polish introduces nothing
}
```

`load_stage_init` now always loads non-strict and **adjudicates**:

1. `unexpected` keys → **always fatal** (the ckpt carries something this stack does not build — that
   genuinely is a geometry mismatch, and no allowlist covers it);
2. `missing` keys under an allowed prefix → **introduced**, reported under their own
   `introduced_keys` field so a run row can never confuse *"this stage BUILT a new head"* with
   *"this stage FAILED to load one"*;
3. every other `missing` key → **fatal**, with a message naming them and the allowance;
4. ⛔ an allowed prefix must be **WHOLLY absent** from the checkpoint. A **partially present**
   module is a geometry mismatch wearing an allowance's clothes, and stays fatal — otherwise half a
   head would be random-initialised and the load would report success;
5. a caller that does not name the stage gets **no allowance at all** — an allowance must be asked
   for, never inherited by default.

## 4. A second, quieter defect fixed with it — the C13 family

Under torch's `strict=True` a mismatch **raises**, it does not return. So the `missing_keys` and
`unexpected_keys` this function reported into the run row **could only ever be empty**. The report
was structurally incapable of describing the thing it was named for — *a guard that cannot fail*
(**C13**), in a reporting field rather than a threshold. Loading non-strict and adjudicating makes
both fields carry real content for the first time.

## 5. Evidence

| check | result |
|---|---|
| `tests/test_v6_stage_init_introduction.py` (**NEW**, 10 tests) | **10 passed** |
| whole v6 set (staged + selector + capacity + revalidation + init + ckpt-layout + probe-trunk) | **140 passed** |
| S-T `selector=goal` init from a real S-W ckpt | **OK**, `introduced_keys` = the 4 `cand_score.*` keys |
| S-T `selector=mlp` init from the same ckpt | **OK**, `introduced_keys` = the 5 `cand_score.*` keys |
| missing `encoder.*` | **still fatal** — the case the original guard existed for |
| `unexpected` keys | **still fatal** |
| half a `cand_score` present | **still fatal** |
| `S-S` with an unbuilt selector | **still fatal** (no allowance at that stage) |
| trunk md5 reproducible across two loads with different selectors | **identical** |

⚠️ The trunk md5 is a **per-tensor** hash over `named_parameters()`, not a hash of the file —
`torch.save` writes a zip whose bytes are not canonical, so a file hash proves the container and not
the tensors (**RETRACTION_LOG C68**). A test pins this by loading the same checkpoint into two
different-selector stacks and asserting the trunk hashes agree.

## 6. What this says about the ladder generally

This is the **third** ladder-edge defect found in two days, and they rhyme:

| edge | defect |
|---|---|
| S-S → S-T (backward) | S-S invalidates S-T's frozen selector certificate; **no gate existed** (fixed `dc50dbc`) |
| S-W → S-T (forward) | the init guard refused the designed introduction; **both arms unlaunchable** (this) |
| S-W → S-T (arms) | `"goal"` had no capacity control, so a win would have been unattributable (fixed `b12c190`) |

⇒ **Every ladder edge deserves an executed transition test, not a reasoned one.** All three were
found by *running* the transition, and none by reading the code. The forward edge S-S → S-J and the
resume edge (`--resume auto` across a stage boundary) have **not** been executed end-to-end and
should be, before either is needed.

## Deliverable manifest

| artifact | where it lives | state |
|---|---|---|
| `STAGE_MAY_INTRODUCE` + the adjudicating `load_stage_init` | `stack/scripts/train_v6_staged.py` | staged |
| `stack/tests/test_v6_stage_init_introduction.py` (10 tests) | repo | staged |
| this document | `…/incoming/2026-08-16-init-from-launch-blocker/` | staged |

## Escalation

⭐ **The remaining untested ladder edges are a work item, not a note.** S-S → S-J and a
stage-boundary `--resume auto` have never been executed. Both are 0-GPU to test with a tiny stack,
and both sit on the critical path after S-T. Given that executing the S-W → S-T edge immediately
surfaced a launch blocker that reading it did not, the prior for the untested edges is not good.
