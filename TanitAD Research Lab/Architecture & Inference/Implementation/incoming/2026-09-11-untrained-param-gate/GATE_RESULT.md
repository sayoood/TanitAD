# The post-training gate for untrained parameters — built, mutation-proved, and it found a NEW arm

**Date:** 2026-09-11 · **Agent:** Arch+Inference FlyWheel · **Evidence class:** MEASURED (ours)
**Instrument:** `stack/scripts/check_untrained_params.py` · **Suite:** `stack/tests/test_check_untrained_params.py` (21 tests)
**Predecessor:** `…/incoming/2026-09-10-refcv5v2-zerograd-heads/probe_zerograd3.py` (63 lines, no functions)

## 1. What it is

A checkpoint-level gate for the defect class that has surfaced **five times** in this programme
and was caught too late every time: a head that is built, unit-tested, registered, counted in
the parameter breakdown — and never reached by the loss. ⭐ *"Rollable and trained are
different claims."*

The method is **analytic, not statistical**. `torch.optim.Adam` allocates per-parameter state
**lazily**, on the first `step()` at which `p.grad is not None`. A parameter listed in
`opt["param_groups"][*]["params"]` with **no entry in `opt["state"]`** therefore never received
a gradient. No threshold, no seed, no re-run.

⛔ **The verdict is the JSON file, never the exit code.** MEASURED in this programme: a
25-minute `timeout` killed a pre-launch gate with zero output and no JSON written, and its
wrapper printed `GATE_EXIT=0`. ⭐ *"The admissible evidence that this gate did not run is the
MISSING JSON."* The JSON is written atomically (tmp + `os.replace`) and is pure ASCII, so
neither a crash mid-write nor a cp1252 console can leave a truncated artifact that reads as
finished. `status` is `PASS` **only** when everything needed was actually read.

## 2. It reproduces refcv5-v2 EXACTLY

`C:/Users/Admin/refcv5v2_final/ckpt.pt`, md5 `9405ec73b2d797c4cebd44f82dbce54b` (verified by the
gate itself, recorded in the JSON), step 40,284.

| quantity | required | measured |
|---|---|---|
| registered parameters | 357 | **357** |
| with optimizer state | 351 | **351** |
| untrained total | 18,472 | **18,472** |
| `core.decoder.offset_head` | 6,160 | **6,160** |
| `tac_goal_tok_head.net` | 11,286 | **11,286** |
| `scorer.goal_point` | 1,026 | **1,026**, flagged **exactly zero** |

All three of the predecessor probe's validations are carried forward and recorded in the JSON:
**351/351** shape matches; mapped total **108,257,502** equal to `config.json`'s
`param_breakdown.total` (an **independently authored** reference the gate never touches); and
same-breath controls `str_goal_head` / `scorer.cand_bias` / `core.decoder.conf_head` all
reading TRAINED with a nonzero optimizer moment, which excludes a global-zero artifact.

⭐ **The `exactly_zero` flag has its own discriminator inside the same checkpoint.** The other
four untrained tensors read `absmax` **non-zero with every element non-zero**
(`offset_head.weight` 0.0510, all 6,144; `tac_goal_tok_head.net.weight` 0.0442, all 11,264), so
"exactly zero" is a property of `scorer.goal_point`, not of the test.

Artifact: `raw/refcv5v2_final_gate.json`.

## 3. ⭐ The alignment is now PROVED FORCED, not greedily guessed

The one inferential step is that optimizer `state_dict` keys are **positional ids, not names**.
The predecessor aligned ids to names by **greedy shape matching** and its own docstring flagged
that as the fragile part.

This gate computes the order-preserving embedding **twice — leftmost and rightmost**. An id's
position is **forced** exactly when the two agree, because every feasible embedding places it
at `>= leftmost` and `<= rightmost`. A disagreement is a **proof of ambiguity**, and the gate
returns INCONCLUSIVE rather than emitting a name it cannot justify.

⛔ **The refusal is reachable, not decorative — MEASURED.** Re-running refcv5-v2 with
`--no-buffer-filter` (so BatchNorm's `running_mean`/`running_var` become candidates and collide
in shape with real parameters) takes the candidate list 360 → 493, slack 3 → 136, and
**132 of 357 positions become ambiguous**. The gate returns INCONCLUSIVE and names nothing.
Pinned by `test_ambiguous_alignment_REFUSES_rather_than_guessing`.

## 4. ⛔ The regression arm — watched RED, watched GREEN

The state-set difference (`id_shapes[k] is not None`, i.e. registered-minus-has-state) **is**
the analytic method. It was removed from the real script on disk and the suite re-run.

| arm | result | artifact |
|---|---|---|
| **MUTATED** (difference removed) | ⛔ **12 failed, 9 passed** | `raw/suite_MUTATED_red.log` |
| **REVERTED** (md5-verified back to pristine) | ✅ **21 passed** | `raw/suite_REVERTED_green.log` |

⭐ **Two things the RED run proved that a green run cannot.**
1. `test_reproduces_refcv5v2_exactly` **failed under mutation** (`assert 'PASS' == 'FAIL'`), so
   it really loads and evaluates the 1.3 GB artifact — it is not a silent skip.
2. `test_a_fully_trained_checkpoint_reports_EXACTLY_zero_untrained` **still passed under
   mutation**, because a blind gate also reports 0 on a fully-trained checkpoint. That is
   precisely why the analytic control alone is insufficient and the mutation arms are load-bearing.

Three further mutation arms live **inside** the suite (state-set difference, allow-everything
allow-list, blinded `exactly_zero`). Each asserts its anchor string occurs **exactly once**, so
a refactor that removes the anchor makes the arm go RED rather than quietly vacuous.

## 5. ⭐ THE RESULT THAT OUTRANKS THE TOOL: two findings the gate produced on its first run

### 5a. FINDING.md's open `offset_head` hypothesis is CONFIRMED — at zero GPU cost

The predecessor left this explicitly open: *"HYPOTHESIS: the ddim configuration orphans the
classifier-pass offset head … ⛔ Not confirmed. The discriminating check is cheap: an arm with
`--sampler` unset should show `offset_head` **with** optimizer state."*

Both banked non-ddim arms were run. ⚠️ **Absence from the untrained list is not evidence**, so
the head's presence and state were asserted **positively** through the gate's own control path:

| arm | `--sampler` | `core.decoder.offset_head` |
|---|---|---|
| `refcv5v2_final` | `ddim` | **UNTRAINED** — no optimizer state, 6,160 params at init |
| `refcv3_final` | *unset* | **TRAINED** — has state, moment abs sum **55.818** |
| `refcv4b_final` | *unset* | **TRAINED** — has state, moment abs sum **58.360** |

⇒ **CONFIRMED.** `offset_head` is bypassed **by the ddim sampler**, not accidentally unreached.
That downgrades one of refcv5-v2's three untrained heads from *defect* to *expected under this
sampler*, and it removes it as a candidate mechanism for refcv5-v2's failed `--sel-refined` bar.
⚠️ Scope it: on a **non-ddim** arm the same head being untrained **would** be a defect.

`tac_goal_tok_head` reads `found: false` on both older arms — the head is not built there at
all, which is why its absence from their untrained lists means nothing about it.

### 5b. ⛔ NEW: `scorer.goal_point` is untrained and exactly zero in **THREE** arms, not one

FINDING.md established this for refcv5-v2 only. MEASURED 2026-09-11 across the banked family:

| arm | registered | with state | untrained | `scorer.goal_point` |
|---|---|---|---|---|
| `refcv5v2_final` | 357 | 351 | 18,472 | **1,026, exactly zero** |
| `refcv3_final` | 343 | 341 | **1,026** | **1,026, exactly zero** |
| `refcv4b_final` | 349 | 347 | **1,026** | **1,026, exactly zero** |

`v6.py` zero-initialises this head; zero-init **plus** zero gradient is an **identity**, not a
noisy estimate — the "free regression" half of the goal scorer has emitted the **constant
origin** for every input in **every refc arm checked**. No seed changes it.

⚠️ **This inherits FINDING.md's open question and widens it.** That note flagged that whether
the panel's `goal_setting FDE 0.6607 m / bearing_MAE 1.3730 deg` row reads **this head** or the
param-free `anchor_goal_prior_at_time` path was **NOT established**. It is still not
established, and it now bears on **three** arms rather than one. If that row reads this head,
it is a measurement of a constant. ⛔ Until settled, the `goal_setting` row is UNVERIFIED for
refcv3, refcv4b **and** refcv5-v2. This is the cheapest remaining lever and it is read-only.

Artifacts: `raw/refcv3_final_gate.json`, `raw/refcv4b_final_gate.json`.

## 6. How deliberately frozen is distinguished from accidentally unreached

⛔ *A gate that cannot tell them apart is switched off within a week.* The expected-frozen set
is **explicit** and anything outside it fails. `expect_frozen_refc_ddim.json` carries **one**
entry — `core.decoder.offset_head`, with §5a as its evidence — and deliberately **omits** the
two genuine defects. MEASURED with that allow-list on refcv5-v2:

```
core.decoder.offset_head.weight/.bias   6,160    allowed(core.decoder.offset_head)
tac_goal_tok_head.net.weight/.bias     11,264+22 UNEXPECTED
scorer.goal_point.weight/.bias          1,024+2  EXACTLY-ZERO UNEXPECTED
-> FAIL: 4 parameters totalling 12,312 not allow-listed
```

An allowed parameter is **still named in the JSON** with the entry that covered it — a gate
that hides what it allowed cannot be audited. An entry matching nothing is reported as
**stale** and fails under `--strict-allowlist`, because an allow-list nobody prunes is how a
gate rots into a rubber stamp.

## 7. The false-positive generators it refuses to become

| hazard | what the gate does | pinned by |
|---|---|---|
| **Plain SGD** allocates *no* state ⇒ would name **every** parameter untrained | detects the precondition fails, returns INCONCLUSIVE, names **nothing** | `test_plain_sgd_is_INCONCLUSIVE_not_everything_untrained` |
| ambiguous id→name alignment | refuses; names nothing | `test_ambiguous_alignment_REFUSES_rather_than_guessing` |
| mapped total ≠ independent reference | INCONCLUSIVE, **never** PASS — even when the allow-list covers every name | `test_a_wrong_parameter_total_is_INCONCLUSIVE_never_PASS` |
| cross-check 2 silently not happening | INCONCLUSIVE unless waived **in the argv**, and the waiver is recorded in the JSON | `test_a_missing_parameter_total_reference_must_be_WAIVED_explicitly` |
| unreadable / absent / corrupt checkpoint | INCONCLUSIVE **and still writes the JSON** | `test_an_unreadable_checkpoint_emits_INCONCLUSIVE_and_still_writes_the_JSON` |
| a control that itself reads untrained | INCONCLUSIVE — the gate has no standing to report | `test_a_named_control_that_reads_untrained_is_INCONCLUSIVE` |

⚠️ Every zero asserted in the suite is paired with a **same-breath control that must read
non-zero**, because a count of 0 from a probe that read nothing is indistinguishable from a
genuine 0. Every expectation is a **literal** hand-derived from the fixture architecture
(`nn.Linear(4, 8)` is 40 parameters because that is what `nn.Linear` means) — never an
expression over the code under test.

## 8. ⚠️ Found in passing, NOT fixed — two RED tests at the tip, same defect family

`pytest --collect-only` on `stack/` reports **7,712 tests collected, 2 errors**. Both are
**pre-existing** (my working-tree footprint is exactly two new files) and both are
import failures for symbols that do not exist at HEAD:

* `tests/test_refa_v1_dk_hook.py` → `cannot import name 'DistanceKeepingSpec' from 'tanitad.refs.refa_v1'`
* `tests/test_metric_decode_refusal.py` → `cannot import name 'UntrainedMetricReadout' from 'tanitad.models.v6'`

Verified positively: `git show HEAD:stack/tanitad/models/v6.py | grep -c UntrainedMetricReadout`
reads **0**, with a same-breath control on the same file reading **38** `class ` hits, so the
grep really read the file.

⭐ **This is the fifth row of the brief's own table** — *"the posttrain spec guard existed only
inside its own test; the tip carried a RED test while the register said CLOSED"* — live at the
tip right now. It is outside this task's scope and is **not** fixed here, but it should not be
discovered again in six weeks.

## 9. Next levers, cheapest first

1. **Settle the `goal_setting` provenance** (read-only, minutes). Does the panel's goal row
   read `scorer.goal_point` or the param-free anchor prior? It decides whether a published row
   on **three** arms is a measurement of a constant. This is the top item.
2. **Wire the gate into the post-training path** so a checkpoint is gated at the artifact
   rather than after evaluation and publication. It is a single CPU invocation and needs no GPU.
3. **Run it across the remaining banked checkpoints** (`refcv5cmp`, `a40-rescue`, the frozen
   428 MB variants) — the instrument is generic and each run is minutes.
4. **Decide whether `scorer.goal_point` should be supervised or removed.** It has never
   trained in any arm; a head that emits the constant origin is either a missing loss term or
   dead weight, and either way it should not be silently registered.

## 10. Deliverable manifest

| artifact | where it lives | also elsewhere? |
|---|---|---|
| `stack/scripts/check_untrained_params.py` | `repo:` staged | no — repo only |
| `stack/tests/test_check_untrained_params.py` (21 tests) | `repo:` staged | no — repo only |
| `expect_frozen_refc_ddim.json` | `repo:` staged (this dir) | no |
| `raw/refcv5v2_final_gate.json` | `repo:` staged | no |
| `raw/refcv3_final_gate.json` | `repo:` staged | no |
| `raw/refcv4b_final_gate.json` | `repo:` staged | no |
| `raw/suite_MUTATED_red.log` | `repo:` staged | no |
| `raw/suite_REVERTED_green.log` | `repo:` staged | no |
| `GATE_RESULT.md` (this file) | `repo:` staged | no |

⛔ **STAGED, NOT COMMITTED** — per the operating standard. The Master Mind commits.
⚠️ **Escalation, in the headline and not in a README:** item 9.1 is a **read-only** check that
decides whether a published `goal_setting` row on three arms is a constant, and item 9.2 wires
this gate into the post-training path so the sixth instance of this defect is caught at the
checkpoint instead of after publication.

⚠️ The checkpoints themselves (`C:/Users/Admin/refcv5v2_final/`,
`D:/Projects/TanitAD-artifacts/`) are **off-repo, single-disk** artifacts and are not staged
here; only their gate verdicts are.
