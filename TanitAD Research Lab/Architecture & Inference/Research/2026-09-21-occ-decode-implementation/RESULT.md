# `occ_from_geometry` — the measured fix, implemented, opt-in, and mutation-proven

**Evidence class:** MEASURED (ours). **Compute:** CPU only, no GPU, no training.
**Date:** 2026-09-21. **Lands:** a code change to a live module + its guard.

## Why

`6c5fb62` measured that the box head's **learned** `occ_logit` loses to a two-parameter read of a
number the head already emits — its own predicted azimuth — by **0.1276 log-loss,
CI [−0.1844, −0.0376]**, on 848 scored pairs over 18 episodes. The whole box (d = 5) adds nothing
over azimuth alone (0.16211 vs 0.16055), exactly as the identity predicts. It survives the
matcher-selection confound in every stratum and **never wins**: it merely ties where the box is
badly wrong, and is 6.5× worse where the box is accurate.

That finding named a fix with **zero training cost**, and then sat undone across two firings. This
lands it.

## What changed — three edits, all additive

| site | change |
|---|---|
| `agent_slots.py` | `OCC_HALF_ANGLE_RAD = radians(60.0)` and `OCC_TEMPERATURE = 6.6902`, each carrying the measurement that produced it |
| `agent_slots.py` | `occ_logit_from_centre(cx, cy, …)` — the identity as a logit, zero parameters |
| `agent_slots.py` | `AgentSlotDecoder.occ_from_geometry: bool = False` + a conditional in `decode` |

⛔ **The default is OFF and the learned expression stays spelled out in `decode`.** Nothing any
existing caller emits changes. Flipping the flag changes what the model *emits at inference*, which
is a contract change and therefore **the PI's call, not a default**.

⭐ **The flag is an attribute, not a constructor argument, on purpose.** `Box3DSlotDecoder`
subclasses `AgentSlotDecoder` and replaces `self.head`; an attribute is inherited with no
forwarding and needs no config plumbing, so the blast radius is one file. A test pins that the 3-D
head — the one actually scored — really does inherit it.

⚠️ **Scope, stated:** the derivation is exact in the TARGET, so its accuracy is bounded by the
accuracy of the predicted centre it reads. It removes a lossy re-encoding; it does **not**
manufacture localisation the head does not have.

## The guard — mutation-proven, 5/5 with named catchers

`code/mutate_occ_geometry.py` reintroduces each real defect and requires the suite to go RED.
Inspection would not do: an AST census once read 0 suspects on both the fixed and the broken
trainer.

| arm | defect reintroduced | caught by |
|---|---|---|
| M1 | half-angle drifts out of sync with `bev_raster.fov_mask` | `test_half_angle_is_pinned_to_fov_mask_not_merely_documented` + a sign case |
| M2 | logit sign inverted | 4 hand-computed sign cases |
| M3 | `abs()` dropped from the azimuth | the two negative-`cy` / behind-ego cases |
| M4 | decode reads the RAW slice, not the DECODED centre | `test_it_reads_the_predicted_centre_not_the_raw_slice` + 2 |
| M5 | the opt-in silently defaults ON | `test_default_is_off_so_the_emitted_contract_is_unchanged` + 2 |

Baseline green, **all 5 arms RED with named failures**, target restored **byte-identical**
(md5 `055a22c8…`), final clean run green. `raw/mutation_proof_occ_geometry.json`.

⭐ **The expectations are LITERALS, never expressions over the code under test.** Asserting
`sign(logit) == (az > OCC_HALF_ANGLE_RAD)` with the module's own constant would move both sides
together and be green forever; the boundary cases carry their own hand-computed geometry
(`tan 60° = √3`, so `|cy| = √3·cx` is exactly on the boundary).

⭐ **And the half-angle is pinned to `bev_raster.fov_mask` by reading its signature default**, not
by a comment. Two sites carrying one physical fact is precisely how the gate-value drift happened.

## ⛔⛔ The prover's own defect, which it reported as a finding about the tests

The first run scored **4/5**, marking M4 — the UNITS arm — as *not caught*, with an empty catcher
list. Hand-running the identical mutation failed **3 named assertions**. The prover was wrong, and
the cause is a trap already in this programme's record:

> `subprocess.run(..., text=True)` decodes the CHILD's output with the **parent's** locale, which is
> **cp1252** here. M4's pytest traceback echoes the failing test's docstring, which carries a
> non-cp1252 byte. The reader thread raised `UnicodeDecodeError`, **both stdout and stderr came back
> EMPTY**, and rc was 1.

⇒ the prover read "RED with no named failures" and, under the hardened rule that a syntax-error red
does not count, scored it as uncaught. **An empty read is a claim about the PIPE, never about the
tests** — the same family as `grep` reporting 0 hits for a file it could not open.

⚠️ **And the arms most likely to trip it are the ones that are WORKING**, because they produce the
richest failure output. A mutation prover with this bug systematically under-counts exactly the
guards that function — and would have had me weaken a correct test.

Fixed by decoding `utf-8` with `errors="replace"`, and by making **empty output INCONCLUSIVE**
rather than scoring it either way.

## Suite

`stack/tests/` filtered to `agent or slot or box3d or occ`: **368 passed, 5 skipped, 1 failed**.
The one failure is `test_P2_every_knob_is_recoverable_from_the_stamp_BY_VALUE`, on
`--w-r7-wta` / `--refcv7`. ⛔ **Verified PRE-EXISTING, not inherited as an assumption:** the tip's
own `agent_slots.py` was checked out over mine and the test fails identically. It is a real
provenance gap in the refcv7 knobs and is left for its own landing rather than folded in here.

## Manifest

| file | what |
|---|---|
| `stack/tanitad/models/agent_slots.py` | the constants, `occ_logit_from_centre`, the opt-in flag (REWRITE, superset-checked) |
| `stack/tests/test_occ_from_geometry.py` | 15 tests, literal expectations, the `fov_mask` pin |
| `code/mutate_occ_geometry.py` | the mutation prover, with its own cp1252 defect fixed and documented |
| `raw/mutation_proof_occ_geometry.json` | 5/5 RED, named catchers, restore verified by md5 |
