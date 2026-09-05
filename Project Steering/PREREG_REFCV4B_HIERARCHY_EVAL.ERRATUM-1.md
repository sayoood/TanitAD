# ERRATUM 1 to `PREREG_REFCV4B_HIERARCHY_EVAL.md` — the registered `H19-OFF` mechanism is a NO-OP on the build it is registered for

`Issued 2026-09-05 by the Master Mind. Staged separately and NOT edited into the
pre-registration: that document's falsifiable object is its staged blob id, and silently
rewriting a registered mechanism after the fact is the thing pre-registration exists to
prevent. Read the two together.`

## What the pre-registration says

§3, row `H19-OFF`: *"`decoder.maneuver_to_anchor = None` (the kin3-derived 5-way prior removed
from the confidence)"*, with `H-H19-1` predicting that switching it off degrades tactical
decision quality and anchor-selection agreement.

## What the code does — MEASURED at source

`stack/tanitad/refs/refc.py:1237` initialises `self.maneuver_to_anchor: nn.Linear | None = None`
and `:1247` assigns a real layer **only in the non-factored branch**. Every REF-C v3 / v4 build
runs `factored_maneuver = True` (forced at `refc_v3.py:437` and `:615` — *"both arms, never a
lever"*), so on `refcv4b-b1-v72-40k` the attribute **is already `None`**, and the live H19 prior
is carried by `lat_to_anchor` + `lon_to_anchor` instead.

⇒ **Executing the registered mechanism removes nothing.** The arm would have run, produced "no
change from FULL", and been read as *"the H19 seam is inert at eval"* — one of the three
conditions `§5` names as refuting the hierarchy thesis.

## Why this is the dangerous kind of error

It is not a crash and not a wrong number. It is **an ablation that cannot fail**, in a panel whose
entire purpose is to be able to fail. The same class has now been caught four times in this
programme: an ADE-scored gate that certified an echoing arm (`H-ECHO-4`); a route metric whose
label was a bijection of its own input, scoring 1.0000 while measuring nothing; a launcher whose
process filter matched its own command line; and now a switch that switches nothing off. In each
case the instrument reported success and the failure was invisible from the output alone.

⚠️ And it would have produced a **false negative about the programme's central thesis** — the
hierarchy declared inert on evidence that never touched it.

## The correction

`H19-OFF` removes **what the build actually carries**, and the arm's record states which:

* `factored_maneuver = True` (every v3/v4 build, including refcv4b) ⇒ zero
  `lat_to_anchor` **and** `lon_to_anchor`, the live kin3-derived prior on the anchor confidence.
* `factored_maneuver = False` (no shipped build) ⇒ the registered `maneuver_to_anchor = None`.

Implemented in `taniteval/tools/refcv3_arm.py` under `--ablate h19_off`, which **refuses rather
than runs** where the switch would be silently inert, and stamps the branch it took into the arm's
own manifest and every per-arm block. `H-H19-1`'s prediction and its committed outcomes are
**unchanged** — only the mechanism that realises it is corrected.

## Scope

No result is retracted: the panel has not run. This erratum exists so that when it runs, the
`H19-OFF` arm tests the seam the hypothesis names. The pre-registration's §5 refutation conditions
stand as written.

**Provenance:** found by the Rung A1 harness stream while implementing the twelve ablation flags
(`TanitAD Research Lab/Benchmarks & Evals/Implementation/incoming/2026-09-05-rung-a1-harness/`,
escalation E2); mechanism re-verified at source by the Master Mind before this erratum was issued
(`refc.py:1237`, `:1247`, `:1645`; `refc_v3.py:437`).
