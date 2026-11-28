# -*- coding: utf-8 -*-
"""Sharpen E-READOUT-CEILING-1 now that the FULL real-trunk panel has landed.

The first pass said "the fine structure is NOT present in the tokens either".
The completed panel says something more precise and more useful: it IS present as
MEMORISABLE structure (the fit-AP column reproduces the pooling ladder) and is
absent from TRANSFER. Same CAS discipline as register_update.py.
"""
import os
import shutil
import subprocess
import sys

REG = (r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
       r"\Project Steering\GOALS_AND_CLAIMS.md")
TMP = r"C:\Users\Admin\wpa-readout\_register_refined.md"
GUARD = "fit-AP column reproduces the pooling ladder"

OLD_ROW = ("\u2b50 **This ANSWERS `E-DEC-29`'s open NEXT** \u2014 the fine per-column structure is "
           "**not** present in the TOKENS either, so on this trunk the deficit is **upstream of the "
           "readout**, not the readout.")
NEW_ROW = ("\u2b50 **This ANSWERS `E-DEC-29`'s open NEXT, and the answer is a DISSOCIATION.** The "
           "**fit-AP column reproduces the pooling ladder** on the real trunk \u2014 16x40 **0.2049** "
           "\u2192 8x20 0.1743 \u2192 the 4-row rungs ~0.13\u20130.14 \u2192 1x1 **0.0581** \u2248 the marginal "
           "control's 0.0359, at IDENTICAL head parameter count \u2014 so the pool really does destroy "
           "addressable content here too. \u26d4 **But NONE of it transfers**: on test every feature arm "
           "is 0.027\u20130.034 against `pos_only` **0.0325**, and the `shuffled` control lands on "
           "**fit 0.0361 / test 0.0325**, indistinguishable from `pos_only`'s **0.0359 / 0.0325** \u2014 "
           "which is what proves the fit ladder is feature-driven and not an artefact of the head. "
           "\u21d2 the encoder's addressable content is **EPISODE-SPECIFIC, not agent-generic**, so on "
           "this trunk the deficit is **upstream of the readout** \u2014 at the objective and the corpus, "
           "not at the pooling.")

OLD_NOTE = ("The same probe on the 16x40 TOKEN grid of `k8clip05p30k` reads "
            "**negative R\u00b2 on every lateral target** (linear), and an azimuth-indexed nonlinear "
            "head **fits its training split (AP 0.2049 vs the marginal control's 0.0359) and "
            "transfers nothing** (test 0.0295 vs pos_only 0.0325).")
NEW_NOTE = ("The same probe on the 16x40 TOKEN grid of `k8clip05p30k` reads **negative R\u00b2 on every "
            "lateral target** (linear), and an azimuth-indexed nonlinear head shows a **DISSOCIATION**: "
            "its **fit-AP column reproduces the pooling ladder** (16x40 **0.2049** \u2192 1x1 **0.0581** "
            "\u2248 the marginal control's 0.0359, at identical parameter count) while **test AP is flat "
            "at 0.027\u20130.034 against `pos_only` 0.0325**, with the `shuffled` control on "
            "fit 0.0361 / test 0.0325.")

OLD_HEAD = ("\u2b50 **ANSWERED 2026-09-07 \u2014 `E-READOUT-CEILING-1`: it is NOT present in the "
            "tokens either.**")
NEW_HEAD = ("\u2b50 **ANSWERED 2026-09-07 \u2014 `E-READOUT-CEILING-1`: it IS in the tokens as "
            "MEMORISABLE structure and ABSENT from transfer.**")


def blob(path):
    out = subprocess.run(["git", "hash-object", path], capture_output=True,
                         text=True, cwd=os.path.dirname(REG))
    h = out.stdout.strip()
    return h if len(h) == 40 else None


before = blob(REG)
if before is None:
    sys.exit("INCONCLUSIVE: could not hash the register - retry")
src = open(REG, encoding="utf-8").read()
if src.count(GUARD):
    sys.exit(f"ALREADY APPLIED: guard appears {src.count(GUARD)}x - refusing")
for name, old in (("row", OLD_ROW), ("note", OLD_NOTE), ("head", OLD_HEAD)):
    assert src.count(old) == 1, f"{name} anchor count {src.count(old)} != 1"

new = src.replace(OLD_ROW, NEW_ROW).replace(OLD_NOTE, NEW_NOTE).replace(OLD_HEAD, NEW_HEAD)
assert new.count(GUARD) == 2, new.count(GUARD)
assert new.count("E-READOUT-CEILING-1") == 2
assert src.count("| E-DEC-29 |") == new.count("| E-DEC-29 |") == 1
assert len(new) > len(src)
open(TMP, "w", encoding="utf-8", newline="").write(new)

now = blob(REG)
if now is None:
    sys.exit("INCONCLUSIVE: could not re-hash before write - retry")
if now != before:
    sys.exit(f"CONFLICT: register moved {before} -> {now}; re-run to rebase")
shutil.copyfile(TMP, REG)

back = open(REG, encoding="utf-8").read()
print(f"[verify] guard {back.count(GUARD)} (want 2) \u00b7 claim id {back.count('E-READOUT-CEILING-1')} "
      f"(want 2) \u00b7 len {len(back)} (want {len(new)})")
assert back.count(GUARD) == 2 and len(back) == len(new)
print("OK")
