#!/usr/bin/env python3
"""YAXN seam: YAXD's axd_seam.py UNCHANGED (A0 = the seam's own call, IDT fresh-trunk identity control, IDTm memo control, IDT gain arms), with the candidate-transform arms removed.
axd_common is patched BEFORE axd_seam is imported (UNPERT is computed at import): CANDIDATES = (), gain_arms() keeps only the IDT arms (IDT@+, IDT@-, IDT@abs+, IDT@abs-).
Everything else (the multi-arm wrap of REFePlanner.infer, trunk memo, records.npz, --regress RED arms) is axd_seam's own code, pinned by CODE_PIN_AXD.json.
    python axn_seam.py --arms-out <dir> [--regress none|transform_not_applied|idt_perturbed|memo_stale] -- <refe_navtest_seam.py args>
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import axn_common as N  # noqa: E402

A = N.A
_orig_gain_arms = A.gain_arms
A.CANDIDATES = ()
A.TRANSFORMS = ("IDT",)
A.gain_arms = lambda: [g for g in _orig_gain_arms() if g[1] == "IDT"]
import axd_seam  # noqa: E402  (YAXD; imports after the patch)

assert axd_seam.UNPERT == ("IDT", "IDTm"), axd_seam.UNPERT
assert [g[0] for g in A.gain_arms()] == ["IDT@+", "IDT@-", "IDT@abs+", "IDT@abs-"], A.gain_arms()

if __name__ == "__main__":
    sys.exit(axd_seam.main())
