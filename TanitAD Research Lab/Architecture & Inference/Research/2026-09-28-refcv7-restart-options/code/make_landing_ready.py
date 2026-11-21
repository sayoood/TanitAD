#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write LANDING_READY.txt for this package (coordinated landing, PI ruling 2026-09-26).

Every file of the package lands AT ITS PACKAGE PATH (the Research Lab tree is outside the launch
gate's stack/ taniteval/ tools/ tree, so landing it touches no token and no closure). The
closure-touching code (and the files that only make sense with it) is listed a SECOND time in its
own section, with NO `->` repo target: landing any of it into stack/ would void both binding overfit
records for every future restart. Each line carries the file's git blob in a `#` comment.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parents[3]
HEAD = "## 2026-09-28 REFCV7-RESTART-OPTIONS"
#: package file -> the stack path it would replace/add AT A RESTART (never landed there now)
CLOSURE = {
    "code/freeze/stack/tanitad/refs/refc_v3.py":
        ("stack/tanitad/refs/refc_v3.py", "IN BOTH closures (launch blob a5e71567)"),
    "code/freeze/stack/tanitad/train/declared_vs_built.py":
        ("stack/tanitad/train/declared_vs_built.py", "IN BOTH closures (launch blob 87cad54e)"),
    "code/getstate/stack/tanitad/data/v2_dataset.py":
        ("stack/tanitad/data/v2_dataset.py", "IN BOTH closures (launch blob ee61ea93)"),
    "code/i3/stack/tanitad/eval/refcv6_loader.py":
        ("stack/tanitad/eval/refcv6_loader.py", "IN the box closure (launch blob 14450c78)"),
    "code/i3/stack/scripts/g_box_overfit.py":
        ("stack/scripts/g_box_overfit.py", "IN the box closure (launch blob ad88b61e)"),
    "code/i3/stack/tests/test_g_box_overfit.py":
        ("stack/tests/test_g_box_overfit.py", "not in a closure; re-pins the I3 loader blob -- lands WITH it"),
    "code/freeze/stack/scripts/refcv7_ckpt_freeze_convert.py":
        ("stack/scripts/refcv7_ckpt_freeze_convert.py", "not in a closure; only meaningful with the freeze"),
    "code/freeze/stack/tests/test_refcv7_admitted_freeze.py":
        ("stack/tests/test_refcv7_admitted_freeze.py", "not in a closure; RED without the freeze (12/12)"),
    "code/getstate/stack/tests/test_v2_cache_pickle_state.py":
        ("stack/tests/test_v2_cache_pickle_state.py", "not in a closure; RED without the fix (5/5)"),
    "code/i3/stack/tests/test_refcv6_loader_stamped_queries.py":
        ("stack/tests/test_refcv6_loader_stamped_queries.py", "not in a closure; RED without the fix (3/4)"),
    "code/argv/refcv7-r101-s0.conflict50.argv.json":
        ("stack/ops/runs.d/refcv7-r101-s0.argv.json (its `argv` LIST only; apply_bundle.py keeps the "
         "canonical file's metadata -> blob 033a6507, gate sha e46ad3eb)",
         "the argv lever: not a closure MODULE, but the box closure hashes this file and both records "
         "bind the launch argv sha256"),
}
LEVER2_NOTE = ("# lever 2 itself is NOT a file here: it is applied from the landed step-cost package "
               "(…/2026-09-27-refcv7-step-cost/code/apply_lever2.py, stack/scripts/refc_v3_train.py "
               "203b437f -> fd874293, IN BOTH closures) by code/apply_bundle.py")


def blob(p: Path) -> str:
    b = p.read_bytes()
    return hashlib.sha1(b"blob %d\x00" % len(b) + b).hexdigest()


def main() -> int:
    rel_pkg = PKG.relative_to(REPO).as_posix()
    files = []
    for dp, dn, fn in os.walk(PKG):
        dn[:] = sorted(d for d in dn if d not in ("__pycache__", ".pytest_cache"))
        for f in sorted(fn):
            p = Path(dp) / f
            if f == "LANDING_READY.txt" or f.endswith(".pyc"):
                continue
            files.append(p)
    lines = [HEAD + ": restart-options package (decision table, closure map, freeze + getstate + "
             "I3 fixes, speed levers, numerics battery) -- package paths only; nothing lands in "
             "stack/ now"]
    lines.append(f"{rel_pkg}/LANDING_READY.txt")
    for p in sorted(files, key=lambda q: q.relative_to(PKG).as_posix()):
        r = p.relative_to(PKG).as_posix()
        lines.append(f"{rel_pkg}/{r}  # blob {blob(p)}")
    lines.append("")
    lines.append(HEAD + " CLOSURE-TOUCHING (package only)")
    lines.append("# ⛔ NO repo target. Each file below is what code/apply_bundle.py writes into a CLEAN "
                 "tree at a restart the PI decides; landing any of it into stack/ now voids "
                 "G-MAP-OVERFIT and G-BOX-OVERFIT for every future restart.")
    lines.append(LEVER2_NOTE)
    lines.append("# the conflict-50 argv also moves stack/tests/test_launch_gate.py's _FINAL_ARGV_SHA256 "
                 "pin (374dd421 -> e1ad8320); code/apply_bundle.py writes it, no package copy is kept")
    for r, (dest, why) in CLOSURE.items():
        p = PKG / r
        lines.append(f"{rel_pkg}/{r}  # blob {blob(p)} -- would become {dest}: {why}")
    (PKG / "LANDING_READY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:5]), "...", len(lines), "lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
