"""Refuse a REWRITE-class MOD that silently DEMOTES a live figure into a retraction note.

⛔ THE HOLE THIS CLOSES. `superset_check.py:40` asks whether a value *"appears ANYWHERE in the new
text"*. That is an EXISTENCE predicate, and the claim it guards is about PLACE. A rewrite that
moves a live number out of its table row and into a sentence saying it was wrong therefore PASSES
— the token still occurs. The guard catches silent DELETION; it is blind to silent DEMOTION.

⭐ Found 2026-09-20 because the TrainingFlyWheel's own doc-checker had the identical defect and its
mutation proof exposed it: asserting the string `13.91` appeared *somewhere* stayed GREEN when the
table cell was reverted to the retracted `0.05 %`, because `13.91` still occurred elsewhere.

⛔ DELIBERATELY NARROW. Requiring a number to stay on the *same* line would refuse every honest
rewrite and block both sessions. This fires only on the specific, checkable transition:

    the value appears on at least one NON-retraction line in the TIP,
    and in the NEW text it appears ONLY on retraction-context lines.

⇒ demotion is not forbidden — `a9e75c6` deliberately kept `6.06` and marked it superseded, which is
correct. It must be DISCLOSED: waive it in `allowdemote/<basename>.allowdemote`, one token per
line, reason after `#`. The point is that demoting a figure becomes a recorded act, not a silent one.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from superset_check import NUM, CODE, norm_num          # noqa: E402  the canonical tokenisers

# A line is RETRACTION CONTEXT when it carries one of these. Conservative on purpose: a marker
# missing here only means a demotion goes unflagged (the status quo), whereas a marker that is too
# loose would flag honest prose and block a landing.
RETRACT = re.compile(
    r"RETRACT|SUPERSED|WITHDRAWN|CORRECTION|\bWRONG\b|\bwas wrong\b|NOT the floor|"
    r"\bI said\b|\bpreviously\b|\bformerly\b|no longer|OUTDATED|STALE|\bvoid\b|VOID",
    re.IGNORECASE)
TRIVIAL = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-1"}


def _positions(text: str):
    """value -> True if it occurs on at least one NON-retraction line."""
    live_n, live_c = {}, {}
    for line in text.split("\n"):
        is_live = not RETRACT.search(line)
        for m in NUM.finditer(line):
            v = norm_num(m.group(0))
            if v not in TRIVIAL:
                live_n[v] = live_n.get(v, False) or is_live
        for m in CODE.finditer(line):
            c = m.group(1).strip()
            live_c[c] = live_c.get(c, False) or is_live
    return live_n, live_c


def check(tip_text: str, new_text: str, allow: set[str]):
    t_n, t_c = _positions(tip_text)
    n_n, n_c = _positions(new_text)
    dem_n = sorted(v for v, live in t_n.items()
                   if live and v in n_n and not n_n[v] and v not in allow)
    dem_c = sorted(c for c, live in t_c.items()
                   if live and c in n_c and not n_c[c] and c not in allow)
    return dem_n, dem_c


def main(argv=None) -> int:
    a = argv or sys.argv[1:]
    tip_blob, new_file = a[0], a[1]
    allow_file = a[2] if len(a) > 2 else None
    env = dict(os.environ)
    env["GIT_DIR"] = "C:/Users/Admin/tanitad-push/.git"
    tip = subprocess.run(["git", "cat-file", "-p", tip_blob], env=env,
                         capture_output=True).stdout.decode("utf-8", "replace")
    if not tip.strip():
        print("ZZDEMOTE-INCONCLUSIVE-EMPTY-TIPZZ")
        return 2
    new = pathlib.Path(new_file).read_text(encoding="utf-8")
    allow = set()
    if allow_file and pathlib.Path(allow_file).exists():
        allow = {l.split("#")[0].strip() for l in
                 pathlib.Path(allow_file).read_text(encoding="utf-8").splitlines()
                 if l.split("#")[0].strip()}
    dn, dc = check(tip, new, allow)
    for n in dn:
        print(f"  DEMOTED-NUMBER {n}  (was live on the tip; now only in retraction context)")
    for c in dc:
        print(f"  DEMOTED-CODE   `{c}`")
    if dn or dc:
        print(f"ZZDEMOTE-FAIL {len(dn)} numbers, {len(dc)} code spans DEMOTED -- each must be "
              f"waived WITH A REASON in allowdemote/<basename>.allowdemoteZZ")
        return 1
    print("ZZDEMOTE-OKZZ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
