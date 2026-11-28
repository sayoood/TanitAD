#!/usr/bin/env python3
"""run_frozen_selector.py -- run the FROZEN ``gbo_select.py`` UNCHANGED (``runpy``, its own ``__main__``) and record
every file the process OPENED (a ``sys.addaudithook`` hook), so the frame set carries the evidence that the selector
read its one input file and nothing else (no model output, no checkpoint).

    python run_frozen_selector.py <gbo_select.py> <selector input .json> <frame-set out .json> <trace out .json>

The selector's md5 (f9f93f92b60a979b581b8f78be2ec6db) and ``vis_rules.py``'s (99cd007c0fd4e539ca38fa299a948fbf) are
checked BEFORE it runs; a different byte refuses.
"""
import hashlib
import json
import os
import runpy
import sys

WANT = {"gbo_select.py": "f9f93f92b60a979b581b8f78be2ec6db", "vis_rules.py": "99cd007c0fd4e539ca38fa299a948fbf"}
OPENED: list = []


def hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0])
        if p not in OPENED:
            OPENED.append(p)


def main() -> int:
    sel, src, out, trace = sys.argv[1:5]
    d = os.path.dirname(os.path.abspath(sel))
    got = {n: hashlib.md5(open(os.path.join(d, n), "rb").read()).hexdigest() for n in WANT}
    if got != WANT:
        raise SystemExit(f"[sel] the selector is not the frozen one: {got} != {WANT}")
    sys.addaudithook(hook)
    sys.argv = [sel, src, out]
    try:
        runpy.run_path(sel, run_name="__main__")
    finally:
        own = (sys.prefix, sys.base_prefix)
        rec = {"selector_md5s": got, "argv": [os.path.basename(src), os.path.basename(out)],
               "opened": [p for p in OPENED if not p.startswith(own) and "/site-packages/" not in p
                          and not p.endswith(".pyc")],
               "out_md5": hashlib.md5(open(out, "rb").read()).hexdigest() if os.path.exists(out) else None}
        with open(trace, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
