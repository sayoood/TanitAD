#!/usr/bin/env python3
"""E-LAB-BE-0901 -- is the LONGITUDINAL distance-keeping family FED on the eval path?

`LAB_BACKLOG` row 19 (verify-first): the instrument (`lead_metrics.py`, admitted by the
pre-registered D-LEAD-1 control) existed UNFED as of 2026-08-03. A binding metric family
reported UNAVAILABLE while its instrument sits idle is a criteria-completeness violation.

This censuses every JSON artifact in `taniteval/results/` and buckets it:
  NUMERIC     - carries an actual distance_keeping / headway / time_gap / min_ttc number
  NON-NUMERIC - carries the key but with an unavailability reason (and WHICH reason)
  NO-KEY      - does not mention the family at all
  UNREADABLE  - could not be opened (see below)

⚠️ THE UNREADABLE BUCKET IS NOT OPTIONAL AND MUST NOT BE FOLDED IN. CLAUDE.md: the G: mount
"FORGES TEST FAILURES ... a G: failure count is admissible ONLY with zero Errno 22 in the run".
An unread file is UNKNOWN, never "absent" -- collapsing the two would let a mount fault
manufacture a cleaner result than the evidence supports. The count is reported over READABLE
files, with the unreadable ones named.
"""
from __future__ import annotations

import glob
import json
import os
import sys

KEYS = ("distance_keeping", "headway", "time_gap", "min_ttc")


def walk(o, found):
    if isinstance(o, dict):
        for k, v in o.items():
            if any(s in str(k) for s in KEYS):
                found.append((k, v))
            walk(v, found)
    elif isinstance(o, list):
        for v in o:
            walk(v, found)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "taniteval/results"
    dst = sys.argv[2] if len(sys.argv) > 2 else "dk_census.json"
    files = sorted(glob.glob(os.path.join(root, "*.json")))

    numeric, nonnum, nokey, unreadable = [], [], [], []
    reasons: dict[str, int] = {}

    for f in files:
        base = os.path.basename(f)
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
        except OSError as e:                      # the Errno 22 / mount class
            unreadable.append({"file": base, "error": f"{type(e).__name__}: {e}",
                               "size_bytes": (os.path.getsize(f)
                                              if os.path.exists(f) else None)})
            continue
        except Exception as e:                    # genuine parse failure
            unreadable.append({"file": base, "error": f"PARSE {type(e).__name__}: {e}"})
            continue

        found: list = []
        walk(d, found)
        if not found:
            nokey.append(base)
        elif any(isinstance(v, (int, float)) for _, v in found):
            numeric.append({"file": base,
                            "values": [(k, v) for k, v in found
                                       if isinstance(v, (int, float))][:5]})
        else:
            nonnum.append(base)
            for _, v in found:
                reasons[str(v)[:120]] = reasons.get(str(v)[:120], 0) + 1

    out = {
        "meta": {
            "root": root, "n_files": len(files),
            "n_readable": len(files) - len(unreadable),
            "what_this_is": ("census of the LONGITUDINAL distance-keeping family across banked "
                             "eval artifacts. Counts are over READABLE files; unreadable files "
                             "are NAMED and excluded, never counted as absent."),
        },
        "counts": {
            "numeric": len(numeric), "key_present_non_numeric": len(nonnum),
            "no_key": len(nokey), "unreadable": len(unreadable),
        },
        "numeric_files": numeric,
        "non_numeric_files": nonnum,
        "distinct_unavailable_reasons": reasons,
        "no_key_files": nokey,
        "unreadable_files": unreadable,
    }
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out["counts"], indent=1))
    print("distinct reasons:", json.dumps(reasons, indent=1))
    print("wrote", dst)


if __name__ == "__main__":
    main()
