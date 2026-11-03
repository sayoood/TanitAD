#!/usr/bin/env python3
"""ka_rarm_r.py -- the known answer of the --r-arm extension (RECOVER-NEXT card section 3): ``--r-arm R`` must reproduce dagland/recover_read/raw/RECOVER_READ.json EXACTLY.

Compares raw/RECOVER_READ_R.json (this package's reader, arm R, recomputed from the existing rollouts: file controls, 15 fresh extractions, every row at B = 2000, seed 0) with the source's JSON, key for key and value for value
(floats compared with ``==`` after the JSON round trip: no tolerance).  The only paths ignored are the two that cannot be equal by construction: ``run_clock`` and ``reader_sha256`` (the extension is a different file).
Also compares each freshly extracted per-window array npz with the source package's, bit for bit.  Writes raw/KA_RARM_R.json; exit 0 iff zero differences.
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
SRC_DIR = "E:/Projects/TanitAD-artifacts/dagland/recover_read/raw"
IGNORED = ("run_clock", "reader_sha256")


def diff(a, b, path=""):
    """-> list of (path, kind, a_value, b_value); kind in only_new | only_source | type | value."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            p = f"{path}.{k}" if path else k
            if not path and k in IGNORED:
                continue
            if k not in b:
                out.append((p, "only_new", None, None))
            elif k not in a:
                out.append((p, "only_source", None, None))
            else:
                out += diff(a[k], b[k], p)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, "length", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
    elif type(a) is not type(b):
        out.append((path, "type", repr(a)[:60], repr(b)[:60]))
    elif a != b:
        out.append((path, "value", repr(a)[:60], repr(b)[:60]))
    return out


def count_leaves(o):
    if isinstance(o, dict):
        return sum(count_leaves(v) for v in o.values())
    if isinstance(o, list):
        return sum(count_leaves(v) for v in o)
    return 1


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(argv=None):
    raw = (argv or sys.argv[1:] or [f"{PK}/raw"])[0]
    new = json.load(open(f"{raw}/RECOVER_READ_R.json", encoding="utf-8"))
    old = json.load(open(f"{SRC_DIR}/RECOVER_READ.json", encoding="utf-8"))
    d = diff(new, old)
    arrs = {}
    for f in sorted(os.listdir(f"{SRC_DIR}/arrays")):
        if not f.endswith(".npz"):
            continue
        p_new = f"{raw}/arrays/{f}"
        if not os.path.exists(p_new):
            arrs[f] = {"equal": False, "why": "absent in the new package"}
            continue
        za, zb = np.load(p_new, allow_pickle=True), np.load(f"{SRC_DIR}/arrays/{f}", allow_pickle=True)
        bad = [k for k in sorted(set(za.files) | set(zb.files)) if k not in za.files or k not in zb.files or not np.array_equal(za[k], zb[k], equal_nan=(za[k].dtype.kind in "fc"))]
        arrs[f] = {"equal": not bad, "keys": len(za.files), "differing_keys": bad}
    res = {"known_answer": "--r-arm R reproduces recover_read/raw/RECOVER_READ.json exactly", "new": f"{raw}/RECOVER_READ_R.json", "new_sha256": sha(f"{raw}/RECOVER_READ_R.json"),
           "source": f"{SRC_DIR}/RECOVER_READ.json", "source_sha256": sha(f"{SRC_DIR}/RECOVER_READ.json"), "ignored_top_level_keys": list(IGNORED),
           "ignored_values": {k: {"new": new.get(k), "source": old.get(k)} for k in IGNORED}, "n_leaves_compared": count_leaves(old), "n_differences": len(d),
           "differences": [{"path": p, "kind": k, "new": a, "source": b} for p, k, a, b in d][:200], "arrays_bit_identical_to_the_source": arrs, "n_arrays": len(arrs),
           "label_new": new.get("label"), "label_source": old.get("label")}
    res["PASS"] = bool(not d and arrs and all(v["equal"] for v in arrs.values()))
    json.dump(res, open(f"{raw}/KA_RARM_R.json", "w", encoding="utf-8", newline="\n"), indent=1)
    print("KA_RARM_R", "PASS" if res["PASS"] else "FAIL", "| leaves", res["n_leaves_compared"], "| differences", len(d), "| arrays bit-identical", sum(v["equal"] for v in arrs.values()), "/", len(arrs))
    for p, k, a, b in d[:25]:
        print("  DIFF", k, p, a, b)
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
