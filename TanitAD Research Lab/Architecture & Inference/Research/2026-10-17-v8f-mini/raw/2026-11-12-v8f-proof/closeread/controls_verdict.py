#!/usr/bin/env python
"""controls_verdict.py -- card ADD77: which controls of CLOSEREAD_CONTROLS.json passed. Prints flags and counts of rows only, never a value of a read.
  python controls_verdict.py CLOSEREAD_CONTROLS.json
"""
import json
import sys


def main(path):
    d = json.load(open(path, encoding="utf-8"))
    c = d.get("controls", {})
    parts = ["%s %s" % (k, "PASS" if c[k].get("pass") else "FAIL") for k in sorted(c)]
    print("VERDICT controls_all_pass %s | %s" % (bool(d.get("controls_all_pass")), " | ".join(parts)))
    for k in sorted(c):
        if not c[k].get("pass"):
            det = c[k].get("detail", c[k])
            flags = {kk: vv for kk, vv in (det.items() if isinstance(det, dict) else []) if isinstance(vv, (bool, str)) or kk.startswith("n_")}
            print("FAILED %s flags %s" % (k, json.dumps(flags)[:400]))
            if isinstance(det, dict):
                for kk, vv in det.items():
                    if isinstance(vv, dict) and vv.get("pass") is False:
                        sub = {a: b for a, b in vv.items() if isinstance(b, (bool, str)) or a.startswith("n_")}
                        print("  %s %s" % (kk, json.dumps(sub)[:400]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
