#!/usr/bin/env python
"""handsrc_verdict.py -- prints, for a result file of read_handsrc_decomp.py, whether the controls of part 1 and of part 2 passed and the names of the controls that did not.
It prints no value of the card. Exit code 0 when part 1 passed, 1 when not, 2 when the file cannot be read.
Usage: python handsrc_verdict.py RESULT.json"""
import io
import json
import sys


def main(path):
    try:
        d = json.load(io.open(path, encoding="utf-8"))
    except Exception as ex:                                                                   # noqa: BLE001
        print("VERDICT unreadable: %s" % type(ex).__name__)
        return 2
    c1 = d.get("controls") or {}
    fam = d.get("family_read") or {}
    c2 = fam.get("controls") or {}
    bad1 = [k for k, v in c1.items() if not (isinstance(v, dict) and v.get("PASS") is True)]
    bad2 = [k for k, v in c2.items() if not (isinstance(v, dict) and v.get("PASS") is True)]
    p1, p2 = d.get("controls_all_pass") is True, fam.get("controls_all_pass") is True
    print("VERDICT card %s part1 %s (%d controls, not passed: %s) part2 %s (%d controls, not passed: %s)" % (d.get("card"), p1, len(c1), ",".join(bad1) or "none", p2, len(c2), ",".join(bad2) or "none"))
    return 0 if p1 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
