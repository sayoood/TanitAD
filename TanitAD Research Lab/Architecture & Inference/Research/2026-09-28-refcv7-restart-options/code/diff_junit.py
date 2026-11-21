#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-test outcome diff of `run_neighbour_suites.sh` (combined tree vs launch tree).

Every test id is classified pass / fail / error / skip on each side from the JUnit XML; the report
lists NEW failures (fail on combo, pass on base), FIXED (the reverse), tests only on one side, and
the per-file counts. An absent XML is reported as UNREAD, never as zero tests.

Usage: diff_junit.py <suites out dir> [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def outcomes(xml: Path) -> dict[str, str]:
    out = {}
    root = ET.parse(xml).getroot()
    for tc in root.iter("testcase"):
        tid = f"{tc.get('classname')}::{tc.get('name')}"
        st = "pass"
        for child in tc:
            if child.tag in ("failure", "error"):
                st = child.tag
            elif child.tag == "skipped":
                st = "skip"
        out[tid] = st
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.dir)
    files = sorted({p.stem for p in (d / "combo").glob("*.xml")} | {p.stem for p in (d / "base").glob("*.xml")})
    rep = {"files": {}, "new_failures": [], "fixed": [], "only_combo": [], "only_base": [],
           "unread": []}
    for f in files:
        c, b = d / "combo" / f"{f}.xml", d / "base" / f"{f}.xml"
        oc = outcomes(c) if c.is_file() else None
        ob = outcomes(b) if b.is_file() else None
        if oc is None:
            rep["unread"].append(f"combo/{f}")
        if ob is None:
            rep["unread"].append(f"base/{f}") if f not in ("test_refcv7_admitted_freeze",
                                                          "test_v2_cache_pickle_state",
                                                          "test_refcv6_loader_stamped_queries") \
                else rep["only_combo"].append(f"{f} (NEW file)")
        cnt = lambda o: None if o is None else {k: sum(1 for v in o.values() if v == k)
                                               for k in ("pass", "failure", "error", "skip")}
        rep["files"][f] = {"combo": cnt(oc), "base": cnt(ob)}
        if oc is not None and ob is not None:
            for t in sorted(set(oc) | set(ob)):
                if t not in ob:
                    rep["only_combo"].append(t)
                elif t not in oc:
                    rep["only_base"].append(t)
                elif oc[t] in ("failure", "error") and ob[t] == "pass":
                    rep["new_failures"].append(t)
                elif ob[t] in ("failure", "error") and oc[t] == "pass":
                    rep["fixed"].append(t)
    txt = json.dumps(rep, indent=1)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)
    return 0 if not rep["new_failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
