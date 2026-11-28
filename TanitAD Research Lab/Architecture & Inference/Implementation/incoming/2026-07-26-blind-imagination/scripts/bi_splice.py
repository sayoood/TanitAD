#!/usr/bin/env python3
"""Splice the rendered tables into BLIND_IMAGINATION.md at its markers.

`bi_report.py` renders `artifacts/_tables.md` as blocks delimited by
`<!-- TABLES:NAME -->` / `<!-- /TABLES:NAME -->`. This script replaces the
matching region in the report, so re-running the analysis re-renders the report
and no number is ever hand-edited.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def blocks(text):
    out = {}
    for m in re.finditer(r"<!-- TABLES:(\w+) -->(.*?)<!-- /TABLES:\1 -->",
                         text, re.S):
        out[m.group(1)] = m.group(0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tables")
    ap.add_argument("report")
    a = ap.parse_args()
    src = blocks(Path(a.tables).read_text(encoding="utf-8"))
    doc = Path(a.report).read_text(encoding="utf-8")
    n = 0
    for name, block in src.items():
        pat = re.compile(rf"<!-- TABLES:{name} -->.*?<!-- /TABLES:{name} -->",
                         re.S)
        if pat.search(doc):
            doc = pat.sub(lambda _m: block, doc)
            n += 1
        else:
            print(f"  [warn] no marker for {name} in the report")
    Path(a.report).write_text(doc, encoding="utf-8")
    print(f"spliced {n}/{len(src)} blocks into {a.report}")


if __name__ == "__main__":
    main()
