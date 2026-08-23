"""Splice `artifacts/t1_tables.md` into SITUATION_SEMANTICS.md at its markers.

usage:  python t1_splice.py <artifacts dir> <doc.md>
"""
from __future__ import annotations

import sys

A, B = "<!-- TABLES:T1 -->", "<!-- /TABLES:T1 -->"


def main():
    tbl = open(f"{sys.argv[1]}/t1_tables.md", encoding="utf-8").read().rstrip()
    doc = open(sys.argv[2], encoding="utf-8").read()
    i, j = doc.index(A), doc.index(B)
    out = doc[:i + len(A)] + "\n\n" + tbl + "\n\n" + doc[j:]
    open(sys.argv[2], "w", encoding="utf-8").write(out)
    print(f"spliced {len(tbl)} chars into {sys.argv[2]}")


if __name__ == "__main__":
    main()
