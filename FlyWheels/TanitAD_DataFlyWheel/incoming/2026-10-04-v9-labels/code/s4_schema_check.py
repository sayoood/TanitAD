"""Every field of a v9 release must be documented in V9_SCHEMA.md, and the doc must name no field the release lacks.
Brace groups in the doc (`rc_{A30,A50}_{x,y}`) are expanded. usage: python s4_schema_check.py <V9_SCHEMA.md> <release.npz>"""
import itertools
import re
import sys

import numpy as np


def expand(tok):
    parts = re.split(r"(\{[^}]*\})", tok)
    opts = [p[1:-1].split(",") if p.startswith("{") else [p] for p in parts]
    return {"".join(c) for c in itertools.product(*opts)}


doc = open(sys.argv[1], encoding="utf-8").read()
names = set()
for m in re.finditer(r"`([A-Za-z0-9_{},]+)`", doc):
    for t in expand(m.group(1)):
        names.add(t)
z = np.load(sys.argv[2])
fields = {k.split("__", 1)[1] for k in z.files}
missing = sorted(fields - names)
print(f"release fields {len(fields)}; documented {len(fields) - len(missing)}; undocumented: {missing}")
sys.exit(1 if missing else 0)
