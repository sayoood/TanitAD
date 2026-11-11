"""Dev tool (not shipped to box 2): theory/wmpool/argvs_wmpool.txt -> frozen/argvs_wmpool.json (token lists), prints its sha256; --patch writes that sha256 into wmpool_check.py's ARGVS_SHA256 line.
Usage: python -I make_frozen_wmpool.py <argvs_wmpool.txt> <out.json> [--patch <wmpool_check.py>]"""
import hashlib
import json
import re
import sys

WHAT = {"P-ATT_s0": "attention pool of the 2 x 5 adapter cells, seed 0 on M2b_s0", "P-ATT_s1": "attention pool of the 2 x 5 adapter cells, seed 1 on M2b_s1",
        "P-ATT_rep_s2": "attention pool of the 2 x 5 adapter cells; replicate, seed 2 on the M2b_s0 start state and cell cache"}
src, out = sys.argv[1], sys.argv[2]
d = {}
for line in open(src, encoding="utf-8"):
    if line.strip() and not line.startswith("#"):
        tag, argv = line.rstrip("\r\n").split("\t", 1)
        d[tag] = {"what": WHAT[tag], "argv": argv.split()}
assert list(d) == list(WHAT), list(d)
with open(out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(d, f, indent=1)
    f.write("\n")
sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
print(sha)
if len(sys.argv) > 4 and sys.argv[3] == "--patch":
    p = sys.argv[4]
    t = open(p, encoding="utf-8", newline="").read()
    t2 = re.sub(r'ARGVS_SHA256 = "[^"]*"', f'ARGVS_SHA256 = "{sha}"', t, count=1)
    assert t2 != t or sha in t
    open(p, "w", encoding="utf-8", newline="").write(t2)
