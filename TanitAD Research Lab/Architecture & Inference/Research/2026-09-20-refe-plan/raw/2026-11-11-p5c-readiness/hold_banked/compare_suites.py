"""Compare per-suite rc between the unmodified tip (res/base) and the hold (res/work) from runall.sh summaries. stdlib only.  usage: compare_suites.py <res dir>"""
import sys, re, glob, os
res = sys.argv[1]
def read(t):
    d = {}
    for f in sorted(glob.glob(os.path.join(res, t, "_summary*.txt"))):
        for ln in open(f, encoding="utf-8"):
            m = re.match(r"(test_\S+) rc=(\d+) (\d+)s ?(.*)", ln.strip())
            if m:
                d[m.group(1)] = (int(m.group(2)), m.group(4).strip())
    return d
b, w = read("base"), read("work")
for k in sorted(set(b) | set(w)):
    bb, ww = b.get(k), w.get(k)
    print(f"{k:34s} base {bb}  work {ww}  {'SAME' if bb == ww else 'DIFF'}")
