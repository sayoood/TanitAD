"""Compare two run_suite.py outputs (base tip export vs the edited tree). usage:
   compare_suites.py BASE.json MINE.json OUT.json OUT.md
A REGRESSION is a test that passed (or was skipped) at the base and fails / errors in the edited tree."""
import json
import sys

b = json.load(open(sys.argv[1]))["results"]
m = json.load(open(sys.argv[2]))["results"]
bad = ("FAILED", "ERROR")
reg = sorted(k for k, v in m.items() if v in bad and b.get(k) not in bad and k in b)
both_fail = sorted(k for k, v in m.items() if v in bad and b.get(k) in bad)
fixed = sorted(k for k, v in b.items() if v in bad and m.get(k) not in bad and k in m)
new_only = sorted(k for k in m if k not in b)
new_bad = [k for k in new_only if m[k] in bad]
gone = sorted(k for k in b if k not in m)


def cnt(d):
    c = {}
    for v in d.values():
        c[v] = c.get(v, 0) + 1
    return c


out = {"base": cnt(b), "mine": cnt(m), "regressions": reg, "failing_in_both": both_fail,
       "fixed_in_mine": fixed, "new_tests_only_in_mine": len(new_only), "new_tests_failing": new_bad,
       "tests_missing_from_mine": gone}
json.dump(out, open(sys.argv[3], "w"), indent=1)
L = ["# base (tip export) vs edited tree -- per-test outcomes", "",
     "| | base | mine |", "|---|---|---|"]
for k in sorted(set(out["base"]) | set(out["mine"])):
    L.append("| %s | %d | %d |" % (k, out["base"].get(k, 0), out["mine"].get(k, 0)))
L += ["", "regressions (pass/skip at base -> fail/error in mine): **%d**" % len(reg)]
L += ["- `%s`" % k for k in reg]
L += ["", "failing in BOTH (pre-existing on this box): %d" % len(both_fail),
      "fixed in mine: %d" % len(fixed), "new tests only in mine: %d (failing: %d)" % (len(new_only), len(new_bad)),
      "tests present at base but missing in mine: %d" % len(gone)]
open(sys.argv[4], "w", encoding="utf-8").write("\n".join(L) + "\n")
print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in out.items()}))
