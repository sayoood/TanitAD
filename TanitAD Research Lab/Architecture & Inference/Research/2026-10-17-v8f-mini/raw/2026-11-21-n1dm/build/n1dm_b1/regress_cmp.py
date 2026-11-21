"""regress_cmp.py -- junit diff of the patched tree (regress_new) against the 1f217bf0 base (regress_base, run only on the files that failed on the patched tree). Writes REGRESSION.json."""
import glob
import json
import os
import xml.etree.ElementTree as ET

H = "F:/TanitAD-scratch/n1dc/build/n1dm_b1"


def failed(path):
    r = ET.parse(path).getroot()
    ts = r if r.tag == "testsuite" else r.find("testsuite")
    cases = list(ts.iter("testcase"))
    bad = sorted(c.get("classname", "") + "::" + c.get("name", "") for c in cases if c.find("failure") is not None or c.find("error") is not None)
    return len(cases), bad


new, base, rows = {}, {}, []
for f in sorted(glob.glob(f"{H}/regress_new/*.junit.xml")):
    n = os.path.basename(f)[:-10]
    new[n] = failed(f)
for f in sorted(glob.glob(f"{H}/regress_base/*.junit.xml")):
    n = os.path.basename(f)[:-10]
    base[n] = failed(f)
no_junit = [l.split()[0] for l in open(f"{H}/regress_new/_summary.txt", encoding="utf-8").read().splitlines() if l.strip() and "rc=" in l and not os.path.exists(f"{H}/regress_new/{l.split()[0]}.junit.xml")]
regress = []
for n, (cnt, bad) in new.items():
    if not bad:
        continue
    b = set(base.get(n, (0, []))[1])
    only_new = sorted(set(bad) - b)
    rows.append({"file": n, "tests": cnt, "failed_patched": len(bad), "failed_base": len(b) if n in base else None, "base_run": n in base, "failed_only_on_patched": only_new})
    regress += only_new
out = {"files_run": len(new), "tests_run": sum(v[0] for v in new.values()), "files_with_failures": len(rows), "files_without_junit": no_junit, "regressions": regress, "zero_regressions": not regress and not [r for r in rows if not r["base_run"]], "rows": rows}
json.dump(out, open(f"{H}/REGRESSION.json", "w", encoding="utf-8"), indent=1)
print(json.dumps({k: out[k] for k in ("files_run", "tests_run", "files_with_failures", "regressions", "zero_regressions")}))
