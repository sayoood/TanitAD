"""Build suite_compare.md from the two sibling-suite JSONs (+ the extra re-runs that supersede a layout-fixed entry), then rebuild the package.  usage: python finish_pkg.py
Reads suite_mine.json / suite_base.json (suite_runner.py outputs) and suite_extra_{mine,base}.json (re-runs after a layout export: they SUPERSEDE the main entry of the same file).
Verdict per file: SAME (both rc 0 with equal pass counts), MORE (both rc 0, mine has a higher count: new checks), FAILS-BOTH (rc != 0 on both: pre-existing in this layout/environment, not caused by the change),
NEW FAILURE (base rc 0, mine rc != 0), DIFFERENT COUNT (equal rc 0 but mine LOWER: a lost check), PENDING (not run yet)."""
import json
import os
import subprocess
import sys

ROOT = "C:/Users/Admin/par18/hnfp2"


def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {}


mine, base = load(f"{ROOT}/suite_mine.json"), load(f"{ROOT}/suite_base.json")
for tag, d in (("mine", mine), ("base", base)):
    d.update(load(f"{ROOT}/suite_extra_{tag}.json"))
order = [ln.strip() for ln in open(f"{ROOT}/list_mine.txt", encoding="utf-8") if ln.strip() and not ln.startswith("#")]


def cnt(e):
    return "/".join(str(c[1]) for c in e["counts"]) if e and e.get("counts") else "-"


rows, new_fail, lost, pending = [], [], [], []
for spec in order:
    m, b = mine.get(spec), base.get(spec)
    if spec == "refe/test_hn_content_fp.py":
        v = "NEW FILE (mine only)" if m and m["rc"] == 0 else ("PENDING" if not m else "NEW FILE FAILS")
    elif not m or not b:
        v = "PENDING"
        pending.append(spec)
    elif m["rc"] == 0 and b["rc"] == 0:
        cm, cb = [c[1] for c in m["counts"]], [c[1] for c in b["counts"]]
        v = "SAME" if cm == cb else ("MORE (new checks)" if sum(cm) > sum(cb) else "LOWER COUNT")
        if v == "LOWER COUNT":
            lost.append(spec)
    elif m["rc"] != 0 and b["rc"] != 0:
        v = "FAILS ON BOTH (pre-existing in this layout/environment)"
    elif b["rc"] == 0:
        v = "NEW FAILURE"
        new_fail.append(spec)
    else:
        v = "base FAILS, mine passes"
    rows.append(f"| {spec} | {b['rc'] if b else '-'} | {m['rc'] if m else '-'} | {cnt(b)} | {cnt(m)} | {b['skips'] if b else '-'}/{m['skips'] if m else '-'} | {v} |")
txt = ["| file | base rc | mine rc | base count | mine count | skips b/m | verdict |", "|---|---|---|---|---|---|---|", *rows, "",
       f"NEW FAILURES: {new_fail or 'none'}; LOWER COUNTS: {lost or 'none'}; PENDING: {pending or 'none'}"]
open(f"{ROOT}/suite_compare.md", "w", encoding="utf-8", newline="\n").write("\n".join(txt) + "\n")
print("\n".join(txt))
if os.path.isfile(f"{ROOT}/logs/suite_mine/refe__test_hn_content_fp.py.log"):
    import shutil
    shutil.copyfile(f"{ROOT}/logs/suite_mine/refe__test_hn_content_fp.py.log", f"{ROOT}/logs/test_hn_content_fp_final.log")
# the generated table goes into RESULT.md between its two markers
rp = f"{ROOT}/RESULT.md"
if os.path.isfile(rp):
    t = open(rp, encoding="utf-8").read()
    a, b = t.find("<!--SIBLINGS-->"), t.find("<!--/SIBLINGS-->")
    if a >= 0 and b > a:
        NL = chr(10)
        open(rp, "w", encoding="utf-8", newline=NL).write(t[:a] + "<!--SIBLINGS-->" + NL + NL.join(txt) + NL + t[b:])
subprocess.run([sys.executable, f"{ROOT}/make_pkg.py"], check=True)
