"""Numbers audit of report v46: every numeric token of the lines ADDED to the v45 base (python -I audit_v46.py tip [out_dir [audit_path]]) must be found in a source (the dev-box theory folders
of check_numbers_v46.py and the tip register), in the v45 base text, be structural (integers below 100, years, section ids) or be in the DERIVED list (each recomputed by check_numbers_v46.py).
Exit 1 on any unmatched token."""
import os, re, subprocess, sys
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TIP = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"
OUT = (sys.argv[2] if len(sys.argv) > 2 else HERE + "out").rstrip("/\\") + "/"
AUD = sys.argv[3] if len(sys.argv) > 3 else HERE + "v46_NUMBERS_AUDIT.md"
TH = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/"
base = open("E:/Projects/TanitAD-artifacts/dagland/report_v45/out/V8F_SCIENTIFIC_REPORT.md", encoding="utf-8").read()
new = open(OUT + "V8F_SCIENTIFIC_REPORT.md", encoding="utf-8").read()
reg = subprocess.run(["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*", "show", TIP + ":Project Steering/GOALS_AND_CLAIMS.md"], capture_output=True).stdout.decode("utf-8")
norm = lambda t: re.sub(r"(?<=\d),(?=\d{3})", "", t.replace("\u2212", "-").replace("**", ""))
tok = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:e-?\d+)?(?![\w])")
srcs = {}
for d in ("plantest2", "plantest3", "ladder_read", "ladder_plantest", "ladder", "decinfo", "otband"):
    for r, _, fs in os.walk(TH + d):
        for f in fs:
            p = os.path.join(r, f)
            if f.endswith((".md", ".json", ".log", ".txt")) and os.path.getsize(p) < 8_000_000 and "__pycache__" not in r:
                for x in tok.findall(norm(open(p, encoding="utf-8", errors="replace").read())):
                    srcs.setdefault(x.lstrip("+-"), p.replace(TH, ""))
for x in tok.findall(norm(reg)):
    srcs.setdefault(x.lstrip("+-"), "register")
inbase = set(tok.findall(norm(base)))
DERIVED = {"0.29", "35", "45", "0.2571", "0.2573"}
STRUCT = re.compile(r"7\.\d\d(?:\.\d+)?|5\.3|2026|\d{1,2}")
c0, c1 = Counter(base.split("\n")), Counter(new.split("\n"))
added = [l for l in (c1 - c0).elements() if l.strip()]
miss, cnt = {}, Counter()
for l in added:
    for t in tok.findall(norm(re.sub(r"`[^`\n]*`", " ", l))):
        if t in srcs:
            cnt["source"] += 1
        elif t in inbase:
            cnt["v45 text"] += 1
        elif t in DERIVED:
            cnt["derived"] += 1
        elif STRUCT.fullmatch(t) or (float(t) < 100 and "." not in t):
            cnt["structural"] += 1
        elif t.rstrip("0") in srcs or ("%g" % float(t)) in srcs:
            cnt["trailing zero"] += 1
        else:
            miss.setdefault(t, l[:90])
out = ["# v46 numbers audit", "", "tip " + TIP, "", "added lines %d; tokens by origin: %s; unmatched: %d" % (len(added), dict(cnt), len(miss)), ""]
out += ["UNMATCHED %s | %s" % kv for kv in sorted(miss.items())]
open(AUD, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print(out[4])
for x in out[6:]:
    print(x)
sys.exit(1 if miss else 0)
