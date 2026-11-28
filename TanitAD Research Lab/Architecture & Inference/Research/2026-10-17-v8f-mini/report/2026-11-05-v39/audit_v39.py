"""Numbers audit of report v39: every number token of the ADDED non-table lines is re-found in the source blobs of the tip (every md / txt / json blob under raw/2026-11-02-* to raw/2026-11-05-*: the RPT, motion-c, level-head
and LLR files, the earlier 7.34 to 7.36 reads they cite, the register) or in the carried v38 text, or it is structural (integers below 100, years, ids). Unmatched tokens stop the audit. Also lists, per source file, how many were found where.
python -I audit_v39.py tip [out_dir [audit_path]]   (git reads only; CPU only)"""
import os, re, subprocess, sys
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TIP = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"
OUT = (sys.argv[2] if len(sys.argv) > 2 else HERE + "out").rstrip("/\\") + "/"
AUD = sys.argv[3] if len(sys.argv) > 3 else HERE + "v39_NUMBERS_AUDIT.md"
M = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"


def git(args):
    r = subprocess.run(["git", "-C", M, "-c", "safe.directory=*"] + args, capture_output=True)
    assert r.returncode == 0, (args, r.stderr[:200])
    return r.stdout


def show(p):
    return git(["show", f"{TIP}:{p}"]).decode("utf-8").replace("\r", "")


names = [n for n in git(["ls-tree", "-r", "--name-only", TIP, "--", R + "raw"]).decode("utf-8").split("\n")
         if re.search(r"/2026-11-0[2-5]-", n) and n.endswith((".md", ".txt", ".json"))]
pr = subprocess.Popen(["git", "-C", M, "-c", "safe.directory=*", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
srcs = []
for n in sorted(names):
    pr.stdin.write(f"{TIP}:{n}\n".encode()); pr.stdin.flush()
    size = int(pr.stdout.readline().decode().split()[2])
    data = pr.stdout.read(size); pr.stdout.read(1)
    if size <= 3_000_000:
        srcs.append((n.replace(R + "raw/", ""), data.decode("utf-8", "replace").replace("\r", "")))
pr.stdin.close(); pr.wait()
srcs.append(("register", show("Project Steering/GOALS_AND_CLAIMS.md")))
v38 = show(R + "report/2026-11-04-v38/V8F_SCIENTIFIC_REPORT.md")  # the carried base (v38)
new = open(OUT + "V8F_SCIENTIFIC_REPORT.md", encoding="utf-8").read()
norm = lambda t: t.replace("\u2212", "-").replace(",", "")
tok = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:e-?\d+)?(?![\w])")
where = {}
for n, t in srcs:
    for i, ln in enumerate(norm(t).split(chr(10)), 1):
        for x in tok.findall(ln):
            where.setdefault(x.lstrip("+-"), (n, i))
union38 = set(tok.findall(norm(v38)))
c38, c39 = Counter(v38.split("\n")), Counter(new.split("\n"))
added = [l for l in (c39 - c38).elements() if l.strip()]
added_text = [l for l in added if not l.startswith("|")]
miss = {}; cnt = {}; D = 0; rows = []
for l in added_text:
    for t in tok.findall(norm(re.sub(r"`[^`\n]*`", " ", l))):
        if t in where:
            cnt[where[t][0]] = cnt.get(where[t][0], 0) + 1; rows.append((t, "%s line %d" % where[t])); continue
        if t in union38:
            cnt["v38 text"] = cnt.get("v38 text", 0) + 1; rows.append((t, "v38 text")); continue
        if re.fullmatch(r"7\.3[3-9](?:\.\d)?|\d{1,2}|2026|0\d|1\d\d?", t) or (float(t) < 100 and "." not in t):
            D += 1; continue
        miss.setdefault(t, []).append(l[:90])
out = ["# v39 numbers audit", "", "tip " + TIP, "", "added non-table lines: %d; sources scanned: %d files (+ register, + v38 text); tokens found in a source (first match, by file): %s; structural (integers below 100, years, ids): %d; unmatched: %d"
       % (len(added_text), len(srcs), ", ".join("%s %d" % kv for kv in sorted(cnt.items(), key=lambda kv: -kv[1])[:12]) + (" ..." if len(cnt) > 12 else ""), D, len(miss)), ""]
for t, ls in sorted(miss.items()):
    out.append("UNMATCHED %s | %s" % (t, ls[0]))
out += ["", "## number -> first source file and line (every distinct non-structural token of the added non-table lines)", ""]
seen = set()
for t, f in rows:
    if t not in seen:
        seen.add(t); out.append("%s | %s" % (t, f))
open(AUD, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print(out[4]); [print(x) for x in out[6:30] if x.startswith("UNMATCHED")]
sys.exit(1 if miss else 0)
