"""Numbers audit of report v42: every number token of the ADDED lines is re-found in the source blobs of the tip (every md / txt / json blob under raw/2026-11-02-* to raw/2026-11-07-*: the T2 cards, reads,
amendments and results, the LoRA pod amendment and identity read, the theory notes and WSP reads, the earlier 7.30 to 7.41 reads they cite, the register) or in the carried v41 text, or it is structural (integers
below 100, years, ids), or it is in the explicit DERIVED allow-list below (each derived number is recomputed here by python from its inputs). The added lines are all lines that are not exactly a source table line
(the four directive tables of 7.42 are source lines, copied verbatim; the hand-written tables, the three added 8.4 rows and the appendix table ARE audited). Unmatched tokens stop the audit.
Also lists, per source file, how many were found where.
python -I audit_v42.py tip [out_dir [audit_path]]   (git reads only; CPU only)"""
import os, re, subprocess, sys
from collections import Counter
from statistics import NormalDist
from decimal import Decimal, ROUND_HALF_UP
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TIP = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"
OUT = (sys.argv[2] if len(sys.argv) > 2 else HERE + "out").rstrip("/\\") + "/"
AUD = sys.argv[3] if len(sys.argv) > 3 else HERE + "v42_NUMBERS_AUDIT.md"
M = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"


def git(args):
    r = subprocess.run(["git", "-C", M, "-c", "safe.directory=*"] + args, capture_output=True)
    assert r.returncode == 0, (args, r.stderr[:200])
    return r.stdout


def show(p):
    return git(["show", f"{TIP}:{p}"]).decode("utf-8").replace("\r", "")


def norm_pipes(line):  # as patch_v42.py
    if re.fullmatch(r"\|[\s:\-|]+", line):
        return line
    n, o = len(line), []
    for i, ch in enumerate(line):
        o.append("\\|" if (ch == "|" and 0 < i < n - 1 and line[i - 1] != "\\" and not (line[i - 1] == " " and line[i + 1] == " ")) else ch)
    return "".join(o)


# ---- the DERIVED allow-list: numbers that are computed in the text from source numbers, each recomputed here ---------------------------------------------------
_N = NormalDist()
_beta = lambda auc: _N.cdf((2 ** 0.5) * _N.inv_cdf(auc) - _N.inv_cdf(0.95))
_rh = lambda x: str((Decimal(x) / Decimal("0.001")).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
DERIVED = {
    "0.194": "%.3f" % (1.000 - 0.806) == "0.194",          # 7.44.5: base gain 1.000 minus total gain 0.806 (v0, s0)
    "0.634": "%.3f" % (1.899 - 1.265) == "0.634",          # 1.899 minus 1.265 (a0, s0)
    "0.237": "%.3f" % (1.000 - 0.763) == "0.237",          # s1
    "0.787": "%.3f" % (1.899 - 1.112) == "0.787",          # s1
    "0.30": "%.2f" % _beta(0.785) == "0.30",               # binormal illustration, AUC 0.785, alpha 5 %
    "0.16": "%.2f" % _beta(0.680) == "0.16",               # AUC 0.680
    "3145728": 24 * 2 * 2 * 1024 * 32 == 3145728,          # 24 blocks x 2 matrices x 2 factors of 1,024 x 32
    "21": _rh("0.0211") == "21", "23": _rh("0.0225") == "23",  # 7.43.4 "21 to 23 times": 2.11e-2 and 2.25e-2 over 1e-3
}
assert all(DERIVED.values()), [k for k, v in DERIVED.items() if not v]

names = [n for n in git(["ls-tree", "-r", "--name-only", TIP, "--", R + "raw"]).decode("utf-8").split("\n")
         if re.search(r"/2026-11-0[2-7]-", n) and n.endswith((".md", ".txt", ".json"))]
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
v41 = show(R + "report/2026-11-06-v41/V8F_SCIENTIFIC_REPORT.md")  # the carried base (v41)
new = open(OUT + "V8F_SCIENTIFIC_REPORT.md", encoding="utf-8").read()
norm = lambda t: t.replace("\u2212", "-").replace(",", "")
tok = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:e-?\d+)?(?![\w])")
where = {}
SRCL = set()
for n, t in srcs:
    for i, ln in enumerate(norm(t).split(chr(10)), 1):
        for x in tok.findall(ln):
            where.setdefault(x.lstrip("+-"), (n, i))
    if n.endswith((".md", ".txt")):
        SRCL |= {norm_pipes(l) for l in t.split("\n") if l.startswith("|")}
# E: only files (cited in 7.43 and 7.44 as "not yet landed"): a token that no tip file holds may be found here; reported as such, never silently
EDIR = "E:/Projects/TanitAD-artifacts/dagland/motion_c/"
E_ONLY = ["lora_pod/HANDOFF_LORAPOD2.md", "lora_pod/HANDOFF_LORAPOD3.md", "lora_pod/AMENDMENT_V8F_LORA_4_LOG.md", "theory/wsp_edit/HANDOFF_WSPEDIT.md", "theory/wsp_read/P1P_RESULT.md", "theory/wsp_read/HANDOFF_P1P.md", "t2v8f/HANDOFF_Q.md"]
where_e = {}
for fn in E_ONLY:
    if os.path.exists(EDIR + fn):
        for i, ln in enumerate(norm(open(EDIR + fn, encoding="utf-8").read().replace(chr(13), "")).split(chr(10)), 1):
            for x in tok.findall(ln):
                where_e.setdefault(x.lstrip("+-"), ("E-only:" + fn, i))
if os.path.exists(EDIR + "theory/wsp_read/P1P_RESULT.md"):
    assert "73640c67" in open(EDIR + "theory/wsp_read/P1P_RESULT.md", encoding="utf-8").read(), "the SPEC_P1P sha256 prefix named in 7.44.2"
if os.path.exists(EDIR + "t2v8f/HANDOFF_Q.md"):
    assert "7fd65ffd" in open(EDIR + "t2v8f/HANDOFF_Q.md", encoding="utf-8").read(), "the fourth pin prefix named in 7.42.6"
union41 = set(tok.findall(norm(v41)))
c41, c42 = Counter(v41.split("\n")), Counter(new.split("\n"))
added = [l for l in (c42 - c41).elements() if l.strip()]
added_text = [l for l in added if l not in SRCL]
n_srclines = len(added) - len(added_text)
miss = {}; cnt = {}; D = 0; rows = []
for l in added_text:
    for t in tok.findall(norm(re.sub(r"`[^`\n]*`", " ", l))):
        if t in where:
            cnt[where[t][0]] = cnt.get(where[t][0], 0) + 1; rows.append((t, "%s line %d" % where[t])); continue
        if t in union41:
            cnt["v41 text"] = cnt.get("v41 text", 0) + 1; rows.append((t, "v41 text")); continue
        if t in where_e:
            cnt["E-only files"] = cnt.get("E-only files", 0) + 1; rows.append((t, "%s line %d" % where_e[t])); continue
        if t in DERIVED:
            cnt["derived"] = cnt.get("derived", 0) + 1; rows.append((t, "derived (recomputed in this script)")); continue
        if re.fullmatch(r"7\.3[3-9](?:\.\d)?|7\.4[0-4](?:\.\d)?|\d{1,2}|2026|0\d|1\d\d?", t) or (float(t) < 100 and "." not in t):
            D += 1; continue
        miss.setdefault(t, []).append(l[:90])
out = ["# v42 numbers audit", "", "tip " + TIP, "", "added lines: %d (of which %d are exact source table lines and are not re-audited; audited: %d); sources scanned: %d files (+ register, + v41 text); tokens found in a source (first match, by file): %s; structural (integers below 100, years, ids): %d; unmatched: %d"
       % (len(added), n_srclines, len(added_text), len(srcs), ", ".join("%s %d" % kv for kv in sorted(cnt.items(), key=lambda kv: -kv[1])[:12]) + (" ..." if len(cnt) > 12 else ""), D, len(miss)), ""]
for t, ls in sorted(miss.items()):
    out.append("UNMATCHED %s | %s" % (t, ls[0]))
out += ["", "## number -> first source file and line (every distinct non-structural token of the audited added lines)", ""]
seen = set()
for t, f in rows:
    if t not in seen:
        seen.add(t); out.append("%s | %s" % (t, f))
open(AUD, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print(out[4]); [print(x) for x in out[6:60] if x.startswith("UNMATCHED")]
sys.exit(1 if miss else 0)
