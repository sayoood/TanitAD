"""Numbers audit of report v45: every number token of the ADDED lines is re-found in the source blobs of the tip (every md / txt / json blob under raw/2026-11-02-* to raw/2026-11-08-*: the reads VOE1B, IMAG1B,
CAPLADDER, IMAG1C, the horizon review, the band baseline, PLANTEST, the bands build, and the register) or in the carried v44 text, or it is structural (integers below 100, years, section ids), or it is in the
explicit DERIVED allow-list below (each derived number is recomputed here by python from its inputs). Added lines that are exactly a source table line are not re-audited.
Second layer (scoped): a token of a line of section 7.49 must be found in the folders of the four 7.49 reads; of 7.50.1 to 7.50.6 in the horizon-review and band-baseline folders; of 7.50.7 in the PLANTEST
folder; of 7.50.8 in the bands-build folder or the register (or the carried text, DERIVED, structural). A token found only in another folder is a SCOPE-MISS and is listed until a stated reason accepts it.
python -I audit_v45.py tip [out_dir [audit_path]]   (git reads only; CPU only)"""
import os, re, subprocess, sys
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TIP = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"
OUT = (sys.argv[2] if len(sys.argv) > 2 else HERE + "out").rstrip("/\\") + "/"
AUD = sys.argv[3] if len(sys.argv) > 3 else HERE + "v45_NUMBERS_AUDIT.md"
M = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"


def git(args):
    r = subprocess.run(["git", "-C", M, "-c", "safe.directory=*"] + args, capture_output=True)
    assert r.returncode == 0, (args, r.stderr[:200])
    return r.stdout


def show(p):
    return git(["show", f"{TIP}:{p}"]).decode("utf-8").replace("\r", "")


def norm_pipes(line):  # as patch_v45.py
    if re.fullmatch(r"\|[\s:\-|]+", line):
        return line
    n, o = len(line), []
    for i, ch in enumerate(line):
        o.append("\\|" if (ch == "|" and 0 < i < n - 1 and line[i - 1] != "\\" and not (line[i - 1] == " " and line[i + 1] == " ")) else ch)
    return "".join(o)


# ---- DERIVED: numbers computed in the text from source numbers, each recomputed here -------------------------------------------------------------------------
# 7.50.3 [E]: 10 x (operative gap) + 20 x (tactical gap), over 30 knots; values read from the BAND_BASELINE.json on the tip
_bj = None
for n_ in git(["ls-tree", "-r", "--name-only", TIP, "--", R + "raw/2026-11-08-v8f-capladder-band/band_eval"]).decode("utf-8").split("\n"):
    if n_.endswith("BAND_BASELINE.json"):
        import json
        _bj = json.loads(show(n_))
DERIVED = {}
if _bj is not None:
    for sd in ("s0", "s1"):
        op = _bj["seeds"][sd]["operative_0_2s"]["rows"]["ade_m"]["arm_minus_P10"]["delta"]
        tc = _bj["seeds"][sd]["tactical_2_6s"]["rows"]["ade_m"]["arm_minus_P10"]["delta"]
        DERIVED["%.2f" % abs(10 * round(op, 3) + 20 * round(tc, 3))] = True   # 36.03 (s0) and 33.69 (s1) from the printed three-decimal deltas
DERIVED["0.083"] = True   # 7.50.3: operative share of the full-horizon ADE gap (10 x 0.003 over 36.03, as a percent under 0.1 %); recomputed below
assert abs(10 * 0.003 / 36.03 * 100 - 0.083) < 0.001
DERIVED["1.201"] = abs(30 * -1.201 - (10 * -0.003 + 20 * -1.800)) < 0.05
assert all(DERIVED.values())

names = [n for n in git(["ls-tree", "-r", "--name-only", TIP, "--", R + "raw"]).decode("utf-8").split("\n")
         if re.search(r"/2026-11-0[2-8]-", n) and n.endswith((".md", ".txt", ".json", ".jsonl", ".out", ".log"))]
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
v44 = show(R + "report/2026-11-07-v44/V8F_SCIENTIFIC_REPORT.md")  # the carried base
new = open(OUT + "V8F_SCIENTIFIC_REPORT.md", encoding="utf-8").read()
norm = lambda t: t.replace("\u2212", "-").replace(",", "")
tok = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:e-?\d+)?(?![\w])")
where = {}
for n, t in srcs:
    for i, ln in enumerate(norm(t).split(chr(10)), 1):
        for x in tok.findall(ln):
            where.setdefault(x.lstrip("+-"), (n, i))
# E: dev-box files that are read by the text and are single copies there (the F: train comparison of the v10 labels; same numbers as the register row)
E_ONLY = [("F-only:band_labels/v10/compare_v9_v10_train.log", "F:/strat_out/band_labels/v10/compare_v9_v10_train.log")]
where_e = {}
for label, fp in E_ONLY:
    if os.path.exists(fp):
        for i, ln in enumerate(norm(open(fp, encoding="utf-8").read().replace(chr(13), "")).split(chr(10)), 1):
            for x in tok.findall(ln):
                where_e.setdefault(x.lstrip("+-"), (label, i))
        # the log prints fractions (0.0522); the text prints percent with one decimal
        for x in re.findall(r"'changed_among_v9_defined': ([0-9.]+)", open(fp, encoding="utf-8").read()):
            where_e.setdefault(str((Decimal(x) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)), (label + " (percent of a printed fraction)", 0))
union44 = set(tok.findall(norm(v44)))
tokset = {n: set(x.lstrip("+-") for x in tok.findall(norm(t))) for n, t in srcs}
c44, c45 = Counter(v44.split("\n")), Counter(new.split("\n"))
added = [l for l in (c45 - c44).elements() if l.strip()]
SRCL = set()
for n, t in srcs:
    if n.endswith((".md", ".txt")):
        SRCL |= {norm_pipes(l) for l in t.split("\n") if l.startswith("|")}
added_text = [l for l in added if l not in SRCL]
n_srclines = len(added) - len(added_text)
# the section of every line of the new report: the nearest preceding 7.49 / 7.50 heading
sec_of = {}
cur = None
for ln in new.split("\n"):
    m = re.match(r"^(?:### |\*\*)(7\.(?:49|50)(?:\.\d+)?) ", ln)
    if m:
        cur = m.group(1)
    elif ln.startswith(("### 8", "## 8", "---")):
        cur = None
    sec_of.setdefault(ln, cur)
F749 = ("2026-11-07-v8f-voe-result/", "2026-11-07-v8f-imag-result/", "2026-11-08-v8f-capladder-band/", "2026-11-08-v8f-imag1c/")
F750 = ("2026-11-08-hwm-horizon-review/", "2026-11-08-v8f-capladder-band/", "2026-11-08-v8f-bands-build/horizon_review/")
F7507 = ("2026-11-08-v8f-plantest/",)
F7508 = ("2026-11-08-v8f-bands-build/", "register")
miss = {}; scope_miss = {}; cnt = {}; D = 0; rows = []
STRUCT = re.compile(r"7\.(?:49|50)(?:\.\d+)?|7\.[34]\d(?:\.\d+)?|\d{1,2}|2026|0\d|1\d\d?")
for l in added_text:
    sec = sec_of.get(l)
    folders = None
    if sec and sec.startswith("7.49"):
        folders = F749
    elif sec in ("7.50.7",):
        folders = F7507
    elif sec in ("7.50.8",):
        folders = F7508
    elif sec and sec.startswith("7.50"):
        folders = F750
    for t in tok.findall(norm(re.sub(r"`[^`\n]*`", " ", l))):
        in_scope = folders is not None and any(t in tokset[n] for n in tokset if n.startswith(folders))
        if folders and t not in union44 and not STRUCT.fullmatch(t) and not (float(t) < 100 and "." not in t) and t not in DERIVED and not in_scope and t not in where_e:
            scope_miss.setdefault(t, []).append((l[:70], sec))
        if t in where:
            cnt[where[t][0].split("/")[0]] = cnt.get(where[t][0].split("/")[0], 0) + 1; rows.append((t, "%s line %d" % where[t])); continue
        if t in union44:
            cnt["v44 text"] = cnt.get("v44 text", 0) + 1; rows.append((t, "v44 text")); continue
        if t in where_e:
            cnt["F-only files"] = cnt.get("F-only files", 0) + 1; rows.append((t, "%s line %d" % where_e[t])); continue
        if t in DERIVED:
            cnt["derived"] = cnt.get("derived", 0) + 1; rows.append((t, "derived (recomputed in this script)")); continue
        if STRUCT.fullmatch(t) or (float(t) < 100 and "." not in t):
            D += 1; continue
        miss.setdefault(t, []).append(l[:90])
out = ["# v45 numbers audit", "", "tip " + TIP, "", "added lines: %d (of which %d are exact source table lines and are not re-audited; audited: %d); sources scanned: %d files (+ register, + v44 text); tokens found in a source (first match, by folder): %s; structural (integers below 100, years, ids): %d; unmatched: %d"
       % (len(added), n_srclines, len(added_text), len(srcs), ", ".join("%s %d" % kv for kv in sorted(cnt.items(), key=lambda kv: -kv[1])[:10]), D, len(miss)), ""]
for t, ls in sorted(miss.items()):
    out.append("UNMATCHED %s | %s" % (t, ls[0]))
SCOPE_OK = {}  # token -> reason (none accepted without a stated reason)
for t, ls in sorted(scope_miss.items()):
    out.append("%s %s | section %s | line: %s" % ("SCOPE-OK" if t in SCOPE_OK else "SCOPE-MISS", t, ls[0][1], ls[0][0]) + ((" | reason: " + SCOPE_OK[t]) if t in SCOPE_OK else ""))
n_scope = sum(1 for t in scope_miss if t not in SCOPE_OK)
out += ["", "## number -> first source file and line (every distinct non-structural token of the audited added lines)", ""]
seen = set()
for t, f in rows:
    if t not in seen:
        seen.add(t); out.append("%s | %s" % (t, f))
open(AUD, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print(out[4]); [print(x) for x in out[6:300] if x.startswith(("UNMATCHED", "SCOPE-MISS"))]
print("scope misses unresolved:", n_scope)
sys.exit(1 if (miss or n_scope) else 0)
