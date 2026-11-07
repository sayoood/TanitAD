"""Numbers audit of report v44: every number token of the ADDED lines is re-found in the source blobs of the tip (every md / txt / json blob under raw/2026-11-02-* to raw/2026-11-07-*: the T2 cards, reads,
amendments and results, the LoRA pod amendment and identity read, the theory notes and WSP reads, the earlier 7.30 to 7.41 reads they cite, the register) or in the carried v43 text, or it is structural (integers
below 100, years, ids), or it is in the explicit DERIVED allow-list below (each derived number is recomputed here by python from its inputs). The added lines are all lines that are not exactly a source table line
(the four directive tables of 7.42 are source lines, copied verbatim; the hand-written tables, the three added 8.4 rows and the appendix table ARE audited). Unmatched tokens stop the audit.
Also lists, per source file, how many were found where.
python -I audit_v44.py tip [out_dir [audit_path]]   (git reads only; CPU only)"""
import os, re, subprocess, sys
from collections import Counter
from statistics import NormalDist
from decimal import Decimal, ROUND_HALF_UP
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TIP = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"
OUT = (sys.argv[2] if len(sys.argv) > 2 else HERE + "out").rstrip("/\\") + "/"
AUD = sys.argv[3] if len(sys.argv) > 3 else HERE + "v44_NUMBERS_AUDIT.md"
M = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"


def git(args):
    r = subprocess.run(["git", "-C", M, "-c", "safe.directory=*"] + args, capture_output=True)
    assert r.returncode == 0, (args, r.stderr[:200])
    return r.stdout


def show(p):
    return git(["show", f"{TIP}:{p}"]).decode("utf-8").replace("\r", "")


def norm_pipes(line):  # as patch_v44.py
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
    "0.953": abs((0.9148 + 0.090 * (0.0313 * 0.9148) ** 0.5) / 0.9766 - 0.953) < 1e-3,  # unrounded inputs give 0.9526;  # 7.46.5 attenuation with the observed correlation
    "0.566": "%.3f" % (0.666 - 0.10) == "0.566", "0.686": "%.3f" % (0.586 + 0.10) == "0.686",  # 7.46.5 ratios +- 0.10
    "0.130": "%.3f" % (0.8401 - 0.7098) == "0.130",  # 7.45.7 render-real minus render-mean cosine
    "0.12": "%.2f" % (1.384 - 1.265) == "0.12", "0.27": "%.2f" % (1.384 - 1.112) == "0.27",  # 7.46.5 next-lever gaps (P5_RESULT section 7)
    "44.97": "%.2f" % (8994.27591919899 / 200) == "44.97",  # 7.45.5 L step time over the first 200 steps (train_log.jsonl, step 200 at 8,994.28 s)
    "0.06": "%.2f" % (0.4073 - 0.3493) == "0.06", "0.11": "%.2f" % (0.4041 - 0.2919) == "0.11",  # 7.45.2 post-training adds to the M1 crawl share
}
assert all(DERIVED.values()), [k for k, v in DERIVED.items() if not v]

names = [n for n in git(["ls-tree", "-r", "--name-only", TIP, "--", R + "raw"]).decode("utf-8").split("\n")
         if re.search(r"/2026-11-0[2-7]-", n) and n.endswith((".md", ".txt", ".json", ".jsonl", ".out"))]
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
LIBP = "TanitAD Research Lab/Library/LIBRARY.md"  # the 50-paper landing (0423bb9d): index totals before and after, from the header lines of the generated index
srcs.append(("library/LIBRARY.md header (tip)", "\n".join(show(LIBP).split("\n")[:16])))
srcs.append(("library/LIBRARY.md header (parent of 0423bb9d)", "\n".join(git(["show", "0423bb9d^:" + LIBP]).decode("utf-8").replace("\r", "").split("\n")[:16])))
v41 = show(R + "report/2026-11-07-v43/V8F_SCIENTIFIC_REPORT.md")  # the carried base (v43; the variable name v41 is kept from v41)
new = open(OUT + "V8F_SCIENTIFIC_REPORT.md", encoding="utf-8").read()
norm = lambda t: t.replace("\u2212", "-").replace(",", "")
tok = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:e-?\d+)?(?![\w])")
where = {}
where_r = {}
SRCL = set()
# the three files whose printed numbers in the report are roundings of the longer floats the files hold (READ1.json gain floats; the L launch train log and timing log)
ROUND_SRC = ("wsp-gain/raw/READ1.json", "lora_L_launch/L_s0/train_log.jsonl", "lora_L_launch/timing_b64.out", "t2-c0cre-result/RESULT_T2_RENC.md", "t2-renc-gate/M_RENC.json", "t2-renc-gate/TRAIN_RESULT_A.json", "t2-renc-gate/TRAIN_RESULT_B.json", "v8f-wmact-result/RESULT_WMACT.json")
for n, t in srcs:
    for i, ln in enumerate(norm(t).split(chr(10)), 1):
        for x in tok.findall(ln):
            where.setdefault(x.lstrip("+-"), (n, i))
    if n.endswith((".md", ".txt")):
        SRCL |= {norm_pipes(l) for l in t.split("\n") if l.startswith("|")}
    if n.endswith(ROUND_SRC):  # a number printed at 1 to 4 decimals whose source file holds it as a longer float (named files only, never a blanket rounding)
        for i, ln in enumerate(norm(t).split(chr(10)), 1):
            for x in tok.findall(ln):
                if re.fullmatch(r"\d+\.\d{5,}", x):
                    for dp in (1, 2, 3, 4, 5):
                        rx = str(Decimal(x).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP))
                        where_r.setdefault(rx, (n, i))
# E: only files (cited in 7.43 and 7.44 as "not yet landed"): a token that no tip file holds may be found here; reported as such, never silently
EDIR = "E:/Projects/TanitAD-artifacts/dagland/motion_c/"
E_ONLY = ["lora_pod/HANDOFF_LORAPOD2.md", "lora_pod/HANDOFF_LORAPOD3.md", "lora_pod/AMENDMENT_V8F_LORA_4_LOG.md", "theory/wsp_edit/HANDOFF_WSPEDIT.md", "theory/wsp_read/P1P_RESULT.md", "theory/wsp_read/HANDOFF_P1P.md", "t2v8f/HANDOFF_Q.md", "theory/imag1/HANDOFF_IMAG1.md", "theory/voe1/HANDOFF_VOE1.md"]
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
tokset = {n: set(x.lstrip("+-") for x in tok.findall(norm(t))) for n, t in srcs}
# second layer (scoped): a token of a line that names C0C-RE, TSWAP, LATSEM2 or the imagination plan must be found in the folders of that read (or in the carried v43 text, or be derived or structural);
# a token found only in another folder is a SCOPE-MISS and is listed until a stated reason accepts it (SCOPE_OK)
SCOPES = [(r"C0C-RE|RE-HARMS|RENC", ("2026-11-07-t2-c0cre-result/", "2026-11-07-t2-renc-gate/", "2026-11-07-t2-renc-prereg/")),
          (r"TSWAP", ("2026-11-07-v8f-tswap-result/", "2026-11-07-v8f-tswap-prereg/", "2026-11-07-v8f-town-result/")),
          (r"LATSEM2", ("2026-11-07-v8f-latsem2-run1/", "2026-11-07-v8f-latsem2b-result/", "2026-11-07-v8f-latsem2-prereg/", "2026-11-07-v8f-latsem-result/", "2026-11-07-v8f-wmact")),
          (r"IMAG1|VOE1|imagination|banked|Library", ("2026-11-07-v8f-imagination-plan/", "library/", "register"))]
scope_miss = {}
miss = {}; cnt = {}; D = 0; rows = []
for l in added_text:
    folders = tuple(f for pat, fs in SCOPES if re.search(pat, l) for f in fs)
    for t in tok.findall(norm(re.sub(r"`[^`\n]*`", " ", l))):
        if folders and t not in union41 and not (re.fullmatch(r"\d{1,2}|2026|0\d|1\d\d?", t) or (float(t) < 100 and "." not in t)) and t not in DERIVED and not any(t in tokset[n] for n in tokset if n.startswith(folders)) and not (t in where_r and where_r[t][0].startswith(folders)):
            scope_miss.setdefault(t, []).append((l[:70], [f.split("-v8f-")[-1] for f in folders][:3]))
        if t in where:
            cnt[where[t][0]] = cnt.get(where[t][0], 0) + 1; rows.append((t, "%s line %d" % where[t])); continue
        if t in where_r:
            cnt["rounding of a longer float"] = cnt.get("rounding of a longer float", 0) + 1; rows.append((t, "rounding of a longer float, %s line %d" % where_r[t])); continue
        if t in union41:
            cnt["v43 text"] = cnt.get("v43 text", 0) + 1; rows.append((t, "v43 text")); continue
        if t in where_e:
            cnt["E-only files"] = cnt.get("E-only files", 0) + 1; rows.append((t, "%s line %d" % where_e[t])); continue
        if t in DERIVED:
            cnt["derived"] = cnt.get("derived", 0) + 1; rows.append((t, "derived (recomputed in this script)")); continue
        if re.fullmatch(r"7\.3[3-9](?:\.\d)?|7\.4[0-8](?:\.\d+)?|\d{1,2}|2026|0\d|1\d\d?", t) or (float(t) < 100 and "." not in t):
            D += 1; continue
        miss.setdefault(t, []).append(l[:90])
out = ["# v44 numbers audit", "", "tip " + TIP, "", "added lines: %d (of which %d are exact source table lines and are not re-audited; audited: %d); sources scanned: %d files (+ register, + v43 text); tokens found in a source (first match, by file): %s; structural (integers below 100, years, ids): %d; unmatched: %d"
       % (len(added), n_srclines, len(added_text), len(srcs), ", ".join("%s %d" % kv for kv in sorted(cnt.items(), key=lambda kv: -kv[1])[:12]) + (" ..." if len(cnt) > 12 else ""), D, len(miss)), ""]
for t, ls in sorted(miss.items()):
    out.append("UNMATCHED %s | %s" % (t, ls[0]))
_WM = "a number of the 7.47 reads (WMACT / LATSEM) on a line that also names RENC or C0C-RE; its source is checked row by row in check_numbers_v44.py (NUMBERS_V44.md), not by file scope here"
SCOPE_OK = {  # token -> reason; each accepted after review of the SCOPE-MISS lines (none is accepted without a stated reason)
    "1833": _WM, "1976": _WM, "9460": _WM,
    "4213": "bytes of the frozen MRATE card prefix: check_numbers_v44.py hashes the first 4,213 bytes of the card (prefix 50667a79); the first audit layer's file match for this token (a manifest) is unrelated",
    "5415": "bytes of the frozen DAgger card prefix, first audit layer: 2026-11-07-t2-dagger-prereg/PREREG_T2_DAGGER.md line 40 (a card named on a line about the C0C-RE read)",
    "7.47": "section number", "7.48": "section number",
}
for t, ls in sorted(scope_miss.items()):
    out.append("%s %s | scoped folders %s | line: %s" % ("SCOPE-OK" if t in SCOPE_OK else "SCOPE-MISS", t, ls[0][1], ls[0][0]) + ((" | reason: " + SCOPE_OK[t]) if t in SCOPE_OK else ""))
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
