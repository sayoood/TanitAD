"""Numbers audit of report v37: every number token of the ADDED non-table lines is re-found in the source blobs of the tip (the RESULTS, cards, freeze records and raw JSONs of 7.35.4 and 7.36, the register) or in the carried v36 text,
or it is structural (integers below 100, years, ids). Unmatched tokens stop the audit. Also lists, per token class, how many were found where.
python -I audit_v37.py tip [out_dir [audit_path]]   (git show reads only; CPU only)"""
import difflib, os, re, subprocess, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TIP = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"
OUT = (sys.argv[2] if len(sys.argv) > 2 else HERE + "out").rstrip("/\\") + "/"
AUD = sys.argv[3] if len(sys.argv) > 3 else HERE + "v37_NUMBERS_AUDIT.md"
M = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
def show(p):
    r = subprocess.run(["git", "-C", M, "-c", "safe.directory=*", "show", f"{TIP}:{p}"], capture_output=True); assert r.returncode == 0, p
    return r.stdout.decode("utf-8").replace("\r", "")
FILES = [("RF RESULTS", "raw/2026-11-03-recover-read-result/RESULTS.md"), ("RT RESULTS", "raw/2026-11-04-recover-next-rtl/RESULTS.md"), ("RP RESULTS", "raw/2026-11-04-recover-next-rp01/RESULTS.md"),
         ("RA RESULTS", "raw/2026-11-04-recover-next-rapr/RESULTS.md"), ("LP RESULTS", "raw/2026-11-04-levelprobe/RESULTS.md"), ("LP2 RESULTS", "raw/2026-11-04-levelprobe/RESULTS2.md"),
         ("LP amend", "raw/2026-11-04-levelprobe/AMEND_A1_PROPOSED.md"), ("LP freeze", "raw/2026-11-04-levelprobe/FROZEN_SHA.txt"), ("LP2 freeze", "raw/2026-11-04-levelprobe/FROZEN_SHA2.txt"),
         ("FT card", "raw/2026-11-02-recover-ft/PREREG_RECOVER_FT_FROZEN.md"), ("CO card", "raw/2026-11-02-recover-read/PREREG_COMPOSITE_FROZEN.md"),
         ("NX card", "raw/2026-11-03-recover-next/PREREG_RECOVER_NEXT_FROZEN.md"), ("A1 card", "raw/2026-11-04-recover-next-rtl/PREREG_RECOVER_NEXT_ADDENDUM_RTLRP01.md"),
         ("A2 card", "raw/2026-11-04-recover-next-rapr/PREREG_RECOVER_NEXT_ADDENDUM2_RAPR.md"), ("A2 freeze", "raw/2026-11-04-recover-next-rapr/FROZEN_SHA_ADDENDUM2.txt"),
         ("RA raw json", "raw/2026-11-04-recover-next-rapr/RECOVER_READ_RAPR.json"), ("NX raw json", "raw/2026-11-04-recover-next-rapr/RECOVER_NEXT_after_RAPR.json")]
srcs = [(n, show(R + p)) for n, p in FILES] + [("register", show("Project Steering/GOALS_AND_CLAIMS.md"))]
v35 = show(R + "report/2026-11-04-v36/V8F_SCIENTIFIC_REPORT.md")  # the carried base (v36); the name is kept from audit_v36.py
new = open(OUT + "V8F_SCIENTIFIC_REPORT.md", encoding="utf-8").read()
norm = lambda t: t.replace("\u2212", "-").replace(",", "")
tok = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:e-?\d+)?(?![\w])")
where = {}
for n, t in srcs:
    for i, ln in enumerate(norm(t).split(chr(10)), 1):
        for x in tok.findall(ln): where.setdefault(x.lstrip("+-"), (n, i))
union35 = set(tok.findall(norm(v35)))
added = [l[2:] for l in difflib.ndiff(v35.split("\n"), new.split("\n")) if l.startswith("+ ")]
added_text = [l for l in added if not l.startswith("|")]
miss = {}; cnt = {}; D = 0; rows = []
for l in added_text:
    for t in tok.findall(norm(re.sub(r"`[^`\n]*`", " ", l))):
        if t in where: cnt[where[t][0]] = cnt.get(where[t][0], 0) + 1; rows.append((t, "%s line %d" % where[t])); continue
        if t in union35: cnt["v36 text"] = cnt.get("v36 text", 0) + 1; rows.append((t, "v36 text")); continue
        if re.fullmatch(r"7\.3[3456](?:\.\d)?|\d{1,2}|2026|0\d|1\d\d?", t) or (float(t) < 100 and "." not in t): D += 1; continue
        miss.setdefault(t, []).append(l[:90])
out = ["# v37 numbers audit", "", "tip " + TIP, "", "added non-table lines: %d; tokens found in a source (first match) %s; structural (integers below 100, years, ids): %d; unmatched: %d" % (len(added_text), ", ".join("%s %d" % kv for kv in sorted(cnt.items())), D, len(miss)), ""]
for t, ls in sorted(miss.items()): out.append("UNMATCHED %s | %s" % (t, ls[0]))
out += ["", "## number -> first source file and line (every distinct non-structural token of the added non-table lines)", ""]
seen = set()
for t, f in rows:
    if t not in seen: seen.add(t); out.append("%s | %s" % (t, f))
open(AUD, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print(out[4]); [print(x) for x in out[6:30] if x.startswith("UNMATCHED")]
sys.exit(1 if miss else 0)
