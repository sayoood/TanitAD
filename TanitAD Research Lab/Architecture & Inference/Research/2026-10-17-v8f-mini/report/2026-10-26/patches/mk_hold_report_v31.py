"""Hold: report v31 (7.22 FLAT-DAG NOT ESTABLISHED, 7.23 DAG-OACC LONGITUDINAL FIXED, the abstract, 8.1-8.4) + its patches and numbers audit.
python mk_hold_report_v31.py <live tip>"""
import difflib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
HERE = "E:/Projects/TanitAD-artifacts/dagland/report_v31/"
D = HERE + "report/2026-10-26/"
H = "C:/Users/Admin/qland/hold_report_v31/"
RP = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/report/2026-10-26/"
assert not os.path.exists(H), H


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


rd = lambda p: open(p, "rb").read()
md, html = rd(D + "V8F_SCIENTIFIC_REPORT.md"), rd(D + "v8f_report.html")
tpl = blob(RP + "page_template_v30.html")
assert html == tpl.replace(b"%%MD%%", md)
old = blob(RP + "V8F_SCIENTIFIC_REPORT.md").decode("utf-8")
added = [l for l in difflib.ndiff(old.splitlines(), md.decode("utf-8").splitlines()) if l.startswith("+ ")]
for w in ("void", "superseded", "retracted", "formerly", "no longer", "correction", "withdrawn"):
    hits = [l for l in added if re.search(r"\b" + w + r"\b", l.lower())]
    assert not hits, (w, hits[:2])
assert b"\r" not in md and b"\r" not in html and b"results through repository tip `569a9d9b`" in md
put("mods-rw/" + RP + "V8F_SCIENTIFIC_REPORT.md", md)
put("mods-rw/" + RP + "v8f_report.html", html)
for rel, src in (("patches/patch_v31.py", D + "patches/patch_v31.py"), ("patches/patch_v31_mm.py", D + "patches/patch_v31_mm.py"),
                 ("patches/v31_NUMBERS_AUDIT.md", HERE + "NUMBERS_AUDIT.md"), ("patches/v31_checks.txt", HERE + "checks.txt"),
                 ("patches/mk_hold_report_v31.py", HERE + "mk_hold_report_v31.py")):
    b = rd(src)
    assert blob(RP + rel) is None and b"\r" not in b, rel
    put("stage/" + RP + rel, b)
put("msg.txt", ("Report v31: 7.22 FLAT-DAG (NOT ESTABLISHED: the DAgger recipe helps flat too; the hierarchy keeps its ADE6 margin at s0, not at"
                " s1), 7.23 DAG-OACC (LONGITUDINAL FIXED: accel MAE -56 %% / -58 %%, every longitudinal row clean against P10; the crawl rows"
                " stay open, so not DOMINANT), the abstract and 8.1-8.4 (rows 1u-1x). DAG-ORC re-based and SPEEDREAD are stated as"
                " pre-registered, with no numbers. Tables copied verbatim from the RESULT files; numbers audit and the MM review pass are in"
                " patches/.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n").encode())
print("hold written", H, len(added), "added lines")
