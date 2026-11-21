"""Build qland/hold_v8f_report_v20: the post-doc report's v20 source (the T1-kin result, section 7.15, rows 1m / 1n / 6, the T1-kin wording) beside
the banked 2026-10-25 edition; the figures and videos it references are the 2026-10-25 edition's.  python mk_hold88.py <tip>"""
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
R = SC + "report/"
H = "C:/Users/Admin/qland/hold_v8f_report_v20"
MIR = "C:/Users/Admin/tanitad-push"
D = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/report/2026-10-26"
TOKEN = re.compile(rb"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")


def git(*a):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


for n in ("hold_v8f_report_v20", "landed_hold_v8f_report_v20"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
for src, name in ((R + "V8F_SCIENTIFIC_REPORT.md", "V8F_SCIENTIFIC_REPORT.md"), (R + "v8f_report.html", "v8f_report.html"),
                  (R + "patch_v20.py", "patch_v20.py"), (SC + "grid_a2/mk_hold88.py", "mk_hold88.py")):
    data = open(src, "rb").read()
    assert not TOKEN.search(data), name
    rel = f"{D}/{name}"
    assert git("rev-parse", f"{TIP}:{rel}") is None, rel
    p = os.path.join(H, "stage", rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data.replace(b"\r\n", b"\n"))
open(os.path.join(H, "msg.txt"), "wb").write(
    b"v8F-mini report v20 banked: the T1-kin result (T1 BEATS THE FLOOR, conditional on the speed bucket), section 7.15, HACC-LAT row\n\n"
    b"Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
print("hold ok")
