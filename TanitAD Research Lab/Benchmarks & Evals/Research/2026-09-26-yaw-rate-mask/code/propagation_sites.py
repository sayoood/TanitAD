"""List every landed .md line that repeats one of the numbers D-YAWMASK-1 moves, so the owner can
annotate them (this package edits none of them). Reads the branch tip through git plumbing on the
local object store (never the G: mount); a hit needs the NUMBER and a CONTEXT word on the same line.

usage: python propagation_sites.py <out.txt>
"""
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
GD, BR = "C:/Users/Admin/tanitad-push/.git", "agent/arch-inf-20260803"
GROUPS = {
    "M26 / D-FEASDEC-T1-1 yaw headline": (["0.2176", "80.4 %", "80.3 %", "−80.4", "-80.4"],
                                          r"(?i)yaw|rad/s|LAT_"),
    "refcv3 40284 yaw paired cell (+0.1700 / +0.1849)": (["+0.1700", "0.1700 rad", "+0.1849"],
                                                        r"(?i)yaw|LAT_"),
    "replicate false-positive rate (6/42, 14.3 %, 3 of 14)": (
        ["6 of 42", "6/42", "6 / 42", "3 of 14", "3/14", "14.3 %", "14.3%"],
        r"(?i)replicate|false.positive|H-ESTIM-SEED|FPR|zero levers|A0b"),
}


def grep(pat):
    r = subprocess.run(["git", f"--git-dir={GD}", "grep", "-n", "-I", "-F", "-e", pat, BR, "--",
                        "*.md"], capture_output=True)
    if r.returncode not in (0, 1):
        raise SystemExit(f"git grep failed rc={r.returncode}")
    return r.stdout.decode("utf-8", "replace").splitlines()


out = []
for g, (pats, ctx) in GROUPS.items():
    hits = {}
    for p in pats:
        for ln in grep(p):
            _, rest = ln.split(":", 1)
            path, lno, text = rest.split(":", 2)
            if re.search(ctx, text):
                hits[(path, int(lno))] = text.strip()[:160]
    out.append(f"## {g}: {len(hits)} lines")
    for (path, lno), t in sorted(hits.items()):
        out.append(f"{path}:{lno}  |  {t}")
    out.append("")
open(sys.argv[1], "w", encoding="utf-8").write("\n".join(out))
print("\n".join(x for x in out if x.startswith("##")))
