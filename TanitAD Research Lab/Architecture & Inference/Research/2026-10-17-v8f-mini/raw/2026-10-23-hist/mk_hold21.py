"""Build qland/hold_v8fhist_a1: PREREG_V8F_HIST Amendment 1 (before any HIST data; the instruments' F-1 finding and the --window-start fix).
   python mk_hold21.py <remote tip sha>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/grid_a2/"
H = "C:/Users/Admin/qland/hold_v8fhist_a1"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


for n in ("hold_v8fhist_a1", "landed_hold_v8fhist_a1"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
pz = "Project Steering/PREREG_V8F_HIST.md"
cur = show(TIP, pz)
assert cur is not None and b"## Amendment 1" not in cur
put("mods", pz, cur.rstrip(b"\n") + b"\n" + open(S + "hist_amend1.md", "rb").read().replace(b"\r\n", b"\n"))
rel = f"{PK}/raw/2026-10-23-hist/mk_hold21.py"
assert show(TIP, rel) is None
put("stage", rel, open(S + "mk_hold21.py", "rb").read().replace(b"\r\n", b"\n"))
open(os.path.join(H, "msg.txt"), "wb").write(open(S + "msg21.txt", "rb").read().replace(b"\r\n", b"\n"))
print("hold ok")
