"""Build qland/hold_v8fd1_prereg: PREREG_V8F_HIST (before any instrument or data) + its register row after H-V8F-GRID-1.
   python mk_hold23.py <remote tip sha>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/grid_a2/"
H = "C:/Users/Admin/qland/hold_v8fd1_prereg"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


for n in ("hold_v8fd1_prereg", "landed_hold_v8fd1_prereg"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
for rel, src in (("Project Steering/PREREG_V8F_D1.md", S + "PREREG_V8F_D1.md"), (f"{PK}/raw/2026-10-23-d1/mk_hold23.py", S + "mk_hold23.py")):
    assert show(TIP, rel) is None, rel
    put("stage", rel, open(src, "rb").read().replace(b"\r\n", b"\n"))

reg = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8")
lines = reg.split("\n")
assert not any(x.startswith("| **H-V8F-D1-1**") for x in lines)
ig = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-HIST-1** (2026-10-23) |")]
assert len(ig) == 1
row = ("| **H-V8F-D1-1** (2026-10-23) | Lateral-event-stratified training batches (turn / lane-change windows at 20 % of every batch, the same clips, windows, steps "
       "and eval) make v8F-mini drive better and T's lateral constraints more accurate: arm E = B + `--event-share 0.2`, ONE lever, B = M2b-h10 iff "
       "PREREG_V8F_HIST's final read is SUCCESS else M2b, seeds 0 then 1 | **PRE-REGISTERED, NOT RUN (box 2 after HIST's chain; awaits the PI's go).** "
       "`Project Steering/PREREG_V8F_D1.md` (plan section 13.6, the lever after more history frames). PRIMARY: ADE6 E − B < 0, separated, more than 2F at both "
       "seeds. MECHANISM: T's latcons RMSE falls ≥ 12.5 %, separated. Beside: turn vs lane-keep windows separately (the trade-off). Instruments being built. |")
lines.insert(ig[0] + 1, row)
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
open(os.path.join(H, "msg.txt"), "wb").write(open(S + "msg23.txt", "rb").read().replace(b"\r\n", b"\n"))
print("hold ok")
