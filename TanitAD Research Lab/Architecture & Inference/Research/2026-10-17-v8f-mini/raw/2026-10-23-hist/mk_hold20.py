"""Build qland/hold_v8fhist_prereg: PREREG_V8F_HIST (before any instrument or data) + its register row after H-V8F-GRID-1.
   python mk_hold20.py <remote tip sha>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/grid_a2/"
H = "C:/Users/Admin/qland/hold_v8fhist_prereg"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


for n in ("hold_v8fhist_prereg", "landed_hold_v8fhist_prereg"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
for rel, src in (("Project Steering/PREREG_V8F_HIST.md", S + "PREREG_V8F_HIST.md"), (f"{PK}/raw/2026-10-23-hist/mk_hold20.py", S + "mk_hold20.py")):
    assert show(TIP, rel) is None, rel
    put("stage", rel, open(src, "rb").read().replace(b"\r\n", b"\n"))

reg = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8")
lines = reg.split("\n")
assert not any(x.startswith("| **H-V8F-HIST-1**") for x in lines)
ig = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-GRID-1** (2026-10-22) |")]
assert len(ig) == 1
row = ("| **H-V8F-HIST-1** (2026-10-23) | The stack's scene input over 10 history frames instead of 4 (1.8 s of video instead of 0.6 s) makes v8F-mini drive better "
       "and T's lateral constraints more accurate: arm H = M2b + `--hist 10`, ONE lever (hmax unchanged at 16, so the batch, the windows and the cache are B's), "
       "seeds 0 then 1 | **PRE-REGISTERED, NOT RUN (box 2, ~4.5–5 h per seed ESTIMATED; awaits the PI's go).** `Project Steering/PREREG_V8F_HIST.md`, the "
       "committed FAIL-NO-MECHANISM branch of `PREREG_V8F_GRID`. PRIMARY: ADE6 H − M2b < 0, separated, more than 2F (0.2415) at both seeds. MECHANISM: T's "
       "latcons RMSE falls ≥ 12.5 %, separated. Declared with the lever: +1,536 parameters in `adapter`, the init RNG stream, the held-age clamp near clip "
       "starts (training only), padded history at the 282 eval windows with t < 11 (the PRIMARY is also reported on the 2,117 others). Instruments being built. |")
lines.insert(ig[0] + 1, row)
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
open(os.path.join(H, "msg.txt"), "wb").write(open(S + "msg20.txt", "rb").read().replace(b"\r\n", b"\n"))
print("hold ok")
