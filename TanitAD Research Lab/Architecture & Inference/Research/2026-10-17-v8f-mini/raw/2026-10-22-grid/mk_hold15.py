"""Build qland/hold_v8fgrid_prereg: PREREG_V8F_GRID (before its data) + the register: H-V8F-GRID-1 PRE-REGISTERED, H-V8F-PGT-1 RUNNING.
   python mk_hold15.py <remote tip sha> "<PGT running note>" """
import os
import subprocess
import sys

TIP, NOTE = sys.argv[1], sys.argv[2]
DL = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/dkland/"
H = "C:/Users/Admin/qland/hold_v8fgrid_prereg"
MIR = "C:/Users/Admin/tanitad-push"


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


for n in ("hold_v8fgrid_prereg", "landed_hold_v8fgrid_prereg"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
rel = "Project Steering/PREREG_V8F_GRID.md"
assert show(TIP, rel) is None
put("stage", rel, open(DL + "PREREG_V8F_GRID.md", "rb").read().replace(b"\r\n", b"\n"))
MK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-22-grid/mk_hold15.py"
assert show(TIP, MK) is None
put("stage", MK, open(DL + "mk_hold15.py", "rb").read().replace(b"\r\n", b"\n"))

reg = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8")
lines = reg.split("\n")
ip = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-PGT-1** (2026-10-22) |")]
assert len(ip) == 1
pre = "| **PRE-REGISTERED, NOT RUN (box 2, ~5 h if seed 0 stops it, ~10 h in all; awaits the PI's go).** "
assert lines[ip[0]].count(pre) == 1
lines[ip[0]] = lines[ip[0]].replace(pre, f"| **RUNNING on box 2 ({NOTE}).** ~~PRE-REGISTERED, NOT RUN (box 2, ~5 h if seed 0 stops it, ~10 h in all; awaits the PI's go).~~ ")
assert not any(x.startswith("| **H-V8F-GRID-1**") for x in lines)
row = (
    "| **H-V8F-GRID-1** (2026-10-22) | The stack's scene input at 8×20 tokens per frame (160, vs 4×10 = 40) makes v8F-mini drive better and T's lateral "
    "constraints more accurate: arm G = B + `--grid pool8x20`, ONE lever, where B = M2b-pgt iff `PREREG_V8F_PGT` reads SUCCESS, else M2b (fixed before "
    "PGT's data); seeds 0 then 1 | **PRE-REGISTERED, NOT RUN (PI go 2026-10-22; box 2 after PGT's chain ends, ~6–8 h per seed, ESTIMATED).** "
    "`Project Steering/PREREG_V8F_GRID.md`. PRIMARY: ADE6 G − B < 0, separated, |δ| > 2F at both seeds (F from B's two seeds; 0.1207 m if B = M2b). "
    "MECHANISM with a size bar fixed now: T's latcons present-slot RMSE falls ≥ 12.5 %, separated (half the α = 0.25 move that the frozen-O dose curve "
    "prices at ≈ 0.19 m). Proofs required before any run: the eval of an 8×20 checkpoint reads `pool8x20` (`v8f_mini_eval.py:268` defaults to "
    "`pool4x10`), and a token-count mismatch is refused. Instruments being built. |")
lines[ip[0]:ip[0]] = [row]
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
open(os.path.join(H, "msg.txt"), "wb").write(open(DL + "msg15.txt", "rb").read().replace(b"\r\n", b"\n"))
print("hold ok")
