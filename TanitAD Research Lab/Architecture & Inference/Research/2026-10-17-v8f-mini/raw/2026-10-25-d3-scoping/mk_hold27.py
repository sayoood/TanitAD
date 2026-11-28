"""Build qland/hold_v8f_d3scope: the D3 scoping (stage only) + a pointer on D-DATA-COVERAGE-1.   python mk_hold27.py <tip>"""
import glob, os, subprocess, sys
TIP = sys.argv[1]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
X, A2 = S + "hist/", S + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8f_d3scope"
MIR = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-25-d3-scoping/"
def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None
def put(kind, rel, data):
    p = os.path.join(H, kind, rel); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "wb").write(data)
lf = lambda b: b.replace(b"\r\n", b"\n")
for n in ("hold_v8f_d3scope", "landed_hold_v8f_d3scope"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
fs = sorted(p for p in glob.glob(X + R + "*") if os.path.isfile(p))
assert any(p.endswith("D3_SCOPING.md") for p in fs) and len(fs) <= 12, fs
for p in fs:
    rel = R + os.path.basename(p); assert show(TIP, rel) is None, rel
    put("stage", rel, lf(open(p, "rb").read()))
put("stage", R + "mk_hold27.py", lf(open(A2 + "mk_hold27.py", "rb").read()))
lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **D-DATA-COVERAGE-1** (2026-10-17) |")]
assert len(ix) == 1 and lines[ix[0]].endswith(" |") and lines[ix[0]].count(" | ") == 2
lines[ix[0]] = lines[ix[0]][:-2] + (" **D3 scoped 2026-10-25** (`…/2026-10-17-v8f-mini/raw/2026-10-25-d3-scoping/D3_SCOPING.md`):"
    " +4,512 clips ESTIMATED at 65–103 GB download, +19.7 GB cache, ~6.3 h box-2 encode. MEASURED traps: the RA-1 holdout is a"
    " positional sha12 slice (`v8f_mini_train.py:328`), the encoder refuses non-B1 clips, and `cons_stats` would re-standardise T's"
    " targets; each must be fixed before a D3 arm. |")
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
open(os.path.join(H, "msg.txt"), "wb").write(lf(open(A2 + "msg27.txt", "rb").read()))
print("hold ok", len(fs), "files")
