"""Build qland/hold_v8flora_route: PREREG_V8F_LORA Amendment 2 + R68 + the route evidence + the register correction.   python mk_hold30.py <tip>"""
import glob, os, subprocess, sys
TIP = sys.argv[1]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
X, A2 = S + "lora/", S + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8flora_route"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-25-lora-route/"


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    assert len(rel) <= 220, rel
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


lf = lambda b: b.replace(b"\r\n", b"\n")
for n in ("hold_v8flora_route", "landed_hold_v8flora_route"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
# the 8.3 MB CPU reference .pt is regenerable by run_repro.sh and stays out of git
fs = sorted(p for p in glob.glob(X + RAW + "*") if os.path.isfile(p) and not p.endswith(".pt"))
assert any(p.endswith("ROUTE.md") for p in fs) and len(fs) == 35, len(fs)
for p in fs:
    rel = RAW + os.path.basename(p)
    assert show(TIP, rel) is None, rel
    put("stage", rel, lf(open(p, "rb").read()))
put("stage", RAW + "mk_hold30.py", lf(open(A2 + "mk_hold30.py", "rb").read()))
pz = "Project Steering/PREREG_V8F_LORA.md"
cur = show(TIP, pz)
assert cur is not None and b"## Amendment 1" in cur and b"## Amendment 2" not in cur
put("mods", pz, cur.rstrip(b"\n") + b"\n" + lf(open(A2 + "lora_amend2.md", "rb").read()))
rl = "Project Steering/RETRACTION_LOG.md"
cur = show(TIP, rl)
assert cur is not None and b"### R68 " not in cur and b"### R67 " in cur
put("mods", rl, cur.rstrip(b"\n") + b"\n" + lf(open(A2 + "r68.md", "rb").read()))
lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-LORA-1** (2026-10-25) |")]
assert len(ix) == 1
w = ("the train-frame ROUTE is NOT yet reproduced (native-256 rebuild vs the thor416 train cache rel-L2 0.079–0.090, τ_route 1e-3), "
     "which blocks the AR-2 smoke until the route is reproduced or the base is rebuilt (Amendment 1 items 1–3)")
assert lines[ix[0]].count(w) == 1
lines[ix[0]] = lines[ix[0]].replace(w, "~~" + w + "~~ **CORRECTED the same day (R68, Amendment 2):** the production train cache is `local256` on "
                                    "4,572 / 4,572 rows and its frames rebuild byte-identically; the identity control had compared against a smoke "
                                    "cache. It now runs SAME-DEVICE on box 2's GPU (τ_device 1e-3, `--amp` OFF), which needs its own go after DSCALE")
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
open(os.path.join(H, "msg.txt"), "wb").write(lf(open(A2 + "msg30.txt", "rb").read()))
print("hold ok", len(fs), "files")
