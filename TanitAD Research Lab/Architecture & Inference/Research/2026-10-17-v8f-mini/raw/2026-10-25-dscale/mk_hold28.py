"""Build qland/hold_v8fdscale_instr: DSCALE's instruments from the export + Amendment 1 + the register.   python mk_hold28.py <tip>"""
import glob, os, subprocess, sys
TIP = sys.argv[1]
BASE = "343f366d"                                # the export's base; every edited file must be unchanged on the tip since
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
X, A2 = S + "hist/", S + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8fdscale_instr"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-25-dscale/"
def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None
def put(kind, rel, data):
    assert len(rel) <= 220, rel
    p = os.path.join(H, kind, rel); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "wb").write(data)
lf = lambda b: b.replace(b"\r\n", b"\n")
for n in ("hold_v8fdscale_instr", "landed_hold_v8fdscale_instr"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
q = f"{PK}/code/v8f_mini_box2_queue.cmd"
old = show(TIP, q); assert old is not None and old == show(BASE, q), "the queue moved on the tip"
new = open(X + q, "rb").read(); assert new != old and new.count(b"\r\n") == new.count(b"\n"), "the queue must stay CRLF"
put("mods-rw", q, new)
NEW = [f"{PK}/code/v8f_dscale_read.py", f"{PK}/code/box2_ops/chain_v8f_dscale.sh", "stack/tests/test_v8f_dscale.py"]
NEW += sorted(RAW + os.path.basename(p) for p in glob.glob(X + RAW + "*") if os.path.isfile(p))
assert len(NEW) == 3 + 12, NEW
for rel in NEW:
    assert show(TIP, rel) is None, rel
    put("stage", rel, lf(open(X + rel, "rb").read()))
put("stage", RAW + "mk_hold28.py", lf(open(A2 + "mk_hold28.py", "rb").read()))
pz = "Project Steering/PREREG_V8F_DSCALE.md"
cur = show(TIP, pz); assert cur is not None and b"## Amendment 1" not in cur
put("mods", pz, cur.rstrip(b"\n") + b"\n" + lf(open(A2 + "dscale_amend1.md", "rb").read()))
lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-DSCALE-1** (2026-10-25) |")]
assert len(ix) == 1
o = " Instruments being built."
assert lines[ix[0]].count(o) == 1
lines[ix[0]] = lines[ix[0]].replace(o, " Instruments landed with Amendment 1 (459 CPU tests pass, 49/49 mutants RED; the clip list MEASURED: Q 2,256 nested clips, 153,271 windows, 19,164 steps).")
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
open(os.path.join(H, "msg.txt"), "wb").write(lf(open(A2 + "msg28.txt", "rb").read()))
print("hold ok", len(NEW), "new files")
