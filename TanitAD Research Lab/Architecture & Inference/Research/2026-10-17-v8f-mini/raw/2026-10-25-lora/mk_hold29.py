"""Build qland/hold_v8flora_instr: PREREG_V8F_LORA's instruments (from the lora export) + Amendment 1 + the register.
   python mk_hold29.py <remote tip sha>"""
import glob, os, subprocess, sys
TIP = sys.argv[1]
BASE = "190be80a"
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
X, A2 = S + "lora/", S + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8flora_instr"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-25-lora/"
def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None
def put(kind, rel, data):
    assert len(rel) <= 220, rel
    p = os.path.join(H, kind, rel); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "wb").write(data)
lf = lambda b: b.replace(b"\r\n", b"\n")
for n in ("hold_v8flora_instr", "landed_hold_v8flora_instr"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
EDITED = ["stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_mini_data.py", "stack/tests/test_v8f_window_start.py"]
for rel in EDITED:
    old = show(TIP, rel); assert old is not None and old == show(BASE, rel), f"{rel}: the tip moved since {BASE}"
    new = lf(open(X + rel, "rb").read()); assert new != old, rel
    put("mods-rw", rel, new)
assert show(TIP, "stack/tanitad/models/v8f_mini.py") == lf(open(X + "stack/tanitad/models/v8f_mini.py", "rb").read()), "v8f_mini.py must be unchanged"
NEW = ["stack/tanitad/models/v8f_lora_encoder.py", "stack/tests/test_v8f_lora.py", f"{PK}/code/v8f_lora_eval_cache.py", f"{PK}/code/v8f_lora_smoke.py"]
NEW += sorted(RAW + os.path.basename(p) for p in glob.glob(X + RAW + "*") if os.path.isfile(p) and not p.endswith(".pyc"))
for rel in NEW:
    assert show(TIP, rel) is None, rel
    put("stage", rel, lf(open(X + rel, "rb").read()))
put("stage", RAW + "mm_suites_after_fix.out", lf(open(S + "lora_work/final/mm_suites.out", "rb").read()))
put("stage", RAW + "mk_hold29.py", lf(open(A2 + "mk_hold29.py", "rb").read()))
pz = "Project Steering/PREREG_V8F_LORA.md"
cur = show(TIP, pz); assert cur is not None and b"## Amendment 1" not in cur
put("mods", pz, cur.rstrip(b"\n") + b"\n" + lf(open(A2 + "lora_amend1.md", "rb").read()))
lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-LORA-1** (2026-10-25) |")]
assert len(ix) == 1
o = "instruments being built with no GPU, PI go 2026-10-25"
assert lines[ix[0]].count(o) == 1
lines[ix[0]] = lines[ix[0]].replace(o, "instruments LANDED with Amendment 1, no GPU, PI go 2026-10-25. Same-device identity EXACT (0.0); the train-frame ROUTE is NOT yet reproduced (native-256 rebuild vs the thor416 train cache rel-L2 0.079–0.090, τ_route 1e-3), which blocks the AR-2 smoke until the route is reproduced or the base is rebuilt (Amendment 1 items 1–3)")
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
open(os.path.join(H, "msg.txt"), "wb").write(lf(open(A2 + "msg29.txt", "rb").read()))
print("hold ok", len(NEW), "new files")
