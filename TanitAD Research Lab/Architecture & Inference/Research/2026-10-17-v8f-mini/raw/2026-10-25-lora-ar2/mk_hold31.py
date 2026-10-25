"""Build qland/hold_v8flora_ar2: the AR-2 box-2 chain (same-device identity driver, reader, queue stages, chain, tests) + the guard widening.
   python mk_hold31.py <remote tip sha>"""
import glob
import os
import subprocess
import sys

TIP = sys.argv[1]
BASE = "99d59527"
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
X, A2 = S + "lora/", S + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8flora_ar2"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-25-lora-ar2/"


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    assert len(rel) <= 220, rel
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


lf = lambda b: b.replace(b"\r\n", b"\n")
for n in ("hold_v8flora_ar2", "landed_hold_v8flora_ar2"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
q = f"{PK}/code/v8f_mini_box2_queue.cmd"
EDITED = [q, f"{PK}/code/v8f_lora_smoke.py", f"{PK}/code/box2_ops/v8f_guard.sh"]
for rel in EDITED:
    old = show(TIP, rel)
    assert old is not None and old == show(BASE, rel), f"{rel}: the tip moved since {BASE}"
    new = open(X + rel, "rb").read()
    assert new != old, rel
    if rel == q:
        assert new.count(b"\r\n") == new.count(b"\n"), "the queue must stay CRLF"
    else:
        new = lf(new)
    put("mods-rw", rel, new)
NEW = [f"{PK}/code/v8f_lora_identity_dev.py", f"{PK}/code/v8f_lora_ar2_read.py", f"{PK}/code/box2_ops/chain_v8f_lora_ar2.sh", "stack/tests/test_v8f_lora_ar2.py"]
NEW += sorted(RAW + os.path.basename(p) for p in glob.glob(X + RAW + "*") if os.path.isfile(p) and not p.endswith((".pt", ".pyc")))
for rel in NEW:
    assert show(TIP, rel) is None, rel
    put("stage", rel, lf(open(X + rel, "rb").read()))
for src, name in (("lora_work/final/mm_suites_ar2.out", "mm_suites_ar2.out"), ("lora_work/final/mm_ar2_rerun.out", "mm_ar2_rerun.out")):
    put("stage", RAW + name, lf(open(S + src, "rb").read()))
put("stage", RAW + "mk_hold31.py", lf(open(A2 + "mk_hold31.py", "rb").read()))
lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-LORA-1** (2026-10-25) |")]
assert len(ix) == 1 and lines[ix[0]].endswith(" |") and lines[ix[0]].count(" | ") == 2
lines[ix[0]] = lines[ix[0]][:-2] + (" **AR-2 chain landed 2026-10-25** (`code/box2_ops/chain_v8f_lora_ar2.sh`, not armed):"
                                    " the same-device identity on box 2's cuda plus its mutant, then the 200-step smoke. AR-2 also fails if the TRAINING"
                                    " path's own identity probe misses τ_device. The guard can now kill both LoRA scripts. Arming needs the PI's go after DSCALE. |")
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
open(os.path.join(H, "msg.txt"), "wb").write(lf(open(A2 + "msg31.txt", "rb").read()))
print("hold ok", len(NEW), "new files")
