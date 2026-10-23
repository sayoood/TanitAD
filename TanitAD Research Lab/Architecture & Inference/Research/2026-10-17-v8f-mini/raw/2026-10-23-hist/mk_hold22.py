"""Build qland/hold_v8fhist_instr: PREREG_V8F_HIST's instruments + Amendment 1's --window-start (trainer, data, model, both evals) + Amendment 2 + the register.
   python mk_hold22.py <remote tip sha>"""
import glob
import os
import subprocess
import sys

TIP = sys.argv[1]
BASE = "6bfeced5"                                # the tip the export was taken from; every edited file must be unchanged on the tip since
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
X, A2 = S + "hist/", S + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8fhist_instr"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-23-hist/"


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    assert len(rel) <= 220, rel
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def lf(b):
    return b.replace(b"\r\n", b"\n")


for n in ("hold_v8fhist_instr", "landed_hold_v8fhist_instr"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n

# ---- edited files (mods-rw): the tip must still hold the export's base version
EDITED = ["stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_mini_data.py", "stack/tanitad/models/v8f_mini.py",
          "stack/scripts/v8f_mini_eval.py", "stack/scripts/v8f_t1_eval.py", f"{PK}/code/v8f_mini_box2_queue.cmd"]
for rel in EDITED:
    old = show(TIP, rel)
    assert old is not None and old == show(BASE, rel), f"{rel}: the tip moved since {BASE}"
    new = open(X + rel, "rb").read()
    assert new != old, f"{rel} is unchanged"
    if rel.endswith(".cmd"):
        assert new.count(b"\r\n") == new.count(b"\n"), "the queue must stay CRLF"
    else:
        new = lf(new)
    put("mods-rw", rel, new)

# ---- new files (stage)
NEW = ["stack/tests/test_v8f_hist.py", "stack/tests/test_v8f_window_start.py", f"{PK}/code/v8f_hist_read.py", f"{PK}/code/box2_ops/chain_v8f_hist.sh"]
for rel in NEW:
    assert show(TIP, rel) is None, f"{rel} already exists on the tip"
    put("stage", rel, lf(open(X + rel, "rb").read()))
n_raw = 0
for p in sorted(glob.glob(X + RAW + "**/*", recursive=True)):
    if os.path.isdir(p) or "__pycache__" in p or p.endswith(".pyc"):
        continue
    rel = RAW + os.path.relpath(p, X + RAW).replace("\\", "/")
    b = open(p, "rb").read()
    on_tip = show(TIP, rel)
    if on_tip is not None:                       # mk_hold20.py / mk_hold21.py landed earlier
        assert lf(b) == on_tip, f"{rel} differs from the tip"
        continue
    put("stage", rel, b if p.endswith((".ref", ".cmd", ".npz")) else lf(b))      # .ref: byte copies of the pre-change sources (the queue's is CRLF)
    n_raw += 1
put("stage", RAW + "mk_hold22.py", lf(open(A2 + "mk_hold22.py", "rb").read()))
print("raw files staged:", n_raw)

# ---- Amendment 2, appended
pz = "Project Steering/PREREG_V8F_HIST.md"
cur = show(TIP, pz)
assert cur is not None and b"## Amendment 1" in cur and b"## Amendment 2" not in cur
put("mods", pz, cur.rstrip(b"\n") + b"\n" + lf(open(A2 + "hist_amend2.md", "rb").read()))

# ---- the register
reg = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8")
lines = reg.split("\n")
ih = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-HIST-1** (2026-10-23) |")]
assert len(ih) == 1
o = " Instruments being built. |"
assert lines[ih[0]].endswith(o)
lines[ih[0]] = lines[ih[0]][: -len(o)] + " " + open(A2 + "reg22.txt", "rb").read().decode("utf-8").strip() + " |"
assert lines[ih[0]].count(" | ") == 2, "the row must keep three cells"
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
open(os.path.join(H, "msg.txt"), "wb").write(lf(open(A2 + "msg22.txt", "rb").read()))
print("hold ok")
