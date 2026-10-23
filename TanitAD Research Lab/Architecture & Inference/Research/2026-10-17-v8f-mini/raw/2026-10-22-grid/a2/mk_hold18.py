"""Build qland/hold_v8fgrid_a2: PREREG_V8F_GRID Amendment 2 (the reader's per-key loss-weight check, written BEFORE any GRID number was read).
   python mk_hold18.py <remote tip sha>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
BASE = "ed67a216"                                   # the tip the fix was built on; the two edited files must be unchanged since
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
A2, X = S + "grid_a2/", S + "dkv_abccc45/"
H = "C:/Users/Admin/qland/hold_v8fgrid_a2"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-22-grid/a2/"
LG = "C:/Users/Admin/box2_ship/logs/"


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


def stage(src, rel):
    assert show(TIP, rel) is None, f"{rel} already exists on the tip"
    b = open(src, "rb").read()
    assert b, src
    put("stage", rel, lf(b))


def rw(rel, src):
    old = show(TIP, rel)
    assert old is not None and old == show(BASE, rel), f"{rel}: the tip moved since {BASE}"
    assert old == open(A2 + "clean/" + os.path.basename(rel), "rb").read(), f"{rel}: the clean copy is not the base"
    new = lf(open(src, "rb").read())
    assert new != old, rel
    put("mods-rw", rel, new)


for n in ("hold_v8fgrid_a2", "landed_hold_v8fgrid_a2"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n

rw(f"{PK}/code/v8f_grid_read.py", X + f"{PK}/code/v8f_grid_read.py")
rw("stack/tests/test_v8f_grid.py", X + "stack/tests/test_v8f_grid.py")
for f in ("mutate_a2.py", "mutate_a2.out", "tests_a2.out", "mk_hold18.py"):
    stage(A2 + f, RAW + f)
stage(LG + "chain_v8f_grid_run1.out", RAW + "chain_v8f_grid_run1.out")      # the refusal, as the first run logged it

pz = "Project Steering/PREREG_V8F_GRID.md"
cur = show(TIP, pz)
assert cur is not None and b"## Amendment 1" in cur and b"## Amendment 2" not in cur
put("mods", pz, cur.rstrip(b"\n") + b"\n" + lf(open(A2 + "grid_amend2.md", "rb").read()))

reg = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8")
lines = reg.split("\n")
ig = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-GRID-1** (2026-10-22) |")]
assert len(ig) == 1
o = " The chain is armed behind PGT's. |"
assert lines[ig[0]].endswith(o)
lines[ig[0]] = lines[ig[0]][: -len(o)] + (
    " **RUNNING on box 2 (2026-10-22/23):** base B = M2b (PGT read FAIL-NO-MECHANISM); G gate PASS (+30,720 parameters exactly, lru 43); G s0 trained "
    "22:22–03:07 and evaluated. The first seed-0 read REFUSED on a lever-record schema difference (M2b's banked record predates `w_tac` / `w_cons`; both "
    "arms train at the trainer's 1.0 / 0.5), fixed by **Amendment 2 before any GRID number was read** (per-key `w_diff`; 6 / 6 mutants RED); chain relaunched "
    "from the amended tip. |")
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
open(os.path.join(H, "msg.txt"), "wb").write(lf(open(A2 + "msg18.txt", "rb").read()))
print("hold ok")
