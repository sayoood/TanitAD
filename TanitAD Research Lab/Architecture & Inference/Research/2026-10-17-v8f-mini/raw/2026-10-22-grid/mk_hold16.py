"""Build qland/hold_v8fgrid_instr: PREREG_V8F_GRID's instruments + Amendment 1 (before any GRID data) + the register note.
   python mk_hold16.py <remote tip sha>"""
import glob
import os
import subprocess
import sys

TIP = sys.argv[1]
BASE = "c61f0167"                                    # the tip the instruments were built on; the queue must be unchanged since
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
G, DL = S + "grid/", S + "dkland/"
H = "C:/Users/Admin/qland/hold_v8fgrid_instr"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-22-grid/"


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


for n in ("hold_v8fgrid_instr", "landed_hold_v8fgrid_instr"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n

stage(G + "stack/tests/test_v8f_grid.py", "stack/tests/test_v8f_grid.py")
for f in ("v8f_grid_read.py", "check_grid_cache.py", "box2_ops/chain_v8f_grid.sh"):
    stage(G + f"{PK}/code/{f}", f"{PK}/code/{f}")
WANT = sorted([f"SYNTHETIC_grid_read_{k}_{b}.json" for k in ("mechanism_only", "regression", "success") for b in ("M2b", "M2b-pgt")]
              + [f"grid_read_check_inputs_{b}.json" for b in ("M2b", "M2b-pgt")]
              + [f"{x}.{e}" for x in ("mutate_grid", "probe_eval_mismatch", "real_proofs") for e in ("py", "out")])
got = sorted(os.path.basename(p) for p in glob.glob(G + f"{PK}/raw/2026-10-22-grid/*"))
assert got == WANT, (got, WANT)                     # the 14 files BUILD.md's manifest lists, by name
for f in got:
    stage(G + f"{PK}/raw/2026-10-22-grid/{f}", RAW + f)
stage(G + "RESULT_GRID_BUILD.md", RAW + "BUILD.md")
for f in ("mutation_wait_1320.out", "tests_live_tip.out", "mk_hold16.py"):
    stage(DL + f, RAW + f)

q = f"{PK}/code/v8f_mini_box2_queue.cmd"
old = show(TIP, q)
assert old == show(BASE, q), "the queue moved since the instruments' base"
new = open(G + q, "rb").read()
assert new != old and new.count(b"\r\n") == new.count(b"\n"), "the queue must stay CRLF"
put("mods-rw", q, new)

pz = "Project Steering/PREREG_V8F_GRID.md"
cur = show(TIP, pz)
assert cur is not None and b"## Amendment 1" not in cur
put("mods", pz, cur.rstrip(b"\n") + b"\n" + lf(open(DL + "grid_amend1.md", "rb").read()))

reg = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8")
lines = reg.split("\n")
ig = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-GRID-1** (2026-10-22) |")]
assert len(ig) == 1
o = "Instruments being built. |"
assert lines[ig[0]].endswith(o)
lines[ig[0]] = lines[ig[0]][: -len(o)] + (
    "Instruments LANDED with Amendment 1 (before any data): `code/v8f_grid_read.py`, `code/check_grid_cache.py`, `code/box2_ops/chain_v8f_grid.sh`, "
    "queue stages `gategrid` / `traingrid` / `evalgrid`, `stack/tests/test_v8f_grid.py` (127 tests; 179 / 179 mutants RED). Amendment 1 declares the "
    "sampler chunk (48 → ~41) and +30,720 parameters part of the lever, and fixes M-LAT to the slot-pooled RMSE. The chain is armed behind PGT's. |")
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
open(os.path.join(H, "msg.txt"), "wb").write(lf(open(DL + "msg16.txt", "rb").read()))
print("hold ok")
