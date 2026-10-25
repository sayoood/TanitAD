"""Build qland/hold_v8ftattrib: the T-error attribution (raw/2026-10-25-t-attrib/), R70, RESULT_V8F_DSCALE corrected in place, register rows
   D-V8F-T-ATTRIB-1 (new) and H-V8F-DSCALE-1 (corrected).   python mk_hold33.py <remote tip sha>"""
import glob
import os
import subprocess
import sys

TIP = sys.argv[1]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/grid_a2/"
TA = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/t_attrib/"
H = "C:/Users/Admin/qland/hold_v8ftattrib"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-25-t-attrib/"


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


def stage(src, rel, binary=False, data=None):
    assert show(TIP, rel) is None, f"{rel} already exists on the tip"
    b = open(src, "rb").read() if data is None else data
    assert b, src
    put("stage", rel, b if binary else lf(b))


for n in ("hold_v8ftattrib", "landed_hold_v8ftattrib"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
stage(S + "raw_readme_tattrib.md", RAW + "README.md")
stage(S + "mk_hold33.py", RAW + "mk_hold33.py")
stage(TA + "T_ATTRIB.md", RAW + "T_ATTRIB.md")
tc = open(TA + "tcommon.py", "rb").read().decode("utf-8")
old_s = 'S = "' + TA + '"'
assert tc.count(old_s) == 1 and "import numpy as np\n" in tc
tc = tc.replace("import numpy as np\n", "import os\n\nimport numpy as np\n", 1)
tc = tc.replace(old_s, 'S = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/") + "/"')
stage(None, RAW + "tcommon.py", data=tc.encode("utf-8"))
scripts = sorted(glob.glob(TA + "s*.py"))
assert len(scripts) == 15, [os.path.basename(p) for p in scripts]
for p in scripts:
    o = p[:-3] + ".out"
    assert os.path.isfile(o), o
    stage(p, RAW + os.path.basename(p))
    stage(o, RAW + os.path.basename(o))
for f in ("table.npz", "gap_M2b_s0.npz", "gap_M2b_s1.npz"):
    stage(TA + f, RAW + f, binary=True)
for p in sorted(glob.glob(TA + "raw_in/*")):
    b = os.path.basename(p)
    assert open(p, "rb").read() == show(TIP, f"{PK}/raw/{b}"), f"raw_in/{b} is not a byte copy of the banked raw/{b}"
    stage(p, RAW + "raw_in/" + b, binary=True)

rl = "Project Steering/RETRACTION_LOG.md"
old = show(TIP, rl)
assert old.endswith(b"\n") and b"### R70 " not in old and b"### R69 " in old
put("mods", rl, old + lf(open(S + "r70.md", "rb").read()))

rs = f"{PK}/RESULT_V8F_DSCALE.md"
t = show(TIP, rs).decode("utf-8")
a = ("* **The generalisation signature:** T's TRAINING constraint loss over the last 10 % of steps was LOWER with half the data (0.1209 against 0.1264, −4.3 %), "
     "while its eval error rose. T fits its smaller set better and generalises worse. That is the same reading WC gave.")
b = "3. **T is generalisation-limited, and the levers that act on its inputs, loss and data volume move it by single-digit percent.**"
assert t.count(a) == 1 and t.count(b) == 1, (t.count(a), t.count(b))
t = t.replace(a, "* ~~" + a[2:] + "~~\n  **CORRECTED 2026-10-25 (R70, `raw/2026-10-25-t-attrib/s6_trainval.out`):** the −4.3 % is about one standard error"
              " (z ≈ 0.96 over 40 single logged batches), so it is no signature. In the trainer's own units M2b_s0's eval Huber, 0.1355 [0.1178, 0.1548],"
              " contains its train Huber, 0.1264: at full data T is FIT-limited, not generalisation-limited.")
t = t.replace(b, "3. ~~**T is generalisation-limited, and**~~ **CORRECTED (R70): T is fit-limited at full data (eval ≈ train error), and the levers that act on"
              " its inputs, loss and data volume move it by single-digit percent.**")
put("mods-rw", rs, t.encode("utf-8"))

lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-DSCALE-1** (2026-10-25) |")]
assert len(ix) == 1
c = "T's train constraint loss −4.3 % with half the data (generalisation-limited signature);"
assert lines[ix[0]].count(c) == 1
lines[ix[0]] = lines[ix[0]].replace(c, "~~" + c[:-1] + "~~ (CORRECTED R70: one SE, no signature; at full data T is fit-limited, eval ≈ train);")
new = open(S + "reg33_tattrib.txt", "rb").read().decode("utf-8").strip()
assert new.startswith("| **D-V8F-T-ATTRIB-1**") and new.count(" | ") == 2 and not any(x.startswith("| **D-V8F-T-ATTRIB-1**") for x in lines)
lines.insert(ix[0] + 1, new)
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
open(os.path.join(H, "msg.txt"), "wb").write(lf(open(S + "msg33.txt", "rb").read()))
print("hold ok")
