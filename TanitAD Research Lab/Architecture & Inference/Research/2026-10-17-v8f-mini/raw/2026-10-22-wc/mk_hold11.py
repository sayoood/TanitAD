"""Build qland/hold_v8fwc: PREREG_V8F_WC (pre-registered, NOT run) + its instruments, proven before any run.
   python mk_hold11.py <remote tip sha> <n RED> <n mutations>"""
import os
import subprocess
import sys

TIP, NRED, NMUT = sys.argv[1], sys.argv[2], sys.argv[3]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/tac/"
T4 = S + "t4/"
BASE = "63eb6c5aa78ee36c7e89dc24cc958c0390ec7723"          # the tip t4 was exported from; every whole-file edit is checked against it
H = "C:/Users/Admin/qland/hold_v8fwc"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-22-wc/"


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def lf(b):
    return b.replace(b"\r\n", b"\n")


def stage(src, rel, binary=False):
    assert show(TIP, rel) is None, f"{rel} already exists on the tip"
    b = open(src, "rb").read()
    put("stage", rel, b if binary else lf(b))


def rw(rel, new, crlf=False, base_check=True):
    old = show(TIP, rel)
    assert old is not None, rel
    if base_check:
        assert old == show(BASE, rel), f"{rel}: the tip moved since the edit base {BASE[:8]}"
    new = new if crlf else lf(new)
    assert new != old, rel
    put("mods-rw", rel, new)


for n in ("hold_v8fwc", "landed_hold_v8fwc"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n

# ---- instruments
rw("stack/tanitad/models/v8f_mini.py", open(T4 + "stack/tanitad/models/v8f_mini.py", "rb").read())
q = f"{PK}/code/v8f_mini_box2_queue.cmd"
rw(q, open(T4 + q, "rb").read(), crlf=True)
stage(T4 + "stack/tests/test_v8f_wc.py", "stack/tests/test_v8f_wc.py")
for f in ("v8f_wc_read.py", "box2_ops/chain_v8f_wc.sh"):
    stage(T4 + f"{PK}/code/{f}", f"{PK}/code/{f}")
stage(S + "PREREG_V8F_WC.md", "Project Steering/PREREG_V8F_WC.md")
for f in ("mutate_wc.py", "mutate_wc.out", "real_queue_probe.py", "real_queue_probe.out", "mk_hold11.py"):
    stage(S + f, RAW + f)
stage(S + "raw_readme_wc.md", RAW + "README.md")

# ---- RESULT_V8F_CZ points at the pre-registered next lever (append-only inside a whole-file edit)
rc = f"{PK}/RESULT_V8F_CZ.md"
old = show(TIP, rc).decode("utf-8")
rw(rc, (old.rstrip("\n") + "\n* **Pre-registered (2026-10-22): `PREREG_V8F_WC`** (H-V8F-WC-1): M2b-wc = M2b + `--w-cons 4.0`, read in training. "
        "Seed 0 decides whether seed 1 trains. It awaits the PI's box-2 go.\n").encode("utf-8"), base_check=False)

# ---- the register
reg = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8")
lines = reg.split("\n")
ic = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-CZ-1** (2026-10-21) |")]
assert len(ic) == 1 and "H-V8F-WC-1" not in reg
row = ("| **H-V8F-WC-1** (2026-10-22) | Up-weighting T's constraint loss makes T's PRESENT constraint slots more accurate and v8F-mini drive "
       "better: arm `M2b-wc` = `--arm M2b --w-cons 4.0` (M2b: 0.5), ONE lever moved, seeds 0 then 1 (`PREREG_V8F_WC`, written before its "
       "data; the committed FAIL branch of H-V8F-CZ-1) | **PRE-REGISTERED, NOT RUN (box 2, ~5 h if seed 0 stops it, ~10 h in all; awaits "
       "the PI's go).** PRIMARY (T1-class open loop, eval-141): per seed, ADE6 M2b-wc − M2b < 0, separated, > 2F (2F 0.2415); SUCCESS "
       "needs both seeds, and seed 1 trains only if seed 0 passes. MECHANISM (reads a FAIL): T's present-slot (T − GT)² lower, separated, "
       "from the absmask + GT-oracle evals. Next per verdict: SUCCESS ⇒ the T1 closed-loop read; FAIL-MECHANISM-OK ⇒ O's use of the "
       "constraints; FAIL-NO-MECHANISM ⇒ T's input / capacity (PI go). Launch gate: the trainer's G-DVB incl. `w_cons` (now recorded in "
       f"`levers.w`), G-LIVE; re-asserted by the chain. Instruments: an exactly-zero gradient control; {NRED} / {NMUT} mutations RED. |")
lines[ic[0]] = row + "\n" + lines[ic[0]]
rw("Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"), base_check=False)
open(os.path.join(H, "msg.txt"), "wb").write(lf(open(S + "msg11.txt", "rb").read()).replace(b"<NRED>", NRED.encode()).replace(b"<NMUT>", NMUT.encode()))
print("hold ok")
