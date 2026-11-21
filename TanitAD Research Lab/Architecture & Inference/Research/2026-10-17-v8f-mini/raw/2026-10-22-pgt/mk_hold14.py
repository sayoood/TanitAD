"""Build qland/hold_v8fpgt_prereg: PREREG_V8F_PGT + its instruments (reader, chain, queue stages, test, built_levers line), NOT armed.
   python mk_hold14.py <remote tip sha>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
BASE = "0a827845a05795000afbe17ea52c1ef1d475dc08"      # the tip the instruments were built on; every edited file must be unchanged since
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
P, DL = S + "pgt/", S + "dkland/"
H = "C:/Users/Admin/qland/hold_v8fpgt_prereg"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-22-pgt/"


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


def rw_bytes(rel, src):
    """Replace a file with the agent's edited copy, BYTE FOR BYTE (keeps CRLF where the tip has it), only if the tip is unchanged since BASE."""
    old = show(TIP, rel)
    assert old is not None and old == show(BASE, rel), f"{rel}: the tip moved since {BASE[:8]}"
    new = open(src, "rb").read()
    assert new != old and (b"\r\n" in new) == (b"\r\n" in old), f"{rel}: line endings changed"
    put("mods-rw", rel, new)


for n in ("hold_v8fpgt_prereg", "landed_hold_v8fpgt_prereg"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n

stage(DL + "PREREG_V8F_PGT.md", "Project Steering/PREREG_V8F_PGT.md")
stage(P + "stack/tests/test_v8f_pgt.py", "stack/tests/test_v8f_pgt.py")
stage(P + f"{PK}/code/v8f_pgt_read.py", f"{PK}/code/v8f_pgt_read.py")
stage(P + f"{PK}/code/box2_ops/chain_v8f_pgt.sh", f"{PK}/code/box2_ops/chain_v8f_pgt.sh")
for f in ("mutate_pgt.py", "mutate_pgt.out", "real_proofs.py", "real_proofs.out", "check_inputs_identity.json",
          "SYNTHETIC_pgt_read_regression.json", "SYNTHETIC_pgt_read_positive.json"):
    stage(P + f"{PK}/raw/2026-10-22-pgt/" + f, RAW + f)
stage(P + "RESULT_PGT_BUILD.md", RAW + "BUILD.md")
stage(P + "family_final.out", RAW + "family_final.out")
stage(DL + "mk_hold14.py", RAW + "mk_hold14.py")

rw_bytes("stack/tanitad/models/v8f_mini.py", P + "stack/tanitad/models/v8f_mini.py")
rw_bytes(f"{PK}/code/v8f_mini_box2_queue.cmd", P + f"{PK}/code/v8f_mini_box2_queue.cmd")

reg = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8")
lines = reg.split("\n")
iw = [k for k, x in enumerate(lines) if x.startswith("| **D-V8F-DK-1** (2026-10-22) |")]
assert len(iw) == 1
assert not any(x.startswith("| **H-V8F-PGT-1**") for x in lines)
row = (
    "| **H-V8F-PGT-1** (2026-10-22) | Training the stack on its own upstream outputs (scheduled sampling 0.5 → 0.0; M2b: 1.0 → 0.2) makes v8F-mini "
    "drive better and removes the latcons leak (`D-V8F-DK-1`): arm `M2b-pgt` = `--arm M2b --p-gt-start 0.5 --p-gt-end 0.0`, ONE lever set by two "
    "flags, acting on three reads (S→T and S→O share one draw, T→O a second), seeds 0 then 1 | **PRE-REGISTERED, NOT RUN (box 2, ~5 h if seed 0 "
    "stops it, ~10 h in all; awaits the PI's go).** `Project Steering/PREREG_V8F_PGT.md`, the committed FAIL-MECHANISM-OK branch of "
    "`PREREG_V8F_WC`. PRIMARY: ADE6 M2b-pgt − M2b < 0, separated, |δ| > 2F (0.2415) at both seeds. MECHANISM, with size bars fixed in advance: "
    "the latcons leak Λ (+0.268 / +0.167 m at M2b) and the 2 s < 5 s TTC excess over P10 (+4.4 / +2.5 pp) must each at least halve. "
    "Instruments (built and proved before any run, not armed): `code/v8f_pgt_read.py`, `code/box2_ops/chain_v8f_pgt.sh`, queue stages "
    "`gatepgt` / `trainpgt` / `evalpgt`, `stack/tests/test_v8f_pgt.py` (29 tests; 194 / 194 mutants RED); `built_levers()` now records "
    "`p_gt_start` / `p_gt_end`. |")
lines[iw[0]:iw[0]] = [row]
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
open(os.path.join(H, "msg.txt"), "wb").write(lf(open(DL + "msg14.txt", "rb").read()))
print("hold ok")
