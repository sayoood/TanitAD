"""Build qland/hold_v8f_causalbucket_result: the read of PREREG_V8F_T1KIN_CAUSALBUCKET, its runs, controls and code (the `--limit causal` mode
into stack/, with its tests), the register row, the PI-queue note and the TRAPS entry.  python mk_hold90.py <tip> <register text file> <piq note file>"""
import glob
import hashlib
import os
import re
import subprocess
import sys

TIP, REGTXT, PIQ = sys.argv[1:4]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
T = SC + "t1kin/"
RUN = "C:/Users/Admin/t1kc_run/"
W = "C:/Users/Admin/t1kc/"
H = "C:/Users/Admin/qland/hold_v8f_causalbucket_result"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RD = f"{PK}/raw/2026-10-26-t1kin-causalbucket"
TOKEN = re.compile(rb"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")
TOOL_RAW = "d0ac8ce685ea9083083d380d4764510ce43677439eaee27cf3ab8760df0303ce"


def git(*a):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data, text=True):
    assert not TOKEN.search(data), f"token-shaped string in {rel}"
    if kind == "stage":
        assert git("rev-parse", f"{TIP}:{rel}") is None, f"NEW path exists at the tip: {rel}"
    else:
        assert git("rev-parse", f"{TIP}:{rel}") is not None, f"modified path absent at the tip: {rel}"
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data.replace(b"\r\n", b"\n") if text else data)


for n in ("hold_v8f_causalbucket_result", "landed_hold_v8f_causalbucket_result"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
chain = open(RUN + "logs/chain.out", encoding="utf-8").read()
assert "ZZT1KC-DONE 4" in chain.split("RELAUNCH 2")[-1], "the relaunched chain has not finished all 4 runs"
tool = open(W + "stack/scripts/v8f_t1_eval.py", "rb").read()
assert hashlib.sha256(tool).hexdigest() == TOOL_RAW, "the tool on disk is not the one the runs read"
# the tool the runs read, into stack/ (the tip's file must still be the base it was patched from: CR-stripped content == the d2ed136d blob)
base = git("show", "d2ed136d:stack/scripts/v8f_t1_eval.py")
assert git("show", f"{TIP}:stack/scripts/v8f_t1_eval.py") == base, "stack/scripts/v8f_t1_eval.py moved at the tip since d2ed136d: merge first"
put("mods", "stack/scripts/v8f_t1_eval.py", tool)
put("stage", "stack/tests/test_v8f_t1_causal.py", open(W + "stack/tests/test_v8f_t1_causal.py", "rb").read())
put("stage", f"{PK}/RESULT_V8F_T1KIN_CAUSALBUCKET.md", open(T + "RESULT_V8F_T1KIN_CAUSALBUCKET.md", "rb").read())
put("stage", f"{RD}/t1kc_read.py", open(T + "t1kc_read.py", "rb").read())
for f in glob.glob(T + "readc/*"):
    put("stage", f"{RD}/read/{os.path.basename(f)}", open(f, "rb").read())
for f in sorted(glob.glob(RUN + "t1v8f_*.json")) + [RUN + "c_l3.json", RUN + "c_l2_pytest.log"]:
    put("stage", f"{RD}/run/{os.path.basename(f)}", open(f, "rb").read())
for f in sorted(glob.glob(RUN + "logs/*")):
    put("stage", f"{RD}/run/logs/{os.path.basename(f)}", open(f, "rb").read())
for f in sorted(glob.glob(RUN + "killed_0119/*")):
    put("stage", f"{RD}/run/killed_0119/{os.path.basename(f)}", open(f, "rb").read())
for side in ("cl1_unpatched", "cl1_patched"):
    for f in sorted(glob.glob(RUN + side + "/t1v8f_*.json")):
        put("stage", f"{RD}/run/{side}/{os.path.basename(f)}", open(f, "rb").read())
for f in sorted(glob.glob(T + "consolidated_c/*")):
    put("stage", f"{RD}/run/dumps/{os.path.basename(f)}", open(f, "rb").read(), text=not f.endswith(".npz"))
for src, name in (("C:/Users/Admin/t1kc_chain.sh", "t1kc_chain.sh"), ("C:/Users/Admin/t1kc_cl3.py", "t1kc_cl3.py"),
                  (SC + "grid_a2/mk_hold90.py", "mk_hold90.py")):
    put("stage", f"{RD}/run/{name}" if name != "mk_hold90.py" else f"{RD}/{name}", open(src, "rb").read())
put("stage", f"{RD}/run/t1kc_chain.cmd", open("C:/Users/Admin/t1kc_chain.cmd", "rb").read(), text=False)

g = "Project Steering/GOALS_AND_CLAIMS.md"
lines = git("show", f"{TIP}:{g}").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-CAUSALBUCKET-1** (2026-10-26) |")]
assert len(ix) == 1
lines[ix[0]] = lines[ix[0]].rstrip().rstrip("|").rstrip() + " " + open(REGTXT, encoding="utf-8").read().strip() + " |"
put("mods-rw", g, "\n".join(lines).encode("utf-8"))
q = "Project Steering/PI_DECISION_QUEUE.md"
qt = git("show", f"{TIP}:{q}").decode("utf-8")
anchor = "**What does NOT wait on this:** `PREREG_V8F_HACC_LAT` (the lateral lever), the m = 5 stress read, FLAT and OACC on box 2.\n"
assert qt.count(anchor) == 1
put("mods-rw", q, qt.replace(anchor, anchor + "\n" + open(PIQ, encoding="utf-8").read().strip() + "\n").encode("utf-8"))
tr = "Project Steering/TRAPS.md"
tt = git("show", f"{TIP}:{tr}").decode("utf-8")
a = "\n## Git hygiene"
assert tt.count(a) == 1 and "A MUTATION PROOF ON A TREE THAT A LIVE CHAIN READS" not in tt
put("mods-rw", tr, tt.replace(a, "\n" + open(T + "TRAP_ADD.md", encoding="utf-8").read().strip() + "\n" + a).encode("utf-8"))
open(os.path.join(H, "msg.txt"), "wb").write(open(T + "msg90.txt", "rb").read())
print("hold ok")
