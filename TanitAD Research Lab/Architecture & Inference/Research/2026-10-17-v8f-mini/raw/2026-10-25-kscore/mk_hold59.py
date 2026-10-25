"""Build qland/hold_v8f_kscore_instr: PREREG_V8F_KSCORE instruments (3 new files), the one-line guard wiring (v8f_kscore in kill_ours),
   the build record under raw/2026-10-25-kscore/, and a register note. python mk_hold59.py <remote tip sha>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
KS = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/kscore/"
A2 = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/grid_a2/"
H = "C:/Users/Admin/qland/hold_v8f_kscore_instr"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-25-kscore/"
GUARD = f"{PK}/code/box2_ops/v8f_guard.sh"


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def stage(src, rel):
    assert show(TIP, rel) is None, f"{rel} already exists on the tip"
    put("stage", rel, open(src, "rb").read().replace(b"\r\n", b"\n"))


for n in ("hold_v8f_kscore_instr", "landed_hold_v8f_kscore_instr"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
for rel in ("stack/scripts/v8f_kscore.py", "stack/scripts/chain_v8f_kscore.sh", "stack/tests/test_v8f_kscore.py"):
    stage(KS + "tree/" + rel, rel)
for f in ("FILES.txt", "HANDOFF.md", "WIRING.patch", "pytest_log.txt", "pytest_log_full.txt"):
    stage(KS + f, RAW + f)
stage(A2 + "mk_hold59.py", RAW + "mk_hold59.py")
g = show(TIP, GUARD).decode("utf-8")
a = "v8f_lora_smoke / v8f_lora_identity_dev) and their workers"
b = "'v8f_mini_train|v8f_mini_eval|v8f_lora_smoke|v8f_lora_identity_dev'"
assert g.count(a) == 1 and g.count(b) == 1 and "v8f_kscore" not in g
g = g.replace(a, "v8f_lora_smoke / v8f_lora_identity_dev / v8f_kscore) and their workers").replace(
    b, "'v8f_mini_train|v8f_mini_eval|v8f_lora_smoke|v8f_lora_identity_dev|v8f_kscore'")
put("mods-rw", GUARD, g.encode("utf-8"))
lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-KSCORE-1**")]
assert len(ix) == 1 and lines[ix[0]].rstrip().endswith("|")
lines[ix[0]] = lines[ix[0]].rstrip()[:-1].rstrip() + (" **Instruments landed 2026-10-25** (`stack/scripts/v8f_kscore.py`, `chain_v8f_kscore.sh`,"
    " `stack/tests/test_v8f_kscore.py`: 28 passed on CPU; mutations RED: B unfrozen, scene input zeroed, outcomes swapped, COSTS precedence"
    " swapped; the 256-state outcome map is complete; the guard now stops `v8f_kscore`). ⚠ Before launch the chain must take FLAT's"
    " predecessor classification (proceed on DONE or a verdict-level STOP; hold only on guard / thermal / reboot / unreachable): as built it"
    " holds on ANY predecessor STOP, which would idle box 2. Box-2 order: MIXO → FLAT → KSCORE → OACC → DSEFF. |")
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("PREREG_V8F_KSCORE instruments: scorer (hyps / fit with launch gate / eval / read), box-2 chain, 28 tests; guard stops v8f_kscore\n\n"
       "Built against 4a59a711, new files only plus the one-line guard wiring. Mutations RED (B unfrozen, scene zeroed, outcome swaps).\n"
       "Open before launch: the chain's predecessor STOP classification (FLAT's form), recorded in the register.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
