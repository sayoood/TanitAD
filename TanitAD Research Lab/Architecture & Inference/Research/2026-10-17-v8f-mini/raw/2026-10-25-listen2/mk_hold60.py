"""Build qland/hold_v8f_listen2_prereg: PREREG_V8F_LISTEN2 (before any data) + register row H-V8F-LISTEN2-1. python mk_hold60.py <tip>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
A2 = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/grid_a2/"
H = "C:/Users/Admin/qland/hold_v8f_listen2_prereg"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


for n in ("hold_v8f_listen2_prereg", "landed_hold_v8f_listen2_prereg"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
for src, rel in (("PREREG_V8F_LISTEN2.md", "Project Steering/PREREG_V8F_LISTEN2.md"), ("mk_hold60.py", f"{PK}/raw/2026-10-25-listen2/mk_hold60.py")):
    assert show(TIP, rel) is None, rel
    put("stage", rel, open(A2 + src, "rb").read().replace(b"\r\n", b"\n"))
lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-LISTEN-1** (2026-10-25) |")]
assert len(ix) == 1 and not any(x.startswith("| **H-V8F-LISTEN2-1**") for x in lines)
row = ("| **H-V8F-LISTEN2-1** (2026-10-25) | The route travels down the hierarchy: a donor nav injected at S ONLY (T and O keep their own nav) lowers"
       " T-lateral's macro-F1, separated at both seeds and above the donor-seed floor (E1). Beside it, E2 (the grid's per-row ordering survives"
       " training-mean replacement) and E3 (O hurts more from contradictory than from coherent foreign inputs) |"
       " **PRE-REGISTERED, NOT RUN (2026-10-25).** `Project Steering/PREREG_V8F_LISTEN2.md`. Zero training, dev-box CPU, the LISTEN harness."
       " Outcomes: ROUTE FLOWS DOWN / PARALLEL PATH (the claim narrows to S → O and nav → T) / SEEDS DISAGREE. |")
lines[ix[0] + 1:ix[0] + 1] = [row]
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("PREREG_V8F_LISTEN2: does the route travel down the hierarchy? (pre-registered before data)\n\n"
       "E1 (primary): donor nav at S only, read T-lateral F1. E2: the grid's ordering under training-mean replacement. E3: O's coherence.\n"
       "Register: H-V8F-LISTEN2-1.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
