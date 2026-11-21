"""Build qland/hold_v8f_kindx_prereg: PREREG_V8F_KINDX (before any data) + register row H-V8F-KINDX-1 after H-V8F-MIXO-1. python mk_hold64.py <tip>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
A2 = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/grid_a2/"
H = "C:/Users/Admin/qland/hold_v8f_kindx_prereg"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


for n in ("hold_v8f_kindx_prereg", "landed_hold_v8f_kindx_prereg"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
for src, rel in (("PREREG_V8F_KINDX.md", "Project Steering/PREREG_V8F_KINDX.md"), ("mk_hold64.py", f"{PK}/raw/2026-10-25-kindx/mk_hold64.py")):
    assert show(TIP, rel) is None, rel
    put("stage", rel, open(A2 + src, "rb").read().replace(b"\r\n", b"\n"))
lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-MIXO-1** (2026-10-25) |")]
assert len(ix) == 1 and not any(x.startswith("| **H-V8F-KINDX-1**") for x in lines)
row = ("| **H-V8F-KINDX-1** (2026-10-25) | Once O consumes partial codes (MIXO s0: r(0.5) 0.644 against B's 0.365), some kind of T's code is worth at least 2F to O"
       " on its own: gain_k(X) = ADE6_X(own) − ADE6_X(GT in kind k only) ≥ 0.2415 m and separated at s0, on X = M2b-mixo_s0 (KIND's six kinds, T0) |"
       " **PRE-REGISTERED, NOT RUN (2026-10-25).** `Project Steering/PREREG_V8F_KINDX.md`. Zero training, dev-box CPU. CANDIDATE → train X at s1 on box 2"
       " and re-read (a carrier needs both seeds), then a T-side arm on that kind; NO CANDIDATE → single-kind T-side arms stay unrankable and the"
       " longitudinal line (KSCORE, then OACC) stays the top lever. |")
lines[ix[0] + 1:ix[0] + 1] = [row]
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("PREREG_V8F_KINDX: a seed-0 screen of KIND on MIXO's checkpoint (pre-registered before data)\n\n"
       "Which kind of T's code is worth at least 2F to an O that consumes partial codes? CANDIDATE -> train X at s1 and re-read;\n"
       "NO CANDIDATE -> the longitudinal line stays top. Register: H-V8F-KINDX-1.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
