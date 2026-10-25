"""Build qland/hold_v8f_listen_prereg: PREREG_V8F_LISTEN (before any LISTEN data) and the register row H-V8F-LISTEN-1.
   python mk_hold48.py <remote tip sha>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
A2 = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/grid_a2/"
H = "C:/Users/Admin/qland/hold_v8f_listen_prereg"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"


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


for n in ("hold_v8f_listen_prereg", "landed_hold_v8f_listen_prereg"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
stage(A2 + "PREREG_V8F_LISTEN.md", "Project Steering/PREREG_V8F_LISTEN.md")
stage(A2 + "mk_hold48.py", f"{PK}/raw/2026-10-25-listen/mk_hold48.py")
lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-HGATE-1** (2026-10-25) |")]
assert len(ix) == 1
assert not any(x.startswith("| **H-V8F-LISTEN-1**") for x in lines)
row = ("| **H-V8F-LISTEN-1** (2026-10-25) | The three layers depend on different inputs: S's and T's largest causal dependence is the scene or nav,"
       " and O's is an input from above (T's code or S's checkpoints), not the scene. Measured as a grid of one-input donor swaps injected at ONE"
       " layer at a time (row metric at that layer's own output, normalised by swapping all of its inputs). Beside it, Q-NAV: does nav reach the"
       " trajectory through T or around it (ADE6 with donor nav at T only minus at O only) |"
       " **PRE-REGISTERED, NOT RUN (2026-10-25).** `Project Steering/PREREG_V8F_LISTEN.md`. Requested by the PI on 2026-10-25 (\"who listens to"
       " what\"). Eval-only, banked M2b s0 / s1, dev-box CPU, under `PI-V8F-2026-10-25b`. Controls: identity Δ 0.0, banked-eval reproduction,"
       " structural zeros on a mutated copy, donor-seed noise. Outcomes: DIVISION OF LABOUR SHOWN / PARTIAL / NOT SHOWN / SEEDS DISAGREE;"
       " Q-NAV: THROUGH T / AROUND T (⇒ a no-nav-at-O arm, PI go) / NOT ESTABLISHED. |")
lines[ix[0] + 1:ix[0] + 1] = [row]
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("PREREG_V8F_LISTEN: the causal who-listens-to-what grid for S, T and O (pre-registered before data)\n\n"
       "One-input donor swaps injected at one layer at a time, row metric at that layer's own output (S @50 m, T lat/lon F1, O ADE6),\n"
       "normalised by the all-inputs swap. Hypotheses from the attention result: S/T listen to scene or nav; O to T's code or S's checkpoints.\n"
       "Q-NAV separates nav -> T -> O from nav -> O. Eval-only, dev-box CPU, banked M2b s0/s1. Register: H-V8F-LISTEN-1.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
