"""Build qland/hold_v8f_hproj_fail: PREREG_V8F_HPROJ's run banked (RESULT_V8F_HPROJ.md + raw/2026-10-25-hproj/), its committed next
   PREREG_V8F_HGATE (before any HGATE data), and the register: H-V8F-HPROJ-1 (FAIL, CONSISTENCY NOT RESTORED), H-V8F-HGATE-1 (new).
   python mk_hold46.py <remote tip sha>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
HP, A2 = S + "hproj/", S + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8f_hproj_fail"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-25-hproj/"
BIN = (".npz", ".npy", ".pt")


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    assert len(rel) <= 220, rel
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def stage(src, rel):
    assert show(TIP, rel) is None, f"{rel} already exists on the tip"
    b = open(src, "rb").read()
    put("stage", rel, b if rel.endswith(BIN) else b.replace(b"\r\n", b"\n"))


for n in ("hold_v8f_hproj_fail", "landed_hold_v8f_hproj_fail"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
n = 0
for root, dirs, files in os.walk(HP):
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), HP).replace("\\", "/")
        if "__pycache__" in rel:
            continue
        assert os.path.getsize(HP + rel) < 10_000_000, rel
        stage(HP + rel, RAW + rel); n += 1
stage(A2 + "mk_hold46.py", RAW + "mk_hold46.py")
stage(HP + "HPROJ_RESULT.md", f"{PK}/RESULT_V8F_HPROJ.md")
stage(A2 + "PREREG_V8F_HGATE.md", "Project Steering/PREREG_V8F_HGATE.md")
lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-HPROJ-1** (2026-10-25) |")]
assert len(ix) == 1
pre = "| **PRE-REGISTERED, NOT RUN (2026-10-25).** "
assert lines[ix[0]].count(pre) == 1
new = ("| **FAIL as committed (CONSISTENCY NOT RESTORED), 2026-10-25. MEASURED, T1-class open loop, eval-141, paired episode-cluster bootstrap;"
       " `…/2026-10-17-v8f-mini/raw/2026-10-25-hproj/READ.json`, `RESULT_V8F_HPROJ.md`.**"
       " Reconstruction controls passed: traj60 recovered to ≤ 2.1e-5 m; the banked `projected_r1_kamm` ADE6 reproduced to 2.3e-5 m; the CPU model's"
       " 0.1 s midpoints matched to 2.8e-5 m. PRIMARY-C 'any' HP − B = +0.0538 [+0.037, +0.072] / +0.0313 [+0.015, +0.049] (bar ≤ +0.02)."
       " PRIMARY-K ADE6 HP − B = +0.007 [−0.123, +0.155] / +0.127 [+0.011, +0.271]: the gain is lost."
       " Why (MEASURED, post hoc, no bar): the SPEED rule caps at min(posted limit, GT v_hi), and v_hi binds in 2,084 of 2,399 windows, which a"
       " posted-limit projection cannot see. LEAD is next. The GT plan itself breaks the posted limit in 6.3 % of windows. The PREREG's"
       " stop/lead/turn explanation was incomplete. ⇒ Committed next: `PREREG_V8F_HGATE`. ~~" + pre[2:].strip() + "~~ ")
lines[ix[0]] = lines[ix[0]].replace(pre, new)
assert not any(x.startswith("| **H-V8F-HGATE-1**") for x in lines)
hg = ("| **H-V8F-HGATE-1** (2026-10-25) | The constraint-aware form of HBLEND keeps its 2 s and 6 s gains without its consistency cost."
      " The weights are re-fitted on training windows whose GT STOP / TURN / LEAD rules are inactive. At inference H is used only where T itself"
      " predicts no FOLLOW, no STOP goal ≥ 0.5, no TURN goal ≥ 0.5 and no exceedance of its own v_hi at t ≥ 2 s; elsewhere B's plan is used."
      " Bars: CONSISTENCY(a) 'any' HG − B upper bound ≤ +0.02 AND ADE6 / ADE2 HG − B < 0 separated, at BOTH seeds |"
      " **PRE-REGISTERED, NOT RUN (2026-10-25).** `Project Steering/PREREG_V8F_HGATE.md`. HPROJ's committed next, with \"in the projection\""
      " made a GATE because a lead constraint cannot be projected at inference. Zero training, dev-box CPU, under `PI-V8F-2026-10-25b`."
      " A FAIL ends the post-processing line: the lever moves to T. |")
lines[ix[0] + 1:ix[0] + 1] = [hg]
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("PREREG_V8F_HPROJ: FAIL (CONSISTENCY NOT RESTORED) -> PREREG_V8F_HGATE pre-registered (T-gated, constraint-aware HBLEND)\n\n"
       "MEASURED, T1-class open loop, eval-141, paired episode-cluster bootstrap; reconstruction controls passed (<= 2.8e-5 m).\n"
       "'any' HP-B +0.054/+0.031 (bar <= +0.02 upper bound); ADE6 +0.007 (ns) / +0.127 (sep. worse): the posted-limit projection cannot see\n"
       "the GT v_hi cap that binds in 2,084/2,399 windows. HGATE (committed next): weights re-fitted on GT-unconstrained training windows,\n"
       "H used only where T predicts no FOLLOW / STOP / TURN and no exceedance of its own v_hi; else B. Same bars. A FAIL ends the\n"
       "post-processing line. Register: H-V8F-HPROJ-1 verdict, H-V8F-HGATE-1.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok", n, "raw files")
