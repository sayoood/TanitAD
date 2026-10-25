"""Build qland/hold_v8f_kindx_result: RESULT_V8F_KINDX.md + raw/2026-10-25-kindx/ (READ, controls, 16 evals, logs, scripts) + register H-V8F-KINDX-1 status. python mk_hold66.py <tip>"""
import glob
import os
import subprocess
import sys

TIP = sys.argv[1]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
K = SC + "kindx/"
A2 = SC + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8f_kindx_result"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-25-kindx"
TEXT = (".md", ".json", ".py", ".log", ".txt", ".out", ".sh", ".ps1")


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def stage(src, rel):
    assert show(TIP, rel) is None, rel
    b = open(src, "rb").read()
    if src.endswith(TEXT):
        b = b.replace(b"\r\n", b"\n")
    put("stage", rel, b)


for n in ("hold_v8f_kindx_result", "landed_hold_v8f_kindx_result"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
stage(K + "KINDX_RESULT.md", f"{PK}/RESULT_V8F_KINDX.md")
for f in sorted(os.listdir(K)):
    if os.path.isfile(K + f) and f != "KINDX_RESULT.md":
        stage(K + f, f"{RAW}/{f}")
for d in ("eval", "logs"):
    for f in sorted(glob.glob(K + d + "/*")):
        if os.path.isfile(f):
            stage(f, f"{RAW}/{d}/{os.path.basename(f)}")
stage(A2 + "mk_hold66.py", f"{RAW}/mk_hold66.py")
put("stage", f"{RAW}/NOT_BANKED.txt", b"Not banked: code/ (79 MB), an export of tip eeb2043e (in git); b_kind/ (32 MB), copies of KIND's banked B s0 evals (raw/2026-10-25-kind/run/, in git).\n")

lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-KINDX-1** (2026-10-25) |")]
assert len(ix) == 1
old = "**PRE-REGISTERED, NOT RUN (2026-10-25).**"
assert lines[ix[0]].count(old) == 1
lines[ix[0]] = lines[ix[0]].replace(old, (
    "**CANDIDATE: SPEED (s0 screen, MEASURED 2026-10-25; T0; one seed, so no lever claim).** gain_speed(X) +0.716 [+0.623, +0.818] m against the"
    " 2F = 0.2415 bar (B s0 was +0.270); angle +0.095, tte +0.110, geom +0.108 (all separated, all below 2F); timegap, leadgap ns. Paired episode-cluster"
    " bootstrap, B = 2,000, 141 episodes (another draw of EPISODES). Beside: union +0.801 [+0.691, +0.922]; additivity A_X 1.28 (B −0.30): on X the kinds"
    " overlap instead of interfering; geom moved from harmful (−0.43 on B) to +0.108. Necessity addendum (after the outcome): only speed (+0.412) and angle"
    " (+0.066) are necessary on X. Controls: own / union / full-GT reproduce the banked box-2 evals to ≤ 2.2e-5 m; identity oracle exactly 0.0 on 2,399"
    " windows; the reader reproduces KIND's banked B s0 gains exactly. Committed next: train X at s1 on box 2 and re-read gain_speed (a carrier needs"
    " both seeds). `RESULT_V8F_KINDX.md`, `raw/2026-10-25-kindx/READ.json`. Was: **PRE-REGISTERED, NOT RUN (2026-10-25).**"))
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("RESULT_V8F_KINDX: CANDIDATE, speed (s0 screen on MIXO's checkpoint)\n\n"
       "gain_speed(X) +0.716 [+0.623, +0.818] m (bar 2F 0.2415; B had +0.270); every other kind <= 0.11 m. Controls exact.\n"
       "Committed next: train X at s1 and re-read gain_speed. Register: H-V8F-KINDX-1.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
