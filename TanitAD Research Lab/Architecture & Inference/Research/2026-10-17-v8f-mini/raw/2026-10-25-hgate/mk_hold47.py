"""Build qland/hold_v8f_hgate_fail: PREREG_V8F_HGATE's run banked (RESULT_V8F_HGATE.md + raw/2026-10-25-hgate/) and the register
   row H-V8F-HGATE-1 (FAIL, CONSISTENCY NOT RESTORED). python mk_hold47.py <remote tip sha>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
HG, A2 = S + "hgate/", S + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8f_hgate_fail"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-25-hgate/"
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


for n in ("hold_v8f_hgate_fail", "landed_hold_v8f_hgate_fail"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
n = 0
for root, dirs, files in os.walk(HG):
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), HG).replace("\\", "/")
        if "__pycache__" in rel or rel.startswith(".pytest_cache"):
            continue
        assert os.path.getsize(HG + rel) < 10_000_000, rel
        stage(HG + rel, RAW + rel); n += 1
stage(A2 + "mk_hold47.py", RAW + "mk_hold47.py")
stage(HG + "HGATE_RESULT.md", f"{PK}/RESULT_V8F_HGATE.md")
lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-HGATE-1** (2026-10-25) |")]
assert len(ix) == 1
pre = "| **PRE-REGISTERED, NOT RUN (2026-10-25).** "
assert lines[ix[0]].count(pre) == 1
new = ("| **FAIL as committed (CONSISTENCY NOT RESTORED), 2026-10-25. MEASURED, T1-class open loop, eval-141, paired episode-cluster bootstrap;"
       " `…/2026-10-17-v8f-mini/raw/2026-10-25-hgate/READ.json`, `RESULT_V8F_HGATE.md`.**"
       " Identity controls passed at both seeds (gate closed = B, Δ 0.0; gate open with banked w = H, Δ 0.0; fit on all windows = banked w, diff 0.0);"
       " the fit set is 5,541 of 9,582 training windows with STOP, TURN and LEAD inactive. PRIMARY-C 'any' HG − B = +0.0196 [+0.0075, +0.0338] /"
       " +0.0288 [+0.0179, +0.0404] (bar: upper bound ≤ +0.02; not moved). PRIMARY-K passes at both seeds: ADE6 −0.037 [−0.068, −0.009] /"
       " −0.027 [−0.052, −0.004], ADE2 −0.020 / −0.021, separated. HG removes 80 % / 62 % of H's consistency cost and keeps 18 % / 38 % of its"
       " ADE6 gain; its ADE2 is above P10's (+0.074 / +0.048, sep.). Post hoc, no bar: the predicted-v_hi trigger is the weak link (precision"
       " 0.06 / 0.14 on the SPEED violations that matter), and a perfect any-violation gate would pass, which supports the T-side attribution."
       " ⇒ Committed next: no further post-processing arm; the lever moves to T after MIXO and FLAT read. ~~" + pre[2:].strip() + "~~ ")
lines[ix[0]] = lines[ix[0]].replace(pre, new)
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("PREREG_V8F_HGATE: FAIL (CONSISTENCY NOT RESTORED); the post-processing line ends, the lever moves to T\n\n"
       "MEASURED, T1-class open loop, eval-141, paired episode-cluster bootstrap; identity controls passed (delta 0.0).\n"
       "'any' HG-B +0.0196 [+0.0075,+0.0338] / +0.0288 [+0.0179,+0.0404] (bar: upper bound <= +0.02). ADE6 -0.037/-0.027 and\n"
       "ADE2 -0.020/-0.021, separated at both seeds. The predicted-v_hi trigger is the weak link (precision 0.06/0.14).\n"
       "Register: H-V8F-HGATE-1 verdict.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok", n, "raw files")
