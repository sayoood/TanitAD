"""Build qland/hold_v8f_listen: PREREG_V8F_LISTEN's run banked (RESULT_V8F_LISTEN.md + raw/2026-10-25-listen/, without the code export
   and pytest temp) and the register row H-V8F-LISTEN-1 (NOT SHOWN; Q-NAV THROUGH T). python mk_hold53.py <remote tip sha>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
LI, A2 = S + "listen/", S + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8f_listen"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-25-listen/"
BIN = (".npz", ".npy", ".pt", ".png")
SKIP = ("code/", "pt_tmp/")


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


for n in ("hold_v8f_listen", "landed_hold_v8f_listen"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
n = 0
for root, dirs, files in os.walk(LI):
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), LI).replace("\\", "/")
        if rel.startswith(SKIP) or "__pycache__" in rel or ".pytest_cache" in rel:
            continue
        assert os.path.getsize(LI + rel) < 10_000_000, rel
        stage(LI + rel, RAW + rel); n += 1
stage(A2 + "mk_hold53.py", RAW + "mk_hold53.py")
stage(LI + "LISTEN_RESULT.md", f"{PK}/RESULT_V8F_LISTEN.md")
lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-LISTEN-1** (2026-10-25) |")]
assert len(ix) == 1
pre = "| **PRE-REGISTERED, NOT RUN (2026-10-25).** "
assert lines[ix[0]].count(pre) == 1
new = ("| **NOT SHOWN as hypothesised; Q-NAV: THROUGH T. 2026-10-25. MEASURED, T1-class open loop, DIAGNOSTIC, eval-141, paired episode-cluster"
       " bootstrap; `…/2026-10-17-v8f-mini/raw/2026-10-25-listen/READ.json`, `RESULT_V8F_LISTEN.md`, figure `fig_listen_grid.png`.** Controls passed:"
       " the clean pass reproduces the banked eval to ≤ 4.9e-5; identity cells exactly 0.0 (23 per seed); structural zeros exactly 0.0 on 4"
       " regression copies; O × T code matches the banked `--tac shuffle` to 1.4e-5 m. Largest cell per row (s0 / s1): S = nav +4.20 / +3.99 m;"
       " T-lat = S checkpoints +0.391 / +0.406 F1 (scene not separated at either seed); T-lon = ego +0.247 / +0.261 F1; O = ego token +5.70 /"
       " +4.72 m, then T code +5.40 / +4.13 m (O × scene +0.31 / +0.01 m, O × nav +0.015 / +0.011 m). So P-UPPER (T-lat on scene or nav) and"
       " P-LOWER (O on code or checkpoints) both fail at both seeds. Q-NAV: D = ADE6(nav at T only) − ADE6(nav at O only) = +0.461 [+0.254,"
       " +0.687] / +0.510 [+0.273, +0.793] m, separated; nav at S only +0.73 / +0.82 m. Post hoc, no bar: the dependence runs down the chain"
       " (nav → S → T-lat → O), and attention share is not dependence (T's 97–100 % scene attention, T-lat × scene ≈ 0). ⇒ Committed next per the"
       " table: the abstraction claim rests on the oracle and donor reads, stated per cell; three CPU follow-ups named in the result need their own"
       " PREREG. ~~" + pre[2:].strip() + "~~ ")
lines[ix[0]] = lines[ix[0]].replace(pre, new)
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("PREREG_V8F_LISTEN: NOT SHOWN as hypothesised; Q-NAV THROUGH T -- the who-listens-to-what grid banked with its figure\n\n"
       "MEASURED, T1-class open loop, DIAGNOSTIC, eval-141, paired episode-cluster bootstrap; controls passed (reproduction <= 4.9e-5,\n"
       "identity and structural zeros exactly 0.0). Largest cell per row: S = nav, T-lat = S checkpoints, T-lon = ego, O = ego then T code.\n"
       "Nav reaches the plan through S and T (D = +0.46 / +0.51 m, separated); O's own nav token is +0.01 m.\n"
       "Register: H-V8F-LISTEN-1 verdict.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok", n, "raw files")
