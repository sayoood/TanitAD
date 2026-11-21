"""Hold: NAV_INPROGRESS (exploratory: why window 2158 turns left against the road; what the in-progress-turn class costs DAG at T1-kin)
+ one BACKLOG row (V8F-NAV-EXTENT).  python mk_hold_nav_inprogress.py <live tip>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/navturn/"
H = "C:/Users/Admin/qland/hold_nav_inprogress/"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
D = PK + "raw/2026-10-27-nav-inprogress/"
assert not os.path.exists(H), H


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


def lf(p):
    b = open(p, "rb").read().replace(b"\r\n", b"\n")
    assert b"\r" not in b
    return b


for f in ("NAV_INPROGRESS.md", "b1_inprogress_cost.py", "b1_inprogress_cost.json"):
    assert blob(D + f) is None
    put("stage/" + D + f, lf(S + f))
put("stage/" + D + "mk_hold_nav_inprogress.py", lf("E:/Projects/TanitAD-artifacts/dagland/mk_hold_nav_inprogress.py"))
assert blob(PK + "raw/2026-10-25-navturn/repro_eval141.npz") is not None

bl = blob("Project Steering/BACKLOG.md")
assert bl.endswith(b"\n") and b"V8F-NAV-EXTENT" not in bl
row = ("| V8F-NAV-EXTENT | **OPEN (0 GPU to build; box 2 to train), 2026-10-27, from the PI's window-2158 question**"
       " (`" + D + "NAV_INPROGRESS.md`, EXPLORATORY): the model's nav input is 5 numbers (one-hot, d_next / 100, valid), so a turn IN PROGRESS"
       " (d_next = 0) reaches it with no angle and no end point, although the v9 release carries both (`nav_d_end_m`, `nav_dyaw_next_deg`)."
       " At window 2158 the nav says TURN_L for a +20 deg turn ending about 8 m ahead; the following -40 deg bend (vmin 8.8 m/s) fails the"
       " builder's vmin <= 8 m/s gate and reads FOLLOW, and T / S / O plan a 6 s left arc (DAG ADE6 19.32 / 17.84 m, P10 13.42). Cost on all"
       " 2,399 windows at T1-kin: the in-progress class is 117 windows / 24 episodes (DAG 4.25 / 4.38 m, about 2x FOLLOW, still about 2 m"
       " better than P10); its stale subset is 25 windows and 0.059 m of DAG's mean ADE6. Candidate arm: add the remaining turn extent and"
       " the announced angle to the nav input (same supplier as d_next; no classifier output). Ceiling about 0.06-0.2 m of mean ADE6, so it"
       " ranks behind DAGGER2 (crawl) and DAG-OACC (longitudinal) for dominance. Not proposed: lifting the vmin gate (it would make the nav"
       " token the situation label) |\n")
put("mods/Project Steering/BACKLOG.md", bl + row.encode("utf-8"))
put("msg.txt", ("NAV_INPROGRESS (exploratory): why video v5's window 2158 turns left against the road. The nav says TURN_L in progress for a"
                " +20 deg turn ending ~8 m ahead, the model's 5-number nav input carries no extent or angle, and the following -40 deg bend"
                " fails the builder's vmin <= 8 m/s announce gate. The in-progress class is DAG's hardest (117 windows, ~2x FOLLOW error);"
                " the screenshot's stale subset is 25 windows / 0.059 m of mean ADE6. BACKLOG: V8F-NAV-EXTENT\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n").encode())
print("hold written", H)
