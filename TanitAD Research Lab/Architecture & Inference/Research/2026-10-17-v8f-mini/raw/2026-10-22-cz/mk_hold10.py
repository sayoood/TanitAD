"""Build qland/hold_v8fcz_result: PREREG_V8F_CZ stage 1 (GATE FAIL, no training) banked, R63, errata, H-V8F-CZ-1.
   python mk_hold10.py <remote tip sha>"""
import glob
import os
import subprocess
import sys

TIP = sys.argv[1]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/tac/"
H = "C:/Users/Admin/qland/hold_v8fcz_result"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-22-cz/"
LG = "C:/Users/Admin/box2_ship/logs/"


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def lf(b):
    return b.replace(b"\r\n", b"\n")


def stage(src, rel, binary=False):
    assert show(TIP, rel) is None, f"{rel} already exists on the tip"
    b = open(src, "rb").read()
    put("stage", rel, b if binary else lf(b))


def rw(rel, new):
    old = show(TIP, rel)
    assert old is not None and lf(new) != old, rel
    put("mods-rw", rel, lf(new))


for n in ("hold_v8fcz_result", "landed_hold_v8fcz_result"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n

# ---- the result + raw
stage(S + "RESULT_V8F_CZ.md", f"{PK}/RESULT_V8F_CZ.md")
stage(LG + "v8f_cz/cz_gate.json", RAW + "cz_gate.json")
stage(LG + "v8f_cz/telemetry_v8fcz.csv", RAW + "telemetry_v8fcz.csv")
for f in ("chain_v8f_cz.out", "v8fcz_guard.out", "v8fcz10220702_refresh.log", "v8fcz10220702_tacabs.log"):
    stage(LG + f, RAW + "box2_logs/" + f)
n = 0
for p in sorted(glob.glob(LG + "v8f_cz/eval/eval_*-tacabs.*")):
    stage(p, RAW + "eval/" + os.path.basename(p), binary=p.endswith(".npz"))
    n += 1
assert n == 8, f"{n} tacabs files, expected 4 npz + 4 json"
for f in ("cons_curve.py", "cons_curve.out", "mk_hold10.py"):
    stage(S + f, RAW + f)
stage(S + "raw_readme_cz.md", RAW + "README.md")

# ---- errata (append-only inside whole-file edits: every old line is kept)
a2 = f"{PK}/RESULT_V8F_TAC_A2.md"
old = show(TIP, a2).decode("utf-8")
anchor = "  * **stage 2:** arm `M2b-cz` trains T to write 0 in absent slots, one lever moved vs M2b, 2 seeds, under the launch gate.\n"
assert old.count(anchor) == 1
err = ("* ⛔ **ERRATUM 2026-10-22 (R63):** the reading \"MISMATCH\" is refuted by `PREREG_V8F_CZ`'s own gate. Zeroing the absent slots made "
       "driving WORSE: ADE6 +1.042 / +1.020 m at M2b, separated (`RESULT_V8F_CZ.md`). O reads T's own code for most of its training, so "
       "T's absent-slot values are O's input, not error. The arithmetic in the table stands; the word \"error\" does not.\n")
rw(a2, old.replace(anchor, anchor + err).encode("utf-8"))
pz = "Project Steering/PREREG_V8F_CZ.md"
old = show(TIP, pz).decode("utf-8")
rw(pz, (old.rstrip("\n") + "\n\n## Outcome (2026-10-22, appended after the data)\n\n"
        "* **Stage 1: GATE FAIL, as committed. No training ran.** absmask − pred ADE6 **+1.042 / +1.020 m** at M2b (2F 0.2415), "
        "+1.107 / +0.878 at M1, all separated (`…/2026-10-17-v8f-mini/raw/2026-10-22-cz/cz_gate.json`, `RESULT_V8F_CZ.md`).\n"
        "* ⛔ **The framing above, \"a train/inference mismatch, not a perception limit\", is refuted (R63).** O co-adapted to T's "
        "absent-slot values, so they are its input.\n"
        "* The committed FAIL branch's next lever is T's PRESENT-slot accuracy, under its own PREREG.\n").encode("utf-8"))

# ---- registers
rl = show(TIP, "Project Steering/RETRACTION_LOG.md")
assert b"### R63 " not in rl and b"### R62 " in rl
put("mods", "Project Steering/RETRACTION_LOG.md", rl.rstrip(b"\n") + b"\n\n" + lf(open(S + "r63.md", "rb").read()))
reg = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8")
lines = reg.split("\n")
ic = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-CZ-1** (2026-10-21) |")]
assert len(ic) == 1
pre = "| **PRE-REGISTERED, NOT RUN (box 2, ~9 h incl. training; awaits the PI's go).** "
assert lines[ic[0]].count(pre) == 1
lines[ic[0]] = lines[ic[0]].replace(pre, (
    "| **GATE FAIL (as committed), 2026-10-22; no training ran. MEASURED, T0, paired episode-cluster bootstrap, 141 episodes; "
    "`…/2026-10-17-v8f-mini/raw/2026-10-22-cz/cz_gate.json`.** absmask − pred ADE6 **+1.042 / +1.020 m** at M2b (2F 0.2415), "
    "+1.107 / +0.878 at M1, all separated: zeroing the absent slots makes driving WORSE. O co-adapted to T's absent-slot values "
    "(it reads T's own code for most of training), so the \"mismatch\" reading is refuted (R63). Next: T's PRESENT-slot accuracy, "
    "under its own PREREG. `RESULT_V8F_CZ.md`. ~~PRE-REGISTERED, NOT RUN (box 2, ~9 h incl. training; awaits the PI's go).~~ "))
rw("Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
open(os.path.join(H, "msg.txt"), "wb").write(lf(open(S + "msg10.txt", "rb").read()))
print("hold ok")
