"""Build qland/hold_v8fcz: Amendment 2's result + PREREG_V8F_CZ's instruments (before any CZ run).
   python mk_hold9.py <remote tip sha> <n RED> <n mutations>"""
import glob
import os
import re
import subprocess
import sys

TIP, NRED, NMUT = sys.argv[1], sys.argv[2], sys.argv[3]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/tac/"
T3, BG = S + "t3/", S + "base_cz/"
H = "C:/Users/Admin/qland/hold_v8fcz"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-21-tacgrp2/"
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


def rw(rel, new, base=None, crlf=False):
    old = show(TIP, rel)
    if base is not None:
        assert old is not None and old == open(base, "rb").read(), f"{rel}: the tip moved since the edit was made"
    new = new if crlf else lf(new)
    assert old is not None and new != old, rel
    put("mods-rw", rel, new)


for n in ("hold_v8fcz", "landed_hold_v8fcz", "aborted_hold_v8fcz"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n

# ---- instruments: code edits against the snapshot bases (asserted equal to the tip)
for rel in ("stack/tanitad/models/v8f_mini.py", "stack/scripts/v8f_mini_eval.py", "stack/scripts/v8f_mini_train.py"):
    rw(rel, open(T3 + rel, "rb").read(), base=BG + os.path.basename(rel))
q = f"{PK}/code/v8f_mini_box2_queue.cmd"
rw(q, open(T3 + q, "rb").read(), base=BG + "v8f_mini_box2_queue.cmd", crlf=True)
stage(T3 + "stack/tests/test_v8f_cz.py", "stack/tests/test_v8f_cz.py")
for f in ("v8f_cz_gate_read.py", "v8f_cz_read.py", "box2_ops/chain_v8f_cz.sh"):
    stage(T3 + f"{PK}/code/{f}", f"{PK}/code/{f}")
stage(S + "PREREG_V8F_CZ.md", "Project Steering/PREREG_V8F_CZ.md")

# ---- Amendment 2's result + raw
stage(S + "RESULT_V8F_TAC_A2.md", f"{PK}/RESULT_V8F_TAC_A2.md")
stage(LG + "v8f_tacgrp2/tac_groups2_read.json", RAW + "tac_groups2_read.json")
stage(LG + "v8f_tacgrp2/telemetry_v8ftacgrp2.csv", RAW + "telemetry_v8ftacgrp2.csv")
for f in ("chain_v8f_tacgrp2.out", "v8ftacgrp2_guard.out"):
    stage(LG + f, RAW + "box2_logs/" + f)
stage(LG + "v8f_tacgrp2/v8ftacgrp2_run.log", RAW + "box2_logs/v8ftacgrp2_run.log")
n = 0
pat = re.compile(r"-tacorc-(lonspeed\+londec|all-[a-z]+|constraints-a\d\d)\.(npz|json)$")
for p in sorted(glob.glob(LG + "v8f_tacgrp2/eval/eval_*")):
    if pat.search(os.path.basename(p)):
        stage(p, RAW + "eval/" + os.path.basename(p), binary=p.endswith(".npz"))
        n += 1
assert n == 80, f"{n} Amendment 2 eval files, expected 40 npz + 40 json"
for f in ("absent_slot_error.py", "absent_slot_error.out", "mutate_cz.py", "mutate_cz.out", "mk_hold9.py"):
    stage(S + f, RAW + f)
stage(S + "raw_readme_tacgrp2.md", RAW + "README.md")

# ---- the A1 erratum (R61)
a1 = f"{PK}/RESULT_V8F_TAC_A1.md"
old = show(TIP, a1).decode("utf-8")
line = "* **STRATEGIC:** PAUSED (PI ruling 2026-09-19, `PI_DECISION_QUEUE.md` item 25).\n"
assert old.count(line) == 1
new = old.replace(line, "* **STRATEGIC:** ~~PAUSED (PI ruling 2026-09-19, `PI_DECISION_QUEUE.md` item 25).~~ ⛔ **ERRATUM 2026-10-21 (R61):** that is the "
                        "programme-wide pause; v8F-mini's strategic family is **UN-PAUSED** (PI ruling 2026-10-17, "
                        "quoted in `v8f_mini_full_verdict.py`). In these arms it is unchanged by construction: S is upstream of every intervention.\n")
rw(a1, new.encode("utf-8"))

# ---- registers
rl = show(TIP, "Project Steering/RETRACTION_LOG.md")
assert b"### R61 " not in rl and b"### R60 " in rl
put("mods", "Project Steering/RETRACTION_LOG.md", rl.rstrip(b"\n") + b"\n" + lf(open(S + "r61.md", "rb").read()))

reg = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8")
lines = reg.split("\n")
i3 = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-TAC-3** (2026-10-21) |")]
assert len(i3) == 1 and "H-V8F-CZ-1" not in reg and "PRE-REGISTERED, NOT RUN" in lines[i3[0]]
row3 = ("| **H-V8F-TAC-3** (2026-10-21) | Inside the constraints, which group is NECESSARY, and does PARTIAL accuracy pay? (A) `lonspeed+londec`; "
        "(B) leave-one-out ×6; (C) the constraints moved α = 0.25 / 0.5 / 0.75 of the way to the GT (`PREREG_V8F_TAC` Amendment 2, written "
        "before its data) | **CONSTRAINT-ACCURACY (as committed), 2026-10-21. MEASURED, T0, paired episode-cluster bootstrap, 141 episodes; "
        "`…/2026-10-17-v8f-mini/raw/2026-10-21-tacgrp2/tac_groups2_read.json`.** (A) carries 0.24–0.30 (does not carry); NECESSARY at M2b: "
        "latcons (0.55/0.64), lonspeed (1.65/1.66), stoplead (0.35/0.37); the decision groups ≤ 0.10. Dose INTERMEDIATE and convex: r(0.25 / "
        "0.5 / 0.75) = 0.16 / 0.36 / 0.73 (s0), 0.08 / 0.21 / 0.48 (s1). ⭐ The saved codes then showed 92–95 % of T's squared latcons / "
        "stoplead error (52–54 % lonspeed) sits in ABSENT slots: T's loss masks them while O is teacher-forced on 0 there ⇒ "
        "H-V8F-CZ-1. `RESULT_V8F_TAC_A2.md`. |")
rowc = ("| **H-V8F-CZ-1** (2026-10-21) | Training T to write 0 in ABSENT constraint slots (the value O is teacher-forced on) makes v8F-mini "
        "drive better: arm `M2b-cz` = M2b + `cons_absent zero`, ONE lever moved, 2 seeds (`PREREG_V8F_CZ`, written before its data) | "
        "**PRE-REGISTERED, NOT RUN (box 2, ~9 h incl. training; awaits the PI's go).** Stage 1 (eval only, T0): the absent-slot oracle "
        "must buy > 2F at both M2b seeds, else no training. Stage 2 (T1-class open loop): SUCCESS iff M2b-cz − M2b ADE6 < 0, separated, "
        "> 2F at both seeds; the mechanism (absmask code shift ≤ half of M2b's) reads a FAIL. Launch gate: G-DVB entry for the new lever, "
        f"G-LIVE; 4,512 clips, the RA-1 holdout, M2b's 19,164 steps asserted. Instruments: an exactly-zero gradient control; {NRED} / {NMUT} "
        "mutations RED. |")
lines[i3[0]] = rowc + "\n" + row3
rw("Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
open(os.path.join(H, "msg.txt"), "wb").write(lf(open(S + "msg9.txt", "rb").read()).replace(b"<NRED>", NRED.encode()).replace(b"<NMUT>", NMUT.encode()))
print("hold ok")
