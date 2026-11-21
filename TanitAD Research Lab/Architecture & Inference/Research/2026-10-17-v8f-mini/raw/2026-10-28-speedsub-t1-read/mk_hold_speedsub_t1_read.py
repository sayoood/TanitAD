"""Hold: the SPEEDSUB-T1 read banked (raw/2026-10-28-speedsub-t1-read/: the reader's READ / BESIDE JSONs, pins, manifest, smoke and
preflight records, the chain log) + register row H-V8F-SPEEDSUB-T1-1 updated to the measured outcome (mods-rw, line count unchanged).
The RESULT document follows separately.  python mk_hold_speedsub_t1_read.py <live tip>"""
import hashlib
import json
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
O = "F:/TanitAD-scratch/speedsub_t1/out/"
H = "C:/Users/Admin/qland/hold_speedsub_t1_read/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-28-speedsub-t1-read/"
assert not os.path.exists(H), H
BANNED = (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn")


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, (rel, len(H + rel))
    assert not re.search(rb"hf_[A-Za-z0-9]{8,}", b), rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


rd = lambda p: open(p, "rb").read()
sha = lambda b: hashlib.sha256(b).hexdigest()

R = json.loads(rd(O + "SPEEDSUB_T1_READ.json").decode("utf-8"))
assert R["OUTCOME"] == "NULL" and R["label"] == "NULL-ONE" and R["controls_PASS"] is True
assert R["hits"] == {"CTRL_s0": False, "CTRL_s1": False, "REP_s0": True, "REP_s1": False}
assert not any(g["fires"] for g in R["guards"].values())
d = R["d_ADE6_egoc_minus_identity"]
f = lambda c: "%+.4f [%+.4f, %+.4f]" % (c["delta"], c["lo"], c["hi"])
FILES = ["SPEEDSUB_T1_READ.json", "SPEEDSUB_T1_BESIDE.json", "pins.json", "pins.txt", "runs_manifest.json", "smoke_check_CTRL_s0.json",
         "preflight_CTRL_s0.json", "chain.log"]
for fn in FILES:
    b = rd(O + fn)
    assert blob(PK + fn) is None and len(b) < 400_000, fn
    put("stage/" + PK + fn, b)
put("stage/" + PK + "chain_speedsub_t1.out", rd("F:/TanitAD-scratch/speedsub_t1/logs/chain_speedsub_t1.out"))
put("stage/" + PK + "mk_hold_speedsub_t1_read.py", rd("E:/Projects/TanitAD-artifacts/dagland/speedsub_t1/mk_hold_speedsub_t1_read.py"))
RSHA = sha(rd(O + "SPEEDSUB_T1_READ.json"))[:8]

GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
lines = g.split(b"\n")
idx = [i for i, l in enumerate(lines) if l.startswith(b"| **H-V8F-SPEEDSUB-T1-1** (2026-10-28) |")]
assert len(idx) == 1
row = lines[idx[0]].decode("utf-8")
cells = row.split(" | ")
OLD_ST = "**PRE-REGISTERED, NOT RUN (2026-10-28).**"
st_i = [i for i, c in enumerate(cells) if c.startswith(OLD_ST)]
assert len(st_i) == 1 and cells[st_i[0] + 1] == "PRE-REGISTERED"
new_st = ("**NULL (label NULL-ONE; MEASURED 2026-10-28, T1-kin, Amendment 1 in force; every control PASS, no guard fires).** ADE6 `egoc \u2212"
          " identity`: CTRL s0 " + f(d["CTRL_s0"]) + ", CTRL s1 " + f(d["CTRL_s1"]) + ", REP s0 " + f(d["REP_s0"]) + " (a hit), REP s1 "
          + f(d["REP_s1"]) + " m; the bar is a hit at all four runs. The one hit is the slow-T draw (REP s0, whose T speed cells read"
          " 0.73\u20131.17 m/s slow at T0): there `egoc` also moves yaw-rate, heading, curvature and L2 separated-better and VIOL-ANY"
          " separated-worse (+0.0421). M1 crawl share moves at no run. Floor rule: NOT COLLAPSE (the REP s0 \u2212 CTRL s0 ADE6 gap 0.4025"
          " \u2192 0.2309 m, above half). Beside, decides nothing: the logged-lane read-out (a declared open channel) moves CTRL's ADE6 by"
          " +0.0008 / \u22120.0185 m, not separated. Paired episode-cluster bootstrap, B = 2,000, 141 episodes, per run. Committed next:"
          " section 8 (a), the history-aware causal read-out. `raw/2026-10-28-speedsub-t1-read/SPEEDSUB_T1_READ.json` (sha256 " + RSHA
          + "\u2026); RESULT to follow. Was: ")
cells[st_i[0]] = new_st + cells[st_i[0]]
cells[st_i[0] + 1] = "MEASURED (NULL)"
nrow = " | ".join(cells).encode("utf-8")
for w in BANNED:
    assert w not in new_st.encode("utf-8").lower(), w
assert nrow.count(b" | ") == lines[idx[0]].count(b" | ")
lines[idx[0]] = nrow
ng = b"\n".join(lines)
assert ng.count(b"\n") == g.count(b"\n")
put("mods-rw/" + GC, ng)
put("msg.txt", ("SPEEDSUB-T1 read: NULL (NULL-ONE), H-V8F-SPEEDSUB-T1-1. Every control PASS, no guard fires (Amendment 1's G-VIOL included)."
                " ADE6 egoc - identity at T1-kin: CTRL s0 " + f(d["CTRL_s0"]) + ", CTRL s1 " + f(d["CTRL_s1"]) + ", REP s0 " + f(d["REP_s0"])
                + " (the only hit, the slow-T draw), REP s1 " + f(d["REP_s1"]) + ". Floor rule NOT COLLAPSE. Committed next: the"
                " history-aware causal read-out (PREREG section 8 (a)). Read banked in raw/2026-10-28-speedsub-t1-read/; register row"
                " updated; RESULT to follow.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n").encode("utf-8"))
print("hold written", H)
