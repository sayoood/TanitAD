"""Hold: the LAUNCH-INTENT read banked (raw/2026-10-28-launch-intent-read/: the read of record under Amendment 1, the T1-kin chain's own refused
read, the manifest, the rollout summaries, both chains' logs, the box-2 run records), the T1-kin chain package (raw/2026-10-28-launch-intent-t1-chain/),
and register row H-V8F-LAUNCH-INTENT-1 updated to the measured outcome (mods-rw, line count unchanged). The RESULT follows separately.
python mk_hold_li_read.py <live tip>"""
import hashlib
import json
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
H = "C:/Users/Admin/qland/hold_li_read/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-28-launch-intent-read/"
PC = V8 + "raw/2026-10-28-launch-intent-t1-chain/"
O = "F:/TanitAD-scratch/launch_intent_t1_out/"
T1 = "E:/Projects/TanitAD-artifacts/launch_intent_t1/"
ART = "E:/Projects/TanitAD-artifacts/v8f-launch-intent/"
assert not os.path.exists(H), H
BANNED = (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn")
MINUS = "\u2212"


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, (rel, len(H + rel))
    assert not re.search(rb"hf_[A-Za-z0-9]{8,}", b), rel
    if rel.endswith((".py", ".sh", ".md")):
        assert b"\r" not in b, rel
    assert len(b) < 400_000 or rel.startswith("mods"), rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


def rd(p):
    return open(p, "rb").read()


def sha(b):
    return hashlib.sha256(b).hexdigest()


RB = rd(O + "read_A1/LAUNCH_INTENT_READ.json")
R = json.loads(RB.decode("utf-8"))
assert R["OUTCOME"]["OUTCOME"] == "HARMS" and R["OUTCOME"]["adopts"] is False
assert all(R["controls_PASS"].values()) and len(R["controls_PASS"]) == 206
assert R["provenance"]["reader_sha256"] == "fb3d2d7d945b54209b62d5f906d4dcea30925a24f0b807459b8cbf3d7f0747a9"
fl = R["OUTCOME"]["flags"]
assert fl["R-MOVE"] == {"s0": False, "s1": False} and fl["R-CRAWL"] == {"s0": False, "s1": False}
assert fl["guards_held_per_seed"]["s0"]["R-ADE"] is False and fl["guards_held_per_seed"]["s1"]["R-ADE"] is False
assert b"K-BUILD ['stack/tanitad/data/v8f_mini_data.py differs from aeb54468" in rd(T1 + "logs/launch_intent_read.out")

files = {"read_A1/LAUNCH_INTENT_READ.json": RB, "read_A1/read_A1.out": rd(O + "read_A1/read_A1.out"),
         "read_A1/run_read_A1.sh": rd(O + "read_A1/run_read_A1.sh"), "runs.json": rd(O + "runs.json"),
         "PINS_BOX2.txt": rd(O + "PINS_BOX2.txt"), "LAUNCH_COMMIT.txt": rd(O + "LAUNCH_COMMIT.txt"),
         "lit1_logs/launch_intent_read_REFUSED.out": rd(T1 + "logs/launch_intent_read.out"),
         "box2/chain_v8f_launchintent.out": rd("C:/Users/Admin/box2_ship/logs/chain_v8f_launchintent.out"),
         "box2/PINS.txt": rd(ART + "PINS.txt"), "box2/K-BUILD.txt": rd(ART + "K-BUILD.txt")}
for f in sorted(os.listdir(O)):
    if (f.startswith("t1v8f_") and f.endswith(".json")) or (f.startswith("done_") and f.endswith(".txt")):
        files["t1/" + f] = rd(O + f)
for f in sorted(os.listdir(O + "t0fresh")):
    if f.endswith(".json"):
        files["t0fresh/" + f] = rd(O + "t0fresh/" + f)
for f in sorted(os.listdir(T1 + "logs")):
    if (f.endswith(".out") or f.startswith("chain.out")) and f != "launch_intent_read.out":
        files["lit1_logs/" + f] = rd(T1 + "logs/" + f)
for run in ("M2b_s0_launchintent", "M2b_s1_launchintent", "M2b_s0_launchintentrep", "M2b_s1_launchintentrep"):
    for f in sorted(os.listdir(ART + run)):
        if f.endswith((".json", ".txt", ".log")):
            files[f"box2/{run}/{f}"] = rd(ART + run + "/" + f)
files["mk_hold_li_read.py"] = rd("E:/Projects/TanitAD-artifacts/dagland/launch_intent_read/mk_hold_li_read.py")
n = 0
for f, b in files.items():
    if not b:
        continue
    assert blob(PK + f) is None, f
    put("stage/" + PK + f, b)
    n += 1

# the T1-kin chain package (it lived only on E:)
m = 0
for root, dirs, fs in os.walk(T1):
    dirs[:] = [d for d in dirs if d not in ("logs", "__pycache__")]
    for f in fs:
        full = os.path.join(root, f).replace("\\", "/")
        b = rd(full)
        if not b or len(b) >= 400_000:
            continue
        put("stage/" + PC + full[len(T1):], b)
        m += 1
RSHA = sha(RB)[:8]


def cell(pair, s, row="ADE6"):
    c = R["read_a"][pair][s][row]
    return "%+.4f [%+.4f, %+.4f]" % (c["delta"], c["lo"], c["hi"])


l2 = R["read_a"]["Arep_REP"]["s0"]["L2 curvature MAE (GT>=5.2)"]["delta"]
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
lines = g.split(b"\n")
idx = [i for i, l in enumerate(lines) if l.startswith(b"| **H-V8F-LAUNCH-INTENT-1** (2026-10-28) |")]
assert len(idx) == 1
row = lines[idx[0]].decode("utf-8")
cells = row.split(" | ")
st_i = [i for i, c in enumerate(cells) if c.startswith("**Amendment 1 (A1-2026-10-28")]
assert len(st_i) == 1 and cells[st_i[0] + 1] == "PRE-REGISTERED"
new_st = ("**HARMS (MEASURED 2026-10-28, T1-kin, Amendment 1 in force; all 206 controls PASS; arm A not adopted).** No crawl row moved by"
          " its bar in both pairs at either seed (R-MOVE false at s0 and s1). The one crawl row clearing its bar in one pair is L2 curvature"
          " in the replicate pair at s0, " + "%+.4f" % l2 + " against a bar of 0.0136. R-ADE fails at both seeds, each in one pair: ADE6"
          " A_rep " + MINUS + " REP s0 " + cell("Arep_REP", "s0") + " m, A " + MINUS + " DAG-OACC s1 " + cell("A_CTRL", "s1") + " m. At each"
          " seed the other pair moves ADE6 the other way, also separated (A " + MINUS + " DAG-OACC s0 " + cell("A_CTRL", "s0") + ", A_rep "
          + MINUS + " REP s1 " + cell("Arep_REP", "s1") + "): the separated ADE6 cells flip sign between the two same-seed pairs, the"
          " pattern the pair design exists to expose (`H-ESTIM-SEED-1`). R-T0, R-KEEP, R-NOBASE, R-TLAT, R-TLON and R-NEW hold at both"
          " seeds. Paired episode-cluster bootstrap, B = 2,000, 141 episodes, per pair. The T1-kin chain's own read with the launch-time"
          " reader REFUSED on K-BUILD alone, as Amendment 1 states. Committed next (section 10): B (W 0.726175, 4 box-2 runs) with its own"
          " PI go; M-LAUNCH, the section 11 routing variable, is not built by the reader. Read"
          " `raw/2026-10-28-launch-intent-read/read_A1/LAUNCH_INTENT_READ.json` (sha256 " + RSHA + "\u2026); RESULT to follow. Was: ")
for w in BANNED:
    assert w not in new_st.encode("utf-8").lower(), w
cells[st_i[0]] = new_st + cells[st_i[0]]
cells[st_i[0] + 1] = "MEASURED (HARMS)"
nrow = " | ".join(cells).encode("utf-8")
assert nrow.count(b" | ") == lines[idx[0]].count(b" | ")
lines[idx[0]] = nrow
ng = b"\n".join(lines)
assert ng.count(b"\n") == g.count(b"\n")
put("mods-rw/" + GC, ng)
msg = ("LAUNCH-INTENT read: HARMS (arm A, not adopted), H-V8F-LAUNCH-INTENT-1. T1-kin, Amendment 1 in force, all 206 controls PASS."
       " No crawl row moved by its bar in both pairs at either seed. R-ADE fails at both seeds in one pair each: ADE6 A_rep - REP s0 "
       + cell("Arep_REP", "s0") + ", A - DAG-OACC s1 " + cell("A_CTRL", "s1") + " m; the other pair at each seed is separated the other way ("
       + cell("A_CTRL", "s0") + ", " + cell("Arep_REP", "s1") + "). The T1-kin chain's own read (launch-time reader) REFUSED on K-BUILD alone."
       " Committed next: B with its own PI go. Read banked in raw/2026-10-28-launch-intent-read/ (%d files); the T1-kin chain package in"
       " raw/2026-10-28-launch-intent-t1-chain/ (%d files); register row updated; RESULT to follow.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % (n, m))
put("msg.txt", msg.encode("utf-8"))
print("hold written", H, n, m, RSHA)
