"""Hold: RESULT_V8F_LAUNCH_INTENT.md (HARMS, arm A not adopted; the M-LAUNCH slot filled: NOT MOVED), the M-LAUNCH package
(raw/2026-10-28-launch-intent-mlaunch/), the RESULT's checker and build package (raw/2026-10-28-launch-intent-result/), and an M-LAUNCH
prefix on register row H-V8F-LAUNCH-INTENT-1 (mods-rw; prefix only).  python mk_hold_li_result.py <live tip>"""
import hashlib
import json
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
H = "C:/Users/Admin/qland/hold_li_result/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
ML = "E:/Projects/TanitAD-artifacts/dagland/launch_intent_mlaunch/"
RS = "E:/Projects/TanitAD-artifacts/dagland/launch_intent_result/"
PML = V8 + "raw/2026-10-28-launch-intent-mlaunch/"
PRS = V8 + "raw/2026-10-28-launch-intent-result/"
assert not os.path.exists(H), H
BANNED = (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn")
MINUS = "\u2212"


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b, limit=400_000):
    assert len(b) > 0 and len(H + rel) < 260, (rel, len(H + rel))
    assert not re.search(rb"hf_[A-Za-z0-9]{8,}", b), rel
    if rel.endswith((".py", ".sh", ".md")):
        assert b"\r" not in b, rel
    assert len(b) < limit or rel.startswith("mods"), rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


def rd(p):
    return open(p, "rb").read()


def sha(b):
    return hashlib.sha256(b).hexdigest()


doc = rd(RS + "RESULT_V8F_LAUNCH_INTENT.md")
assert b"@@MLAUNCH@@" not in doc and b"**M-LAUNCH: NOT MOVED**" in doc
for w in BANNED:
    assert w not in doc.lower(), w
assert b"ALL CHECKS PASS" in rd(RS + "check_result_filled.out") and b"133 checks passed, 0 failed" in rd(RS + "check_result_filled.out")
mlb = rd(ML + "MLAUNCH.json")
assert sha(mlb) == "f5ebb4b209a5072e8b8171d4103d026a682cf2af84c14c891c1169a7e11aaaa2"
M = json.loads(mlb.decode("utf-8"))
assert M["M_LAUNCH"] == "not moved"
RP = V8 + "RESULT_V8F_LAUNCH_INTENT.md"
assert blob(RP) is None
put("stage/" + RP, doc)

n = 0
for base, pk, skip in ((ML, PML, ()), (RS, PRS, ("RESULT_V8F_LAUNCH_INTENT.md",))):
    for root, dirs, fs in os.walk(base):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in fs:
            full = os.path.join(root, f).replace("\\", "/")
            rel = full[len(base):]
            if rel in skip:
                continue
            b = rd(full)
            if not b:
                continue
            assert blob(pk + rel) is None, rel
            put("stage/" + pk + rel, b, limit=(700_000 if rel.endswith(".npz") else 400_000))
            n += 1
for f, src in (("checks/check_result_mutated_mlaunch.out", "F:/TanitAD-scratch/RESULT_LI_mut_mlaunch.out"),
               ("checks/check_result_original_53c4d527.py", "F:/TanitAD-scratch/check_result_53c4d527.py")):
    put("stage/" + PRS + f, rd(src))
    n += 1

C = M["primary"]["cells"]


def fm(c):
    s = lambda v: ("+" if v >= 0 else MINUS) + "%.4f" % abs(v)
    return "%s [%s, %s]" % (s(c["delta"]), s(c["lo"]), s(c["hi"]))


GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
lines = g.split(b"\n")
idx = [i for i, l in enumerate(lines) if l.startswith(b"| **H-V8F-LAUNCH-INTENT-1** (2026-10-28) |")]
assert len(idx) == 1
row = lines[idx[0]].decode("utf-8")
OLD = "**HARMS (MEASURED 2026-10-28, T1-kin, Amendment 1 in force;"
assert row.count(OLD) == 1
PRE = ("**M-LAUNCH NOT MOVED (MEASURED 2026-10-28, T1-kin, 327 GT-launch windows / 49 episodes, paired episode-cluster bootstrap, B = 2,000):"
       " X " + MINUS + " Y of the mean (v_own " + MINUS + " v_GT) at ticks 1\u20132 is main s0 " + fm(C["main_s0"]["x_minus_y"]) + ", s1 "
       + fm(C["main_s1"]["x_minus_y"]) + ", replicate s0 " + fm(C["replicate_s0"]["x_minus_y"]) + ", s1 " + fm(C["replicate_s1"]["x_minus_y"])
       + " m/s, none separated; section 11 routes to LAUNCH-INTENT-HI (W 4.17852, its own PI go). Beside: the training draw alone (REP "
       + MINUS + " DAG-OACC s0) is " + fm(M["beside_training_seed_floor"]["REP_minus_CTRL_s0"]["x_minus_y"]) + " m/s."
       " `RESULT_V8F_LAUNCH_INTENT.md`.** ")
for w in BANNED:
    assert w not in PRE.encode("utf-8").lower(), w
row = row.replace(OLD, PRE + OLD)
lines[idx[0]] = row.encode("utf-8")
ng = b"\n".join(lines)
assert ng.count(b"\n") == g.count(b"\n") and ng.replace(PRE.encode("utf-8"), b"") == g
put("mods-rw/" + GC, ng)
put("msg.txt", ("RESULT_V8F_LAUNCH_INTENT: HARMS (arm A not adopted), reported as registered; four families per seed and per pair;"
                " disclosures (Amendment 1, the RAM-guard stop, the pin-fill time). M-LAUNCH (the PREREG section 11 routing variable) built"
                " and read: NOT MOVED in all four cells (main s0 " + fm(C["main_s0"]["x_minus_y"]) + ", s1 " + fm(C["main_s1"]["x_minus_y"])
                + ", replicate s0 " + fm(C["replicate_s0"]["x_minus_y"]) + ", s1 " + fm(C["replicate_s1"]["x_minus_y"]) + " m/s), so section 11"
                " routes to LAUNCH-INTENT-HI with its own PI go; the registered next after HARMS is B, also with its own PI go. The"
                " RESULT's checker re-finds every number (133 checks, 0 failed; a planted wrong M-LAUNCH cell is caught). Packages"
                " raw/2026-10-28-launch-intent-mlaunch/ and raw/2026-10-28-launch-intent-result/ (%d files); register prefix.\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % n).encode("utf-8"))
print("hold written", H, n)
