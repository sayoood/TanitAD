"""Hold: RECOVER-FT stack patch + packages recover_ft and framefreeze (H-RECOVER-FT-1)
trainer into stack/, packages banked as raw/2026-11-02-recover-ft/ and raw/2026-11-02-framefreeze/, one register row (append). F: scratch stays (NOT_LANDED.txt).
python mk_hold_recover_ft.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
H = "C:/Users/Admin/qland/hold_recover_ft/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
assert not os.path.exists(H), H
BANNED = (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn")
LIMIT = 3_000_000
M = "\u2212"


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def sha(b):
    return hashlib.sha256(b).hexdigest()


def put(rel, b):
    assert len(b) > 0 and len(rel) <= 220, (rel, len(rel))
    assert not re.search(rb"hf_[A-Za-z0-9]{8,}", b), rel
    if rel.endswith((".py", ".sh", ".md")):
        assert b"\r" not in b, rel
    assert len(b) < LIMIT or rel.startswith("mods"), rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


import json
D = "E:/Projects/TanitAD-artifacts/dagland/"
RF = D + "recover_ft/"
FZ = D + "framefreeze/"
PKR = V8 + "raw/2026-11-02-recover-ft/"
PKF = V8 + "raw/2026-11-02-framefreeze/"
TRAINER = "stack/scripts/v8f_mini_train.py"
n, tot, nl = 0, 0, 0
# 1. the stack patch: the trainer replaces the tip blob it was built on (0d708b3c); new files must not exist on the tip
base = blob(TRAINER)
assert base is not None and sha(base).startswith("0d708b3c"), "trainer base moved on the tip: rebuild the patch"
for rel in ("stack/tanitad/data/v8f_recover.py", "stack/tests/test_v8f_recover.py", "stack/tests/test_v8f_recover_wiring.py", TRAINER):
    b = open(RF + "stack_patch/" + rel, "rb").read()
    assert b"\r" not in b, rel
    if rel == TRAINER:
        put("mods-rw/" + rel, b)
    else:
        assert blob(rel) is None, rel
        put("stage/" + rel, b)
    n += 1
verdict = open(RF + "logs/REGRESSION_VERDICT.txt", "rb").read()
assert b"VERDICT: 0 regressions" in verdict
# 2. the two packages
for S, PK, pin_need, FS in ((RF, PKR, {"PREREG_RECOVER_FT_FROZEN.md": "57caba3b"}, "F:/TanitAD-scratch/recover_ft/"),
                            (FZ, PKF, {}, "F:/TanitAD-scratch/framefreeze/")):
    pins = {}
    for line in open(S + "MANIFEST.txt", encoding="utf-8"):
        m = re.match(r"\s*(\S.*?)\s*\|\s*\d+\s*\|\s*([0-9a-f]{64})\s*$", line)
        if m:
            pins[m.group(1).replace(S, "")] = m.group(2)
    for f, p in pin_need.items():
        assert pins.get(f, "").startswith(p), (S, f)
    notl = []
    for root, dirs, fs in os.walk(S):
        for f in fs:
            full = os.path.join(root, f).replace("\\", "/")
            rel = full[len(S):]
            b = open(full, "rb").read()
            if rel in pins and not rel.startswith("logs/") and rel != "FROZEN_SHA.txt":
                assert sha(b) == pins[rel], (S, rel)
            why = ("cache" if "__pycache__" in rel or ".pytest_cache" in rel else
                   "log over 512 kB (kept on E:)" if rel.startswith("logs/") and len(b) >= 512_000 else
                   "over 3 MB" if len(b) >= LIMIT else None)
            if why:
                notl.append(f"{rel} | {len(b)} | {sha(b)} | {why}")
                continue
            if not b:
                continue
            if rel.endswith((".py", ".sh", ".md")) and b"\r" in b:
                b = b.replace(b"\r\n", b"\n")
            assert blob(PK + rel) is None, (S, rel)
            put("stage/" + PK + rel, b)
            n += 1
            tot += len(b)
    notl.append(f"{FS} | - | - | scratch trees, regression logs and rollout outputs (regenerable)")
    put("stage/" + PK + "NOT_LANDED.txt", ("# NOT_LANDED\n# path | bytes | sha256 | why\n" + "\n".join(sorted(notl)) + "\n").encode("utf-8"))
    n += 1
    nl += len(notl)
fz = json.load(open(FZ + "raw/FZ_CONTROLS.json", encoding="utf-8"))
k4 = fz["kfz4"]
assert k4["PASS"] and fz["recover_ft_contrast_usable"] and not fz["failed_controls"]
assert f"{k4['share_closed']['s0']:.3f}" == "0.569" and f"{k4['share_closed']['s1']:.3f}" == "0.608"
fe = fz["freeze_effect_FZY_minus_Y"]["s0"]["ADE6"]
assert f"{fe['delta']:+.3f}" == "+0.010" and not fe["separated"]
amd = open(RF + "AMENDMENT_1.md", "rb").read()
assert sha(amd).startswith("05cff3f8")
put("stage/" + PKR + "mk_hold_recover_ft.py", open(__file__.replace("\\", "/"), "rb").read().replace(b"\r\n", b"\n"))
n += 1
ROW = ("**H-RECOVER-FT-1** (2026-11-02)",
       "A recovery fine-tune of O from DAG-OACC (RECOVER-FT: at p = 0.3 the ego-history input is deviated by the model family's own"
       " executed-minus-logged deviations from the DAG2 TRAIN own-state bank, and the target is the logged future re-expressed in the"
       " deviated frame with a 10-knot rejoin) closes v8F's M1 crawl excess at T1-kin: M1 R " + M + " C0 separated-better at both seeds,"
       " >= 0.5 of (C0 " + M + " P10)'s excess closed, above the C0rep replicate floor, ADE6 R " + M + " C0 <= +0.0604 m, no non-crawl"
       " row separated-worse, and ROBUST under the frame-freeze control (T1-kin's logged frames move at the human's pace).",
       "**PRE-REGISTERED 2026-11-02, BUILT; not yet trained.** Card frozen 19:40:09 (sha256 57caba3b…) before any code touched real"
       " data; AMENDMENT_1 (pre-data, 19:44:35, sha256 05cff3f8…): the rejoin continuation is the logged manoeuvre, so zero deviation"
       " gives the logged target exactly (the frozen straight-line continuation would have taught under-turning on curves). PI box-2"
       " go 2026-11-02 for six runs (R, C0, C0rep at s0 / s1, about 3.8 h), launch only after the trainer's self-check and the"
       " launch-gate PASS. Build (MEASURED): `stack/tanitad/data/v8f_recover.py` + trainer flags with FLAG_CHECKS, declared-vs-built"
       " and a gradient-reach probe (p = 0 reads exactly 0.0; every mutation RED); p = 0 is bit-identical to the tip trainer (276 / 276"
       " tensors); 0 regressions on the v8F trainer suites (10 failures that fail on the tip too). Frame-freeze controls on DAG-OACC"
       " (MEASURED, T1-kin, paired episode-cluster bootstrap B = 2,000): all PASS; the oracle level arm keeps its closure under the"
       " freeze (share 0.569 / 0.608), so the contrast is usable; the freeze's own effect on the base is ADE6 +0.010 m, not separated."
       " Real deviation quantiles of the TRAIN bank are UNVERIFIED until box 2's self-check.",
       "PRE-REGISTERED (built; training pending the launch gate)",
       "`" + PKR + "PREREG_RECOVER_FT_FROZEN.md`")
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
assert g.endswith(b"\n") and b"**H-RECOVER-FT-1**" not in g and b"\r" not in g
add = ("| " + " | ".join(ROW) + " |\n").encode("utf-8")
for w in BANNED:
    assert w not in add.lower(), w
assert add.count(b" | ") == 4 and g.rstrip(b"\n").split(b"\n")[-1].count(b" | ") == 4
put("mods/" + GC, g + add)
put("msg.txt", ("RECOVER-FT (H-RECOVER-FT-1): trainer lever for v8F's M1 crawl (recovery fine-tune of O from DAG-OACC with the DAG2 TRAIN"
                " own-state deviation bank and a re-expressed rejoin target) landed into stack/ with its card (frozen before data) and the"
                " pre-data AMENDMENT_1; p = 0 bit-identical to the tip trainer, 0 regressions on the v8F trainer suites. Frame-freeze"
                " T1-kin tool and its controls (all PASS) banked. Packages raw/2026-11-02-recover-ft/ and raw/2026-11-02-framefreeze/ ("
                + str(n) + " files incl. 4 stack files; NOT_LANDED lines " + str(nl) + "); register row. Training not started (box 2"
                " after the launch gate)." + chr(10) + chr(10) + "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" + chr(10)).encode("utf-8"))
print("hold written", H, n, "kB", tot // 1000, "not landed", nl)
