"""Hold: RECOVER-FT read result (H-RECOVER-FT-1, H-RECOVER-COMPOSITE-1) as raw/2026-11-03-recover-read-result/ and the next-arm card
(H-RECOVER-NEXT-1) as raw/2026-11-03-recover-next/, with three register rows (append). python mk_hold_recover_result.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
H = "C:/Users/Admin/qland/hold_recover_result/"
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
RR = D + "recover_read/"
OLDPK = V8 + "raw/2026-11-02-recover-read/"
PKR = V8 + "raw/2026-11-03-recover-read-result/"
NX = D + "recover_next/"
PKN = V8 + "raw/2026-11-03-recover-next/"
n, tot, nl = 0, 0, 0


def bank(S, PK, pick, scratch_line):
    global n, tot, nl
    notl = []
    for root, dirs, fs in os.walk(S):
        for f in fs:
            full = os.path.join(root, f).replace("\\", "/")
            rel = full[len(S):]
            b = open(full, "rb").read()
            if rel.endswith((".py", ".sh", ".md")) and b"\r" in b:
                b = b.replace(b"\r\n", b"\n")
            if not pick(rel, b):
                continue
            why = ("cache" if "__pycache__" in rel or ".pytest_cache" in rel else
                   "log over 512 kB (kept on E:)" if rel.startswith("logs/") and len(b) >= 512_000 else
                   "over 3 MB" if len(b) >= LIMIT else None)
            if why:
                notl.append(f"{rel} | {len(b)} | {sha(b)} | {why}")
                continue
            if not b:
                continue
            assert not re.search(rb"hf_[A-Za-z0-9]{8,}", b), rel
            assert blob(PK + rel) is None, rel
            put("stage/" + PK + rel, b)
            n += 1
            tot += len(b)
    notl.append(scratch_line)
    put("stage/" + PK + "NOT_LANDED.txt", ("# NOT_LANDED\n# path | bytes | sha256 | why\n" + "\n".join(sorted(notl)) + "\n").encode("utf-8"))
    n += 1
    nl += len(notl)


def changed_since_landed(rel, b):      # only the files the ef4b4462 package does not already hold byte for byte
    old = blob(OLDPK + rel)
    return old is None or old != b


# 1. the read's result package (new path; the pre-registered package at OLDPK is not touched)
for f in ("RESULTS.md", "raw/RECOVER_READ.json", "raw/RESULTS_TABLES.md", "READER_FIX_RREAD1.md", "READER_FIX_RREAD2.md", "raw/SWAPCTL.txt"):
    assert os.path.exists(RR + f), f
res = open(RR + "RESULTS.md", "rb").read()
for w in BANNED:
    assert w not in res.lower(), w
bank(RR, PKR, changed_since_landed, "C:/Users/Admin/rr_scratch/out/ and F:/TanitAD-scratch/recover_read/ | - | - | the 15 rollouts' ticks / dumps (28 MB each; regenerable by the chain)")
# 2. the next-arm package
assert open(NX + "FROZEN_SHA.txt", encoding="utf-8").read().split()[2].startswith("01046fe6")
bank(NX, PKN, lambda rel, b: True, "box 2 D:/tanitad/rnext/ | - | - | not shipped yet (needs the PI's box-2 go)")
put("stage/" + PKR + "mk_hold_recover_result.py", open(__file__.replace("\\", "/"), "rb").read().replace(b"\r\n", b"\n"))
n += 1
# 3. rows, every number formatted from the read JSON
j = json.load(open(RR + "raw/RECOVER_READ.json", encoding="utf-8"))
assert j["mode"] == "production" and not j.get("refused") and j["label"] == "HARMS"
sd = j["seeds"]


def ci(blk, s, row, nd=4):
    x = sd[s][blk][row]
    f = lambda v: (M if v < 0 else "+") + f"{abs(v):.{nd}f}"
    return f"{f(x['delta'])} [{f(x['lo'])}, {f(x['hi'])}]"


P, G, F, FZ, CO = j["primary"], j["guards"], j["floor"], j["frame_freeze"], j["composite"]
assert not P["PASS"] and not G["PASS"] and FZ["reading"] == "PARTIAL" and CO["status"] == "NOT DOMINANT"
assert CO["C0C_dominant_seeds"] == 0 and CO["attribution"] == "R-SPECIFIC" and CO["counts"] is False and CO["R_freeze_reading"] == "PARTIAL"
assert all(CO["by_seed"][s]["separated_worse_rows"] == ["M1 crawl share"] and CO["by_seed"][s]["D1"] and CO["by_seed"][s]["D3"] for s in ("s0", "s1"))
EST = " paired episode-cluster bootstrap B = 2,000, seed 0 (another draw of episodes only)"
wr = "; ".join(f"{s}: " + ", ".join(G["per_seed"][s]["non_crawl_separated_worse"]) for s in ("s0", "s1"))
ROWS = [
    ("**H-RECOVER-FT-1** (2026-11-03, read)",
     "Read of the pre-registered RECOVER-FT arm (the recovery fine-tune of O from DAG-OACC, p = 0.3, empirical TRAIN-bank deviation law, rejoin target) against its no-perturbation control C0 and the C0rep replicate floor, with the frame-freeze control.",
     "**MEASURED 2026-11-03, T1-kin, eval-141 causal lane, s0 / s1, after S," + EST + "; cards frozen before data (57caba3b…, AMENDMENT_1 05cff3f8…).** **Label HARMS.**"
     f" PRIMARY not met: M1 crawl share R {M} C0 {ci('R_minus_C0', 's0', 'M1 crawl share')} / {ci('R_minus_C0', 's1', 'M1 crawl share')}, separated-better at both seeds,"
     f" but share of (C0 {M} P10)'s excess closed {P['share_closed']['s0']:.3f} / {P['share_closed']['s1']:.3f} (bar 0.5). Guards fail: ADE6 R {M} C0 {ci('R_minus_C0', 's0', 'ADE6')} / {ci('R_minus_C0', 's1', 'ADE6')} m"
     f" (guard +0.0604); non-crawl rows separated-worse {wr}. Floor: C0rep {M} C0 M1 {ci('C0REP_minus_C0', 's0', 'M1 crawl share')} / {ci('C0REP_minus_C0', 's1', 'M1 crawl share')},"
     f" ADE6 {ci('C0REP_minus_C0', 's0', 'ADE6')} / {ci('C0REP_minus_C0', 's1', 'ADE6')} m (a same-flag replicate moves s0 more than the lever does). Frame-freeze PARTIAL"
     f" (share closed under the freeze {FZ['share_closed']['s0']:.3f} / {FZ['share_closed']['s1']:.3f}). Two instrument fixes before any number, disclosed (READER_FIX_RREAD1: eval-tree model module"
     " swapped to the trained tree's, 35 arrays byte-equal; READER_FIX_RREAD2: write root moved off a full F:). Next (card section 8, HARMS): the dose arm p 0.1 and the attribution arm"
     " target logged, queued with R-APR in H-RECOVER-NEXT-1.",
     "MEASURED (HARMS)", "`" + PKR + "RESULTS.md`"),
    ("**H-RECOVER-COMPOSITE-1** (2026-11-03, read)",
     "Read of the pre-registered composite RC_s = R_s + `--smooth sg72 --feasible kbox` against P10 (Read A at both seeds, counted only if R's frame-freeze reading is ROBUST or PARTIAL).",
     "**MEASURED 2026-11-03, T1-kin," + EST + "; card frozen 21:53:40 2026-11-02 (18ed37bc…).** **NOT DOMINANT.** At both seeds D1 holds (ADE6 RC " + M + " P10 "
     + ci("RC_minus_P10", "s0", "ADE6") + " / " + ci("RC_minus_P10", "s1", "ADE6") + " m) and D3 holds over the four families; D2 fails on M1 crawl share alone (RC "
     + M + " P10 " + ci("RC_minus_P10", "s0", "M1 crawl share") + " / " + ci("RC_minus_P10", "s1", "M1 crawl share") + "; C0C " + M + " P10 "
     + ci("C0C_minus_P10", "s0", "M1 crawl share") + " / " + ci("C0C_minus_P10", "s1", "M1 crawl share") + ", so R removes part of the excess). R's freeze reading is PARTIAL, so a DOMINANT"
     " would have counted. Attribution R-SPECIFIC (C0C dominant at 0 seeds). Next: H-RECOVER-NEXT-1.",
     "MEASURED (NOT DOMINANT; D2 fails on M1 alone)", "`" + PKR + "RESULTS.md`"),
    ("**H-RECOVER-NEXT-1** (2026-11-03)",
     "The a-priori heavier deviation law (R-APR: dv ~ U[" + M + "3, 0] m/s, fixed in RECOVER-FT section 3 before any data), composed with S + kbox, makes v8F DOMINANT against P10 at T1-kin"
     " at both seeds (counted only if R-APR's frame-freeze reading is ROBUST or PARTIAL); with the committed HARMS attribution arms RTL (target logged) and RP01 (p 0.1).",
     "**PRE-REGISTERED 2026-11-03; card frozen 05:30:52 (sha256 01046fe6…) before any of the six runs exists.** Disclosed post-data choice: R-APR is added and ordered first"
     " because D2 fails on M1 alone and D1 has about 0.7 m slack over its bar; its law and every reading rule are pre-specified. Reader extended only by `--r-arm` (known answer:"
     " `--r-arm R` reproduces RECOVER_READ.json). Box-2 chain RAPR_0, RAPR_1, RTL_0, RTL_1, RP01_0, RP01_1 (about 9.5 h), built and tested on the dev box, not shipped:"
     " needs the PI's box-2 go. ESTIMATED P(any composite DOMINANT, counted) about 0.12.",
     "PRE-REGISTERED (box-2 go pending)", "`" + PKN + "PREREG_RECOVER_NEXT_FROZEN.md`"),
]
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
assert g.endswith(b"\n") and b"\r" not in g and b"**H-RECOVER-NEXT-1**" not in g and b"**H-RECOVER-FT-1** (2026-11-03, read)" not in g
add = b""
for ROW in ROWS:
    a = ("| " + " | ".join(ROW) + " |\n").encode("utf-8")
    for w in BANNED:
        assert w not in a.lower(), w
    assert a.count(b" | ") == 4, ROW[0]
    add += a
assert g.rstrip(b"\n").split(b"\n")[-1].count(b" | ") == 4
put("mods/" + GC, g + add)
put("msg.txt", ("RECOVER-FT read (H-RECOVER-FT-1 HARMS; H-RECOVER-COMPOSITE-1 NOT DOMINANT, D2 fails on M1 crawl share alone at both seeds) and the next-arm card"
                " (H-RECOVER-NEXT-1, pre-registered: R-APR + the committed HARMS attribution arms). T1-kin, paired episode-cluster bootstrap B = 2,000. Packages"
                " raw/2026-11-03-recover-read-result/ and raw/2026-11-03-recover-next/ (" + str(n) + " files; NOT_LANDED lines " + str(nl) + "); three register rows."
                + chr(10) + chr(10) + "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" + chr(10)).encode("utf-8"))
print("hold written", H, n, "kB", tot // 1000, "not landed", nl)
