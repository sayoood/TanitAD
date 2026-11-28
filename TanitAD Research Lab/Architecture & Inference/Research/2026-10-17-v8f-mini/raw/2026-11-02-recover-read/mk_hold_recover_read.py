"""Hold: recover_read + recover_ft_launch (H-RECOVER-COMPOSITE-1)
banked as raw/2026-11-02-recover-read/ and raw/2026-11-02-recover-ft-launch/, one register row (append). F: scratch stays (NOT_LANDED.txt).
python mk_hold_recover_read.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
H = "C:/Users/Admin/qland/hold_recover_read/"
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


D = "E:/Projects/TanitAD-artifacts/dagland/"
PKGS = (("recover_read", V8 + "raw/2026-11-02-recover-read/", {"PREREG_COMPOSITE_FROZEN.md": "18ed37bc"}, "F:/TanitAD-scratch/recover_read/"),
        ("recover_ft_launch", V8 + "raw/2026-11-02-recover-ft-launch/", {"HANDOFF.md": "a85f6cdc"}, "box 2 D:/tanitad/runs/v8f_mini_full/recft/"))
n, tot, nl = 0, 0, 0
for name, PK, need, FS in PKGS:
    S = D + name + "/"
    pins = {}
    for line in open(S + "MANIFEST.txt", encoding="utf-8"):
        m = re.match(r"\s*(\S.*?)\s*\|\s*\d+\s*\|\s*([0-9a-f]{64})\s*$", line)
        if m:
            pins[m.group(1).replace(S, "")] = m.group(2)
    for f, p in need.items():
        assert pins.get(f, "").startswith(p), (name, f)
    notl = []
    for root, dirs, fs in os.walk(S):
        for f in fs:
            full = os.path.join(root, f).replace("\\", "/")
            rel = full[len(S):]
            b = open(full, "rb").read()
            if rel in pins and not rel.startswith("logs/") and rel != "FROZEN_SHA.txt":
                assert sha(b) == pins[rel], (name, rel)
            why = ("cache" if "__pycache__" in rel or ".pytest_cache" in rel else
                   "log over 512 kB (kept on E:)" if rel.startswith("logs/") and len(b) >= 512_000 else
                   "over 3 MB (the shipped tip tar; regenerable by git archive of ad0bcb72)" if len(b) >= LIMIT else None)
            if why:
                notl.append(f"{rel} | {len(b)} | {sha(b)} | {why}")
                continue
            if not b:
                continue
            if rel.endswith((".py", ".sh", ".md")) and b"\r" in b:
                b = b.replace(b"\r\n", b"\n")
            assert not re.search(rb"hf_[A-Za-z0-9]{8,}", b), (name, rel)
            assert blob(PK + rel) is None, (name, rel)
            put("stage/" + PK + rel, b)
            n += 1
            tot += len(b)
    notl.append(f"{FS} | - | - | outputs of the running chain (pulled and banked after it ends)")
    put("stage/" + PK + "NOT_LANDED.txt", ("# NOT_LANDED\n# path | bytes | sha256 | why\n" + "\n".join(sorted(notl)) + "\n").encode("utf-8"))
    n += 1
    nl += len(notl)
put("stage/" + PKGS[0][1] + "mk_hold_recover_read.py", open(__file__.replace("\\", "/"), "rb").read().replace(b"\r\n", b"\n"))
n += 1
PKR = PKGS[0][1]
ROW = ("**H-RECOVER-COMPOSITE-1** (2026-11-02)",
       "The RECOVER-FT checkpoint composed with the SG(7, 2) smoother and the curvature-only clamp (RC_s = R_s + `--smooth sg72"
       " --feasible kbox`) makes v8F DOMINANT against P10 at T1-kin: Read A at BOTH seeds (D1 ADE6 separated-better with point <= "
       + M + "0.2415 m; D2 no row of 19 separated-worse; D3 a separated-better row in each of the four families, STRATEGIC included),"
       " counted only if R's frame-freeze reading (H-RECOVER-FT-1) is ROBUST or PARTIAL.",
       "**PRE-REGISTERED 2026-11-02; card frozen 21:53:40 (sha256 18ed37bc…) before any R checkpoint existed.** Basis (MEASURED,"
       " H-FEAS2-T1-1): DAG-OACC + S + kbox passes D1 and D3 against P10 at both seeds and fails D2 on M1 crawl share alone. ESTIMATED"
       " P(DOMINANT, counted) about 0.07. Reader `code/recover_read.py` (one JSON for the RECOVER-FT primary, the C0rep floor, Read A,"
       " the frame-freeze rule and the composite; the three-family D3 reported beside the four-family verdict): 27 tests pass, six"
       " mutations RED; known answers on existing runs (R := C0: every delta exactly 0.0; R := the HISTSWAP run: M1 share 0.9395, as"
       " H-HISTSWAP-1; RC := FEAS2's kbox runs: rows equal, D2 fails on M1 only). Dev-box chain of 15 T1-kin rollouts staged, to start"
       " after the box-2 chain (launched 21:42:35, six runs, both seed-0 self-checks PASS) and the checkpoint pull.",
       "PRE-REGISTERED",
       "`" + PKR + "PREREG_COMPOSITE_FROZEN.md`")
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
assert g.endswith(b"\n") and b"**H-RECOVER-COMPOSITE-1**" not in g and b"\r" not in g
add = ("| " + " | ".join(ROW) + " |\n").encode("utf-8")
for w in BANNED:
    assert w not in add.lower(), w
assert add.count(b" | ") == 4 and g.rstrip(b"\n").split(b"\n")[-1].count(b" | ") == 4
put("mods/" + GC, g + add)
put("msg.txt", ("RECOVER-FT read prep (H-RECOVER-COMPOSITE-1, pre-registered): composite card frozen before any R checkpoint (R + S + curvature-only"
                " clamp, DOMINANT iff Read A vs P10 at both seeds and frame-freeze ROBUST/PARTIAL); reader with known answers (27 tests, six"
                " mutations RED); dev-box chain staged; box-2 launch record (both seed-0 self-checks PASS, real TRAIN-bank deviation"
                " quantiles). Packages raw/2026-11-02-recover-read/ and raw/2026-11-02-recover-ft-launch/ (" + str(n) + " files; NOT_LANDED"
                " lines " + str(nl) + "); register row." + chr(10) + chr(10) + "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" + chr(10)).encode("utf-8"))
print("hold written", H, n, "kB", tot // 1000, "not landed", nl)
