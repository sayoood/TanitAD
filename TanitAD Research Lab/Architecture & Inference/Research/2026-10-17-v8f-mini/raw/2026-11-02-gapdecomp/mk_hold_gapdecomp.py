"""Hold: the gap split by window class (H-GAPDECOMP-1) banked as raw/2026-11-02-gapdecomp/, and its register row (append).
Scratch on F: stays there (NOT_LANDED.txt). python mk_hold_gapdecomp.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
H = "C:/Users/Admin/qland/hold_gapdecomp/"
S = "E:/Projects/TanitAD-artifacts/dagland/gapdecomp/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-11-02-gapdecomp/"
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


pins = {}
for line in open(S + "MANIFEST.txt", encoding="utf-8"):
    m = re.match(r"\s*(\S.*?)\s*\|\s*\d+\s*\|\s*([0-9a-f]{64})\s*$", line)
    if m:
        pins[m.group(1)] = m.group(2)
assert pins.get("PREREG_FROZEN.md", "").startswith("d5c9caa8") and pins.get("RESULTS.md", "").startswith("ea815b34"), "manifest"
txt = b"".join(open(S + f, "rb").read() for f in ("RESULTS.md", "raw/RESULTS_TABLES.md"))
n, notl = 0, []
for root, dirs, fs in os.walk(S):
    for f in fs:
        full = os.path.join(root, f).replace("\\", "/")
        rel = full[len(S):]
        b = open(full, "rb").read()
        if rel in pins:
            assert sha(b) == pins[rel], rel
        why = ("cache" if "__pycache__" in rel or ".pytest_cache" in rel else "over 3 MB" if len(b) >= LIMIT else None)
        if why:
            notl.append(f"{rel} | {len(b)} | {sha(b)} | {why}")
            continue
        if not b:
            continue
        if rel.endswith((".py", ".sh", ".md")) and b"\r" in b:
            b = b.replace(b"\r\n", b"\n")
        assert blob(PK + rel) is None, rel
        put("stage/" + PK + rel, b)
        n += 1
notl.append("F:/TanitAD-scratch/gapdecomp/ | - | - | scratch and a second copy of raw/arrays (regenerable by extract_arrays.py)")
put("stage/" + PK + "NOT_LANDED.txt", ("# NOT_LANDED\n# path | bytes | sha256 | why\n" + "\n".join(sorted(notl)) + "\n").encode("utf-8"))
put("stage/" + PK + "mk_hold_gapdecomp.py", open(__file__.replace("\\", "/"), "rb").read().replace(b"\r\n", b"\n"))
n += 2
for q in ("+0.771", "+0.660", "+0.747", "+0.768", "+0.123", "+0.107", "0.424", "+0.205", "+0.475", "+0.502", "+0.606", "+0.545", "+0.010",
          "0.561", "0.910", "0.463", "0.815", "0.004", "0.019"):
    assert q.encode() in txt, q
ROW = ("**H-GAPDECOMP-1** (2026-11-02)",
       "The Read-A gap of the v8F arms against P10 is TRIGGER-BOUND: the go/stay trigger windows (GT-launch + GT-stay, 359 of the"
       " 2,399 eval-141 windows) carry >= 0.7 of each crawl row's X " + M + " P10 gap with CI lower > 0.5 at both seeds for DAG-OACC;"
       " NOT TRIGGER-BOUND iff <= 0.3 with CI upper < 0.5; otherwise MIXED. Decides whether a perception-only trigger fix can clear D2.",
       "**MEASURED 2026-11-02, T1-kin, the SOUP read's per-window dumps, paired episode-cluster bootstrap B = 2,000 (another draw of"
       " episodes only); card frozen 14:07:12 before the run (sha256 d5c9caa8\u2026).** Known answers PASS: the classes partition the 2,399"
       " windows (GT-launch 327, GT-stay 32, other-stopped 8, moving 2,032); 114 / 114 arm-seed-rows reproduce the banked row and"
       " interval within 6e-5; (X, X) reads exactly 0. **No crawl row is TRIGGER-BOUND for DAG-OACC** (trigger share s0 / s1): ADE6"
       " +0.123 / +0.107 and yaw-rate " + M + "0.424 / +0.205 NOT; curvature +0.475 / +0.502, L2 +0.771 [+0.561, +0.910] / +0.660"
       " [+0.463, +0.815], M1 +0.606 / +0.545 MIXED. SOUP: L2 TRIGGER-BOUND at both seeds (+0.747 / +0.768). The trigger windows (15.0 %"
       " of windows) carry 3.0-6.4x their share of the curvature, L2 and M1 gaps (descriptive, not pre-registered). Beside (ESTIMATED):"
       " with the trigger windows' gap set to 0, yaw-rate, curvature, L2 and M1 stay separated-worse for DAG-OACC and SOUP at both"
       " seeds (DAG-OACC s0 L2 +0.010 [+0.004, +0.019]), so a trigger-only fix would not clear D2; the rest lives in the moving"
       " windows. Next (zero training): split the moving class by turning vs straight and speed bucket.",
       "MEASURED (NOT TRIGGER-BOUND for DAG-OACC: no crawl row; curvature, L2, M1 MIXED)",
       "`" + PK + "RESULTS.md`")
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
assert g.endswith(b"\n") and b"**H-GAPDECOMP-1**" not in g and b"\r" not in g
add = ("| " + " | ".join(ROW) + " |\n").encode("utf-8")
for w in BANNED:
    assert w not in add.lower(), w
assert add.count(b" | ") == 4 and g.rstrip(b"\n").split(b"\n")[-1].count(b" | ") == 4
put("mods/" + GC, g + add)
put("msg.txt", ("Gap split vs P10 by window class (H-GAPDECOMP-1): NOT TRIGGER-BOUND for DAG-OACC. The go/stay windows (15 % of eval-141) carry"
                " 3-6x their share of the curvature, L2 and M1 gaps, but with their gap set to 0 the four crawl rows stay separated-worse,"
                " so a trigger-only perception fix cannot clear D2; the rest lives in the moving windows. Known answers pass (114 / 114"
                " rows reproduce the SOUP read). Package raw/2026-11-02-gapdecomp/ (" + str(n) + " files); register row." + chr(10) + chr(10)
                + "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" + chr(10)).encode("utf-8"))
print("hold written", H, n, "not landed", len(notl))
