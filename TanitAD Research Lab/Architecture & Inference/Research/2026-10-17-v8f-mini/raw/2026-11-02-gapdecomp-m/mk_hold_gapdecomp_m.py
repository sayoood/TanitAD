"""Hold: the moving-class split (H-GAPDECOMP-M-1) banked as raw/2026-11-02-gapdecomp-m/, and its register row (append).
Scratch on F: stays there (NOT_LANDED.txt). python mk_hold_gapdecomp.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
H = "C:/Users/Admin/qland/hold_gapdecomp_m/"
S = "E:/Projects/TanitAD-artifacts/dagland/gapdecomp_m/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-11-02-gapdecomp-m/"
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
assert pins.get("PREREG_FROZEN.md", "").startswith("1c5dbaf0") and pins.get("RESULTS.md", "").startswith("eb2e2820"), "manifest"
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
notl.append("F:/TanitAD-scratch/gapdecomp_m/ | - | - | scratch; inputs are the landed raw/2026-11-02-gapdecomp/ arrays (sha-pinned)")
put("stage/" + PK + "NOT_LANDED.txt", ("# NOT_LANDED\n# path | bytes | sha256 | why\n" + "\n".join(sorted(notl)) + "\n").encode("utf-8"))
put("stage/" + PK + "mk_hold_gapdecomp_m.py", open(__file__.replace("\\", "/"), "rb").read().replace(b"\r\n", b"\n"))
n += 2
for q in ("+0.653", "+0.684", "+0.934", "+0.915", "+0.792", "+0.751", "+0.473", "+0.549", "+0.100", "+0.084", "0.074", "413", "0.0135", "+0.398", "+0.241", "+1.599", "0.1845", "0.2621", "0.0623", "9.0", "11.4"):
    assert q.encode() in txt, q
ROW = ("**H-GAPDECOMP-M-1** (2026-11-02)",
       "Within the moving windows (class M, 2,032 of the 2,399 eval-141 windows), each crawl row's X " + M + " P10 gap is LOCALISED: one"
       " subclass (GT path shape STRAIGHT / GENTLE / TURN x v0 bucket) holding <= 30 % of M's windows carries >= 0.5 of the row's M gap"
       " with CI lower > 0.3 at both seeds for DAG-OACC; DIFFUSE otherwise.",
       "**MEASURED 2026-11-02, T1-kin, the landed gap split's per-window arrays, paired episode-cluster bootstrap B = 2,000 (another draw"
       " of episodes only); card frozen 14:19:53 (sha256 1c5dbaf0…) before any subclass number.** Known answers PASS (subclass sums"
       " reproduce class M within 5e-15). **LOCALISED to v0 3-8 m/s** (413 windows, 20.3 % of M), DAG-OACC share s0 / s1: yaw-rate"
       " +0.653 / +0.684, L2 +0.934 / +0.915, M1 +0.792 / +0.751, every lower bound above 0.5; SOUP the same by the rule. DIFFUSE:"
       " curvature MAE (+0.473 at s0, +0.549 at s1) and ADE6. Path shape localises nothing: TURN holds +0.100 / +0.084 of yaw-rate with"
       " 0.074 of the windows, so the lateral excess does not live in turns. Beside (MEASURED, 3-8 m/s): plan mean |kappa| 0.1845 / 0.2621"
       " against P10 0.0623 and GT 0.0135 with a signed error of about 0, and plan speed only +0.398 / +0.241 m/s above GT (P10 +1.599): the"
       " excess is curvature jitter, not a bias or a speed collapse. POST-HOC: 9.0 / 11.4 % of DAG-OACC's 3-8 windows have a plan step"
       " below 0.5 m/s (P10 0.7 %). ESTIMATED: with the 3-8 gap set to 0, D2 still fails. Next: the pre-registered SG(7, 2) residual"
       " smoother (H-V8F-SMOOTH-1, PASS at T0 on 10-20) read at T1-kin.",
       "MEASURED (LOCALISED to 3-8 m/s on yaw-rate, L2, M1; curvature MAE and ADE6 DIFFUSE)",
       "`" + PK + "RESULTS.md`")
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
assert g.endswith(b"\n") and b"**H-GAPDECOMP-M-1**" not in g and b"\r" not in g
add = ("| " + " | ".join(ROW) + " |\n").encode("utf-8")
for w in BANNED:
    assert w not in add.lower(), w
assert add.count(b" | ") == 4 and g.rstrip(b"\n").split(b"\n")[-1].count(b" | ") == 4
put("mods/" + GC, g + add)
put("msg.txt", ("Moving-class split vs P10 (H-GAPDECOMP-M-1): the lateral excess of yaw-rate, L2 and M1 is LOCALISED to v0 3-8 m/s (20 % of"
                " moving windows); path shape localises nothing. There DAG-OACC's plan |kappa| is 0.18-0.26 1/m against GT 0.0135 with no signed"
                " bias: curvature jitter. Known answers pass. Package raw/2026-11-02-gapdecomp-m/ (" + str(n) + " files); register row."
                + chr(10) + chr(10) + "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" + chr(10)).encode("utf-8"))
print("hold written", H, n, "not landed", len(notl))
