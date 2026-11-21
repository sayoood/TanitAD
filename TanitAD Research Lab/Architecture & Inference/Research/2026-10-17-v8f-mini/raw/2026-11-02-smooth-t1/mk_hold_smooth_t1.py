"""Hold: SMOOTH-T1 (H-SMOOTH-T1-1: the first read and its AMENDMENT_1 read) banked as raw/2026-11-02-smooth-t1/, and its register row (append).
Scratch on F: stays there (NOT_LANDED.txt). python mk_hold_gapdecomp.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
H = "C:/Users/Admin/qland/hold_smooth_t1/"
S = "E:/Projects/TanitAD-artifacts/dagland/smooth_t1/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-11-02-smooth-t1/"
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
assert pins.get("PREREG_FROZEN.md", "").startswith("4030d22f") and pins.get("AMENDMENT_1.md", "").startswith("717ff177") and pins.get("RESULTS_A1.md", "").startswith("2be30c96"), "manifest"
txt = b"".join(open(S + f, "rb").read() for f in ("RESULTS_A1.md", "RESULTS.md"))
n, notl = 0, []
for root, dirs, fs in os.walk(S):
    for f in fs:
        full = os.path.join(root, f).replace("\\", "/")
        rel = full[len(S):]
        b = open(full, "rb").read()
        if rel in pins and not rel.startswith("logs/"):  # logs grow after the manifest; everything else is pinned
            assert sha(b) == pins[rel], rel
        why = ("cache" if "__pycache__" in rel or ".pytest_cache" in rel else "over 3 MB (rollout dumps / tick arrays; regenerable by the chains)" if len(b) >= LIMIT else None)
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
notl.append("F:/TanitAD-scratch/smooth_t1/ | - | - | rollout outputs out/, out_m16/ (regenerable by the chains; per-window dumps also listed above when over 3 MB)")
put("stage/" + PK + "NOT_LANDED.txt", ("# NOT_LANDED\n# path | bytes | sha256 | why\n" + "\n".join(sorted(notl)) + "\n").encode("utf-8"))
put("stage/" + PK + "mk_hold_smooth_t1.py", open(__file__.replace("\\", "/"), "rb").read().replace(b"\r\n", b"\n"))
n += 2
for q in ("-1.2556", "-1.1903", "-1.1928", "-1.1895", "-0.0544", "-0.0671", "-0.0687", "-0.0558", "QUOTABLE ON Y'", "MOVED", "NO EFFECT",
          "1.71e-02"):
    assert q.encode() in txt, q
ROW = ("**H-SMOOTH-T1-1** (2026-11-02)",
       "The pre-registered SG(7, 2) residual smoother (H-V8F-SMOOTH-1, PASS at T0 on 2026-10-20), applied to the plan at every replan"
       " inside the T1-kin rollout, makes v8F DOMINANT against P10 (Read A at both seeds), or at least MOVES every crawl row (yaw-rate"
       " MAE, curvature MAE, L2, M1) separated-better against the unsmoothed arm at both seeds with ADE6 not worse than +0.0604 m.",
       "**MEASURED 2026-11-02, T1-kin, eval-141 causal lane (2,399 windows), DAG-OACC and SOUP at s0 / s1, paired episode-cluster"
       " bootstrap B = 2,000 (another draw of episodes only); card frozen 14:40:42 (sha256 4030d22f…) before any smoothed number.**"
       " The first read was REFUSED by the card's controls (1-thread rollouts against a 16-thread banked baseline; at 16 threads the"
       " P10 floor recomputed inside the rollout differed from the banked one). AMENDMENT_1 (control-only, frozen 15:41:09, sha256"
       " 717ff177…, disclosing that the 1-thread tables had been seen): fresh unsmoothed runs Y' at 16 threads are the decision"
       " baseline; two identical fresh runs are bit-identical, so every 1e-4 tolerance stands; the banked baseline differs from a"
       " fresh run by up to 1.71e-02 m on the floor, so the status is QUOTABLE ON Y'. **SOUP + S: MOVED (both seeds). DAG-OACC + S:"
       " NO EFFECT (MOVED at s0, NO EFFECT at s1). Not DOMINANT. No HARMS** (0 of 19 rows separated-worse against Y' in any run)."
       " ADE6 X " + M + " Y' " + M + "0.0544 / " + M + "0.0671 m (DAG-OACC), " + M + "0.0687 / " + M + "0.0558 m (SOUP),"
       " separated-better. Read A against P10: D1 PASS (ADE6 " + M + "1.2556 / " + M + "1.1903 / " + M + "1.1928 / " + M + "1.1895 m),"
       " D3 PASS (each family has a separated-better row), D2 FAILS on curvature MAE, L2 curvature MAE and M1 crawl share (and"
       " yaw-rate MAE at s1). The same outcome words hold on the banked baseline. Post-hoc (seam diagnostic): what still blocks D2 is a"
       " near-uniform per-knot curvature overshoot over P10, which a within-plan smoother does not act on. The patched tool"
       " (`--smooth`, default off, bit-identical off) is banked in the package, not in `stack/` (it would move the tool pin of"
       " `v8f_t1_speedsub.py`). Next: the curvature lever (`dagland/kappa_t1/`, gate pre-registered).",
       "MEASURED (QUOTABLE ON Y': SOUP + S MOVED, DAG-OACC + S NO EFFECT; not DOMINANT)",
       "`" + PK + "RESULTS_A1.md`")
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
assert g.endswith(b"\n") and b"**H-SMOOTH-T1-1**" not in g and b"\r" not in g
add = ("| " + " | ".join(ROW) + " |\n").encode("utf-8")
for w in BANNED:
    assert w not in add.lower(), w
assert add.count(b" | ") == 4 and g.rstrip(b"\n").split(b"\n")[-1].count(b" | ") == 4
put("mods/" + GC, g + add)
put("msg.txt", ("SMOOTH-T1 (H-SMOOTH-T1-1), T1-kin: the SG(7, 2) residual smoother inside the rollout MOVES every crawl row for SOUP at both"
                " seeds (DAG-OACC: NO EFFECT, s1 not all separated) and improves ADE6 by 0.054-0.069 m against fresh unsmoothed runs, with no"
                " row harmed; not DOMINANT (D2 still fails on curvature, L2 and M1 vs P10). First read refused by its controls (thread count);"
                " AMENDMENT_1 (control-only, frozen first): fresh baseline Y', status QUOTABLE ON Y'. Package raw/2026-11-02-smooth-t1/ ("
                + str(n) + " files; NOT_LANDED.txt lists " + str(len(notl)) + "); register row." + chr(10) + chr(10)
                + "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" + chr(10)).encode("utf-8"))
print("hold written", H, n, "not landed", len(notl))
