"""Hold: FEAS2-T1 + FEAS3 (`dagland/feas2_t1/`, H-FEAS2-T1-1 and H-FEAS3-T1-1) and HISTSWAP (`dagland/histswap/`, H-HISTSWAP-1)
banked as raw/2026-11-02-feas2-t1/ and raw/2026-11-02-histswap/, with three register rows (append). F: scratch stays (NOT_LANDED.txt).
python mk_hold_feas_hsw.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
H = "C:/Users/Admin/qland/hold_feas_hsw/"
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
PKGS = {"feas2_t1": (V8 + "raw/2026-11-02-feas2-t1/", {"PREREG_FROZEN.md": "ecf9bb5d", "PREREG_FEAS3_FROZEN.md": "e817abcf", "RESULTS.md": "8339868d",
                                                       "RESULTS_FEAS3.md": "d09478ca", "READER_FIX_FEAS3.md": "0289a4a4"}, "F:/TanitAD-scratch/feas2_t1/"),
        "histswap": (V8 + "raw/2026-11-02-histswap/", {"PREREG_HISTSWAP_FROZEN.md": "7e91a7c7", "RESULTS.md": "398943de"}, "F:/TanitAD-scratch/histswap/")}
txt, n, tot, nl = {}, 0, 0, 0
for name, (PK, need, FS) in PKGS.items():
    S = D + name + "/"
    pins = {}
    for line in open(S + "MANIFEST.txt", encoding="utf-8"):
        m = re.match(r"\s*(\S.*?)\s*\|\s*\d+\s*\|\s*([0-9a-f]{64})\s*$", line)
        if m:
            pins[m.group(1).replace(S, "")] = m.group(2)
    for f, p in need.items():
        assert pins.get(f, "").startswith(p), (name, f)
    txt[name] = b"".join(open(S + f, "rb").read() for f in need if f.startswith("RESULTS"))
    for w in BANNED:
        assert w not in txt[name].lower(), (name, w)
    notl = []
    for root, dirs, fs in os.walk(S):
        for f in fs:
            full = os.path.join(root, f).replace("\\", "/")
            rel = full[len(S):]
            b = open(full, "rb").read()
            if rel in pins and not rel.startswith("logs/"):
                assert sha(b) == pins[rel], (name, rel)
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
            assert blob(PK + rel) is None, (name, rel)
            put("stage/" + PK + rel, b)
            n += 1
            tot += len(b)
    notl.append(f"{FS} | - | - | rollout outputs and tool trees (regenerable by the package's chain)")
    put("stage/" + PK + "NOT_LANDED.txt", ("# NOT_LANDED\n# path | bytes | sha256 | why\n" + "\n".join(sorted(notl)) + "\n").encode("utf-8"))
    n += 1
    nl += len(notl)
put("stage/" + PKGS["feas2_t1"][0] + "mk_hold_feas_hsw.py", open(__file__.replace("\\", "/"), "rb").read().replace(b"\r\n", b"\n"))
n += 1
for name, qs in (("feas2_t1", ("+0.0753", "+1.7413", "+1.3432", "+0.1283", "+0.1442", "+0.1342", "+0.1287", "0.0713", "0.0230", "0.1786", "0.0480",
                               "6.6", "import-path")),
                 ("histswap", ("SUPPORTED", "0.939", "0.937", "-0.0476", "-0.0523", "-0.0612", "-0.0537", "-1.2325", "-1.2352", "-1.2971", "-1.2806",
                               "ORACLE"))):
    for q in qs:
        assert q.encode() in txt[name], (name, q)
F2 = V8 + "raw/2026-11-02-feas2-t1/"
HS = V8 + "raw/2026-11-02-histswap/"
EST = " paired episode-cluster bootstrap B = 2,000, seed 0 (another draw of episodes only)"
ROWS = [
    ("**H-FEAS2-T1-1** (2026-11-02)",
     "After S, a variant of the curvature-bounded decode that removes KAPPA-T1's harm makes v8F DOMINANT against P10 at T1-kin, or MOVES"
     " the crawl rows against Y = the fresh S run without HARMS: kbox (kappa-only, a_max = inf, the KAPPA-T1 card's committed arm for"
     " HARMS), abox (accel-bounded longitudinal decode with entry clamp, aimed at M1), kabox (both).",
     "**MEASURED 2026-11-02, T1-kin, eval-141 causal lane, DAG-OACC and SOUP at s0 / s1," + EST + "; card frozen 16:43:48 (sha256"
     " ecf9bb5d…) before any X number; a per-tick pre-check on KAPPA-T1's seen ticks, disclosed in the card, attributed the heading harm"
     " to the curvature clamp and found plan-level crawl below P10's.** All six control groups PASS. **Outcome: HARMS for every lever"
     " on both bases.** kbox and kabox: heading MAE X " + M + " Y separated-worse (e.g. +1.7413 / +1.3432 deg) and S-POS rows; kbox"
     " ADE6 X " + M + " Y +0.0753 m at DAG-OACC s1, beyond the +0.0604 m guard; against P10, M1 crawl share is the only D2 row failing"
     " for kbox at 4/4 cells. abox closes no crawl row and fails D2 on curvature, L2 and M1. M1 is not separated-better under any lever."
     " Next: the card's committed arms (H-FEAS3-T1-1).",
     "MEASURED (HARMS, all three levers)",
     "`" + F2 + "RESULTS.md`"),
    ("**H-FEAS3-T1-1** (2026-11-02)",
     "The FEAS2 card's committed next arms make v8F DOMINANT against P10 at T1-kin, or MOVE the crawl rows without HARMS: kbox2"
     " (kappa clamp with min_speed 2.0), lonfloor (executed step speed floored at the P10 prior's step speed " + M + " 2 m/s), kbox2lf"
     " (both).",
     "**MEASURED 2026-11-02, T1-kin, eval-141 causal lane, DAG-OACC and SOUP at s0 / s1," + EST + "; card frozen 17:43:04 (sha256"
     " e817abcf…).** The first read stopped inside the card's beside checks on a stack-shadow guard; READER_FIX_FEAS3.md records an"
     " import-path-only fix (the operator file is byte-identical in both trees) and the re-run's 12 per-cell lines equal the first"
     " run's. **Outcome: HARMS for all three arms on both bases.** kbox2 ADE6 X " + M + " Y +0.1283 / +0.1442 m (DAG-OACC), +0.1342 /"
     " +0.1287 m (SOUP); lonfloor moves M1 separated-better at 4/4 cells but closes only 2.0-6.6 % of its excess and fails D2 on"
     " curvature and L2 again. Disclosure (i), the M1 margins: the crawl is not a threshold artefact (CTRL s0, row definition Y"
     " 0.0713 vs P10 0.0230; at a 1 m/s margin 0.1786 vs 0.0480). Seven decode levers since S leave M1 the D2 blocker; the cause sits"
     " upstream of the decode (H-HISTSWAP-1).",
     "MEASURED (HARMS, all three arms)",
     "`" + F2 + "RESULTS_FEAS3.md`"),
    ("**H-HISTSWAP-1** (2026-11-02)",
     "v8F's M1 crawl excess over P10 at T1-kin enters through O's ego-history input (O follows its own slowed executed history):"
     " replacing ONLY that input by the LOGGED ego history inside the rollout (plan applied from the executed state, after S) makes M1"
     " X " + M + " Y separated-better at all four cells and closes >= 0.5 of Y " + M + " P10's M1 excess; NOT SUPPORTED iff not"
     " separated-better at >= 3 of 4 cells.",
     "**MEASURED 2026-11-02, T1-kin, eval-141 causal lane, DAG-OACC and SOUP at s0 / s1," + EST + "; card frozen 17:57:30 (sha256"
     " 7e91a7c7…) before any swap number.** **SUPPORTED**: M1 X " + M + " Y " + M + "0.0476 / " + M + "0.0523 (DAG-OACC), " + M
     + "0.0612 / " + M + "0.0537 (SOUP), separated-better at 4/4 cells; share of the M1 excess closed 0.939 mean (0.937 pooled);"
     " ADE6 X " + M + " Y " + M + "1.2325 / " + M + "1.2352 / " + M + "1.2971 / " + M + "1.2806 m. **X is an ORACLE**: the logged"
     " history carries the human's speed at the tick, so the result locates the drift in the ego-history channel and is not a"
     " deployable fix. Next: the leak-free arms (constant-velocity history at the executed speed, deployable; at the logged speed,"
     " level only) in `dagland/histswap_lf/`, which separate inertia from a missing speed level.",
     "MEASURED (SUPPORTED, oracle)",
     "`" + HS + "RESULTS.md`"),
]
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
assert g.endswith(b"\n") and b"\r" not in g
add = b""
for ROW in ROWS:
    assert ROW[0].encode() not in g
    a = ("| " + " | ".join(ROW) + " |\n").encode("utf-8")
    for w in BANNED:
        assert w not in a.lower(), w
    assert a.count(b" | ") == 4
    add += a
assert g.rstrip(b"\n").split(b"\n")[-1].count(b" | ") == 4
put("mods/" + GC, g + add)
put("msg.txt", ("FEAS2-T1, FEAS3 and HISTSWAP (T1-kin, v8F after S): six decode levers (kbox, abox, kabox, kbox2, lonfloor, kbox2lf) all read HARMS;"
                " M1 crawl share stays the only D2 blocker against P10 and is not a threshold artefact. HISTSWAP (oracle: logged ego history"
                " as O's input) is SUPPORTED: it closes 0.939 of the M1 excess at 4/4 cells, locating the drift in the ego-history channel."
                " FEAS3 reader import-path fix disclosed. Packages raw/2026-11-02-feas2-t1/ and raw/2026-11-02-histswap/ (" + str(n)
                + " files, " + str(tot // 1000) + " kB; NOT_LANDED lines " + str(nl) + "); three register rows." + chr(10) + chr(10)
                + "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" + chr(10)).encode("utf-8"))
print("hold written", H, n, "kB", tot // 1000, "not landed", nl)
