"""Hold: the SOUP read of record banked (raw/2026-10-29-soup-read/: SOUP_READ.json, the manifest, rollout / T0 summaries, done records,
argv, chain and reader logs, the launcher as launched, the promoted dry run), and register row H-V8F-SOUP-1 set to MEASURED (HARMS)
(mods-rw: the evidence cell gains a prefix, the status cell is replaced; every other byte kept). Per-window dumps and tick arrays stay
on F: (NOT_LANDED.txt).  python mk_hold_soup_read.py <live tip>"""
import hashlib
import json
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
H = "C:/Users/Admin/qland/hold_soup_read/"
O = "F:/TanitAD-scratch/soup_t1_out/"
SP = "E:/Projects/TanitAD-artifacts/dagland/soup/"
SL = "E:/Projects/TanitAD-artifacts/dagland/soup_land/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-29-soup-read/"
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


RB = open(O + "read/SOUP_READ.json", "rb").read()
R = json.loads(RB.decode("utf-8"))
OC = R["OUTCOME"]
assert OC["OUTCOME"] == "HARMS" and OC["adopts"] is False and OC["route"] == "FLAT" and OC["next_soup"] == "SOUP-K"
assert len(R["controls_PASS"]) == 368 and all(R["controls_PASS"].values())
assert R["provenance"]["prereg_sha256"] == "830d9c9143850628188f7d8489566e92f2a9cbdf050689331f1eaa7d3c78993c"
assert R["provenance"]["reader_sha256"] == "de39aefe8d5770889309e874e33eceb89fbbb2046b766a049de26ec5a6321c67"
assert sha(open(SP + "read/soup_read.py", "rb").read()) == R["provenance"]["reader_sha256"]

n, notl = 0, []
for root, dirs, fs in os.walk(O):
    for f in fs:
        full = os.path.join(root, f).replace("\\", "/")
        rel = full[len(O):]
        b = open(full, "rb").read()
        why = ("per-window dump" if rel.startswith("dump_") else "tick arrays" if rel.startswith("ticks/") else
               "chain temp" if rel.startswith("_tmp/") else "over 3 MB" if len(b) >= LIMIT else None)
        if why:
            notl.append(f"{rel} | {len(b)} | {sha(b)} | {why}")
            continue
        if not b:
            continue
        put("stage/" + PK + "out/" + rel, b)
        n += 1
for f in sorted(os.listdir(SP + "logs")):
    b = open(SP + "logs/" + f, "rb").read()
    if b and len(b) < LIMIT:
        put("stage/" + PK + "logs/" + f, b)
        n += 1
for f, src in (("as_launched/soup_chain.cmd", SP + "soup_chain.cmd"), ("as_launched/soup_chain.cmd.as_shipped", SL + "soup_chain.cmd.as_shipped"),
               ("as_launched/dry_run_promoted.out", SL + "dry_run_promoted.out"), ("as_launched/wake.out", SL + "wake.out"),
               ("mk_hold_soup_read.py", SL + "mk_hold_soup_read.py")):
    b = open(src, "rb").read()
    put("stage/" + PK + f, b.replace(b"\r\n", b"\n") if f.endswith(".py") else b)
    n += 1
put("stage/" + PK + "NOT_LANDED.txt", ("# NOT_LANDED: files of the run kept on the dev box (" + O + ")\n# path | bytes | sha256 | why\n"
                                       + "\n".join(sorted(notl)) + "\n").encode("utf-8"))
n += 1

T = R["tables"]
SG = R["sg"]


def c(t, s, r="ADE6"):
    x = T[t][s][r]

    def g(v):
        return ("+" if v >= 0 else M) + "%.4f" % abs(v)
    return "%s [%s, %s]" % (g(x["delta"]), g(x["lo"]), g(x["hi"]))


def sg(k):
    x = SG[k]

    def g(v):
        return ("+" if v >= 0 else M) + "%.4f" % abs(v)
    return "%s [%s, %s]" % (g(x["delta"]), g(x["lo"]), g(x["hi"]))


RSHA = sha(RB)[:8]
PRE = ("**HARMS (MEASURED 2026-10-29, T1-kin, all 368 controls PASS; not adopted).** H fails at s0 only: ADE6 SOUP_0 " + M + " DAG-OACC_0 "
       + c("SOUP_CTRL", "s0") + " m and SOUP'_0 " + M + " A_0 " + c("SOUPR_A", "s0") + " m (R-ADE), and TAC-LAT macro-F1 SOUP_0 " + M
       + " DAG-OACC_0 " + c("SOUP_CTRL", "s0", "TAC-LAT") + " (R-TLAT); at s1 every guard holds (SOUP_1 " + M + " DAG-OACC_1 "
       + c("SOUP_CTRL", "s1") + " m; SOUP'_1 " + M + " A_1 " + c("SOUPR_A", "s1") + " m, separated-better). No crawl row moved by its bar"
       " in either pair at either seed (R-MOVE false), and every crawl row stays separated-worse against P10 (R-CRAWL false). The soup"
       " gain is real and FLAT-routed: SG on ADE6 (the soup minus the mean of its members' errors) " + sg("SOUP:s0") + " (s0), "
       + sg("SOUP:s1") + " (s1); SOUP' " + sg("SOUPR:s0") + " / " + sg("SOUPR:s1") + " m; no barrier. Against its slow member, SOUP_0 "
       + M + " REP_0 reads " + c("SOUP_REP", "s0") + " m: averaging recovers most of the bad draw, and the soup still lands worse than"
       " the good one. Paired episode-cluster bootstrap, B = 2,000, 141 episodes / 2,399 windows, per pair. Committed next (section 11,"
       " FLAT): SOUP-K, K = 4 pure-recipe draws per seed, 4 box-2 runs, its own PI go. Read `" + PK + "out/read/SOUP_READ.json` (sha256 "
       + RSHA + "\u2026); RESULT to follow. Was: ")
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
assert g.endswith(b"\n") and b"\r" not in g
lines = g.split(b"\n")
idx = [i for i, l in enumerate(lines) if l.startswith(b"| **H-V8F-SOUP-1** (2026-10-29) | ")]
assert len(idx) == 1
cells = lines[idx[0]].decode("utf-8").split(" | ")
assert cells[2].startswith("Pre-registered 2026-10-29; nothing rolled out or read.") and cells[3] == "PRE-REGISTERED"
cells[2] = PRE + cells[2]
cells[3] = "MEASURED (HARMS)"
for w in BANNED:
    assert w not in PRE.encode("utf-8").lower(), w
lines[idx[0]] = " | ".join(cells).encode("utf-8")
ng = b"\n".join(lines)
assert ng.count(b"\n") == g.count(b"\n")
put("mods-rw/" + GC, ng)
put("msg.txt", ("SOUP read: HARMS (H-V8F-SOUP-1), T1-kin, all 368 controls PASS, not adopted. H fails at s0 only (ADE6 SOUP_0 - DAG-OACC_0 "
                + c("SOUP_CTRL", "s0") + " m; TAC-LAT " + c("SOUP_CTRL", "s0", "TAC-LAT") + "); s1 holds (SOUP'_1 - A_1 "
                + c("SOUPR_A", "s1") + " m). No crawl row moved. The soup gain is real: SG " + sg("SOUP:s0") + " (s0) / "
                + sg("SOUP:s1") + " (s1) m, no barrier, so the FLAT route commits SOUP-K (4 box-2 runs, its own PI go). Read banked in"
                " raw/2026-10-29-soup-read/ (" + str(n) + " files; per-window dumps and tick arrays on F:, NOT_LANDED.txt);"
                " register row updated; RESULT to follow.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n").encode("utf-8"))
print("hold written", H, n, "not landed", len(notl), RSHA)
