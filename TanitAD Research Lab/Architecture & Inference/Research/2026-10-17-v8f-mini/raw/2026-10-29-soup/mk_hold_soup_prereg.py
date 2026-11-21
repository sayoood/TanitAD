"""Hold: PREREG_V8F_SOUP promoted (status line + section 0 item 5 only; no soup has been rolled out, evaluated or read), the
SOUP package (reader, builder, local chain, tests, evidence) as raw/2026-10-29-soup/, and register row H-V8F-SOUP-1
(PRE-REGISTERED). The six soup checkpoints (29 MB each) stay on E: and are listed with sha256 in NOT_LANDED.txt.
python mk_hold_soup_prereg.py <live tip>"""
import hashlib
import importlib.util
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
H = "C:/Users/Admin/qland/hold_soup_prereg/"
S = "E:/Projects/TanitAD-artifacts/dagland/soup/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-29-soup/"
PR = "Project Steering/PREREG_V8F_SOUP.md"
assert not os.path.exists(H), H
BANNED = (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn")
LIMIT = 3_000_000


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


# ---- the PREREG: draft -> promoted (two exact replacements, nothing else)
draft = open(S + "PREREG_V8F_SOUP.md", "rb").read()
assert sha(draft).startswith("4caafe7935b4aa42"), "the draft is not the one the finishing agent handed back"
assert blob(PR) is None
OLD_ST = (b"**Status: DRAFT for the Master Mind, NOT RUN (2026-10-28).**")
NEW_ST = (b"**Status: PRE-REGISTERED, NOT RUN (promoted by the Master Mind 2026-10-29, from the draft sha256 4caafe79\xe2\x80\xa6; the"
          b" promotion changed this line and section 0 item 5 only).**")
OLD_I5 = (b"| BLOCKED, named: the banking stays blocked here; the Master Mind banks it later, and until then literature statements"
          b" decide nothing |")
NEW_I5 = (b"| DONE (Master Mind, 2026-10-29): banked in the Library at `5cc7d7b1` (key `2203.05482`, sha256 226ac912\xe2\x80\xa6);"
          b" the literature statements in section 1 still decide nothing in this read |")
assert draft.count(OLD_ST) == 1 and draft.count(OLD_I5) == 1
prom = draft.replace(OLD_ST, NEW_ST).replace(OLD_I5, NEW_I5)
assert b"\r" not in prom
lib = blob("TanitAD Research Lab/Library/library.json")
assert b'"2203.05482"' in lib and b"226ac91243bdf4ebe79652991f59fa1c0c84d5d624a50b88210bf7e9316869e3" in lib
# the reader's own PREREG-TEXT check must pass on the promoted bytes (section 10 is untouched)
tmp = "F:/TanitAD-scratch/soup_prereg_check/PREREG_V8F_SOUP.md"
os.makedirs(os.path.dirname(tmp), exist_ok=True)
open(tmp, "wb").write(prom)
spec = importlib.util.spec_from_file_location("soup_read", S + "read/soup_read.py")
rdr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rdr)
probs = rdr.prereg_text_problems(tmp, sha(prom))
assert probs == [], probs
put("stage/" + PR, prom)

# ---- the package
n, notl = 0, []
for root, dirs, fs in os.walk(S):
    for f in fs:
        full = os.path.join(root, f).replace("\\", "/")
        rel = full[len(S):]
        b = open(full, "rb").read()
        why = ("checkpoint: rebuilt bit-for-bit by soup_build.py from banked members; PINS.txt carries its sha256"
               if rel.startswith("ckpt/") else
               "chain log" if rel.startswith("logs/") else
               "cache" if "__pycache__" in rel or ".pytest_cache" in rel else
               "over 3 MB" if len(b) >= LIMIT else None)
        if why:
            notl.append(f"{rel} | {len(b)} | {sha(b)} | {why}")
            continue
        if not b:
            continue
        if rel == "PREREG_V8F_SOUP.md":
            rel = "PREREG_V8F_SOUP_DRAFT_4caafe79.md"
        if rel.endswith((".py", ".sh", ".md")) and b"\r" in b:
            b = b.replace(b"\r\n", b"\n")
            assert b"\r" not in b, rel
        assert blob(PK + rel) is None, rel
        put("stage/" + PK + rel, b)
        n += 1
put("stage/" + PK + "NOT_LANDED.txt", ("# NOT_LANDED: files kept on the dev box (" + S + ")\n# path | bytes | sha256 | why\n"
                                       + "\n".join(sorted(notl)) + "\n").encode("utf-8"))
put("stage/" + PK + "mk_hold_soup_prereg.py", open(__file__.replace("\\", "/"), "rb").read().replace(b"\r\n", b"\n"))
n += 2

ROW = ("**H-V8F-SOUP-1** (2026-10-29)",
       "A uniform same-init weight average of two DAG-OACC-recipe O fine-tunes (SOUP_s = DAG-OACC_s + REP_s; replicate SOUP'_s = A_s +"
       " A_rep,s), zero training, read at T1-kin against its pair's control (DAG-OACC_s; A_s) and against P10 on the 19 rows of the four"
       " families: it removes run-to-run draw variance and moves the crawl rows by their bars without breaking a guard.",
       "Pre-registered 2026-10-29; nothing rolled out or read. Bar 2 x max(F, RF0) per row as PREREG_V8F_LAUNCH_INTENT section 6 (F ="
       " 0.1207 m); paired episode-cluster bootstrap B = 2,000 over 141 episodes, per pair, at both seeds; R-T0 read from the dev-box T0"
       " of both arms (K-T0DEV). Outcomes REFUSED / HARMS / DOMINANT (ADOPTED or NOT) / MOVED (ADOPTED or NOT) / ONE SEED / NO EFFECT"
       " (section 10); routing by the soup gain SG on ADE6: BARRIER -> ENS2 (output ensemble), FLAT -> SOUP-K (4 box-2 runs, its own PI"
       " go). First-order prediction for SOUP_0 - DAG-OACC_0 on ADE6 about +0.20 m worse (ESTIMATED: REP_0 is a known slow draw, +0.4025"
       " m). Instrument: reader sha256 de39aefe\u2026 (127 tests, 286 / 288 mutants killed, 2 named equivalent; dry run reproduces the"
       " LAUNCH-INTENT tables with 0 differences); local chain 28 steps, about 80 min of dev-box CPU (ESTIMATED).",
       "PRE-REGISTERED",
       "`" + PR + "`")
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
assert g.endswith(b"\n") and b"**H-V8F-SOUP-1**" not in g and b"\r" not in g
add = ("| " + " | ".join(ROW) + " |\n").encode("utf-8")
for w in BANNED:
    assert w not in add.lower() and w not in NEW_ST.lower() and w not in NEW_I5.lower(), w
assert add.count(b" | ") == 4 and g.rstrip(b"\n").split(b"\n")[-1].count(b" | ") == 4
put("mods/" + GC, g + add)
open(S + "../soup_land/PROMOTED_SHA.txt", "w").write(sha(prom) + "  " + PR + "\n")
put("msg.txt", ("PREREG_V8F_SOUP promoted (PRE-REGISTERED, NOT RUN; status line and section 0 item 5 only; sha256 " + sha(prom)[:12]
                + "...): a same-init weight average of two DAG-OACC-recipe O fine-tunes, read at T1-kin against its pair's control and P10,"
                " zero training, about 80 min of dev-box CPU. The package raw/2026-10-29-soup/ (" + str(n) + " files: reader, soup"
                " builder, local chain, tests, evidence; the draft kept as PREREG_V8F_SOUP_DRAFT_4caafe79.md; NOT_LANDED.txt lists "
                + str(len(notl)) + " files kept on E:, the six soup checkpoints among them). Register row H-V8F-SOUP-1.\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n").encode("utf-8"))
print("hold written", H, n, "not landed", len(notl), "promoted sha", sha(prom))
