"""Hold: RESULT_V8F_SOUP.md (HARMS, not adopted; route FLAT -> SOUP-K), its builder, checker and evidence (raw/2026-10-29-soup-result/),
and a prefix on register row H-V8F-SOUP-1 that points at the RESULT and states what its TAC-LAT cell is (mods-rw; prefix only).
python mk_hold_soup_result.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
H = "C:/Users/Admin/qland/hold_soup_result/"
S = "E:/Projects/TanitAD-artifacts/dagland/soup_result/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-29-soup-result/"
RP = V8 + "RESULT_V8F_SOUP.md"
assert not os.path.exists(H), H
BANNED = (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn")
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
    assert len(b) < 400_000 or rel.startswith("mods"), rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


doc = open(S + "RESULT_V8F_SOUP.md", "rb").read()
assert sha(doc).startswith("c039bf11f1a3672e")
for w in BANNED:
    assert w not in doc.lower(), w
assert b"ALL CHECKS PASS" in open(S + "evidence/check_result_PASS.out", "rb").read()
assert blob(RP) is None
put("stage/" + RP, doc)
n = 1
for root, dirs, fs in os.walk(S):
    dirs[:] = [d for d in dirs if d not in ("__pycache__", ".pytest_cache")]
    for f in fs:
        full = os.path.join(root, f).replace("\\", "/")
        rel = full[len(S):]
        if rel == "RESULT_V8F_SOUP.md":
            continue
        b = open(full, "rb").read()
        if not b:
            continue
        if rel.endswith((".py", ".sh", ".md")) and b"\r" in b:
            b = b.replace(b"\r\n", b"\n")
        assert blob(PK + rel) is None, rel
        put("stage/" + PK + rel, b)
        n += 1
put("stage/" + PK + "mk_hold_soup_result.py", open(__file__.replace("\\", "/"), "rb").read().replace(b"\r\n", b"\n"))
n += 1

GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
assert g.endswith(b"\n") and b"\r" not in g
lines = g.split(b"\n")
idx = [i for i, l in enumerate(lines) if l.startswith(b"| **H-V8F-SOUP-1** (2026-10-29) | ")]
assert len(idx) == 1
row = lines[idx[0]].decode("utf-8")
OLD = "**HARMS (MEASURED 2026-10-29, T1-kin, all 368 controls PASS; not adopted).**"
assert row.count(OLD) == 1 and row.count("TAC-LAT macro-F1") == 1
PRE = ("**RESULT 2026-10-29: `" + RP + "` (175 checks re-find every number and table cell from the read; three planted errors are"
       " caught). It states what the TAC-LAT cell below is: the T1 rig's plan-derived TAC-LAT agreement row (" + M + "0.0125 [" + M
       + "0.0250, " + M + "0.0004] at s0 main), not the T-head macro-F1 that guard R-TLAT reads; R-TLAT's verdict (it fails at s0 in the"
       " main pair) is the reader's. R-KEEP also fails at s0 in both pairs; Read A reads MARGIN, NOT DOMINANT for every soup at both"
       " seeds.** ")
for w in BANNED:
    assert w not in PRE.encode("utf-8").lower(), w
row = row.replace(OLD, PRE + OLD)
lines[idx[0]] = row.encode("utf-8")
ng = b"\n".join(lines)
assert ng.count(b"\n") == g.count(b"\n") and ng.replace(PRE.encode("utf-8"), b"") == g
put("mods-rw/" + GC, ng)
put("msg.txt", ("RESULT_V8F_SOUP: HARMS (not adopted), reported as registered. H fails at s0 only (R-ADE in both pairs, R-TLAT in the main"
                " pair); s1 holds. No crawl row moved. The soup gain is real (SG negative in all six cells, separated in five; no"
                " barrier): the soup recovers about 80 % of the slow draw's ADE6 excess at s0 and still lands worse than the good draw."
                " Route FLAT, so SOUP-K is the committed next (its own PI go); the Rule Zero section ties the crawl to perception"
                " (register rows H-CRAWL-01/02, H-DD-FITGAP-M2B, H-CRAWL-06-PROBE/2) and names a finer-token probe as PROPOSED. Checker"
                " 175 / 175; three planted errors caught. Package raw/2026-10-29-soup-result/ (" + str(n) + " files); register row"
                " prefix (points at the RESULT; states what the TAC-LAT cell is).\n\nCo-Authored-By: Claude Opus 5.5"
                " <noreply@anthropic.com>\n").encode("utf-8"))
print("hold written", H, n)
