"""Hold: PREREG_V8F_LAUNCH_INTENT Amendment 1 (A1-2026-10-28, K-BUILD's data-module clause, before any LAUNCH-INTENT row is read), appended
(mods); the amended reader with its tests, mutation pass and the real-record check in raw/2026-10-28-launch-intent-amend1/; a prefix on register
row H-V8F-LAUNCH-INTENT-1 (mods-rw; prefix only).  python mk_hold_li_amend1.py <live tip>"""
import hashlib
import json
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
D = "E:/Projects/TanitAD-artifacts/dagland/launch_intent_v2/amend1/"
H = "C:/Users/Admin/qland/hold_li_amend1/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-28-launch-intent-amend1/"
assert not os.path.exists(H), H
BANNED = (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn")


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, (rel, len(H + rel))
    assert not re.search(rb"hf_[A-Za-z0-9]{8,}", b), rel
    if rel.endswith((".py", ".sh", ".md")):
        assert b"\r" not in b, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


rd = lambda p: open(p, "rb").read()
sha = lambda b: hashlib.sha256(b).hexdigest()

PR = "Project Steering/PREREG_V8F_LAUNCH_INTENT.md"
p = blob(PR)
assert sha(p).startswith("469bac1f3918") and b"## Amendment 1" not in p
a1 = rd(D + "AMEND1.md")
assert a1.startswith(b"\n## Amendment 1 (A1-2026-10-28)")
for w in BANNED:
    assert w not in a1.lower(), w
new = p + a1
assert new == rd(D + "PREREG_V8F_LAUNCH_INTENT.md")
PSHA = sha(new)
put("mods/" + PR, new)

M = json.loads(rd(D + "mutation_reader_amend1.json").decode("utf-8"))
assert M["baseline"]["rc"] == 0 and M["n_mutants"] == 247 and M["n_killed"] == 247, (M.get("n_mutants"), M.get("n_killed"))
st = rd(D + "static_check_amend1.txt")
assert b"LAUNCH-INTENT-READ-STATIC-OK" in st and b"87 passed" in rd(D + "pytest_amend1.txt")
chk = rd(D + "kbuild_real_record_check.txt")
assert b"landed reader 5e950dcf REFUSED" in chk and b"amended reader (A1) PASS" in chk and b"deliberate regression" in chk
RSHA = sha(rd(D + "read/launch_intent_read.py"))
n = 0
for f in ("AMEND1.md", "read/launch_intent_read.py", "read/test_launch_intent_read.py", "read/mutation_launch_intent_reader.py",
          "read/ref/rep_s0_speedcells.json", "reader_amend1.diff", "static_check_amend1.txt", "pytest_amend1.txt",
          "mutation_reader_amend1.json", "mutation_amend1.out", "mutation_reader_amend1_pre_stampfix.json",
          "mutation_amend1_pre_stampfix.out.txt", "kbuild_record_d8c9f7bf.json", "kbuild_real_record_check.txt",
          "mk_hold_li_amend1.py"):
    b = rd(D + f)
    assert blob(PK + f) is None and len(b) < 400_000, f
    put("stage/" + PK + f, b)
    n += 1

GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
lines = g.split(b"\n")
idx = [i for i, l in enumerate(lines) if l.startswith(b"| **H-V8F-LAUNCH-INTENT-1** (2026-10-28) |")]
assert len(idx) == 1
row = lines[idx[0]].decode("utf-8")
OLD = "**PRE-REGISTERED, NOT RUN (2026-10-28).**"
assert row.count(OLD) == 1
PRE = ("**Amendment 1 (A1-2026-10-28, before any LAUNCH-INTENT row is read): K-BUILD admits the launch commit's data module when it is DAG-ORC"
       " r2's build (sha256 `c117e1cf\u2026`) with its twelve flag-gated hunks PASS; the reader changes in that clause alone (sha256 `" + RSHA[:8]
       + "\u2026`, 87 tests, 247 / 247 mutants killed). No bar, row, outcome or other control changes.** ")
for w in BANNED:
    assert w not in PRE.encode("utf-8").lower(), w
row = row.replace(OLD, PRE + OLD)
lines[idx[0]] = row.encode("utf-8")
ng = b"\n".join(lines)
assert ng.count(b"\n") == g.count(b"\n") and ng.replace(PRE.encode("utf-8"), b"") == g
put("mods-rw/" + GC, ng)
put("msg.txt", ("PREREG_V8F_LAUNCH_INTENT Amendment 1 (A1-2026-10-28), before any LAUNCH-INTENT row is read: K-BUILD's added sentence"
                " 'v8f_mini_data.py is byte-identical' was measured against d7208328, not against aeb54468 (the commit DAG-OACC trained at)."
                " MEASURED: the launch commit's data module equals DAG-ORC r2's build (c117e1cf) and differs from aeb54468 by DAG-ORC's twelve"
                " compose hunks, each PASS in DAG-ORC r2's accepted K-BUILD record. K-BUILD is read by its registered rule with the data module"
                " pinned to DAG-ORC r2's bytes. Reader: that clause only (sha256 %s); 87 tests GREEN on the amended text (sha256 %s), 247 /"
                " 247 mutants killed; on the real record at d8c9f7bf the landed reader REFUSES on this clause alone and the amended reader"
                " PASSES, and a foreign data module is REFUSED. Package raw/2026-10-28-launch-intent-amend1/ (%d files); register prefix."
                "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % (RSHA[:12], PSHA[:12], n)).encode("utf-8"))
print("hold written", H, n, "PREREG", PSHA, "reader", RSHA)
