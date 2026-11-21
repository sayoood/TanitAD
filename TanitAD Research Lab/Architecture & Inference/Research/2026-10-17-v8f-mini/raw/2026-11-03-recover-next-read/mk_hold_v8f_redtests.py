"""Hold: v8F red-test fixes (groups 1, 2, 3, 5, 6, 7 of the RefCV9 session's list; group 4 = secret-scan pins HELD for a PI decision; 7c =
test_r9_e1_floor.py left to its owner) + the RECOVER-NEXT read package (reader --r-arm, combiner, pre-check, chains) as
raw/2026-11-03-recover-next-read/. python mk_hold_v8f_redtests.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
H = "C:/Users/Admin/qland/hold_v8f_redtests/"
RT = "E:/Projects/TanitAD-artifacts/dagland/v8f_redtests/"
RD = "E:/Projects/TanitAD-artifacts/dagland/recover_next/read/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PKR = V8 + "raw/2026-11-03-recover-next-read/"
LANDED_ARR = V8 + "raw/2026-11-03-recover-read-result/raw/arrays/"
assert not os.path.exists(H), H
BANNED = (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn")
HELD = {"stack/tests/test_secret_scan.py": "group 4: 20 new exact pins change the literal PI decision table; needs the PI's yes",
        "tools/secret_scan.py": "group 4: as above",
        "stack/tests/test_r9_e1_floor.py": "group 7c: the RefCV9 session's file; told it the 4 sites instead"}


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def sha(b):
    return hashlib.sha256(b).hexdigest()


def put(rel, b):
    assert len(b) > 0 and len(rel) <= 220, rel
    assert not re.search(rb"hf_[A-Za-z0-9]{8,}", b), rel
    if rel.endswith((".py", ".sh", ".md")):
        assert b"\r" not in b, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


n_rw = n_new = 0
for line in open(RT + "BASE.txt", encoding="utf-8"):
    if line.startswith("#") or not line.strip():
        continue
    path, base, new = [x.strip() for x in line.split("|")[:3]]
    if path in HELD:
        continue
    b = open(RT + "stack_patch/" + path, "rb").read()
    assert sha(b) == new and b"\r" not in b, path
    tip = blob(path)
    if base == "NEW":
        assert tip is None, path
        put("stage/" + path, b)
        n_new += 1
    else:
        assert tip is not None and sha(tip) == base, f"{path}: the tip blob moved since the patch's base; rebase"
        put("mods-rw/" + path, b)
        n_rw += 1
assert (n_rw, n_new) == (9, 1), (n_rw, n_new)
# the read package
n, tot, notl = 0, 0, []
for root, dirs, fs in os.walk(RD):
    for f in fs:
        full = os.path.join(root, f).replace("\\", "/")
        rel = full[len(RD):]
        b = open(full, "rb").read()
        if rel.endswith((".py", ".sh", ".md")) and b"\r" in b:
            b = b.replace(b"\r\n", b"\n")
        why = None
        if "__pycache__" in rel or ".pytest_cache" in rel:
            why = "cache"
        elif rel.startswith("raw/arrays/") and rel.endswith(".npz") and blob(LANDED_ARR + rel[len("raw/arrays/"):]) == b:
            why = "byte-identical to the landed " + LANDED_ARR + rel[len("raw/arrays/"):]
        elif len(b) >= 3_000_000:
            why = "over 3 MB"
        if why:
            notl.append(f"{rel} | {len(b)} | {sha(b)} | {why}")
            continue
        if not b:
            continue
        assert blob(PKR + rel) is None, rel
        put("stage/" + PKR + rel, b)
        n += 1
        tot += len(b)
notl.append("C:/Users/Admin/rr_scratch/rnext_work/ | - | - | test scratch (regenerable)")
put("stage/" + PKR + "NOT_LANDED.txt", ("# NOT_LANDED\n# path | bytes | sha256 | why\n" + "\n".join(sorted(notl)) + "\n").encode("utf-8"))
put("stage/" + PKR + "mk_hold_v8f_redtests.py", open(__file__.replace("\\", "/"), "rb").read().replace(b"\r\n", b"\n"))
put("stage/" + PKR + "HELD_FROM_THIS_LAND.txt", ("".join(f"{k} | {v}\n" for k, v in HELD.items())).encode("utf-8"))
msg = ("v8F red tests for the RefCV9 launch gate (groups 1, 2, 3, 5, 6, 7): saved-module configs drop launch_intent_w at its off value; the queue-label"
       " tests subtract the later edits' labels (new stack/tests/v8f_queue_later_labels.py); speedread passes window_start (unchanged behaviour,"
       " -1 = hist - 1); the flag-off golden tests allow exactly the six RECOVER-FT summary keys at their flag-off values; explicit utf-8 at three"
       " speedsub sites. 0 new failures over the touched suites + speedread / t1_speedsub (60 / 60). Secret-scan pins (group 4) held for a PI"
       " decision. Plus the RECOVER-NEXT read package raw/2026-11-03-recover-next-read/ (reader --r-arm: known answer 0 diffs; 27 + 30 + 18 tests;"
       " pulled-file pre-check): " + str(n) + " files, NOT_LANDED lines " + str(len(notl)) + "."
       + chr(10) + chr(10) + "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" + chr(10))
for w in BANNED:
    assert w not in msg.lower().encode(), w
put("msg.txt", msg.encode("utf-8"))
print("hold written", H, "rw", n_rw, "new", n_new, "pkg", n, "kB", tot // 1000, "not landed", len(notl))
