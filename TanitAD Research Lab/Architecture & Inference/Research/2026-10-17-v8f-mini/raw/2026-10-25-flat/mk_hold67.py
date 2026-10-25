"""Build qland/hold_v8f_flat_instruments: the PREREG_V8F_FLAT instruments from the export tree S/flat (FILES.txt: MODIFIED vs tip 18c62306 -> mods-rw, NEW -> stage),
bytes exactly as in the tree (FILES.txt hashes the raw blobs). The PREREG itself is skipped (byte-identical to the tip's Amendment-1 blob). python mk_hold67.py <tip>"""
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
T = SC + "flat/"
A2 = SC + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8f_flat_instruments"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
BASE = "18c623068626d5f9f57d5aeb09d2fddb6b89e1fb"


def git(*a):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def blob(rev, p):
    b = git("rev-parse", f"{rev}:{p}")
    return b.decode().strip() if b else None


def put(kind, rel, data):
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


for n in ("hold_v8f_flat_instruments", "landed_hold_v8f_flat_instruments"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
sec, mod, new = None, [], []
for line in open(T + "FILES.txt", encoding="utf-8").read().splitlines():
    if line.strip() in ("MODIFIED", "NEW"):
        sec = line.strip()
        continue
    m = re.match(r"^  (.+?)  \((\d+) bytes\)", line)
    if m and sec:
        (mod if sec == "MODIFIED" else new).append(m.group(1))
assert len(mod) == 9 and len(new) >= 14, (len(mod), len(new))
for p in mod:
    data = open(T + p, "rb").read()
    if p == "Project Steering/PREREG_V8F_FLAT.md":
        h = subprocess.run(["git", "hash-object", "--stdin"], input=data, capture_output=True).stdout.decode().strip()
        assert h == blob(TIP, p), "the PREREG copy is not the tip's Amendment-1 blob"
        continue
    assert blob(BASE, p) == blob(TIP, p), f"{p} moved on the tip since {BASE[:8]}"
    put("mods-rw", p, data)
for p in new:
    assert blob(TIP, p) is None, p
    put("stage", p, open(T + p, "rb").read())
put("stage", f"{PK}/raw/2026-10-25-flat/mk_hold67.py", open(A2 + "mk_hold67.py", "rb").read().replace(b"\r\n", b"\n"))

lines = git("show", f"{TIP}:Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-FLAT-1**")]
assert len(ix) == 1
old = "**PRE-REGISTERED, NOT RUN (2026-10-25).**"
assert lines[ix[0]].count(old) == 1
lines[ix[0]] = lines[ix[0]].replace(old, (
    "**PRE-REGISTERED, INSTRUMENTS LANDED, NOT RUN (2026-10-25).** The O-interface cut (`--o-interface cut`: O reads constants in place of S's checkpoints"
    " and T's code), its in-process probe, the reader and box-2 chain. Mutation proof 172 / 172 RED by an assertion: 164 in the first run, the 8 survivors"
    " killed by 9 new tests, the 4 kills made during the disk-full window re-run clean, and the 7 cut-path mutants caught behaviourally once the import-time"
    " anchor pin was lifted in a scratch copy (`raw/2026-10-25-flat/behaviour_only.out`). v8F suites 1,261 passed, 0 failed, 2 skipped"
    " (`raw/2026-10-25-flat/suites_after_survivor_tests.out`). Box 2 after KSCORE."))
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("PREREG_V8F_FLAT instruments: the O-interface cut, its probe, reader and box-2 chain (172/172 mutants RED, suites 1261 passed)\n\n"
       "O reads constants in place of S's checkpoints and T's code; S and T still train. Survivors killed by 9 new tests; the cut-path mutants\n"
       "caught behaviourally. Register: H-V8F-FLAT-1.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok:", len(mod) - 1, "mods,", len(new), "new")
