"""Hold: PREREG_V8F_SPEEDREAD Amendment 1 (A1-2026-10-27: the ego vector takes the data's own width, 22 dims, before any SPEEDREAD number)
+ the fixed instrument and its test (mods-rw, 535 waived with its reason) + the package raw/2026-10-27-speedread-a1/ + a register note.
python mk_hold_speedread_a1.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
HERE = "E:/Projects/TanitAD-artifacts/dagland/speedread_a1/"
RUN = "E:/Projects/TanitAD-artifacts/speedread_run/"
W = "F:/TanitAD-scratch/speedread/"
H = "C:/Users/Admin/qland/hold_speedread_a1/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-27-speedread-a1/"
INS, TST = "stack/scripts/v8f_mini_eval_speedread.py", "stack/tests/test_v8f_speedread.py"
assert not os.path.exists(H), H


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


rd = lambda p: open(p, "rb").read()
sha = lambda b: hashlib.sha256(b).hexdigest()
assert sha(blob(INS)) == "d554575169ea393cf5783959d354ca6a5a0acd859e583b0b4e1f520fbb776119" == sha(rd(HERE + "tip_instrument.py"))
assert sha(blob(TST)) == "caa124a6de5e7abf2c80bd3ae001aa01168224012af2e5b67259d3ad04481f46" == sha(rd(HERE + "tip_test.py"))
ins, tst = rd(HERE + "stage/" + INS), rd(HERE + "stage/" + TST)
assert sha(ins) == "fd5ce0bea37c3e0e608144e041f54a337fd4378b38fb312df65787bc864be4fe" and b"\r" not in ins and b"\r" not in tst
assert b"fd5ce0bea37c3e0e608144e041f54a337fd4378b38fb312df65787bc864be4fe" in rd(RUN + "run_speedread.sh")
put("mods-rw/" + INS, ins)
put("mods-rw/" + TST, tst)
put("allowdrop/v8f_mini_eval_speedread.py.allowdrop", rd(HERE + "allowdrop/v8f_mini_eval_speedread.py.allowdrop"))

am = rd(HERE + "AMEND1.md")
assert b"\r" not in am and am.startswith(b"\n## Amendment 1 (A1-2026-10-27): the ego vector takes the data's own width")
PR = "Project Steering/PREREG_V8F_SPEEDREAD.md"
p = blob(PR)
assert p.endswith(b"\n") and b"## Amendment 1" not in p
put("mods/" + PR, p + am)

files = {"AMEND1.md": HERE + "AMEND1.md", "pytest_patched.txt": HERE + "pytest_patched.txt",
         "mutation_old_instrument.txt": HERE + "mutation_old_instrument.txt", "mutation_results_a1.json": HERE + "mutation_results_a1.json",
         "mutation_output_a1.txt": HERE + "mutation_output_a1.txt",
         "run/run_speedread.sh": RUN + "run_speedread.sh", "run/run_speedread_pre_a1.sh": HERE + "run_speedread_pre_a1.sh",
         "run/run_with_taniteval.py": RUN + "run_with_taniteval.py", "run/pull_speedread.sh": RUN + "pull_speedread.sh",
         "run/pull.out": W + "pull.out",
         "stops/stop1_extract_no_taniteval.log": W + "stop1/extract_no_taniteval.log", "stops/stop1_run.out": W + "stop1/run_stop1.out",
         "stops/stop2_extract_ego23.log": W + "stop2/extract_ego23.log", "stops/stop2_run.out": W + "stop2/run_stop2.out",
         "landing/mk_hold_speedread_a1.py": HERE + "mk_hold_speedread_a1.py"}
for rel, src in sorted(files.items()):
    b = rd(src)
    assert b and blob(PK + rel) is None, rel
    if rel.endswith((".py", ".sh", ".md")):
        assert b"\r" not in b, rel
    assert not re.search(rb"hf_[A-Za-z0-9]{20,}", b), rel
    put("stage/" + PK + rel, b)

GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
END = b"(25 tests; 19 / 19 mutants RED). | HYPOTHESIS | `Project Steering/PREREG_V8F_SPEEDREAD.md` |"
assert g.count(END) == 1
add = ("(25 tests; 19 / 19 mutants RED). **Amendment 1 (A1-2026-10-27), before any SPEEDREAD number:** the real cache's `limit` is 4 wide,"
       " so T's ego vector is 22 dims, not the 23 §3 counted. The first extract stopped on the broadcast after decoding one batch of 128"
       " TRAIN windows; nothing was written but a partial scene memmap, moved aside unread, and no eval batch was opened. The instrument now"
       " takes the data's width and records `d_ego` (sha256 `fd5ce0be…`): 27 tests pass, the new test fails on the promoted instrument,"
       " 19 / 19 mutants KILLED. Features, read-outs, controls and §5's table are unchanged (`raw/2026-10-27-speedread-a1/`)."
       " | HYPOTHESIS | `Project Steering/PREREG_V8F_SPEEDREAD.md` |").encode("utf-8")
for w in (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn"):
    assert w not in add.lower() and w not in am.lower(), w
g2 = g.replace(END, add)
assert len(g2.split(b"\n")) == len(g.split(b"\n"))
put("mods-rw/" + GC, g2)
put("msg.txt", ("PREREG_V8F_SPEEDREAD Amendment 1 (A1-2026-10-27), before any SPEEDREAD number: the real cache's `limit` is 4 wide, so T's ego"
                " vector is 22 dims, not the 23 the PREREG counted. The first extract stopped on the broadcast after one TRAIN batch (no number,"
                " no eval batch). The instrument now takes the data's width and records d_ego; new test (27 pass; it fails on the promoted"
                " instrument), 19 / 19 mutants KILLED. Package raw/2026-10-27-speedread-a1/ (%d files), register note.\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % len(files)).encode())
print("hold written", H, len(files))
