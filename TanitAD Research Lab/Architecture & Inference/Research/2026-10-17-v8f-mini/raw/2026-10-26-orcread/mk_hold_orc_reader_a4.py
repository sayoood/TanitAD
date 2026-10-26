"""Hold: the O-RECOVER reader at Amendment 4 (K4 reads `t0_fresh`; T0-AGES and T0F-CKPT controls; 4 tests, 5 mutants) and the dev-box T1 chain
that feeds it (orct1: waits for the four pulls, runs the T1-kin rollouts and the fresh-age T0 evals, writes runs.json, runs the reader).
python mk_hold_orc_reader_a4.py <tip>. The reader files come from the A4 snapshot (snap_a4_SHA256.txt, taken before the mutation proof; R77)."""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
OR = "C:/Users/Admin/orcread/"
OT = "C:/Users/Admin/orct1/"
H = "C:/Users/Admin/qland/hold_orc_reader_a4/"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
RD = PK + "raw/2026-10-26-orcread/"
TD = PK + "raw/2026-10-26-orct1/"
assert not os.path.exists(H), H


def blob(p):
    return subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True, check=True).stdout


def exists(p):
    return subprocess.run(GIT + ["cat-file", "-e", f"{TIP}:{p}"], capture_output=True).returncode == 0


def rd(p):
    return open(p, "rb").read()


def put(rel, b):
    assert len(b) > 0, rel
    assert len(H + rel) < 260 and len(rel) <= 220, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


def stage(rel, b):
    assert not exists(rel), f"{rel} exists on the tip"
    put("stage/" + rel, b)


# 1. the reader: the A4 files, pinned by the snapshot taken before the proof, on a tip that still carries the edit's base
pins = {ln.split(" *", 1)[1]: ln.split()[0] for ln in rd(OR + "snap_a4_SHA256.txt").decode().splitlines() if " *" in ln}
for f in ("orc_read.py", "test_orc_read.py", "mutate_orc.py"):
    b = rd(OR + f)
    assert hashlib.sha256(b).hexdigest() == pins[f], f"{f}: moved since the snapshot"
    assert blob(RD + f) == rd(OR + f + ".pre_a4"), f"{f}: the tip is not the A4 edit's base"
    assert b"\r\n" not in b, f
    put("mods-rw/" + RD + f, b)
n = rd(OR + "NOTES.md")
assert n.startswith(blob(RD + "NOTES.md")) and b"## Amendment 4" in n[len(blob(RD + "NOTES.md")):]
put("mods/" + RD + "NOTES.md", n)
a4 = rd(OR + "PREREG_V8F_ORECOVER.A4.md")
assert a4 == blob("Project Steering/PREREG_V8F_ORECOVER.md") and hashlib.sha256(a4).hexdigest() == pins["PREREG_V8F_ORECOVER.A4.md"]
suite = rd(OR + "out/pytest_full_a4.out").decode()
assert re.search(r"^57 passed", suite.strip().splitlines()[-1])
mut = rd(OR + "out/mutation_a4.out").decode()
assert "# 30 of 30 mutants KILLED; survived: none" in mut
assert rd(OR + "out/snap_a4_check.out").decode().count(": OK") == 5
for f in ("PREREG_V8F_ORECOVER.A4.md", "apply_a4.py", "run_a4_tests.sh", "snap_a4_SHA256.txt", "mk_hold_orc_reader_a4.py",
          "out/pytest_full_a4.out", "out/mutation_a4.out", "out/mutation_a4.log", "out/snap_a4_check.out"):
    stage(RD + f, rd(OR + f))
# 2. the dev-box T1 chain (new package dir)
sel = rd(OT + "smoke/selftest_output_a4.txt").decode()
assert "40/40 as expected" in sel
for f in ("orc_t1_chain.sh", "orc_t1_chain.cmd", "orct1_pullcheck.py", "orct1_manifest.py", "orct1_selftest.py", "NOTES.md", "HANDOFF.md"):
    b = rd(OT + f)
    assert f.endswith(".cmd") or b"\r\n" not in b, f
    stage(TD + f, b)
assert b"t0_use" not in rd(OT + "orct1_manifest.py") and b"T0SEL" not in rd(OT + "orc_t1_chain.sh") and b"rt4/tree" in rd(OT + "orc_t1_chain.sh")
for f in sorted(os.listdir(OT + "smoke")):                           # the smoke evidence: top-level text and scripts (data dirs stay local)
    p = OT + "smoke/" + f
    if os.path.isfile(p) and os.path.getsize(p) > 0:
        stage(TD + "smoke/" + f, rd(p))
for d in ("logs", "logs_t0f", "halfpull/logs"):
    for f in sorted(os.listdir(OT + "smoke/" + d)):
        p = OT + "smoke/" + d + "/" + f
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            stage(TD + "smoke/" + d + "/" + f, rd(p))
for f in ("halfpull/test.out", "halfpull/resume_changed_ckpt.stdout"):
    if os.path.exists(OT + "smoke/" + f):
        stage(TD + "smoke/" + f, rd(OT + "smoke/" + f))
for root, _, files in os.walk(OT + "pre_a4"):
    for f in files:
        stage(TD + "pre_a4/" + f, rd(os.path.join(root, f)))
# 3. the register
g = blob("Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8")
row = [ln for ln in g.split("\n") if ln.startswith("| **H-V8F-ORECOVER-1**")]
assert len(row) == 1
cells = row[0].split(" | ")
assert len(cells) == 5 and cells[3] == "HYPOTHESIS"
cells[2] = cells[2] + (
    " **2026-10-26 ~13:05: all four launch gates PASS on box 2** (commit 853f2fc3; ORC s0's probe exempted exactly 1 row at rest), and ORC s0"
    " is training. **The reader is at A4:** K4 reads `t0_fresh` (the `--ages fresh` eval), and R-T0 reads box 2's standard eval. The T0-AGES and"
    " T0F-CKPT controls keep them apart: 57 tests, 30/30 mutants. The dev-box T1 chain (`raw/2026-10-26-orct1/`) always runs the fresh eval:"
    " self-test 40/40. Built, not launched.")
put("mods-rw/Project Steering/GOALS_AND_CLAIMS.md", g.replace(row[0], " | ".join(cells)).encode("utf-8"))
put("msg.txt", ("O-RECOVER reader at Amendment 4: K4 reads t0_fresh (the --ages fresh T0 eval), R-T0 / T0-CKPT / the families stay on box 2's standard"
                " eval; T0-AGES + T0F-CKPT controls; 57 passed, 30/30 mutants KILLED. + the dev-box T1 chain (orct1): always runs the fresh-age"
                " eval, no switch repoints t0; self-test 40/40\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n").encode())
print("hold written", H)
