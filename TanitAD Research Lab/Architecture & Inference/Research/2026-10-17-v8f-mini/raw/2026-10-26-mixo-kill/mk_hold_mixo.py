"""Hold: the MIXO reader's mutation proof made whole (14 killing tests + the mutant anchors) and its re-check on the final trainer bytes.
python mk_hold_mixo.py <tip>. Run AFTER the OACC Amendment 2 hold has landed (the re-check tree carries its trainer)."""
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
K = "C:/Users/Admin/mixo_kill/"
X = "E:/Projects/TanitAD-artifacts/mx/"
H = "C:/Users/Admin/qland/hold_mixo_kill/"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
TEST, MUT, TRAIN = "stack/tests/test_v8f_mixo.py", PK + "raw/2026-10-25-mixo/mutate_mixo.py", "stack/scripts/v8f_mini_train.py"
assert not os.path.exists(H), H


def blob(p, rev=TIP):
    return subprocess.run(GIT + ["show", f"{rev}:{p}"], capture_output=True, check=True).stdout


def put(rel, b):
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


def rd(p):
    return open(p, "rb").read()


# the two edited files: the tip still carries the bytes the proof started from, and the proof's tree carries the edit
assert blob(MUT) == rd(K + "orig/mutate_mixo.py"), "the mutator moved on the tip: rebase"
ORC = '{"orcarm", "gateorc", "trainorc", "evalorc"}'
WIDEN = [('assert set(blocks) - FLAT_LABELS - OACC_LABELS == set(_K.PRE_EXISTING)', 'assert set(blocks) - FLAT_LABELS - OACC_LABELS - ORC_LABELS == set(_K.PRE_EXISTING)'),
         ('OACC_LABELS = {"waccoacc", "oaccwacc", "gateoacc", "trainoacc", "evaloacc", "fitoacc"}', 'ORC_LABELS = ' + ORC + '           # PREREG_V8F_ORECOVER appended these (test_v8f_orc_chain.py pins them); widened here, nothing else\nOACC_LABELS = {"waccoacc", "oaccwacc", "gateoacc", "trainoacc", "evaloacc", "fitoacc"}')]


def widen(b):                                     # the test_v8f_mixo.py entry of orcchain/tools/widen_pins.py, verbatim
    s = b.decode("utf-8")
    for a, c in WIDEN:
        assert s.count(a) == 1, a[:70]
        s = s.replace(a, c)
    return s.encode("utf-8")


WIDENED = blob(TEST) != rd(K + "orig/test_v8f_mixo.py")
if WIDENED:                                       # O-RECOVER landed first: its pin widening must be the tip's only change to this test
    assert widen(rd(K + "orig/test_v8f_mixo.py")) == blob(TEST), "the tip's test is not orig + the O-RECOVER widening: rebase by hand"
for rel in (TEST, MUT):
    b = rd(K + "tree/" + rel)
    assert b"\r\n" not in b and b == rd(X + "tree/" + rel), rel
    put("mods-rw/" + rel, widen(b) if WIDENED and rel == TEST else b)
# the re-check ran on the trainer that is now on the tip (OACC Amendment 2 landed first)
assert blob(TRAIN) == rd(X + "tree/" + TRAIN), "the re-check's trainer is not the tip's: land OACC A2 first, or re-run"
# the evidence must say what this hold claims
proof = rd(K + "out/mutation_proof_kill.out").decode("utf-8", "replace")
assert "162/162 RED" in proof and "byte-identical: True" in proof and "over the tip 6df75ad202b70aa22c3acadc99dfba4de94ef92f" in proof
chk = rd(X + "out/mutate_mixo_check.out").decode()
assert "162 mutants over 6 files, each anchor found exactly once" in chk
tr = rd(X + "out/mutate_mixo_train.out").decode("utf-8", "replace")
m = re.search(r"(\d+)/(\d+) RED", tr)
assert m and m.group(1) == m.group(2) and "byte-identical: True" in tr, "trainer mutants: not all RED, or not restored"
n_tr = int(m.group(2))
cc = rd(X + "out/coupled_check.out").decode()
assert re.search(r"\b2 passed\b", cc) and "error" not in cc.lower().split("passed")[-1], cc[-200:]
R = "stage/" + PK + "raw/2026-10-26-mixo-kill/"
for src, dst in ((K + "out/mutation_proof_kill.out", "out/mutation_proof_kill_6df75ad2.out"), (K + "out/coupled_check.out", "out/coupled_check_6df75ad2.out"),
                 (K + "out/test_v8f_mixo.diff", "out/test_v8f_mixo.diff"), (K + "out/mutate_mixo.diff", "out/mutate_mixo.diff"),
                 (K + "run_proof.cmd", "run_proof.cmd"), (K + "wait_no_pytest.ps1", "wait_no_pytest.ps1"), (K + "move_proof_to_e.py", "move_proof_to_e.py"),
                 (X + "run_mx.sh", "run_mx.sh"), ("E:/Projects/TanitAD-artifacts/oc/run_oc.sh", "run_oc.sh"), ("E:/Projects/TanitAD-artifacts/tv/run_tv.sh", "run_tv.sh"), ("E:/Projects/TanitAD-artifacts/tv/run_tv2.sh", "run_tv2.sh"), (X + "mk_hold_mixo.py", "mk_hold_mixo.py"), (X + "NOTES.md", "NOTES.md"),
                 (X + "out/tree_files.out", "out/recheck_tree_files.out"), (X + "out/mutate_mixo_check.out", "out/recheck_mutate_mixo_check.out"),
                 (X + "out/mutate_mixo_train.out", "out/recheck_mutate_mixo_train.out"), (X + "out/coupled_check.out", "out/recheck_coupled_check.out")):
    put(R + dst, rd(src))
for f in sorted(os.listdir(X + "out/void_r77")):                       # R77: the void first re-check, banked as a record
    put(R + "out/void_r77/" + f, rd(X + "out/void_r77/" + f))
t7 = rd(X + "out/on_7f06ff00/mutate_mixo_train.out").decode("utf-8", "replace")     # the second re-check, on 7f06ff00's trainer (before Amendment 4)
assert re.search(r"^15/15 RED", t7, re.M) and "byte-identical: True" in t7
for f in sorted(os.listdir(X + "out/on_7f06ff00")):
    put(R + "out/recheck_on_7f06ff00/" + f, rd(X + "out/on_7f06ff00/" + f))
g = blob("Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8")
row = [ln for ln in g.split("\n") if ln.startswith("| **H-V8F-MIXO-1**")]
assert len(row) == 1
cells = row[0].split(" | ")
note = (f" **2026-10-26: the reader's mutation proof is now whole, 162/162 RED** (14 killing tests, among them the −2F edge and the O-LEARNED"
        f" branch that had survived; files restored byte-identical). It ran on 6df75ad2. O-RECOVER, OACC Amendment 2 and O-RECOVER Amendment 4 changed"
        f" the trainer after that, so its {n_tr} trainer mutants were re-run: on 7f06ff00 (15/15 RED; an earlier re-check had run on a mutated trainer copy and is banked as a record, R77), and again on the"
        f" tip's trainer after Amendment 4: {n_tr}/{n_tr} RED, every anchor still found once. The banked verdict"
        f" does not move: it had been re-derived by hand. `raw/2026-10-26-mixo-kill/`.**")
assert len(cells) == 3 and cells[2].endswith(" |"), "row layout: id | claim | status |"
cells[2] = cells[2][:-2] + note + " |"
put("mods-rw/Project Steering/GOALS_AND_CLAIMS.md", g.replace(row[0], " | ".join(cells)).encode("utf-8"))
put("msg.txt", f"MIXO reader mutation proof made whole: 14 killing tests + mutant anchors, 162/162 RED on 6df75ad2; the {n_tr} trainer mutants"
    f" re-run on the tip's trainer after O-RECOVER Amendment 4, {n_tr}/{n_tr} RED (and on 7f06ff00's before it, 15/15). The verdict does not move.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n".encode())
print("hold written", H, "trainer mutants", n_tr)
