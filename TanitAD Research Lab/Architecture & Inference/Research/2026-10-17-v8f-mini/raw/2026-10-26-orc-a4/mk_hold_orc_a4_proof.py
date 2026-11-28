"""Hold: the Amendment 4 proof that was still running when 853f2fc3 landed (O-RECOVER 23/23 RED; OACC #193/#194 2/2 RED; 309 anchors unique;
the tree byte-identical to the snapshot after both restores). python mk_hold_orc_a4_proof.py <tip>"""
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
EF = "E:/Projects/TanitAD-artifacts/ef/"
H = "C:/Users/Admin/qland/hold_orc_a4_proof/"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-26-orc-a4/"
assert not os.path.exists(H), H


def blob(p):
    return subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True, check=True).stdout


def exists(p):
    return subprocess.run(GIT + ["cat-file", "-e", f"{TIP}:{p}"], capture_output=True).returncode == 0


def rd(p):
    return open(p, "rb").read()


def put(rel, b):
    assert len(b) > 0, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


mo = rd(EF + "out/mutate_orecover.out").decode("utf-8", "replace")
assert re.search(r"^23/23 RED", mo, re.M) and "byte-identical: True" in mo
m2 = rd(EF + "out/mutate_oacc_193_194.out").decode("utf-8", "replace")
assert re.search(r"^2/2 RED", m2, re.M) and "byte-identical: True" in m2
assert "309 mutants over 6 files, each anchor found exactly once" in rd(EF + "out/mutate_oacc_check_respelled.out").decode()
sv = rd(EF + "out/snapshot_vs_restored.out").decode()
assert sv.count(": OK") == 4 and "DIFFERS" not in sv and sv.count("tree == ") == 5
for f in ("mutate_orecover.out", "mutate_oacc_check_respelled.out", "mutate_oacc_193_194.out", "snapshot_vs_restored.out", "v8f_mini_train.py.vs7f06ff00.diff"):
    assert not exists(R + "out/" + f), f
    put("stage/" + R + "out/" + f, rd(EF + "out/" + f))
n = blob(R + "NOTES.md").decode("utf-8")
a = ("and the full O-RECOVER proof\n  (`out/mutate_orecover.out`). The proof and the two re-run mutate_oacc anchors were still running at landing (box 2 was idle), so they are\n"
     "  banked by a follow-up commit.")
assert n.count(a) == 1
n = n.replace(a, "and the full O-RECOVER proof\n  (`out/mutate_orecover.out`): **23/23 RED**, files restored byte-identical. It and the two re-run mutate_oacc anchors"
              " (`out/mutate_oacc_193_194.out`: **2/2 RED**, restored byte-identical;\n  `out/mutate_oacc_check_respelled.out`: 309 anchors, each found once) were"
              " still running at landing (box 2 was idle), so they are banked by this follow-up commit.")
b = "  (`out/snapshot_vs_restored.out`)."
assert n.count(b) == 1
n = n.replace(b, "  (`out/snapshot_vs_restored.out`): all four snapshot files match, and the queue (not snapshotted, not edited) equals 7f06ff00's.\n"
              "  `out/v8f_mini_train.py.vs7f06ff00.diff` is the restored tree's trainer against 7f06ff00, identical to the snapshot's diff.")
put("mods-rw/" + R + "NOTES.md", n.encode("utf-8"))
put("msg.txt", ("PREREG_V8F_ORECOVER Amendment 4, the proof banked after 853f2fc3: O-RECOVER mutants 23/23 RED and OACC #193/#194 2/2 RED (both restored"
                " byte-identical), 309 OACC anchors unique, the tree equal to the snapshot\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n").encode())
print("hold written", H)
