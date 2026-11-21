"""Hold: PREREG_V8F_ORECOVER Amendment 4 (before any run) -- the ego probe's at-rest exemption (trainer + 3 tests + 3 mutants), the diagnosis
on box 2's gate batch, the K4 comparator rule, and the mutate_oacc anchor respelling. python mk_hold_orc_a4.py <tip> [--allow-pending-proof]
The code files come ONLY from the snapshot (E:/.../ef/snap, taken before any mutation run; R77), never from the tree."""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
PENDING = "--allow-pending-proof" in sys.argv          # land before the O-RECOVER mutation proof ends (box 2 idle); the proof is banked by a follow-up
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
EF = "E:/Projects/TanitAD-artifacts/ef/"
DG = "C:/Users/Admin/box2_ship/diag/"
H = "C:/Users/Admin/qland/hold_orc_a4/"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
CODE = ("stack/scripts/v8f_mini_train.py", "stack/tests/test_v8f_ego_perturb.py", PK + "raw/2026-10-26-orecover/mutate_orecover.py",
        PK + "raw/2026-10-25-oacc/mutate_oacc.py")
assert not os.path.exists(H), H


def blob(p, rev=TIP):
    return subprocess.run(GIT + ["show", f"{rev}:{p}"], capture_output=True, check=True).stdout


def rd(p):
    return open(p, "rb").read()


def put(rel, b):
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


# 1. code, from the snapshot, pinned by its own sha256 file, on a tip that still carries the edit's base
pins = {ln.split("*./", 1)[1]: ln.split()[0] for ln in open(EF + "snap_SHA256.txt", encoding="utf-8").read().splitlines() if "*./" in ln}
for rel in CODE:
    b = rd(EF + "snap/" + rel)
    assert hashlib.sha256(b).hexdigest() == pins[rel], f"{rel}: the snapshot moved"
    assert blob(rel) == blob(rel, "7f06ff00"), f"{rel} moved on the tip since 7f06ff00: rebase"
    assert b"\r\n" not in b, rel
    put("mods-rw/" + rel, b)
# 2. the evidence must say what the hold claims
ego = rd(EF + "out/pytest_ego_perturb.out").decode("utf-8", "replace")
oc = rd(EF + "out/pytest_orc_chain_oacc.out").decode("utf-8", "replace")
for name, t in (("ego", ego), ("orc_chain+oacc", oc)):
    last = t.strip().splitlines()[-1]
    assert re.search(r"\d+ passed", last) and "failed" not in last and "error" not in last, f"{name}: {last}"
assert "162 mutants over 6 files, each anchor found exactly once" in rd(EF + "out/mutate_mixo_check.out").decode()
evid = ["pytest_ego_perturb.out", "pytest_orc_chain_oacc.out", "mutate_mixo_check.out", "mutate_oacc_check.out", "v8f_mini_train.py.snap.vs7f06ff00.diff"]  # the snapshot (the landed bytes), not the tree a mutation run edits
for f in ("mutate_oacc_check_respelled.out", "mutate_oacc_193_194.out", "mutate_orecover.out", "snapshot_vs_restored.out"):
    done = os.path.exists(EF + "out/" + f) and (f != "mutate_orecover.out" or b"byte-identical" in rd(EF + "out/" + f))  # still being written
    if done:
        evid.append(f)
    else:
        assert PENDING, f"{f} missing (pass --allow-pending-proof to land before the proof ends)"
if "mutate_orecover.out" in evid:
    mo = rd(EF + "out/mutate_orecover.out").decode("utf-8", "replace")
    m = re.search(r"(\d+)/(\d+) RED", mo)
    assert m and m.group(1) == m.group(2) and "byte-identical: True" in mo, mo[-400:]
R = "stage/" + PK + "raw/2026-10-26-orc-a4/"
for f in evid:
    put(R + "out/" + f, rd(EF + "out/" + f))
for src, dst in ((DG + "diag_egoprobe.py", "diag_egoprobe.py"), (DG + "diag_gate_line.txt", "diag_gate_line.txt"),
                 (DG + "diag_egoprobe_s0.cmd", "diag_egoprobe_s0.cmd"), (DG + "diag_egoprobe_s0.json", "diag_egoprobe_s0.json"),
                 (DG + "out/diag_egoprobe_s0.log", "out/diag_egoprobe_s0.log"), ("C:/Users/Admin/orct1/smoke/k4_demo.out", "k4_demo.out"),
                 (EF + "run_ef.sh", "run_ef.sh"), (EF + "run_ef2.sh", "run_ef2.sh"), (EF + "mk_hold_orc_a4.py", "mk_hold_orc_a4.py"),
                 (EF + "snap_SHA256.txt", "snap_SHA256.txt"), (EF + "NOTES.md", "NOTES.md"), ("C:/Users/Admin/orcread/AMEND_ORC4.md", "AMEND_ORC4.md")):
    put(R + dst, rd(src))
d = __import__("json").load(open(DG + "diag_egoprobe_s0.json"))
assert d["n_selected"] == 29 and len(d["unchanged"]) == 1 and d["unchanged"][0]["kin3_before"][:2] == [0.0, 0.0] and d["unchanged"][0]["base_absmax_after"] == 0.0
# 3. PREREG Amendment 4 (append)
pre = blob("Project Steering/PREREG_V8F_ORECOVER.md")
a4 = rd("C:/Users/Admin/orcread/AMEND_ORC4.md")
assert b"## Amendment 4" not in pre and b"## Amendment 3" in pre and b"\r\n" not in a4
put("mods/Project Steering/PREREG_V8F_ORECOVER.md", pre + a4)
# 4. the register
g = blob("Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8")
row = [ln for ln in g.split("\n") if ln.startswith("| **H-V8F-ORECOVER-1**")]
assert len(row) == 1
cells = row[0].split(" | ")
assert len(cells) == 5 and cells[3] == "HYPOTHESIS"
cells[2] = cells[2] + (
    " **2026-10-26 12:20, before any run: the first launch gate REFUSED** (\"ego probe: base is unchanged on 1 of 29 selected rows\"), and nothing"
    " trained. MEASURED on box 2's gate batch: that row was a car at rest (v0 = a0 = 0, |base| 0 before and after), and its base cannot move."
    " **Amendment 4** exempts exactly the rows at rest from the probe's base clause and counts them (3 tests, 3 mutants). It also fixes K4's"
    " comparator to the arm's fresh-age T0 eval (MEASURED: fresh passes at 4.4e-5 m, box 2's cycle-age eval fails at 6.05 m). `raw/2026-10-26-orc-a4/`.")
put("mods-rw/Project Steering/GOALS_AND_CLAIMS.md", g.replace(row[0], " | ".join(cells)).encode("utf-8"))
put("msg.txt", ("PREREG_V8F_ORECOVER Amendment 4 (before any run): the ego probe exempts exactly the rows at rest from its base clause (box 2's"
                " gate batch: v0 = a0 = 0, base cannot move) + 3 tests + 3 mutants; K4's comparator = the fresh-age T0 eval; mutate_oacc anchors"
                " #193/#194 made unique again (evalorc repeats evaloacc's lines)" + (" [O-RECOVER mutation proof banked by a follow-up]" if PENDING and "mutate_orecover.out" not in evid else "")
                + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n").encode())
print("hold written", H, "evidence:", evid)
