"""Hold: PREREG_V8F_DAGGER2 Amendment 1 (the outcome table's four unplaced cells, placed before any DAGGER2 row) + the register row's note
+ the cell enumeration that proves no placed cell moves.   python mk_hold_dag2_amend1.py <live tip>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
D = "E:/Projects/TanitAD-artifacts/dagland/dag2_amend1/"
H = "C:/Users/Admin/qland/hold_dag2_amend1/"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-27-dagger2-amend1/"
assert not os.path.exists(H), H


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and b"\r" not in b and len(H + rel) < 260, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


PR = "Project Steering/PREREG_V8F_DAGGER2.md"
old = blob(PR)
am = open(D + "amend1.md", "rb").read()
assert old.endswith(b"\n") and b"Amendment 1" not in old and am.startswith(b"\n## Amendment 1")
put("mods/" + PR, old + am)

GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
anchor = b" Replicate floor fixed at DAGGER's measured values (lat 0.0067 / 0.0043, lon 0.0251 / 0.0139), disclosed as an estimate from another init. | HYPOTHESIS |"
assert g.count(anchor) == 1 and g.count(b"**H-V8F-DAGGER2-1**") == 1
note = (b" Replicate floor fixed at DAGGER's measured values (lat 0.0067 / 0.0043, lon 0.0251 / 0.0139), disclosed as an estimate from another init."
        b" **Amendment `A1-2026-10-27`, written before any DAGGER2 row exists:** the outcome table left 4 of its 16 guarded cells unplaced (R-MOVE"
        b" at exactly one seed, R-CRAWL failing at that seed); they are PARTIAL now, and none of the 12 placed cells moves"
        b" (`raw/2026-10-27-dagger2-amend1/enumerate_cells.py`). A one-seed R-MOVE adopts nothing and is never called an effect of the lever."
        b" Box 2: bank s0 / s1 built; DAG2 s0, DAG2-0 s0 and DAG2 s1 trained and pulled; DAG2-0 s1 training. | HYPOTHESIS |")
g2 = g.replace(anchor, note)
assert len(g2.split(b"\n")) == len(g.split(b"\n"))
put("mods-rw/" + GC, g2)

for f in ("enumerate_cells.py", "enumerate_cells.out", "mk_hold_dag2_amend1.py", "amend1.md"):
    assert blob(PK + f) is None
    put("stage/" + PK + f, open(D + f, "rb").read().replace(b"\r\n", b"\n"))

put("msg.txt", ("PREREG_V8F_DAGGER2 Amendment 1 (A1-2026-10-27), written before any DAGGER2 row exists: the outcome table, which says it is"
                " complete, left 4 of its 16 guarded (R-CRAWL, R-MOVE) x 2-seed cells unplaced (R-MOVE at exactly one seed with R-CRAWL failing"
                " there). The guarded cells are now partitioned on R-MOVE: FIXED 1, PARTIAL 11, NO EFFECT 4; none of the 12 placed cells moves"
                " (enumerate_cells.py, rc 0). PARTIAL's committed next runs on the base's states; the change-3 bars enter no outcome and are"
                " reported beside; a one-seed R-MOVE adopts nothing (H-ESTIM-SEED-1). Found by the DAGGER2 reader's cell-by-cell coding."
                " No T1-kin rollout of DAG2 / DAG2-0 exists; no T0 eval file was opened. Register row H-V8F-DAGGER2-1 carries the note.\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n").encode())
print("hold written", H)
