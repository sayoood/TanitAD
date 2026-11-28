"""Hold: PREREG_V8F_DAG_OACC Amendment 1 (A1-2026-10-27), before any DAG-OACC T1-kin rollout or value: append to the PREREG, the package
raw/2026-10-27-dagoacc-amend1/ (AMEND1.md, the enumeration that found the gap, the reader files before the amendment), register note.
python mk_hold_dagoacc_amend1.py <live tip>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
HERE = "E:/Projects/TanitAD-artifacts/dagland/dagoacc_amend1/"
H = "C:/Users/Admin/qland/hold_dagoacc_amend1/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-27-dagoacc-amend1/"
assert not os.path.exists(H), H


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


rd = lambda p: open(p, "rb").read()
am = rd(HERE + "AMEND1.md")
assert b"\r" not in am and am.startswith(b"\n## Amendment 1 (A1-2026-10-27)")
PR = "Project Steering/PREREG_V8F_DAG_OACC.md"
p = blob(PR)
assert p.endswith(b"\n") and b"## Amendment 1" not in p and b"A1-2026-10-27" not in p
put("mods/" + PR, p + am)                                   # append-only: the tip's file + the amendment
files = {"AMEND1.md": HERE + "AMEND1.md", "enumerate_cells_before.out": HERE + "enumerate_cells_before.out",
         "landing/mk_hold_dagoacc_amend1.py": HERE + "mk_hold_dagoacc_amend1.py"}
for f in os.listdir(HERE + "before"):
    files["before/" + f] = HERE + "before/" + f
for rel, src in files.items():
    assert blob(PK + rel) is None, rel
    put("stage/" + PK + rel, rd(src))

GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
old = (b"Box 2 after DAGGER2's chain; dev-box CPU for the T1-kin rollouts and reader. | HYPOTHESIS | `Project Steering/PREREG_V8F_DAG_OACC.md` |")
assert g.count(old) == 1
new = ("Box 2 after DAGGER2's chain; dev-box CPU for the T1-kin rollouts and reader. **Amendment 1 (A1-2026-10-27), before any T1-kin rollout"
       " or DAG-OACC value:** the reader's enumeration found §4 left 255 of 524,288 cells unplaced; LONGITUDINAL FIXED now reads PRIMARY-ACC"
       " and KEEP pass, LON-CLEAR at both seeds, not DOMINANT; DOMINANT also needs Read A's crawl rows to pass (L2, M1); LONGITUDINAL FIXED's"
       " next is the open row's lever (crawl rows: DAG-ORC re-based on DAG-OACC), since DAGGER2 read HARMS. Both runs trained and pulled"
       " (21:51, w_acc 49.6969). `raw/2026-10-27-dagoacc-amend1/`. | HYPOTHESIS | `Project Steering/PREREG_V8F_DAG_OACC.md` |").encode("utf-8")
for w in (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn"):
    assert w not in new.lower(), w
g2 = g.replace(old, new)
assert len(g2.split(b"\n")) == len(g.split(b"\n"))
put("mods-rw/" + GC, g2)
put("msg.txt", ("PREREG_V8F_DAG_OACC Amendment 1 (A1-2026-10-27), before any DAG-OACC T1-kin rollout or value (both runs trained and pulled,"
                " nothing read). The reader's enumeration found the section-4 table left 255 of 524,288 cells unplaced: LONGITUDINAL FIXED is"
                " made the not-DOMINANT remainder of LON-CLEAR; DOMINANT also needs Read A's crawl rows to pass (the table's own LF text names a"
                " crawl row as not dominant); LONGITUDINAL FIXED's next is re-routed because DAGGER2 read HARMS. Package"
                " raw/2026-10-27-dagoacc-amend1/ (the amendment, the enumeration before it, the reader files before it). Register note.\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n").encode())
print("hold written", H, len(files))
