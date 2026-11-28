"""Hold: PREREG_V8F_DAG_OACC (new) + PREREG_V8F_FLATDAG (promoted from the 5eaa0232 draft), both with the PI's go "yes queue it", before any
of their weight values, banks, runs or rows; register rows H-V8F-DAG-OACC-1 and H-V8F-FLATDAG-1.  python mk_hold_oacc_flatdag_prereg.py <tip>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
A = "E:/Projects/TanitAD-artifacts/"
H = "C:/Users/Admin/qland/hold_oacc_flatdag_prereg/"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
assert not os.path.exists(H), H


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


def stage(rel, b):
    assert blob(rel) is None, f"{rel} exists on the tip"
    put("stage/" + rel, b)


def lf(p):
    b = open(p, "rb").read().replace(b"\r\n", b"\n")
    assert b"\r" not in b
    return b


oacc = lf(A + "dagland/PREREG_V8F_DAG_OACC.md")
flat = lf(A + "dagland/PREREG_V8F_FLATDAG.md")
assert oacc.startswith(b"# PREREG_V8F_DAG_OACC") and flat.startswith(b"# PREREG_V8F_FLATDAG:")
assert all(b'*"yes queue it"*' in x for x in (oacc, flat))
assert blob(PK + "raw/2026-10-27-flatdag-draft/PREREG_V8F_FLATDAG.md") is not None
stage("Project Steering/PREREG_V8F_DAG_OACC.md", oacc)
stage("Project Steering/PREREG_V8F_FLATDAG.md", flat)
stage(PK + "raw/2026-10-27-dagger2-prereg/mk_hold_oacc_flatdag_prereg.py", lf(A + "dagland/mk_hold_oacc_flatdag_prereg.py"))

g = blob("Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8")
row = [ln for ln in g.split("\n") if ln.startswith("| **H-V8F-DAGGER2-1**")]
assert len(row) == 1 and "| **H-V8F-DAG-OACC-1**" not in g and "| **H-V8F-FLATDAG-1**" not in g
r_oacc = " | ".join([
    "| **H-V8F-DAG-OACC-1** (2026-10-27)",
    "OACC's acceleration-matching term (--w-acc W, W by OACC's rule with B = DAG: 0.5 L_traj(DAG_s0) / L_acc(DAG_s0) over HBLEND's fit"
    " windows), added to DAG's fine-tune recipe as its ONE lever, lowers accel MAE against DAG (separated, both seeds) without costing"
    " position (DAG-OACC - DAG ADE6 upper bound <= +F/2 = 0.0604 m), and leaves no LONGITUDINAL row separated-worse against P10 at T1-kin;"
    " DOMINANT if Read A's D1 / D2 / D3 also hold at both seeds",
    "**PRE-REGISTERED (2026-10-27), NOT RUN.** RESULT_V8F_DAG_HACC's committed next (COSTS POSITION: the post-map cleared the longitudinal"
    " rows and failed KEEP at s0, +0.1264 m). PI go 2026-10-27, verbatim: *\"yes queue it\"*. Written before any DAG-OACC weight value, gate,"
    " run or row. Control = the banked DAG run (one-flag pair). Box 2 after DAGGER2's chain; dev-box CPU for the T1-kin rollouts and reader.",
    "HYPOTHESIS",
    "`Project Steering/PREREG_V8F_DAG_OACC.md` |",
])
r_flat = " | ".join([
    "| **H-V8F-FLATDAG-1** (2026-10-27)",
    "With own-state supervision on BOTH sides (FLAT-DAG = FLAT_s fine-tuned with DAG's DAgger recipe on FLAT's own bank, --o-interface cut"
    " kept), the hierarchy still beats flat at T1-kin: DAG - FLAT-DAG ADE6 <= -2F = -0.2415 m, separated, at both seeds, with no Read-A family"
    " row separated-worse",
    "**PRE-REGISTERED (2026-10-27), NOT RUN.** Promoted from the draft banked at 5eaa0232 (written before any DAG_REREAD row). Run condition:"
    " B-2 read ESTABLISHED, under PREREG_V8F_DAG_REREAD's Amendment 1 (post-data, control-only; disclosed). PI go 2026-10-27, verbatim:"
    " *\"yes queue it\"*. Written before any FLAT-DAG bank, run or row. Launch blocker until the gate shows --o-interface cut and --dagger-*"
    " composing (cut and swap reached in the same run). Box 2 after DAG-OACC.",
    "HYPOTHESIS",
    "`Project Steering/PREREG_V8F_FLATDAG.md` |",
])
g2 = g.replace(row[0], row[0] + "\n" + r_oacc + "\n" + r_flat)
assert g2.count("| **H-V8F-DAG-OACC-1**") == 1 and g2.count("| **H-V8F-FLATDAG-1**") == 1
put("mods-rw/Project Steering/GOALS_AND_CLAIMS.md", g2.encode("utf-8"))
put("msg.txt", ("PREREG_V8F_DAG_OACC (DAG-HACC's committed next: OACC's acceleration term inside DAG's fine-tune, one lever, control = banked DAG,"
                " W by OACC's rule with B = DAG) and PREREG_V8F_FLATDAG (promoted from the 5eaa0232 draft: FLAT given its own DAgger bank and"
                " DAG's recipe, the attribution read for B-2). Both with the PI's go (\"yes queue it\"), before any of their weight values,"
                " banks, runs or rows. Register: H-V8F-DAG-OACC-1, H-V8F-FLATDAG-1 new\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n").encode())
print("hold written", H)
