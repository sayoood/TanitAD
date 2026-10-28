"""Hold: PREREG_V8F_SPEEDSUB promoted (before any SPEEDSUB number), its instrument mode (mods-rw), new tests + families runner, the draft
package, and the register row (append).  python mk_hold_speedsub.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
S = "E:/Projects/TanitAD-artifacts/dagland/speedsub/"
H = "C:/Users/Admin/qland/hold_speedsub/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-28-speedsub-draft/"
assert not os.path.exists(H), H


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, (rel, len(H + rel))
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


rd = lambda p: open(p, "rb").read()
sha = lambda b: hashlib.sha256(b).hexdigest()
BANNED = (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn")

# ---- the instrument (rewrite of a tracked file) and the new stack files
INS = "stack/scripts/v8f_mini_eval_speedread.py"
ib = rd(S + "stage/" + INS)
assert sha(ib) == "a95f4fb14da3fd8f6f89e55dfcb23d5d26a81b09dd24285aba394f36546634ca" and b"\r" not in ib
assert sha(blob(INS)).startswith("fd5ce0bea37c3e0e"), "the tip's instrument is not SPEEDREAD A1's"
put("mods-rw/" + INS, ib)
assert rd(S + "stage/stack/tests/test_v8f_speedread.py") == blob("stack/tests/test_v8f_speedread.py")      # unchanged: not banked again
assert sha(rd(S + "stage/stack/tests/golden/v8f_mini_eval_speedread_fd5ce0be.py")).startswith("fd5ce0bea37c3e0e")
NEW = ["stack/tests/test_v8f_speedsub.py", "stack/tests/make_golden_speedread.py",
       "stack/tests/golden/golden_speedread_modes.json", "stack/tests/golden/v8f_mini_eval_speedread_fd5ce0be.py"]
for rel in NEW:
    b = rd(S + "stage/" + rel)
    assert b"\r" not in b and blob(rel) is None, rel
    put("stage/" + rel, b)

# ---- the PREREG, promoted
d = rd(S + "PREREG_V8F_SPEEDSUB.md")
assert b"\r" not in d
OLD = b"**Status: PRE-REGISTERED DRAFT, NOT RUN.** Drafted 2026-10-28 for the Master Mind to promote."
assert d.count(OLD) == 1
p = d.replace(OLD, ("**Status: PRE-REGISTERED, NOT RUN (2026-10-28).** Promoted by the Master Mind from the draft (`raw/2026-10-28-speedsub-draft/`,"
                    " sha256 " + sha(d)[:8] + "\u2026) before any SPEEDSUB number; the Master Mind rewrote one sentence of \u00a72 before promotion"
                    " (the `limit` admissibility: `limit` is in v8F's own `INFERENCE_INPUT_KEYS`).").encode("utf-8"))
for w in BANNED:
    assert w not in p.lower(), w
PR = "Project Steering/PREREG_V8F_SPEEDSUB.md"
assert blob(PR) is None
put("stage/" + PR, p)

# ---- the draft package
n = 0
for root, label in ((S, ""),):
    for dp, _, fs in os.walk(root):
        for f in fs:
            pth = os.path.join(dp, f).replace("\\", "/")
            rel = pth[len(root):]
            if "__pycache__" in rel or rel.startswith("stage/") or f.endswith((".pyc", ".pt", ".npz")):
                continue
            b = rd(pth)
            if rel.endswith((".py", ".sh", ".md")):
                assert b"\r" not in b, rel
            assert b and len(b) < 400_000 and not re.search(rb"hf_[A-Za-z0-9]{20,}", b), rel
            assert blob(PK + rel) is None, rel
            put("stage/" + PK + rel, b)
            n += 1
put("stage/" + PK + "mk_hold_speedsub.py", rd("E:/Projects/TanitAD-artifacts/dagland/speedsub_land/mk_hold_speedsub.py")); n += 1

# ---- the register row (append after the last row, which is SPEEDREAD's)
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
assert g.endswith(b"\n") and g.rstrip(b"\n").split(b"\n")[-1].startswith(b"| **H-V8F-SPEEDREAD-1** (2026-10-27)")
row = ("| **H-V8F-SPEEDSUB-1** (2026-10-28) | Replacing only the five speed cells O reads by the banked ego-only read-out `mlp_ego` (SPEEDREAD's"
       " fit, `readout.pt` sha256 `a205609a\u2026`, no re-fit; it reads T's own non-visual inference inputs at the tick: P10 kinematics, nav, the"
       " speed-limit bucket, frame validity) cuts X_s's own-code ADE6 on eval-141 by at least F/2 = 0.06035 m, separated, at both seeds"
       " (X_s = M2b-mixo_s, SPEEDREAD's base) | **PRE-REGISTERED, NOT RUN (2026-10-28).** Commissioned by `RESULT_V8F_SPEEDREAD.md` \u00a75"
       " (the `mlp_ego` substitution was not run); the arm was chosen after SPEEDREAD's beside table was read, so the selection rule is"
       " restated on TRAIN fold 0 only (control S13) and a PASS is a screening result. Zero training; dev-box CPU, \u2264 4 threads, about"
       " 1.3\u20132.7 CPU-h for the decision (ESTIMATED). Tier: T1-class open loop (the `gt` ceiling is T0, a control); paired episode-cluster"
       " bootstrap, B = 2,000, 141 episodes. Controls S1\u2013S13 incl. a deliberate-regression arm (`egoshuf`, must read separated-worse at"
       " both seeds). Outcomes: PASS \u2192 a T1-kin zero-training substitution on the DAG-OACC / DAG-ORC base under its own PREREG, after a"
       " nav-free re-fit attributes the gain; NULL or HARMS \u2192 PREREG_V8F_SPEEDCELL (per-cell attribution of the 0.7 m `gt` ceiling)."
       " Instrument `stack/scripts/v8f_mini_eval_speedread.py` (sha256 `a95f4fb1\u2026`; modes `ego`, `egoshuf`; earlier modes bit-identical),"
       " `stack/tests/test_v8f_speedsub.py` (39 tests; RED 31 / 39 on the earlier instrument; 28 / 28 mutants killed)."
       " `Project Steering/PREREG_V8F_SPEEDSUB.md`, `raw/2026-10-28-speedsub-draft/`. | PRE-REGISTERED | `Project Steering/PREREG_V8F_SPEEDSUB.md` |\n")
rb_ = row.encode("utf-8")
for w in BANNED:
    assert w not in rb_.lower(), w
put("mods/" + GC, g + rb_)
put("msg.txt", ("PREREG_V8F_SPEEDSUB promoted before any SPEEDSUB number: the banked ego-only read-out mlp_ego substituted into the five speed"
                " cells O reads (zero training, dev-box CPU, T1-class open loop, PASS at <= -F/2 separated at both seeds; NULL/HARMS -> SPEEDCELL)."
                " Instrument modes ego / egoshuf (earlier modes bit-identical), 39 tests (31 RED on the earlier instrument), 28/28 mutants killed."
                " Register row H-V8F-SPEEDSUB-1. Draft package raw/2026-10-28-speedsub-draft/ (%d files).\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % n).encode())
print("hold written", H, n, "package files")
