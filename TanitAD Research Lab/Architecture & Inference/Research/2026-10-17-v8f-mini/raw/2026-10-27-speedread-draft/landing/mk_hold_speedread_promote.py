"""Hold: promote PREREG_V8F_SPEEDREAD (draft sha256 2ea672da..., written before any SPEEDREAD number) + its instrument and self-test into
stack/ + the draft package raw/2026-10-27-speedread-draft/ + register row H-V8F-SPEEDREAD-1 (PRE-REGISTERED, NOT RUN).
python mk_hold_speedread_promote.py <live tip>"""
import hashlib
import os
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
D = "E:/Projects/TanitAD-artifacts/speedread_draft/"
HERE = "E:/Projects/TanitAD-artifacts/dagland/speedread_promote/"
H = "C:/Users/Admin/qland/hold_speedread_promote/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-27-speedread-draft/"
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

# the draft, byte-exact, and every file its MANIFEST pins
man = {}
for line in rd(D + "MANIFEST_SHA256.txt").decode().splitlines():
    h, f = line.split(" *", 1)
    man[f] = h
for f, h in man.items():
    assert sha(rd(D + f)) == h, f
    assert b"\r" not in rd(D + f), f
dr = rd(D + "PREREG_V8F_SPEEDREAD.md")
assert sha(dr) == "2ea672dabb644a0b4d7c41c0fdb94092678535ca74de8f66bebce64a530d9141"
for f in list(man) + ["MANIFEST_SHA256.txt"]:
    assert blob(PK + f) is None, f
    put("stage/" + PK + f, rd(D + f))
put("stage/" + PK + "landing/mk_hold_speedread_promote.py", rd(HERE + "mk_hold_speedread_promote.py"))
put("stage/" + PK + "landing/selftest_mm_rerun.txt", rd(HERE + "selftest_mm_rerun.txt"))

# the instrument and its test into stack/ (byte-identical to the pinned draft files)
for f in ("stack/scripts/v8f_mini_eval_speedread.py", "stack/tests/test_v8f_speedread.py"):
    assert blob(f) is None, f
    put("stage/" + f, rd(D + f))

# the promoted PREREG: the draft with its title and status line changed, everything else verbatim
PS = "Project Steering/PREREG_V8F_SPEEDREAD.md"
assert blob(PS) is None
t = dr.decode("utf-8")
a = "# PREREG_V8F_SPEEDREAD (DRAFT): is the per-window speed information O needs present in T's own INPUT?\n"
b = ("**Status: DRAFT, not promoted.** Written 2026-10-27, before any SPEEDREAD number. No real train or eval feature has been opened by this\n"
     "work; the instrument has run on synthetic data only. Claim row to file on promotion: `H-V8F-SPEEDREAD-1`.")
assert t.count(a) == 1 and t.count(b) == 1
t = t.replace(a, "# PREREG_V8F_SPEEDREAD: is the per-window speed information O needs present in T's own INPUT?\n")
t = t.replace(b, ("**Status: PRE-REGISTERED, NOT RUN.** Promoted 2026-10-27 by the Master Mind from the draft banked at\n"
                  "`raw/2026-10-27-speedread-draft/PREREG_V8F_SPEEDREAD.md` (sha256 `2ea672da…`), before any SPEEDREAD number: no real train or eval\n"
                  "feature has been opened; the instrument has run on synthetic data only (25 tests, re-run by the MM on a fresh export of `95063711`:\n"
                  "25 passed). Only this status line and the title differ from the draft. Claim row `H-V8F-SPEEDREAD-1`."))
assert "\r" not in t
put("stage/" + PS, t.encode("utf-8"))

# the register row, inserted after H-V8F-FLATDAG-1
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
assert g.count(b"**H-V8F-SPEEDREAD-1**") == 0 and g.count(b"| **H-V8F-FLATDAG-1**") == 1
i = g.index(b"| **H-V8F-FLATDAG-1**")
j = g.index(b"\n", i)
row = ("| **H-V8F-SPEEDREAD-1** (2026-10-27) | A read-out of the 5 GT speed cells computed from T's own input only (its 4 tick frames of the cached"
       " `pool4x10` grid plus the ego-motion history T reads), fitted on the 4,512 TRAIN clips only and substituted for T's 5 speed cells in the"
       " code O reads, cuts X_s's own-code ADE6 on eval-141 by at least F = 0.1207 m, separated, at both seeds (X_s = M2b-mixo_s, the KINDX /"
       " SPEEDCAL base) | **PRE-REGISTERED, NOT RUN (2026-10-27).** Promoted from the draft (`raw/2026-10-27-speedread-draft/`, sha256"
       " 2ea672da…) before any SPEEDREAD number; commissioned by `RESULT_V8F_FLATDAG.md` §4 (a), the box-2-feasible form of the T-side speed"
       " lever while perception waits for the A40. Zero training of the driving model; dev-box CPU, ~1.5–2.5 h ESTIMATED, after a 94.27 GB pull"
       " of the train cache from box 2 while it is idle between chains. Tier: T1-class open loop (the tier of SPEEDCAL's and KINDX's ADE6 rows;"
       " the `gt` ceiling is T0); paired episode-cluster bootstrap, B = 2,000, 141 episodes. Outcomes: INFO IN INPUT / PARTIAL → the box-2 arm"
       " SPEEDHEAD (sub-head to build, launch-gate PASS and the PI's go); NOT IN INPUT → perception stays the lever, blocked on the A40; MIXED →"
       " a T1-kin Stage Z decides. Instrument `stack/scripts/v8f_mini_eval_speedread.py`, `stack/tests/test_v8f_speedread.py` (25 tests;"
       " 19 / 19 mutants RED). | HYPOTHESIS | `Project Steering/PREREG_V8F_SPEEDREAD.md` |").encode("utf-8")
g2 = g[: j + 1] + row + b"\n" + g[j + 1:]
assert len(g2.split(b"\n")) == len(g.split(b"\n")) + 1
for w in (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn"):
    assert w not in row.lower(), w
put("mods-rw/" + GC, g2)
put("msg.txt", ("PREREG_V8F_SPEEDREAD promoted (H-V8F-SPEEDREAD-1, PRE-REGISTERED, NOT RUN), before any SPEEDREAD number. The cheapest"
                " discriminating experiment for the T-side speed lever (RESULT_V8F_FLATDAG s4 (a)): a read-out of the GT speed cells from T's own"
                " cached input, fitted on TRAIN only, substituted into the code O reads; INFO IN INPUT routes to a box-2 T-side arm, NOT IN INPUT"
                " keeps perception (A40) as the lever. Instrument stack/scripts/v8f_mini_eval_speedread.py + stack/tests/test_v8f_speedread.py"
                " (25 tests, re-run by the MM: 25 passed; 19/19 mutants RED). Draft package raw/2026-10-27-speedread-draft/. Register row added.\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n").encode())
print("hold written", H)
