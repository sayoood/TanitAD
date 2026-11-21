"""Hold: PREREG_V8F_DAG_ORC promoted from its draft (banked unchanged under raw/2026-10-27-dag-orc-draft/), + register row H-V8F-DAG-ORC-1.
Written before any DAG-ORC gate, run or row, and before N0 (the nav probe) runs.   python mk_hold_dag_orc_prereg.py <live tip>"""
import hashlib
import os
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
S = "E:/Projects/TanitAD-artifacts/dagland/dag_orc_draft/"
H = "C:/Users/Admin/qland/hold_dag_orc_prereg/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
DR = V8 + "raw/2026-10-27-dag-orc-draft/"
assert not os.path.exists(H), H


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and b"\r" not in b and len(H + rel) < 260, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


draft = open(S + "PREREG_V8F_DAG_ORC.md", "rb").read()
dsha = hashlib.sha256(draft).hexdigest()
for f in ("PREREG_V8F_DAG_ORC.md", "NOTES.md", "enumerate_cells.py", "enumerate_cells.out", "enumerate_cells_mutation.txt"):
    assert blob(DR + f) is None
    put("stage/" + DR + f, open(S + f, "rb").read().replace(b"\r\n", b"\n"))
put("stage/" + DR + "landing/mk_hold_dag_orc_prereg.py", open(__file__, "rb").read().replace(b"\r\n", b"\n"))

old_head = b"**DRAFT, 2026-10-27.** Written after `RESULT_V8F_DAGGER2` and before any DAG-ORC gate, run or row. No DAG-ORC checkpoint, rollout,\neval or floor value exists. Proposed claim row `H-V8F-DAG-ORC-1` (text in `NOTES.md`). The Master Mind promotes it; it is not landed."
assert draft.count(old_head) == 1
new_head = ("Promoted 2026-10-27 by the Master Mind from the draft banked unchanged at `raw/2026-10-27-dag-orc-draft/PREREG_V8F_DAG_ORC.md`"
            " (sha256 `%s`). Written after `RESULT_V8F_DAGGER2` and **before any DAG-ORC gate, run or row, and before N0** (§8): no DAG-ORC"
            " checkpoint, rollout, eval or floor value exists, and no `--nav logged` rollout of DAG has run. Claim row `H-V8F-DAG-ORC-1`.\n"
            "* **MM decision on the control (the draft's open question):** kept as drafted, CTRL = DAG2-0 (one lever, clean attribution). The"
            " base gap it carries (DAG2-0 trails DAG on ADE6) is what R-KEEP / R-NOBASE test, and DAG-ORC-SWAP is the registered route if A"
            " is attributed but not adopted.\n"
            "* **N0 runs first** (launch precondition 3; 0 GPU, dev-box CPU). The box-2 arms A, REP (and B) need the PI's go for box-2 time and"
            " queue after FLAT-DAG and DAG-OACC.") % dsha
promoted = draft.replace(old_head, new_head.encode("utf-8"))
P = "Project Steering/PREREG_V8F_DAG_ORC.md"
assert blob(P) is None
put("stage/" + P, promoted)

GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
i = g.index(b"| **H-V8F-DAGGER2-1** ")
j = g.index(b"\n", i) + 1
row = ("| **H-V8F-DAG-ORC-1** (2026-10-27) | Fine-tuning DAG_s for 3,000 steps with O-RECOVER's ego-state perturbation (p 0.3; v0 × U(0.5, 1.0),"
       " a0 − U(0, 1.5) m/s² where v0 < 8 m/s; yaw rate + N(0, 2.0 deg/s); targets unchanged), arm A = DAG2-0's argv + the perturbation flags,"
       " moves DAG's crawl rows at T1-kin: every row of C (s0: curvature, L2, M1; s1: + yaw-rate) clears 2 × max(F_r, RF0_r, RF_r) for"
       " A − DAG2-0 at both seeds (R-MOVE; RF0 from the banked DAG-0 / ORC-0 pair, RF from a new same-recipe replicate REP at --seed 10 / 11),"
       " with R-CRAWL deciding FIXED vs PARTIAL; guards R-TLAT / R-TLON / R-ADE not separated-worse, R-T0 <= +0.0604 m, R-NEW (no row newly"
       " worse beyond its floor); adoption R-KEEP (A − DAG ADE6 point <= +0.0604 m) and R-NOBASE; on HARMS the dose fallback B (p 0.1) is read"
       " under the same table. N0 (0 GPU) first: DAG at T1-kin with --nav logged against the banked --nav deadreckon runs, NAVMOVE = every row"
       " of C separated-better at both seeds | **PRE-REGISTERED (2026-10-27), NOT RUN.** Committed by RESULT_V8F_DAGGER2's HARMS row (the"
       " launch-recovery target on DAG states). Written before any DAG-ORC gate, run or row and before N0. Outcome table complete over 524,288"
       " cells, 8 rows, 13 / 13 mutants caught (`raw/2026-10-27-dag-orc-draft/enumerate_cells.py`). Box 2 ESTIMATED ≈ 3.0 h for A + REP"
       " (≈ 4.5 h with B), needs the PI's go; N0 runs on the dev-box CPU. | HYPOTHESIS | `Project Steering/PREREG_V8F_DAG_ORC.md` |\n")
g2 = g[:j] + row.encode("utf-8") + g[j:]
put("mods-rw/" + GC, g2)
put("msg.txt", ("PREREG_V8F_DAG_ORC promoted (the launch-recovery perturbation on DAG's states, committed by RESULT_V8F_DAGGER2's HARMS row),"
                " written before any DAG-ORC gate, run or row and before its 0-GPU nav probe N0. Draft banked unchanged at"
                " raw/2026-10-27-dag-orc-draft/ (sha256 %s); the promoted text differs only in its opening paragraph (promotion, the MM's control"
                " decision: CTRL = DAG2-0 as drafted, N0 first). New per-row replicate floors (RF0 from DAG-0 / ORC-0, RF from a new replicate)"
                " and adoption tests against DAG. Register row H-V8F-DAG-ORC-1 added after H-V8F-DAGGER2-1.\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % dsha).encode())
print("hold written", H, dsha[:12])
