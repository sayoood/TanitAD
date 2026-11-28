"""Hold: PREREG_V8F_DAG_ORC's N0 (the nav probe) read: NOT NAVMOVE. Package raw/2026-10-27-dag-orc-n0/ (no dumps / ticks: they stay on E:)
+ the register row H-V8F-DAG-ORC-1's status.   python mk_hold_dag_orc_n0.py <live tip>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
S = "E:/Projects/TanitAD-artifacts/dag-orc-n0/"
H = "C:/Users/Admin/qland/hold_dag_orc_n0/"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-27-dag-orc-n0/"
assert not os.path.exists(H), H


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


n = 0
for root, dirs, files in os.walk(S):
    dirs[:] = [x for x in dirs if not x.startswith(("dump_", "ticks"))]
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), S).replace("\\", "/")
        if rel.startswith("PREREG_V8F_DAG_ORC_"):
            continue                                          # the landed PREREG (85d0b125) is in the repo
        b = open(S + rel, "rb").read()
        if rel.endswith((".py", ".sh", ".md")):
            b = b.replace(b"\r\n", b"\n")
        assert blob(PK + rel) is None, rel
        put("stage/" + PK + rel, b); n += 1
put("stage/" + PK + "landing/mk_hold_dag_orc_n0.py", open(__file__, "rb").read().replace(b"\r\n", b"\n")); n += 1

GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
old = b"| **PRE-REGISTERED (2026-10-27), NOT RUN.** Committed by RESULT_V8F_DAGGER2's HARMS row"
assert g.count(old) == 1 and g.count(b"**H-V8F-DAG-ORC-1**") == 1
new = ("| **N0 READ: NOT NAVMOVE (MEASURED 2026-10-27, all 17 controls PASS; A, REP, B NOT RUN).** DAG at T1-kin with --nav logged"
       " against the banked --nav deadreckon runs, same checkpoint, paired episode-cluster bootstrap, B = 2,000, 141 episodes: s0 curvature"
       " MAE −0.0001 [−0.0011, +0.0008], L2 +0.0004 [−0.0001, +0.0012], M1 +0.0001 [−0.0004, +0.0007], none separated; s1 curvature −0.0004"
       " n.s., L2 −0.00068 [−0.00183, −0.00001] separated-better, M1 −0.0004 n.s., yaw-rate −0.107 separated-better. Nav-logged moves"
       " 0.2–1.4 % of DAG's excess over P10 on the rows of C; beside, T-head lat F1 +0.023 / +0.029 (separated). The dead-reckoned nav is"
       " eliminated as the crawl's cause. Committed (§8): DAG-ORC is not held and launches as written on the PI's go; NEXT-CRAWL later starts"
       " from LAUNCH-INTENT (DAG-OACC's read does not exist yet) or COMPOSE if DAG-OACC moves the rows of C at both seeds. T1-kin self-action"
       " open loop; nav-logged is an open channel. `raw/2026-10-27-dag-orc-n0/N0_READ.json`. Was: **PRE-REGISTERED (2026-10-27), NOT RUN.**"
       " Committed by RESULT_V8F_DAGGER2's HARMS row").encode("utf-8")
g2 = g.replace(old, new)
i = g2.index(b"**H-V8F-DAG-ORC-1**"); j = g2.index(b"\n", i)
row = g2[i:j]
tail_old = b"| HYPOTHESIS | `Project Steering/PREREG_V8F_DAG_ORC.md` |"
assert row.endswith(tail_old)
g2 = g2[:i] + row[: -len(tail_old)] + b"| MEASURED (N0); HYPOTHESIS (A) | `Project Steering/PREREG_V8F_DAG_ORC.md` |" + g2[j:]
assert len(g2.split(b"\n")) == len(g.split(b"\n"))
put("mods-rw/" + GC, g2)
put("msg.txt", ("PREREG_V8F_DAG_ORC N0 (the nav probe, 0 GPU): NOT NAVMOVE. DAG at T1-kin with --nav logged against its banked --nav"
                " deadreckon runs moves none of the crawl rows at s0 and two of four at s1, by 0.2-1.4 %% of DAG's excess over P10; all 17 controls"
                " PASS (argv identity but --nav, identity re-read 0.0, RC 0 differences, window set bit-equal). The dead-reckoned nav is eliminated"
                " as the crawl's cause; DAG-ORC is not held. Package raw/2026-10-27-dag-orc-n0/ (%d files; per-episode dumps and ticks stay on E:)."
                " Register row H-V8F-DAG-ORC-1 updated.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % n).encode())
print("hold written", H, n)
