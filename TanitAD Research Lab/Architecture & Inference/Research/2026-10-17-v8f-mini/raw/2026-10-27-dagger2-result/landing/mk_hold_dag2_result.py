"""Hold: RESULT_V8F_DAGGER2 (HARMS) + the read package raw/2026-10-27-dagger2-result/ + register row H-V8F-DAGGER2-1.
python mk_hold_dag2_result.py <live tip>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
T1 = "E:/Projects/TanitAD-artifacts/v8f-dagger2-t1/"
D2 = "E:/Projects/TanitAD-artifacts/dag2t1/"
HERE = "E:/Projects/TanitAD-artifacts/dagland/dag2_result/"
H = "C:/Users/Admin/qland/hold_dag2_result/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-27-dagger2-result/"
assert not os.path.exists(H), H


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


rd = lambda p: open(p, "rb").read()
res = rd(HERE + "RESULT_V8F_DAGGER2.md")
assert b"\r" not in res and blob(V8 + "RESULT_V8F_DAGGER2.md") is None
put("stage/" + V8 + "RESULT_V8F_DAGGER2.md", res)
files = {"read/DAG2_READ.json": T1 + "read/DAG2_READ.json", "read/readA_compact.txt": T1 + "read/readA_compact.txt",
         "runs.json": T1 + "runs.json", "PINS.txt": T1 + "PINS.txt", "ids/eval141_clip_ids.json": T1 + "ids/eval141_clip_ids.json",
         "ids/train_clip_ids.json": T1 + "ids/train_clip_ids.json", "selftest_real_pulls.txt": D2 + "selftest_real_pulls.txt",
         "dry_run_real_config_amend1.txt": D2 + "dry_run_real_config_amend1.txt", "landing/mk_hold_dag2_result.py": HERE + "mk_hold_dag2_result.py"}
for f in os.listdir(T1 + "read/_dag2_work"):
    files["read/_dag2_work/" + f] = T1 + "read/_dag2_work/" + f
for f in os.listdir(T1):
    if f.startswith("t1v8f_") and f.endswith(".json") or f.startswith("done_"):
        files[f] = T1 + f
for f in os.listdir(T1 + "t0fresh"):
    if f.endswith(".json"):
        files["t0fresh/" + f] = T1 + "t0fresh/" + f
for rel, src in files.items():
    b = rd(src)
    if b"" == b:
        continue
    assert blob(PK + rel) is None, rel
    put("stage/" + PK + rel, b.replace(b"\r\n", b"\n") if rel.endswith(".py") else b)

GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
old = b"| **PRE-REGISTERED (2026-10-27), NOT RUN.** Run basis: PREREG_V8F_DAGGER's ROBUST clause"
assert g.count(old) == 1 and g.count(b"**H-V8F-DAGGER2-1**") == 1
new = ("| **HARMS (MEASURED 2026-10-27, read by the pre-registered reader at Amendment 1; all 199 controls PASS).** Guards held but one: the"
       " newly-worse clause fails at s0 on yaw-rate MAE (DAG2 − P10 +6.2521 [+3.276, +9.143] °/s, separated-worse; DAG − P10 was −0.0400)."
       " The crawl rows did not move: R-CRAWL and R-MOVE fail at both seeds (M1 DAG2 − P10 +0.0524 / +0.0424, L2 +0.0357 / +0.0249, curvature"
       " +0.1022 / +0.0785, all separated-worse), so with the guards held the table would read NO EFFECT. Beside: DAG2 − DAG2-0 T1-kin ADE6"
       " −0.2247 [−0.3021, −0.1536] / −0.1630 [−0.2582, −0.0596] m, lat F1 +0.0761 / +0.0791 (both clear 2F), accel MAE +0.5846 / +0.6034"
       " (separated-worse); DAG2 − DAG ADE6 +0.0426 (s0, separated-worse) / −0.0309 (s1, n.s.); yaw-rate against P10 swings −0.04 / +8.66"
       " (DAG) and +6.25 / +5.63 (DAG2), a run-to-run spread the clause has no floor for. T1-kin self-action open loop (R-T0 at T0); paired"
       " episode-cluster bootstrap, B = 2,000, 141 episodes. Committed next: DAG2 not adopted, DAG stays the base; the launch-recovery target"
       " on DAG states under its own PREREG. `RESULT_V8F_DAGGER2.md`, `raw/2026-10-27-dagger2-result/read/DAG2_READ.json`. Was:"
       " **PRE-REGISTERED (2026-10-27), NOT RUN.** Run basis: PREREG_V8F_DAGGER's ROBUST clause").encode("utf-8")
g2 = g.replace(old, new)
i = g2.index(b"**H-V8F-DAGGER2-1**")
j = g2.index(b"\n", i)
row = g2[i:j]
assert row.endswith(b"| HYPOTHESIS | `Project Steering/PREREG_V8F_DAGGER2.md` |"), row[-120:]
row2 = row[: -len(b"| HYPOTHESIS | `Project Steering/PREREG_V8F_DAGGER2.md` |")] + b"| MEASURED | `Project Steering/PREREG_V8F_DAGGER2.md` |"
g2 = g2[:i] + row2 + g2[j:]
assert len(g2.split(b"\n")) == len(g.split(b"\n"))
put("mods-rw/" + GC, g2)
n = len(files) + 1
put("msg.txt", ("RESULT_V8F_DAGGER2: HARMS (H-V8F-DAGGER2-1). Read by the pre-registered reader at Amendment 1 on 8/8 T1-kin rollouts + 4 fresh-T0"
                " evals (dev-box CPU), all 199 controls PASS. The guard that fails: newly-worse yaw-rate MAE at s0. The crawl rows did not move"
                " (R-CRAWL, R-MOVE fail at both seeds). Beside: the swap beats its own control on T1-kin ADE6 (-0.22 / -0.16 m) and lat F1, costs"
                " accel MAE (+0.58 / +0.60), and does not beat DAG. DAG stays the base; committed next is the launch-recovery target on DAG states."
                " Package raw/2026-10-27-dagger2-result/ (%d files; per-episode dumps and tick arrays, 1.3 GB, stay on E:). Register row updated.\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % n).encode())
print("hold written", H, n)
