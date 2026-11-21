"""N4-DAC Q3b: (1) the EXECUTED (tracked) path change a repair causes, vs the base label's execution and vs the teacher's own plan;
(2) W-TRANSP-format rows for the families: banked RJ5 rows with arm R replaced by the repaired label (pipeline P1 = REPAIR-D on RJ5.R) and by
CERT-A + REPAIR-D (pipeline P2). Rows go to F:/tanitad_scratch/n4dac/spliced/ (scratch, deleted at the end)."""
import json
import os
import sys
import numpy as np
sys.path.insert(0, ".")
sys.path.insert(0, "F:/tanitad_scratch/n4dac/src/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-11-11-teacher-fidelity/n2")
import n4_geom as G
import n4_common as N
import tfn2_probe as P

Rl, S = N.load_rolls(), N.load_scores()
Q = json.load(open("Q3_repairs.json", encoding="utf-8"))["instances"]
common = sorted(set.intersection(*[set(v) for v in S.values()]))
ORDER = json.load(open("Q2_cert_wider.json", encoding="utf-8"))["order"]
OUT = "F:/tanitad_scratch/n4dac/spliced"
os.makedirs(OUT, exist_ok=True)

need = {}
for q in Q:
    need.setdefault(q["log"], set()).add(q["token"])
need = {k: sorted(v) for k, v in need.items()}
exe, newarm = [], {}
for lg, tk, cx in G.contexts(need):
    for q in [x for x in Q if x["token"] == tk]:
        run = q["run"]
        t20 = np.asarray(Rl[run][tk]["traj20"], float)
        lab = cx.labels(t20)["R"]

        def exec8(p):
            return G.tracked8(G.ND, G.ND.to_navsim(p)[None], cx.ego)[0]
        e_base = exec8(lab)
        for m, v in q["methods"].items():
            if not v.get("repaired"):
                continue
            p = np.asarray(v["plan20"], float)
            e_new = exec8(p)
            rec = {"run": run, "token": tk, "method": m,
                   "exec_shift_vs_base_exec_xy_max_m": round(float(np.linalg.norm(e_new[:, :2] - e_base[:, :2], axis=1).max()), 4),
                   "exec_shift_vs_teacher_plan_xy_max_m": round(float(np.linalg.norm(e_new[:, :2] - G.ND.to_navsim(t20)[:, :2], axis=1).max()), 4),
                   "exec_base_vs_teacher_plan_xy_max_m": round(float(np.linalg.norm(e_base[:, :2] - G.ND.to_navsim(t20)[:, :2], axis=1).max()), 4)}
            exe.append(rec)
            if run == "RJ5" and m == "D":
                arm, _ = P.score_arm(G.TS, cx.parts, cx.ego, cx.s0, p)
                newarm[tk] = arm
json.dump(exe, open("Q3b_exec_shift.json", "w", encoding="utf-8"), indent=1)
for m in "BHAD":
    rr = [r for r in exe if r["method"] == m]
    if rr:
        print(m, len(rr), "exec shift vs base exec: median %.3f max %.3f" % (np.median([r["exec_shift_vs_base_exec_xy_max_m"] for r in rr]), max(r["exec_shift_vs_base_exec_xy_max_m"] for r in rr)),
              "| vs teacher plan: median %.3f max %.3f" % (np.median([r["exec_shift_vs_teacher_plan_xy_max_m"] for r in rr]), max(r["exec_shift_vs_teacher_plan_xy_max_m"] for r in rr)),
              "| base exec vs teacher plan median %.3f" % np.median([r["exec_base_vs_teacher_plan_xy_max_m"] for r in rr]))

# ---- spliced rows ----
def rows_of(run):
    d = {}
    for i in (0, 1):
        for ln in open(f"{N.ROWS}/scores/{run}_s{i}.jsonl", encoding="utf-8"):
            r = json.loads(ln)
            if "error" not in r:
                d[r["token"]] = r
    return d


RJ5 = rows_of("RJ5")
allrows = {r: rows_of(r) for r in N.RUNS}
P1, P2, chosen = [], [], {}
for tk in common:
    r1 = json.loads(json.dumps(RJ5[tk]))
    if tk in newarm:
        r1["arms"]["R"] = newarm[tk]
    P1.append(r1)
    r2 = json.loads(json.dumps(RJ5[tk]))
    sel = None
    for nm in ORDER:
        run, f = nm.split(".")
        if allrows[run][tk]["arms"][f]["tracked"]["dac"] == 1.0:
            sel = nm
            break
    if sel is None:
        assert tk in newarm, tk
        r2["arms"]["R"] = newarm[tk]
        chosen[tk] = "REPAIR-D"
    else:
        run, f = sel.split(".")
        assert allrows[run][tk]["shared"]["human8"] == RJ5[tk]["shared"]["human8"], "shared differs across runs"
        r2["arms"]["R"] = allrows[run][tk]["arms"][f]
        chosen[tk] = sel
    P2.append(r2)
for nm, rows in (("P1_repairD", P1), ("P2_certA_repairD", P2)):
    with open(f"{OUT}/{nm}.jsonl", "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
json.dump(chosen, open("Q3b_pipeline_choices.json", "w"), indent=0)
import collections
print("P2 choices:", collections.Counter(chosen.values()))
