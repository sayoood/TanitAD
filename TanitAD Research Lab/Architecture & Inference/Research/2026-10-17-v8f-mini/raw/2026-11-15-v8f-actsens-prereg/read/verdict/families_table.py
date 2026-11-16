"""ACTSENS four-family table (MM 2026-11-16). Descriptive companion to verdict_actsens.py, which alone decides the label.
For every cell of the arm JSONs (hwm planner, oracle goal, tier T0), the lever arm's delta vs R35 at its chassis with the paired episode-cluster
bootstrap CI, marked against the card's floor applied to that cell: Floor_k = max(|NULL_k - R35_k|, F_I), F_I = max over both chassis of the
lever arm's and R35's |inference seed 1 - seed 0|. B = CI below 0 and |delta| > floor, W = CI above 0 and delta > floor.
NULL columns show the training floor itself (s = its CI excludes 0). STRATEGIC is PAUSED (PI item 25).
usage: python families_table.py ARMS_DIR OUT_MD"""
import json
import sys

D, OUT = sys.argv[1], sys.argv[2]
ARMS = {k: json.load(open(f"{D}/arm_{k}.json", encoding="utf-8")) for k in ("IDM_s0", "DIV_s0", "NULL_s2", "IDM_s1", "DIV_s1", "NULL_s3")}
NULL = {0: "NULL_s2", 1: "NULL_s3"}
cells = list(ARMS["NULL_s2"]["hwm"]["cells"].keys())


def c(a, k, slot):
    return ARMS[a]["hwm"]["cells"][k][slot]


def fmt(v):
    return f"{v:+.4f}"


rows = []
tally = {}
for k in cells:
    fam = k.split("|")[0]
    out = [k.replace("|", " ")]
    for a in ("IDM_s0", "DIV_s0", "NULL_s2", "IDM_s1", "DIV_s1", "NULL_s3"):
        ch = int(a[-1]) % 2 if a.startswith("NULL") else int(a[-1])
        ch = 0 if a == "NULL_s2" else 1 if a == "NULL_s3" else ch
        v = c(a, k, "vs_r35")
        cell = f"{fmt(v['delta'])} [{v['lo']:+.3f}, {v['hi']:+.3f}]"
        if a.startswith("NULL"):
            if v["lo"] > 0 or v["hi"] < 0:
                cell += " s"
        else:
            lev = a.split("_")[0]
            reps = [ARMS[f"{lev}_s{j}"]["hwm"]["cells"][k].get(r) for j in (0, 1) for r in ("rep_arm", "rep_r35")]
            fi = max([abs(r["delta"]) for r in reps if r] or [0.0])
            dag = "" if all(reps) else " +"
            fl = max(abs(c(NULL[ch], k, "vs_r35")["delta"]), fi)
            mark = "B" if (v["hi"] < 0 and abs(v["delta"]) > fl) else "W" if (v["lo"] > 0 and v["delta"] > fl) else ""
            cell += dag
            if mark:
                cell += " **" + mark + "**"
                tally.setdefault((a, fam, mark), 0)
                tally[(a, fam, mark)] += 1
        out.append(cell)
    rows.append(out)
n = ARMS["NULL_s3"]
L = ["# ACTSENS: four families, six arms (T0, oracle goal: a world-model diagnostic, never driving performance)", "",
     f"Estimator: {n.get('estimator')}. Windows per chassis {ARMS['NULL_s2']['n_windows']} / {n['n_windows']}, episodes {ARMS['NULL_s2']['n_episodes']} / {n['n_episodes']}. "
     "Each cell is delta vs R35 at the arm's chassis (lower is better), CI in brackets. B / W = beyond the card's floor on that cell (see families_table.py). "
     "A + marks a cell without an inference replicate, where the floor is the training floor alone. The label comes from `verdict_actsens.py` alone, on the two deciding quantities (LONGITUDINAL 0-2 s acc_mae, LATERAL 0-2 s yaw_mae).", "",
     "**STRATEGIC: PAUSED (PI item 25), not a pass.**", "",
     "| cell | IDM_s0 | DIV_s0 | NULL_s2 (floor c0) | IDM_s1 | DIV_s1 | NULL_s3 (floor c1) |", "|---|---|---|---|---|---|---|"]
L += ["| " + " | ".join(r) + " |" for r in rows]
L += ["", "## Count of cells beyond the floor (B better, W worse)", "", "| arm | family | B | W |", "|---|---|---|---|"]
for a in ("IDM_s0", "IDM_s1", "DIV_s0", "DIV_s1"):
    for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL"):
        L.append(f"| {a} | {fam} | {tally.get((a, fam, 'B'), 0)} | {tally.get((a, fam, 'W'), 0)} |")
open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("\n".join(L[-14:]))
