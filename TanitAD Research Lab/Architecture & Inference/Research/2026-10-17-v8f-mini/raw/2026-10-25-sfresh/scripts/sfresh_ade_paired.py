"""descriptive: paired ADE6 / ADE2 / FDE6, fresh - banked cycle, per seed (same 2,399 windows / 141 episodes); and the seed floor on Q-LAYER / Q-STALE."""
import json, os, sys
import numpy as np
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(OUT, "scripts"))
import sfresh_analyze as A
grid, J, X = A.load_ctx()
res = {}
for s in (0, 1):
    fr = np.load(os.path.join(A.EVD, f"eval_M2b_s{s}-sfresh-fresh.npz")); cy = A.banked_cycle_npz(s)
    res[f"s{s}"] = {k: A.pr(A.E.paired(fr[k], cy[k], X["eid"], A.NB)) for k in ("ade6", "ade2", "fde6")}
n = json.load(open(os.path.join(OUT, "sfresh_numbers.json")))
fl = {}
for q in ("Q_STALE_fresh_minus_cycle_pos_err", "Q_LAYER_fresh_S_minus_navonly_pos_err"):
    for arc in ("@30", "@50", "@80"):
        d0, d1 = n["seeds"]["s0"][q][arc]["delta"], n["seeds"]["s1"][q][arc]["delta"]
        fl[f"{q}{arc}"] = {"s0": d0, "s1": d1, "F_abs_s0_minus_s1": abs(d0 - d1)}
res["seed_floor_F_on_the_Q_reads"] = fl
A.jd(res, os.path.join(OUT, "sfresh_ade_paired.json"))
print(json.dumps(res, indent=1))
