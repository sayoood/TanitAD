"""tele_posthoc.py -- post-hoc read beside card ADD122 (names nothing; run after the names were opened). CPU only.
Asks where the tele heads lose against the wide-only head: frames whose lead box centre lies inside / outside the fixed tele windows, per arm.
Usage: python tele_posthoc.py <reader.py> <read_args.txt> <out.json>"""
import importlib.util
import json
import os
import sys

import numpy as np

reader, args_file, out = sys.argv[1], sys.argv[2], sys.argv[3]
spec = importlib.util.spec_from_file_location("TG", reader)
TG = importlib.util.module_from_spec(spec)
sys.modules["TG"] = TG
spec.loader.exec_module(TG)
cap = {}
orig = TG.compute_values


def grab(LK, RL, J, L, box, *a, **k):
    cap.update(LK=LK, RL=RL, J=J, L=L, box=box)
    return orig(LK, RL, J, L, box, *a, **k)


TG.compute_values = grab
argv = [x.strip() for x in open(args_file, encoding="utf-8") if x.strip()]
tmp = os.path.join(os.path.dirname(out), "posthoc_controls_scratch.json")
try:
    TG.main(["--stage", "controls", "--lookgap-reader", "/workspace/v8proof/lookgap/kit_read/read_lookgap.py", "--lookat-reader", "/workspace/v8proof/lookat/kit/read_lookat.py",
             "--out", tmp] + argv)
except SystemExit:
    pass
L, box = cap["L"], cap["box"]
com = TG.common_rows(L)
rows = np.flatnonzero(com)
lab = L["D_0"]["lab_gap"][rows]
has = box["has"][rows]
az, el = box["az"][rows], box["el"][rows]
T8, T4 = TG.T8, TG.T4
in8 = has & (np.abs(az - T8["az"]) <= T8["half"]) & (np.abs(el - T8["el"]) <= T8["half"])
in4 = has & (np.abs(az - T4["az"]) <= T4["half"]) & (np.abs(el - T4["el"]) <= T4["half"])
W16 = has & (np.abs(az) <= 8.0) & (np.abs(el + 2.0) <= 8.0)
res = {"note": "post-hoc, names nothing; common frame set of the read of record", "n": int(len(rows)), "n_in_T8": int(in8.sum()), "n_out_T8": int((has & ~in8).sum()), "n_in_T4": int(in4.sum()), "arms": {}}
for X, ks in TG.ARMS.items():
    e = {k: np.abs(L[k]["gap"][rows] - lab) for k in ks}
    s = {k: (L[k]["gap"][rows] - lab) for k in ks}
    m = np.mean([e[k] for k in ks], axis=0)
    sg = np.mean([s[k] for k in ks], axis=0)
    d = {"E_in_T8": float(m[in8].mean()), "E_out_T8": float(m[has & ~in8].mean()), "E_in_T4": float(m[in4].mean()), "E_in_T8_not_T4": float(m[in8 & ~in4].mean()),
         "per_fit_in_T8": [float(e[k][in8].mean()) for k in ks], "per_fit_out_T8": [float(e[k][has & ~in8].mean()) for k in ks], "by_gap_in_T8": {}, "by_gap_out_T8": {}}
    for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 60)):
        g = (lab >= lo) & (lab < hi)
        for nm, mk in (("by_gap_in_T8", in8), ("by_gap_out_T8", has & ~in8)):
            mm = g & mk
            d[nm]["%d-%d" % (lo, hi)] = {"n": int(mm.sum()), "E": float(m[mm].mean()) if mm.any() else None, "signed": float(sg[mm].mean()) if mm.any() else None}
    res["arms"][X] = d
json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
print("n", res["n"], "in T8", res["n_in_T8"], "out T8", res["n_out_T8"], "in T4", res["n_in_T4"])
for X, d in res["arms"].items():
    print(X, "E in T8 %.3f | out T8 %.3f | in T4 %.3f | in T8 not T4 %.3f | per fit in T8 %s | per fit out T8 %s" % (
        d["E_in_T8"], d["E_out_T8"], d["E_in_T4"], d["E_in_T8_not_T4"], ["%.2f" % v for v in d["per_fit_in_T8"]], ["%.2f" % v for v in d["per_fit_out_T8"]]))
    print("  in T8 by gap:", {k: (v["n"], None if v["E"] is None else round(v["E"], 2), None if v["signed"] is None else round(v["signed"], 2)) for k, v in d["by_gap_in_T8"].items()})
    print("  out T8 by gap:", {k: (v["n"], None if v["E"] is None else round(v["E"], 2), None if v["signed"] is None else round(v["signed"], 2)) for k, v in d["by_gap_out_T8"].items()})
