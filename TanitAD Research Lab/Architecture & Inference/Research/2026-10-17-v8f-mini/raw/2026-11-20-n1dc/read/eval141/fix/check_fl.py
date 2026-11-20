import sys, json, numpy as np
sys.path.insert(0, "F:/TanitAD-scratch/strat1/read/theory/obd/read")
import obd_read as OR
E = "F:/TanitAD-scratch/n1dc/read/eval141"
ref = OR.load_dump("F:/TanitAD-scratch/strat1/read/t1/dump_STRAT1-C0_s0_m10_causal_heads")
hb = {t: OR.load_dump(f"{E}/dumps141_hb02/dump_N1DC-{t}_m10_causal_heads") for t in ("N1D_s0","N1D_s0Z","N1D_s1","N1D_s1Z","N1DC_s0","N1DC_s0Z","N1DC_s1","N1DC_s1Z")}
old = {t: OR.load_dump(f"{E}/dumps141_old/dump_N1DC-{t}_m10_causal_heads") for t in ("N1DC_s0","N1DC_s1")}
out = {"hb_vs_C0old": {}, "old_vs_C0old": {}, "hb_fl_vs_hbN1D_s0": {}}
for t, x in hb.items():
    out["hb_vs_C0old"][t] = {k: bool(np.array_equal(x[k], ref[k])) for k in OR.SAME}
    out["hb_fl_vs_hbN1D_s0"][t] = bool(np.array_equal(x["fl"], hb["N1D_s0"]["fl"]))
for t, x in old.items():
    out["old_vs_C0old"][t] = {k: bool(np.array_equal(x[k], ref[k])) for k in OR.SAME}
a, b = hb["N1D_s0"]["fl"].astype(np.float64), ref["fl"].astype(np.float64)
d = np.abs(a - b).reshape(len(a), -1).max(1)
out["fl_hb_minus_old"] = {"shape": list(a.shape), "windows_differ": int((d > 0).sum()), "n": int(len(d)), "max_abs": float(d.max()), "median_where_differ": float(np.median(d[d > 0])) if (d > 0).any() else 0.0}
print(json.dumps(out, indent=1))
json.dump(out, open(f"{E}/fix/check_fl.json", "w"), indent=1)
