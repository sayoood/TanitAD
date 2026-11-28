"""TRK-MIN C1 table read (AMENDMENT_TRKMIN.md, sha d8f5462d..., frozen before this ran). PROVISIONAL until the harness confirmation and the C0 replicate.
usage: python trkmin_read.py --pk <PK root of a landed export> --dump <trk_c1.npz> --table "<full_C1_s*of2.jsonl>" --trk-csv <refe_stage2_trk_c1.csv> --out <json>"""
import argparse, csv, glob, json, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pk", required=True); ap.add_argument("--dump", required=True); ap.add_argument("--table", required=True)
ap.add_argument("--trk-csv", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, a.pk + "/refe"); sys.path.insert(0, a.pk + "/eval/p5c")
import safehead_data as D  # noqa: E402  (agg32: the ONE selection arithmetic)
import p5c_read as P  # noqa: E402  (Boot)

d = np.load(a.dump)
tok = d["token"].astype(str); k_ship = d["k_ship"]; k_trk = d["k_trk"]
A_plan = D.agg32(d["s_plan"]); A_trk = D.agg32(d["s_trk"])
A_min = np.minimum(A_plan, A_trk)
k_min = A_min.argmax(-1)
rows = {}
for f in sorted(glob.glob(a.table)):
    for line in open(f, encoding="utf-8"):
        r = json.loads(line); rows[r["token"]] = r
n = len(tok)
ctrl = {}
ctrl["CTRL-SET"] = {"n_tokens": n, "in_table": int(sum(t in rows for t in tok)), "n_logs": len({rows[t]["log"] for t in tok if t in rows})}
ctrl["CTRL-SET"]["pass"] = n == 12146 and ctrl["CTRL-SET"]["in_table"] == n and ctrl["CTRL-SET"]["n_logs"] == 136
idp = int((A_plan.argmax(-1) == k_ship).sum()); idt = int((A_trk.argmax(-1) == k_trk).sum())
M = ("nc", "dac", "ttc", "ep", "c", "ddc", "pdms")
def at(k):
    return {m: np.array([rows[t]["tr"][m][int(k[i])] for i, t in enumerate(tok)], np.float64) * 100 for m in M}
V_c1, V_trk, V_min = at(k_ship), at(k_trk), at(k_min)
ctrl["CTRL-ID-T"] = {"plan_argmax_eq_k_ship": idp, "trk_argmax_eq_k_trk": idt, "c1_table_mean": float(V_c1["pdms"].mean()),
                     "pass": idp == n and idt == n and abs(V_c1["pdms"].mean() - 92.5548) <= 5e-5}
trk_csv = {r["token"]: float(r["score"]) * 100 for r in csv.DictReader(open(a.trk_csv)) if r["token"] != "average"}
dt = np.array([abs(V_trk["pdms"][i] - trk_csv[t]) for i, t in enumerate(tok)])
ctrl["CTRL-T-TAB"] = {"max_abs_diff_x100": float(dt.max()), "n_over_1e-4": int((dt > 1e-4).sum()), "pass": bool(dt.max() <= 1e-4)}   # 1e-6 on the 0..1 scale
both = (k_trk == k_ship)
ctrl["CTRL-MIN"] = {"agg_min_le_both": bool((A_min <= A_plan + 0).all() and (A_min <= A_trk + 0).all()),
                    "agree_where_ship_eq_trk": int((k_min[both] == k_ship[both]).sum()), "n_ship_eq_trk": int(both.sum()),
                    "n_kmin_neither": int(((k_min != k_ship) & (k_min != k_trk)).sum()),
                    "n_kmin_eq_ship": int((k_min == k_ship).sum()), "n_kmin_eq_trk": int((k_min == k_trk).sum())}
# where k_ship == k_trk it is the argmax of BOTH sets, so it is also the argmax of their min
ctrl["CTRL-MIN"]["pass"] = ctrl["CTRL-MIN"]["agg_min_le_both"] and ctrl["CTRL-MIN"]["agree_where_ship_eq_trk"] == ctrl["CTRL-MIN"]["n_ship_eq_trk"]
logs = np.array([rows[t]["log"] for t in tok])
B = P.Boot(logs, n_boot=P.NBOOT, seed=P.SEED)
res = {"amendment_sha256": "d8f5462dd05682eb9ca587e76241d84d173383602007b6ea02f4cd0dbec96184", "status": "PROVISIONAL (harness confirmation + C0 replicate pending)",
       "tier": "T1 (banked harness-exact 64-proposal table)", "estimator": "p5c_read.Boot paired log-cluster, 10,000 draws, seed 20260927, 95 % percentile, 136 logs",
       "controls": ctrl, "all_controls_pass": all(v["pass"] for v in ctrl.values()), "arms": {}}
for name, V in (("TRK", V_trk), ("TRKMIN", V_min)):
    res["arms"][name] = {m: {"mean": float(V[m].mean()), "delta_vs_C1": B.ci(V[m] - V_c1[m])} for m in M}
res["C1_means"] = {m: float(V_c1[m].mean()) for m in M}
json.dump(res, open(a.out, "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
print("controls", {k: v["pass"] for k, v in ctrl.items()}, "kmin:", {k: ctrl["CTRL-MIN"][k] for k in ("n_kmin_eq_ship", "n_kmin_eq_trk", "n_kmin_neither")})
for name in ("TRK", "TRKMIN"):
    for m in ("pdms", "nc", "dac", "ttc", "ep", "c", "ddc"):
        c = res["arms"][name][m]["delta_vs_C1"]
        print(f"{name:6s} {m:4s} {res['arms'][name][m]['mean']:8.4f} delta {c['point']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" + ("  SEPARATED" if c["lo"] > 0 or c["hi"] < 0 else ""))
