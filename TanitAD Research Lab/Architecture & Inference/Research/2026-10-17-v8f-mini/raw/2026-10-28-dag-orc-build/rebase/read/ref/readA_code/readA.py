"""PREREG_V8F_DAG_REREAD Read A, the comps-based rows (arm - P10, T1-kin, per seed): ADE6, speed / along / accel MAE, heading / yaw-rate / curvature / cross-track MAE,
L1, L2, plus M1 (per-window crawl share) and M2.  TAC-LAT / TAC-LON / DK-* come from t1kf_read_a1 (run separately); S-POS from DAG_READ.json.
usage: readA.py <packed prefix e.g. DAG> <cfg: causal|logged|frozen_causal> <out json> [--vs HACC]"""
import ast
import json
import math
import os
import sys
import types

sys.path.insert(0, "E:/Projects/TanitAD-artifacts/dagreread/code")
import numpy as np
import rr_lib as R

RD, S = R.RD, "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad"
# m_rows (M1 pooled / M2) AST-extracted from hmc_controls.py (REUSE, NEVER RE-IMPLEMENT); its `L.VMIN` binds to a namespace carrying VMIN only.
_ns2 = {"np": np, "L": types.SimpleNamespace(VMIN=R.VMIN), "pair_err": R.pair_err}
for _n in ast.parse(open(f"{RD}/hmc_controls.py", encoding="utf-8").read()).body:
    if isinstance(_n, ast.FunctionDef) and _n.name == "m_rows":
        exec(compile(ast.Module([_n], []), "hmc_controls.py", "exec"), _ns2)
m_rows = _ns2["m_rows"]

ROWS_A = [  # (label, key in rows_of, family, role)
    ("ADE6", "ade6_m", "PRIMARY"),
    ("speed MAE", "speed_mae_mps", "LONGITUDINAL"), ("along-track MAE", "along_mae_m", "LONGITUDINAL"), ("accel MAE", "accel_mae_mps2", "LONGITUDINAL"),
    ("heading MAE", "heading_mae_deg", "LATERAL"), ("yaw-rate MAE", "yaw_rate_mae_degps", "LATERAL"), ("curvature MAE", "curvature_mae_1pm", "LATERAL"),
    ("cross-track MAE", "cross_mae_m", "LATERAL"), ("L1 lateral-accel MAE", "L1_lat_accel_mae_mps2", "LATERAL"), ("L2 curvature MAE (GT>=5.2)", "L2_curvature_mae_fast_1pm", "LATERAL"),
    ("ADE2 (beside)", "ade2_m", "beside"), ("L3 yaw-rate fast (beside)", "L3_yaw_rate_mae_fast_degps", "beside"),
]


def prow(a, b, eid):
    a, b, eid = np.asarray(a, np.float64), np.asarray(b, np.float64), np.asarray(eid)
    m = np.isfinite(a) & np.isfinite(b)
    r = R.paired(a, b, eid)
    r.update({"mean_a": float(a[m].mean()) if m.any() else None, "mean_b": float(b[m].mean()) if m.any() else None})
    r["separated_better"] = bool(r["separated"] and r["delta"] < 0)
    r["separated_worse"] = bool(r["separated"] and r["delta"] > 0)
    return r


def rows_block(X, Y, G, eid):
    """X, Y [n, 30, 2] tracks (error rows: negative delta = X better), vs the same GT G."""
    cx, cy = R.rows_of(X, G), R.rows_of(Y, G)
    out = {lab: prow(cx[k], cy[k], eid) for lab, k, fam in ROWS_A}
    for lab, k, fam in ROWS_A:
        out[lab]["family"], out[lab]["key"] = fam, k
    sx, px, nx, fx = R.m1_window(X, G)
    sy, py, ny, fy = R.m1_window(Y, G)
    m1 = prow(sx, sy, eid)
    m1.update({"family": "crawl", "key": "M1 per-window share", "pooled_M1_a": px, "pooled_M1_b": py, "n_arm_crawl_pairs_a": nx, "n_fast_pairs_a": fx,
               "n_crawl_pairs_b": ny, "n_fast_pairs_b": fy})
    out["M1 crawl share"] = m1
    mr = m_rows(X, Y, G)
    out["_M2"] = {"M1_pooled": mr["M1"], "M1_P10_pooled": mr["M1_P10"], "M2": mr["M2"], "n_arm_crawl_pairs": mr["n_arm_crawl_pairs"], "n_fast_pairs": mr["n_fast_pairs"]}
    return out


def ident_block(X, G, eid):
    """C2 for the comps-based readers: (X, X) reads exactly +0 [0, 0] on every row."""
    b = rows_block(X, X, G, eid)
    return {k: (v["delta"], v["lo"], v["hi"], v["n_windows"]) for k, v in b.items() if not k.startswith("_")}


if __name__ == "__main__":
    arm, cfg, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
    vs = sys.argv[sys.argv.index("--vs") + 1] if "--vs" in sys.argv else None
    P = "E:/Projects/TanitAD-artifacts/dagreread/packed"
    res = {"arm": arm, "cfg": cfg, "tier": "T1-kin (self-action OPEN loop; m=10, nav deadreckon, periodic, causal bucket unless cfg says otherwise)", "seeds": {}}
    for s in R.SEEDS:
        d = R.load_npz(f"{P}/dumps_{arm}_s{s}_m10_{cfg}.npz")
        eid = d["eid"]
        blk = rows_block(d["cl"], d["fl"], d["g"], eid)
        ident = ident_block(d["cl"], d["g"], eid)
        ident_ok = all(v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0 for v in ident.values())
        res["seeds"][f"s{s}"] = {"rows_arm_minus_P10": blk, "C2_identity_comps_rows": {"all_exactly_zero": ident_ok, "rows": ident}, "n_windows_total": int(len(eid)),
                                 "n_episodes_total": int(len(np.unique(eid)))}
        if vs:
            h = R.load_dir(f"{S}/hacc/dump_{vs}_s{s}_m10_causal") if vs == "HACC" else None
            same = bool(np.array_equal(h["clip_index"], d["clip_index"]) and np.array_equal(h["ws"], d["ws"]) and np.array_equal(h["eid"], eid) and np.array_equal(h["g"], d["g"]))
            cx, ch = R.rows_of(d["cl"], d["g"]), R.rows_of(h["cl"], h["g"])
            res["seeds"][f"s{s}"][f"beside_arm_minus_{vs}"] = {"same_windows_and_gt": same, "ADE6": prow(cx["ade6_m"], ch["ade6_m"], eid),
                                                               "L2": prow(cx["L2_curvature_mae_fast_1pm"], ch["L2_curvature_mae_fast_1pm"], eid)}
    json.dump(res, open(out_p, "w"), indent=1)
    for s in R.SEEDS:
        for lab, v in res["seeds"][f"s{s}"]["rows_arm_minus_P10"].items():
            if lab.startswith("_"):
                print(s, lab, v)
            else:
                print(f"s{s} {lab:30s} {v['delta']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] sep={v['separated']} n={v['n_windows']}/{v['n_episodes']} a={v['mean_a']:.4f} b={v['mean_b']:.4f}")
        print(f"s{s} C2 comps identity all exactly zero:", res["seeds"][f"s{s}"]["C2_identity_comps_rows"]["all_exactly_zero"])
