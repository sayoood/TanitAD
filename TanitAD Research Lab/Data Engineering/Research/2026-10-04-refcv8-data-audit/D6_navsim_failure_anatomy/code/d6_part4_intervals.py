"""D6 part 4 -- log-cluster bootstrap intervals for the headline shares and the max-speed difference-in-differences.

ESTIMATOR (stated, per the programme rule): resample the scorer's LOG units with replacement (navhard: 76 logs; navtest:
136 logs), B = 2000, seed 0, percentile 95 % interval.  It answers "would another draw of LOGS say this?" and is BLIND to
training variance (H-ESTIM-SEED-1) and to inference variance; the seed-1 replicate overlap (part 1) is the only
inference-variance evidence quoted beside it.

Reads raw/d6_scene_geometry_step30000.csv (part 2), raw/d6_scene_table_step30000.csv (part 1), raw/d6_navtest_geometry_step30000.csv (part 3).
"""
from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d6_common as C

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "raw")
B = 2000


def boot(units: pd.Series, stat, B=B, seed=0):
    """units = Series log -> (list of per-log arrays); stat(sum_of_selected) -> float.  Here: per-log count tensors."""
    rng = np.random.default_rng(seed)
    M = np.stack(units.values)            # [n_logs, k]
    n = M.shape[0]
    out = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        out[b] = stat(M[idx].sum(0))
    return out


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def share_cis(df, logcol, classcol, subset_mask, classes):
    """share of each class within the subset, CI over logs."""
    d = df[subset_mask]
    logs = df[logcol].unique()
    res = {}
    cnt = {c: d[d[classcol] == c].groupby(logcol).size().reindex(logs, fill_value=0).values for c in classes}
    tot = d.groupby(logcol).size().reindex(logs, fill_value=0).values
    for c in classes:
        M = pd.Series(list(np.stack([cnt[c], tot], 1)), index=logs)
        pt = cnt[c].sum() / tot.sum() if tot.sum() else float("nan")
        res[c] = {"n": int(cnt[c].sum()), "of": int(tot.sum()), "share": float(pt), "ci95_logs": ci(boot(M, lambda s: s[0] / s[1] if s[1] else np.nan))}
    return res


def rate_diff(df, logcol, mask_a, mask_b, flag_a, flag_b, seed=0):
    """(zero-rate of flag_a on mask_a) - (zero-rate of flag_b on mask_b)  with log-cluster CI.  [na, ka, nb, kb] per log."""
    logs = df[logcol].unique()
    g = pd.DataFrame({"log": df[logcol], "na": mask_a.astype(int), "ka": (mask_a & (flag_a == 0)).astype(int),
                      "nb": mask_b.astype(int), "kb": (mask_b & (flag_b == 0)).astype(int)}).groupby("log").sum().reindex(logs, fill_value=0)
    M = pd.Series(list(g[["na", "ka", "nb", "kb"]].values), index=g.index)
    f = lambda s: s[1] / s[0] - s[3] / s[2] if s[0] and s[2] else np.nan
    pt = f(g[["na", "ka", "nb", "kb"]].values.sum(0))
    return {"point": float(pt), "ci95_logs": ci(boot(M, f, seed=seed)), "n_a": int(g.na.sum()), "n_b": int(g.nb.sum())}


def did(df, logcol, term, arm, known_col="vmax_known", stage_mask=None, seed=0, ref="STOP"):
    """difference-in-differences: [A1 - STOP zero-rate]_unknown - [A1 - STOP zero-rate]_known."""
    logs = df[logcol].unique()
    m = np.ones(len(df), bool) if stage_mask is None else stage_mask
    unk, kn = (~df[known_col]).values & m, df[known_col].values & m
    a, s = df[f"{arm}_{term}"].values == 0, df[f"{ref}_{term}"].values == 0
    g = pd.DataFrame({"log": df[logcol].values, "nu": unk.astype(int), "ku": (unk & a).astype(int), "su": (unk & s).astype(int),
                      "nk": kn.astype(int), "kk": (kn & a).astype(int), "sk": (kn & s).astype(int)}).groupby("log").sum().reindex(logs, fill_value=0)
    M = pd.Series(list(g[["nu", "ku", "su", "nk", "kk", "sk"]].values), index=g.index)
    f = lambda v: (v[1] - v[2]) / v[0] - (v[4] - v[5]) / v[3] if v[0] and v[3] else np.nan
    pt = f(g[["nu", "ku", "su", "nk", "kk", "sk"]].values.sum(0))
    return {"did": float(pt), "ci95_logs": ci(boot(M, f, seed=seed)), "n_unknown": int(g.nu.sum()), "n_known": int(g.nk.sum())}


def main():
    D = pd.read_csv(os.path.join(RAW, "d6_scene_geometry_step30000.csv")).set_index("token")
    T = pd.read_csv(os.path.join(RAW, "d6_scene_table_step30000.csv")).set_index("token")
    D["log"] = T.loc[D.index, "log_name"]
    out = {"estimator": "log-cluster bootstrap, B=2000, seed 0, percentile 95%", "n_logs_navhard": int(D.log.nunique())}
    classes = ["ON-ROUTE", "ROUTE-FOLLOWING", "WRONG-SIDE", "OVER-STEER", "LATERAL-DRIFT", "NO-RECOVERY", "SPEED", "STOP-LIKE"]
    for s, name in ((2, "stage2"), (1, "stage1"), (0, "both")):
        m = (D.A1_DAC == 0) & ((D.stage == s) if s else True)
        out[f"DAC_geometry_class_shares_{name}"] = share_cis(D, "log", "cls", m.values, classes)
    # scene-level flags (rescored)
    R = D[D.resc_ok].copy()
    R["scene"] = np.where(R.nd_first == 0, "INITIAL-STATE", np.where(R.ref_dac == 0, "REF-ALSO-FAILS", "PLAN-INDUCED"))
    Dz = D.copy()
    Dz["scene"] = "NA"
    Dz.loc[R.index, "scene"] = R.scene
    for s, name in ((2, "stage2"), (1, "stage1"), (0, "both")):
        m = (D.A1_DAC == 0) & D.resc_ok & ((D.stage == s) if s else True)
        out[f"DAC_scene_flag_shares_{name}"] = share_cis(Dz, "log", "scene", m.values, ["INITIAL-STATE", "REF-ALSO-FAILS", "PLAN-INDUCED"])
    # PLAN-INDUCED subset: geometry classes
    for s, name in ((2, "stage2"), (0, "both")):
        m = (Dz.scene == "PLAN-INDUCED") & ((D.stage == s) if s else True)
        out[f"PLAN_INDUCED_geometry_class_shares_{name}"] = share_cis(Dz, "log", "cls", m.values, classes)
    if "nc_class" in D.columns:
        ncc = [c for c in D.nc_class.dropna().unique()]
        for s, name in ((2, "stage2"), (0, "both")):
            m = (D.A1_NC == 0) & D.nc_class.notna() & ((D.stage == s) if s else True)
            out[f"NC_class_shares_{name}"] = share_cis(D.fillna({"nc_class": "NA"}), "log", "nc_class", m.values, ncc)
    # max-speed known / unknown: navhard
    out["vmax_navhard"] = {}
    for s, name in ((2, "stage2"), (1, "stage1"), (0, "both")):
        m = ((D.stage == s) if s else np.ones(len(D), bool))
        for term in ("DAC", "NC"):
            out["vmax_navhard"][f"{name}_{term}"] = {
                "unknown_minus_known_A1_zero_rate": rate_diff(D, "log", (~D.vmax_known) & m, D.vmax_known & m, D[f"A1_{term}"], D[f"A1_{term}"]),
                "unknown_minus_known_STOP_zero_rate": rate_diff(D, "log", (~D.vmax_known) & m, D.vmax_known & m, D[f"STOP_{term}"], D[f"STOP_{term}"]),
                "difference_in_differences_A1_vs_STOP": did(D, "log", term, "A1", stage_mask=m.values if hasattr(m, "values") else m),
                "difference_in_differences_A1_vs_PRIOR": did(D, "log", term, "A1", stage_mask=m.values if hasattr(m, "values") else m, ref="PRIOR")}
    # navtest
    N = pd.read_csv(os.path.join(RAW, "d6_navtest_geometry_step30000.csv")).set_index("token")
    nt = json.load(gzip.open(C.__dict__.get("NT", "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"), "rt", encoding="utf-8"))["tokens"]
    N["log"] = [nt[t]["log_name"] for t in N.index]
    del nt
    out["n_logs_navtest"] = int(N.log.nunique())
    out["vmax_navtest"] = {}
    for term in ("DAC", "NC"):
        out["vmax_navtest"][term] = {
            "unknown_minus_known_A1_zero_rate": rate_diff(N, "log", ~N.vmax_known, N.vmax_known, N[f"A1_{term}"], N[f"A1_{term}"]),
            "unknown_minus_known_STOP_zero_rate": rate_diff(N, "log", ~N.vmax_known, N.vmax_known, N[f"STOP_{term}"], N[f"STOP_{term}"]),
            "difference_in_differences_A1_vs_STOP": did(N, "log", term, "A1"),
            "difference_in_differences_A1_vs_PRIOR": did(N, "log", term, "A1", ref="PRIOR")}
    ncls = ["ON-HUMAN", "ROUTE-FOLLOWING", "OVER-STEER", "LATERAL-DRIFT", "SPEED", "WRONG-SIDE", "STOP-LIKE"]
    out["DAC_navtest_class_shares"] = share_cis(N, "log", "cls_A1", (N.A1_DAC == 0).values, ncls)
    out["DAC_navtest_class_shares_5k"] = share_cis(N, "log", "cls_A1k5", (N.A1k5_DAC == 0).values, ncls)
    json.dump(out, open(os.path.join(RAW, "d6_part4_intervals.json"), "w", encoding="utf-8"), indent=1)
    for k in ("vmax_navhard", "vmax_navtest"):
        print(k, json.dumps(out[k], default=float)[:1500])


if __name__ == "__main__":
    main()
