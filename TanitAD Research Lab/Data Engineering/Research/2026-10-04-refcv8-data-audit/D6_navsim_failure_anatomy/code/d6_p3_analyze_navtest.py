"""D6 P3 analysis on NAVTEST -- exact re-score (navsim-1.1 devkit) + perfect-fix oracles.   tanitad venv.
Reads raw/p3_navtest.jsonl (d6_p3_navtest.py), raw/d6_navtest_geometry_step30000.csv, the navtest inputs export (logs).  Writes raw/d6_p3_navtest.json.
Estimator: log-cluster bootstrap (navtest logs), B = 2000, seed 0.  Counterfactual PDMS rows hold every un-edited sub-score at its banked value (ESTIMATED).
"""
from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "raw")
NT = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"


def boot_share(flags, logs, B=2000, seed=0):
    ul = logs.unique()
    k = pd.Series(flags.values.astype(int), index=logs.values).groupby(level=0).sum().reindex(ul).values
    n = pd.Series(np.ones(len(flags), int), index=logs.values).groupby(level=0).sum().reindex(ul).values
    rng = np.random.default_rng(seed)
    res = np.empty(B)
    for b in range(B):
        i = rng.integers(0, len(ul), len(ul))
        res[b] = k[i].sum() / max(n[i].sum(), 1)
    return float(k.sum() / n.sum()), [float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))]


def main():
    N = pd.read_csv(os.path.join(RAW, "d6_navtest_geometry_step30000.csv")).set_index("token")
    inp = json.load(gzip.open(NT, "rt", encoding="utf-8"))["tokens"]
    logs_all = pd.Series({t: r["log_name"] for t, r in inp.items()})
    del inp
    recs = {}
    for ln in open(os.path.join(RAW, "p3_navtest.jsonl"), encoding="utf-8"):
        try:
            r = json.loads(ln)
        except Exception:                                           # noqa: BLE001
            continue
        recs[r["token"]] = r
    ok = {t: r for t, r in recs.items() if r["status"] == "OK"}
    out = {"n_records": len(recs), "n_ok": len(ok), "n_err": len(recs) - len(ok)}
    # ---- controls
    out["controls"] = {"rescore_repro_max_abs_all": float(max(r["repro_max_abs"] for r in ok.values())),
                       "rescore_repro_exact_share": float(np.mean([r["repro_max_abs"] <= 1e-9 for r in ok.values()])),
                       "identity_snap_DAC_equals_banked": float(np.mean([r["snap"]["ID"]["dac"] == N.loc[t, "A1_DAC"] for t, r in ok.items()]))}
    out["controls"]["PASS"] = bool(out["controls"]["rescore_repro_max_abs_all"] <= 1e-9 and out["controls"]["identity_snap_DAC_equals_banked"] == 1.0)
    D = N.loc[list(ok)].copy()
    D["log"] = logs_all[D.index]
    dz = D[D.A1_DAC == 0]
    nz = D[D.A1_NC == 0]
    out["n_DAC_zero_scored"] = int(len(dz))
    out["n_NC_zero_scored"] = int(len(nz))
    # ---- DAC anatomy: first non-drivable instant, clean path
    nd = pd.Series({t: ok[t]["nd_first"] for t in dz.index})
    bins = pd.cut(nd * 0.1, [-0.01, 0.05, 1.0, 2.0, 3.0, 4.01], labels=["t=0", "(0,1]", "(1,2]", "(2,3]", "(3,4]"]).astype(str)
    out["DAC_first_nondrivable_time_s"] = {k: int(v) for k, v in bins.value_counts().items()}
    out["DAC_t0_violations"] = int((nd == 0).sum())
    out["DAC_clean_path_share"] = float(np.mean([ok[t]["ref_dac"] == 1 for t in dz.index]))
    out["DAC_ref_also_fails"] = int(sum(ok[t]["ref_dac"] == 0 for t in dz.index))
    # ---- NC anatomy
    ncl = {}
    for t in nz.index:
        ev = [e for e in ok[t]["nc_events"] if e["at_fault"]]
        if not ev:
            c = "NO-EVENT"
        else:
            e = ev[0]
            vru = e["obj_type"] in ("PEDESTRIAN", "BICYCLE")
            c = ("INITIAL-OVERLAP" if e["t_idx"] <= 1 else "LATERAL-AFTER-LANE-DEPARTURE" if e["ctype"] == "ACTIVE_LATERAL_COLLISION" else
                 ("STOPPED-TRACK-" + ("VRU" if vru else "VEHICLE/OBJECT")) if e["ctype"] == "STOPPED_TRACK_COLLISION" else ("ACTIVE-FRONT-" + ("VRU" if vru else "VEHICLE")))
        ncl[t] = c
    S = pd.Series(ncl)
    out["NC_classes"] = {k: {"n": int(v), "share": float(v / len(S))} for k, v in S.value_counts().items()}
    fe = [ok[t]["nc_events"] for t in nz.index]
    out["NC_front_collision_share"] = float(np.mean([c in ("ACTIVE-FRONT-VEHICLE", "ACTIVE-FRONT-VRU", "STOPPED-TRACK-VEHICLE/OBJECT", "STOPPED-TRACK-VRU") for c in S]))
    # ---- speed oracle on NC-zero
    sp = {}
    for s in ("S0.8", "S0.6", "S0.4"):
        cl = pd.Series({t: ok[t]["speed"][s]["no_at_fault_collisions"] == 1 for t in nz.index})
        pt, ci = boot_share(cl, nz["log"])
        sp[s] = {"NC_cleared_share": pt, "ci95_logs": ci, "n_cleared": int(cl.sum()), "n": int(len(cl)),
                 "median_EP_of_cleared": float(np.median([ok[t]["speed"][s]["ego_progress"] for t in nz.index if ok[t]["speed"][s]["no_at_fault_collisions"] == 1])) if cl.any() else None}
    anyc = pd.Series({t: any(ok[t]["speed"][s]["no_at_fault_collisions"] == 1 for s in ("S0.8", "S0.6", "S0.4")) for t in nz.index})
    pt, ci = boot_share(anyc, nz["log"])
    sp["any_scale"] = {"NC_cleared_share": pt, "ci95_logs": ci, "n_cleared": int(anyc.sum()), "n": int(len(anyc))}
    sp["not_fixable_STOP_also_NC_zero"] = int((nz.STOP_NC == 0).sum())
    sp["by_nc_class"] = {c: {"n": int((S == c).sum()), "any_cleared": int(anyc[S[S == c].index].sum())} for c in S.unique()}
    out["speed_oracle_NC"] = sp
    # ---- snap oracle on DAC-zero
    sn = {}
    for v in ("B075", "B150", "FULL", "V80", "V60", "V40"):
        cl = pd.Series({t: ok[t]["snap"][v]["dac"] == 1 for t in dz.index})
        pt, ci = boot_share(cl, dz["log"])
        sn[v] = {"DAC_cleared_share": pt, "ci95_logs": ci, "n_cleared": int(cl.sum()), "n": int(len(cl)),
                 "cleared_in_clean_path": int(sum(cl[t] and ok[t]["ref_dac"] == 1 for t in dz.index)), "n_clean_path": int(sum(ok[t]["ref_dac"] == 1 for t in dz.index))}
    u = pd.Series({t: any(ok[t]["snap"][v]["dac"] == 1 for v in ("FULL", "V60", "V40")) for t in dz.index})
    sn["union_FULL_or_V60_or_V40"] = {"n_cleared": int(u.sum()), "share": float(u.mean())}
    out["snap_oracle_DAC"] = sn
    # ---- PDMS ceilings (ESTIMATED): PDMS = mean NC*DAC*(5EP+5TTC+2CF)/12 over tokens
    def pdms(df):
        return float((df.A1_NC * df.A1_DAC * (5 * df.A1_EP + 5 * df.A1_TTC + 2 * df.A1_CF) / 12).mean() * 100)
    base = pdms(N)
    # (a) oracle speed: per NC-zero token best of {banked, 0.8, 0.6, 0.4} (score from the re-scored row; the official `score` column)
    N2 = N.copy()
    sc = N2["A1_score"].copy()
    for t in nz.index:
        best = N2.loc[t, "A1_score"]
        for s in ("S0.8", "S0.6", "S0.4"):
            best = max(best, ok[t]["speed"][s]["score"])
        sc[t] = best
    N2["A1_score"] = sc
    out["PDMS_x100"] = {"A1_banked": float(N["A1_score"].mean() * 100), "STOP": float(N["STOP_score"].mean() * 100),
                        "oracle_speed_on_NC0": float(N2["A1_score"].mean() * 100),
                        "DAC_to_1_where_FULL_snap_clears": None}
    N3 = N.copy()
    idx = [t for t in dz.index if ok[t]["snap"]["FULL"]["dac"] == 1]
    N3.loc[idx, "A1_DAC"] = 1.0
    N3["A1_score"] = N3.A1_NC * N3.A1_DAC * (5 * N3.A1_EP + 5 * N3.A1_TTC + 2 * N3.A1_CF) / 12
    out["PDMS_x100"]["DAC_to_1_where_FULL_snap_clears"] = float(N3["A1_score"].mean() * 100)
    json.dump(out, open(os.path.join(RAW, "d6_p3_navtest.json"), "w", encoding="utf-8"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({"controls": out["controls"], "speed": {k: (round(v["NC_cleared_share"], 3)) for k, v in sp.items() if isinstance(v, dict) and "NC_cleared_share" in v},
                      "snap": {k: round(v["DAC_cleared_share"], 3) for k, v in sn.items() if "DAC_cleared_share" in v}, "PDMS": out["PDMS_x100"], "NCfront": out["NC_front_collision_share"]}, indent=1))


if __name__ == "__main__":
    main()
