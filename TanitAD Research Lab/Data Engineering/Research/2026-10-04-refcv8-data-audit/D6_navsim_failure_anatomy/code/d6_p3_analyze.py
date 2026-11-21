"""D6 P3 analysis -- what share of the failing scenes would a PERFECT fix of each kind clear?   tanitad venv.

Reads raw/p3_controls.jsonl, raw/p3_speed_nc0_s{0,1}.jsonl, raw/p3_snap_dac0.jsonl (d6_p3_oracles.py), the part-2 geometry table and the banked frames.
Writes raw/d6_p3_navhard.json and raw/d6_p3_tables.md.
Estimator for every interval: log-cluster bootstrap (76 navhard logs, B = 2000, seed 0).  All counterfactual EPDMS rows hold every un-edited
sub-score (and the stage-2 weights) at their banked values -> ESTIMATED upper bounds, not results.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d6_common as C

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "raw")
KEYS = {"no_at_fault_collisions": "NC", "drivable_area_compliance": "DAC", "driving_direction_compliance": "DDC", "traffic_light_compliance": "TLC",
        "ego_progress": "EP", "time_to_collision_within_bound": "TTC", "lane_keeping": "LK", "history_comfort": "HC"}


def load(pattern):
    out = {}
    for p in sorted(glob.glob(os.path.join(RAW, pattern))):
        for l in open(p, encoding="utf-8"):
            try:
                r = json.loads(l)
            except Exception:                                       # noqa: BLE001
                continue
            out[r["token"]] = r
    return out


def boot_share(flags: pd.Series, logs: pd.Series, B=2000, seed=0):
    """share of True in flags, log-cluster 95 % CI."""
    rng = np.random.default_rng(seed)
    ul = logs.unique()
    k = pd.Series(flags.values.astype(int), index=logs.values).groupby(level=0).sum().reindex(ul).values
    n = pd.Series(np.ones(len(flags), int), index=logs.values).groupby(level=0).sum().reindex(ul).values
    pt = k.sum() / n.sum()
    res = np.empty(B)
    for b in range(B):
        i = rng.integers(0, len(ul), len(ul))
        res[b] = k[i].sum() / max(n[i].sum(), 1)
    return float(pt), [float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))]


def main():
    ci = C.load_ci()
    inp, mapping = C.load_inputs()
    G = pd.read_csv(os.path.join(RAW, "d6_scene_geometry_step30000.csv")).set_index("token")
    T = pd.read_csv(os.path.join(RAW, "d6_scene_table_step30000.csv")).set_index("token")
    A1 = C.load_arm(30000, "R7_A1", with_hooks=False)
    out = {}
    # ------------------------------------------------------------------ controls
    ctl = load("p3_controls.jsonl")
    stop = C.load_floor("STOP_zero")
    mx_id = mx_stop = 0.0
    for t, r in ctl.items():
        v = r["variants"]
        mx_id = max(mx_id, max(abs(v["S1.0"][k] - r["banked"][k]) for k in r["banked"]))
        mx_stop = max(mx_stop, max(abs(v["S0.0"][k] - float(stop.loc[t, KEYS[k]])) for k in KEYS))
    out["controls"] = {"n_tokens": len(ctl), "identity_plan_max_abs_vs_banked": mx_id, "scale0_vs_official_STOP_frame_max_abs": mx_stop,
                       "PASS": bool(mx_id <= 1e-9 and mx_stop <= 1e-9)}
    # ------------------------------------------------------------------ SPEED oracle on NC-zero
    sp = load("p3_speed_nc0_s*.jsonl")
    nz = [t for t in A1.index if A1.loc[t, "NC"] == 0]
    out["speed"] = {"n_NC_zero": len(nz), "n_scored": len(sp), "n_err": int(sum(1 for r in sp.values() if r["status"] != "OK"))}
    sp_ok = {t: r for t, r in sp.items() if r["status"] == "OK"}
    nc_class = G["nc_class"] if "nc_class" in G.columns else pd.Series(index=G.index, dtype=object)
    rows = []
    for t, r in sp_ok.items():
        v = r["variants"]
        rec = {"token": t, "stage": inp[t]["stage"], "log": inp[t]["log_name"], "nc_class": nc_class.get(t), "STOP_NC": float(stop.loc[t, "NC"]),
               "banked_DAC": r["banked"]["drivable_area_compliance"], "banked_EP": r["banked"]["ego_progress"]}
        for s in ("S0.8", "S0.6", "S0.4"):
            rec[f"{s}_NC"] = v[s]["no_at_fault_collisions"]
            rec[f"{s}_DAC"] = v[s]["drivable_area_compliance"]
            rec[f"{s}_DDC"] = v[s]["driving_direction_compliance"]
            rec[f"{s}_EP"] = v[s]["ego_progress"]
            rec[f"{s}_end"] = v[s]["end_dist"]
        rows.append(rec)
    S = pd.DataFrame(rows).set_index("token")
    if len(S):
        logs = S["log"]
        sp_res = {}
        for s in ("S0.8", "S0.6", "S0.4"):
            cl = S[f"{s}_NC"] == 1
            pt, c = boot_share(cl, logs)
            clean = (S[f"{s}_NC"] == 1) & (S[f"{s}_DAC"] >= S["banked_DAC"])
            sp_res[s] = {"NC_cleared_share": pt, "ci95_logs": c, "n_cleared": int(cl.sum()), "n": int(len(S)),
                         "cleared_and_DAC_not_worse": int(clean.sum()),
                         "median_end_dist_of_cleared": float(S.loc[cl, f"{s}_end"].median()) if cl.any() else None,
                         "median_end_dist_all": float(S[f"{s}_end"].median()),
                         "median_banked_end_dist": float(T.loc[S.index, "plan_end_dist"].median()),
                         "median_reference_end_dist": float(G.loc[S.index, "d_ref"].median()),
                         "cleared_with_end_dist_ge_half_reference": int((cl & (S[f"{s}_end"] >= 0.5 * G.loc[S.index, "d_ref"])).sum())}
        anyc = (S[["S0.8_NC", "S0.6_NC", "S0.4_NC"]] == 1).any(axis=1)
        pt, c = boot_share(anyc, logs)
        sp_res["any_scale"] = {"NC_cleared_share": pt, "ci95_logs": c, "n_cleared": int(anyc.sum()), "n": int(len(S))}
        unfix = (S["STOP_NC"] == 0)
        sp_res["not_fixable_by_any_speed_STOP_also_NC_zero"] = int(unfix.sum())
        sp_res["cleared_any_given_STOP_passes"] = {"n": int((~unfix).sum()), "cleared": int((anyc & ~unfix).sum()),
                                                  "share": float((anyc & ~unfix).sum() / max((~unfix).sum(), 1))}
        sp_res["by_stage"] = {str(st): {"n": int((S.stage == st).sum()), "any_cleared": int((anyc & (S.stage == st)).sum())} for st in (1, 2)}
        if S["nc_class"].notna().any():
            sp_res["by_nc_class"] = {c: {"n": int(len(g)), "any_cleared": int(anyc[g.index].sum()),
                                         "S0.8": int((g["S0.8_NC"] == 1).sum()), "S0.6": int((g["S0.6_NC"] == 1).sum()), "S0.4": int((g["S0.4_NC"] == 1).sum())}
                                     for c, g in S.groupby("nc_class")}
        out["speed"]["results"] = sp_res
        # official-EPDMS ceiling (ESTIMATED): oracle per-token choice among the original and the three scales, EC held
        A = A1.copy()
        sc_new = A["score"].copy()
        for t, r in sp_ok.items():
            cands = [("orig", r["banked"])] + [(s, r["variants"][s]) for s in ("S0.8", "S0.6", "S0.4")]
            best = -1.0
            for nm, row in cands:
                m = row["no_at_fault_collisions"] * row["drivable_area_compliance"] * row["driving_direction_compliance"] * row["traffic_light_compliance"]
                w = (5 * row["ego_progress"] + 5 * row["time_to_collision_within_bound"] + 2 * row["lane_keeping"] + 2 * row["history_comfort"] + 2 * A.loc[t, "EC"]) / 16.0
                best = max(best, m * w)
            sc_new[t] = best
        A["score"] = sc_new
        base = C.official_epdms(A1, mapping, ci)
        new = C.official_epdms(A, mapping, ci)
        out["speed"]["official_epdms_oracle_speed_on_NC0_tokens"] = {"base": base, "oracle": new, "delta": new - base,
                                                                    "note": "per-token best of {banked, x0.8, x0.6, x0.4}; EC and stage-2 weights held; scored tokens only", "n_tokens_edited": len(sp_ok)}
    # ------------------------------------------------------------------ SNAP oracle on DAC-zero (map-only)
    sn = load("p3_snap_dac0.jsonl")
    sn_ok = {t: r for t, r in sn.items() if r["status"] == "OK"}
    out["snap"] = {"n_DAC_zero": int((A1.DAC == 0).sum()), "n_scored": len(sn), "n_ok": len(sn_ok)}
    D = pd.DataFrame({"stage": {t: inp[t]["stage"] for t in sn_ok}, "log": {t: inp[t]["log_name"] for t in sn_ok}})
    for v in ("ID", "B075", "B150", "FULL", "V80", "V60", "V40"):
        D[f"{v}_dac"] = {t: r["variants"][v]["dac"] for t, r in sn_ok.items()}
        D[f"{v}_ddc"] = {t: r["variants"][v]["ddc"] for t, r in sn_ok.items()}
    D["ref_dac"] = G.loc[D.index, "ref_dac"]
    D["cls"] = G.loc[D.index, "cls"]
    D["route_turn"] = G.loc[D.index, "route_turn"]
    out["snap"]["control_ID_dac_is_zero_on_all"] = bool((D["ID_dac"] == 0).all())
    sres = {}
    for v in ("B075", "B150", "FULL", "V80", "V60", "V40"):
        cl = (D[f"{v}_dac"] == 1)
        pt, c = boot_share(cl, D["log"])
        cl_all4 = cl & (D[f"{v}_ddc"] == 1)
        sres[v] = {"DAC_cleared_share": pt, "ci95_logs": c, "n_cleared": int(cl.sum()), "n": int(len(D)),
                   "cleared_and_DDC_ok": int(cl_all4.sum()),
                   "cleared_in_clean_path_scenes": {"n": int((D.ref_dac == 1).sum()), "cleared": int((cl & (D.ref_dac == 1)).sum())},
                   "cleared_in_ref_fails_scenes": {"n": int((D.ref_dac == 0).sum()), "cleared": int((cl & (D.ref_dac == 0)).sum())},
                   "by_stage": {str(st): {"n": int((D.stage == st).sum()), "cleared": int((cl & (D.stage == st)).sum())} for st in (1, 2)},
                   "by_class": {c_: {"n": int(len(g)), "cleared": int(cl[g.index].sum())} for c_, g in D.groupby("cls")}}
    out["snap"]["results"] = sres
    # union: cleared by FULL snap OR by a speed variant
    u = (D["FULL_dac"] == 1) | (D["V60_dac"] == 1) | (D["V40_dac"] == 1)
    out["snap"]["union_FULL_or_V60_or_V40"] = {"n_cleared": int(u.sum()), "share": float(u.mean())}
    # official-EPDMS ceiling for DAC clearance by variant (T1 style: DAC := 1 on cleared tokens, others held)
    cf = {}
    for v in ("B075", "B150", "FULL", "V60"):
        Ax = A1.copy()
        idx = D.index[D[f"{v}_dac"] == 1]
        Ax.loc[idx, "DAC"] = 1.0
        Ax["score"] = Ax[list(C.MULT)].prod(axis=1) * sum(Ax[k] * w for k, w in C.WEIGHTED.items()) / sum(C.WEIGHTED.values())
        cf[v] = {"official_epdms": C.official_epdms(Ax, mapping, ci), "delta_vs_A1": C.official_epdms(Ax, mapping, ci) - C.official_epdms(A1, mapping, ci)}
    out["snap"]["official_epdms_DAC_cleared_by_variant"] = cf
    json.dump(out, open(os.path.join(RAW, "d6_p3_navhard.json"), "w", encoding="utf-8"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: v for k, v in out["controls"].items()}))
    if "results" in out["speed"]:
        print({k: (round(v["NC_cleared_share"], 3), v["n_cleared"]) for k, v in out["speed"]["results"].items() if isinstance(v, dict) and "NC_cleared_share" in v})
    print({k: (round(v["DAC_cleared_share"], 3), v["n_cleared"]) for k, v in sres.items()}, out["snap"]["union_FULL_or_V60_or_V40"])


if __name__ == "__main__":
    main()
