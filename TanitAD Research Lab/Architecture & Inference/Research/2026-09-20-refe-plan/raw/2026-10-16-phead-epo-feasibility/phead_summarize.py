#!/usr/bin/env python3
"""PHEAD summary: f, the safety proxies and the row of PREREG_PHEAD.md s.8, from closed_out/*.pkl, full_ref.pkl, open_*.pkl, proxy_validity.json.  Writes summary.json and prints the tables."""
import glob, json, os, pickle, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import phead_common as PC

FR = pickle.load(open(os.path.join(HERE, "full_ref.pkl"), "rb"))
LOGS = FR["logs"]; POOLS = ("train", "val")
LOGS_ALL = np.concatenate([LOGS["train"], LOGS["val"]])
PV = json.load(open(os.path.join(HERE, "proxy_validity.json")))["verdict"]["proxy_validated"] if os.path.exists(os.path.join(HERE, "proxy_validity.json")) else None
COLS = {"TTC": 3, "NC": 0, "DAC": 1}


def closed_runs():
    d = {}
    for sub in ("closed_cached", "closed_out"):                     # the real pipeline (closed_out) overrides the cached-feature path where both exist
        for p in glob.glob(os.path.join(HERE, sub, "closed_s0_P_*.pkl")):
            m = re.match(r"closed_s0_P_(.+)_(R|E)_lr([0-9.]+)_lam([0-9.]+)(.*)\.pkl", os.path.basename(p))
            if m and m.group(5) == "":
                r = pickle.load(open(p, "rb"))
                r["source"] = sub
                d[(m.group(1), m.group(2), float(m.group(3)), float(m.group(4)))] = r
    return d


def prox(a1, a0):
    """a1, a0: {pool: eval dict} -> {metric: {pool means x100, pooled mean + CI}} for TTC / NC / DAC under P1 (pick) and P2 (policy expectation)"""
    out = {}
    for nm, col in COLS.items():
        for pn_, key in (("P1", "pick_comp"), ("P2", "pi_comp")):
            per = {p: float(((a1[p][key][:, col] - a0[p][key][:, col]) * 100).mean()) for p in POOLS}
            d = np.concatenate([(a1[p][key][:, col] - a0[p][key][:, col]) * 100 for p in POOLS])
            b = PC.boot_ratio(LOGS_ALL, d)
            out[f"{pn_}_{nm}"] = {**per, "pooled": b["mean_num"], "pooled_ci": b["ci_num"]}
    return out


def row_of(fmin, ttc_ok):
    if ttc_ok is False:
        return "R3 CANNOT CARRY (TTC proxy worse)"
    base = "R1 CARRIES" if fmin >= 0.5 else ("R2 PARTIAL" if fmin >= 0.2 else "R3 CANNOT CARRY (f<0.2)")
    if ttc_ok is None:
        return base + " (TTC unverifiable in sim; rests on K3)"
    return base


def main():
    runs = closed_runs()
    res = {"proxy_validated": PV, "closed": {}, "open": {}}
    for K in (25, 50):
        fz1, fz0 = FR[(1.0, K)], FR[(0.0, K)]
        dF = {p: fz1[p]["J"] - fz0[p]["J"] for p in POOLS}
        res["closed"][f"K{K}"] = {"FULL": {"dJ": {p: PC.boot_ratio(LOGS[p], dF[p]) for p in POOLS}, "T_match": prox(fz1, fz0), "dJ_pooled": float(np.concatenate([dF[p] for p in POOLS]).mean()),
                                          "J_theta0": {p: float(FR["theta0"][p]["J"].mean()) for p in POOLS},
                                          "J_lam0": {p: float(fz0[p]["J"].mean()) for p in POOLS}, "J_lam1": {p: float(fz1[p]["J"].mean()) for p in POOLS}}}
        for (g, kind, lrm, lam), r in sorted(runs.items()):
            if lam != 1.0 or K not in r["snaps"]:
                continue
            e1 = r["snaps"][K]["eval"]
            if kind == "R":
                if (g, "R", lrm, 0.0) not in runs:
                    continue
                e0 = runs[(g, "R", lrm, 0.0)]["snaps"][K]["eval"]
            else:
                e0 = fz0
            dG = {p: e1[p]["J"] - e0[p]["J"] for p in POOLS}
            fb = {p: PC.boot_ratio(LOGS[p], dG[p], dF[p]) for p in POOLS}
            fA = {p: PC.boot_ratio(LOGS[p], e1[p]["J"] - fz0[p]["J"], dF[p]) for p in POOLS}
            fmin = min(fb[p]["f"] for p in POOLS)
            tm, ta = prox(e1, e0), prox(e1, fz0)
            ttc_ok = None if not PV else bool(all(tm[f"{m}_TTC"][p] >= -0.5 for m in ("P1", "P2") for p in POOLS))
            nc_ok = bool(all(tm[f"{m}_NC"][p] >= -0.5 for m in ("P1", "P2") for p in POOLS))
            res["closed"][f"K{K}"][f"{g}|{kind}|lr{lrm:g}"] = {
                "dJ_G": {p: {"mean": fb[p]["mean_num"], "ci": fb[p]["ci_num"]} for p in POOLS}, "f": {p: {"f": fb[p]["f"], "ci": fb[p]["ci_f"], "den_straddles_zero": fb[p]["den_straddles_zero"]} for p in POOLS},
                "f_min": fmin, "f_A": {p: {"f": fA[p]["f"], "ci": fA[p]["ci_f"]} for p in POOLS},
                "J_G_lam1": {p: float(e1[p]["J"].mean()) for p in POOLS}, "J_G_ctrl": {p: float(e0[p]["J"].mean()) for p in POOLS},
                "source": r["source"], "disp_norm": r["snaps"][K]["disp_norm"], "outside_group_max_abs": r["snaps"][K]["outside"],
                "kl_to_base_lam1": {p: float(e1[p]["kl"].mean()) for p in POOLS},
                "T_match": tm, "T_A": ta, "dJ_pooled": float(np.concatenate([dG[p] for p in POOLS]).mean()),
                "P2_TTC_x100_per_unit_dJ": tm["P2_TTC"]["pooled"] / float(np.concatenate([dG[p] for p in POOLS]).mean()), "ttc_not_worse_match": ttc_ok, "nc_not_worse_match": nc_ok, "row": row_of(fmin, ttc_ok)}
    # the bit-level checks that the closed-loop restriction held
    res["closed_outside_checks"] = {f"{g}|{kind}|lr{lrm:g}|lam{lam:g}": r["snaps"][max(r["snaps"])]["outside"] for (g, kind, lrm, lam), r in runs.items() if kind == "R"}
    # open loop
    o = {}
    for pth in sorted(glob.glob(os.path.join(HERE, "open_out", "open_*.pkl"))):
        o.update(pickle.load(open(pth, "rb"))["runs"])
    if True:
        ks = sorted({(s, rg, g, kind, lrm, k) for (s, rg, g, kind, lrm, lam, k), v in o.items() if "eval" in v})
        for (s, rg, g, kind, lrm, k) in ks:
            r1, r0 = o.get((s, rg, g, kind, lrm, 1.0, k)), (o.get((s, rg, g, kind, lrm, 0.0, k)) if kind == "R" else o.get((s, rg, "FULL", "R", 1.0, 0.0, k)))
            f1, f0 = o.get((s, rg, "FULL", "R", 1.0, 1.0, k)), o.get((s, rg, "FULL", "R", 1.0, 0.0, k))
            if not (r1 and r0 and f1 and f0 and "eval" in r1 and "eval" in r0 and "eval" in f1 and "eval" in f0):
                continue
            dG = {p: r1["eval"][p]["J"] - r0["eval"][p]["J"] for p in POOLS}
            dF = {p: f1["eval"][p]["J"] - f0["eval"][p]["J"] for p in POOLS}
            fb = {p: PC.boot_ratio(LOGS[p], dG[p], dF[p]) for p in POOLS}
            res["open"][f"s{s}|{rg}|{g}|{kind}|lr{lrm:g}|K{k}"] = {"f": {p: fb[p]["f"] for p in POOLS}, "f_min": min(fb[p]["f"] for p in POOLS),
                                                                  "dJ_G": {p: fb[p]["mean_num"] for p in POOLS}, "dJ_FULL": {p: fb[p]["mean_den"] for p in POOLS},
                                                                  "T_match_P2_TTC": prox(r1["eval"], r0["eval"])["P2_TTC"]["pooled"]}
    json.dump(res, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
    for K in (25, 50):
        print(f"--- closed loop, regime P, K={K}  (FULL dJ train {res['closed'][f'K{K}']['FULL']['dJ']['train']['mean_num']:+.4f}  val {res['closed'][f'K{K}']['FULL']['dJ']['val']['mean_num']:+.4f})")
        fl = res["closed"][f"K{K}"]["FULL"]
        print(f"FULL: TTC match P1 {fl['T_match']['P1_TTC']['pooled']:+.2f} P2 {fl['T_match']['P2_TTC']['pooled']:+.2f} (P2/dJ {fl['T_match']['P2_TTC']['pooled']/fl['dJ_pooled']:+.1f}) NC P2 {fl['T_match']['P2_NC']['pooled']:+.2f}")
        for k, v in res["closed"][f"K{K}"].items():
            if k == "FULL":
                continue
            print(f"{k:28s} dJ train {v['dJ_G']['train']['mean']:+.4f} val {v['dJ_G']['val']['mean']:+.4f} | f train {v['f']['train']['f']:+.3f} val {v['f']['val']['f']:+.3f} fmin {v['f_min']:+.3f} | "
                  f"TTC match P1 {v['T_match']['P1_TTC']['pooled']:+.2f} P2 {v['T_match']['P2_TTC']['pooled']:+.2f} (P2/dJ {v['P2_TTC_x100_per_unit_dJ']:+.1f}) | {v['row']}")
    print("open-loop f_min:", {k: round(v["f_min"], 3) for k, v in res["open"].items()})


if __name__ == "__main__":
    main()
