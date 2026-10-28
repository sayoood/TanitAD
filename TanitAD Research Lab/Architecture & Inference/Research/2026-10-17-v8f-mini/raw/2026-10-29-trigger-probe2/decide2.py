#!/usr/bin/env python3
"""decide2.py -- applies the COMMITTED decision rules of PREREG_FROZEN.md 7 to the banked results, evaluates the controls (section 8) and prints the tables.  Writes raw/VERDICT.json.
Nothing is tuned here: thresholds are the frozen ones.   CUDA_VISIBLE_DEVICES=-1  python -I decide2.py"""
from __future__ import annotations

import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
warnings.filterwarnings("ignore")
import numpy as np

import common2 as C  # noqa: E402

L = C.L


def J(f):
    p = f"{C.RAW}/{f}"
    return json.load(open(p)) if os.path.exists(p) else None


def znn_rule(delta, lo):
    if delta >= 0.05 and lo > 0:
        return "VISION-THROUGH-TRUNK"
    if delta < 0.02:
        return "NAV-ONLY"
    return "INCONCLUSIVE"


def v4_rule(delta, lo):
    if delta >= 0.03 and lo > 0:
        return "MOTION"
    if delta < 0.01:
        return "NO-MOTION"
    return "INCONCLUSIVE"


def covers_half(a):
    return a["lo"] <= 0.5 <= a["hi"]


def controls(pop):
    d = C.load_pop(pop)
    y, clip, outer = d["y"], d["clip"], d["outer"]
    counts = L.boot_counts(clip)
    out = {}
    ov = np.load(f"{C.OLDRAW}/oof_{pop}_V_linear.npz")
    r1 = C.rec(ov["C1__shuffle__E"], L.shuffle_labels(y, 1), clip, outer, counts)["fold"]
    r2 = C.rec(ov["C2__donor__E"], L.donor_labels(y, clip, 2), clip, outer, counts)["fold"]
    eE = C.rec(ov["E"], y, clip, outer, counts)["fold"]
    out["E"] = {"C1_shuffle": {**r1, "covers_0.5": covers_half(r1), "pass": covers_half(r1)}, "C2_donor_labels": {**r2, "covers_0.5": covers_half(r2), "not_above_true": r2["auc"] <= eE["auc"], "pass": covers_half(r2) and r2["auc"] <= eE["auc"]}}
    for src in ("Znn", "Znl", "V4", "Vdiff", "nav"):
        res = J(f"res2_{src}_{pop}_linear.json" if src != "nav" else f"res2_nav_{pop}_linear.json")
        if res is None:
            continue
        key = "N5" if src == "nav" else src
        A = res["arms"]
        o = {}
        c1 = A.get(f"C1:shuffle:{key}+E")
        c3 = A.get(f"C3:leak:{key}+E")
        c2 = A.get(f"C2:donor:{key}+E")
        true = A.get(f"{key}+E")
        if c1:
            o["C1_shuffle"] = {**c1["fold"], "pass": covers_half(c1["fold"])}
        if c2 and true:
            o["C2_donor_labels"] = {**c2["fold"], "not_above_true": c2["fold"]["auc"] <= true["fold"]["auc"], "pass": covers_half(c2["fold"]) and c2["fold"]["auc"] <= true["fold"]["auc"]}
        if c3:
            o["C3_leak"] = {**c3["fold"], "pass": c3["fold"]["auc"] >= 0.99}
        out[key] = o
    return out


def band_reading(pop, P):
    """POST-FREEZE (raw/POSTFREEZE_LOG.md 5): C2 read against the null band of 10 further donor draws (seeds 3..12, c2_band.py) plus the frozen seed 2.  Never decisive."""
    b = J(f"c2_band_{pop}.json")
    if b is None:
        return {}
    out = {}
    for key, armkey in (("E", "E"), ("Znn", "Znn+E"), ("Znl", "Znl+E"), ("V4", "V4+E"), ("Vdiff", "Vdiff+E")):
        if armkey not in b["arms"] or key not in P["controls"] or "C2_donor_labels" not in P["controls"][key]:
            continue
        seed2 = P["controls"][key]["C2_donor_labels"]["auc"]
        allv = np.array([seed2] + b["arms"][armkey]["aucs"])
        se = allv.std(ddof=1) / np.sqrt(len(allv))
        t = float((allv.mean() - 0.5) / se)
        res = J(f"res2_{key}_{pop}_linear.json") if key != "E" else None
        true_auc = (res["arms"][f"{key}+E"]["fold"]["auc"] if res else C.rec(np.load(f"{C.OLDRAW}/oof_{pop}_V_linear.npz")["E"], C.load_pop(pop)["y"], C.load_pop(pop)["clip"], C.load_pop(pop)["outer"], L.boot_counts(C.load_pop(pop)["clip"]))["fold"]["auc"])
        out[key] = {"n_draws": int(len(allv)), "mean": float(allv.mean()), "sd": float(allv.std(ddof=1)), "min": float(allv.min()), "max": float(allv.max()), "seed2_value": float(seed2), "seed2_is_minimum": bool(seed2 == allv.min()),
                    "t_of_mean_vs_0.5": t, "true_arm_auc": float(true_auc), "true_minus_null_max": float(true_auc - allv.max()), "band_centred_within_0.03_of_chance": bool(abs(allv.mean() - 0.5) < 0.03), "band_pass": bool(abs(allv.mean() - 0.5) < 0.03 and true_auc > allv.max())}
    return out


def main():
    V = {"frozen_sha256": open(f"{C.HERE}/FROZEN_SHA.txt").read().split()[2], "pops": {}}
    n1 = J("control_znn.json")
    v1 = J("control_v4.json")
    ev_ctl = J("control_eval2.json")
    ev2 = J("results_eval2.json")
    V["control_N1_N2"] = n1
    V["control_V1"] = v1
    V["control_eval2"] = ev_ctl
    n1_pass = bool(n1 and n1["N1a_true_forward_vs_banked_Z_max_abs"] <= 1e-5 and n1["N1b_selfdonor_swap_vs_banked_Z_max_abs"] <= 1e-5)
    n2_pass = bool(n1 and n1["N2_nn_windows_nav_differs"]["frac_relL2_gt_0"] is not None and n1["N2_nn_windows_nav_differs"]["frac_relL2_gt_0"] >= 0.99 and n1["N2_nn_windows_nav_differs"]["median_max_abs_dtraj60_m"] > 0)
    v1_pass = bool(v1 and all(p["max_abs_frame_t_old_vs_cache"] == 0.0 and p["share_rows_where_lag_frame_differs_from_t"]["3"] >= 0.99 and p["all_zero_lag3_rows"] == 0 for p in v1["pops"].values()))
    V["control_pass"] = {"N1_identity": n1_pass, "N2_reachability": n2_pass, "V1_frames": v1_pass}
    for pop in C.POPS:
        P = {"controls": controls(pop)}
        P["postfreeze_c2_band"] = band_reading(pop, P)
        # ---- Z_nn
        z = J(f"results2_Znn_{pop}.json")
        if z:
            lin = z["paired"]["linear"]["Znn+E_minus_E"]["fold"]
            mlp = z["paired"].get("mlp", {}).get("Znn+E_minus_E", {}).get("fold")
            ctl_ok = all(v.get("pass", True) for k, v in P["controls"].get("Znn", {}).items()) and P["controls"]["E"]["C1_shuffle"]["pass"] and P["controls"]["E"]["C2_donor_labels"]["pass"] and n1_pass and n2_pass
            verdict = znn_rule(lin["delta"], lin["lo"]) if ctl_ok else "FAILED-CONTROL"
            zz = {"delta_Znn+E_minus_E_fold": lin, "pooled": z["paired"]["linear"]["Znn+E_minus_E"]["pooled"], "controls_ok": ctl_ok, "verdict_committed": verdict,
                  "dAUC(Z+E - Znn+E)": z["paired"]["linear"]["Z+E_minus_Znn+E"], "dAUC(Znn+E - V+E)": z["paired"]["linear"]["Znn+E_minus_V+E"],
                  "AUC_Znn+E_fold": z["arms"]["linear"]["Znn+E"]["fold"], "AUC_Znn_fold": z["arms"]["linear"].get("Znn", {}).get("fold"), "AUC_E_fold": z["arms"]["linear"]["banked:E"]["fold"], "AUC_Z+E_fold": z["arms"]["linear"]["banked:Z+E"]["fold"]}
            bandp = P["postfreeze_c2_band"].get("Znn", {}).get("band_pass")
            zz["POSTFREEZE_reading_if_C2_is_read_against_its_null_band"] = znn_rule(lin["delta"], lin["lo"]) if (bandp and P["controls"]["E"]["C1_shuffle"]["pass"] and n1_pass and n2_pass) else "n/a"
            if mlp:
                zz["flag_F1_mlp_delta"] = mlp
                zz["flag_F1_not_robust_to_nonlinear_ego"] = bool(verdict == "VISION-THROUGH-TRUNK" and (mlp["delta"] < 0.02 or mlp["lo"] <= 0))
            if ev2:
                e = ev2["pops"][pop]["families"]["linear"]["paired"].get("Znn+E_minus_E")
                if e:
                    zz["eval141_dAUC(Znn+E - E)"] = e
                    zz["flag_F2_memorisation_suspect"] = bool(verdict == "VISION-THROUGH-TRUNK" and e["delta"] < 0.02)
            zl = J(f"results2_Znl_{pop}.json")
            if zl:
                l2 = zl["paired"]["linear"]["Znl+E_minus_E"]["fold"]
                zz["POSTFREEZE_Znl_(nav+limit swapped)_dAUC(Znl+E - E)"] = l2
                zz["POSTFREEZE_Znl_rule_reading"] = znn_rule(l2["delta"], l2["lo"])
                zz["POSTFREEZE_dAUC(Z+E - Znl+E)"] = zl["paired"]["linear"]["Z+E_minus_Znl+E"]
            P["Znn"] = zz
        # ---- V4
        arms = {}
        for src in ("V4", "Vdiff"):
            r = J(f"results2_{src}_{pop}.json")
            if r:
                arms[src] = r
        if len(arms) == 2:
            auc = {s: arms[s]["arms"]["linear"][f"{s}+E"]["fold"]["auc"] for s in arms}
            best = max(auc, key=auc.get)
            b = arms[best]["paired"]["linear"][f"{best}+E_minus_V+E"]["fold"]
            ctl_ok = all(v.get("pass", True) for s in arms for k, v in P["controls"].get(s, {}).items()) and v1_pass
            verdict = v4_rule(b["delta"], b["lo"]) if ctl_ok else "FAILED-CONTROL"
            bandv = P["postfreeze_c2_band"].get("V4", {}).get("band_pass")
            vv = {"best_arm_by_linear_fold_mean_AUC": best, "POSTFREEZE_reading_if_C2_is_read_against_its_null_band": v4_rule(b["delta"], b["lo"]) if (bandv is not False and v1_pass) else "n/a", "AUC_fold": {s: arms[s]["arms"]["linear"][f"{s}+E"]["fold"] for s in arms}, "AUC_V+E_fold": arms[best]["arms"]["linear"]["banked:V+E"]["fold"],
                  "delta_best+E_minus_V+E_fold": b, "controls_ok": ctl_ok, "verdict_committed": verdict,
                  "both_arms_delta_vs_V+E": {s: arms[s]["paired"]["linear"][f"{s}+E_minus_V+E"]["fold"] for s in arms},
                  "both_arms_delta_vs_E": {s: arms[s]["paired"]["linear"][f"{s}+E_minus_E"]["fold"] for s in arms}}
            m = arms[best]["paired"].get("mlp", {}).get(f"{best}+E_minus_V+E", {}).get("fold")
            if m:
                vv["flag_F1_mlp_delta_vs_V+E"] = m
                vv["flag_F1_not_robust_to_nonlinear_ego"] = bool(verdict == "MOTION" and (m["delta"] < 0.01 or m["lo"] <= 0))
            P["V4"] = vv
        V["pops"][pop] = P
    json.dump(V, open(f"{C.RAW}/VERDICT.json", "w"), indent=1, default=float)
    f = lambda d: f"{d['delta']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]"
    print("controls pass:", V["control_pass"])
    for pop, P in V["pops"].items():
        print("==", pop)
        for k, v in P["controls"].items():
            print("  ctl", k, {kk: (round(vv["auc"], 4), vv["pass"]) for kk, vv in v.items()})
        print("  postfreeze C2 bands:", {k: (round(v["mean"], 4), round(v["sd"], 4), "seed2 min" if v["seed2_is_minimum"] else "", "true-nullmax %+.3f" % v["true_minus_null_max"], v["band_pass"]) for k, v in P["postfreeze_c2_band"].items()})
        if "Znn" in P:
            z = P["Znn"]
            print("  Znn:", z["verdict_committed"], "dAUC(Znn+E-E) fold", f(z["delta_Znn+E_minus_E_fold"]), "| Z+E-Znn+E", f(z["dAUC(Z+E - Znn+E)"]["fold"]), "| Znn+E-V+E", f(z["dAUC(Znn+E - V+E)"]["fold"]))
            print("        AUC E", round(z["AUC_E_fold"]["auc"], 4), "Znn+E", round(z["AUC_Znn+E_fold"]["auc"], 4), "Z+E", round(z["AUC_Z+E_fold"]["auc"], 4), "| flags", {k: v for k, v in z.items() if k.startswith("flag")})
            for k, v in z.items():
                if k.startswith("POSTFREEZE") and isinstance(v, dict) and "fold" in v:
                    print("   ", k, f(v["fold"]))
                elif k.startswith("POSTFREEZE") and isinstance(v, dict):
                    print("   ", k, f(v))
                elif k.startswith("POSTFREEZE"):
                    print("   ", k, v)
        if "V4" in P:
            v = P["V4"]
            print("  V4:", v["verdict_committed"], "best", v["best_arm_by_linear_fold_mean_AUC"], f(v["delta_best+E_minus_V+E_fold"]), {s: f(d) for s, d in v["both_arms_delta_vs_V+E"].items()}, "flags", {k: val for k, val in v.items() if k.startswith("flag")})


if __name__ == "__main__":
    main()
