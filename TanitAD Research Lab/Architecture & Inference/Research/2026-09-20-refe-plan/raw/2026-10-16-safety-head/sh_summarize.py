#!/usr/bin/env python3
"""SAFEHEAD summary: applies PREREG_SAFEHEAD.md s.8 AS WRITTEN.  Inputs: out/sh_*.pkl, out/banked_P.pkl, phead full_ref.pkl (FULL at P), the MM's phead k250T pickles (FULL at T).
Writes summary.json and prints the tables.  Safe to re-run whenever a new arm lands (missing inputs -> PENDING / PROVISIONAL, never a silent pass).
    python sh_summarize.py"""
import glob, json, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sh_common as SH
PC = SH.PC

POOLS = ("train", "val")
SAFETY = {"TTC": 3, "NC": 0, "DAC": 1}
REPORT = {**SAFETY, "EP": 2}
BAR = -0.5
MM_T = os.environ.get("SH_MM_T", "C:/Users/Admin/AppData/Local/Temp/phead/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-16-phead-epo-feasibility/closed_out")
VIEW = {"SH_F": "SH_F", "SH_A": "SH_A", "SH_SG": "own", "FULL": "own"}
FR = pickle.load(open(os.path.join(SH.PHD, "full_ref.pkl"), "rb"))
LOGS = FR["logs"]
LOGS_ALL = np.concatenate([LOGS["train"], LOGS["val"]])


def load(p):
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None


def prox(e1, e0):
    out = {}
    for nm, col in REPORT.items():
        for pn_, key in (("P1", "pick_comp"), ("P2", "pi_comp")):
            d = np.concatenate([(e1[p][key][:, col] - e0[p][key][:, col]) * 100 for p in POOLS])
            b = PC.boot_ratio(LOGS_ALL, d)
            out[f"{pn_}_{nm}"] = {**{p: float(((e1[p][key][:, col] - e0[p][key][:, col]) * 100).mean()) for p in POOLS}, "pooled": b["mean_num"], "pooled_ci": b["ci_num"]}
    d = np.concatenate([(e1[p]["pick_pdms"] - e0[p]["pick_pdms"]) * 100 for p in POOLS])
    out["P1_PDMS"] = {"pooled": float(d.mean()), "pooled_ci": PC.boot_ratio(LOGS_ALL, d)["ci_num"]}
    return out


def safety_ok(px):
    return {nm: bool(px[f"P1_{nm}"]["pooled"] >= BAR and px[f"P2_{nm}"]["pooled"] >= BAR) for nm in SAFETY}


def read_arm(e1, e0, F1, F0):
    dG = {p: e1[p]["J"] - e0[p]["J"] for p in POOLS}
    dF = {p: F1[p]["J"] - F0[p]["J"] for p in POOLS}
    fb = {p: PC.boot_ratio(LOGS[p], dG[p], dF[p]) for p in POOLS}
    px = prox(e1, e0)
    return {"dJ": {p: {"mean": fb[p]["mean_num"], "ci": fb[p]["ci_num"]} for p in POOLS}, "dJ_FULL": {p: fb[p]["mean_den"] for p in POOLS},
            "f": {p: {"f": fb[p]["f"], "ci": fb[p]["ci_f"], "den_straddles_zero": fb[p]["den_straddles_zero"]} for p in POOLS},
            "f_min": min(fb[p]["f"] for p in POOLS), "proxies_vs_matched_ctrl": px, "safety_ok": safety_ok(px),
            "kl_to_base": {p: float(e1[p]["kl"].mean()) for p in POOLS}, "J_lam1": {p: float(e1[p]["J"].mean()) for p in POOLS}, "J_ctrl": {p: float(e0[p]["J"].mean()) for p in POOLS}}


def sh_eval(regime, d, K, k):
    r = load(os.path.join(HERE, "out", f"sh_s0_{regime}_{d}_sepon_lr1_lam1_K{K}.pkl"))
    if r is None or k not in r["snaps"]:
        return None, r
    return {p: r["snaps"][k]["eval"][p][VIEW[d]] for p in POOLS}, r


def horizon(regime, K, k):
    """-> dict for one horizon: FULL read (its own safety) + each design read vs its matched control, f against FULL."""
    res = {"regime": regime, "K": k}
    shA_e, shA_r = sh_eval(regime, "SH_A", K, k)
    coA = {p: shA_r["snaps"][k]["eval"][p]["A_own"] for p in POOLS} if shA_e is not None else None
    coF = {p: shA_r["snaps"][k]["eval"][p]["SH_F_ctrl"] for p in POOLS} if shA_e is not None else None
    if regime == "P":
        F1, F0 = FR[(1.0, k)], FR[(0.0, k)]
        bk = load(os.path.join(HERE, "out", "banked_P.pkl"))
        ctrlF = {p: bk[(0.0, k)][p]["SH_F"] for p in POOLS} if bk else None
        res["control_source"] = "banked closed_P_0.0.pt D[k] (= full_ref.pkl)"
        if coA is not None:
            res["xcheck_coA_vs_banked_lam0_J_maxabs"] = {p: float(np.abs(coA[p]["J"] - F0[p]["J"]).max()) for p in POOLS}
    else:
        m1 = load(os.path.join(MM_T, "closed_s0_T_FULL_R_lr1_lam1k250T.pkl"))
        m0 = load(os.path.join(MM_T, "closed_s0_T_FULL_R_lr1_lam0k250T.pkl"))
        F1 = m1["snaps"][k]["eval"] if m1 and k in m1["snaps"] else None
        F0 = m0["snaps"][k]["eval"] if m0 and k in m0["snaps"] else coA
        res["control_source"] = "MM FULL lam0 k250T" if (m0 and k in m0["snaps"]) else ("co-trained arm A of the SH_A T run (MM lam0 not landed)" if coA is not None else "MISSING")
        ctrlF = coF
        if m0 and k in m0["snaps"] and coA is not None:
            res["xcheck_coA_vs_MM_lam0_J_maxabs"] = {p: float(np.abs(coA[p]["J"] - m0["snaps"][k]["eval"][p]["J"]).max()) for p in POOLS}
        if F1 is None or F0 is None:
            res["FULL"] = None
            res["validity"] = "V-PENDING"
            return res
    res["FULL"] = {"proxies_vs_lam0": prox(F1, F0), "dJ": {p: float((F1[p]["J"] - F0[p]["J"]).mean()) for p in POOLS},
                   "dJ_ci": {p: PC.boot_ratio(LOGS[p], F1[p]["J"] - F0[p]["J"])["ci_num"] for p in POOLS}}
    ttc = res["FULL"]["proxies_vs_lam0"]["P2_TTC"]["pooled"]
    res["FULL"]["P2_TTC_pooled"] = ttc
    if regime == "T":
        res["validity"] = "V-OK" if ttc < BAR else "V-NO"
    for d in ("SH_F", "SH_A", "SH_SG"):
        e1, _ = sh_eval(regime, d, K, k)
        e0 = ctrlF if d == "SH_F" else F0
        if e1 is None or e0 is None:
            res[d] = None
            continue
        res[d] = read_arm(e1, e0, F1, F0)
        res[d]["proxies_vs_FULL_lam0_g5_as_registered"] = prox(e1, F0)
    return res


def row(d, hP, hT, V):
    rp, rt = hP.get(d), (hT.get(d) if hT else None)
    if rp is None:
        return "PENDING (no P reading)"
    sP = all(rp["safety_ok"].values())
    sT = all(rt["safety_ok"].values()) if rt else None
    fP = rp["f_min"]
    fT = rt["f_min"] if rt else None
    fw = min(fP, fT) if rt else fP
    tag = ""
    if V == "V-NO":
        return f"SH-UNTESTABLE-TTC (V-NO): f_min P {fP:.3f} / T {fT if fT is None else round(fT, 3)}; NC/DAC P {rp['safety_ok']} T {rt['safety_ok'] if rt else None}"
    if rt is None:
        tag = "PROVISIONAL (T missing) "
        if not sP or fP < 0.2:
            return tag + "SH-NO"
        return tag + ("GO-on-P-only (never GO without T)" if fP >= 0.5 else "PARTIAL-on-P-only")
    if V != "V-OK":
        tag = f"PROVISIONAL ({V}) "
    if not sT or fw < 0.2:
        return tag + "SH-NO"
    if sP and sT and fP >= 0.5 and fT >= 0.5:
        return tag + "SH-GO"
    if sP and sT and 0.2 <= fw < 0.5:
        return tag + "SH-PARTIAL"
    if fP >= 0.5 and fT >= 0.5 and (sP != sT):
        return tag + "SH-PARTIAL"
    return tag + "UNASSIGNED (prereg gap; reported, no row)"


def main():
    hP = {k: horizon("P", 50, k) for k in (25, 50)}
    hT = {k: horizon("T", 250, k) for k in (50, 100, 250)}
    V = hT[250].get("validity", "V-PENDING")
    rows = {d: row(d, hP[50], hT[250] if hT[250].get("FULL") else None, V) for d in ("SH_F", "SH_A", "SH_SG")}
    res = {"prereg_sha256": open(os.path.join(HERE, "PREREG_SAFEHEAD.sha256")).read().split()[0], "validity": V, "rows": rows, "P": {str(k): v for k, v in hP.items()}, "T": {str(k): v for k, v in hT.items()}}
    json.dump(res, open(os.path.join(HERE, "summary.json"), "w"), indent=1, default=float)
    print("VALIDITY", V)
    for hz, hh in (("P", hP), ("T", hT)):
        for k, h in hh.items():
            if not h.get("FULL"):
                print(f"--- {hz} K={k}: FULL reading missing ({h.get('control_source')})")
                continue
            fl = h["FULL"]
            print(f"--- {hz} K={k} ctrl: {h['control_source']} | FULL dJ train {fl['dJ']['train']:+.4f} val {fl['dJ']['val']:+.4f} | FULL vs lam0 P2 TTC {fl['P2_TTC_pooled']:+.2f} "
                  f"P1 TTC {fl['proxies_vs_lam0']['P1_TTC']['pooled']:+.2f} P2 NC {fl['proxies_vs_lam0']['P2_NC']['pooled']:+.2f} P2 DAC {fl['proxies_vs_lam0']['P2_DAC']['pooled']:+.2f}")
            for d in ("SH_F", "SH_A", "SH_SG"):
                v = h.get(d)
                if not v:
                    print(f"   {d}: pending")
                    continue
                px = v["proxies_vs_matched_ctrl"]
                print(f"   {d}: f train {v['f']['train']['f']:+.3f} val {v['f']['val']['f']:+.3f} fmin {v['f_min']:+.3f} | TTC P1 {px['P1_TTC']['pooled']:+.2f} P2 {px['P2_TTC']['pooled']:+.2f} | "
                      f"NC P1 {px['P1_NC']['pooled']:+.2f} P2 {px['P2_NC']['pooled']:+.2f} | DAC P1 {px['P1_DAC']['pooled']:+.2f} P2 {px['P2_DAC']['pooled']:+.2f} | EP P1 {px['P1_EP']['pooled']:+.2f} | PDMS P1 {px['P1_PDMS']['pooled']:+.2f} | ok {v['safety_ok']}")
    print("ROWS", json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
