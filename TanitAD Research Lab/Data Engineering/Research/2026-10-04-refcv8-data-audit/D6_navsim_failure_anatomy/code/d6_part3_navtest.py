"""D6 part 3 -- the same compact anatomy on NAVTEST (single-stage PDMS v1, 12,146 tokens, ORIGINAL frames with a banked
human future), step 30,000 and step 5,000.  BANKED FILES ONLY: scorer CSVs, hooks (emitted plans), the navtest inputs
export (human future + driving command), the bridge rows (max-speed input).

Classes use the HUMAN future as the reference (navtest has no cached route here; the same T_* literals as part 2):
    STOP-LIKE | WRONG-SIDE | ROUTE-FOLLOWING (under-turn vs the human) | OVER-STEER | LATERAL-DRIFT | SPEED | ON-HUMAN
Not available without re-scoring on the navsim-1.1 devkit: time of first non-drivable instant, collision type (stated in RESULT).
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d6_common as C
from d6_part2_anatomy import T_TURN, T_HEAD, T_LAT, T_STOP, SPEED_RATIO, SPEED_ABS, wrap

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "raw")
NT = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
SUBN = {"no_at_fault_collisions": "NC", "drivable_area_compliance": "DAC", "ego_progress": "EP", "time_to_collision_within_bound": "TTC",
        "comfort": "CF", "driving_direction_compliance": "DDC"}


def load_scores(path):
    df = pd.read_csv(path)
    df = df[df.token != "average"].copy()
    assert len(df) == 12146 and df.token.is_unique, path
    return df.rename(columns=SUBN).set_index("token")


def classify_h(P, H):
    """plan P vs human H, both [8,3] ego frame."""
    dist_p = float(np.hypot(*P[-1, :2]))
    dist_h = float(np.hypot(*H[-1, :2]))
    if dist_p < T_STOP:
        return "STOP-LIKE"
    gt = float(H[-1, 2])
    yaw = float(P[-1, 2])
    if abs(gt) >= T_TURN:
        rs = 1.0 if gt > 0 else -1.0
        if np.sign(yaw) == -rs and abs(yaw) >= T_HEAD:
            return "WRONG-SIDE"
        err = rs * float(wrap(yaw - gt))
        lat_in = rs * (P[-1, 1] - H[-1, 1])
        if err <= -T_HEAD or lat_in <= -T_LAT:
            return "ROUTE-FOLLOWING"
        if err >= T_HEAD or lat_in >= T_LAT:
            return "OVER-STEER"
    else:
        if abs(P[-1, 1] - H[-1, 1]) >= T_LAT or abs(float(wrap(yaw - gt))) >= T_HEAD:
            return "LATERAL-DRIFT"
    if dist_p >= SPEED_RATIO * dist_h and dist_p - dist_h >= SPEED_ABS:
        return "SPEED"
    return "ON-HUMAN"


def main():
    base30 = f"{C.milestone_dir(30000)}/scores_navtest"
    base5 = f"{C.milestone_dir(5000)}/scores_navtest"
    sc = {"A1": load_scores(f"{base30}/r7s30000_R7_A1/r7s30000_R7_A1.csv"),
          "A1s1": load_scores(f"{base30}/r7s30000_R7_A1_s1/r7s30000_R7_A1_s1.csv"),
          "PRIOR": load_scores(f"{base30}/r7s30000_PRIOR_ha0p/r7s30000_PRIOR_ha0p.csv"),
          "STOP": load_scores(f"{C.PKG}/raw/harness_navtest/r7kh_STOP/r7kh_STOP.csv"),
          "A1k5": load_scores(f"{base5}/r7s5000_R7_A1/r7s5000_R7_A1.csv"),
          "PRIORk5": load_scores(f"{base5}/r7s5000_PRIOR_ha0p/r7s5000_PRIOR_ha0p.csv")}
    out = {"controls": {}}
    # controls: PDMS values (known), STOP analytic, formula
    for k, df in sc.items():
        out["controls"][f"PDMS_x100_{k}"] = float(df.score.mean() * 100)
    a = sc["A1"]
    f = a.NC * a.DAC * (5 * a.EP + 5 * a.TTC + 2 * a.CF) / 12
    out["controls"]["pdms_formula_max_abs_diff_A1"] = float((f - a.score).abs().max())
    out["controls"]["expected"] = {"A1_30k_PDMS_x100": 71.8849, "STOP_PDMS_x100": 61.82023, "A1_5k_PDMS_x100": 65.5976}
    out["controls"]["PASS_A1_30k"] = abs(out["controls"]["PDMS_x100_A1"] - 71.8849) < 5e-3
    out["controls"]["PASS_STOP"] = abs(out["controls"]["PDMS_x100_STOP"] - 61.82023) < 5e-4
    out["controls"]["PASS_A1_5k"] = abs(out["controls"]["PDMS_x100_A1k5"] - 65.5976) < 5e-3
    # inputs: human future, command, log
    d = json.load(gzip.open(NT, "rt", encoding="utf-8"))["tokens"]
    assert len(d) == 12146
    H, cmd, logn = {}, {}, {}
    for t, r in d.items():
        H[t] = np.asarray(r["human_future_poses"], dtype=np.float64)
        cmd[t] = int(np.argmax(r["ego_statuses"][-1]["driving_command"]))
        logn[t] = r["log_name"]
    del d
    # hooks: emitted plans + v0
    plans = {}
    for key, p in (("A1", f"{base30}/r7s30000_R7_A1/r7s30000_R7_A1_hooks.json"), ("A1k5", f"{base5}/r7s5000_R7_A1/r7s5000_R7_A1_hooks.json"),
                   ("PRIOR", f"{base30}/r7s30000_PRIOR_ha0p/r7s30000_PRIOR_ha0p_hooks.json")):
        hk = json.load(open(p, encoding="utf-8"))["pdm_score_calls"]
        assert len(hk) == 12146
        plans[key] = {c["token"]: np.asarray(c["agent_poses"], dtype=np.float64) for c in hk}
        if key == "A1":
            v0 = {c["token"]: float(c["v0_mps"]) for c in hk}
        del hk
    # max-speed input
    vk = {}
    for line in open(f"{C.milestone_dir(30000)}/bridge_navtest/rows_R7_A1.jsonl", encoding="utf-8"):
        r = json.loads(line)
        vk[r["token"]] = float(r["vmax"].get("v_max_valid", 0.0)) == 1.0
    toks = list(sc["A1"].index)
    assert set(toks) == set(H) == set(plans["A1"]) == set(vk)
    # analytic control of the classifier: the human against itself is ON-HUMAN (or STOP-LIKE if it barely moves)
    self_cls = Counter(classify_h(H[t], H[t]) for t in toks)
    out["controls"]["human_vs_itself_classes"] = dict(self_cls)
    out["controls"]["human_vs_itself_only_ON_HUMAN_or_STOP"] = bool(set(self_cls) <= {"ON-HUMAN", "STOP-LIKE"})
    # command order control (human 4-s yaw by command)
    cdf = pd.DataFrame({"cmd": [cmd[t] for t in toks], "yaw4": [H[t][-1, 2] for t in toks]})
    out["controls"]["command_order_human_median_yaw4"] = {C.CMD_NAME[k]: float(g.yaw4.median()) for k, g in cdf.groupby("cmd")}
    m = out["controls"]["command_order_human_median_yaw4"]
    out["controls"]["PASS_left_gt_straight_gt_right"] = bool(m["LEFT"] > m["STRAIGHT"] > m["RIGHT"])

    D = pd.DataFrame(index=toks)
    D["cmd"] = [C.CMD_NAME[cmd[t]] for t in toks]
    D["v0"] = [v0[t] for t in toks]
    D["v0_band"] = ["0-2" if v < 2 else "2-5" if v < 5 else "5-8" if v < 8 else "8-12" if v < 12 else ">=12" for v in D.v0]
    D["vmax_known"] = [vk[t] for t in toks]
    D["gt_turn"] = [abs(H[t][-1, 2]) >= T_TURN for t in toks]
    D["gt_dir"] = ["L" if H[t][-1, 2] >= T_TURN else "R" if H[t][-1, 2] <= -T_TURN else "S" for t in toks]
    for key in ("A1", "A1k5", "PRIOR"):
        D[f"cls_{key}"] = [classify_h(plans[key][t], H[t]) for t in toks]
    D["dist_plan"] = [float(np.hypot(*plans["A1"][t][-1, :2])) for t in toks]
    D["dist_h"] = [float(np.hypot(*H[t][-1, :2])) for t in toks]
    D["yaw_plan"] = [plans["A1"][t][-1, 2] for t in toks]
    D["yaw_h"] = [H[t][-1, 2] for t in toks]
    for k in sc:
        for term in ("NC", "DAC", "EP", "TTC", "CF", "DDC", "score"):
            D[f"{k}_{term}"] = sc[k].loc[toks, term].values

    def share(series):
        c = series.value_counts()
        return {k: {"n": int(v), "share": float(v / len(series))} for k, v in c.items()}

    res = {}
    for key, flagcol, clscol in (("A1_30k", "A1_DAC", "cls_A1"), ("A1_5k", "A1k5_DAC", "cls_A1k5"), ("PRIOR_30k", "PRIOR_DAC", "cls_PRIOR")):
        z = D[D[flagcol] == 0]
        res[key] = {"n_DACzero": int(len(z)), "rate": float(len(z) / len(D)), "class_mix": share(z[clscol]),
                    "rate_by_class": {c: {"n": int(len(g)), "DAC0_rate": float((g[flagcol] == 0).mean())} for c, g in D.groupby(clscol)}}
    out["DAC_anatomy"] = res
    # STOP / PRIOR cross patterns on A1's DAC-zero
    z = D[D.A1_DAC == 0]
    out["DAC_cross_arm"] = {"n": int(len(z)), "STOP_also_zero": int((z.STOP_DAC == 0).sum()), "STOP_passes": int((z.STOP_DAC != 0).sum()),
                            "PRIOR_also_zero": int((z.PRIOR_DAC == 0).sum()), "A1s1_also_zero": int((z.A1s1_DAC == 0).sum()),
                            "A1_5k_also_zero": int((z.A1k5_DAC == 0).sum()),
                            "STOP_and_PRIOR_pass": int(((z.STOP_DAC != 0) & (z.PRIOR_DAC != 0)).sum()),
                            "seed_jaccard": float(((D.A1_DAC == 0) & (D.A1s1_DAC == 0)).sum() / ((D.A1_DAC == 0) | (D.A1s1_DAC == 0)).sum())}
    zn = D[D.A1_NC == 0]
    out["NC_cross_arm"] = {"n": int(len(zn)), "rate": float(len(zn) / len(D)), "STOP_also_zero": int((zn.STOP_NC == 0).sum()), "PRIOR_also_zero": int((zn.PRIOR_NC == 0).sum()),
                           "A1s1_also_zero": int((zn.A1s1_NC == 0).sum()), "A1_5k_also_zero": int((zn.A1k5_NC == 0).sum()),
                           "also_DAC_zero": int((zn.A1_DAC == 0).sum()),
                           "seed_jaccard": float(((D.A1_NC == 0) & (D.A1s1_NC == 0)).sum() / ((D.A1_NC == 0) | (D.A1s1_NC == 0)).sum())}
    out["NC_class_mix"] = {"A1_30k": share(zn.cls_A1), "rate_by_class": {c: {"n": int(len(g)), "NC0_rate": float((g.A1_NC == 0).mean()), "STOP_NC0_rate": float((g.STOP_NC == 0).mean())}
                                                                        for c, g in D.groupby("cls_A1")}}
    out["NC_by_v0_band"] = {b: {"n": int(len(g)), "A1_NC0": float((g.A1_NC == 0).mean()), "STOP_NC0": float((g.STOP_NC == 0).mean()), "PRIOR_NC0": float((g.PRIOR_NC == 0).mean())}
                            for b, g in D.groupby("v0_band")}
    out["NC_plan_vs_human_distance"] = {"NC_zero_median_ratio": float((zn.dist_plan / zn.dist_h.clip(lower=0.5)).median()),
                                        "all_median_ratio": float((D.dist_plan / D.dist_h.clip(lower=0.5)).median()),
                                        "NC_zero_plan_faster_than_human_by_4m_share": float(((zn.dist_plan - zn.dist_h) >= 4.0).mean()),
                                        "all_plan_faster_than_human_by_4m_share": float(((D.dist_plan - D.dist_h) >= 4.0).mean())}
    # command splits (D4 finding 2): command x GT turn; DAC-zero by command and compliance
    D["complied"] = [(yp >= T_HEAD) if c == "LEFT" else (yp <= -T_HEAD) if c == "RIGHT" else (abs(yp) < 2 * T_HEAD) for c, yp in zip(D.cmd, D.yaw_plan)]
    cs = {}
    for cmd_, g in D.groupby("cmd"):
        cs[cmd_] = {"n": int(len(g)), "gt_turn_in_4s": {k: int(v) for k, v in g.gt_dir.value_counts().items()},
                    "A1_DAC0": float((g.A1_DAC == 0).mean()), "STOP_DAC0": float((g.STOP_DAC == 0).mean()), "PRIOR_DAC0": float((g.PRIOR_DAC == 0).mean()),
                    "A1_NC0": float((g.A1_NC == 0).mean()), "STOP_NC0": float((g.STOP_NC == 0).mean()),
                    "plan_complied": float(g.complied.mean())}
        for gd, h in g.groupby("gt_dir"):
            cs[cmd_][f"gt_{gd}"] = {"n": int(len(h)), "A1_DAC0": float((h.A1_DAC == 0).mean()), "STOP_DAC0": float((h.STOP_DAC == 0).mean()),
                                    "plan_complied": float(h.complied.mean()), "A1_DAC0_if_complied": float((h[h.complied].A1_DAC == 0).mean()) if h.complied.any() else None,
                                    "n_complied": int(h.complied.sum()),
                                    "A1_DAC0_if_not": float((h[~h.complied].A1_DAC == 0).mean()) if (~h.complied).any() else None, "n_not": int((~h.complied).sum())}
    out["command_split"] = cs
    xc = {}
    for cl in ("ROUTE-FOLLOWING", "WRONG-SIDE", "OVER-STEER", "LATERAL-DRIFT", "SPEED", "ON-HUMAN", "STOP-LIKE"):
        sub = z[z.cls_A1 == cl]
        allc = D[D.cls_A1 == cl]
        xc[cl] = {"n_DACzero": int(len(sub)), "n_all": int(len(allc)), "DACzero_by_command": {k: int(v) for k, v in sub.cmd.value_counts().items()},
                  "all_by_command": {k: int(v) for k, v in allc.cmd.value_counts().items()}}
    out["class_x_command"] = xc
    # max-speed known/unknown on navtest
    vmx = {}
    for known, g in D.groupby("vmax_known"):
        k = "known" if known else "unknown"
        vmx[k] = {"n": int(len(g)), "share_of_scenes": float(len(g) / len(D)),
                  **{f"{a}_{t}0": float((g[f"{a}_{t}"] == 0).mean()) for a in ("A1", "STOP", "PRIOR", "A1k5") for t in ("DAC", "NC")},
                  "share_of_A1_DAC0": float((g.A1_DAC == 0).sum() / (D.A1_DAC == 0).sum()), "share_of_A1_NC0": float((g.A1_NC == 0).sum() / (D.A1_NC == 0).sum()),
                  "PDMS_x100_A1": float(g.A1_score.mean() * 100), "PDMS_x100_STOP": float(g.STOP_score.mean() * 100), "PDMS_x100_PRIOR": float(g.PRIOR_score.mean() * 100)}
        vmx[k]["A1_minus_STOP_DAC0"] = vmx[k]["A1_DAC0"] - vmx[k]["STOP_DAC0"]
        vmx[k]["A1_minus_STOP_NC0"] = vmx[k]["A1_NC0"] - vmx[k]["STOP_NC0"]
    out["vmax_split_navtest"] = vmx
    out["vmax_strata_composition_navtest"] = {("known" if k else "unknown"): {"cmd": {c: float(v) for c, v in g.cmd.value_counts(normalize=True).items()},
                                                                               "v0_band": {c: float(v) for c, v in g.v0_band.value_counts(normalize=True).items()}, "median_v0": float(g.v0.median())}
                                              for k, g in D.groupby("vmax_known")}
    num = den = 0.0
    sd = {}
    for term in ("DAC", "NC"):
        for a_ in ("A1", "STOP", "PRIOR"):
            num = den = 0.0
            for _, g in D.groupby(["v0_band", "cmd"]):
                u, kn = g[~g.vmax_known], g[g.vmax_known]
                if len(u) >= 5 and len(kn) >= 5:
                    num += len(g) * ((u[f"{a_}_{term}"] == 0).mean() - (kn[f"{a_}_{term}"] == 0).mean())
                    den += len(g)
            sd[f"{a_}_{term}0"] = num / den if den else None
    out["vmax_standardised_diff_unknown_minus_known_navtest"] = sd
    # within-stratum class mix of the DAC zeros
    out["vmax_x_class_DACzero_navtest"] = {("known" if k else "unknown"): share(g[g.A1_DAC == 0].cls_A1) for k, g in D.groupby("vmax_known")}
    D.to_csv(os.path.join(RAW, "d6_navtest_geometry_step30000.csv"), float_format="%.5g", index_label="token")
    json.dump(out, open(os.path.join(RAW, "d6_part3_navtest.json"), "w", encoding="utf-8"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print({k: v for k, v in out["controls"].items() if k.startswith(("PASS", "PDMS", "pdms", "human_vs"))})
    print("DAC class mix 30k:", {k: v["n"] for k, v in res["A1_30k"]["class_mix"].items()})
    print("vmax", {k: (v["n"], round(v["A1_DAC0"], 3), round(v["STOP_DAC0"], 3), round(v["A1_NC0"], 3), round(v["STOP_NC0"], 3)) for k, v in vmx.items()})


if __name__ == "__main__":
    main()
