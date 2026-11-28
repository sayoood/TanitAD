"""D6 part 1 -- controls, per-scene failure table, cross-arm patterns, official-EPDMS counterfactuals.

BANKED FILES ONLY (no metric cache): the PRE-CSV score frames + hooks of the refcv7 NavSim milestone runs.
    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 C:/Users/Admin/venvs/tanitad/Scripts/python.exe d6_part1_table.py

Writes  raw/d6_scene_table_step30000.csv   (5,912 rows: one per scorer token, both stages)
        raw/d6_part1.json                  (controls + aggregates; no tokens)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d6_common as C

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "raw")
TERMS = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]


def band(v):
    return "0-2" if v < 2 else "2-5" if v < 5 else "5-8" if v < 8 else "8-12" if v < 12 else ">=12"


def main():
    ci = C.load_ci()
    inp, mapping = C.load_inputs()
    out = {"controls": {}, "n_tokens": len(inp)}
    arms = {
        "A1": C.load_arm(30000, "R7_A1"),
        "A1s1": C.load_arm(30000, "R7_A1_s1", with_hooks=False),
        "A1k5": C.load_arm(5000, "R7_A1", with_hooks=False),
        "PRIOR": C.load_arm(30000, "PRIOR_ha0p", with_hooks=False),
        "STOP": C.load_floor("STOP_zero"),
        "CV": C.load_floor("CV_official"),
    }
    # ------------------------------------------------------------------ controls (known values)
    summ30 = json.load(open(f"{C.milestone_dir(30000)}/summary_navhard.json", encoding="utf-8"))
    summ5 = json.load(open(f"{C.milestone_dir(5000)}/summary_navhard.json", encoding="utf-8"))
    ref = {"A1": summ30["arms"]["R7_A1"], "A1s1": summ30["arms"]["R7_A1_s1"], "PRIOR": summ30["arms"]["PRIOR_ha0p"],
           "STOP": summ30["arms"]["STOP_zero"], "CV": summ30["arms"]["CV_official"], "A1k5": summ5["arms"]["R7_A1"]}
    ctl = out["controls"]
    for k, df in arms.items():
        off = C.official_epdms(df, mapping, ci)
        want = ref[k]["official_two_stage_EPDMS"]
        s2 = df[df.frame_type == 1]
        z = {t: float((s2[t] == 0).mean()) for t in C.MULT}
        zr = ref[k]["S2_multiplier_zero_rates"]
        ctl[f"official_epdms_{k}"] = {"recomputed": off, "summary_json": want, "abs_diff": abs(off - want),
                                      "PASS": abs(off - want) < 1e-9}
        ctl[f"zero_rates_{k}"] = {"recomputed": z, "summary_json": zr,
                                  "max_abs_diff": max(abs(z[t] - zr[t]) for t in C.MULT),
                                  "PASS": max(abs(z[t] - zr[t]) for t in C.MULT) < 1e-12}
        assert len(s2) == 5462 and (df.frame_type == 0).sum() == 450
    # formula control: score == prod(NC,DAC,DDC,TLC) * (5EP+5TTC+2LK+2HC+2EC)/16 ?
    a = arms["A1"]
    f = a[list(C.MULT)].prod(axis=1) * sum(a[k] * v for k, v in C.WEIGHTED.items()) / sum(C.WEIGHTED.values())
    ctl["score_formula_A1"] = {"max_abs_diff": float((f - a.score).abs().max()),
                               "n_diff_gt_1e-9": int(((f - a.score).abs() > 1e-9).sum())}
    # STOP analytic control: a zero plan; stage-2 count; STOP never makes progress beyond its own EP floor
    st = arms["STOP"]
    ctl["STOP_ep_range"] = [float(st.EP.min()), float(st.EP.max())]
    # token-set agreement across all arms and with the inputs export
    ctl["token_sets_equal"] = all(set(df.index) == set(inp) for df in arms.values())
    # command-order control (analytic): human 4 s lateral displacement by NavSim one-hot index, stage 1
    hum = arms["A1"]["human"]
    rows = []
    for t, r in inp.items():
        if r["stage"] == 1:
            h = hum[t]
            rows.append((r["cmd"], float(h[-1, 1]), float(h[-1, 2])))
    cdf = pd.DataFrame(rows, columns=["cmd", "y4", "yaw4"])
    ctl["command_order_human_stage1"] = {C.CMD_NAME[k]: {"n": int(len(g)), "median_y4_m": float(g.y4.median()),
                                                         "median_yaw4_rad": float(g.yaw4.median())}
                                         for k, g in cdf.groupby("cmd")}
    m = ctl["command_order_human_stage1"]
    ctl["command_order_human_stage1"]["PASS_left_gt_straight_gt_right"] = bool(
        m["LEFT"]["median_yaw4_rad"] > m["STRAIGHT"]["median_yaw4_rad"] > m["RIGHT"]["median_yaw4_rad"])

    # ------------------------------------------------------------------ the per-scene table
    tab = pd.DataFrame(index=arms["A1"].index)
    tab["stage"] = [inp[t]["stage"] for t in tab.index]
    tab["log_name"] = [inp[t]["log_name"] for t in tab.index]
    tab["map"] = [inp[t]["map"] for t in tab.index]
    tab["cmd"] = [C.CMD_NAME[inp[t]["cmd"]] for t in tab.index]
    tab["v0"] = arms["A1"]["v0"]
    tab["v0_band"] = [band(v) for v in tab["v0"]]
    tab["orig_init"] = [inp[t]["orig_init"] for t in tab.index]
    for k, df in arms.items():
        for t in TERMS:
            tab[f"{k}_{t}"] = df[t]
        tab[f"{k}_score"] = df["score"]
        tab[f"{k}_weight"] = df["weight"]
    # zero flags and patterns
    for k in arms:
        tab[f"{k}_zeros"] = ["+".join(t for t in C.MULT if tab.loc[i, f"{k}_{t}"] == 0) or "-" for i in tab.index]
    # plan geometry summary (raw plan, ego frame): endpoint distance + heading at 4 s
    P = arms["A1"]["plan"]
    tab["plan_end_x"] = [P[t][-1, 0] for t in tab.index]
    tab["plan_end_y"] = [P[t][-1, 1] for t in tab.index]
    tab["plan_end_yaw"] = [P[t][-1, 2] for t in tab.index]
    tab["plan_end_dist"] = np.hypot(tab.plan_end_x, tab.plan_end_y)
    tab.index.name = "token"
    tab.reset_index().to_csv(os.path.join(RAW, "d6_scene_table_step30000.csv"), index=False, float_format="%.6g")

    # ------------------------------------------------------------------ cross-arm patterns (per multiplier)
    pat = {}
    for stg in (1, 2):
        d = tab[tab.stage == stg]
        pat[f"stage{stg}"] = {"n": int(len(d))}
        for term in ("DAC", "NC", "DDC", "TLC"):
            z = d[d[f"A1_{term}"] == 0]
            n = len(z)
            rec = {"n_A1_zero": int(n), "rate_A1": n / len(d),
                   "rate_STOP": float((d[f"STOP_{term}"] == 0).mean()),
                   "rate_PRIOR": float((d[f"PRIOR_{term}"] == 0).mean()),
                   "rate_CV": float((d[f"CV_{term}"] == 0).mean()),
                   "rate_A1s1": float((d[f"A1s1_{term}"] == 0).mean()),
                   "rate_A1_5k": float((d[f"A1k5_{term}"] == 0).mean())}
            if n:
                rec["of_A1_zero__STOP_also_zero"] = int((z[f"STOP_{term}"] == 0).sum())
                rec["of_A1_zero__STOP_passes"] = int((z[f"STOP_{term}"] != 0).sum())
                rec["of_A1_zero__PRIOR_also_zero"] = int((z[f"PRIOR_{term}"] == 0).sum())
                rec["of_A1_zero__CV_also_zero"] = int((z[f"CV_{term}"] == 0).sum())
                rec["of_A1_zero__all_of_STOP_PRIOR_CV_zero"] = int(((z[f"STOP_{term}"] == 0) & (z[f"PRIOR_{term}"] == 0) & (z[f"CV_{term}"] == 0)).sum())
                rec["of_A1_zero__A1s1_also_zero"] = int((z[f"A1s1_{term}"] == 0).sum())
                rec["of_A1_zero__A1s1_passes"] = int((z[f"A1s1_{term}"] != 0).sum())
                rec["of_A1_zero__5k_also_zero"] = int((z[f"A1k5_{term}"] == 0).sum())
                rec["of_A1_zero__5k_passes"] = int((z[f"A1k5_{term}"] != 0).sum())
                rec["of_A1_zero__STOP_passes_and_PRIOR_passes"] = int(((z[f"STOP_{term}"] != 0) & (z[f"PRIOR_{term}"] != 0)).sum())
                rec["of_A1_zero__STOP_passes_PRIOR_passes_s1_passes"] = int(((z[f"STOP_{term}"] != 0) & (z[f"PRIOR_{term}"] != 0) & (z[f"A1s1_{term}"] != 0)).sum())
                rec["of_A1_zero__only_A1_zero_among_A1_s1_PRIOR_STOP_CV"] = int(((z[f"STOP_{term}"] != 0) & (z[f"PRIOR_{term}"] != 0) & (z[f"CV_{term}"] != 0) & (z[f"A1s1_{term}"] != 0)).sum())
            # reverse: scenes where STOP zero and A1 passes (A1 rescues)
            rec["STOP_zero_A1_passes"] = int(((d[f"STOP_{term}"] == 0) & (d[f"A1_{term}"] != 0)).sum())
            # two-seed overlap (training-fixed, sampler-varying)
            both = ((d[f"A1_{term}"] == 0) & (d[f"A1s1_{term}"] == 0)).sum()
            either = ((d[f"A1_{term}"] == 0) | (d[f"A1s1_{term}"] == 0)).sum()
            rec["seed_jaccard"] = float(both / either) if either else None
            # by command and by speed band
            rec["by_command"] = {c: {"n": int(len(g)), "A1_zero_rate": float((g[f"A1_{term}"] == 0).mean()),
                                     "STOP_zero_rate": float((g[f"STOP_{term}"] == 0).mean()),
                                     "PRIOR_zero_rate": float((g[f"PRIOR_{term}"] == 0).mean())}
                                 for c, g in d.groupby("cmd")}
            rec["by_v0_band"] = {b: {"n": int(len(g)), "A1_zero_rate": float((g[f"A1_{term}"] == 0).mean()),
                                     "STOP_zero_rate": float((g[f"STOP_{term}"] == 0).mean()),
                                     "PRIOR_zero_rate": float((g[f"PRIOR_{term}"] == 0).mean())}
                                 for b, g in d.groupby("v0_band")}
            pat[f"stage{stg}"][term] = rec
    out["cross_arm_patterns"] = pat
    # co-occurrence of zero terms among A1 stage-2
    d2 = tab[tab.stage == 2]
    out["A1_stage2_zero_combos"] = d2["A1_zeros"].value_counts().to_dict()
    out["A1_stage1_zero_combos"] = tab[tab.stage == 1]["A1_zeros"].value_counts().to_dict()

    # ------------------------------------------------------------------ official-EPDMS counterfactuals
    def cf(df, fix_terms=(), replace_from=None, mask=None):
        """official EPDMS after setting `fix_terms` to 1 on rows `mask` (or taking them from `replace_from`)."""
        d = df.copy()
        sel = np.ones(len(d), bool) if mask is None else mask
        for t in fix_terms:
            if replace_from is None:
                d.loc[sel, t] = 1.0
            else:
                d.loc[sel, t] = replace_from.loc[d.index[sel], t].values
        sc = d[list(C.MULT)].prod(axis=1) * sum(d[k] * v for k, v in C.WEIGHTED.items()) / sum(C.WEIGHTED.values())
        d["score"] = sc
        return C.official_epdms(d, mapping, ci)

    base = C.official_epdms(arms["A1"], mapping, ci)
    stop_off = C.official_epdms(arms["STOP"], mapping, ci)
    cfd = {"A1_official": base, "STOP_official": stop_off, "gap_A1_minus_STOP": base - stop_off}
    A = arms["A1"]
    for t in ("DAC", "NC", "DDC", "TLC"):
        cfd[f"fix_{t}_to_1_all_scenes"] = cf(A, (t,)) - base
    cfd["fix_DAC_NC_DDC_TLC_to_1"] = cf(A, ("DAC", "NC", "DDC", "TLC")) - base
    # "match STOP's multiplier wherever A1 is zero and STOP is not" (A1 only ever ABOVE STOP's multiplier)
    for t in ("DAC", "NC", "DDC", "TLC"):
        m = ((A[t] == 0) & (arms["STOP"].loc[A.index, t] != 0)).values
        cfd[f"match_STOP_on_{t}_where_A1_zero_STOP_passes"] = cf(A, (t,), replace_from=arms["STOP"], mask=m) - base
    # the same for stage-1 only vs stage-2 only (stage 1 multiplies stage 2 inside every key)
    for stg_name, stg_frame_type in (("stage1_rows_only", 0), ("stage2_rows_only", 1)):
        for t in ("DAC", "NC"):
            m = (A.frame_type == stg_frame_type).values
            cfd[f"fix_{t}_to_1_{stg_name}"] = cf(A, (t,), mask=m) - base
    out["official_epdms_counterfactuals_A1_30k"] = cfd

    # stage-1 -> official key structure: how many keys have a zero stage-1 factor
    rows = {t: {"score": float(r.score), "weight": float(r.weight)} for t, r in A[["score", "weight"]].iterrows()}
    n_keys = len(mapping)
    s1z = 0
    for e in mapping:
        if rows[str(e[0])]["score"] == 0:
            s1z += 1
    out["keys"] = {"n_keys": n_keys, "keys_whose_orig_stage1_score_is_zero_A1": s1z}

    json.dump(out, open(os.path.join(RAW, "d6_part1.json"), "w", encoding="utf-8"), indent=1, default=float)
    print("controls:", json.dumps({k: (v.get("PASS") if isinstance(v, dict) and "PASS" in v else v) for k, v in ctl.items() if k.startswith(("official", "zero_rates"))}, indent=0))
    print("formula", ctl["score_formula_A1"], "STOP EP range", ctl["STOP_ep_range"], "token sets equal", ctl["token_sets_equal"])
    print("cmd-order", json.dumps(ctl["command_order_human_stage1"]))
    print("cf:", json.dumps({k: round(v, 4) for k, v in cfd.items()}, indent=0))
    for stg in (1, 2):
        print(f"--- stage {stg}")
        for term in ("DAC", "NC", "DDC", "TLC"):
            r = pat[f"stage{stg}"][term]
            print(term, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, dict)})


if __name__ == "__main__":
    main()
