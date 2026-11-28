"""D6 part 6 -- merge the per-scene deliverable: one row per navhard scorer token (5,912), all arms' sub-scores (part 1) +
command, route/geometry class, plan-vs-command compliance, rescored first-violation times and collision class (part 2).
Writes raw/d6_scene_table_FULL_step30000.csv (the DELIVERABLE-1 table) and raw/d6_failure_patterns.json (counts).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "raw")


def main():
    T = pd.read_csv(os.path.join(RAW, "d6_scene_table_step30000.csv")).set_index("token")
    G = pd.read_csv(os.path.join(RAW, "d6_scene_geometry_step30000.csv")).set_index("token")
    keep = ["route_dir", "route_turn", "junction", "cls", "cls_5k", "cls_prior", "cls_s1", "lat0", "lat4", "yaw4", "head_err", "d_route",
            "dist4", "d_plan_route", "d_ref", "plan_speed4", "ref_speed4", "complied", "vmax_known", "resc_ok", "nd_first", "ref_dac", "raw_dac",
            "nc_first_t", "nc_ctype", "nc_obj", "nc_lat_dep", "nc_ego_speed", "nc_obj_speed", "nc_rel_lon", "nc_rel_lat", "nc_class", "human_dac"]
    keep = [k for k in keep if k in G.columns]
    F = T.join(G[keep], how="left")
    # scene-level flag for DAC-zero
    sc = np.where(F.A1_DAC != 0, "", np.where(F.resc_ok.fillna(False).astype(bool),
                                                np.where(F.nd_first == 0, "INITIAL-STATE", np.where(F.ref_dac == 0, "REF-ALSO-FAILS", "PLAN-INDUCED")), "NOT-RESCORED"))
    F["dac_scene_flag"] = sc
    # cross-arm pattern for each multiplier: who else fails
    def pattern(term):
        a = F[f"A1_{term}"] == 0
        s = F[f"STOP_{term}"] == 0
        p = F[f"PRIOR_{term}"] == 0
        c = F[f"CV_{term}"] == 0
        out = np.where(~a, "", np.where(s & p & c, "ALL-ARMS-FAIL", np.where(s, "STOP-ALSO-FAILS", np.where(p | c, "A1-AND-KINEMATIC-ARMS-FAIL", "A1-ONLY"))))
        return out
    for term in ("DAC", "NC", "DDC", "TLC"):
        F[f"pattern_{term}"] = pattern(term)
    F["refcv7_failed_where_STOP_passed_any_mult"] = [any(F.loc[i, f"A1_{t}"] == 0 and F.loc[i, f"STOP_{t}"] != 0 for t in ("NC", "DAC", "DDC", "TLC")) for i in F.index]
    F.reset_index().to_csv(os.path.join(RAW, "d6_scene_table_FULL_step30000.csv"), index=False, float_format="%.5g")
    cnt = {}
    for stg in (1, 2):
        d = F[F.stage == stg]
        cnt[f"stage{stg}"] = {"n": int(len(d))}
        for term in ("DAC", "NC", "DDC", "TLC"):
            cnt[f"stage{stg}"][term] = {k: int(v) for k, v in d[f"pattern_{term}"].value_counts().items() if k}
        cnt[f"stage{stg}"]["any_mult_failed_where_STOP_passed"] = int(d.refcv7_failed_where_STOP_passed_any_mult.sum())
        cnt[f"stage{stg}"]["A1_zero_score_scenes"] = int((d.A1_score == 0).sum())
    # feasibility cross for the DAC-zero scenes: does a clean path exist (PDM-closed reference passes), does doing nothing pass,
    # does the kinematic prior pass, does the second inference seed pass
    z = F[(F.A1_DAC == 0) & F.resc_ok.fillna(False).astype(bool)]
    cross = {}
    for stg in (1, 2, 0):
        d = z if stg == 0 else z[z.stage == stg]
        c = {}
        for i in d.index:
            key = ("ref_clean" if d.loc[i, "ref_dac"] == 1 else "ref_FAILS") + "|" + ("STOP_clean" if d.loc[i, "STOP_DAC"] != 0 else "STOP_FAILS") + "|" + ("PRIOR_clean" if d.loc[i, "PRIOR_DAC"] != 0 else "PRIOR_FAILS")
            c[key] = c.get(key, 0) + 1
        cross["both" if stg == 0 else f"stage{stg}"] = {"n": int(len(d)), "cells": dict(sorted(c.items(), key=lambda kv: -kv[1]))}
        cross["both" if stg == 0 else f"stage{stg}"]["seed1_clean"] = int((d.A1s1_DAC != 0).sum())
        cross["both" if stg == 0 else f"stage{stg}"]["5k_clean"] = int((d.A1k5_DAC != 0).sum())
        cross["both" if stg == 0 else f"stage{stg}"]["any_of_STOP_PRIOR_seed1_clean_given_ref_clean"] = int(((d.ref_dac == 1) & ((d.STOP_DAC != 0) | (d.PRIOR_DAC != 0) | (d.A1s1_DAC != 0))).sum())
        cross["both" if stg == 0 else f"stage{stg}"]["ref_clean_and_none_of_STOP_PRIOR_seed1_clean"] = int(((d.ref_dac == 1) & (d.STOP_DAC == 0) & (d.PRIOR_DAC == 0) & (d.A1s1_DAC == 0)).sum())
    cnt["DAC_feasibility_cross"] = cross
    json.dump(cnt, open(os.path.join(RAW, "d6_failure_patterns.json"), "w"), indent=1)
    print(json.dumps(cnt, indent=0))
    print("rows", len(F), "cols", len(F.columns))


if __name__ == "__main__":
    main()
