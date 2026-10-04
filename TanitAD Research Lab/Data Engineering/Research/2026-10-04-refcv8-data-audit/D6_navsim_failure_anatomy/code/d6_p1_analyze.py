"""D6 P1 analysis -- the candidate-fan probe, exactly as pre-registered in SPEC_P1P2.md s3/s4/s6.   tanitad venv.

    python d6_p1_analyze.py [--dir raw] [--synthetic]        (--synthetic: label every output SYNTHETIC-TEST, never a result)

Inputs : raw/p1_fan/fan_R7_A1.jsonl (bridge export), raw/p1_scored_s*.jsonl + raw/p1_scored_K4.jsonl (d6_fan_score.py), raw/spec_tokens_*.txt,
         raw/d6_scene_geometry_step30000.csv (classes, stage, command, max-speed flag), the banked A1 / STOP frames, raw/p1_bridge/rows_R7_A1.jsonl.
Outputs: raw/d6_p1.json, raw/d6_p1_tables.md.  Intervals: log-cluster bootstrap (navhard logs), B = 2000, seed 0.
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d6_common as C

HERE = os.path.dirname(os.path.abspath(__file__))
RAWD = os.path.join(os.path.dirname(HERE), "raw")
W_CPASS = 3116 / 200.0
N_FAN = 117


def dec(b64, n):
    return None if b64 is None else np.frombuffer(base64.b64decode(b64), dtype=np.float32)[:n].copy()


_SETS_DIR = [RAWD]


def tokset(name):
    return [l.strip() for l in open(os.path.join(_SETS_DIR[0], f"spec_tokens_{name}.txt")) if l.strip()]


def boot_ratio(df, num, den, logcol="log", B=2000, seed=0):
    """sum(num)/sum(den) with a log-cluster bootstrap; num, den are column names."""
    if len(df) == 0 or df[den].sum() == 0:
        return None, [None, None]
    g = df.groupby(logcol)[[num, den]].sum()
    M = g.values
    rng = np.random.default_rng(seed)
    pt = M[:, 0].sum() / M[:, 1].sum()
    res = np.empty(B)
    for b in range(B):
        i = rng.integers(0, len(M), len(M))
        d = M[i, 1].sum()
        res[b] = M[i, 0].sum() / d if d else np.nan
    return float(pt), [float(np.nanpercentile(res, 2.5)), float(np.nanpercentile(res, 97.5))]


def band(F):
    return "SELECTION" if F >= 0.70 else "GENERATION" if F <= 0.30 else "MIXED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=RAWD)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--p1x", action="store_true", help="SPEC_P1P2_A2_P1PRIME: step-50,400 inputs / outputs, K4s (state-shift, bar 0.10), sets by rule")
    ap.add_argument("--hooks", default="", help="override the banked-hooks path (tests only)")
    ap.add_argument("--sets-dir", default=RAWD, help="where spec_tokens_p1x_*.txt and spec_p1x_token_sets.json live (default raw/)")
    a = ap.parse_args()
    tag = "SYNTHETIC-TEST (not a result)" if a.synthetic else "MEASURED"
    global W_CPASS
    if a.p1x:
        pfx, fan_dir, scored_glob, k4_file, k4_thr, k4_key = "p1x", "p1x_fan", "p1x_scored_s*.jsonl", "p1x_scored_K4s.jsonl", 0.10, "K4s_state_shift_plus30m_F_DAC"
        bridge_dir = "p1x_bridge"
        setfile = {"S1_dac_cleanpath": "p1x_S1", "S1b_dac_ref_fails": "p1x_S1b", "S2_nc": "p1x_S2", "Cpass": "p1x_Cpass", "P1_K1_fail": "p1x_K1fail"}
        hooks_p = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step50400/scores_navhard/"
                   "score_R7_A1__navhard_two_stage_wrapper/R7_A1__navhard_two_stage_hooks.json")
        W_CPASS = json.load(open(os.path.join(a.sets_dir, "spec_p1x_token_sets.json"), encoding="utf-8"))["N_pass"] / 200.0
        out_name, withheld_name = "d6_p1x.json", "d6_p1x_WITHHELD_controls_failed.json"
    else:
        pfx, fan_dir, scored_glob, k4_file, k4_thr, k4_key = "p1", "p1_fan", "p1_scored_s*.jsonl", "p1_scored_K4.jsonl", 0.05, "K4_mutation_plus30m_F_DAC"
        bridge_dir = "p1_bridge"
        setfile = {k: k for k in ("S1_dac_cleanpath", "S1b_dac_ref_fails", "S2_nc", "Cpass", "P1_K1_fail")}
        hooks_p = None
        out_name, withheld_name = "d6_p1.json", "d6_p1_WITHHELD_controls_failed.json"
    inp, mapping = C.load_inputs()
    G = pd.read_csv(os.path.join(RAWD, "d6_scene_geometry_step30000.csv")).set_index("token")
    T = pd.read_csv(os.path.join(RAWD, "d6_scene_table_step30000.csv")).set_index("token")
    stop = C.load_floor("STOP_zero")
    hooks_p = a.hooks or hooks_p
    A1h = {c["token"]: c for c in json.load(open(hooks_p if hooks_p else
        f"{C.milestone_dir(30000)}/scores_navhard/score_R7_A1__navhard_two_stage_wrapper/R7_A1__navhard_two_stage_hooks.json", encoding="utf-8"))["pdm_score_calls"]}
    _SETS_DIR[0] = a.sets_dir if a.p1x else RAWD
    sets = {k: set(tokset(setfile[k])) for k in ("S1_dac_cleanpath", "S1b_dac_ref_fails", "S2_nc", "Cpass", "P1_K1_fail")}
    fan = {}
    for ln in open(os.path.join(a.dir, fan_dir, "fan_R7_A1.jsonl"), encoding="utf-8"):
        try:
            r = json.loads(ln)
        except Exception:                                           # noqa: BLE001
            continue
        fan[r["token"]] = r
    sc = {}
    for p in sorted(glob.glob(os.path.join(a.dir, scored_glob))):
        for ln in open(p, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except Exception:                                       # noqa: BLE001
                continue
            sc[r["token"]] = r
    out = {"tag": tag, "n_fan_records": len(fan), "n_scored": len(sc), "n_scored_err": sum(1 for r in sc.values() if r["status"] != "OK")}
    rows = []
    for tok, r in sc.items():
        if r["status"] != "OK" or tok not in fan:
            continue
        f = fan[tok]
        n = r["n_cands"]
        nc, dac, ddc, tlc = (np.asarray(r[k][:n]) for k in ("nc", "dac", "ddc", "tlc"))
        cd = (dac == 1) & (ddc == 1)
        cf = cd & (nc == 1) & (tlc == 1)
        s9 = dec(f["sel_score_v3_b64"], N_FAN)
        rk = dec(f["reach_keep_b64"], N_FAN)
        s9m = np.where(rk > 0, s9, -np.inf) if rk is not None else s9
        s7 = dec(f.get("r7_score_b64"), n)
        if s7 is None:                          # amendment A1: no r7 scorer -> the E9 selection score ranks the 117 fan
            s7 = s9m[:n]
        order7 = np.argsort(-s7, kind="stable")
        rank7 = np.empty(n, int)
        rank7[order7] = np.arange(1, n + 1)
        order9 = np.argsort(-s9m, kind="stable")
        rank9 = np.empty(N_FAN, int)
        rank9[order9] = np.arange(1, N_FAN + 1)
        sel = f["sel_idx"]
        pe = np.asarray(r["end_dist"][:n])
        prog = np.asarray(r["prog_raw"][:n])
        rec = {"token": tok, "stage": inp[tok]["stage"], "log": inp[tok]["log_name"], "sel_idx": sel, "n": n}
        for lvl, cl in (("DAC", cd), ("FULL", cf)):
            rec[f"has_{lvl}"] = int(cl.any())
            rec[f"has117_{lvl}"] = int(cl[:N_FAN].any())
            rec[f"hasWTA_{lvl}"] = int(cl[N_FAN:].any())
            rec[f"nclean_{lvl}"] = int(cl.sum())
            rec[f"rnd_{lvl}"] = float(cl.mean())
            rec[f"pick_{lvl}"] = int(cl[sel])
            alt = f.get("r7_sel_idx") if f.get("r7_sel_idx") is not None else f.get("sel_idx_base")     # A1: R4' = the decoder's own pick (E9 graft off)
            rec[f"r4_{lvl}"] = int(cl[alt]) if alt is not None else np.nan
            if cl.any():
                rec[f"rank7_{lvl}"] = int(rank7[cl].min())
            if cl[:N_FAN].any():
                rec[f"rank9_{lvl}"] = int(rank9[cl[:N_FAN]].min())
        if cf.any():
            best = int(np.flatnonzero(cf)[np.argmin(rank7[cf])])
            rec["slower"] = int(pe[best] < pe[sel])
            rec["good_any"] = int(((prog >= prog[sel]) & cf).any())
        rec["tierc_confirmed"] = (None if "tierc" not in r else int(r["tierc"]["confirmed_clean"]))
        rec["pick_member_err"] = r["pick_member_vs_emitted_max_abs"]
        rec["stop_dac"], rec["stop_ddc"], rec["stop_tlc"] = r["dac"][n], r["ddc"][n], r["tlc"][n]
        rows.append(rec)
    R = pd.DataFrame(rows).set_index("token")
    for k in ("S1_dac_cleanpath", "S1b_dac_ref_fails", "S2_nc", "Cpass"):
        R[f"in_{k}"] = R.index.isin(sets[k])
    R["cls"] = G.loc[R.index, "cls"] if not a.p1x else np.where(G.loc[R.index, "route_turn"], "route-turns-in-horizon", "route-straight-in-horizon")
    R["vmax_known"] = G.loc[R.index, "vmax_known"] if "vmax_known" in G.columns else np.nan
    R["cmd"] = G.loc[R.index, "cmd"]
    out["n_analysed_scenes"] = int(len(R))
    # ---------------------------------------------------------------- the reads
    res = {}
    for name in ("S1_dac_cleanpath", "S1b_dac_ref_fails", "S2_nc", "Cpass"):
        d = R[R[f"in_{name}"]]
        res[name] = {"n": int(len(d))}
        for lvl in ("DAC", "FULL"):
            for key in (f"has_{lvl}", f"has117_{lvl}", f"hasWTA_{lvl}", f"r4_{lvl}"):
                pt, ci = boot_ratio(d.assign(one=1), key, "one")
                res[name][key] = {"value": pt, "ci95_logs": ci}
            res[name][f"rnd_{lvl}"] = float(d[f"rnd_{lvl}"].mean()) if len(d) else None
            hc = d[d[f"has_{lvl}"] == 1]
            if len(hc):
                r7 = hc[f"rank7_{lvl}"]
                res[name][f"rank7_{lvl}"] = {"n": int(len(hc)), "median": float(r7.median()), "top1": float((r7 <= 1).mean()), "top3": float((r7 <= 3).mean()),
                                             "top5": float((r7 <= 5).mean()), "top10": float((r7 <= 10).mean())}
            h9 = d[d[f"has117_{lvl}"] == 1]
            if len(h9):
                r9 = h9[f"rank9_{lvl}"]
                res[name][f"rank9_{lvl}"] = {"n": int(len(h9)), "median": float(r9.median()), "top1": float((r9 <= 1).mean()), "top5": float((r9 <= 5).mean())}
        if name == "S2_nc":
            hc = d[d["has_FULL"] == 1]
            res[name]["SLOWER"] = {"n": int(len(hc)), "share": float(hc["slower"].mean()) if len(hc) else None}
            res[name]["GOOD_any"] = {"n": int(len(hc)), "share": float(hc["good_any"].mean()) if len(hc) else None}
        tc = d["tierc_confirmed"].dropna()
        res[name]["tierC_confirmed"] = {"n": int(len(tc)), "share": float(tc.mean()) if len(tc) else None}
        # by class / stage / vmax / command (F at both levels), n >= 50 or UNDERPOWERED
        strata = {}
        for col in ("cls", "stage", "vmax_known", "cmd"):
            for k, g in d.groupby(col):
                strata[f"{col}={k}"] = {"n": int(len(g)), "underpowered": bool(len(g) < 50), "F_DAC": float(g["has_DAC"].mean()), "F_FULL": float(g["has_FULL"].mean()),
                                        "PICK_DAC": float(g["pick_DAC"].mean()), "R4_FULL": float(g["r4_FULL"].mean())}
        res[name]["strata"] = strata
    out["reads"] = res
    # ---------------------------------------------------------------- population PTI (Horvitz-Thompson)
    pop = R[R.in_S1_dac_cleanpath | R.in_S1b_dac_ref_fails | R.in_S2_nc | R.in_Cpass].copy()
    pop["w"] = np.where(pop.in_Cpass & ~(pop.in_S1_dac_cleanpath | pop.in_S1b_dac_ref_fails | pop.in_S2_nc), W_CPASS, 1.0)
    pt = {}
    for lvl in ("DAC", "FULL"):
        has = pop[f"has_{lvl}"] == 1
        w = pop["w"]
        pti = float((w * pop[f"pick_{lvl}"] * has).sum() / (w * has).sum()) if has.any() else None
        rnd = float((w * pop[f"rnd_{lvl}"] * has).sum() / (w * has).sum()) if has.any() else None
        orc = float((w * has).sum() / w.sum())
        pt[lvl] = {"PTI": pti, "RND_expected_uniform_pick_given_clean_exists": rnd, "oracle_F_population": orc, "n": int(len(pop)), "weights": {"failure": 1.0, "Cpass": W_CPASS}}
    out["population_PTI"] = pt
    # ---------------------------------------------------------------- controls
    ctl = {}
    ctl["K1_pick_member_equals_emitted_le_1e-4_share"] = float((R.pick_member_err <= 1e-4).mean())
    em = []
    for tok in R.index:
        em.append(float(np.abs(np.asarray(fan[tok]["poses_emitted"], dtype=np.float64) - np.asarray(A1h[tok]["agent_poses"], dtype=np.float64)).max()))
    ctl["K1_sidecar_emitted_vs_banked_plan_max_abs"] = float(np.max(em)) if em else None
    ctl["K1_sidecar_emitted_vs_banked_plan_share_le_1e-4"] = float(np.mean(np.asarray(em) <= 1e-4)) if em else None
    ex_ok, ex_n = 0, 0
    st_ok, st_n = 0, 0
    for tok, r in sc.items():
        if r.get("exact"):
            ex_n += 1
            d_ = max(abs(r["exact"]["pick"][k] - r["exact"]["banked_pick"][k]) for k in r["exact"]["banked_pick"])
            ex_ok += d_ <= 1e-9
            sf = stop.loc[tok]
            kmap = {"no_at_fault_collisions": "NC", "drivable_area_compliance": "DAC", "driving_direction_compliance": "DDC", "traffic_light_compliance": "TLC",
                    "ego_progress": "EP", "time_to_collision_within_bound": "TTC", "lane_keeping": "LK", "history_comfort": "HC"}
            st_n += 1
            st_ok += max(abs(r["exact"]["stop"][k] - float(sf[v])) for k, v in kmap.items()) <= 1e-9
    ctl["K1_exact_pick_equals_banked_row"] = {"ok": int(ex_ok), "n": int(ex_n)}
    ctl["K3_exact_STOP_equals_banked_STOP_row"] = {"ok": int(st_ok), "n": int(st_n)}
    s2 = R[R.stage == 2]
    sf2 = stop.loc[s2.index]
    ctl["K3_STOP_batch_DAC_DDC_TLC_equal_banked_stage2"] = float(((s2.stop_dac == sf2.DAC) & (s2.stop_ddc == sf2.DDC) & (s2.stop_tlc == sf2.TLC)).mean()) if len(s2) else None
    cp = sorted(sets["Cpass"] & set(R.index))
    if cp:
        ctl["K2_first_Cpass_token_has_clean_full"] = bool(R.loc[cp[0], "has_FULL"] == 1)
        ctl["K2_F_FULL_Cpass"] = float(R.loc[cp, "has_FULL"].mean())
        ctl["K2_F_DAC_Cpass"] = float(R.loc[cp, "has_DAC"].mean())
    k4 = {}
    for ln in open(os.path.join(a.dir, k4_file), encoding="utf-8") if os.path.exists(os.path.join(a.dir, k4_file)) else []:
        r = json.loads(ln)
        if r["status"] == "OK":
            n = r["n_cands"]
            k4[r["token"]] = int(any((r["dac"][i] == 1 and r["ddc"][i] == 1) for i in range(n)))
    ctl[k4_key] = {"n": len(k4), "F": float(np.mean(list(k4.values()))) if k4 else None}
    k5 = [r["k5"] for r in sc.values() if r.get("k5")]
    ctl["K5_batch_vs_single"] = {"n": len(k5), "all_equal": bool(all(x["single_dac"] == x["batch_dac"] and x["single_ddc"] == x["batch_ddc"] for x in k5)) if k5 else None}
    ctl["K6_sel_idx_in_first_117_share"] = float((R.sel_idx < N_FAN).mean())
    rows_p = os.path.join(a.dir, bridge_dir, "rows_R7_A1.jsonl")
    if os.path.exists(rows_p):
        dk = [json.loads(l)["diag"]["derived_ok"] for l in open(rows_p, encoding="utf-8") if l.strip() and json.loads(l).get("diag")]
        ctl["K6_derived_ok_share"] = float(np.mean(dk)) if dk else None
    out["controls"] = ctl
    ctl_pass = {
        "K1": ctl["K1_pick_member_equals_emitted_le_1e-4_share"] >= 0.99 and (ctl["K1_exact_pick_equals_banked_row"]["ok"] == ctl["K1_exact_pick_equals_banked_row"]["n"]),
        "K2": bool(ctl.get("K2_F_FULL_Cpass", 0) == 1.0 and ctl.get("K2_first_Cpass_token_has_clean_full", False)),
        "K3": (ctl["K3_STOP_batch_DAC_DDC_TLC_equal_banked_stage2"] or 0) >= 0.999 and ctl["K3_exact_STOP_equals_banked_STOP_row"]["ok"] == ctl["K3_exact_STOP_equals_banked_STOP_row"]["n"],
        "K4": (ctl[k4_key]["F"] is not None and ctl[k4_key]["F"] <= k4_thr),
        "K5": bool(ctl["K5_batch_vs_single"]["all_equal"]),
        "K6": ctl["K6_sel_idx_in_first_117_share"] == 1.0 and (ctl.get("K6_derived_ok_share") or 0) >= 0.995}
    out["control_verdicts"] = ctl_pass
    out["ALL_CONTROLS_PASS"] = bool(all(ctl_pass.values()))
    # ---------------------------------------------------------------- the pre-registered reading (SPEC s6)
    rd = {}
    s1 = res["S1_dac_cleanpath"]
    if s1["n"]:
        F = s1["has_DAC"]["value"]
        PTI = pt["DAC"]["PTI"]
        R4 = s1["r4_DAC"]["value"]
        rk = s1.get("rank7_DAC", {})
        rd["S1_DAC"] = {"F": F, "band": band(F), "PTI_population": PTI, "R4": R4,
                        "selection_dominant": bool(F >= 0.70 and PTI is not None and PTI < 0.5 * F),
                        "free_lever_E9_graft_off_decoder_pick": bool(F >= 0.70 and R4 >= 0.5 * F),
                        "selector_blind_rank_median_gt_30": bool(rk.get("median", 0) > 30),
                        "mild_rerank_top5_ge_0.5": bool(rk.get("top5", 0) >= 0.5)}
    s2 = res["S2_nc"]
    if s2["n"]:
        F = s2["has_FULL"]["value"]
        sl = s2["SLOWER"]["share"]
        rd["S2_NC"] = {"F": F, "band": band(F), "SLOWER": sl, "reading": ("speed commit (selector speed / L3 / L1 lon)" if (sl is not None and sl >= 0.80) else
                                                                           "evasive / perception (box, lateral generation)" if (sl is not None and sl < 0.50) else "both"),
                       "R4": s2["r4_FULL"]["value"]}
    out["pre_registered_reading"] = rd
    dflt = lambda o: o.item() if hasattr(o, "item") else str(o)
    if not out["ALL_CONTROLS_PASS"]:
        # SPEC s4: "any control failing -> the affected read is INCONCLUSIVE". The numbers are kept apart and the headline file carries NO reading.
        failed = [k for k, v in ctl_pass.items() if not v]
        json.dump({"WITHHELD_controls_failed": failed, "reads": out.pop("reads"), "population_PTI": out.pop("population_PTI"), "pre_registered_reading": out.pop("pre_registered_reading")},
                  open(os.path.join(a.dir, withheld_name), "w", encoding="utf-8"), indent=1, default=dflt)
        out["pre_registered_reading"] = {"status": "INCONCLUSIVE (SPEC s4): controls failed " + ",".join(failed), "numbers": "raw/" + withheld_name + " (not quotable)"}
        json.dump(out, open(os.path.join(a.dir, out_name), "w", encoding="utf-8"), indent=1, default=dflt)
        print(json.dumps({"tag": tag, "controls": ctl_pass, "reading": out["pre_registered_reading"]}, indent=1))
        return
    json.dump(out, open(os.path.join(a.dir, out_name), "w", encoding="utf-8"), indent=1, default=dflt)
    print(json.dumps({"tag": tag, "controls": ctl_pass, "reading": rd}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
