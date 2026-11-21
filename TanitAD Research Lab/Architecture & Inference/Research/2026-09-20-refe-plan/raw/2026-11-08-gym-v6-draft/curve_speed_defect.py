"""Analysis for CURVE_SPEED_DEFECT.md (fix = curve_speed_dedup, min_seg_m 0.05). CPU only, one process. Run with the driverl-eval python.
Inputs : raw/probe_b.json, raw/probe_c.json (Part 1 full probe), raw/cl/*.npz and raw/cl_fast/*.npz (real-map centrelines),
         pilot_v5 stored records, raw/resim/*.json (exact re-simulations, optional).
Output : curve_speed_defect.json
Evidence class: MEASURED (centreline from the real nuPlan maps via the gym's own calls) unless marked APPROX (s_end reconstructed from stored
progress_legacy; vbar := progress_legacy*limit, as reward2_counterfactuals.py Q1b)."""
import glob, gzip, json, os, sys, types
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); V5 = os.path.join(os.path.dirname(HERE), "pilot_v5")
sys.path.insert(0, os.path.join(HERE, "scorer_gym"))
import gym_env as g6
ARC = 0.05                                          # min_seg_m of the fix
RUNGS = ['crawl1.5', 'f0.3', 'f0.5', 'f0.7', 'f0.9', 'f1.0']


def cs(cl, s0, arc):
    return g6.RewardEvaluator.curve_speed(types.SimpleNamespace(cl_pts=cl["pts"], cl_s=cl["s"], curve_min_seg_m=arc), s0)


def load_cl():
    CL = {}
    for sub in ("cl_fast", "cl"):                       # full-planner build (cl) wins over fast build when both exist
        for f in sorted(glob.glob(os.path.join(HERE, "raw", sub, "cl_*.npz"))):
            z = np.load(f)
            CL[int(os.path.basename(f)[3:-4])] = dict(pts=z["pts"], s=z["s"], s0=float(z["s0"]), limit=float(z["limit"]), src=sub)
    return CL


S = {}
for sub in ("run_main", "ladder_20_160"):
    for l in gzip.open(os.path.join(V5, "raw", sub, "states.jsonl.gz"), "rt", encoding="utf-8"):
        d = json.loads(l); S[d["idx"]] = d
res5 = json.load(open(os.path.join(V5, "raw", "results_v5.json"), encoding="utf-8"))
ELIG = [e["idx"] for e in res5["B3"]["eligible_detail"]]
KNOWN = {e["idx"]: e["known"] for e in res5["B3"]["eligible_detail"]}
FAIL = sorted(i for i in ELIG if not KNOWN[i])
CL = load_cl()
out = dict(min_seg_m=ARC, n_states_with_ladder=len(S), n_centrelines=len(CL), missing_centrelines=sorted(set(S) - set(CL)))


def cyc_rows(idx, arc):
    d, cl = S[idx], CL[idx]; L = cl["limit"]; rows = []
    for rk in RUNGS:
        cum = 0.0
        for i, c in enumerate(d["ladder"][rk]["logs"]):
            cum += c["progress_legacy"] * L * 0.5
            se = cl["s0"] + cum
            v5 = min(max(d["limit"], 2.0), cs(cl, se, None)); v6 = min(max(d["limit"], 2.0), cs(cl, se, arc))
            sd = cl["s"][1:][np.diff(cl["s"]) < 0.05]                      # path positions of zero-length segments
            dup = bool(((sd >= se) & (sd <= se + 30.0)).any())
            rows.append(dict(rung=rk, i=i, se=se, v5=v5, v6=v6, dup=dup, free=c["lead_gap"] is None and c["conf_ttc"] is None, hard=bool(c["hard"]),
                             active=bool(c["active"]["q_free"]), prog=c["progress_legacy"], reward=c["reward"]))
    return rows


# ---- Part 1b: all states, per-cycle vstar v5 vs fix
per = {}
for idx in sorted(S):
    if idx not in CL:
        continue
    rows = cyc_rows(idx, ARC)
    free = [r for r in rows if r["free"] and not r["hard"]]
    n_v5_lt1 = sum(r["v5"] < 1.0 for r in free); n_v6_lt1 = sum(r["v6"] < 1.0 for r in free)
    flip = [r for r in free if r["v5"] < 1.0 <= r["v6"]]
    agree = sum((r["v5"] >= 1.0) == r["active"] for r in free)
    nd = [r for r in rows if not r["dup"]]
    relmove = max([abs(r["v6"] - r["v5"]) / r["v5"] for r in nd] or [0.0])
    nmove = sum(1 for r in nd if abs(r["v6"] - r["v5"]) / r["v5"] > 0.15)
    per[idx] = dict(src=CL[idx]["src"], n_free=len(free), n_v5_lt1=n_v5_lt1, n_v6_lt1=n_v6_lt1, n_flip=len(flip), agree_with_stored=agree,
                    max_rel_move_on_unaffected_cycles=relmove, n_cycles_without_dup=len(nd), n_cycles_with_dup=len(rows) - len(nd), n_dup_cycles_moved_gt15pct=sum(1 for r in rows if r["dup"] and abs(r["v6"] - r["v5"]) / r["v5"] > 0.15),
                    n_nodup_cycles_moved_gt15pct=nmove,
                    min_v5=min(r["v5"] for r in rows), min_v6=min(r["v6"] for r in rows))
out["per_state"] = per
aff = [i for i, p in per.items() if p["n_v5_lt1"] > 0]
out["summary"] = dict(
    states_scored=len(per), affected_v5_any_free_cycle_vstar_lt1=len(aff), affected_idx=aff,
    still_lt1_with_fix=[i for i in aff if per[i]["n_v6_lt1"] > 0],
    states_whose_vstar_crosses_1_with_fix=[i for i in aff if per[i]["n_flip"] > 0],
    n_states_cross=sum(per[i]["n_flip"] > 0 for i in aff),
    cycles_flipped=sum(p["n_flip"] for p in per.values()),
    stored_applicability_agreement=[sum(p["agree_with_stored"] for p in per.values()), sum(p["n_free"] for p in per.values())],
    unaffected_states=len(per) - len(aff),
    nodup_cycles=sum(p["n_cycles_without_dup"] for p in per.values()), dup_cycles=sum(p["n_cycles_with_dup"] for p in per.values()),
    nodup_cycles_moved_gt15pct=sum(p["n_nodup_cycles_moved_gt15pct"] for p in per.values()),
    nodup_max_rel_move=max(p["max_rel_move_on_unaffected_cycles"] for p in per.values()),
    dup_cycles_moved_gt15pct=sum(p["n_dup_cycles_moved_gt15pct"] for p in per.values()),
    states_with_a_dup_window_cycle=sum(p["n_cycles_with_dup"] > 0 for p in per.values()))

# ---- counterfactual verdicts on the 39 eligible, with the fixed vstar (APPROX for vbar; scripted rungs do not react to reward)
def q_free_value(vbar, vstar, floor): return float(np.clip(1.0 - 0.7 * (0.9 * vstar - vbar) / (0.4 * vstar), floor, 1.0))


def known_from(rets, hards):
    best = max(rets["f0.9"], rets["f1.0"])
    slow = [rets[n] for n in ("crawl1.5", "f0.3", "f0.5", "f0.7") if not hards[n]]
    return bool(best > 0 and all(r < 0.85 * best for r in slow) and rets["crawl1.5"] <= 0.10 * best)


cf = {}
for idx in ELIG:
    d = S[idx]; L = max(d["limit"], 2.0); rows = cyc_rows(idx, ARC)
    rets0 = {rk: sum(c["reward"] for c in d["ladder"][rk]["logs"]) for rk in RUNGS}
    rets1 = {}; hards = {rk: any(c["hard"] for c in d["ladder"][rk]["logs"]) for rk in RUNGS}
    for rk in RUNGS:
        tot = 0.0
        for i, c in enumerate(d["ladder"][rk]["logs"]):
            r = next(x for x in rows if x["rung"] == rk and x["i"] == i); w = 1.0
            if r["free"] and not r["hard"] and r["v6"] >= 1.0 and abs(r["v6"] - r["v5"]) > 1e-9:
                vbar = c["progress_legacy"] * d["limit"]
                new = q_free_value(vbar, r["v6"], 0.20)
                old = q_free_value(vbar, r["v5"], 0.20) if r["v5"] >= 1.0 else 1.0       # v5: not applicable below 1.0 -> factor 1
                w = new / old
            tot += c["reward"] * w
        rets1[rk] = tot
    b0 = max(rets0["f0.9"], rets0["f1.0"]); b1 = max(rets1["f0.9"], rets1["f1.0"])
    assert known_from(rets0, hards) == KNOWN[idx], idx           # instrument check: stored verdict reproduced before any counterfactual
    changed = sorted(rk for rk in RUNGS if abs(rets1[rk] - rets0[rk]) > 1e-12)
    cf[str(idx)] = dict(rungs_changed=changed, known_v5=KNOWN[idx], known_fix=known_from(rets1, hards), crawl_over_best_v5=rets0["crawl1.5"] / b0, crawl_over_best_fix=rets1["crawl1.5"] / b1,
                        f09_retention=rets1["f0.9"] / rets0["f0.9"], f10_retention=rets1["f1.0"] / rets0["f1.0"], n_flip=per[idx]["n_flip"])
out["counterfactual_39_APPROX"] = cf
out["counterfactual_summary"] = dict(
    changed_to_known=[int(k) for k, v in cf.items() if v["known_fix"] and not v["known_v5"]],
    changed_to_not_known=[int(k) for k, v in cf.items() if v["known_v5"] and not v["known_fix"]],
    known_fix=sum(v["known_fix"] for v in cf.values()), known_v5=sum(v["known_v5"] for v in cf.values()),
    states_touched=[int(k) for k, v in cf.items() if v["n_flip"] > 0],
    worst_f09_retention=min(v["f09_retention"] for v in cf.values()), worst_f10_retention=min(v["f10_retention"] for v in cf.values()))

# ---- exact re-simulation results, if present
rs = {}
for f in sorted(glob.glob(os.path.join(HERE, "raw", "resim", "resim_[0-9]*.json"))):
    r = json.load(open(f, encoding="utf-8")); rs[str(r["idx"])] = r
if rs:
    ex = {}
    for k, r in rs.items():
        d = S[int(k)]; rets0 = {rk: sum(c["reward"] for c in d["ladder"][rk]["logs"]) for rk in RUNGS}
        hards = {rk: bool(r["ladder"][rk]["hard"]) for rk in RUNGS}; rets1 = {rk: r["ladder"][rk]["ret"] for rk in RUNGS}
        b1 = max(rets1["f0.9"], rets1["f1.0"]); b0 = max(rets0["f0.9"], rets0["f1.0"])
        ex[k] = dict(arc=r["arc"], known_v5=KNOWN.get(int(k)), known_fix=known_from(rets1, hards), crawl_over_best_v5=rets0["crawl1.5"] / b0, crawl_over_best_fix=rets1["crawl1.5"] / b1,
                     f09_retention=rets1["f0.9"] / rets0["f0.9"], f10_retention=rets1["f1.0"] / rets0["f1.0"],
                     rets_fix=rets1, rets_v5=rets0, same_hard_flags=hards == {rk: any(c["hard"] for c in d["ladder"][rk]["logs"]) for rk in RUNGS})
    out["exact_resim"] = ex
# ---- sensitivity of the fix to its one parameter, and the alternative (heading over a 3 m arc; probe-side comparison only)
sens = {}
for ms in (0.01, 0.05, 0.1, 0.2):
    n_lt1 = 0; st_lt1 = set(); cyc_tot = 0
    for idx in S:
        if idx not in CL:
            continue
        for r in cyc_rows(idx, ms):
            if r["free"] and not r["hard"]:
                cyc_tot += 1
                if r["v6"] < 1.0:
                    n_lt1 += 1; st_lt1.add(idx)
    sens[str(ms)] = dict(free_cycles=cyc_tot, cycles_vstar_lt1=n_lt1, states_with_such_a_cycle=len(st_lt1))
out["sensitivity_min_seg_m"] = sens
# ---- Part 1 raw numbers per probed state (from curve_speed_probe.py output)
P = {}
for f in ("probe_b.json", "probe_c.json"):
    P.update(json.load(open(os.path.join(HERE, "raw", f), encoding="utf-8")))
affected_34 = {int(k) for k in json.load(open(os.path.join(HERE, "reward2_counterfactuals.json")))["Q1"]["states_crawl_free_geometry_cycles_with_q_free_not_applicable_(n_cycles,total)"]}
rows = []
for k in sorted(P, key=int):
    p = P[k]; w = p["worst_window"]; i = int(k)
    rows.append(dict(idx=i, cls="affected(34)" if i in affected_34 else "control", n_pts=p["n_pts"], zero_len_segments=p["n_ds_lt_1e3"], ds_median=p["ds_median"],
                     max_k_v5_rad_per_m=p["scan_k_v5_max"], max_k_heading_3m_arc=p["scan_k_arc3_max"], worst_ds_raw=w["ds_raw"], worst_dh_rad=w["dh_rad"],
                     worst_pts=w["pts"], applicability_pred_vs_stored=[p["pred_vs_stored_free_cycles"]["agree"], p["pred_vs_stored_free_cycles"]["n"]]))
out["part1_probe_rows"] = rows
json.dump(out, open(os.path.join(HERE, "curve_speed_defect.json"), "w"), indent=1, default=float)
print(json.dumps(out["summary"], indent=1, default=float)); print(json.dumps(out["counterfactual_summary"], indent=1, default=float))
