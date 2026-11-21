"""REWARD-2 summary. usage: summarize_v6.py MAIN_DIR TTCOFF_DIR OUT.json DET1.txt DET2.txt LADDER_DIR [LADDER_DIR ...]
B1, B2, B4-B7 come from the UNCHANGED summarize_v4 (PREREG_M_GYM_4 section 4 text) run on the 20 full-arm states; B3 is the REWARD-1 rule, unchanged
(eligible states only) over every state whose ladder was run (the 20 full-arm states plus the ladder-only blocks). Secondary, never a substitute for B3: B3-F (free-road
states) and the kinematic crawl share (PREREG_REWARD_2.md section 3). curve_min_seg_m is asserted on every run dir and recorded. Bars are PREREG_REWARD_2.md as written."""
import gzip, json, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gym_env as g
import reward1 as r1
import reward2 as r2

main_dir, off_dir, out_path, det1, det2 = sys.argv[1:6]
lad_dirs = sys.argv[6:]
PRE = json.load(open(os.path.join(os.path.dirname(main_dir), "free_presim.json"), encoding="utf-8"))   # AMENDMENT 1: written before the first rollout
basis = os.path.join(os.path.dirname(out_path), "results_v4_basis.json")
subprocess.run([sys.executable, "-I", os.path.join(HERE, "summarize_v4.py"), main_dir, off_dir, basis],
               stdout=open(os.path.join(os.path.dirname(out_path), "summarize_v4_basis.out"), "w", encoding="utf-8"), stderr=subprocess.STDOUT, check=True)
B = json.load(open(basis, encoding="utf-8"))
meta_main = json.load(open(os.path.join(main_dir, "run_meta.json"), encoding="utf-8"))
out = dict(prereg="PREREG_REWARD_2.md", seed=r2.SEED_V6, floor=meta_main["floor"], curve_min_seg_m=meta_main["curve_min_seg_m"], tier="ENVIRONMENT (simulator) validity only")
assert meta_main["floor"] == r1.FLOOR, "the main run did not use FLOOR 0.20"
assert meta_main["curve_min_seg_m"] == r2.CURVE_MIN_SEG_M and meta_main["seed"] == r2.SEED_V6, "the main run did not use curve_min_seg_m 0.05 on seed 20261111"
assert json.load(open(os.path.join(off_dir, "run_meta.json"), encoding="utf-8"))["curve_min_seg_m"] == r2.CURVE_MIN_SEG_M


def rows_of(d, tier):
    out_ = []
    for l in gzip.open(os.path.join(d, "states.jsonl.gz"), "rt", encoding="utf-8"):
        r = json.loads(l)
        rets = {n: v["ret"] for n, v in r["ladder"].items()}
        hards = {n: any(x["hard"] for x in v["logs"]) for n, v in r["ladder"].items()}
        inf, kn = g.ladder_known(rets, hards)
        best = max(rets["f0.9"], rets["f1.0"])
        qf = [x["q"]["q_free"] for v in r["ladder"].values() for x in v["logs"] if x["active"]["q_free"]]
        assert r["curve_min_seg_m"] == r2.CURVE_MIN_SEG_M
        fr = r2.classify_free(r)
        out_.append(dict(idx=r["idx"], tier=tier, sampler=r["state"].get("sampler"), log=r["state"].get("log"), limit=r["limit"], informative=inf,
                         v0=fr["v0"], free_presim=PRE["states"].get(str(r["idx"]), {}).get("free_presim"), free=fr["free"], agent_in_window=fr["agent_in_window"], kin_crawl_over_best=r2.kin_crawl_over_best(fr["v0"], r["limit"]),
                         known=kn, crawl_over_limit=1.5 / r["limit"], crawl_over_best=(rets["crawl1.5"] / best) if best > 0 else None,
                         rets=rets, hard=hards, min_q_free=min(qf) if qf else None, n_cycles_qfree_below_0p30=int(sum(v < 0.30 - 1e-12 for v in qf))))
    return out_


def errors_of(d):
    pe = os.path.join(d, "errors.jsonl")
    return [json.loads(l)["idx"] for l in open(pe, encoding="utf-8")] if os.path.exists(pe) else []


rows = rows_of(main_dir, "full")
n_reached = r1.N_FULL
errs_ladder = []
for d in lad_dirs:
    m = json.load(open(os.path.join(d, "run_meta.json"), encoding="utf-8"))
    assert m["floor"] == r1.FLOOR and m["curve_min_seg_m"] == r2.CURVE_MIN_SEG_M and m["seed"] == r2.SEED_V6
    rows += rows_of(d, "ladder")
    n_reached = max(n_reached, m["range"][1])
    errs_ladder += errors_of(d)
rows.sort(key=lambda x: x["idx"])
errs_main = sorted(set(errors_of(main_dir)) | set(errors_of(off_dir)))
at_cap = n_reached >= r1.CAP
v = r1.b3_verdict(rows, at_cap)
inelig_inf = [dict(idx=r["idx"], limit=r["limit"], crawl_over_limit=r["crawl_over_limit"], crawl_over_best=r["crawl_over_best"], known=r["known"])
              for r in rows if r["informative"] and not r1.eligible(r["informative"], r["limit"])]
out["B3"] = dict(**v, n_states_reached=n_reached, at_cap=at_cap, n_informative=sum(r["informative"] for r in rows),
                 informative_rate=sum(r["informative"] for r in rows) / len(rows),
                 ineligible_informative_reported_not_judged=inelig_inf, build_or_run_failures_ladder_only=sorted(errs_ladder),
                 build_or_run_failures_full_arm_states=errs_main,
                 n_build_or_run_failures=len(errs_ladder) + len(errs_main),
                 per_state=[{k: r[k] for k in ("idx", "tier", "sampler", "log", "limit", "v0", "informative", "known", "free_presim", "free", "agent_in_window", "kin_crawl_over_best",
                                               "crawl_over_limit", "crawl_over_best")} for r in rows])
out["B3"]["eligible_detail"] = [dict(idx=r["idx"], limit=r["limit"], known=r["known"], crawl_over_best=r["crawl_over_best"], rets=r["rets"])
                                for r in rows if r1.eligible(r["informative"], r["limit"])]
# ---- secondary (section 3): B3-F on the free-road states, and the kinematic crawl share. Reported, never a substitute for B3.
out["B3_F"] = dict(**r2.b3f_verdict(rows, free_of=lambda r: bool(r["free_presim"])), definition="PREREG_REWARD_2_AMENDMENT_1: launch speed >= 1.0 m/s AND no lead/conflict agent at the logged ego poses of the 17-cycle window (log only, before any rollout)",
                   free_idx_all_states=[r["idx"] for r in rows if r["free_presim"]], n_presim_failed=PRE["n_failed"], n_presim_states=PRE["n"])
out["B3_F_rollout_any_rung"] = dict(**r2.b3f_verdict(rows), definition="launch speed >= 1.0 m/s AND no lead/conflict agent in any cycle of any of the six rungs. NOT pre-simulation; never decides a section 4 branch",
                                    free_idx_all_states=[r["idx"] for r in rows if r["free"]], n_states_where_the_two_classes_differ=sum(bool(r["free_presim"]) != bool(r["free"]) for r in rows))
elig_rows = [r for r in rows if r1.eligible(r["informative"], r["limit"])]
out["kinematic_crawl_share"] = dict(**r2.kin_share(elig_rows), bar=r2.CRAWL_RATIO,
                                    per_eligible_state=[dict(idx=r["idx"], v0=r["v0"], limit=r["limit"], kin_crawl_over_best=r["kin_crawl_over_best"],
                                                             observed_crawl_over_best=r["crawl_over_best"], known=r["known"]) for r in elig_rows])
# floor reachability evidence on the fresh data
mins = [r["min_q_free"] for r in rows if r["min_q_free"] is not None]
nb = int(sum(r["n_cycles_qfree_below_0p30"] for r in rows))
out["floor_reached_in_fresh_data"] = dict(min_q_free_over_all_ladder_cycles=min(mins) if mins else None, n_ladder_cycles_with_q_free_below_0p30=nb,
                                          ok=bool(mins and min(mins) >= r1.FLOOR - 1e-12 and nb > 0))
# ---- retained bars on the full-arm states (text of PREREG_M_GYM_4 section 4, via summarize_v4)
reg = B["base_regression_PREREG_M_GYM_1"] or {}


def parse(f):
    return json.loads([l for l in open(f, errors="ignore", encoding="utf-8") if l.startswith("JSON")][0][4:])


try:
    a, b = parse(det1), parse(det2)
    det = all(a[m][0] == a[m][1] == b[m][0] == b[m][1] for m in ("log", "idm"))
except Exception as e:
    det = "ERROR %r" % (e,)
ms = B["map_prescreen"]["per_state"]
all_off = [x for x in ms if x["n_offdrv_arms_log"] == 24 or x["n_offdrv_arms_idm"] == 24]
ttc = B["ttc"]
coll_states = sorted(int(i) for i, row in ttc["per_state"].items() if row["n_real_collision_arms"] > 0)
out["retained"] = dict(
    B1_oracle=("PASS" if B["oracle"]["BAR_oracle_pass"] else "FAIL"),
    B1_detail=dict(lex_hard_in_avoidable_cells=B["oracle"]["lex_hard_in_avoidable_cells"], unavoidable_fraction=B["oracle"]["unavoidable_fraction"],
                   unavoidable_cells=B["oracle"]["unavoidable_cells"]),
    B2_ttc=ttc["BAR_retention"],
    B2_detail=dict(real_flagged_by_legacy=ttc["real_flagged_by_legacy"], real_flagged_by_corridor=ttc["real_flagged_by_corridor"],
                   retention=ttc["retention"], n_distinct_real_collision_states=len(coll_states), collision_states=coll_states,
                   n_shadow_vs_live_mismatches=ttc["n_shadow_vs_live_mismatches"], M_GYM_4_distinct_collision_states=1),
    B4_always_slowest=reg.get("always_slowest"), B4_always_fastest=[reg.get("always_fastest_log"), reg.get("always_fastest_idm")],
    B4_do_nothing=reg.get("do_nothing"), B4_determinism_bit_identical=det,
    B5_cost=B["cost"], B6_prescreen=("UNDETERMINED" if not all_off else ("PASS" if all(x["prescreen_flag"] for x in all_off) else "FAIL")),
    B6_states_all_arms_offdrivable=[x["idx"] for x in all_off], B7_activation=B["base_activation_all"], base_hard_event_rates=B["base_hard_event_rates"],
    old_M_GYM_4_B3_text_on_full_arm_states_informational=B["ladder"]["BAR_ladder"])
sl = reg.get("always_slowest") or {}
out["retained"]["any_arm_not_known"] = [k for k, ok in (("always_slowest", sl.get("known")), ("always_fastest_log", (reg.get("always_fastest_log") or {}).get("known")),
                                                      ("always_fastest_idm", (reg.get("always_fastest_idm") or {}).get("known")),
                                                      ("do_nothing", (reg.get("do_nothing") or {}).get("known")), ("determinism", det is True)) if ok is not True]
json.dump(out, open(out_path, "w", encoding="utf-8"), indent=1, default=str)
print(json.dumps({"B3": {k: out["B3"][k] for k in ("verdict", "n_eligible", "n_known", "eligible_idx", "not_known_idx", "n_states_reached", "n_informative",
                                                   "ineligible_informative_reported_not_judged", "build_or_run_failures_ladder_only",
                                                   "build_or_run_failures_full_arm_states", "n_build_or_run_failures")},
                  "B3_F": {k: out["B3_F"][k] for k in ("verdict", "n_free_eligible", "n_known", "not_known_idx")},
                  "kinematic_crawl_share": {k: out["kinematic_crawl_share"][k] for k in ("n", "n_kin_above_bar", "share_above_bar", "mean", "min", "max")},
                  "curve_min_seg_m": out["curve_min_seg_m"],
                  "floor_reached_in_fresh_data": out["floor_reached_in_fresh_data"],
                  "retained": {k: out["retained"][k] for k in ("B1_oracle", "B2_ttc", "B2_detail", "B6_prescreen", "B4_always_slowest", "B4_do_nothing",
                                                                "B4_determinism_bit_identical", "any_arm_not_known")}}, indent=1, default=str))
