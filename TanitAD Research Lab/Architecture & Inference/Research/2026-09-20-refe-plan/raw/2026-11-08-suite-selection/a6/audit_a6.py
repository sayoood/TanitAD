"""AMENDMENT_6 audit + selector/scorer agreement. Reads rows3_a6.pkl, SUITE_CANDIDATES_A6.json (this folder) and ../a3/SUITE_CANDIDATES_A3.json (read only).
Writes AUDIT_GAP.json, AGREEMENT_A6.json, DIFF_VS_A3.json.   Run: python -I audit_a6.py"""
import os, sys, json, pickle, math, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import select_suite_tokens as SS
import a3_slots as A3
import behaviour_score as B
d = pickle.load(open(f"{HERE}/rows3_a6.pkl", "rb")); rows3, safe = d["rows3"], d["safe"]
c6 = json.load(open(f"{HERE}/SUITE_CANDIDATES_A6.json"))["slots"]
c3 = json.load(open(f"{HERE}/../a3/SUITE_CANDIDATES_A3.json"))["slots"]
byt = {r["token"]: r for r in rows3}
q = lambda a: {k: (float(np.percentile(a, p)) if len(a) else None) for k, p in (("p05", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95), ("max", 100))} | {"min": float(np.min(a)) if len(a) else None, "n": int(len(a))}

# ---- 1. diff of candidate lists / pools vs a3
diff = {}
for sl in A3.SLOTS_A3:
    o3 = [e["token"] for e in c3[sl]["ordered_one_per_log"]]; o6 = [e["token"] for e in c6[sl]["ordered_one_per_log"]]
    diff[sl] = dict(candidates_a3=c3[sl]["candidates"], candidates_a6=c6[sl]["candidates"], candidates_identical=c3[sl]["candidates"] == c6[sl]["candidates"],
                    ordered_a3=o3, ordered_a6=o6, ordered_identical=o3 == o6, added_to_ordered=[t for t in o6 if t not in o3], dropped_from_ordered=[t for t in o3 if t not in o6],
                    pool_tokens_a3=c3[sl]["pool_tokens_passing"], pool_tokens_a6=c6[sl]["pool_tokens_passing"], pool_logs_a3=c3[sl]["pool_logs_passing"], pool_logs_a6=c6[sl]["pool_logs_passing"],
                    before_safety_a3=c3[sl]["pool_tokens_before_human_safety"], before_safety_a6=c6[sl]["pool_tokens_before_human_safety"])
json.dump(diff, open(f"{HERE}/DIFF_VS_A3.json", "w"), indent=1)

# ---- 2. audit of the human stop gap to the new line (gap = line station - front-bumper station at the first stopped red tick, rear-axle path, 2 Hz)
def pool(flag, need_safe=True):
    return [r for r in rows3 if r.get(flag) and (not need_safe or safe.get(r["token"]))]
aud = {}
for name, flag in (("TL-G", "TL-G_c2code"), ("TL-S", "TL-S_5s_roll")):
    p_pass = pool(flag)                                           # the slot pool as it stands (cap 6 m applied by the predicate)
    p_all = pool("TL-G" if name == "TL-G" else "TL-S_5s_queued30")  # same predicate with the cap lifted to 30 m (rows carry the gap), human safe
    gaps = lambda L: np.array([r["stop_gap_m"] for r in L if "stop_gap_m" in r])
    over = sorted([dict(token=r["token"], log=r["log"], gap_m=r["stop_gap_m"], in_pool_cap6=bool(r.get(flag))) for r in pool("TL-G_queued30" if name == "TL-G" else "TL-S_5s_queued30") if r["stop_gap_m"] > A3.STOP_CAP], key=lambda x: x["gap_m"])
    aud[name] = dict(definition="stop_gap_m = stop-line station - front-bumper station (rear axle + 4.049 m) at the first tick with v<=0.1 m/s before the line on red; stations on the 2 Hz rear-axle window path, straight extension per AMENDMENT_6; line = connector start",
                     pool_cap6_human_safe=q(gaps(p_pass)), pool_cap6_tokens=[r["token"] for r in p_pass], gaps_cap6=sorted(round(float(x), 3) for x in gaps(p_pass)),
                     human_safe_tokens_above_6m_cap_with_cap_lifted_to_30m=over, n_above_cap=len(over),
                     before_human_safety_cap6=q(gaps([r for r in rows3 if r.get(flag)])), all_stopped_red_human_safe=q(np.array([r["stop_gap_m"] for r in rows3 if "stop_gap_m" in r and safe.get(r["token"])])))
json.dump(aud, open(f"{HERE}/AUDIT_GAP.json", "w"), indent=1)

# ---- 3. agreement: the 13 checked tokens (5 TL-G candidates + 8 TL-S ordered entries), selector station vs scorer station
A3.B  # noqa
tok13 = [(sl, e) for sl in ("TL-G", "TL-S") for e in c6[sl]["ordered_one_per_log"] if e["rank"] <= (5 if sl == "TL-G" else 8)]
import shapely as sh_
out = []
by_log = collections.defaultdict(list)
for sl, e in tok13: by_log[e["log"]].append((sl, e))
for log, lst in sorted(by_log.items()):
    ctx = SS.load_log(f"{SS.DB_DIR}/{log}.db")
    for sl, e in lst:
        mf = e["predicate_values"]["map_fields"]; r = byt[e["token"]]
        r0 = ctx.tok2rank[e["token"]]; K = np.arange(SS.SPEC["k_lo"], SS.SPEC["k_hi"] + 1); idx = r0 + SS.SPEC["step"] * K
        yaw = ctx.eyaw[idx]; v = ctx.ev[idx]
        rear = dict(x=ctx.ex[idx], y=ctx.ey[idx], yaw=np.unwrap(yaw), v=v)
        box = dict(x=ctx.ex[idx] + np.cos(yaw) * SS.EGO_COG_OFF, y=ctx.ey[idx] + np.sin(yaw) * SS.EGO_COG_OFF, yaw=np.unwrap(yaw), v=v)
        pt = np.array([mf["stop_line_xy"]])
        s_rear10 = float(B.HumanPath(rear).project(pt)[0][0])      # scorer's HumanPath on the rear-axle track at 10 Hz
        s_box10 = float(B.HumanPath(box).project(pt)[0][0])        # scorer's HumanPath on the box-centre track (the shipped stop_line_s)
        # selector's station: 2 Hz rear-axle window path of the predicates (a3_slots.station with ext_dir)
        frames = pickle.load(open(f"{A3.LOG_DIR}/{log}.pkl", "rb")); j = r["openscene_list_idx"]
        Wf = [A3.frame_globals(frames[k]) for k in range(j - A3.NB, j + A3.NA_ + 1)]
        P2 = np.array([[F["x"], F["y"]] for F in Wf]); V2 = np.array([F["v"] for F in Wf]); D2, _ = A3.densify(P2); D2 = np.vstack([D2, P2[-1]]) if float(np.hypot(*(D2[-1] - P2[-1]))) > 1e-6 else D2   # same tail rule as token_row
        dd = A3.ext_dir(P2, V2, Wf[-1]["h"]); s_sel = A3.station(sh_, D2, dd, pt[0])
        # the same station with the OLD extension (last heading) to show what ruling 2 changed
        s_sel_old = A3.station(sh_, D2, np.array([math.cos(Wf[-1]["h"]), math.sin(Wf[-1]["h"])]), pt[0])
        out.append(dict(slot=sl, rank=e["rank"], token=e["token"], stop_line_s_shipped=mf["stop_line_s"], s_box10=s_box10, s_rear10=s_rear10, s_selector_rear2hz=s_sel, s_selector_rear2hz_old_ext=s_sel_old, sel_minus_scorer_rear=s_sel - s_rear10, box_minus_rear=s_box10 - s_rear10, ext_dir_selector=r.get("ext_dir"),
                        human_path_length_box=mf["stop_line_s_diag"]["human_path_length_m"], beyond_human_end=mf["stop_line_s_diag"]["beyond_human_end"], stop_gap_m=r.get("stop_gap_m"),
                        in_candidates=e["is_candidate"]))
json.dump(out, open(f"{HERE}/AGREEMENT_A6.json", "w"), indent=1)
print(json.dumps(diff, indent=0)[:3000])
print(json.dumps({k: {x: v[x] for x in ("pool_cap6_human_safe", "n_above_cap")} for k, v in aud.items()}, indent=1))
