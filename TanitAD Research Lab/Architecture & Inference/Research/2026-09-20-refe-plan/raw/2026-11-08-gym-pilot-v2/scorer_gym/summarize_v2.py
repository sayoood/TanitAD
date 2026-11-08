"""M-GYM-2 summary of a run dir -> results.json (verdict thresholds are PREREG_M_GYM_1.md, unchanged; v2 adds before/after rows)."""
import gzip, json, os, sys
import numpy as np
run = sys.argv[1]
R = [json.loads(l) for l in gzip.open(os.path.join(run, "states.jsonl.gz"), "rt")]
meta = json.load(open(os.path.join(run, "run_meta.json")))
TERMS = ["q_free", "q_gap", "q_stall", "q_yield", "q_comf", "q_osc", "q_lane", "q_cons"]
out = {"n_states": len(R), "workers": meta["workers"], "total_wall_s": meta["total_wall_s"], "sampler": meta.get("sampler", "pilot"), "hold_cycles": meta.get("hold_cycles", 1)}
# ---- cost
rows = []
for r in R:
    b = r["build"]; build = b.get("scene_map_build", 0) + b.get("proposals", 0)
    per = {m: sum(a["wall"] for k, a in r["arms"].items() if k.startswith(m + ":")) for m in ("log", "idm")}
    rows.append(dict(idx=r["idx"], log=r["state"]["log"], v0=r["v0"], n_obj0=r["n_obj0"], scene_map_build=b.get("scene_map_build", 0),
                     proposals=b.get("proposals", 0), arms_log=per["log"], arms_idm=per["idm"], total=build + per["log"] + per["idm"],
                     warm_total=None))
med_build = float(np.median([x["scene_map_build"] for x in rows]))
for x in rows:
    x["warm_total"] = x["total"] - x["scene_map_build"] + med_build
tot = np.array([x["total"] for x in rows]); warm = np.array([x["warm_total"] for x in rows])
cost = dict(single_worker_s_mean=float(tot.mean()), single_worker_s_median=float(np.median(tot)), single_worker_s_max=float(tot.max()),
            eff12_mean=float(tot.mean() / 12), eff12_median=float(np.median(tot) / 12), eff12_max=float(tot.max() / 12),
            eff12_warm_mean=float(warm.mean() / 12),
            eff12_log_only=float(np.mean([x["scene_map_build"] + x["proposals"] + x["arms_log"] for x in rows]) / 12),
            eff12_idm_only=float(np.mean([x["scene_map_build"] + x["proposals"] + x["arms_idm"] for x in rows]) / 12),
            measured_6w_wall_per_state=float(meta["total_wall_s"] / len(R)),
            measured_6w_task_wall_mean=float(np.mean([r["wall_task"] for r in R])))
m = cost["eff12_mean"]
cost["verdict"] = "GO" if m <= 26 else ("REDUCE" if m <= 260 else "BLOCKED")
out["cost"] = cost
# bucket split (mean over states, single worker seconds)
bk = {}
for r in R:
    for k, v in r["timer_main"].items():
        bk[k] = bk.get(k, 0) + v / len(R)
out["buckets_mean_s_per_state"] = bk
# ---- activation
rec = []
for r in R:
    for k, a in r["arms"].items():
        mode = k.split(":")[0]
        for x in a["logs"]:
            rec.append((mode, x))
act = {}
for mode in ("log", "idm", "all"):
    sel = [x for md, x in rec if mode == "all" or md == mode]
    d = {}
    for t in TERMS:
        fired = np.mean([x["q"][t] < 1 - 1e-9 for x in sel])
        app = np.mean([x["active"][t] for x in sel]) if t in ("q_free", "q_gap") else None
        d[t] = dict(fired=float(fired), applicable=None if app is None else float(app),
                    mean_q=float(np.mean([x["q"][t] for x in sel])), sparse=bool(fired < 0.01))
    d["n_cycle_records"] = len(sel)
    act[mode] = d
out["activation"] = act
eps = {}
for mode in ("log", "idm"):
    E = [a["logs"] for r in R for k, a in r["arms"].items() if k.startswith(mode + ":")]
    h = [e[-1]["hard"] for e in E]
    eps[mode] = dict(n_episodes=len(E), **{t: float(np.mean([x == t for x in h])) for t in ("collision", "off_drivable", "wrong_way", "ttc")},
                     any=float(np.mean([x is not None for x in h])))
out["hard_event_rates"] = eps
comf = {}
for md, x in rec:
    for c in x["comf"]:
        comf[c] = comf.get(c, 0) + 1
out["comfort_channel_counts"] = comf
# ---- first-action spread (does the first action matter?)
spread = []
for r in R:
    for mode in ("log", "idm"):
        rs = [a["ret"] for k, a in r["arms"].items() if k.startswith(mode + ":")]
        spread.append(dict(idx=r["idx"], mode=mode, ret_min=min(rs), ret_max=max(rs), ret_mean=float(np.mean(rs))))
out["first_action_return_spread"] = spread
# ---- regression arms
reg = {}
HAS_REG = any(r["reg"] for r in R)
def pdm_ret(r, mode):
    return r["arms"][f"{mode}:{r['pdm_best']}"]["ret"] if f"{mode}:{r['pdm_best']}" in r["arms"] else None
if HAS_REG:
    # slowest
    qf, rs, rp = [], [], []
    for r in R:
        if "slowest:log" in r["reg"]:
            L = r["reg"]["slowest:log"]["logs"]
            qf += [x["q"]["q_free"] for x in L if x["active"]["q_free"]]
            rs.append(r["reg"]["slowest:log"]["ret"]); rp.append(pdm_ret(r, "log"))
    reg["always_slowest"] = dict(mean_q_free_applicable=float(np.mean(qf)) if qf else None, n_applicable=len(qf),
                                 mean_return=float(np.mean(rs)), mean_return_pdm=float(np.mean(rp)))
    reg["always_slowest"]["known"] = bool(qf and np.mean(qf) < 0.6 and np.mean(rs) < np.mean(rp))
    # fastest
    for mode in ("log", "idm"):
        f_cnt = p_cnt = 0
        for r in R:
            if f"fastest:{mode}" in r["reg"]:
                for tag, L in (("f", r["reg"][f"fastest:{mode}"]["logs"]), ("p", r["arms"][f"{mode}:{r['pdm_best']}"]["logs"])):
                    c = sum(1 for x in L if x["hard"]) + sum(1 for x in L if x["lead_gap"] is not None and x["q"]["q_gap"] < 0.7)
                    if tag == "f": f_cnt += c
                    else: p_cnt += c
        reg[f"always_fastest_{mode}"] = dict(fail_count_fastest=f_cnt, fail_count_pdm=p_cnt, known=bool(f_cnt > p_cnt))
    reg["always_fastest"] = dict(known=bool(reg["always_fastest_log"]["known"] or reg["always_fastest_idm"]["known"]))
    # stop
    sv = [(r["reg"]["stop:log"]["ret"], len(r["reg"]["stop:log"]["logs"])) for r in R if r["v0"] > 0.5 and "stop:log" in r["reg"]]
    reg["do_nothing"] = dict(n_moving_states=len(sv), mean_reward_per_cycle=float(np.mean([a / b for a, b in sv])) if sv else None,
                             known=bool(sv and np.mean([a / b for a, b in sv]) <= 0.02))
    post = [x["reward"] for r in R if "stop:log" in r["reg"] for x in r["reg"]["stop:log"]["logs"] if x["v"] < 0.05]
    reg["do_nothing"]["POST_HOC_mean_reward_on_cycles_with_v_lt_0.05"] = float(np.mean(post)) if post else None
    reg["do_nothing"]["POST_HOC_n_cycles_v_lt_0.05"] = len(post)
    # oracle
    oc = {"log": [0, 0], "idm": [0, 0]}
    for r in R:
        for mode in ("log", "idm"):
            ks = [k for k in r["arms"] if k.startswith(mode + ":")]
            best = max(ks, key=lambda k: r["arms"][k]["ret"])
            h = r["arms"][best]["logs"][-1]["hard"] is not None
            allh = all(r["arms"][k]["logs"][-1]["hard"] is not None for k in ks)
            oc[mode][0] += int(h); oc[mode][1] += int(h and allh)
    reg["oracle"] = dict(hard_events_log=oc["log"][0], hard_events_idm=oc["idm"][0],
                         hard_events_where_every_arm_also_hard_log=oc["log"][1], hard_events_where_every_arm_also_hard_idm=oc["idm"][1],
                         known=bool(oc["log"][0] + oc["idm"][0] == 0))
    out["regression"] = reg

# ================= v2 additions =================
v2 = {}
# q_comf: smoothed (v2) vs raw (pilot estimator) on the SAME rollouts
def raw_fired(x):
    others = set(x["comf"]) - {"lon_jerk", "mag_jerk"}
    return bool(others | set(x.get("comf_raw_jerk") or []))
cyc = {}
for r in R:
    for k, a in r["arms"].items():
        for x in a["logs"]:
            d = cyc.setdefault(x["cycle"], [0, 0, 0]); d[0] += bool(x["comf"]); d[1] += raw_fired(x); d[2] += 1
v2["comf_by_cycle"] = {c: dict(smoothed=d[0] / d[2], raw=d[1] / d[2], n=d[2]) for c, d in sorted(cyc.items())}
allrec = [x for r in R for a in r["arms"].values() for x in a["logs"]]
v2["comf_overall"] = dict(smoothed=float(np.mean([bool(x["comf"]) for x in allrec])), raw=float(np.mean([raw_fired(x) for x in allrec])), n=len(allrec))
bestrec = [x for r in R for m in ("log", "idm") if f"{m}:{r['pdm_best']}" in r["arms"] for x in r["arms"][f"{m}:{r['pdm_best']}"]["logs"]]
v2["comf_pdm_best_arm"] = dict(smoothed=float(np.mean([bool(x["comf"]) for x in bestrec])), raw=float(np.mean([raw_fired(x) for x in bestrec])), n=len(bestrec))
v2["lon_jerk_max_quantiles"] = {q: float(np.quantile([x["lon_jerk_max"] for x in allrec if x.get("lon_jerk_max") is not None], q)) for q in (0.5, 0.9, 0.99)}
# reward: legacy (pilot formula, same rollouts) vs v2
def ret(logs, key): return float(sum(x[key] for x in logs))
def mean_ret(sel, key): return float(np.mean([ret(a["logs"], key) for a in sel])) if sel else None
rw = {}
for mode in ("log", "idm"):
    arms = [a for r in R for k, a in r["arms"].items() if k.startswith(mode + ":")]
    best = [r["arms"][f"{mode}:{r['pdm_best']}"] for r in R]
    rw[mode] = dict(all24_mean_ret_v2=mean_ret(arms, "reward"), all24_mean_ret_legacy=mean_ret(arms, "reward_legacy"),
                    pdmbest_mean_ret_v2=mean_ret(best, "reward"), pdmbest_mean_ret_legacy=mean_ret(best, "reward_legacy"))
    rs = [(ret(a["logs"], "reward"), ret(a["logs"], "reward_legacy")) for a in arms]
    rw[mode]["spearman_arm_return_v2_vs_legacy_pooled"] = float(np.corrcoef(np.argsort(np.argsort([x[0] for x in rs])), np.argsort(np.argsort([x[1] for x in rs])))[0, 1])
v2["reward_before_after"] = rw
if HAS_REG:
    regr = {}
    for nm in ("slowest:log", "fastest:log", "fastest:idm", "stop:log"):
        L = [r["reg"][nm]["logs"] for r in R if nm in r["reg"]]
        regr[nm] = dict(mean_ret_v2=float(np.mean([ret(l, "reward") for l in L])), mean_ret_legacy=float(np.mean([ret(l, "reward_legacy") for l in L])), n_states=len(L))
    mv = [r for r in R if r["v0"] > 0.5 and "stop:log" in r["reg"]]
    regr["do_nothing_per_cycle_moving"] = dict(
        v2=float(np.mean([ret(r["reg"]["stop:log"]["logs"], "reward") / len(r["reg"]["stop:log"]["logs"]) for r in mv])),
        legacy=float(np.mean([ret(r["reg"]["stop:log"]["logs"], "reward_legacy") / len(r["reg"]["stop:log"]["logs"]) for r in mv])), n=len(mv))
    v2["regression_returns_before_after"] = regr
# oracle detail
od = {}
for mode in ("log", "idm"):
    rows = []
    for r in R:
        ks = [k for k in r["arms"] if k.startswith(mode + ":")]
        best = max(ks, key=lambda k: r["arms"][k]["ret"])
        hs = [r["arms"][k]["logs"][-1]["hard"] for k in ks]
        rows.append(dict(idx=r["idx"], best_hard=r["arms"][best]["logs"][-1]["hard"], best_cycles=len(r["arms"][best]["logs"]),
                         n_arms_hard=sum(h is not None for h in hs), first_hard_cycles=sorted({len(r["arms"][k]["logs"]) - 1 for k in ks if r["arms"][k]["logs"][-1]["hard"]})))
    od[mode] = [x for x in rows if x["best_hard"] is not None]
v2["oracle_hard_states"] = od
# q_stall precondition (stopped for the last 4 cycle-ends incl. v0 padding): how many cycle records, how many fired, why excused
pre = dict(n=0, fired=0, excused_lead=0, excused_conflict=0, excused_other_red=0)
pre_states = {}
for r in R:
    for k, a in r["arms"].items():
        vs = [r["v0"]] * 4
        for x in a["logs"]:
            vs.append(x["v"])
            if all(abs(u) < 0.5 for u in vs[-4:]):
                pre["n"] += 1; f = x["q"]["q_stall"] < 1; pre["fired"] += f
                if not f:
                    if x["conf_ttc"] is not None: pre["excused_conflict"] += 1
                    elif x["lead_gap"] is not None and x["lead_gap"] <= 10.0: pre["excused_lead"] += 1
                    else: pre["excused_other_red"] += 1
                pre_states[r["idx"]] = pre_states.get(r["idx"], 0) + 1
v2["stall_precondition"] = dict(pre, states=pre_states, standing_states_v0_lt_0p5=[r["idx"] for r in R if r["v0"] < 0.5])
# hold knob / ranking helper data: per (state, mode) return vectors
v2["returns_by_state_mode"] = {f"{r['idx']}:{m}": [r["arms"][f"{m}:{k}"]["ret"] for k in range(24) if f"{m}:{k}" in r["arms"]] for r in R for m in ("log", "idm")}
out["v2"] = v2
json.dump(out, open(os.path.join(run, "results.json"), "w"), indent=1)
json.dump(rows, open(os.path.join(run, "cost_rows.json"), "w"), indent=1)
print(json.dumps({k: out.get(k) for k in ("cost", "buckets_mean_s_per_state", "hard_event_rates", "comfort_channel_counts", "regression")}, indent=1))
print("ACTIVATION(all)", json.dumps(out["activation"]["all"]))
