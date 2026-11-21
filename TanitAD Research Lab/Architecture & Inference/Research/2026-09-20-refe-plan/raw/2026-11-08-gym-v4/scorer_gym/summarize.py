"""Summarise a run dir -> results.json + RESULT_TABLES.md (verdict thresholds are PREREG_M_GYM_1.md, unchanged)."""
import gzip, json, os, sys
import numpy as np
run = sys.argv[1]
R = [json.loads(l) for l in gzip.open(os.path.join(run, "states.jsonl.gz"), "rt")]
meta = json.load(open(os.path.join(run, "run_meta.json")))
TERMS = ["q_free", "q_gap", "q_stall", "q_yield", "q_comf", "q_osc", "q_lane", "q_cons"]
out = {"n_states": len(R), "workers": meta["workers"], "total_wall_s": meta["total_wall_s"]}
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
def pdm_ret(r, mode):
    return r["arms"][f"{mode}:{r['pdm_best']}"]["ret"] if f"{mode}:{r['pdm_best']}" in r["arms"] else None
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
json.dump(out, open(os.path.join(run, "results.json"), "w"), indent=1)
json.dump(rows, open(os.path.join(run, "cost_rows.json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ("cost", "buckets_mean_s_per_state", "hard_event_rates", "comfort_channel_counts", "regression")}, indent=1))
print("ACTIVATION(all)", json.dumps(out["activation"]["all"]))
