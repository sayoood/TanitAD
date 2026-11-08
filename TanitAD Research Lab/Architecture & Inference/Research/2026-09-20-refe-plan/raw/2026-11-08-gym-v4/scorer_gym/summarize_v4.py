"""M-GYM-4 summary. usage: summarize_v4.py RUN_MAIN_DIR RUN_TTCOFF_DIR OUT.json
Reads states.jsonl.gz of the fixes-ON run (main) and of the ttc-OFF shadow run (ttcoff). Bars are PREREG_M_GYM_4.md as written."""
import gzip, json, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gym_env as g

main_dir, off_dir, out_path = sys.argv[1:4]
# ---- base summary (PREREG_M_GYM_1 quantities: cost, activation, regression arms) via the unchanged v2 summariser
subprocess.run([sys.executable, os.path.join(HERE, "summarize_v2.py"), main_dir], stdout=open(os.path.join(main_dir, "summarize_v2.out"), "w"), check=True)
base = json.load(open(os.path.join(main_dir, "results.json")))
R = [json.loads(l) for l in gzip.open(os.path.join(main_dir, "states.jsonl.gz"), "rt")]
RO = [json.loads(l) for l in gzip.open(os.path.join(off_dir, "states.jsonl.gz"), "rt")] if os.path.exists(os.path.join(off_dir, "states.jsonl.gz")) else []
meta = json.load(open(os.path.join(main_dir, "run_meta.json")))
K = 24
out = dict(n_states=len(R), seed=g.SEED_V4)


def hard_of(a):
    return a["logs"][-1]["hard"]


# ================= fix 1 + 4: oracle, unavoidable, label-free =================
cells = []
for r in sorted(R, key=lambda x: x["idx"]):
    for mode in ("log", "idm"):
        rets = [r["arms"][f"{mode}:{k}"]["ret"] for k in range(K)]
        hards = [hard_of(r["arms"][f"{mode}:{k}"]) is not None for k in range(K)]
        ncy = [len(r["arms"][f"{mode}:{k}"]["logs"]) for k in range(K)]
        lex, am = g.oracle_pick(rets, hards, "lex"), g.oracle_pick(rets, hards, "argmax")
        st = g.cell_status(rets, hards, ncy)
        cells.append(dict(idx=r["idx"], mode=mode, sampler=r["state"].get("sampler"), n_clean=int(K - sum(hards)), lex_arm=lex, lex_hard=bool(hards[lex]),
                          lex_ret=rets[lex], argmax_arm=am, argmax_hard=bool(hards[am]), argmax_ret=rets[am], regret_lex_vs_argmax=rets[am] - rets[lex],
                          unavoidable=st["unavoidable"], label_free_tie=st["label_free_tie"], any_exact_tie=st["any_exact_tie"],
                          hard_types=sorted({hard_of(r["arms"][f"{mode}:{k}"]) for k in range(K) if hards[k]})))
avoidable = [c for c in cells if c["n_clean"] >= 1]
out["oracle"] = dict(
    n_cells=len(cells), n_cells_with_a_clean_arm=len(avoidable),
    lex_hard_in_avoidable_cells=[(c["idx"], c["mode"]) for c in avoidable if c["lex_hard"]],
    OLD_argmax_hard_in_avoidable_cells=[(c["idx"], c["mode"]) for c in avoidable if c["argmax_hard"]],
    unavoidable_cells=[(c["idx"], c["mode"], c["hard_types"]) for c in cells if c["unavoidable"]],
    label_free_tie_cells=[(c["idx"], c["mode"]) for c in cells if c["label_free_tie"]],
    any_exact_tie_cells=[(c["idx"], c["mode"]) for c in cells if c["any_exact_tie"]],
    n_cells_lex_differs_from_argmax=int(sum(c["lex_arm"] != c["argmax_arm"] and c["lex_ret"] != c["argmax_ret"] for c in cells)),
    mean_regret_lex_vs_argmax=float(np.mean([c["regret_lex_vs_argmax"] for c in cells])),
    max_regret_lex_vs_argmax=float(np.max([c["regret_lex_vs_argmax"] for c in cells])))
out["oracle"]["BAR_oracle_pass"] = bool(len(out["oracle"]["lex_hard_in_avoidable_cells"]) == 0)
out["oracle"]["unavoidable_fraction"] = float(sum(c["unavoidable"] for c in cells) / len(cells))
out["cells"] = cells

# ================= fix 3: map pre-screen =================
ms = []
for r in sorted(R, key=lambda x: x["idx"]):
    m = r["map"]
    any_hard = {mode: [hard_of(r["arms"][f"{mode}:{k}"]) for k in range(K)] for mode in ("log", "idm")}
    ms.append(dict(idx=r["idx"], sampler=r["state"].get("sampler"), centreline_overshoot=m["centreline_overshoot"], prescreen_flag=m["prescreen_flag"],
                   human_slack=m["slack"], unreliable_cap=m["unreliable"],
                   n_offdrv_arms_log=sum(h == "off_drivable" for h in any_hard["log"]), n_offdrv_arms_idm=sum(h == "off_drivable" for h in any_hard["idm"])))
out["map_prescreen"] = dict(per_state=ms, n_flagged=sum(x["prescreen_flag"] for x in ms), tol_m=g.PRESCREEN_TOL_M,
                            flagged_states_with_offdrv=[x["idx"] for x in ms if x["prescreen_flag"] and (x["n_offdrv_arms_log"] or x["n_offdrv_arms_idm"])],
                            UNflagged_states_with_offdrv_in_ANY_arm=[x["idx"] for x in ms if (not x["prescreen_flag"]) and (x["n_offdrv_arms_log"] or x["n_offdrv_arms_idm"])],
                            UNflagged_states_offdrv_ALL_arms=[x["idx"] for x in ms if (not x["prescreen_flag"]) and (x["n_offdrv_arms_log"] == K or x["n_offdrv_arms_idm"] == K)])

# ================= fix 2: ttc corridor, hazard retention (ttc-OFF shadow run is the ground truth) =================
S5 = g.CYCLE_STEPS


def first_event(logs):
    """global step of first legacy / corridor shadow flag and of the first real hard event (ttc never terminates in the off run)."""
    leg = cor = None
    for r in logs:
        sh = r.get("ttc_shadow") or {}
        if leg is None and "legacy_step" in sh:
            leg = r["cycle"] * S5 + sh["legacy_step"]
        if cor is None and "corridor_step" in sh:
            cor = r["cycle"] * S5 + sh["corridor_step"]
    h = logs[-1]["hard"]
    hs = logs[-1]["cycle"] * S5 + logs[-1]["hard_step"] if h else None
    return leg, cor, h, hs


ttc = dict(n_states=len(RO), per_state={})
Rn = leg_n = cor_n = lost = 0
fa_leg = fa_cor = nonreal = 0
mismatch = []
live = {(r["idx"], k): hard_of(a) for r in R for k, a in r["arms"].items()}
live_end = {(r["idx"], k): (a["logs"][-1]["cycle"] * S5 + (a["logs"][-1]["hard_step"] or 0)) for r in R for k, a in r["arms"].items() if hard_of(a)}
for r in sorted(RO, key=lambda x: x["idx"]):
    row = dict(n_real_collision_arms=0, real_flagged_legacy=0, real_flagged_corridor=0, real_lost=0, nonreal_legacy_flag=0, nonreal_corridor_flag=0)
    for key, a in r["arms"].items():
        mode, k = key.split(":")
        leg, cor, h, hs = first_event(a["logs"])
        # derived outcome of the corridor-mode / legacy-mode run from the off-run shadow
        cand = [(t, ty) for t, ty in ((cor, "ttc"), (hs, h)) if t is not None]
        dterm = min(cand) if cand else None
        lv = live.get((r["idx"], key))
        if any(x["idx"] == r["idx"] for x in R):
            lend = live_end.get((r["idx"], key))
            if (dterm is None) != (lv is None) or (dterm is not None and (dterm[1] != lv or dterm[0] != lend)):
                mismatch.append((r["idx"], key, dterm, lv, lend))
        if h == "collision":
            row["n_real_collision_arms"] += 1
            fl, fc = leg is not None and leg < hs, cor is not None and cor < hs
            row["real_flagged_legacy"] += fl
            row["real_flagged_corridor"] += fc
            row["real_lost"] += bool(fl and not fc)
        else:
            row["nonreal_legacy_flag"] += leg is not None and (hs is None or leg < hs)
            row["nonreal_corridor_flag"] += cor is not None and (hs is None or cor < hs)
    ttc["per_state"][r["idx"]] = row
    Rn += row["n_real_collision_arms"]; leg_n += row["real_flagged_legacy"]; cor_n += row["real_flagged_corridor"]; lost += row["real_lost"]
    fa_leg += row["nonreal_legacy_flag"]; fa_cor += row["nonreal_corridor_flag"]
ttc.update(n_real_collision_arm_episodes=Rn, real_flagged_by_legacy=leg_n, real_flagged_by_corridor=cor_n, real_flagged_by_legacy_but_NOT_corridor=lost,
           retention=(cor_n / leg_n) if leg_n else None,
           nonreal_episodes_flagged_by_legacy=fa_leg, nonreal_episodes_flagged_by_corridor=fa_cor,
           shadow_vs_live_mismatches=mismatch[:20], n_shadow_vs_live_mismatches=len(mismatch))
ttc["BAR_retention"] = ("UNDETERMINED_n<10" if leg_n < 10 else ("PASS" if cor_n / leg_n >= 0.80 else "FAIL"))
out["ttc"] = ttc

# ================= fix 6: constant-speed ladder =================
lad = []
for r in sorted(R, key=lambda x: x["idx"]):
    rets = {n: v["ret"] for n, v in r["ladder"].items()}
    hards = {n: any(x["hard"] for x in v["logs"]) for n, v in r["ladder"].items()}
    inf, kn = g.ladder_known(rets, hards)
    qs = sum(bool(x["q"]["q_stall"] < 1) for v in r["ladder"].values() for x in v["logs"])
    lad.append(dict(idx=r["idx"], informative=inf, known=kn, rets=rets, hard=hards, pdm_best_log_ret=r["arms"][f"log:{r['pdm_best']}"]["ret"], q_stall_fired_cycles=qs))
ni = [x for x in lad if x["informative"]]
out["ladder"] = dict(per_state=lad, n_informative=len(ni), n_known=sum(bool(x["known"]) for x in ni),
                     BAR_ladder=("UNDETERMINED_n<2" if len(ni) < 2 else ("PASS" if all(x["known"] for x in ni) else "FAIL")))

# ================= cost / sizing =================
cost = base["cost"]
wall_per_state = meta["total_wall_s"] / len(R)
out["cost"] = dict(single_worker_s_mean=cost["single_worker_s_mean"], single_worker_s_median=cost["single_worker_s_median"], single_worker_s_max=cost["single_worker_s_max"],
                   eff12_mean=cost["eff12_mean"], verdict_PREREG_M_GYM_1=cost["verdict"], measured_wall_s_per_state=wall_per_state, workers=meta["workers"],
                   states_per_3_days_at_measured_wall=259200.0 / wall_per_state,
                   states_per_3_days_at_12_workers_ESTIMATED=259200.0 / cost["eff12_mean"],
                   note="single_worker/eff12 follow the PREREG_M_GYM_1 definition: scene+map build + proposals + 24 arms x {log, idm}; regression policies and the ladder are extra")
task_wall = float(np.mean([r["wall_task"] for r in R]))
out["cost"]["task_wall_s_mean_incl_regression_and_ladder"] = task_wall
out["cost"]["ladder_wall_s_mean"] = float(np.mean([r.get("ladder_wall", 0.0) for r in R]))
out["cost"]["states_per_3_days_labelling_only_ESTIMATED"] = 259200.0 / (wall_per_state * (cost["single_worker_s_mean"] / task_wall))
out["base_regression_PREREG_M_GYM_1"] = base.get("regression")
out["base_activation_all"] = base.get("activation", {}).get("all")
out["base_hard_event_rates"] = base.get("hard_event_rates")
json.dump(out, open(out_path, "w"), indent=1, default=str)
print(json.dumps({k: out[k] for k in ("oracle", "ttc", "ladder", "cost")}, indent=1, default=str)[:12000])
print("map_prescreen", json.dumps(out["map_prescreen"]["per_state"], default=str)[:3000])
