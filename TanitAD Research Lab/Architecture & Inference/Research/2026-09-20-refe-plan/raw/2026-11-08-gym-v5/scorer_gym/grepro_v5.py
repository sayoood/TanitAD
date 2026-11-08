"""G-REPRO + R1/R2/R3 of PREREG_REWARD_1 (sections 1 and 3) on USED states 3, 11, 17 of the M-GYM-4 list ONLY. No fresh state is read or drawn.
usage: grepro_v5.py OUT.json [--workers 3]
 R1 control : FLOOR 0.30 reproduces the 18 stored ladder returns (raw/results_v4.json ladder.per_state[*].rets) bit for bit; run twice (determinism).
 G-REPRO    : FLOOR 0.20 matches the counterfactual table of the card (b3_decomp/attrib.json floor_cf['<idx>']['0.2']) within 1 % relative:
              (a) crawl/best (the table's headline: 0.077 on state 3, 0.044 on state 11, 0.447 on state 17); (b) the 0.9x and 1.0x rungs (the side effect the card bounds);
              (c) the card calls the table "exact for scripted rungs, upper bound on the cut": a rung with a cycle between the old and new floor really loses LESS than the
              transform says, so for every rung the measured return must be >= the counterfactual (never below it). Per-rung deviations are reported, not gated beyond (c).
 R2         : q_free held at 1.0 (floor 1.0): crawl/best 0.3648 / 0.1691 / 0.5214 (state 3 / 11 / 17; 1 % relative) and NOT KNOWN on state 3.
 R3         : B4 always-slowest at FLOOR 0.20, pooled over the three states: mean q_free over applicable cycles < 0.6 AND mean return < PDM-Closed's (same floor).
Prints GREPRO_PASS or GREPRO_FAIL; exit 0 / 1."""
import json, multiprocessing as mp, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gym_env as g
import reward1 as r1

V4 = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v4"
USED = (3, 11, 17)
R2_EXPECT = {3: 0.3648, 11: 0.1691, 17: 0.5214}
REL = 0.01


def ladder(S):
    rets, hards = {}, {}
    for name, vt in S.ladder_targets():
        logs = S.ladder_rollout(vt, "log")
        rets[name] = g.episode_return(logs)
        hards[name] = any(x["hard"] for x in logs)
    return rets, hards


def work(idx):
    st = g.load_pinned_states()[idx]
    out = dict(idx=idx)
    mk = lambda fl: g.GymState(st, fixes="on", ttc_mode=None, ttc_shadow=True, q_free_floor=fl)
    S = mk(0.30)
    out["limit"] = S.limit
    out["r1_run1"], out["hard30"] = ladder(S)
    out["r1_run2"], _ = ladder(S)
    S = mk(r1.FLOOR)
    out["f20"], out["hard20"] = ladder(S)
    # R3 arms at floor 0.20 on the same state object
    ft = S.first_traj_for("slowest")
    lg = S.rollout(-1, "log", policy="slowest", first_traj=ft, timer=g.Timer())
    out["slowest_ret"] = g.episode_return(lg)
    out["slowest_qfree_applicable"] = [x["q"]["q_free"] for x in lg if x["active"]["q_free"]]
    pb = int(np.argmax(S.pdm_scores))
    out["pdm_ret"] = g.episode_return(S.rollout(pb, "log"))
    S = mk(1.0)
    out["f100"], out["hard100"] = ladder(S)
    return out


def best(r):
    return max(r["f0.9"], r["f1.0"])


def main():
    out_path = sys.argv[1]
    t0 = time.time()
    with mp.get_context("spawn").Pool(3) as pool:
        W = {w["idx"]: w for w in pool.map(work, USED)}
    stored = json.load(open(V4 + "/raw/results_v4.json", encoding="utf-8"))["ladder"]["per_state"]
    cf = json.load(open(V4 + "/b3_decomp/attrib.json", encoding="utf-8"))["floor_cf"]
    res = dict(wall_s=time.time() - t0, states={})
    ok = True
    q_all, rs, rp = [], [], []
    for i in USED:
        w = W[i]
        st_rets = stored[i]["rets"]
        assert stored[i]["idx"] == i
        r1_equal = all(w["r1_run1"][k] == st_rets[k] and w["r1_run2"][k] == st_rets[k] for k in st_rets)
        cfr = cf[str(i)]["0.2"]["rets"]
        rel = {k: abs(w["f20"][k] - cfr[k]) / abs(cfr[k]) for k in cfr}
        cfc = cf[str(i)]["0.2"]["crawl_over_best"]
        c20 = w["f20"]["crawl1.5"] / best(w["f20"])
        gate_a = abs(c20 - cfc) / cfc <= REL
        gate_b = all(rel[k] <= REL for k in ("f0.9", "f1.0"))
        gate_c = all(w["f20"][k] >= cfr[k] * (1 - 1e-9) for k in cfr)
        g_ok = bool(gate_a and gate_b and gate_c)
        c100 = w["f100"]["crawl1.5"] / best(w["f100"])
        r2_ok = abs(c100 - R2_EXPECT[i]) / R2_EXPECT[i] <= REL
        known100 = r1_known = None
        inf, kn = g.ladder_known(w["f100"], w["hard100"])
        known100 = (inf, kn)
        q_all += w["slowest_qfree_applicable"]; rs.append(w["slowest_ret"]); rp.append(w["pdm_ret"])
        res["states"][i] = dict(limit=w["limit"], R1_bit_identical_both_runs=r1_equal, n_returns_compared=len(st_rets),
                                GREPRO_max_rel_err_vs_counterfactual=max(rel.values()), GREPRO_rel_err_per_rung=rel,
                                GREPRO_gate_a_crawl_over_best=gate_a, GREPRO_gate_b_f09_f10=gate_b, GREPRO_gate_c_not_below_counterfactual=gate_c, GREPRO_ok=g_ok,
                                crawl_over_best_floor030=w["r1_run1"]["crawl1.5"] / best(w["r1_run1"]), crawl_over_best_floor020=c20,
                                crawl_over_best_floor100_R2=c100, R2_expected=R2_EXPECT[i], R2_ok=r2_ok, R2_known_informative=known100,
                                rets_floor020=w["f20"], rets_floor100=w["f100"], slowest_ret=w["slowest_ret"], pdm_ret=w["pdm_ret"],
                                slowest_mean_qfree=float(np.mean(w["slowest_qfree_applicable"])) if w["slowest_qfree_applicable"] else None)
        ok &= r1_equal and g_ok and r2_ok
    res["R2_state3_not_known"] = bool(res["states"][3]["R2_known_informative"][0] and res["states"][3]["R2_known_informative"][1] is False)
    res["R3"] = dict(pooled_mean_qfree=float(np.mean(q_all)) if q_all else None, n_applicable=len(q_all), mean_slowest_ret=float(np.mean(rs)),
                     mean_pdm_ret=float(np.mean(rp)))
    res["R3"]["known"] = bool(q_all and np.mean(q_all) < 0.6 and np.mean(rs) < np.mean(rp))
    ok &= res["R2_state3_not_known"] and res["R3"]["known"]
    res["R1"] = all(res["states"][i]["R1_bit_identical_both_runs"] for i in USED)
    res["GREPRO"] = all(res["states"][i]["GREPRO_ok"] for i in USED)
    res["R2"] = all(res["states"][i]["R2_ok"] for i in USED) and res["R2_state3_not_known"]
    res["PASS"] = bool(ok)
    json.dump(res, open(out_path, "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps({k: res[k] for k in ("R1", "GREPRO", "R2", "R3", "PASS")}, default=str))
    for i in USED:
        s = res["states"][i]
        print(i, {k: s[k] for k in ("R1_bit_identical_both_runs", "GREPRO_max_rel_err_vs_counterfactual", "crawl_over_best_floor030", "crawl_over_best_floor020", "crawl_over_best_floor100_R2", "R2_known_informative", "slowest_mean_qfree")})
    print("GREPRO_PASS" if ok else "GREPRO_FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
