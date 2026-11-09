"""R1, R2, R3 of PREREG_REWARD_2 section 3 on USED states 3, 11, 17 of the M-GYM-4 list ONLY (g.load_pinned_states()). No state of seed 20261111 is read or drawn.
usage: grepro_v6.py OUT.json [--workers 2]
 R1 : curve_min_seg_m = None (switch OFF), floor 0.20, reproduces the stored M-GYM-5 ladder returns of these states (pilot_v5/raw/grepro_v5.json rets_floor020, and slowest/PDM
      returns) bit for bit, and determinism holds (the ladder run twice in one process and the stored values are identical).
 R2 : switch OFF, q_free held at 1.0 (floor 1.0): crawl/best 0.3648 / 0.1691 / 0.5214 (state 3 / 11 / 17; within 1 % relative) and NOT KNOWN on state 3; also equal to the stored
      rets_floor100 bit for bit.
 R3 : B4 always-slowest with the switch ON (curve_min_seg_m = 0.05, floor 0.20), pooled over the three states: mean q_free over applicable cycles < 0.6 AND mean return < PDM-Closed's
      (same configuration) -> KNOWN. Control (switch OFF) is the stored M-GYM-5 value, compared inside R1.
Every output records curve_min_seg_m per configuration. Prints GREPRO_PASS or GREPRO_FAIL; exit 0 / 1."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gym_env as g
import reward1 as r1
import reward2 as r2
import ramguard

P5 = r2.P5
USED = (3, 11, 17)
R2_EXPECT = {3: 0.3648, 11: 0.1691, 17: 0.5214}
REL = 0.01
ON = r2.CURVE_MIN_SEG_M


def ladder(S):
    rets, hards = {}, {}
    for name, vt in S.ladder_targets():
        logs = S.ladder_rollout(vt, "log")
        rets[name] = g.episode_return(logs)
        hards[name] = any(x["hard"] for x in logs)
    return rets, hards


def slowest_and_pdm(S):
    ft = S.first_traj_for("slowest")
    lg = S.rollout(-1, "log", policy="slowest", first_traj=ft, timer=g.Timer())
    pdm = g.episode_return(S.rollout(int(np.argmax(S.pdm_scores)), "log"))
    return g.episode_return(lg), [x["q"]["q_free"] for x in lg if x["active"]["q_free"]], pdm


def work(args):
    idx, workers_label = args
    st = g.load_pinned_states()[idx]
    out = dict(idx=idx)
    mk = lambda fl, arc: g.GymState(st, fixes="on", ttc_mode=None, ttc_shadow=True, q_free_floor=fl, curve_min_seg_m=arc)
    S = mk(0.20, None)                                  # R1: switch OFF
    assert S.evaluator.curve_min_seg_m is None
    out["limit"] = S.limit
    out["r1_run1"], out["r1_hard1"] = ladder(S)
    out["r1_run2"], _ = ladder(S)
    out["r1_slowest_ret"], out["r1_slowest_q"], out["r1_pdm_ret"] = slowest_and_pdm(S)
    S = mk(1.0, None)                                   # R2: q_free held at 1.0, switch OFF
    assert S.evaluator.curve_min_seg_m is None
    out["r2_rets"], out["r2_hard"] = ladder(S)
    S = mk(0.20, ON)                                    # R3: switch ON
    assert S.evaluator.curve_min_seg_m == ON
    out["r3_slowest_ret"], out["r3_slowest_q"], out["r3_pdm_ret"] = slowest_and_pdm(S)
    return out


def best(r):
    return max(r["f0.9"], r["f1.0"])


def main():
    out_path = sys.argv[1]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 3
    t0 = time.time()
    W = {}
    ramguard.run_tasks(work, [(i, "grepro") for i in USED], workers, lambda r, t: W.__setitem__(r["idx"], r),
                       label_fn=lambda t: f"grepro state {t[0]}", log=lambda s: print(time.strftime("%H:%M:%S"), s, flush=True))
    stored = json.load(open(os.path.join(P5, "raw", "grepro_v5.json"), encoding="utf-8"))["states"]
    res = dict(wall_s=time.time() - t0, curve_min_seg_m_on=ON, curve_min_seg_m_off=None, floor=0.20, states={})
    ok = True
    q_all, rs, rp = [], [], []
    for i in USED:
        w, s5 = W[i], stored[str(i)]
        r1_equal = all(w["r1_run1"][k] == s5["rets_floor020"][k] and w["r1_run2"][k] == s5["rets_floor020"][k] for k in s5["rets_floor020"])
        r1_equal_arms = bool(w["r1_slowest_ret"] == s5["slowest_ret"] and w["r1_pdm_ret"] == s5["pdm_ret"])
        r2_equal = all(w["r2_rets"][k] == s5["rets_floor100"][k] for k in s5["rets_floor100"])
        c100 = w["r2_rets"]["crawl1.5"] / best(w["r2_rets"])
        r2_ok = abs(c100 - R2_EXPECT[i]) / R2_EXPECT[i] <= REL
        inf, kn = g.ladder_known(w["r2_rets"], w["r2_hard"])
        q_all += w["r3_slowest_q"]; rs.append(w["r3_slowest_ret"]); rp.append(w["r3_pdm_ret"])
        res["states"][i] = dict(limit=w["limit"], R1_ladder_bit_identical_both_runs=r1_equal, n_returns_compared=len(s5["rets_floor020"]),
                                R1_slowest_pdm_bit_identical=r1_equal_arms, R1_crawl_over_best=w["r1_run1"]["crawl1.5"] / best(w["r1_run1"]),
                                R2_crawl_over_best=c100, R2_expected=R2_EXPECT[i], R2_ok=bool(r2_ok), R2_bit_identical_to_stored=bool(r2_equal),
                                R2_known_informative=(inf, kn), R3_slowest_ret_switch_on=w["r3_slowest_ret"], R3_pdm_ret_switch_on=w["r3_pdm_ret"],
                                R3_slowest_mean_qfree_switch_on=float(np.mean(w["r3_slowest_q"])) if w["r3_slowest_q"] else None,
                                rets_switch_off_floor020=w["r1_run1"], rets_floor100=w["r2_rets"])
        ok &= bool(r1_equal and r1_equal_arms and r2_ok)
    res["R2_state3_not_known"] = bool(res["states"][3]["R2_known_informative"][0] and res["states"][3]["R2_known_informative"][1] is False)
    res["R3"] = dict(pooled_mean_qfree=float(np.mean(q_all)) if q_all else None, n_applicable=len(q_all), mean_slowest_ret=float(np.mean(rs)), mean_pdm_ret=float(np.mean(rp)))
    res["R3"]["known"] = bool(q_all and np.mean(q_all) < 0.6 and np.mean(rs) < np.mean(rp))
    ok &= res["R2_state3_not_known"] and res["R3"]["known"]
    res["R1"] = all(res["states"][i]["R1_ladder_bit_identical_both_runs"] and res["states"][i]["R1_slowest_pdm_bit_identical"] for i in USED)
    res["R2"] = all(res["states"][i]["R2_ok"] for i in USED) and res["R2_state3_not_known"]
    res["PASS"] = bool(ok)
    json.dump(res, open(out_path, "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps({k: res[k] for k in ("R1", "R2", "R3", "PASS")}, default=str))
    for i in USED:
        s = res["states"][i]
        print(i, {k: s[k] for k in ("R1_ladder_bit_identical_both_runs", "R1_slowest_pdm_bit_identical", "R1_crawl_over_best", "R2_crawl_over_best", "R2_known_informative",
                                    "R2_bit_identical_to_stored", "R3_slowest_mean_qfree_switch_on")})
    print("GREPRO_PASS" if ok else "GREPRO_FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
