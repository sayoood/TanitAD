"""Amendment 1 proof (CPU, no model): new frame control on all 5,877 tokens, old reference reproduces the 17, RED time-shift and RED rotation fail it."""
import os, sys, json, gzip, math, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
D = "E:/Projects/TanitAD-artifacts/refe_selection_confirm_20261107"
sys.path.insert(0, D)
import numpy as np
import selc_dump as S
_exp = None

def one_log(args):
    lg, toks = args
    global _exp
    import navtrain_scenarios as NS
    if _exp is None:
        _exp = json.load(gzip.open(S.navsim_root() + "/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz", "rt", encoding="utf-8"))["tokens"]
    mt = S.meta_times(lg)
    out = []
    for sc in NS.build_scenarios_for_log(os.path.join(S.DB_DIR, lg + ".db"), toks, history_rows=1, future_rows=80):
        tok = sc._initial_lidar_token
        ego = sc.get_ego_state_at_iteration(0)
        th = np.asarray(_exp[tok]["human_future_poses"], np.float64)
        a, nx = mt[tok]
        nx_shift = [x + 500000 for x in nx]                      # RED 1: one-step (0.5 s) time shift of the export's timestamps
        ang = 0.05; c, s = math.cos(ang), math.sin(ang)
        th_rot = th.copy(); th_rot[:, 0] = c * th[:, 0] - s * th[:, 1]; th_rot[:, 1] = s * th[:, 0] + c * th[:, 1]   # RED 2: rotated frame (0.05 rad)
        out.append({"token": tok, "log": lg, "new": S.frame_ctrl_meta(sc, ego, th, a, nx), "old": S.frame_ctrl_old(sc, ego, th),
                    "red_shift": S.frame_ctrl_meta(sc, ego, th, a, nx_shift), "red_rot": S.frame_ctrl_meta(sc, ego, th_rot, a, nx),
                    "moving": bool(np.abs(th[:, :2]).max() > 1.0)})
    return lg, out, sorted(set(toks) - {r["token"] for r in out})

if __name__ == "__main__":
    from multiprocessing import Pool
    doc = json.load(open(D + "/tokens_confirm.json", encoding="utf-8"))
    by = {}
    for t in doc["tokens"]:
        by.setdefault(doc["token_log"][t], []).append(t)
    t0 = time.time()
    with Pool(4) as p:
        res = p.map(one_log, [(lg, by[lg]) for lg in sorted(by)], chunksize=1)
    R = [r for _, o, _ in res for r in o]
    miss = sum(len(m) for _, _, m in res)
    ok = lambda k: np.array([r[k] <= S.FC_TOL_M for r in R])        # nan -> False (a FAIL)
    cen = json.load(open(D + "/fc_diag/fc_census.json"))
    c17 = sorted(r["token"] for r in cen["failing"])
    old_fail = sorted(r["token"] for r in R if not r["old"] <= S.FC_TOL_M)
    mv = np.array([r["moving"] for r in R])
    res_ = {"n_tokens": len(R), "n_missing_from_builder": miss, "wall_s": round(time.time() - t0),
            "new_pass": int(ok("new").sum()), "new_max_err_m": float(np.nanmax([r["new"] for r in R])), "new_n_nan": int(sum(math.isnan(r["new"]) for r in R)),
            "old_fail_n": len(old_fail), "old_fail_equals_census_17": old_fail == c17,
            "red_time_shift_fail_n": int((~ok("red_shift")).sum()), "red_time_shift_fail_share": float((~ok("red_shift")).mean()),
            "red_rotation_0p05rad_fail_n": int((~ok("red_rot")).sum()), "red_rotation_fail_share_of_moving": float((~ok("red_rot"))[mv].mean()), "n_moving": int(mv.sum()),
            "red_time_shift_any_fails": bool((~ok("red_shift")).any()), "red_rotation_any_fails": bool((~ok("red_rot")).any())}
    res_["GATE"] = bool(res_["new_pass"] == len(R) == 5877 and miss == 0 and res_["old_fail_equals_census_17"] and res_["red_time_shift_any_fails"] and res_["red_rotation_any_fails"])
    json.dump(res_, open(D + "/fc_diag/fc_fix_result.json", "w"), indent=1)
    print(json.dumps(res_, indent=1))
