"""No-model frame-control census (CPU). Reuses selc_dump's env, NS.build_scenarios_for_log and the exact check at selc_dump.py:236-244."""
import os, sys, json, math, gzip, pickle, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
D = "E:/Projects/TanitAD-artifacts/refe_selection_confirm_20261107"
sys.path.insert(0, D)
import numpy as np
import selc_dump as S   # sets env + sys.path; imports no torch at module level
META = "F:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test/"
_exp = None

def one_log(args):
    lg, toks = args
    global _exp
    import navtrain_scenarios as NS
    if _exp is None:
        _exp = json.load(gzip.open(S.navsim_root() + "/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz", "rt", encoding="utf-8"))["tokens"]
    meta = pickle.load(open(META + lg + ".pkl", "rb"))
    mt = {f["token"]: i for i, f in enumerate(meta)}
    mts = np.array([f["timestamp"] for f in meta], np.int64)
    out, got = [], set()
    db = os.path.join(S.DB_DIR, lg + ".db")
    for sc in NS.build_scenarios_for_log(db, toks, history_rows=1, future_rows=80):
        tok = sc._initial_lidar_token; got.add(tok)
        ego = sc.get_ego_state_at_iteration(0)
        theirs = np.asarray(_exp[tok]["human_future_poses"], np.float64)
        px, py, pyaw = ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading
        c, s = math.cos(-pyaw), math.sin(-pyaw)
        f8 = list(sc.get_ego_future_trajectory(0, 4.0, 8))[:8]
        r = {"token": tok, "log": lg, "n_f8": len(f8)}
        t0 = int(ego.time_point.time_us)
        if len(f8) == 8:
            loc = np.array([[(f.rear_axle.x - px) * c - (f.rear_axle.y - py) * s, (f.rear_axle.x - px) * s + (f.rear_axle.y - py) * c] for f in f8])
            e = np.linalg.norm(loc - theirs[:, :2], axis=1)
            r["step_err_m"] = e.tolist(); r["max_err_m"] = float(e.max())
            r["db_dt_s"] = [(int(f.time_point.time_us) - t0) / 1e6 for f in f8]
        else:
            r["max_err_m"] = float("nan")
        f9 = list(sc.get_ego_future_trajectory(0, 4.5, 9))[:9]
        r["n_f9"] = len(f9)
        i = mt.get(tok)
        if i is not None:
            nxt = mts[i + 1:i + 9]
            r["meta_dt_s"] = ((nxt - mts[i]) / 1e6).tolist() if len(nxt) == 8 else None
            r["meta_ts_match_db"] = bool(mts[i] == t0)
            if len(nxt) == 8 and len(f9) == 9:
                # explain: pair export step k with the DB state at the 0.5 s slot nearest to the META timestamp of that step
                loc9 = np.array([[(f.rear_axle.x - px) * c - (f.rear_axle.y - py) * s, (f.rear_axle.x - px) * s + (f.rear_axle.y - py) * c] for f in f9])
                slot = [int(round(d / 0.5)) - 1 for d in r["meta_dt_s"]]
                if all(0 <= q < 9 for q in slot):
                    r["slots"] = slot
                    r["explained_err_m"] = float(np.linalg.norm(loc9[slot] - theirs[:, :2], axis=1).max())
        else:
            r["meta_dt_s"] = None
        out.append(r)
    miss = set(toks) - got
    return lg, out, sorted(miss)

if __name__ == "__main__":
    from multiprocessing import Pool
    doc = json.load(open(D + "/tokens_confirm.json", encoding="utf-8"))
    by = {}
    for t in doc["tokens"]:
        by.setdefault(doc["token_log"][t], []).append(t)
    sel = sys.argv[1]
    if sel == "fail1":
        lgs = ["2021.06.03.12.02.06_veh-35_00233_00609"]
    elif sel == "pending":
        lgs = [lg for lg in sorted(by) if not os.path.exists(S.chunk_path(D + "/dump", lg))]
    else:
        lgs = sorted(by)
    t0 = time.time()
    jobs = [(lg, by[lg]) for lg in lgs]
    with Pool(4) as p:
        res = p.map(one_log, jobs, chunksize=1)
    recs = [r for _, o, _ in res for r in o]
    miss = {lg: m for lg, _, m in res if m}
    json.dump({"logs": lgs, "wall_s": time.time() - t0, "recs": recs, "missing": miss}, open(D + f"/fc_diag/raw_{sel}.json", "w"))
    bad = [r for r in recs if not (r["max_err_m"] <= 1e-3)]
    print(sel, "logs", len(lgs), "tokens", len(recs), "fail", len(bad), "wall", round(time.time() - t0), "missing", sum(len(m) for m in miss.values()))
