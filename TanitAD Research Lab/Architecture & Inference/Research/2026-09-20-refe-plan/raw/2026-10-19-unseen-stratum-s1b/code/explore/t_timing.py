import sys, time, json
sys.path.insert(0, "C:/Users/Admin/par18/unseen_s1b/code")
import s1b_lib as L
L.setup_env()
import numpy as np
import unseen_token_rule as R
import navtrain_scenarios as NS
import navsim_pdm_targets as PT
t0 = time.perf_counter(); PT._setup(); print("setup", round(time.perf_counter()-t0,2), flush=True)
lg = sys.argv[1]; n = int(sys.argv[2])
db = f"{L.DB_TRAINVAL}/{lg}.db"
rec = R.read_log(db); c = R.candidates(rec)
toks = c["tokens"]; print("cand", len(toks))
rng = np.random.default_rng(1); pick = sorted(rng.choice(len(toks), n, replace=False).tolist())
want = [toks[i] for i in pick]
t0 = time.perf_counter(); c0 = time.process_time()
scs = list(NS.build_scenarios_for_log(db, want, map_root=L.MAPS, history_rows=0, future_rows=100))
print("scenarios", len(scs), "build s", round(time.perf_counter()-t0,2), flush=True)
for sc in scs:
    t1 = time.perf_counter(); c1 = time.process_time()
    o = L.score_token(PT, sc)
    print(sc._initial_lidar_token[:8], "v=%.2f" % o["speed"], "cv", round(o["cv"]["pdms"],3), "hum", round(o["human"]["pdms"],3), "wall %.2f cpu %.2f cache %.2f score %.2f" % (time.perf_counter()-t1, time.process_time()-c1, o["t_cache"], o["t_score"]), flush=True)
print("total wall", round(time.perf_counter()-t0,1), "cpu", round(time.process_time()-c0,1))
