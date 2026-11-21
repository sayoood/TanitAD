import sys, time, os
sys.path.insert(0, "C:/Users/Admin/par18/unseen_s1b/code")
import s1b_lib as L
L.setup_env()
import psutil
import numpy as np
import unseen_token_rule as R
import navtrain_scenarios as NS
import navsim_pdm_targets as PT
PT._setup()
p = psutil.Process()
print("rss after setup MB", p.memory_info().rss/1e6, flush=True)
lg = "2021.06.07.12.42.11_veh-38_02445_02843"
db = f"{L.DB_TRAINVAL}/{lg}.db"
rec = R.read_log(db); toks = R.candidates(rec)["tokens"]
rng=np.random.default_rng(3); want=[toks[i] for i in sorted(rng.choice(len(toks), 60, replace=False))]
t0=time.perf_counter(); c0=time.process_time(); n=0
for sc in NS.build_scenarios_for_log(db, want, map_root=L.MAPS, history_rows=0, future_rows=100):
    L.score_token(PT, sc); n+=1
    if n%15==0: print(n, "rss MB %.0f" % (p.memory_info().rss/1e6), "wall/tok %.2f cpu/tok %.2f" % ((time.perf_counter()-t0)/n, (time.process_time()-c0)/n), flush=True)
