import sys, time, cProfile, pstats, io
sys.path.insert(0, "C:/Users/Admin/par18/unseen_s1b/code")
import s1b_lib as L
L.setup_env()
import numpy as np
import unseen_token_rule as R
import navtrain_scenarios as NS
import navsim_pdm_targets as PT
PT._setup()
lg = "2021.06.07.11.59.52_veh-35_02283_02464"
db = f"{L.DB_TRAINVAL}/{lg}.db"
rec = R.read_log(db); toks = R.candidates(rec)["tokens"]
want = toks[100:112]
scs = list(NS.build_scenarios_for_log(db, want, map_root=L.MAPS, history_rows=0, future_rows=100))
L.score_token(PT, scs[0])
S = PT._setup(); proc = S["proc"]
pr = cProfile.Profile(); pr.enable()
for sc in scs[1:]:
    L.score_token(PT, sc)
pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(28); print(s.getvalue()[:6500])
# split: planner vs observation
t=[0,0]
for sc in scs[1:]:
    t0=time.perf_counter(); pi, pin = proc._get_planner_inputs(sc); proc._pdm_closed.initialize(pin); tr = proc._pdm_closed.compute_planner_trajectory(pi); t1=time.perf_counter()
    ob = proc._interpolate_gt_observation(sc); t2=time.perf_counter(); t[0]+=t1-t0; t[1]+=t2-t1
print("per token: pdm_closed %.3f obs %.3f" % (t[0]/len(scs[1:]), t[1]/len(scs[1:])))
