import os,sys,json,pickle,lzma,csv
os.environ.update({"REFE_SIM_HZ":"10","NUPLAN_MAPS_ROOT":"E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0","OMP_NUM_THREADS":"1","CUDA_VISIBLE_DEVICES":"-1"})
sys.path[:0]=["C:/Users/Admin/n1_run/w/pk/refe"]
import numpy as np
import navsim_pdm_targets as PT, navtrain_scenarios as NS
S=PT._setup()
from navsim.evaluate.pdm_score import get_trajectory_as_array
W={}
for l in open("n1t_only.jsonl"):
    d=json.loads(l); W[d["token"]]=d
H={r["token"]:r for r in csv.DictReader(open("C:/Users/Admin/n1_run/inputs/refe_p4_teacher_navtest.csv",encoding="utf-8")) if r["token"]!="average"}
sel=json.load(open("diag_tokens.json"))
dbs=NS.index_dbs(db_root="E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits")
def raw0(parts):
    pdm_traj,ego,obs,cl,lanes,dam=parts
    st=get_trajectory_as_array(pdm_traj,S["samp"],ego.time_point)
    sim=S["sim"].simulate_proposals(np.stack([st],0),ego)
    sc=S["scorer"]; sc.score_proposals(sim,obs,cl,lanes,dam)
    return float(sc._progress_raw[0]*sc._multi_metrics.prod(axis=0)[0])
rows=[]
for grp in ("epdiff","same","flip"):
    for tok in sel[grp]:
        log=W[tok]["log"]
        mc=pickle.loads(lzma.open(f"F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest/{log}/unknown/{tok}/metric_cache.pkl","rb").read())
        sc=next(NS.build_scenarios_for_log(dbs[log],[tok],history_rows=1,future_rows=80))
        a=raw0(PT.metric_cache_parts(sc)); b=raw0((mc.trajectory,mc.ego_state,mc.observation,mc.centerline,mc.route_lane_ids,mc.drivable_area_map))
        epw=W[tok]["tr"]["ep"]; eph=float(H[tok]["ego_progress"])
        r={"grp":grp,"token":tok,"raw0_wtransp":a,"raw0_cache":b,"raw0_ratio_cache_over_wtransp":b/a if a else None,
           "ep_wtransp":epw,"ep_harness":eph,"ep_ratio_w_over_h":(epw/eph if eph else None)}
        rows.append(r); print(json.dumps(r),flush=True)
json.dump(rows,open("diag_norm.json","w"),indent=1)
