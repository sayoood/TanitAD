import sys, json, time
sys.path.insert(0, "C:/Users/Admin/par18/p6gaps/sandbox/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval")
import selection_quality as SQ
t=time.time()
r = SQ.read_table("C:/Users/Admin/p2full_work/final/table_p2_full.npz")
print(json.dumps({k:r[k] for k in ("n_tokens","n_logs","levels_x100","paired_differences_x100","n_pick_zero","n_best_zero","n_best_strictly_above_pick")}, indent=0)[:1800], round(time.time()-t,1))
