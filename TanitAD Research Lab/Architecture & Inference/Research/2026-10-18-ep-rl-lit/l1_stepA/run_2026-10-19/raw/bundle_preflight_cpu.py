# CPU-only, model-free preflight of make_stepA_bundle.py's imports and data-side keys (no checkpoint load, no forward). Evidence class: MEASURED (this run).
import json, os, sys, time
import numpy as np
HERE = sys.argv[1]; W = sys.argv[2]
sys.path.insert(0, HERE)
t0 = time.time()
import make_stepA_bundle as M      # runs the sys.path inserts for TREE/refe, TREE/eval
import stepA as A
print("require_lock_for_gpu:", callable(getattr(A, "require_lock_for_gpu", None)))
import torch; print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
import ctx_cache as CC, navtrain_scenarios as NS, planner as PLM
from planner import REFePlanner
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
print("imports ok", round(time.time() - t0, 1), "s")
args = dict(plan="C:/Users/Admin/p2heldout/plan/heldout_tokens.json", table="D:/Projects/TanitAD-artifacts/p2_heldout_table_full_2026-10-18/table_p2_heldout.npz",
            p20="D:/Projects/TanitAD-artifacts/p2_heldout_table_full_2026-10-18/proposals.npz")
plan = json.load(open(args["plan"], encoding="utf-8")); sample = json.load(open(W + "/sample.json", encoding="utf-8"))["tokens"]; toks = sorted(sample)
print("plan keys", sorted(plan.keys()))
bank = M.banked_rows(args["table"], toks); print("banked_rows ok", {k: v.shape for k, v in bank.items()})
p20 = np.load(args["p20"], allow_pickle=True); p20i = {str(k): i for i, k in enumerate(p20["token"])}
print("tokens in proposals20:", sum(t in p20i for t in toks), "/", len(toks))
miss = [t for t in toks if t not in plan["token_log"] or t not in plan["token_root"]]; print("tokens missing token_log/token_root:", len(miss))
logs = sorted({plan["token_log"][t] for t in toks}); print("logs", len(logs), "db paths exist:", sum(os.path.exists(plan["db"][l]) for l in logs), "/", len(logs))
roots = sorted({plan["token_root"][t] for t in toks}); print("frame roots", roots, [os.path.isdir(r) for r in roots])
print("L1_TREE refe dir exists:", os.path.isdir(M.TREE + "/refe"))
