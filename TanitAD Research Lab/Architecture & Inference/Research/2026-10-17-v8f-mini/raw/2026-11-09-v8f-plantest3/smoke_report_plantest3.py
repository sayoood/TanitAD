"""Smoke test of report_plantest3.py on SYNTHETIC plan arrays (built from the logged actions plus noise; not results). Writes under F:/TanitAD-scratch/pt3/rep (deleted by the caller)."""
import json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import plantest3_real as R3
import report_plantest3 as REP
import run_plantest3 as RUN

OUT = "F:/TanitAD-scratch/pt3/rep"
runs = f"{OUT}/runs"
os.makedirs(runs, exist_ok=True)
rng = np.random.default_rng(0)
ctx = {t: R3.prepare3(t) for t in RUN.TAGS}
sc = ctx["M2b_s0"].score_idx
choice, tab = {}, {}
for t in RUN.TAGS:
    for a in RUN.ARMS:
        for f in RUN.FORMS:
            tab[f"{t}|{a}|{f}"] = {RUN.lam_s(l): dict(acc_mae=1.0 + 0.1 * i, yaw_mae=0.1 + 0.01 * (2 - i)) for i, l in enumerate(RUN.LAMS)}
            for fam in RUN.FAMS:
                choice[f"{t}|{a}|{f}|{fam}"] = 0.0 if fam == "LONGITUDINAL" else 1.0
td3 = {f"{t}|{fam}": list(RUN.td3_pick(choice, tab, t, fam)) for t in RUN.TAGS for fam in RUN.FAMS}
json.dump(dict(choice=choice, tune_table=tab, td3_pick=td3, split_salt="plantest2-v1", lambdas=list(RUN.LAMS)), open(f"{OUT}/tune_choice.json", "w"))
B1, B2, B3 = RUN.jobs_b(choice, tab)
for job in B1 + B2 + B3:
    _, t, kind, arm, form, goal, lam, ps = job
    L = ctx[t].A[sc][:, :12]
    noise = 0.05 if goal == "true" else 1.0
    plan = L + noise * rng.normal(size=L.shape) + (0.02 if ps else 0.0)
    r = dict(plan_raw=plan, idx=sc, mono1=rng.uniform(-1, 1, len(sc)), dmono1=rng.uniform(-1, 1, len(sc)), cost1=rng.uniform(0, 1, len(sc)), init1=rng.uniform(0.5, 1.5, len(sc)))
    np.savez(f"{runs}/score_{t}_{kind}_{arm}_{form}_{goal}_L{lam:g}_p{ps}.npz", **r)
for t in RUN.TAGS:
    L = ctx[t].A[sc][:, :12]
    for g, nz in (("true", 0.3), ("rand", 1.5)):
        np.savez(f"{runs}/kin_{t}_{g}.npz", plan_raw=L + nz * rng.normal(size=L.shape), idx=sc)
    json.dump({"tag": t, "DECODER_WEAK_this_seed": False, **{k: dict(r2_speed=0.9, r2_yaw=0.6, n_speed=100, n_yaw=120) for k in ("enc_L1", "enc_L1_k12", "rollout_L1_k12_logged_actions", "enc_L2")}}, open(f"{runs}/pre_{t}.json", "w"))
out = REP.main(runs, f"{OUT}/tune_choice.json", f"{OUT}/RESULT_SMOKE", 0)
print(json.dumps(out["labels"], indent=1))
print(out["triggers"], out["kin_beats_hold"], out["hier_adds"]["LATERAL"]["HIER_ADDS"])
