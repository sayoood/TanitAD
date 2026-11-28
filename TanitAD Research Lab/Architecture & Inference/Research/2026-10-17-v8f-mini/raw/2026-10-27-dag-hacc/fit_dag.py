"""fit_dag.py <seed 0|1>: PREREG_V8F_DAG_HACC section 1. v8f_hacc.fit_seed (Phase A, TRAIN arrays only) UNCHANGED, with only the plans changed (DAG's in place of M2b's).
DECLARATIONS D1: the reference weight of lambda0 ('HBLEND's banked w') = the HBLEND recipe (H.fit_weights) on DAG's own plans; the literal alternative (M2b's banked w) is a sensitivity beside."""
import hashlib, json, os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TANITEVAL_STACK_OVERRIDE", "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/stack")
sys.dont_write_bytecode = True
PK = "E:/Projects/TanitAD-artifacts/daghacc"
sys.path.insert(0, f"{PK}/hacc")
import numpy as np
import v8f_hacc as A
H = A.H
s = sys.argv[1]
seed = f"DAG_s{s}"
fp = f"{PK}/plans/fitplans_{seed}.npz"
fw = np.load(fp)
assert fw["gt_ok"].all() and len(fw["traj30"]) == A.N_TRAIN
tau, P, y = (fw[k].astype(np.float64) for k in ("traj30", "p30", "gt"))
ref = H.fit_weights(tau, P, y)                       # D1: HBLEND's recipe on DAG's plans
canon = hashlib.sha256(np.ascontiguousarray(ref["w"]).tobytes()).hexdigest()
A.TRAIN_SHA[seed] = {"fit": H.sha256_file(fp), "w": canon}
A.train_bundle = lambda sd, _b=A.TrainBundle(tau=tau, P=P, y=y, banked_w=ref["w"], split="train", source=fp): _b
rec = A.fit_seed(seed, f"{PK}/fit")
# the weights table in the format V8FMini.load_hacc_weights reads (HACC_KEYS)
w = np.asarray(rec["w_star"], float)
out = {"seed": seed, "arm": "DAG-HACC", "lambda0": rec["lambda0"], "m_star": rec["m_star"], "lam_star": rec["lam_star"], "rule_branch": rec["rule_branch"],
       "w_along_by_step": w[:, 0].tolist(), "w_cross_by_step": w[:, 1].tolist(), "tier": A.TIER, "fit_windows_sha256": H.sha256_file(fp),
       "note": "H = P + w (.) (tau - P) at inference; w is a fixed function of the step index (30 x 2 per model); NO isotonic step; tau/P/y = DAG's own T0 plans on HBLEND's W_fit (PREREG_V8F_DAG_HACC section 1)"}
json.dump(out, open(f"{PK}/fit/weights_HACC_{seed}.json", "w"), indent=1)
# D1 sensitivity, BESIDE: lambda0 with M2b's literal banked w
bw = json.load(open(f"{PK}/hb/weights_M2b_s{s}.json"))
bwm = np.stack([np.asarray(bw["w_along_by_step"], float), np.asarray(bw["w_cross_by_step"], float)], 1)
T = A.TrainBundle(tau=tau, P=P, y=y, banked_w=bwm, split="train", source=fp)
alt = A.select_lambda(T)
sens = {"seed": seed, "primary_lambda0": rec["lambda0"], "primary_lam_star": rec["lam_star"], "primary_m_star": rec["m_star"],
        "alt_M2b_banked_w_lambda0": alt["lambda0"], "alt_lam_star": alt["lam_star"], "alt_m_star": alt["m_star"], "alt_rule_branch": alt["rule_branch"],
        "max_abs_w_alt_vs_primary": float(np.abs(alt["w_star"] - w).max()), "note": "BESIDE only (DECLARATIONS D1); decides nothing"}
json.dump(sens, open(f"{PK}/fit/sensitivity_D1_{seed}.json", "w"), indent=1)
# C-H2 on THIS fit: the selection path refuses anything that is not a TrainBundle of split 'train'
refused = {}
for nm, bad in (("dict_world", {"tau": tau, "P": P, "y": y, "split": "eval"}), ("bare_array", tau)):
    try:
        A.select_lambda(bad)
        refused[nm] = False
    except (TypeError, ValueError, AttributeError):
        refused[nm] = True
try:
    A.TrainBundle(tau=tau, P=P, y=y, banked_w=ref["w"], split="eval")
    refused["eval_split_bundle"] = False
except ValueError:
    refused["eval_split_bundle"] = True
json.dump({"seed": seed, "refuses": refused, "all_refused": all(refused.values())}, open(f"{PK}/fit/CH2_refusal_{seed}.json", "w"), indent=1)
print("ZZFITDAG", seed, "lam_star", rec["lam_star"], "m*", rec["m_star"], "controls", json.dumps(rec["train_controls"]["pass"]), "refused", json.dumps(refused), flush=True)
