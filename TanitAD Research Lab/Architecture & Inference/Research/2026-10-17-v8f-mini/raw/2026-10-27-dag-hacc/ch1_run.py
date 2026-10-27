"""C-H1 driver: the HACC fit code (v8f_hacc.fit_seed, Phase A, TRAIN arrays only) run on M2b's banked fit inputs; compare w* / lambda* with HACC's banked ones."""
import json, os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TANITEVAL_STACK_OVERRIDE", "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/stack")
sys.dont_write_bytecode = True
PK = "E:/Projects/TanitAD-artifacts/daghacc"
sys.path.insert(0, f"{PK}/hacc")
import numpy as np
import v8f_hacc as A
seed = sys.argv[1]
t0 = time.time()
rec = A.fit_seed(seed, f"{PK}/ch1")
bank = json.load(open(f"{PK}/hacc_banked/weights_HACC_{seed}.json"))
bw = np.stack([np.asarray(bank["w_along_by_step"], float), np.asarray(bank["w_cross_by_step"], float)], 1)
bf = json.load(open(f"{PK}/hacc_banked/fit_{seed}.json"))
w = np.asarray(rec["w_star"], float)
out = {"seed": seed, "max_abs_dw_vs_banked_HACC_weights": float(np.abs(w - bw).max()), "lam_star": rec["lam_star"], "banked_lam_star": bf["lam_star"],
       "lam_star_equal": bool(rec["lam_star"] == bf["lam_star"]), "lambda0": rec["lambda0"], "banked_lambda0": bf["lambda0"], "m_star": rec["m_star"], "banked_m_star": bf["m_star"],
       "max_abs_dw_vs_banked_fit_json_w_star": float(np.abs(w - np.asarray(bf["w_star"], float)).max()),
       "train_controls_pass": rec["train_controls"]["pass"], "seconds": time.time() - t0}
out["C_H1_pass"] = bool(out["max_abs_dw_vs_banked_HACC_weights"] <= 1e-12 and out["lam_star_equal"] and rec["train_controls"]["all_pass"])
json.dump(out, open(f"{PK}/ch1/ch1_{seed}.json", "w"), indent=1)
print("ZZCH1", json.dumps(out), flush=True)
