#!/usr/bin/env python3
"""STEP 0 (before PREREG_PHEAD.md): the FULL arm only -- no restricted number is computed here.
  (i)  J(theta_0) on the first 256 train sets must equal late_probe.json['base'].mean_J (the H-ADAM tool's own number);
  (ii) the BANKED closed-loop full-arm states (adam_closed.py output, K=25/50, lam 0 and 1, regimes P and T) are scored with the Evaluator, giving the magnitude and noise of
       DeltaJ_full on the two pools, which the prereg needs to fix its primary pool and its bars.
"""
import json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import phead_common as PC
torch.set_num_threads(int(os.environ.get("PH_THREADS", "16")))
t0 = time.time()
SF, a, holder, cfg, tr, ev = PC.setup()
print("setup", round(time.time() - t0), "s; N train", ev["train"].N, "N val", ev["val"].N, "logs", len(set(ev["train"].logs)), len(set(ev["val"].logs)), flush=True)
th0 = PC.theta0_of(tr)
res = {"tool": "probe0/1", "N": {k: v.N for k, v in ev.items()}, "n_logs": {k: int(len(set(v.logs))) for k, v in ev.items()}}
t1 = time.time()
o0 = {k: v.evaluate(th0) for k, v in ev.items()}
print("eval theta0 both pools", round(time.time() - t1), "s", flush=True)
lp = json.load(open(os.path.join(PC.AD, "late_probe.json")))["states"]["base"]
res["theta0"] = {k: {"J_all": float(o["J"].mean()), "J_first256": float(o["J"][:256].mean())} for k, o in o0.items()}
res["late_probe_base_mean_J"] = lp["mean_J"]
res["identity_J_first256_vs_late_probe_abs_diff"] = abs(res["theta0"]["train"]["J_first256"] - lp["mean_J"])
print("J first256", res["theta0"]["train"]["J_first256"], "late_probe", lp["mean_J"], "diff", res["identity_J_first256_vs_late_probe_abs_diff"], flush=True)
cdir = f"{PC.SCR}/closed"
res["banked_closed_full"] = {}
for regime in ("P", "T"):
    st = {}
    for lam in ("0.0", "0.25", "1.0"):
        z = torch.load(f"{cdir}/closed_{regime}_{lam}.pt", weights_only=False)
        for k in (25, 50):
            th = [t0_ + d for t0_, d in zip(th0, z["D"][k])]
            st[(lam, k)] = {pn: e.evaluate(th) for pn, e in ev.items()}
    out = {}
    for k in (25, 50):
        for pn in ev:
            lg = ev[pn].logs
            for lam in ("0.25", "1.0"):
                d = st[(lam, k)][pn]["J"] - st[("0.0", k)][pn]["J"]
                b = PC.boot_ratio(lg, d)
                out[f"{pn}_K{k}_lam{lam}_vs_lam0"] = {"dJ_epo": b["mean_num"], "ci": b["ci_num"],
                    "J_lam": float(st[(lam, k)][pn]["J"].mean()), "J_lam0": float(st[("0.0", k)][pn]["J"].mean()), "J_theta0": float(o0[pn]["J"].mean()),
                    "pick_ttc_x100_lam_minus_lam0": 100 * float((st[(lam, k)][pn]["pick_comp"][:, 3] - st[("0.0", k)][pn]["pick_comp"][:, 3]).mean()),
                    "pi_ttc_x100_lam_minus_lam0": 100 * float((st[(lam, k)][pn]["pi_comp"][:, 3] - st[("0.0", k)][pn]["pi_comp"][:, 3]).mean()),
                    "kl_to_base_lam": float(st[(lam, k)][pn]["kl"].mean()), "kl_to_base_lam0": float(st[("0.0", k)][pn]["kl"].mean())}
    res["banked_closed_full"][regime] = out
    print(regime, json.dumps({k: (round(v["dJ_epo"], 5), [round(x, 5) for x in v["ci"]], round(v["pick_ttc_x100_lam_minus_lam0"], 3)) for k, v in out.items()}), flush=True)
json.dump(res, open(os.path.join(HERE, "probe0_fullarm_baseline.json"), "w"), indent=1)
print("ZZPROBE0_OK", round(time.time() - t0), "s", flush=True)
