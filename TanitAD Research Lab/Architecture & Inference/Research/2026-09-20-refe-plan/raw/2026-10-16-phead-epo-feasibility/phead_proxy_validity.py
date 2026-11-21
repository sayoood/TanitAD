#!/usr/bin/env python3
"""PHEAD control (d), PREREG_PHEAD.md s.5: does the sim's TTC proxy read the KNOWN R0 trade?  The proxy functions of phead_common.Evaluator are applied to the REAL banked scorer states
(rls1-{r0,r0b,c0}/scorer_states/scorer_states_u{u}.pt, A and B) on the cached sets (302 train + 62 val = 364), B - A x100, and set against the real RL-val B - A in ab_vs_base.json.
Pass: P2 (policy-expected TTC) B - A is negative at >= 2 of the 3 R0 checkpoints AND |P2 B - A| < 0.5 at every C0 (floor) checkpoint."""
import json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import phead_common as PC
t0 = time.time()
SF, a, holder, cfg, tr, ev = PC.setup(int(os.environ.get("PH_THREADS", "4")))
real = json.load(open(os.path.join(PC.AD, "ab_vs_base.json")))["rows"]
STATES = [("r0", 250), ("r0", 500), ("r0", 1000), ("r0b", 250), ("r0b", 500), ("c0", 250), ("c0", 500), ("c0", 1000)]
out = {"tool": "phead_proxy_validity/1", "states": {}}
logs_all = np.concatenate([ev["train"].logs, ev["val"].logs])
for run, u in STATES:
    ss = torch.load(os.path.join(AC_DL := PC.AC.DL, f"rls1-{run}", "scorer_states", f"scorer_states_u{u}.pt"), map_location="cpu", weights_only=False)
    ev_arm = {}
    for which in ("A", "B"):
        for m in SF.SCORER:
            getattr(tr.armB, m).load_state_dict(ss[which][m], strict=True)
        th = [p.detach().clone() for p in tr.armB.parameters()]
        ev_arm[which] = {pn: e.evaluate(th) for pn, e in ev.items()}
    rec = {}
    for nm, col in (("TTC", 3), ("NC", 0), ("DAC", 1), ("EP", 2), ("C", 4)):
        for proxy, key in (("P1_pick", "pick_comp"), ("P2_pi", "pi_comp")):
            d = np.concatenate([ev_arm["B"][pn][key][:, col] - ev_arm["A"][pn][key][:, col] for pn in ("train", "val")]) * 100
            bs = PC.boot_ratio(logs_all, d)
            rec[f"{proxy}_{nm}_B_minus_A_x100"] = {"mean": bs["mean_num"], "ci95": bs["ci_num"]}
    dj = np.concatenate([ev_arm["B"][pn]["J"] - ev_arm["A"][pn]["J"] for pn in ("train", "val")])
    rec["J_B_minus_A"] = {"mean": float(dj.mean()), "ci95": PC.boot_ratio(logs_all, dj)["ci_num"]}
    rec["real_RLval_B_minus_A_TTC_x100"] = real[f"{run}_u{u}"]["B_minus_A"]["TTC"]["mean"]
    rec["real_RLval_B_minus_A_NC_x100"] = real[f"{run}_u{u}"]["B_minus_A"]["NC"]["mean"]
    out["states"][f"{run}_u{u}"] = rec
    print(run, u, {k: round(v["mean"], 3) for k, v in rec.items() if isinstance(v, dict) and k.startswith(("P2_pi_TTC", "P1_pick_TTC", "J_"))}, "real TTC", rec["real_RLval_B_minus_A_TTC_x100"], f"{time.time()-t0:.0f}s", flush=True)
r0 = [out["states"][f"r0_u{u}"]["P2_pi_TTC_B_minus_A_x100"]["mean"] for u in (250, 500, 1000)]
c0 = [out["states"][f"c0_u{u}"]["P2_pi_TTC_B_minus_A_x100"]["mean"] for u in (250, 500, 1000)]
r0p1 = [out["states"][f"r0_u{u}"]["P1_pick_TTC_B_minus_A_x100"]["mean"] for u in (250, 500, 1000)]
out["verdict"] = {"P2_TTC_R0_B_minus_A": r0, "n_negative_of_3": int(sum(x < 0 for x in r0)), "P2_TTC_C0_floor": c0, "C0_all_abs_lt_0.5": bool(all(abs(x) < 0.5 for x in c0)),
                  "P1_TTC_R0_B_minus_A_reported": r0p1,
                  "proxy_validated": bool(sum(x < 0 for x in r0) >= 2 and all(abs(x) < 0.5 for x in c0))}
json.dump(out, open(os.path.join(HERE, "proxy_validity.json"), "w"), indent=1)
print("ZZPROXY_OK", out["verdict"], f"{time.time()-t0:.0f}s", flush=True)
