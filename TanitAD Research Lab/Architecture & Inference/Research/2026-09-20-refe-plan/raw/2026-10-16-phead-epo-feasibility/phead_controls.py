#!/usr/bin/env python3
"""PHEAD identity controls (a1), (a2), (b), (c) -- PREREG_PHEAD.md section 7.  (a3) and the closed-loop halves of (b) and (c) come from the closed-loop queue (see phead_checks_closed.py)."""
import json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import phead_common as PC
from phead_open import run_open
from adam_sim import run as sim_run, stats as sim_stats, draws_for
import adam_common as AC
t0 = time.time()
SF, a, holder, cfg, tr, ev = PC.setup(int(os.environ.get("PH_THREADS", "4")))
th0 = PC.theta0_of(tr); names = tr.names; shapes = [tuple(p.shape) for p in th0]
bank = torch.load(os.path.join(PC.BANK_DIR, "bank_s0.pt"), weights_only=False)
draws = draws_for(0, 250, bank["gB"].shape[0])
full = PC.group_slices("FULL", names, shapes)
banked = json.load(open(os.path.join(PC.AD, "adam_sim.json")))["streams"]["0"]["modes"]["P"]["at"]["50"]
res = {"tool": "phead_controls/1"}

# ---- (a1) FULL through Restricted vs adam_sim.run, open loop, regime P, K=50
mine, ref = {}, {}
for lam in (0.0, 0.25, 1.0):
    Dm, _ = run_open(bank, th0, a, draws, "P", lam, 50, full, "R", 1.0, (25, 50))
    Dr, _ = sim_run(bank, th0, a, draws, "P", lam, 50, snaps=(25, 50))
    mine[lam] = {k: [x - y for x, y in zip(v, th0)] for k, v in Dm.items()}
    ref[lam] = Dr
mx = max(float((x - y).abs().max()) for lam in mine for k in (25, 50) for x, y in zip(mine[lam][k], ref[lam][k]))
st = sim_stats(mine[0.0][50], mine[0.25][50], mine[1.0][50], names)
a1 = {"max_abs_diff_vs_adam_sim_run": mx, "tolerance_abs": 1e-7, "pass_state": bool(mx <= 1e-7), "stats": {}}
for key in ("S_C80", "S_all", "cos_C80", "U_full_norm_ratio", "cos_full"):
    rel = abs(st[key] - banked[key]) / abs(banked[key])
    a1["stats"][key] = {"mine": st[key], "banked_adam_sim.json": banked[key], "rel_diff": rel, "pass": bool(rel <= 1e-6)}
a1["pass"] = bool(a1["pass_state"] and all(v["pass"] for v in a1["stats"].values()))
res["a1_open_loop_FULL_vs_banked"] = a1
print("a1", json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()}) for k, v in a1.items() if k != "stats"}), {k: v["rel_diff"] for k, v in a1["stats"].items()}, flush=True)

# ---- (a2) J(theta_0) on the 256 sets of groups[:32] vs late_probe.json base.mean_J
o0 = {k: v.evaluate(th0) for k, v in ev.items()}
ids = bank["groups"][:32].reshape(-1)
j256 = float(o0["train"]["J"][ids].mean())
lp = json.load(open(os.path.join(PC.AD, "late_probe.json")))["states"]["base"]["mean_J"]
res["a2_J_theta0_vs_late_probe"] = {"mine_J_256_sets": j256, "late_probe_base_mean_J": lp, "abs_diff": abs(j256 - lp), "tolerance_abs": 1e-5, "pass": bool(abs(j256 - lp) <= 1e-5)}
print("a2", res["a2_J_theta0_vs_late_probe"], flush=True)

# ---- (b) EMPTY group: Delta J exactly 0.0 (open loop; the closed-loop half is checked from the queue's *_b3 runs)
sl0 = PC.group_slices("EMPTY", names, shapes)
b = {}
for regime, K in (("P", 50), ("T", 50)):
    th_l = {}
    for lam in (0.0, 1.0):
        D, _ = run_open(bank, th0, a, draws, regime, lam, K, sl0, "R", 1.0, (K,))
        th_l[lam] = D[K]
    maxdisp = max(float((x - y).abs().max()) for lam in th_l for x, y in zip(th_l[lam], th0))
    ej = {pn: e.evaluate(th_l[1.0]) for pn, e in ev.items()}
    ej0 = {pn: e.evaluate(th_l[0.0]) for pn, e in ev.items()}
    dj = {pn: float((ej[pn]["J"] - ej0[pn]["J"]).mean()) for pn in ev}
    exact = {pn: bool(np.array_equal(ej[pn]["J"], o0[pn]["J"]) and np.array_equal(ej0[pn]["J"], o0[pn]["J"])) for pn in ev}
    b[regime] = {"max_abs_theta_minus_theta0": maxdisp, "delta_J": dj, "J_bitwise_equal_to_theta0": exact, "pass": bool(maxdisp == 0.0 and all(v == 0.0 for v in dj.values()) and all(exact.values()))}
# sensitivity: the failure a masked-gradient implementation would have (all params in AdamW, gradient x 0): weight decay alone moves them
Rm = PC.Restricted(th0, full, lr=a.lr)
for _ in range(50):
    Rm.step([torch.zeros_like(t) for t in th0], a.lr)
b["sensitivity_masked_gradient_with_weight_decay"] = {"max_abs_theta_minus_theta0_after_50_steps": max(float((x - y).abs().max()) for x, y in zip(Rm.theta(), th0)),
                                                       "note": "AdamW decays zero-gradient entries: a mask-the-gradient implementation would break (b); Restricted does not hand frozen entries to AdamW"}
b["pass"] = bool(b["P"]["pass"] and b["T"]["pass"] and b["sensitivity_masked_gradient_with_weight_decay"]["max_abs_theta_minus_theta0_after_50_steps"] > 0.0)
res["b_empty_group_open_loop"] = b
print("b", json.dumps(b)[:600], flush=True)

# ---- (c) mutation: delete the line that applies the restriction
slc = PC.group_slices("PH_C_HEAD", names, shapes)
c = {}
for tag, env in (("unmutated", None), ("mutated", "1")):
    if env:
        os.environ["PHEAD_MUTATE_NO_RESTRICT"] = env
    else:
        os.environ.pop("PHEAD_MUTATE_NO_RESTRICT", None)
    th = {}
    for lam in (0.0, 1.0):
        D, _ = run_open(bank, th0, a, draws, "P", lam, 50, slc, "R", 1.0, (50,))
        th[lam] = D[50]
    out_ = PC.outside_group_displacement(th[1.0], th0, slc)
    ejs = {pn: (e.evaluate(th[1.0])["J"] - e.evaluate(th[0.0])["J"]).mean() for pn, e in ev.items()}
    c[tag] = {"outside_group_max_abs_displacement": out_, "check_outside_eq_0.0": bool(out_ == 0.0), "delta_J_epo": {k: float(v) for k, v in ejs.items()}}
os.environ.pop("PHEAD_MUTATE_NO_RESTRICT", None)
c["restricted_result_changes_under_mutation"] = bool(any(abs(c["mutated"]["delta_J_epo"][k] - c["unmutated"]["delta_J_epo"][k]) > 1e-4 for k in c["mutated"]["delta_J_epo"]))
c["pass"] = bool(c["unmutated"]["check_outside_eq_0.0"] and not c["mutated"]["check_outside_eq_0.0"] and c["restricted_result_changes_under_mutation"])
res["c_mutation_open_loop"] = c
print("c", json.dumps(c), flush=True)
json.dump(res, open(os.path.join(HERE, "controls_open.json"), "w"), indent=1)
print("ZZCONTROLS_OPEN_OK", round(time.time() - t0), "s", {k: v.get("pass") for k, v in res.items() if isinstance(v, dict)}, flush=True)
