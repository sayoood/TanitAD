"""RULETRAIN checks t2 (the relabelled target equals distance_rule by hand) and t3 (reach), CPU, tiny model, synthetic world with the synthetic lead track.
Usage: t2_t3.py <tree_root> <out json>   (PYTHONPATH = tree stack;taniteval;stack/scripts)

t2  batch of 64 logged windows from the trainer's own build_data. For EVERY row the test recomputes, from the raw arrays (join lead_valid / lead_gap, cache speed row v[t], p10f[t][0], gt[t]) and NOT from the batch tensors,
    the eligibility (presence 1, 0 <= gap <= 60, finite closing label, gt_ok), v_L = max(0, v + closing), and calls dist_rule.distance_rule with batch size 1 (floor_lifts False, g0 = gap). Required: every relabelled row equals
    the by-hand output EXACTLY (torch.equal), every other row equals the batch's gt EXACTLY, the set of relabelled rows equals the stage's mask, and the count equals info["n_relabel"].
t3  reach, on the full tiny V8FMini loss: batch A (rows with a label lead, rule changes some) -> `ade` and the gradient of the total differ between logged and relabelled target; batch B (rows WITHOUT a label lead) -> relabelled
    target is the logged one (torch.equal) and ade, total and gradient are EXACTLY equal; control: the same batch twice differs by exactly 0.0.
"""
import json
import os
import sys

import numpy as np
import torch

tree, outp = sys.argv[1], sys.argv[2]
torch.set_num_threads(2)
import tanitad  # noqa: E402

assert os.path.abspath(tanitad.__file__).lower().startswith(os.path.abspath(tree).lower()), tanitad.__file__
sys.path.insert(0, os.path.join(tree, "stack", "scripts"))
import v8f_mini_train as TR  # noqa: E402
from tanitad.data import dist_rule as R  # noqa: E402
from tanitad.data import rule_relabel as RRL  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402

assert os.path.abspath(TR.__file__).lower().startswith(os.path.abspath(tree).lower()), TR.__file__
res = {"tanitad_file": tanitad.__file__, "rule_file": R.__file__}
OUT = "C:/Users/Admin/v8proof_c/ruletrain/tmp/t2t3"
os.makedirs(OUT, exist_ok=True)
argv = ["--synthetic", "--batch", "64", "--d", "32", "--n-q", "2", "--threads", "1", "--seed", "0", "--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", "--wm-state", "grid", "--wm-cell-dim", "16",
        "--rule-relabel", "on", "--exclude", "1:3", "--out", OUT]
a = TR.build_parser().parse_args(argv)
cfg = TR.make_config(a)
torch.manual_seed(0)
np.random.seed(0)
store, J, wins, clips, info = TR.build_data(a, cfg)
ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
assert not ds.lead_sup, "the dataset must NOT carry lead_sup labels: the relabel reads the join itself"
RR = RRL.RuleRelabel("on", J, store)
LSA = D.lead_sup_label_arrays(J)
wl = list(wins)
res["n_windows"] = len(wl)


def eligible_by_hand(ci, t):
    """independent of the batch tensors and of RR.label_rows: raw join rows + store."""
    row = int(J["off"][ci]) + int(t)
    lv, g = J["lead_valid"][row] == 1, float(J["lead_gap"][row])
    d = store.get(str(J["clip"][ci]))
    clos = float(LSA["clos"][row])                                   # the closing-label rule itself is pinned against a literal re-implementation by the LEADSUP check t3 l1
    ok = bool(lv and np.isfinite(g) and 0.0 <= g <= 60.0 and np.isfinite(clos) and d["gt_ok"][t])
    return ok, g, clos, d


# ---------------- t2
batch = D.collate([ds.item(*w) for w in wl[:64]])
gt_new, inf = RR.stage(batch)
mism, n_el, n_changed_hand, kept_bad = [], 0, 0, []
mask_hand = np.zeros(len(wl[:64]), bool)
for b in range(len(wl[:64])):
    ci, t = int(batch["ci"][b]), int(batch["t"][b])
    ok, g, clos, d = eligible_by_hand(ci, t)
    if ok:
        v_log, v0 = float(d["v"][t]), float(d["p10f"][t][0])
        ok = ok and np.isfinite(v_log) and np.isfinite(v0) and bool(batch["frame_ok"][b, -1])
    mask_hand[b] = ok
    if ok:
        n_el += 1
        out = R.distance_rule(d["gt"][t][None].astype(np.float32), np.array([v0]), np.array([True]), np.array([g]), np.array([g]), np.array([max(0.0, v_log + clos)]), m=10, floor_lifts=False)
        n_changed_hand += int(out.changed[0])
        if not torch.equal(gt_new[b], torch.as_tensor(out.plan[0])):
            mism.append(b)
    else:
        if not torch.equal(gt_new[b], batch["gt"][b]):
            kept_bad.append(b)
res["t2"] = {"n_rows": len(wl[:64]), "n_eligible_by_hand": n_el, "n_stage_relabel": inf["n_relabel"], "n_changed_by_hand": n_changed_hand, "n_changed_stage": inf["n_changed"], "mask_equal": bool((mask_hand == inf["mask"]).all()),
             "relabelled_rows_not_equal_by_hand": mism, "unlabelled_rows_not_kept": kept_bad,
             "dtype": str(gt_new.dtype), "max_abs_change_m": float((gt_new - batch["gt"]).abs().max())}
res["t2"]["PASS"] = bool(not mism and not kept_bad and res["t2"]["mask_equal"] and n_el == inf["n_relabel"] and n_el > 0 and 0 < n_changed_hand == inf["n_changed"] and n_el < 64)

# ---------------- t2b: a row with a label lead and a NON-FINITE logged speed (v[t], then p10f[t][0]) keeps its target exactly and is not counted
rows_ok = [b for b in range(len(wl[:64])) if mask_hand[b]]
nan_res = {}
for what in ("v", "p10f"):
    b = rows_ok[0]
    ci, t = int(batch["ci"][b]), int(batch["t"][b])
    d = store.get(str(J["clip"][ci]))
    saved = d[what][t].copy() if what == "p10f" else float(d[what][t])
    d[what][t] = np.nan
    g2, i2 = RR.stage(batch)
    d[what][t] = saved
    keep_others = torch.equal(torch.cat([g2[:b], g2[b + 1:]]), torch.cat([gt_new[:b], gt_new[b + 1:]]))
    nan_res[what] = {"row": b, "target_kept_exactly": bool(torch.equal(g2[b], batch["gt"][b])), "in_mask": bool(i2["mask"][b]), "n_relabel": i2["n_relabel"], "n_relabel_before": inf["n_relabel"], "other_rows_equal_to_unmasked_run": keep_others}
g3, i3 = RR.stage(batch)
nan_res["restored_equal_to_first_run"] = bool(torch.equal(g3, gt_new) and i3["n_relabel"] == inf["n_relabel"])
res["t2b_nan_speed"] = nan_res
res["t2b_nan_speed"]["PASS"] = bool(all(nan_res[k]["target_kept_exactly"] and not nan_res[k]["in_mask"] and nan_res[k]["n_relabel"] == inf["n_relabel"] - 1 and nan_res[k]["other_rows_equal_to_unmasked_run"] for k in ("v", "p10f")) and nan_res["restored_equal_to_first_run"])

# ---------------- t3: reach on the full tiny model
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)
torch.manual_seed(0)
model = M.V8FMini(cfg, M.P10Prior.synthetic(0), torch.zeros(M.N_CONS), torch.ones(M.N_CONS))
model.train()
named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def run(b):
    for _, p in named:
        p.grad = None
    torch.manual_seed(20261115)
    T = model.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(11))["terms"]
    T["total"].backward()
    return float(T["ade"].detach()), float(T["total"].detach()), {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for n, p in named}


gd = lambda x, y: float(sum((x[n] - y[n]).abs().sum() for n in x))      # noqa: E731
row_ok = lambda w: eligible_by_hand(int(w[0]), int(w[1]))[0]            # noqa: E731
with_lead = [w for w in wl if row_ok(w)][:16]
without = [w for w in wl if not (int(J["lead_valid"][int(J["off"][int(w[0])]) + int(w[1])] == 1))][:16]
res["t3"] = {"n_with": len(with_lead), "n_without": len(without)}
bA = D.collate([ds.item(*w) for w in with_lead])
gA, iA = RR.stage(bA)
a0, t0, g0 = run(dict(bA))
a1, t1, g1 = run({**bA, "gt": gA})
a1b, t1b, g1b = run({**bA, "gt": gA})
res["t3"]["A"] = {"n_relabel": iA["n_relabel"], "n_changed": iA["n_changed"], "ade_logged": a0, "ade_relabelled": a1, "ade_diff": abs(a1 - a0), "grad_diff": gd(g0, g1), "same_batch_twice_ade_diff": abs(a1 - a1b), "same_batch_twice_grad_diff": gd(g1, g1b)}
bB = D.collate([ds.item(*w) for w in without])
gB, iB = RR.stage(bB)
b0, tb0, gb0 = run(dict(bB))
b1, tb1, gb1 = run({**bB, "gt": gB})
res["t3"]["B"] = {"n_relabel": iB["n_relabel"], "target_equal": bool(torch.equal(gB, bB["gt"])), "ade_diff": abs(b1 - b0), "total_diff": abs(tb1 - tb0), "grad_diff": gd(gb0, gb1)}
A, B = res["t3"]["A"], res["t3"]["B"]
res["t3"]["PASS"] = bool(A["n_changed"] > 0 and A["ade_diff"] > 0 and A["grad_diff"] > 0 and A["same_batch_twice_ade_diff"] == 0.0 and A["same_batch_twice_grad_diff"] == 0.0
                         and B["n_relabel"] == 0 and B["target_equal"] and B["ade_diff"] == 0.0 and B["total_diff"] == 0.0 and B["grad_diff"] == 0.0 and len(without) > 0)
res["PASS"] = bool(res["t2"]["PASS"] and res["t2b_nan_speed"]["PASS"] and res["t3"]["PASS"])
json.dump(res, open(outp, "w", encoding="utf-8", newline="\n"), indent=1)
print(json.dumps(res["t2"]))
print(json.dumps(res["t2b_nan_speed"]))
print(json.dumps(res["t3"]))
print("T2T3", "PASS" if res["PASS"] else "FAIL")
sys.exit(0 if res["PASS"] else 1)
