"""RULEOWNL1 checks o2 (cap and term by hand) and o3 (reach, exact zeros, finite gradient on the cap), form l1. CPU, tiny model, synthetic world with the synthetic lead track.
Usage: o2_o3_l1.py <tree_root> <out json> [form]   (PYTHONPATH = tree stack;taniteval;stack/scripts). form defaults to l1; the mutation script runs it on a mutated tree and expects PASS false.

o2  batch of 64 logged windows from the trainer's own build_data. Mask and rule inputs recomputed from the RAW arrays.
    (a) the cap that the term uses equals dist_rule.distance_rule called by hand on every masked row (exact), on the model's own prediction and on a planted fast plan
    (b) the l1 per-sample term equals a float64 hand computation (mean over the 30 knots of the Euclidean distance in metres, from the hand-called rule); the batch term is the mean over the masked rows; the sq term of the same plan is also printed
    (c) rows without a label lead and rows with a non-finite raw speed contribute exactly 0 and get an exactly zero gradient
o3  (1) a plan that runs into the lead: term > 0, finite non-zero gradient on the trajectory output; (2) a plan inside the rule: term and gradient exactly 0; (3) a batch with no lead: exactly 0;
    (4) a mixed batch is partial, both counts printed; the gradient reaches the model parameters; (5) a constructed plan with knots exactly on the cap: finite gradient, exact zeros on those knots
"""
import json
import os
import sys

import numpy as np
import torch

tree, outp = sys.argv[1], sys.argv[2]
FORM = sys.argv[3] if len(sys.argv) > 3 else "l1"
torch.set_num_threads(2)
import tanitad  # noqa: E402

assert os.path.abspath(tanitad.__file__).lower().startswith(os.path.abspath(tree).lower()), tanitad.__file__
sys.path.insert(0, os.path.join(tree, "stack", "scripts"))
import v8f_mini_train as TR  # noqa: E402
from tanitad.data import dist_rule as R  # noqa: E402
from tanitad.data import rule_own as ROW  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402

assert os.path.abspath(TR.__file__).lower().startswith(os.path.abspath(tree).lower()), TR.__file__
assert os.path.abspath(ROW.__file__).lower().startswith(os.path.abspath(tree).lower()), ROW.__file__
res = {"tanitad_file": tanitad.__file__, "rule_own_file": ROW.__file__, "form": FORM}
OUT = "F:/TanitAD-scratch/v8proof/ruletrain3/work/tmp/o2o3_" + os.path.basename(os.path.abspath(tree))
os.makedirs(OUT, exist_ok=True)
argv = ["--synthetic", "--batch", "64", "--d", "32", "--n-q", "2", "--threads", "1", "--seed", "0", "--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", "--wm-state", "grid", "--wm-cell-dim", "16",
        "--w-rule-own", "1.0", "--rule-own-form", "l1", "--exclude", "1:3", "--out", OUT]
a = TR.build_parser().parse_args(argv)
cfg = TR.make_config(a)
torch.manual_seed(0)
np.random.seed(0)
store, J, wins, clips, info = TR.build_data(a, cfg)
ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
RO = ROW.RuleOwn(a.w_rule_own, a.rule_relabel_weight, J, store, form="l1")
LSA = D.lead_sup_label_arrays(J)
wl = list(wins)
torch.manual_seed(0)
model = M.V8FMini(cfg, M.P10Prior.synthetic(0), torch.zeros(M.N_CONS), torch.ones(M.N_CONS))
model.train()
res["built_form"] = RO.form
res["info_rule_own_form"] = (info.get("rule_own") or {}).get("form")


def eligible_by_hand(ci, t):
    row = int(J["off"][ci]) + int(t)
    lv, g = J["lead_valid"][row] == 1, float(J["lead_gap"][row])
    d = store.get(str(J["clip"][ci]))
    clos = float(LSA["clos"][row])
    ok = bool(lv and np.isfinite(g) and 0.0 <= g <= 60.0 and np.isfinite(clos) and d["gt_ok"][t])
    return ok, g, clos, d


batch = D.collate([ds.item(*w) for w in wl[:64]])
n = len(wl[:64])
inp = RO.stage(batch)
out = model.loss(dict(batch), p_gt=0.5, gen=torch.Generator().manual_seed(11))
pred = out["traj30"].detach().clone()
mask_hand = np.zeros(n, bool)
rows = {}
for b in range(n):
    ci, t = int(batch["ci"][b]), int(batch["t"][b])
    ok, g, clos, d = eligible_by_hand(ci, t)
    if ok:
        v_log, v0 = float(d["v"][t]), float(d["p10f"][t][0])
        ok = ok and np.isfinite(v_log) and np.isfinite(v0) and bool(batch["frame_ok"][b, -1])
    mask_hand[b] = ok
    if ok:
        rows[b] = (g, clos, v0, v_log)
res["o2"] = {"n_rows": n, "n_masked": int(inp["mask"].sum()), "n_masked_hand": len(rows), "mask_equal": bool((mask_hand == inp["mask"]).all())}


def hand_cap(plan_np, b):
    g, clos, v0, v_log = rows[b]
    o = R.distance_rule(plan_np[b:b + 1].astype(np.float64), np.array([v0]), np.array([True]), np.array([g]), np.array([g]), np.array([max(0.0, v_log + clos)]), m=10, floor_lifts=False)
    return o.plan[0], bool(o.changed[0])


def cap_check(plan_t, name):
    sel = np.nonzero(inp["mask"])[0]
    caps = torch.as_tensor(ROW.cap_of_plan(plan_t[sel].numpy().astype(np.float32), inp["v0"][sel], inp["gap"][sel], inp["v_L"][sel])).float()
    bad, n_changed = [], 0
    for k, b in enumerate(sel):
        cp, ch = hand_cap(plan_t.numpy(), int(b))
        n_changed += int(ch)
        if not torch.equal(caps[k], torch.as_tensor(cp).float()):
            bad.append(int(b))
    return {"name": name, "n": int(len(sel)), "n_changed_by_hand": n_changed, "rows_not_equal": bad}


planted = pred.clone()
planted[:, :, 0] = torch.arange(1, 31, dtype=torch.float32)[None, :] * 0.2 * 12.0           # 12 m/s straight on every row
planted[:, :, 1] = 0.0
res["o2"]["a"] = [cap_check(pred, "own_prediction"), cap_check(planted, "planted_fast_plan")]
# ---------------- o2 (b): the term by hand
sel = np.nonzero(inp["mask"])[0]
per_hand = []
for b in sel:
    cp, _ = hand_cap(planted.numpy(), int(b))
    pl = planted[int(b)].numpy().astype(np.float32).astype(np.float64)
    cap32 = torch.as_tensor(cp).float().double().numpy()                              # the term casts the cap to the prediction dtype
    per_hand.append(float(np.sqrt(((pl - cap32) ** 2).sum(-1)).mean()))
t_l1, i_l1 = ROW.own_term(planted.clone(), inp, FORM)
t_sq, i_sq = ROW.own_term(planted.clone(), inp, "sq")
hand_batch = float(np.mean(per_hand))
res["o2"]["b"] = {"form": FORM, "term_trainer_function": float(t_l1), "term_by_hand": hand_batch, "abs_diff": abs(float(t_l1) - hand_batch), "rel_diff": abs(float(t_l1) - hand_batch) / max(hand_batch, 1e-12),
                  "per_sample_hand_min_max": [min(per_hand), max(per_hand)], "n_active": i_l1["n_active"], "n_active_hand": int(sum(x > 0 for x in per_hand)), "sq_term_same_plan_for_the_record": float(t_sq)}
# ---------------- o2 (c)
trq = planted.clone().requires_grad_(True)
term, ti = ROW.own_term(trq, inp, FORM)
term.backward()
g_rows = trq.grad.abs().sum(dim=(1, 2)).numpy()
res["o2"]["c_no_lead_rows"] = {"n_rows_without_mask": int((~inp["mask"]).sum()), "max_abs_grad_on_them": float(g_rows[~inp["mask"]].max()) if (~inp["mask"]).any() else None, "term": float(term.detach()), "n_lead": ti["n_lead"],
                               "n_active": ti["n_active"], "grad_all_finite": bool(np.isfinite(trq.grad.numpy()).all()), "masked_rows_with_grad_equal_n_active": bool(int((g_rows[inp["mask"]] > 0).sum()) == ti["n_active"])}
nan_res = {}
rows_ok = np.nonzero(inp["mask"])[0]
for what in ("v", "p10f"):
    b = int(rows_ok[0])
    ci, t = int(batch["ci"][b]), int(batch["t"][b])
    d = store.get(str(J["clip"][ci]))
    saved = d[what][t].copy() if what == "p10f" else float(d[what][t])
    d[what][t] = np.nan
    inp_n = RO.stage(batch)
    d[what][t] = saved
    tq = planted.clone().requires_grad_(True)
    tn, tin = ROW.own_term(tq, inp_n, FORM)
    tn.backward()
    ones_only = {**inp, "mask": inp["mask"] & (np.arange(n) != b)}
    t_ref, _ = ROW.own_term(planted.clone(), ones_only, FORM)
    nan_res[what] = {"row": b, "in_mask": bool(inp_n["mask"][b]), "n_lead": tin["n_lead"], "n_lead_before": ti["n_lead"], "grad_on_row": float(tq.grad[b].abs().sum()),
                     "term_equals_term_without_that_row_exactly": bool(float(tn.detach()) == float(t_ref.detach()))}
res["o2"]["c_nan_speed"] = nan_res
c = res["o2"]
c["PASS"] = bool(c["mask_equal"] and c["n_masked"] == c["n_masked_hand"] > 0 and not any(x["rows_not_equal"] for x in c["a"]) and c["a"][1]["n_changed_by_hand"] > 0
                 and c["b"]["rel_diff"] <= 1e-5 and c["b"]["n_active"] == c["b"]["n_active_hand"] > 0 and abs(c["b"]["term_trainer_function"] - c["b"]["sq_term_same_plan_for_the_record"]) > 1e-3
                 and c["c_no_lead_rows"]["max_abs_grad_on_them"] == 0.0 and c["c_no_lead_rows"]["grad_all_finite"] and c["c_no_lead_rows"]["masked_rows_with_grad_equal_n_active"] and (~inp["mask"]).any()
                 and all((not v["in_mask"]) and v["n_lead"] == v["n_lead_before"] - 1 and v["grad_on_row"] == 0.0 and v["term_equals_term_without_that_row_exactly"] for v in nan_res.values()))

# ---------------- o3
res["o3"] = {}
mk = lambda k, gap, v0, vl, mask=None: {"mask": np.ones(k, bool) if mask is None else mask, "v0": np.full(k, v0), "gap": np.full(k, gap), "v_L": np.full(k, vl), "n_rows": k}      # noqa: E731
run = torch.as_tensor(np.cumsum(np.zeros((4, 30, 2), np.float32) + np.array([2.0, 0.0], np.float32), axis=1)).requires_grad_(True)           # 10 m/s straight
t1, i1 = ROW.own_term(run, mk(4, 8.0, 10.0, 0.0), FORM)
t1.backward()
res["o3"]["part1_into_the_lead"] = {"term": float(t1.detach()), "n_active": i1["n_active"], "grad_abs_sum": float(run.grad.abs().sum()), "grad_finite": bool(torch.isfinite(run.grad).all())}
slow = torch.as_tensor(np.cumsum(np.zeros((4, 30, 2), np.float32) + np.array([0.1, 0.0], np.float32), axis=1)).requires_grad_(True)
t2, i2 = ROW.own_term(slow, mk(4, 40.0, 0.5, 0.0), FORM)
t2.backward()
res["o3"]["part2_inside_the_rule"] = {"term": float(t2.detach()), "n_lead": i2["n_lead"], "n_active": i2["n_active"], "grad_abs_sum": float(slow.grad.abs().sum()), "grad_finite": bool(torch.isfinite(slow.grad).all())}
nol = run.detach().clone().requires_grad_(True)
t3, i3 = ROW.own_term(nol, mk(4, 8.0, 10.0, 0.0, mask=np.zeros(4, bool)), FORM)
res["o3"]["part3_no_lead"] = {"term": float(t3.detach()), "n_lead": i3["n_lead"], "requires_grad": bool(t3.requires_grad)}
idx_no = [i for i, w in enumerate(wl) if not eligible_by_hand(int(w[0]), int(w[1]))[0]][:16]
b_no = D.collate([ds.item(*wl[i]) for i in idx_no])
inp_no = RO.stage(b_no)
pl_no = model.loss(dict(b_no), p_gt=0.5, gen=torch.Generator().manual_seed(11))["traj30"].detach().clone()
pl_no[:, :, 0] = torch.arange(1, 31, dtype=torch.float32)[None, :] * 0.2 * 12.0
tno, ino = ROW.own_term(pl_no.requires_grad_(True), inp_no, FORM)
res["o3"]["part3b_real_batch_without_lead"] = {"n_rows": len(idx_no), "n_lead": ino["n_lead"], "term": float(tno.detach())}
# part 4: partial on the real batch, gradient reaches the model
half = sel[: max(1, len(sel) // 2)]
rest = sel[len(half):]
pl4 = pred.clone()
pl4[half] = torch.as_tensor(np.cumsum(np.zeros((len(half), 30, 2), np.float32) + np.array([2.4, 0.0], np.float32), axis=1))          # 12 m/s into a lead 8 m away
pl4[rest] = torch.as_tensor(np.cumsum(np.zeros((len(rest), 30, 2), np.float32) + np.array([0.1, 0.0], np.float32), axis=1))          # a crawl, lead 40 m away: inside the rule
gap4 = inp["gap"].copy()
gap4[half], gap4[rest] = 8.0, 40.0
inp4 = dict(inp, gap=gap4, v_L=np.where(inp["mask"], 0.0, inp["v_L"]))
p4 = pl4.clone().requires_grad_(True)
t4, i4 = ROW.own_term(p4, inp4, FORM)
t4.backward()
res["o3"]["part4_partial_planted_half"] = {"n_rows": i4["n_rows"], "n_lead": i4["n_lead"], "n_active": i4["n_active"], "term": float(t4.detach()), "grad_on_inactive_masked_rows": float(p4.grad[rest].abs().sum())}
print("PARTIAL: n_lead", i4["n_lead"], "n_active", i4["n_active"])
named = [(nm, p) for nm, p in model.named_parameters() if p.requires_grad]
for _, p in named:
    p.grad = None
torch.manual_seed(20261115)
o = model.loss(dict(batch), p_gt=0.5, gen=torch.Generator().manual_seed(11))
T = o["terms"]
tr = o["traj30"]
off = torch.zeros_like(tr)
off[sel] = pl4[sel] - tr.detach()[sel]
tot_before = float(T["total"].detach())
inf = ROW.add_own_term(T, tr + off, inp4, 1.0, FORM)
T["total"].backward()
gsum = float(sum(float(p.grad.abs().sum()) for _, p in named if p.grad is not None))
res["o3"]["part4_model_gradient"] = {"total_before": tot_before, "total_after": float(T["total"].detach()), "term": inf["term"], "total_delta_equals_term": bool(abs(float(T["total"].detach()) - tot_before - inf["term"]) <= 1e-4 * (1 + abs(tot_before))),
                                     "param_grad_abs_sum": gsum, "finite": bool(np.isfinite(gsum))}
# part 5: knots exactly on the cap
inp5 = mk(2, 8.0, 12.0, 0.0)
cap0 = torch.as_tensor(ROW.cap_of_plan(np.cumsum(np.zeros((2, 30, 2), np.float32) + np.array([2.4, 0.0], np.float32), axis=1), inp5["v0"], inp5["gap"], inp5["v_L"])).float()
plan5 = cap0.clone()
plan5[:, 20:, 0] += 3.0
cap5 = torch.as_tensor(ROW.cap_of_plan(plan5.numpy(), inp5["v0"], inp5["gap"], inp5["v_L"])).float()
on = (plan5 == cap5).all(-1)
p5 = plan5.clone().requires_grad_(True)
t5, i5 = ROW.own_term(p5, inp5, FORM)
t5.backward()
g5 = p5.grad
q5 = plan5.clone().requires_grad_(True)
((q5 - cap5) ** 2).sum(-1).sqrt().mean().backward()
res["o3"]["part5_knots_on_the_cap"] = {"n_knots_on_cap": int(on.sum()), "n_knots_off_cap": int((~on).sum()), "term": float(t5.detach()), "grad_all_finite": bool(torch.isfinite(g5).all()),
                                       "grad_abs_sum_on_cap_knots": float(g5[on].abs().sum()), "grad_abs_sum_off_cap_knots": float(g5[~on].abs().sum()), "plain_sqrt_gradient_all_finite_for_the_record": bool(torch.isfinite(q5.grad).all())}
o3 = res["o3"]
o3["PASS"] = bool(o3["part1_into_the_lead"]["term"] > 0 and o3["part1_into_the_lead"]["grad_abs_sum"] > 0 and o3["part1_into_the_lead"]["grad_finite"] and o3["part1_into_the_lead"]["n_active"] == 4
                  and o3["part2_inside_the_rule"]["term"] == 0.0 and o3["part2_inside_the_rule"]["grad_abs_sum"] == 0.0 and o3["part2_inside_the_rule"]["n_active"] == 0 and o3["part2_inside_the_rule"]["grad_finite"]
                  and o3["part3_no_lead"]["term"] == 0.0 and o3["part3_no_lead"]["n_lead"] == 0 and not o3["part3_no_lead"]["requires_grad"]
                  and o3["part3b_real_batch_without_lead"]["n_lead"] == 0 and o3["part3b_real_batch_without_lead"]["term"] == 0.0
                  and 0 < o3["part4_partial_planted_half"]["n_active"] < o3["part4_partial_planted_half"]["n_lead"] and o3["part4_partial_planted_half"]["grad_on_inactive_masked_rows"] == 0.0
                  and o3["part4_model_gradient"]["param_grad_abs_sum"] > 0 and o3["part4_model_gradient"]["finite"] and o3["part4_model_gradient"]["total_delta_equals_term"]
                  and o3["part5_knots_on_the_cap"]["n_knots_on_cap"] > 0 and o3["part5_knots_on_the_cap"]["n_knots_off_cap"] > 0 and o3["part5_knots_on_the_cap"]["grad_all_finite"]
                  and o3["part5_knots_on_the_cap"]["grad_abs_sum_on_cap_knots"] == 0.0 and o3["part5_knots_on_the_cap"]["grad_abs_sum_off_cap_knots"] > 0)
res["PASS"] = bool(res["o2"]["PASS"] and res["o3"]["PASS"])
json.dump(res, open(outp, "w", encoding="utf-8", newline="\n"), indent=1)
print(json.dumps({"o2": res["o2"]["PASS"], "o3": res["o3"]["PASS"], "b": res["o2"]["b"], "part4": res["o3"]["part4_partial_planted_half"], "part5": res["o3"]["part5_knots_on_the_cap"]}))
print("O2_O3", "PASS" if res["PASS"] else "FAIL")
