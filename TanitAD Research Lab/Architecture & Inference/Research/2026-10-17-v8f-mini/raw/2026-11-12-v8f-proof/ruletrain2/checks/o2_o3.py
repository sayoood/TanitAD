"""RULEOWN checks o2 (the cap is the rule by hand) and o3 (reach), CPU, tiny model, synthetic world with the synthetic lead track.
Usage: o2_o3.py <tree_root> <out json>   (PYTHONPATH = tree stack;taniteval;stack/scripts)

o2  batch of 64 logged windows from the trainer's own build_data. The mask and the rule inputs are recomputed from the RAW arrays (join lead_valid / lead_gap, store v[t], p10f[t][0]) and NOT from the batch tensors.
    (a) for EVERY masked row the cap that the term uses equals dist_rule.distance_rule called by hand on that row (batch size 1, detached float32 prediction, floor_lifts False): exact (torch.equal); the mask equals the hand mask;
        done on the model's own prediction AND on a planted fast plan (so the rule is active and changes the plan)
    (b) the rule applied twice equals once: max abs difference of cap(cap(plan)) and cap(plan), at most 1e-5
    (c) rows without a label lead, and rows with a non-finite raw speed (v[t] then p10f[t][0], injected into the store), contribute exactly 0 to the term and get an exactly zero gradient
o3  reach. Part 1 a plan that runs into the lead: term > 0, gradient on the trajectory output > 0. Part 2 a plan inside the rule: term and gradient exactly 0. Part 3 a batch with no lead: exactly 0 for both.
    Part 4 on a real batch of the tiny V8FMini the term is partial (some lead samples contribute, some do not; both counts printed) and its gradient reaches the model parameters; the total with the term differs from the total without.
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
from tanitad.data import rule_own as ROW  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402

assert os.path.abspath(TR.__file__).lower().startswith(os.path.abspath(tree).lower()), TR.__file__
assert os.path.abspath(ROW.__file__).lower().startswith(os.path.abspath(tree).lower()), ROW.__file__
res = {"tanitad_file": tanitad.__file__, "rule_own_file": ROW.__file__}
OUT = "F:/TanitAD-scratch/v8proof/ruletrain2/work/tmp/o2o3"
os.makedirs(OUT, exist_ok=True)
argv = ["--synthetic", "--batch", "64", "--d", "32", "--n-q", "2", "--threads", "1", "--seed", "0", "--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", "--wm-state", "grid", "--wm-cell-dim", "16",
        "--w-rule-own", "1.0", "--exclude", "1:3", "--out", OUT]
a = TR.build_parser().parse_args(argv)
cfg = TR.make_config(a)
torch.manual_seed(0)
np.random.seed(0)
store, J, wins, clips, info = TR.build_data(a, cfg)
ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
RO = ROW.RuleOwn(a.w_rule_own, a.rule_relabel_weight, J, store)
LSA = D.lead_sup_label_arrays(J)
wl = list(wins)
torch.manual_seed(0)
model = M.V8FMini(cfg, M.P10Prior.synthetic(0), torch.zeros(M.N_CONS), torch.ones(M.N_CONS))
model.train()


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
# ---------------- o2 (a)
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


def cap_check(plan_t, name):
    sel = np.nonzero(inp["mask"])[0]
    caps = torch.as_tensor(ROW.cap_of_plan(plan_t[sel].numpy().astype(np.float32), inp["v0"][sel], inp["gap"][sel], inp["v_L"][sel])).float()      # float64 call, cast to the prediction dtype as the term does
    bad, n_changed = [], 0
    for k, b in enumerate(sel):
        g, clos, v0, v_log = rows[int(b)]
        o = R.distance_rule(plan_t[b:b + 1].numpy().astype(np.float64), np.array([v0]), np.array([True]), np.array([g]), np.array([g]), np.array([max(0.0, v_log + clos)]), m=10, floor_lifts=False)
        n_changed += int(o.changed[0])
        if not torch.equal(caps[k], torch.as_tensor(o.plan[0]).float()):
            bad.append(int(b))
    return {"name": name, "n": int(len(sel)), "n_changed_by_hand": n_changed, "rows_not_equal": bad}, caps


planted = pred.clone()
planted[:, :, 0] = torch.arange(1, 31, dtype=torch.float32)[None, :] * 0.2 * 12.0           # 12 m/s straight on every row
planted[:, :, 1] = 0.0
ca, _ = cap_check(pred, "own_prediction")
cb, caps_planted = cap_check(planted, "planted_fast_plan")
res["o2"]["a"] = [ca, cb]
# ---------------- o2 (b)
sel = np.nonzero(inp["mask"])[0]
c1 = ROW.cap_of_plan(planted[sel].numpy().astype(np.float32), inp["v0"][sel], inp["gap"][sel], inp["v_L"][sel])
c2 = ROW.cap_of_plan(np.asarray(c1, np.float32), inp["v0"][sel], inp["gap"][sel], inp["v_L"][sel])
c2 = ROW.cap_of_plan(c1, inp["v0"][sel], inp["gap"][sel], inp["v_L"][sel])
e1, e2 = c1, c2                                                      # the float64 path the term uses
s = lambda x: np.asarray(x, np.float64)      # noqa: E731
r1 = R.distance_rule(planted[sel].numpy().astype(np.float32), s(inp["v0"][sel]), np.ones(len(sel), bool), s(inp["gap"][sel]), s(inp["gap"][sel]), s(inp["v_L"][sel]), m=10, floor_lifts=False).plan
r2 = R.distance_rule(np.asarray(r1, np.float32), s(inp["v0"][sel]), np.ones(len(sel), bool), s(inp["gap"][sel]), s(inp["gap"][sel]), s(inp["v_L"][sel]), m=10, floor_lifts=False).plan
d32 = float(np.abs(np.asarray(r1, np.float64) - np.asarray(r2, np.float64)).max())
res["o2"]["b"] = {"max_abs_diff_twice_vs_once_float64_plan": float(np.abs(e1 - e2).max()), "max_abs_diff_twice_vs_once_float32_call_for_the_record": d32,
                  "float32_note": "for the record: the rule called on float32 plans; each pass rounds to float32 and the rule recursion amplifies it. The term calls the rule in float64 and casts the cap back",
                  "n_changed_by_first_pass": int((np.asarray(c1) != planted[sel].numpy()).any(axis=(1, 2)).sum())}
# ---------------- o2 (c)
trq = planted.clone().requires_grad_(True)
term, ti = ROW.own_term(trq, inp)
term.backward()
g_rows = trq.grad.abs().sum(dim=(1, 2)).numpy()
res["o2"]["c_no_lead_rows"] = {"n_rows_without_mask": int((~inp["mask"]).sum()), "max_abs_grad_on_them": float(g_rows[~inp["mask"]].max()) if (~inp["mask"]).any() else None,
                               "term_with_all_rows_planted": float(term.detach()), "n_lead": ti["n_lead"], "n_active": ti["n_active"], "n_masked_rows_with_grad_gt0": int((g_rows[inp["mask"]] > 0).sum()), "masked_rows_with_grad_equal_n_active": bool(int((g_rows[inp["mask"]] > 0).sum()) == ti["n_active"])}
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
    tn, tin = ROW.own_term(tq, inp_n)
    tn.backward()
    ones_only = {**inp, "mask": inp["mask"] & (np.arange(n) != b)}
    t_ref, _ = ROW.own_term(planted.clone(), ones_only)
    nan_res[what] = {"row": b, "in_mask": bool(inp_n["mask"][b]), "n_lead": tin["n_lead"], "n_lead_before": ti["n_lead"], "grad_on_row": float(tq.grad[b].abs().sum()),
                     "term_equals_term_without_that_row_exactly": bool(float(tn.detach()) == float(t_ref.detach()))}
res["o2"]["c_nan_speed"] = nan_res
c = res["o2"]
c["PASS"] = bool(c["mask_equal"] and c["n_masked"] == c["n_masked_hand"] > 0 and not any(x["rows_not_equal"] for x in c["a"]) and c["a"][1]["n_changed_by_hand"] > 0
                 and c["b"]["max_abs_diff_twice_vs_once_float64_plan"] <= 1e-5 and c["b"]["n_changed_by_first_pass"] > 0
                 and c["c_no_lead_rows"]["max_abs_grad_on_them"] == 0.0 and c["c_no_lead_rows"]["masked_rows_with_grad_equal_n_active"] and (~inp["mask"]).any()
                 and all((not v["in_mask"]) and v["n_lead"] == v["n_lead_before"] - 1 and v["grad_on_row"] == 0.0 and v["term_equals_term_without_that_row_exactly"] for v in nan_res.values()))

# ---------------- o3
res["o3"] = {}
run = torch.as_tensor(np.cumsum(np.full((4, 30, 2), 0.0, np.float32) + np.array([2.0, 0.0], np.float32), axis=1)).requires_grad_(True)           # 10 m/s straight
mk = lambda k, gap, v0, vl, mask=None: {"mask": np.ones(k, bool) if mask is None else mask, "v0": np.full(k, v0), "gap": np.full(k, gap), "v_L": np.full(k, vl), "n_rows": k}      # noqa: E731
t1, i1 = ROW.own_term(run, mk(4, 8.0, 10.0, 0.0))
t1.backward()
res["o3"]["part1_into_the_lead"] = {"term": float(t1.detach()), "n_active": i1["n_active"], "grad_abs_sum": float(run.grad.abs().sum())}
slow = torch.as_tensor(np.cumsum(np.full((4, 30, 2), 0.0, np.float32) + np.array([0.1, 0.0], np.float32), axis=1)).requires_grad_(True)
t2, i2 = ROW.own_term(slow, mk(4, 40.0, 0.5, 0.0))
t2.backward()
res["o3"]["part2_inside_the_rule"] = {"term": float(t2.detach()), "n_lead": i2["n_lead"], "n_active": i2["n_active"], "grad_abs_sum": float(slow.grad.abs().sum())}
nol = run.detach().clone().requires_grad_(True)
t3, i3 = ROW.own_term(nol, mk(4, 8.0, 10.0, 0.0, mask=np.zeros(4, bool)))
res["o3"]["part3_no_lead"] = {"term": float(t3.detach()), "n_lead": i3["n_lead"], "requires_grad": bool(t3.requires_grad)}
# part 3b: rows of the real batch WITHOUT a lead, the real model, planted fast plans: exactly 0
idx_no = [i for i, w in enumerate(wl) if not eligible_by_hand(int(w[0]), int(w[1]))[0]][:16]
b_no = D.collate([ds.item(*wl[i]) for i in idx_no])
inp_no = RO.stage(b_no)
o_no = model.loss(dict(b_no), p_gt=0.5, gen=torch.Generator().manual_seed(11))
tr_no = o_no["traj30"]
pl_no = tr_no.detach().clone()
pl_no[:, :, 0] = torch.arange(1, 31, dtype=torch.float32)[None, :] * 0.2 * 12.0
pl_no = pl_no.requires_grad_(True)
tno, ino = ROW.own_term(pl_no, inp_no)
res["o3"]["part3b_real_batch_without_lead"] = {"n_rows": len(idx_no), "n_lead": ino["n_lead"], "term": float(tno.detach())}
# part 4: partial on the real batch, gradient reaches the model
sel = np.nonzero(inp["mask"])[0]
half = sel[: max(1, len(sel) // 2)]
pl4 = pred.clone()
rest = sel[len(half):]
pl4[half] = torch.as_tensor(np.cumsum(np.full((len(half), 30, 2), 0.0, np.float32) + np.array([2.4, 0.0], np.float32), axis=1))          # 12 m/s into a lead 8 m away
pl4[rest] = torch.as_tensor(np.cumsum(np.full((len(rest), 30, 2), 0.0, np.float32) + np.array([0.1, 0.0], np.float32), axis=1))          # a crawl, lead 40 m away: inside the rule
gap4 = inp["gap"].copy()
gap4[half], gap4[rest] = 8.0, 40.0
inp4 = dict(inp, gap=gap4, v_L=np.where(inp["mask"], 0.0, inp["v_L"]))
t4, i4 = ROW.own_term(pl4.clone().requires_grad_(True), inp4)
res["o3"]["part4_partial_planted_half"] = {"n_rows": i4["n_rows"], "n_lead": i4["n_lead"], "n_active": i4["n_active"], "term": float(t4.detach())}
t4n, i4n = ROW.own_term(pred.clone().requires_grad_(True), inp)
res["o3"]["part4_model_own_prediction"] = {"n_lead": i4n["n_lead"], "n_active": i4n["n_active"], "term": float(t4n.detach()), "note": "the untrained tiny model plans a crawl, so the rule rarely binds on its own plan"}
named = [(nm, p) for nm, p in model.named_parameters() if p.requires_grad]


def total_grad(with_term, inp_x):
    for _, p in named:
        p.grad = None
    torch.manual_seed(20261115)
    o = model.loss(dict(batch), p_gt=0.5, gen=torch.Generator().manual_seed(11))
    T = o["terms"]
    tr = o["traj30"]
    if with_term:
        # plant the fast plan through the graph: the term reads the model's own output plus a fixed offset that makes the plan run into the lead on the masked rows
        off = torch.zeros_like(tr)
        off[sel] = pl4[sel] - tr.detach()[sel]
        ROW.add_own_term(T, tr + off, inp_x, 1.0)
    T["total"].backward()
    return float(T["total"].detach()), {nm: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for nm, p in named}


tot0, g0 = total_grad(False, inp4)
tot1, g1 = total_grad(True, inp4)
res["o3"]["part4_model_gradient"] = {"total_without": tot0, "total_with": tot1, "total_delta": tot1 - tot0, "planted_term_value": float(t4.detach()), "grad_diff_abs_sum": float(sum((g0[k] - g1[k]).abs().sum() for k in g0))}
o = res["o3"]
o["PASS"] = bool(o["part1_into_the_lead"]["term"] > 0 and o["part1_into_the_lead"]["grad_abs_sum"] > 0 and o["part1_into_the_lead"]["n_active"] == 4
                 and o["part2_inside_the_rule"]["term"] == 0.0 and o["part2_inside_the_rule"]["grad_abs_sum"] == 0.0 and o["part2_inside_the_rule"]["n_active"] == 0
                 and o["part3_no_lead"]["term"] == 0.0 and o["part3_no_lead"]["n_lead"] == 0 and not o["part3_no_lead"]["requires_grad"]
                 and o["part3b_real_batch_without_lead"]["term"] == 0.0 and o["part3b_real_batch_without_lead"]["n_lead"] == 0
                 and 0 < o["part4_partial_planted_half"]["n_active"] < o["part4_partial_planted_half"]["n_lead"]
                 and abs(o["part4_model_gradient"]["total_delta"] - o["part4_model_gradient"]["planted_term_value"]) <= 1e-3 * (1.0 + o["part4_model_gradient"]["planted_term_value"]) and o["part4_model_gradient"]["total_delta"] > 1.0
                 and o["part4_model_gradient"]["grad_diff_abs_sum"] > 0)
res["PASS"] = bool(res["o2"]["PASS"] and res["o3"]["PASS"])
json.dump(res, open(outp, "w", encoding="utf-8", newline="\n"), indent=1)
print(json.dumps(res["o2"]))
print(json.dumps(res["o3"]))
print("O2O3", "PASS" if res["PASS"] else "FAIL")
sys.exit(0 if res["PASS"] else 1)
