"""a2 and a3 (ACTCONS) with numbers: the term against a hand computation (independent route: HJEPAStack.rollout + a plain head call), invalid samples (exactly 0.0 term and gradient, bit-identical term and
gradient when the invalid rows are scrambled), gradient reach (predictor, action embed > 0) and exact zeros (lead head, start state, every other parameter), the total moving by W x term, the planted
kinematic predictor (term ~ 0) and the mutations (sign, offset not applied, wrong units, wrong horizon, head not detached, start state not detached). Writes a2_term.json and a3_planted.json. Usage: a2_a3_term.py"""
import copy
import json

import torch

import ac_common as C

C.use_tree("patch")
import test_v8f_act_cons as T  # noqa: E402  (the tree's own test module: the same helpers and the same planted objects)
from tanitad.models import act_cons as AC  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402,F401

W = C.world("a2")
batch = W["batch"]
d = T._offsets(8)
_, cfg, m = T._built(W, T.LS)
T._wake(m)
adapt = T._adapt(m, batch)
res2 = {}
r = AC.act_cons_term(m, adapt, batch, d, T.MAXOFF)
hand, n_hand = T._hand_term(m, batch, adapt, d, T.MAXOFF)
res2["term"], res2["hand"], res2["abs_diff"], res2["n_valid"], res2["n_valid_hand"] = float(r["term"]), hand, abs(float(r["term"]) - hand), float(r["n_valid"]), n_hand
res2["term_matches_hand_tol_2e-4_rel"] = bool(res2["abs_diff"] < 2e-4 * max(1.0, hand) and res2["n_valid"] == n_hand and res2["term"] > 0.01)
ctl = {}
b0 = dict(batch)
b0["lead_pres"] = torch.zeros_like(batch["lead_pres"])
r0 = AC.act_cons_term(m, adapt, b0, d, T.MAXOFF)
ctl["no_label_lead_term"], ctl["no_label_lead_has_graph"], ctl["no_label_lead_n"] = float(r0["term"]), bool(r0["term"].requires_grad), float(r0["n_valid"])
b1 = dict(batch)
b1["wm_speed"] = torch.full_like(batch["wm_speed"], T.MAXOFF * AC.SPEED_MULT - 0.01)
r1 = AC.act_cons_term(m, adapt, b1, d, T.MAXOFF)
ctl["speed_below_bound_term"], ctl["speed_below_bound_n"] = float(r1["term"]), float(r1["n_valid"])
b1["wm_speed"] = torch.full_like(batch["wm_speed"], 20.0)
ctl["speed_above_bound_control_n"] = float(AC.act_cons_term(m, adapt, b1, d, T.MAXOFF)["n_valid"])
for tag, val in (("gap_61", 61.0), ("gap_nan", float("nan"))):
    b2 = dict(batch)
    b2["lead_gapm"] = torch.full_like(batch["lead_gapm"], val)
    ctl[f"{tag}_n"] = float(AC.act_cons_term(m, adapt, b2, d, T.MAXOFF)["n_valid"])
b3 = dict(batch)
b3["frame_ok"] = torch.zeros_like(batch["frame_ok"])
ctl["padded_frames_n"] = float(AC.act_cons_term(m, adapt, b3, d, T.MAXOFF)["n_valid"])


def grads(bt, dd):
    mm = copy.deepcopy(m)
    mm.zero_grad(set_to_none=True)
    rr = AC.act_cons_term(mm, T._adapt(mm, bt), bt, dd, T.MAXOFF)
    rr["term"].backward()
    return rr, {n: p.grad.clone() for n, p in mm.named_parameters() if p.grad is not None}


ra, ga = grads(batch, d)
vr = AC.valid_mask(batch, AC.valid_starts(16), T.MAXOFF).any(1)
b4 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
d4 = d.clone()
bad = ~vr
b4["act"][bad] = b4["act"][bad] * 3.0 + 1.0
d4[bad] = 9.0
rs, gs = grads(b4, d4)
ctl["rows_valid"], ctl["rows_invalid"] = int(vr.sum()), int((~vr).sum())
ctl["scrambled_invalid_rows_term_equal"] = float(ra["term"]) == float(rs["term"])
ctl["scrambled_invalid_rows_grad_max_abs_diff"] = max(float((ga[k] - gs[k]).abs().max()) for k in ga) if set(ga) == set(gs) else None
res2["invalid_samples"] = ctl
mm = copy.deepcopy(m)
ad = T._adapt(mm, batch).detach().requires_grad_(True)
mm.zero_grad(set_to_none=True)
AC.act_cons_term(mm, ad, batch, d, T.MAXOFF)["term"].backward()
nm = dict(mm.named_parameters())
sm = lambda pre: float(sum(p.grad.abs().sum() for n, p in nm.items() if n.startswith(pre) and p.grad is not None))  # noqa: E731
res2["gradient"] = {"abs_sum_predictor": sm("wm.stack.predictors.0."), "abs_sum_action_embed": sm("wm.stack.act_embed."), "abs_sum_lead_head": sm("wm.lead_head."),
                    "abs_sum_state_projection_wm_proj": sm("wm.proj."), "abs_sum_adapter": sm("adapter."),
                    "n_other_parameters_holding_a_gradient": sum(1 for n, p in nm.items() if p.grad is not None and not n.startswith(("wm.stack.predictors.0.", "wm.stack.act_embed."))),
                    "start_state_input_grad_abs_sum": 0.0 if ad.grad is None else float(ad.grad.abs().sum())}
out = m.loss(dict(batch), p_gt=0.5, gen=torch.Generator().manual_seed(11))
tot0 = float(out["terms"]["total"].detach())
rr = AC.act_cons_term(m, out["adapt"], batch, d, T.MAXOFF)
res2["total_moves"] = {f"W={Wt}": {"delta": float((out["terms"]["total"] + Wt * rr["term"]).detach()) - tot0, "W_x_term": Wt * float(rr["term"])} for Wt in (0.1, 1.0, 3.0)}
g = res2["gradient"]
res2["PASS"] = bool(res2["term_matches_hand_tol_2e-4_rel"] and ctl["no_label_lead_term"] == 0.0 and not ctl["no_label_lead_has_graph"] and ctl["no_label_lead_n"] == 0.0
                    and ctl["speed_below_bound_term"] == 0.0 and ctl["speed_below_bound_n"] == 0.0 and ctl["speed_above_bound_control_n"] > 0 and ctl["gap_61_n"] == 0 and ctl["gap_nan_n"] == 0 and ctl["padded_frames_n"] == 0
                    and ctl["scrambled_invalid_rows_term_equal"] and ctl["scrambled_invalid_rows_grad_max_abs_diff"] == 0.0 and ctl["rows_valid"] > 0 and ctl["rows_invalid"] > 0
                    and g["abs_sum_predictor"] > 0 and g["abs_sum_action_embed"] > 0 and g["abs_sum_lead_head"] == 0.0 and g["abs_sum_state_projection_wm_proj"] == 0.0 and g["abs_sum_adapter"] == 0.0
                    and g["n_other_parameters_holding_a_gradient"] == 0 and g["start_state_input_grad_abs_sum"] == 0.0
                    and all(abs(v["delta"] - v["W_x_term"]) < 1e-4 * max(1.0, abs(tot0)) for v in res2["total_moves"].values()))
C.dump(f"{C.AC_ROOT}/checks/a2_term.json", res2)
print("a2", "PASS" if res2["PASS"] else "FAIL", json.dumps({k: res2[k] for k in ("term", "hand", "abs_diff", "n_valid")}))

# ---- a3: planted kinematic predictor
mp = T._planted(W)
ap = T._adapt(mp, batch)
res3 = {}
rp = AC.act_cons_term(mp, ap, batch, d, T.MAXOFF)
res3["planted_term"], res3["planted_n_valid"], res3["bar"] = float(rp["term"]), float(rp["n_valid"]), 1e-6
res3["mutations"] = {}
for name, old, new in (("sign_of_kinematic_target_flipped", "+ 0.5 * dr[None, :] * (T ** 2)", "- 0.5 * dr[None, :] * (T ** 2)"),
                       ("offset_not_applied_to_rollout", "st.act_embed(a + off)", "st.act_embed(a)"),
                       ("offset_in_wrong_units_d_not_d_over_scale", "dr / float(M.WM_ACC_SCALE)", "dr"),
                       ("horizon_of_the_1s_read_set_to_2s", "(T ** 2)", "(2.0 ** 2)")):
    f = T._mutant(old, new)
    t = float(f(mp, ap, batch, d, T.MAXOFF)["term"])
    res3["mutations"][name] = {"term": t, "red": bool(t > 1e-3)}


def reach(fn):
    mm = copy.deepcopy(m)
    a_ = T._adapt(mm, batch).detach().requires_grad_(True)
    mm.zero_grad(set_to_none=True)
    fn(mm, a_, batch, d, T.MAXOFF)["term"].backward()
    h = sum(float(p.grad.abs().sum()) for n, p in mm.named_parameters() if n.startswith("wm.lead_head.") and p.grad is not None)
    pj = sum(float(p.grad.abs().sum()) for n, p in mm.named_parameters() if n.startswith(("wm.proj.", "adapter.")) and p.grad is not None)
    return {"lead_head_grad": h, "projection_adapter_grad": pj, "start_state_input_grad": 0.0 if a_.grad is None else float(a_.grad.abs().sum())}


res3["gradient_zero_control_real_function"] = reach(AC.act_cons_term)
res3["gradient_zero_mutations"] = {
    "head_parameters_not_detached": reach(T._mutant("params = {n: p.detach() for n, p in head.named_parameters()}", "params = {n: p for n, p in head.named_parameters()}")),
    "start_state_not_detached": reach(T._mutant("        with torch.no_grad():\n            z = wm.z1(adapt.detach().float()).float()", "        if True:\n            z = wm.z1(adapt.float()).float()"))}
gm = res3["gradient_zero_mutations"]
res3["PASS"] = bool(abs(res3["planted_term"]) < 1e-6 and res3["planted_n_valid"] > 2 and all(v["red"] for v in res3["mutations"].values())
                    and all(v == 0.0 for v in res3["gradient_zero_control_real_function"].values())
                    and gm["head_parameters_not_detached"]["lead_head_grad"] > 0 and (gm["start_state_not_detached"]["projection_adapter_grad"] > 0 or gm["start_state_not_detached"]["start_state_input_grad"] > 0))
C.dump(f"{C.AC_ROOT}/checks/a3_planted.json", res3)
print("a3", "PASS" if res3["PASS"] else "FAIL", json.dumps(res3)[:1500])
