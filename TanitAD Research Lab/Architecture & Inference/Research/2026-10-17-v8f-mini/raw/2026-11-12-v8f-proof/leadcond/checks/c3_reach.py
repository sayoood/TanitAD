"""c3 reach (LEADCOND), in ONE tree (arg: patch | mutO | mutT; the mutants are copies of tree_patch with ONE line deleted: the line adding the token in run_o / run_t).
 a  setting lead_cond.weight non-zero by hand moves the OPERATIVE plan (O alone: the code O reads is held fixed with forward_diag(code_override)) and the TACTICAL outputs (T alone: goal / lat / lon logits)
 b  one real loss step: the gradient on lead_cond.weight is above 0 (and the bias); with u forced to zeros the weight gradient is EXACTLY 0.0 (bias still above 0)
 c  the plan loss (total minus the lead_sup term) gives the lead head's parameters EXACTLY 0.0 gradient through this path (and with the path deleted: the same); control: the total WITH the lead_sup term gives them a gradient above 0
 d  detach into the trunk: the gradient of sum(lead_cond(u)) on the adapter is exactly 0.0; control: the same sum with u built WITHOUT detach gives the adapter a gradient above 0
PASS (for patch) = all of a..d true.  For a mutant the check must be RED (reach_ok false). Usage: c3_reach.py <tree> [out.json]"""
import sys

import torch

tree = sys.argv[1]
import lc_common as C

C.use_tree(tree)
from tanitad.models import v8f_mini as M

batch, J = C.world_batch("c3")
_, cfg, model = C.build_from_args(["--lead-sup", "on", "--w-lead-sup", "1.0", "--lead-cond", "on"], J, seed=3)
g = torch.Generator().manual_seed(1)
with torch.no_grad():
    model.head_o[2].weight.normal_(0, 0.1)                                  # wake O's zero-init head
    for m_ in model.wm.lead_head.mlp:
        if isinstance(m_, torch.nn.Linear):
            m_.weight.copy_(torch.randn(m_.weight.shape, generator=g) * 0.3)
model.eval()
bt = {k: v for k, v in batch.items() if k in M.INFERENCE_INPUT_KEYS}
B = batch["feats"].shape[0]
code = torch.full((B, M.CODE_DIM), 0.25)
with torch.no_grad():
    base = model.forward_diag(bt, code)
    ref = model.forward_infer(bt)
    model.lead_cond.weight.copy_(torch.randn(model.lead_cond.weight.shape, generator=g) * 0.5)
    model.lead_cond.bias.copy_(torch.randn(model.lead_cond.bias.shape, generator=g) * 0.5)
    moved = model.forward_diag(bt, code)
    moved_free = model.forward_infer(bt)
r_o = float((moved["traj60"] - base["traj60"]).abs().max())                   # O alone: the code is fixed
r_t = max(float((moved[k] - base[k]).abs().max()) for k in ("goal_logits", "lat_logits", "lon_logits"))
r_o_free = float((moved_free["traj60"] - ref["traj60"]).abs().max())
res = {"tree": tree, "a_reach": {"operative_traj60_max_abs_change_code_fixed": r_o, "tactical_logits_max_abs_change": r_t, "operative_traj60_max_abs_change_free": r_o_free}}
# b: a real loss step
model.train()
out = model.loss(dict(batch), p_gt=0.5, gen=torch.Generator().manual_seed(11))
model.zero_grad(set_to_none=True)
out["terms"]["total"].backward()
gw, gb = float(model.lead_cond.weight.grad.abs().sum()), float(model.lead_cond.bias.grad.abs().sum())
model.zero_grad(set_to_none=True)
orig = model._lead_u
model._lead_u = lambda b, adapt, last: torch.zeros(adapt.shape[0], 2, dtype=torch.float32)
out0 = model.loss(dict(batch), p_gt=0.5, gen=torch.Generator().manual_seed(11))
out0["terms"]["total"].backward()
gw0, gb0 = float(model.lead_cond.weight.grad.abs().sum()), float(model.lead_cond.bias.grad.abs().sum())
model._lead_u = orig
res["b_grad"] = {"weight": gw, "bias": gb, "control_u_zeros_weight": gw0, "control_u_zeros_bias": gb0}
# c: the plan loss gives the lead head exactly 0
head = [p for n, p in model.named_parameters() if n.startswith("wm.lead_head.")]
model.zero_grad(set_to_none=True)
o1 = model.loss(dict(batch), p_gt=0.5, gen=torch.Generator().manual_seed(11))["terms"]
(o1["total"] - cfg.w_lead_sup * o1["lead_sup"]).backward()
g_plan = max((float(p.grad.abs().max()) if p.grad is not None else 0.0) for p in head)
model.zero_grad(set_to_none=True)
o2 = model.loss(dict(batch), p_gt=0.5, gen=torch.Generator().manual_seed(11))["terms"]
o2["total"].backward()
g_total = max((float(p.grad.abs().max()) if p.grad is not None else 0.0) for p in head)
res["c_head_grad"] = {"plan_loss_through_u_max_abs": g_plan, "control_total_with_lead_sup_term_max_abs": g_total}
# d: detach into the trunk
model.zero_grad(set_to_none=True)
adapt = model._encode(batch)
last = torch.full((B,), adapt.shape[1] - 1, dtype=torch.long)
model.lead_cond(model._lead_u(batch, adapt, last)).sum().backward()
ad = [p for n, p in model.named_parameters() if n.startswith(("adapter", "pos", "time", "wm.proj", "wm.lead_head"))]
d_detached = max((float(p.grad.abs().max()) if p.grad is not None else 0.0) for p in ad)
model.zero_grad(set_to_none=True)
h = cfg.lead_hist
pos = last[:, None] - (h - 1 - torch.arange(h))[None]
win = torch.gather(adapt, 1, pos[:, :, None, None].expand(-1, -1, adapt.shape[2], adapt.shape[3]))
o = model.wm.lead_head(model.wm.z1(win))[:, 0]
u_nd = torch.stack([torch.sigmoid(o[:, 0]), torch.sigmoid(o[:, 0]) * o[:, 1].clamp(0, M.LEAD_COND_GAP_MAX)], -1)
model.lead_cond(u_nd).sum().backward()
d_ctl = max((float(p.grad.abs().max()) if p.grad is not None else 0.0) for p in ad)
res["d_detach_trunk"] = {"adapter_and_head_grad_through_u_max_abs": d_detached, "control_undetached_max_abs": d_ctl}
reach_ok = bool(r_o > 0.0 and r_t > 0.0 and gw > 0.0 and gb > 0.0)
res["reach_ok"] = reach_ok
res["PASS"] = bool(reach_ok and gw0 == 0.0 and gb0 > 0.0 and g_plan == 0.0 and g_total > 0.0 and d_detached == 0.0 and d_ctl > 0.0)
out_path = sys.argv[2] if len(sys.argv) > 2 else f"{C.LC}/checks/c3_reach_{tree}.json"
C.dump(out_path, res)
print(tree, "reach_ok", reach_ok, "PASS", res["PASS"], res["a_reach"])
