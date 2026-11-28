"""Measures the numbers REPORT_B1.json quotes for Z0, Z1, Z2 and Z-ID (the tests assert them; this prints their values). CPU, fp32, ONE thread, no pytest. usage: report_values.py <tree> <tmpdir> <out.json>"""
import json
import sys
import tempfile
from pathlib import Path

TREE, TMP, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(TREE / "stack" / "tests"))
import torch  # noqa: E402

import test_v8f_n1dw as T  # noqa: E402

torch.set_num_threads(1)
M, TR, D = T.M, T.TR, T.D
root = Path(tempfile.mkdtemp(prefix="n1dwrv_", dir=str(TMP)))
w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
J0 = D.load_join(root / "join_train.npz")
world = {"root": root, "rbd": T.write_rbd_npz(root / "rbd.npz", J0), "nav": T.write_nav_npz(root / "nav.npz", J0), **w}
b, J, _ = T.make_batch(world)
res = {"device": "cpu", "dtype": "float32", "num_threads": torch.get_num_threads(), "torch": torch.__version__, "batch": int(b["rb"].shape[0]), "rb_in_batch": int(b["rb"].sum()), "gt_ok_in_batch": int(b["gt_ok"].sum())}
# Z0
b0, _, _ = T.make_batch(world, with_rb=False)
mt, mn = T.build(J, mod=T.base_module()), T.build(J, mod=M)
ot, on = T.run_loss(mt, b0), T.run_loss(mn, b0)
probs, extra = T.equal_runs(ot, on)
res["Z0"] = {"state_dict_keys": len(mn.state_dict()), "state_dict_values_equal": all(torch.equal(v, mn.state_dict()[k]) for k, v in mt.state_dict().items()), "loss_terms": sorted(on["terms"]),
             "n_terms": len(on["terms"]), "terms_and_outputs_problems": probs, "extra_terms": extra, "total_equal": bool(torch.equal(ot["terms"]["total"], on["terms"]["total"])),
             "built_levers_equal": mt.built_levers() == mn.built_levers()}
# Z1
m1, m4 = T.build(J, rb_window_w=1.0), T.build(J, rb_window_w=4.0)
o1, o4 = T.run_loss(m1, b), T.run_loss(m4, b)
probs, extra = T.equal_runs(o1, o4)
ind = T.indep_rb_term(o4["traj30"].detach(), b["gt"], b["gt_ok"].float(), b["rb"])
want = float(m4.cfg.w_traj) * 3.0 * ind
diff = float(o4["terms"]["total"].detach()) - float(o1["terms"]["total"].detach())
res["Z1"] = {"w_traj": float(m4.cfg.w_traj), "existing_terms_and_outputs_problems": probs, "extra_terms": extra, "rb_traj": float(o4["terms"]["rb_traj"].detach()), "rb_traj_independent_float64": ind,
             "rb_traj_rel_err": abs(float(o4["terms"]["rb_traj"].detach()) - ind) / ind, "total_w1": float(o1["terms"]["total"].detach()), "total_w4": float(o4["terms"]["total"].detach()),
             "total_diff": diff, "expected_w_traj_x3_x_rb_traj": want, "total_diff_rel_err": abs(diff - want) / want}
# Z2
r = TR.rb_window_probe(m4, b, m4.cfg)
res["Z2"] = {k: r[k] for k in ("rb_window_w_declared", "rb_n", "rb_n_eligible", "rb_eligible", "term_forced", "term_at_unit_weight", "contribution", "w_traj_x_wm1_x_term", "head_o_grad_diff", "natural_term",
                               "natural_contribution", "natural_head_o_grad_diff", "control_repeat_total_diff", "control_repeat_grad_diff", "control_allfalse_term", "control_allfalse_total_diff",
                               "control_allfalse_grad_diff", "control_prior_grad_diff", "pass")}
# Z-ID
gok = b["gt_ok"].float()
rb = b["rb"]
traj = o4["traj30"].detach()
f = lambda a, fe: a / 3.0 + 0.5 * fe / 6.0                                                      # noqa: E731


def grad_of(fn):
    x = traj.clone().requires_grad_(True)
    fn(x).backward()
    return x.grad


wt = float(m4.cfg.w_traj)
g1 = grad_of(lambda x: wt * f(*M.traj_terms(x, b["gt"], gok)))
g4 = grad_of(lambda x: wt * f(*M.traj_terms(x, b["gt"], gok)) + wt * 3.0 * f(*M.rb_traj_terms(x, b["gt"], gok, rb)))
rbs, oth = rb & (gok > 0), ~rb
ratio = (g4[rbs] / g1[rbs])
res["Z_ID"] = {"w": 4.0, "n_rb_samples": int(rbs.sum()), "n_other_samples": int(oth.sum()), "rb_ratio_min": float(ratio.min()), "rb_ratio_max": float(ratio.max()), "rb_max_rel_err_vs_w": float((ratio - 4.0).abs().max() / 4.0),
               "non_rb_bit_equal": bool(torch.equal(g4[oth], g1[oth]))}
OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res)[:600])
