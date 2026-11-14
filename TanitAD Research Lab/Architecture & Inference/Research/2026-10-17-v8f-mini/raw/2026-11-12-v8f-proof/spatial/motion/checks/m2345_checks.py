"""m2, m3, m4, m5 at the arm's real predictor sizes (grid state 640 = 40 cells x 16; flat width 256, cell width 128, depth 2, 4 heads; K = 10, T = 16), CPU.
Each check prints its value and its control; the json goes to checks/m2345_checks.json. Run with PYTHONPATH = tree_patch stack;taniteval;stack/scripts.
Usage: m2345_checks.py <tree_root>"""
import inspect
import json
import os
import sys
import types

import torch

tree = sys.argv[1]
import tanitad  # noqa: E402

assert os.path.abspath(tanitad.__file__).lower().startswith(os.path.abspath(tree).lower()), tanitad.__file__
from tanitad.models import hjepa as H  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "m2345_checks.json")
res = {"tanitad_file": tanitad.__file__}
K = 10


def wm(pred, residual=False, k=0, w=0.0, seed=0):
    cfg = M.V8FMiniConfig(wm="b", hmax=16, wm_state="grid", wm_l1_pred=pred, wm_l1_residual="on" if residual else "off", wm_l1_roll_k=k, w_wm_roll=w, d=32, ff=64, heads=4, n_q=2,
                          depth_s=1, depth_t=1, depth_o=1, nb=6)
    torch.manual_seed(seed)
    return M.WMAux(cfg).eval()


def seq(b=4, t=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, t, 640, generator=g), torch.randn(b, t - 1, 2, generator=g)


def gsum(ps):
    return float(sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None))


def nonzero_last(st):
    g = torch.Generator().manual_seed(5)
    last = st.predictors[0].inner.out[-1]
    with torch.no_grad():
        last.weight.copy_(torch.randn(last.weight.shape, generator=g) * 0.2)
        last.bias.copy_(torch.randn(last.bias.shape, generator=g) * 0.2)


# ---- m2
m2 = {}
for pred in ("flat", "cell"):
    st = wm(pred, residual=True).stack
    p = st.predictors[0]
    g = torch.Generator().manual_seed(3)
    d, ctl = [], []
    plain = wm(pred, residual=False).stack.predictors[0]
    for L in (1, 2, 3):
        z, a = torch.randn(4, L, 640, generator=g), torch.randn(4, L, 32, generator=g)
        with torch.no_grad():
            d.append(float((p(z, a) - z).abs().max()))
            ctl.append(float((plain(z, a) - z).abs().max()))
    m2[pred] = {"max_abs_diff_output_minus_input_L1_L2_L3": d, "control_plain_predictor_max_abs_diff": ctl, "wrapper": type(p).__name__, "inner": type(p.inner).__name__,
                "last_layer_abs_sum": float(p.inner.out[-1].weight.abs().sum() + p.inner.out[-1].bias.abs().sum()), "PASS": d == [0.0, 0.0, 0.0] and min(ctl) > 1e-3}
res["m2"] = m2

# ---- m3
m3 = {}
for pred in ("flat", "cell"):
    aux = wm(pred, k=K, w=1.0)
    st = aux.stack.train()
    z, raw = seq()
    zl = z.clone().requires_grad_(True)
    a = st.act_embed(raw)
    st._roll_loss(zl, a, K).backward()
    p = st.predictors[0]
    groups = {n: list(c.parameters()) for n, c in p.named_children()}
    groups.update({n: [q] for n, q in p.named_parameters(recurse=False)})
    per_frame = zl.grad.abs().sum(dim=(0, 2)).tolist()
    starts = list(range(0, 16 - K, 2))
    st.zero_grad()
    zc = z.clone().requires_grad_(True)
    st._pred_loss(1, zc, st.act_embed(raw), detach_target=False).backward()
    # state projection, through the start state (WMAux level: z1 = proj(adapt))
    adapt = torch.randn(2, 16, aux.n_tok, 32, generator=torch.Generator().manual_seed(2))
    act = torch.randn(2, 16, 2, generator=torch.Generator().manual_seed(4))
    aux.zero_grad()
    out = aux.stack.loss(aux.z1(adapt), aux.scale_actions(act)[:, :-1])
    out["roll/L1"].backward()
    m3[pred] = {"predictor_group_grad_abs_sums": {n: gsum(v) for n, v in groups.items()}, "all_groups_nonzero": all(gsum(v) > 0 for v in groups.values()),
                "state_path_grad_per_frame_abs_sum": per_frame, "starts": starts, "last_frame_grad_exact": per_frame[-1], "non_start_frames_all_zero": all(per_frame[t] == 0.0 for t in range(16) if t not in starts),
                "start_frames_all_nonzero": all(per_frame[s] > 0 for s in starts),
                "control_one_step_loss_last_frame_grad": float(zc.grad[:, -1].abs().sum()),
                "state_projection_grad_abs_sum_roll_alone": gsum(aux.proj.parameters()),
                "PASS": all(gsum(v) > 0 for v in groups.values()) and per_frame[-1] == 0.0 and all(per_frame[s] > 0 for s in starts) and float(zc.grad[:, -1].abs().sum()) > 0
                and gsum(aux.proj.parameters()) > 0}
res["m3"] = m3

# ---- m4
m4 = {}
for pred in ("flat", "cell"):
    for residual in (False, True):
        st = wm(pred, residual=residual, k=K, w=0.5).stack
        if residual:
            nonzero_last(st)
        z, raw = seq()
        a = st.act_embed(raw)
        ctx = st.predictors[0].context
        with torch.no_grad():
            pr, tgt, starts = st.roll_pairs(z, a, K)
            worst = max(float((pr[i] - st.rollout(1, z[:, s], raw[:, s:s + K], context=ctx)).abs().max()) for i, s in enumerate(starts))
            ctl = float((pr - tgt).abs().max())
        m4[f"{pred}_residual_{'on' if residual else 'off'}"] = {"max_abs_diff_vs_stack_rollout": worst, "context": ctx, "K": K, "starts": starts, "control_rollout_minus_target": ctl,
                                                                "PASS": worst <= 1e-6 and ctl > 1e-3}
res["m4"] = m4

# ---- m5
W = 0.7


def build(Hm, pred, w):
    kw = dict(pred_kind="cell", pred_cells=40, pred_cell_dim=16) if pred == "cell" else {}
    mk = lambda **x: Hm.LevelConfig(pred_depth=2, pred_heads=4, enc_hidden=128, macro_hidden=32, beta=0.08, gamma=100.0, **x)   # noqa: E731
    torch.manual_seed(0)
    return Hm.HJEPAStack([mk(latent_dim=640, action_dim=2, act_embed_dim=32, pred_width=128 if pred == "cell" else 256, roll_k=K, roll_weight=w, **kw),
                          mk(latent_dim=128, action_dim=32, stride=2, pred_width=128), mk(latent_dim=128, action_dim=32, stride=2, pred_width=128)], sigreg_projections=256).eval()


def delta(Hm, pred):
    s1, s0 = build(Hm, pred, W), build(Hm, pred, W)
    s0.cfg[0].roll_weight = 0.0
    s1, s0 = s1.double(), s0.double()      # float64: the difference of two totals of size ~10 is then exact to ~1e-14
    z, a = (t.double() for t in seq())
    o1 = s1.loss(z, a, generator=torch.Generator().manual_seed(9))
    o0 = s0.loss(z, a, generator=torch.Generator().manual_seed(9))
    d = float(o1["total"] - o0["total"])
    return {"total_W_minus_total_0": d, "W_times_roll": W * float(o1["roll/L1"]), "roll_L1": float(o1["roll/L1"]), "abs_err": abs(d - W * float(o1["roll/L1"]))}


src = inspect.getsource(H)
marker = "level_loss = level_loss + c.roll_weight * roll"
lines = src.split("\n")
idx = [i for i, l in enumerate(lines) if l.strip().startswith(marker)]
assert len(idx) == 1
mut = types.ModuleType("hjepa_mutant")
sys.modules["hjepa_mutant"] = mut
exec(compile("\n".join(lines[:idx[0]] + lines[idx[0] + 1:]), "hjepa_mutant.py", "exec"), mut.__dict__)
m5 = {"marker_line_no": idx[0] + 1}
for pred in ("flat", "cell"):
    real, mutant = delta(H, pred), delta(mut, pred)
    m5[pred] = {"real": real, "mutant_line_deleted": mutant, "PASS": real["abs_err"] < 1e-9 and real["roll_L1"] > 1e-3 and mutant["abs_err"] > 1e-3}
res["m5"] = m5
res["PASS"] = all(v["PASS"] for v in (*m2.values(), *m3.values(), *m4.values(), *(m5[p] for p in ("flat", "cell"))))
json.dump(res, open(OUT, "w", encoding="utf-8", newline="\n"), indent=1)
print(json.dumps({k: (v if k in ("PASS",) else {kk: (vv.get("PASS") if isinstance(vv, dict) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in res.items()}, indent=1))
