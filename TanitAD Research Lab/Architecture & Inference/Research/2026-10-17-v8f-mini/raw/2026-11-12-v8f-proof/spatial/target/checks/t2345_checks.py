"""t2, t3, t4, t5 at the arm's real predictor sizes (grid state 640 = 40 cells x 16; flat width 256, cell width 128, depth 2, 4 heads; token width 1024; K = 10, T = 16), CPU.
Each check prints its value and its control; the json goes to checks/t2345_checks.json. PYTHONPATH = tree_patch stack;taniteval;stack/scripts.
  t2  EGOV reach: the column of the action embedding that reads the third channel gets a non-zero gradient; controls that read EXACTLY 0.0 (zero speed; levels 2 and 3); the loss is sensitive to a permuted speed.
  t3  FUTTOK term alone: tok_head, every level-1 predictor group, the action embedding and the state projection non-zero; levels 2 and 3 EXACTLY 0.0 (control: the total reaches them).
  t4  the states inside the FUTTOK term equal HJEPAStack.rollout(1, ..., context=<predictor context>) for the same start, actions and context.
  t5  float64: total(W) - total(0) = W x futtok/L1; the same assertion on a copy of hjepa.py with the adding line deleted goes red.
Usage: t2345_checks.py <tree_root>"""
import inspect
import json
import os
import sys
import types

import torch
import torch.nn.functional as F

tree = sys.argv[1]
import tanitad  # noqa: E402

assert os.path.abspath(tanitad.__file__).lower().startswith(os.path.abspath(tree).lower()), tanitad.__file__
from tanitad.models import hjepa as H  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "t2345_checks.json")
res = {"tanitad_file": tanitad.__file__}
K, B = 10, 2
NT = 40


def wm(pred, fk=0, fw=0.0, act="acc_yaw", seed=0):
    cfg = M.V8FMiniConfig(wm="b", hmax=16, wm_state="grid", wm_l1_pred=pred, wm_l1_futtok_k=fk, w_wm_futtok=fw, wm_l1_act=act, d=32, ff=64, heads=4, n_q=2,
                          depth_s=1, depth_t=1, depth_o=1, nb=6)           # d_feat 1024, n_tok 40, wm_cell_dim 16: the arm's sizes
    torch.manual_seed(seed)
    return cfg, M.WMAux(cfg)


def inputs(cfg, seed=2):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, 16, cfg.n_tok, cfg.d, generator=g), torch.randn(B, 16, 2, generator=g), torch.rand(B, 16, generator=g) * 20.0,
            torch.randn(B, 16, cfg.n_tok, cfg.d_feat, generator=g).half())


gsum = lambda ps: float(sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None))     # noqa: E731
lvl = lambda st, k: [p for m in ([st.predictors[k - 1], st.idm_heads[k - 1]] + ([st.encoders[k - 2], st.macro_encoders[k - 2]] if k >= 2 else [])) for p in m.parameters()]  # noqa: E731

# ---------------------------------------------------------------- t2 EGOV
res["t2"] = {}
for pred in ("flat", "cell"):
    cfg, w = wm(pred, act="speed")
    adapt, act, spd, feats = inputs(cfg)
    w(adapt, act, speed=spd)["pred/L1"].backward()
    W = w.stack.act_embed[0].weight
    r = {"embed_weight_shape": list(W.shape), "grad_abs_sum_speed_column": float(W.grad[:, 2].abs().sum()), "grad_abs_sum_accel_column": float(W.grad[:, 0].abs().sum()),
         "control_levels_2_3_grad": [gsum(lvl(w.stack, 2)), gsum(lvl(w.stack, 3))]}
    w.zero_grad()
    w(adapt, act, speed=torch.zeros_like(spd))["pred/L1"].backward()
    r["control_zero_speed_column_grad"] = float(w.stack.act_embed[0].weight.grad[:, 2].abs().sum())
    w.eval()
    with torch.no_grad():
        base = float(w(adapt, act, speed=spd)["pred/L1"])
        r["pred_L1"] = {"speed": base, "speed_flipped_over_batch": float(w(adapt, act, speed=spd.flip(0))["pred/L1"]),
                        "speed_shuffled_over_frames": float(w(adapt, act, speed=spd[:, torch.randperm(16, generator=torch.Generator().manual_seed(1))])["pred/L1"])}
    cfg0, w0 = wm(pred)
    w0.eval()
    with torch.no_grad():
        r["lever_off_loss_with_other_speed_equal"] = float(w0(adapt, act, speed=spd)["pred/L1"]) == float(w0(adapt, act, speed=spd.flip(0))["pred/L1"])
    r["PASS"] = (r["grad_abs_sum_speed_column"] > 0 and r["control_zero_speed_column_grad"] == 0.0 and r["control_levels_2_3_grad"] == [0.0, 0.0]
                 and abs(base - r["pred_L1"]["speed_flipped_over_batch"]) > 1e-6 and abs(base - r["pred_L1"]["speed_shuffled_over_frames"]) > 1e-6 and r["lever_off_loss_with_other_speed_equal"])
    res["t2"][pred] = r
    print("t2", pred, json.dumps(r))

# ---------------------------------------------------------------- t3 FUTTOK reach
res["t3"] = {}
for pred in ("flat", "cell"):
    cfg, w = wm(pred, fk=K, fw=1.0)
    adapt, act, spd, feats = inputs(cfg)
    w.train()
    w(adapt, act, feats=feats)["futtok/L1"].backward()                      # the term ALONE
    st = w.stack
    groups = {n: gsum(c.parameters()) for n, c in st.predictors[0].named_children()}
    groups.update({n: float(p.grad.abs().sum()) if p.grad is not None else 0.0 for n, p in st.predictors[0].named_parameters(recurse=False)})
    r = {"tok_head": gsum(w.tok_head.parameters()), "level1_predictor_groups": groups, "action_embedding": gsum(st.act_embed.parameters()), "state_projection": gsum(w.proj.parameters()),
         "levels_2_3_modules": [gsum(lvl(st, 2)), gsum(lvl(st, 3))]}
    w.zero_grad()
    w(adapt, act, feats=feats)["total"].backward()
    r["control_total_reaches_levels_2_3"] = [gsum(lvl(st, 2)), gsum(lvl(st, 3))]
    r["PASS"] = (r["tok_head"] > 0 and all(v > 0 for v in groups.values()) and r["action_embedding"] > 0 and r["state_projection"] > 0
                 and r["levels_2_3_modules"] == [0.0, 0.0] and all(v > 0 for v in r["control_total_reaches_levels_2_3"]))
    res["t3"][pred] = r
    print("t3", pred, json.dumps(r))


# ---------------------------------------------------------------- t4 the rollout inside the term vs HJEPAStack.rollout
class Rec(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head, self.seen = head, []

    def forward(self, x):
        self.seen.append(x.detach().clone())
        return self.head(x)


res["t4"] = {}
for pred in ("flat", "cell"):
    cfg, w = wm(pred, fk=K, fw=1.0)
    adapt, act, spd, feats = inputs(cfg)
    w.eval()
    rec = Rec(w.tok_head)
    z1 = w.z1(adapt)
    a = w.scale_actions(act)[:, :-1]
    with torch.no_grad():
        out = w.stack.loss(z1, a, tok_head=rec, tok_tgt=F.layer_norm(feats.float(), (cfg.d_feat,)))
        ctx = w.stack.predictors[0].context
        starts = list(range(0, 16 - K, 2))
        diffs = []
        for x, s in zip(rec.seen, starts):
            ref = w.stack.rollout(1, z1[:, s], a[:, s:s + K], context=ctx)
            diffs.append(float((x.reshape(B, K, -1) - ref).abs().max()))
    r = {"context": ctx, "starts": starts, "max_abs_diff_per_start": diffs, "max_abs_diff": max(diffs), "bar": 1e-6,
         "control_rollout_vs_true_future_states": float((rec.seen[0].reshape(B, K, -1) - z1[:, 1:K + 1]).abs().max())}
    r["PASS"] = r["max_abs_diff"] <= 1e-6 and len(rec.seen) == len(starts) and r["control_rollout_vs_true_future_states"] > 1e-3
    res["t4"][pred] = r
    print("t4", pred, json.dumps(r))


# ---------------------------------------------------------------- t5 total moves by W x term; mutation goes red
def total_delta(Hm, kind):
    Kk, Wt = 6, 0.7
    d1 = 640
    mk = lambda wgt: Hm.HJEPAStack([Hm.LevelConfig(latent_dim=d1, action_dim=2, act_embed_dim=32, pred_width=128 if kind == "cell" else 256, pred_depth=2, pred_heads=4, enc_hidden=128, macro_hidden=32,
                                                   futtok_k=Kk, futtok_weight=wgt, **(dict(pred_kind="cell", pred_cells=40, pred_cell_dim=16) if kind == "cell" else {})),
                                    Hm.LevelConfig(latent_dim=128, action_dim=32, stride=2, pred_width=128, pred_depth=2, pred_heads=4, enc_hidden=128, macro_hidden=32),
                                    Hm.LevelConfig(latent_dim=128, action_dim=32, stride=2, pred_width=128, pred_depth=2, pred_heads=4, enc_hidden=128, macro_hidden=32)],
                                   sigreg_projections=64).eval().double()
    torch.manual_seed(0)
    s1 = mk(Wt)
    torch.manual_seed(0)
    s0 = mk(Wt)
    s0.cfg[0].futtok_weight = 0.0
    g = torch.Generator().manual_seed(1)
    z, a = torch.randn(2, 13, d1, generator=g).double(), torch.randn(2, 12, 2, generator=g).double()
    tgt = F.layer_norm(torch.randn(2, 13, 40, 64, generator=g).double(), (64,))
    torch.manual_seed(5)
    head = torch.nn.Linear(16, 64).double()
    o1 = s1.loss(z, a, generator=torch.Generator().manual_seed(9), tok_head=head, tok_tgt=tgt)
    o0 = s0.loss(z, a, generator=torch.Generator().manual_seed(9), tok_head=head, tok_tgt=tgt)
    delta, term = float(o1["total"] - o0["total"]), float(o1["futtok/L1"])
    err = abs(delta - Wt * term)
    assert term > 1e-3 and err < 1e-9, f"total(W) - total(0) = {delta!r} but W x futtok/L1 = {Wt * term!r}"
    return {"delta_total": delta, "W_x_term": Wt * term, "abs_err": err}


res["t5"] = {}
src = inspect.getsource(H)
marker = "level_loss = level_loss + c.futtok_weight * futt"
lines = src.split("\n")
idx = [i for i, l in enumerate(lines) if l.strip().startswith(marker)]
assert len(idx) == 1
mut = types.ModuleType("hjepa_mutant_ft")
mut.__dict__["__file__"] = "hjepa_mutant_ft.py"
sys.modules["hjepa_mutant_ft"] = mut
exec(compile("\n".join(lines[:idx[0]] + lines[idx[0] + 1:]), "hjepa_mutant_ft.py", "exec"), mut.__dict__)
for kind in ("flat", "cell"):
    ok = total_delta(H, kind)
    try:
        total_delta(mut, kind)
        red = None
    except AssertionError as e:
        red = str(e)
    r = {"real": ok, "mutant_line_deleted": marker, "mutant_goes_red": red is not None, "mutant_assertion": red, "PASS": ok["abs_err"] < 1e-9 and red is not None}
    res["t5"][kind] = r
    print("t5", kind, json.dumps(r))
    print("  RED RUN (line deleted):", red)

res["PASS"] = all(v["PASS"] for t in ("t2", "t3", "t4", "t5") for v in res[t].values())
json.dump(res, open(OUT, "w", encoding="utf-8", newline="\n"), indent=1)
print("T2345", "PASS" if res["PASS"] else "FAIL")
