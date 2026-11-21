"""c2 reachability (+ mutation of the declared-vs-built check) and c3 location. Runs in tree_patch (PYTHONPATH set by the caller). CPU only."""
import json
import os
import sys
import tempfile
from pathlib import Path

TREE = "F:/TanitAD-scratch/v8proof/spatial/stage1/tree_patch"
import numpy as np
import torch
import tanitad

assert os.path.abspath(tanitad.__file__).lower().startswith(os.path.abspath(TREE).lower()), tanitad.__file__
sys.path.insert(0, TREE + "/stack/scripts")
import v8f_mini_train as TR
from tanitad.data import v8f_mini_data as D
from tanitad.models import v8f_mini as M

assert os.path.abspath(TR.__file__).lower().startswith(os.path.abspath(TREE).lower())
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)
OUT = {}

root = Path(tempfile.mkdtemp(prefix="c2c3_", dir="F:/TanitAD-scratch/v8proof/spatial/stage1/tmp"))
w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
J = D.load_join(root / "join_train.npz")
store = D.ClipStore(root, "train", lru=8)
wins = D.train_windows(store, J, w["names"]["train"])
ds = D.V8FMiniWindows(store, J, wins, hmax=16)
idx = np.random.default_rng(0).choice(len(wins), 8, replace=False)
batch = D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx])
mu, sd = M.cons_stats(J)


def args_for(extra):
    return TR.build_parser().parse_args(["--out", "x", "--d", "32", "--n-q", "2", "--arm", "M2b", *extra])


def build(extra, **over):
    a = args_for(extra)
    cfg0 = TR.make_config(a)
    keep = {k: v for k, v in TINY.items() if k not in ("nb", "n_q", "d", "ff")}
    cfg = M.V8FMiniConfig(**{**cfg0.__dict__, **keep, "d": 32, "ff": 64, **over})
    torch.manual_seed(0)
    model = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    info = {"n_clips": 6, "n_clips_expected": 6, "batch": a.batch, "md5_overridden": True, "kin_mode": a.kin_mode, "p10_units": None}
    return a, cfg, model, opt, info


def gsum(params):
    return float(sum(float(p.grad.abs().sum()) for p in params if p.grad is not None))


# ---------------------------------------------------------------- c2 reachability: the WM term alone
def wm_term_grads(extra, w_wm):
    a, cfg, model, opt, info = build(extra, w_wm=w_wm)
    model.train()
    model.zero_grad(set_to_none=True)
    out = model.loss(dict(batch), p_gt=0.5, gen=torch.Generator().manual_seed(11))
    T = out["terms"]
    (cfg.w_wm * T["wm"]).backward()                       # ONLY the WM term of the total (the total adds w_wm x T["wm"])
    return {"wm_term": float(T["wm"]), "grad_proj": gsum(model.wm.proj.parameters()),
            "grad_adapter": gsum(list(model.adapter.parameters())),
            "grad_wm_stack": gsum(model.wm.stack.parameters()), "grad_read_s": gsum(model.read_s.parameters()),
            "proj_shapes": [list(p.shape) for p in model.wm.proj.parameters()]}


c2 = {"grid_w_wm_0.1": wm_term_grads(["--wm-state", "grid"], 0.1), "grid_w_wm_0": wm_term_grads(["--wm-state", "grid"], 0.0),
      "mean_w_wm_0.1": wm_term_grads([], 0.1), "mean640_w_wm_0.1": wm_term_grads(["--wm-dim", "640"], 0.1)}
c2["PASS_reach"] = bool(c2["grid_w_wm_0.1"]["grad_proj"] > 0 and c2["grid_w_wm_0.1"]["grad_adapter"] > 0)
c2["PASS_zero_control"] = bool(c2["grid_w_wm_0"]["grad_proj"] == 0.0 and c2["grid_w_wm_0"]["grad_adapter"] == 0.0)
c2["control_read_s_is_zero_from_wm_term"] = c2["grid_w_wm_0.1"]["grad_read_s"] == 0.0         # the WM term does not reach S's reader: a second exact zero

# ---------------------------------------------------------------- c2 mutation: the declared-vs-built check goes red when z1 ignores the flag
a, cfg, model, opt, info = build(["--wm-state", "grid"])
green0 = TR.declared_vs_built(a, cfg, model, opt, info)
orig_z1 = M.WMAux.z1
M.WMAux.z1 = lambda self, adapt: self.proj(adapt.mean(2))                 # the mutation: z1 ignores the flag and returns the mean path
red = TR.declared_vs_built(a, cfg, model, opt, info)
M.WMAux.z1 = orig_z1
green1 = TR.declared_vs_built(a, cfg, model, opt, info)
# a declared grid on a BUILT mean model, and the reverse, and a width / cell-dim lie
am, cm, mm, om, im = build([])                                              # built mean
red_decl_grid_built_mean = TR.declared_vs_built(args_for(["--wm-state", "grid"]), cm, mm, om, im)
ag, cg, mg, og, ig = build(["--wm-state", "grid"])
red_decl_mean_built_grid = TR.declared_vs_built(args_for([]), cg, mg, og, ig)
red_cell = TR.declared_vs_built(args_for(["--wm-state", "grid", "--wm-cell-dim", "8"]), cg, mg, og, ig)
am6, cm6, mm6, om6, im6 = build(["--wm-dim", "640"])
green_w = TR.declared_vs_built(am6, cm6, mm6, om6, im6)
red_w = TR.declared_vs_built(args_for([]), cm6, mm6, om6, im6)
try:
    build(["--wm", "off", "--wm-state", "grid"])
    off_refused = False
except ValueError as e:
    off_refused = str(e)
a_off, c_off, m_off, o_off, i_off = build(["--arm", "M1"])
red_off = TR.declared_vs_built(args_for(["--arm", "M1", "--wm-state", "grid"]), c_off, m_off, o_off, i_off) if False else None
c2["mutation"] = {"green_before": green0, "red_when_z1_ignores_flag": red, "green_after_restore": green1,
                  "red_declared_grid_built_mean": [m for m in red_decl_grid_built_mean if "wm" in m],
                  "red_declared_mean_built_grid": [m for m in red_decl_mean_built_grid if "wm" in m],
                  "red_declared_cell_8_built_16": [m for m in red_cell if "wm" in m],
                  "green_mean640": green_w, "red_declared_mean128_built_mean640": [m for m in red_w if "wm" in m],
                  "config_refuses_grid_with_wm_off": off_refused}
c2["PASS_mutation"] = bool(green0 == [] and any("wm_state" in m for m in red) and green1 == [] and red_decl_grid_built_mean and red_decl_mean_built_grid
                           and red_cell and green_w == [] and red_w and off_refused)
OUT["c2"] = c2

# ---------------------------------------------------------------- c3 location
def z1_after_permuting_frame(extra, frame=2, seed=5):
    a, cfg, model, opt, info = build(extra)
    model.eval()
    feats = batch["feats"].float()                                          # [B, F, n_tok, d_feat]
    perm = torch.randperm(cfg.n_tok, generator=torch.Generator().manual_seed(seed))
    assert not torch.equal(perm, torch.arange(cfg.n_tok))
    f2 = feats.clone()
    f2[:, frame] = feats[:, frame][:, perm]                                 # the cells of ONE frame, permuted
    with torch.no_grad():
        tok0, tok1 = model.adapter(feats), model.adapter(f2)                # tokens BEFORE the position term (the adapter is per token)
        e0, e1 = model._encode({"feats": feats}), model._encode({"feats": f2})      # AFTER: adapter + self.pos (the real encode)
        z0, z1 = model.wm.z1(e0), model.wm.z1(e1)
        mean_before_pos0, mean_before_pos1 = tok0.mean(2), tok1.mean(2)
        # the permutation done on the adapt tensor itself (the input of WMAux.z1)
        e2 = e0.clone()
        e2[:, frame] = e0[:, frame][:, perm]
        z2 = model.wm.z1(e2)
    d = lambda x, y: float((x - y).abs().max())
    n = lambda x, y: float((x - y).norm())
    other = [i for i in range(feats.shape[1]) if i != frame]
    return {"state_width": int(z0.shape[-1]),
            "tokens_before_pos_mean_maxabs_diff_frame": d(mean_before_pos0[:, frame], mean_before_pos1[:, frame]),
            "z1_maxabs_diff_permuted_frame_features_path": d(z0[:, frame], z1[:, frame]), "z1_norm_diff_permuted_frame_features_path": n(z0[:, frame], z1[:, frame]),
            "z1_maxabs_diff_permuted_frame_adapt_path": d(z0[:, frame], z2[:, frame]), "z1_norm_diff_adapt_path": n(z0[:, frame], z2[:, frame]),
            "z1_maxabs_diff_other_frames": max(d(z0[:, i], z1[:, i]) for i in other), "z1_norm_of_state_frame": float(z0[:, frame].norm())}


c3 = {"mean": z1_after_permuting_frame([]), "grid": z1_after_permuting_frame(["--wm-state", "grid"]), "mean640": z1_after_permuting_frame(["--wm-dim", "640"])}
c3["PASS"] = bool(c3["mean"]["tokens_before_pos_mean_maxabs_diff_frame"] < 1e-5 and c3["mean"]["z1_maxabs_diff_permuted_frame_features_path"] < 1e-5
                  and c3["mean640"]["z1_maxabs_diff_permuted_frame_features_path"] < 1e-5
                  and c3["grid"]["z1_maxabs_diff_permuted_frame_features_path"] > 1e-3 and c3["grid"]["z1_maxabs_diff_permuted_frame_adapt_path"] > 1e-3
                  and c3["grid"]["z1_maxabs_diff_other_frames"] == 0.0)
c3["note"] = ("permuted: the n_tok = 40 cells of frame 2 of the feature tensor [B, F, n_tok, d_feat] (one fixed random permutation, same for every window), either on the features "
              "(the adapter is per token, so this permutes the tokens before the position term) or on the adapt tensor after _encode (adapter + self.pos). "
              "self.pos enters in V8FMini._encode: adapt = adapter(feats) + pos[None, None]; a sum is permutation invariant, so the MEAN state cannot see the permutation on either path; the grid state concatenates per cell and does.")
OUT["c3"] = c3
json.dump(OUT, open("F:/TanitAD-scratch/v8proof/spatial/stage1/checks/c2_c3.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(OUT, indent=1)[:6000])
print("C2 reach", c2["PASS_reach"], "zero", c2["PASS_zero_control"], "mutation", c2["PASS_mutation"], "C3", c3["PASS"])
