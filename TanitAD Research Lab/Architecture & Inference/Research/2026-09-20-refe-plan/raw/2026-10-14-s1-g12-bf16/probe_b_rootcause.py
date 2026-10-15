"""Root-cause probe for G12's `feed_probe_reads_difference: false` on the pod (CUDA + bf16 autocast).

Real modules: refe/aux_drivable_head.AuxDrivable (feed on, zero-init), the REAL trained scorer (score_q_mlp / score_dec / score_head of
D:/Projects/TanitAD-artifacts/refe-final-2026-10-01/model_final.pt, 6 outputs), run on this machine's GPU under torch.autocast("cuda", bfloat16)
-- the pod's numeric path.  Inputs: smooth forward proposals (K=64) and a random visual context (the verdict does not depend on its scale: swept).

Run: PYTHONIOENCODING=utf-8 python raw/2026-10-14-s1-g12-bf16/probe_b_rootcause.py   -> probe_b_rootcause.json beside it.
"""
import json
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "refe")))
from model import REFe, REFeConfig, CrossBlock  # noqa: E402
import aux_drivable_head as AUX  # noqa: E402

assert torch.cuda.is_available(), "this probe is about the CUDA bf16 path"
dev = "cuda"
OUT = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
CKPT = "D:/Projects/TanitAD-artifacts/refe-final-2026-10-01/model_final.pt"


class ScorerCopy(nn.Module):             # the shape of scorer_finetune.ScorerCopy: three modules, forward = REFe.score_trajectories itself
    def __init__(self, sd):
        super().__init__()
        c = REFeConfig()
        self.score_dec = nn.ModuleList([CrossBlock(c.dec_width, c.dec_heads, c.dec_width) for _ in range(c.dec_depth)])
        self.score_q_mlp = nn.Sequential(nn.Linear(60, 256), nn.GELU(), nn.Linear(256, 256))
        self.score_head = nn.Linear(256, 6)
        own = {k: v for k, v in sd.items() if k.split(".")[0] in ("score_q_mlp", "score_dec", "score_head")}
        self.load_state_dict(own, strict=True)

    def forward(self, trajs, sctx):
        return REFe.score_trajectories(self, trajs, sctx)


sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)["model"]
armB = ScorerCopy(sd).to(dev)
for p in armB.parameters():
    p.requires_grad_(True)                    # as in the trainer: the arm is trainable (autocast caches the cast of a requires_grad leaf)


def make_inputs(B=2, K=64, N=2048, ctx_scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(1, 21, dtype=torch.float32) * 0.2
    v = 6.0 + 8.0 * torch.rand(B, K, 1, generator=g)
    lat = (torch.rand(B, K, 1, generator=g) - 0.5) * 0.6
    x = (v * t).expand(B, K, 20)
    y = (lat * t ** 2 / 2.0).expand(B, K, 20)
    yaw = torch.atan2(lat * t, v).expand(B, K, 20)
    trajs = torch.stack([x, y, yaw], -1).to(dev)
    sctx = (torch.randn(B, N, 256, generator=g) * ctx_scale).to(dev)
    return trajs, sctx


def fresh_aux(seed=0):
    return AUX.AuxDrivable(AUX.AuxDrivableConfig(feed="on", w_aux=0.1, seed=seed)).to(dev)


def old_probe(aux, trajs, sctx, clear_after_copy=False):
    """VERBATIM the pre-fix block of g12_probe (scorer_finetune.py 1056-1064), plus one switch that is NOT in the old code."""
    g = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lg0, sd0 = aux.predict(sctx)
        s_plain = armB(trajs, sctx)
        g["feed_zero_init"] = bool(float(aux.feed.proj.weight.abs().max()) == 0.0)
        g["feed_step0_equal_base"] = bool(torch.equal(aux.score(armB, trajs, sctx, sd0), s_plain))
        keep_w = aux.feed.proj.weight.detach().clone()
        aux.feed.proj.weight.copy_(torch.full_like(keep_w, 1e-2))
        if clear_after_copy:
            torch.clear_autocast_cache()
        s_fed = aux.score(armB, trajs, sctx, sd0)
        g["feed_probe_reads_difference"] = bool(not torch.equal(s_fed, s_plain))
        g["max_abs_diff"] = float((s_fed.float() - s_plain.float()).abs().max())
        aux.feed.proj.weight.copy_(keep_w)
    return g


res = {}
trajs, sctx = make_inputs()
aux = fresh_aux()

# ---- E1: reproduce on CUDA bf16 with the old code, at several context scales -------------------------------------
res["E1_old_probe_cuda_bf16"] = {}
for s in (0.05, 1.0, 5.0):
    tj, sc = make_inputs(ctx_scale=s)
    res["E1_old_probe_cuda_bf16"][f"ctx_scale={s}"] = old_probe(fresh_aux(), tj, sc)

# ---- E2: the feed's inputs at step 0 on this path (cause b) ------------------------------------------------------
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    lg0, sd0 = aux.predict(sctx)
feats, oob = AUX.feed_scalars(sd0, trajs)
res["E2_step0_head_and_feed_scalars"] = {
    "sdf_pred_dtype": str(sd0.dtype),
    "sdf_pred_min_mean_max_std": [float(sd0.float().min()), float(sd0.float().mean()), float(sd0.float().max()), float(sd0.float().std())],
    "feat_names": ["min_corner_sdf_m", "mean_centre_sdf_m", "share_corner_lt0"],
    "feat_min": feats.amin((0, 1)).tolist(), "feat_max": feats.amax((0, 1)).tolist(), "feat_mean": feats.mean((0, 1)).tolist(),
    "feat_std_over_proposals": feats.std(1).mean(0).tolist(),
    "oob_share_mean": float(oob.mean()), "n_distinct_feat_rows": int(torch.unique(feats.reshape(-1, 3), dim=0).shape[0]), "n_rows": int(feats.reshape(-1, 3).shape[0])}

# ---- E3: magnitudes: the feed's contribution vs the base layer's bf16 resolution ----------------------------------
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    base_h = armB.score_q_mlp[0](trajs.flatten(2)).float()
contrib = (1e-2 * feats).sum(-1, keepdim=True)          # each of the 256 outputs receives 1e-2 * (f0 + f1 + f2) when the weight is the constant 1e-2
ulp = base_h.abs().clamp_min(1e-30).log2().floor().exp2() * 2 ** -7        # bf16 spacing at the base magnitude (8 significand bits)
res["E3_magnitudes"] = {"base_h_abs_median": float(base_h.abs().median()), "base_h_abs_p95": float(base_h.abs().quantile(0.95)),
                        "contrib_abs_median": float(contrib.abs().median()), "contrib_abs_max": float(contrib.abs().max()),
                        "bf16_half_ulp_at_base_median": float(ulp.median() / 2),
                        "share_of_elements_with_contrib_gt_half_ulp": float((contrib.abs().expand_as(base_h) > ulp / 2).float().mean())}

# ---- E4: the cause (c): the autocast weight-cast cache -----------------------------------------------------------
aux = fresh_aux()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    lg0, sd0 = aux.predict(sctx)
    f0 = aux.feed(sd0, trajs).float()
    keep = aux.feed.proj.weight.detach().clone()
    aux.feed.proj.weight.copy_(torch.full_like(keep, 1e-2))
    f1_stale = aux.feed(sd0, trajs).float()                    # same autocast region, the fp32 weight is now 1e-2
    torch.clear_autocast_cache()
    f1_cleared = aux.feed(sd0, trajs).float()
    aux.feed.proj.weight.copy_(keep)
res["E4_cache"] = {"feed_out_abs_max_w0": float(f0.abs().max()),
                   "feed_out_abs_max_after_copy_same_region_STALE": float(f1_stale.abs().max()),
                   "feed_out_abs_max_after_clear_autocast_cache": float(f1_cleared.abs().max())}

# ---- E5: what changes the verdict ---------------------------------------------------------------------------------
r = {}
r["old_probe_cuda_bf16"] = old_probe(fresh_aux(), trajs, sctx)
r["old_probe + torch.clear_autocast_cache() after copy_"] = old_probe(fresh_aux(), trajs, sctx, clear_after_copy=True)
a_ = fresh_aux()
with torch.no_grad():
    with torch.autocast("cuda", dtype=torch.bfloat16):                      # the cache is cleared when the OUTERMOST autocast region exits
        lg0, sd0 = a_.predict(sctx)
        s_plain = armB(trajs, sctx)
        eq0 = bool(torch.equal(a_.score(armB, trajs, sctx, sd0), s_plain))
    keep = a_.feed.proj.weight.detach().clone()
    a_.feed.proj.weight.copy_(torch.full_like(keep, 1e-2))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        s_fed = a_.score(armB, trajs, sctx, sd0)
    a_.feed.proj.weight.copy_(keep)
r["fresh region per call (bf16), w=1e-2"] = {"step0_equal": eq0, "reads_difference": bool(not torch.equal(s_fed, s_plain)),
                                             "max_abs_diff": float((s_fed.float() - s_plain.float()).abs().max()),
                                             "n_elems_differ": int((s_fed != s_plain).sum()), "n_elems": int(s_fed.numel())}
a_ = fresh_aux()
with torch.no_grad():
    lg0, sd0 = a_.predict(sctx)
    s_plain = armB(trajs, sctx)
    eq0 = bool(torch.equal(a_.score(armB, trajs, sctx, sd0), s_plain))
    keep = a_.feed.proj.weight.detach().clone()
    a_.feed.proj.weight.copy_(torch.full_like(keep, 1e-2))
    s_fed = a_.score(armB, trajs, sctx, sd0)
    a_.feed.proj.weight.copy_(keep)
r["fp32 no autocast, w=1e-2"] = {"step0_equal": eq0, "reads_difference": bool(not torch.equal(s_fed, s_plain)), "max_abs_diff": float((s_fed - s_plain).abs().max())}
res["E5_variants"] = r

# ---- E6: smallest weight constant that survives bf16 (fresh region per call) -------------------------------------
sweep = {}
a_ = fresh_aux()
with torch.no_grad():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg0, sd0 = a_.predict(sctx)
        s_plain = armB(trajs, sctx)
    keep = a_.feed.proj.weight.detach().clone()
    for w in (1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 1e-1):
        a_.feed.proj.weight.copy_(torch.full_like(keep, w))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s_fed = a_.score(armB, trajs, sctx, sd0)
        sweep[f"{w:g}"] = {"differs": bool(not torch.equal(s_fed, s_plain)), "n_elems_differ": int((s_fed != s_plain).sum()), "of": int(s_fed.numel()),
                           "max_abs_diff": float((s_fed.float() - s_plain.float()).abs().max())}
    a_.feed.proj.weight.copy_(keep)
res["E6_weight_sweep_bf16_fresh_region"] = sweep

OUT["results"] = res
p = os.path.join(HERE, "probe_b_rootcause.json")
json.dump(OUT, open(p, "w"), indent=1)
print(json.dumps(OUT, indent=1))
