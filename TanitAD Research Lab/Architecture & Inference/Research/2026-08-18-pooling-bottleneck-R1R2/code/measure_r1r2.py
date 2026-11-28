"""Measure R1/R2 parameter cost by BUILDING the modules. CPU only, zero GPU.

Also re-derives the pooling geometry from the live V6Config defaults so the
spec's numbers come from source, not from prose.
"""
import json
import sys

import torch
from torch import nn

sys.path.insert(0, r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\stack")

from tanitad.models.v6 import (V6Config, MODULE_GROUPS, STAGE_GROUPS,   # noqa: E402
                               LADDER_UNTRAINED_GROUPS, PARAM_BUDGET,
                               MaskedCellPredictor)
from tanitad.models.readout import SpatialGridReadout                    # noqa: E402

out = {}

# ---------------------------------------------------------------- geometry ---
cfg = V6Config()
enc = cfg.encoder
th = enc.image_size // enc.patch_size
tw = enc.image_width // enc.patch_size
gw = cfg.readout.grid if cfg.readout.grid_w is None else cfg.readout.grid_w
ro = SpatialGridReadout(n_tokens=th * tw, d_model=enc.d_model,
                        grid=cfg.readout.grid, d_readout=cfg.readout.d_readout,
                        token_grid=(th, tw), grid_w=cfg.readout.grid_w)
out["geometry"] = {
    "image": [enc.image_size, enc.image_width], "patch": enc.patch_size,
    "d_model": enc.d_model, "token_grid": [th, tw], "n_tokens": th * tw,
    "readout_grid": [cfg.readout.grid, gw], "d_readout": cfg.readout.d_readout,
    "exact_pool": bool(ro.exact_pool),
    "pool_repr": repr(ro.pool),
    "pool_kernel": [th // cfg.readout.grid, tw // gw],
    "tokens_per_cell": (th // cfg.readout.grid) * (tw // gw),
    "px_per_cell": [(th // cfg.readout.grid) * enc.patch_size,
                    (tw // gw) * enc.patch_size],
    "n_cells": cfg.readout.grid * gw,
    "d_op": cfg.readout.grid * gw * cfg.readout.d_readout,
    "readout_params": sum(p.numel() for p in ro.parameters()),
    "pool_params": sum(p.numel() for p in ro.pool.parameters()),
    "param_budget": PARAM_BUDGET,
}

# --- baseline: the EXISTING O3 head (post-pool, self-predictive) --------------
o3 = MaskedCellPredictor(cfg.readout.grid * gw, cfg.readout.d_readout,
                         hidden=cfg.aux_hidden)
out["O3_existing_post_pool"] = {
    "params": sum(p.numel() for p in o3.parameters()),
    "hidden": cfg.aux_hidden, "n_units": cfg.readout.grid * gw,
    "d_unit": cfg.readout.d_readout,
}


# ------------------------------------------------------- R1 candidate module --
class TokenMaskedPredictor(nn.Module):
    """R1 — O3's idiom lifted from the 16 POOLED CELLS to the 640 PATCH TOKENS.

    Identical structure to MaskedCellPredictor, so the ONLY thing the experiment
    changes is WHERE the loss reads. Same masking, same stop-grad target, same
    transformer shape -- only n_units 16 -> 640 and d_unit 128 -> 384.
    """

    def __init__(self, n_tokens: int, d_tok: int, hidden: int = 256,
                 depth: int = 2, n_heads: int = 4):
        super().__init__()
        self.n_tokens, self.d_tok = int(n_tokens), int(d_tok)
        self.inp = nn.Linear(self.d_tok, hidden)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, hidden))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            hidden, n_heads, dim_feedforward=4 * hidden, batch_first=True,
            norm_first=True, activation="gelu", dropout=0.0)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth,
                                            enable_nested_tensor=False)
        self.out = nn.Linear(hidden, self.d_tok)

    def forward(self, tok, mask):
        h = self.inp(tok)
        h = torch.where(mask.unsqueeze(-1), self.mask_token.to(h.dtype), h)
        return self.out(self.blocks(h + self.pos.to(h.dtype)))


r1 = {}
for hidden, depth in ((192, 2), (256, 2), (256, 3), (384, 2)):
    m = TokenMaskedPredictor(th * tw, enc.d_model, hidden=hidden, depth=depth)
    per = {}
    for name, mod in (("inp", m.inp), ("pos", None), ("mask_token", None),
                      ("blocks", m.blocks), ("out", m.out)):
        if mod is None:
            per[name] = int(getattr(m, name).numel())
        else:
            per[name] = sum(p.numel() for p in mod.parameters())
    r1[f"hidden{hidden}_depth{depth}"] = {
        "total": sum(p.numel() for p in m.parameters()), "per_submodule": per}
out["R1_token_masked_predictor"] = r1

# smoke: does it actually run at the production token count?
m = TokenMaskedPredictor(th * tw, enc.d_model, hidden=256, depth=2)
tok = torch.randn(2, th * tw, enc.d_model)
msk = torch.zeros(2, th * tw, dtype=torch.bool)
msk[:, : (th * tw) // 4] = True
with torch.no_grad():
    y = m(tok, msk)
out["R1_smoke"] = {"in": list(tok.shape), "out": list(y.shape),
                   "ok": list(y.shape) == list(tok.shape)}


# ------------------------------------------------------- R2 candidate heads ---
class ResidualFlowHead(nn.Module):
    """R2 — per-UNIT regression onto the ego-compensated residual-flow target.

    `n_out=3`: (residual u, residual v, residual magnitude/confidence logit).
    Deliberately tiny: R2's content is the TARGET, not the head.
    """

    def __init__(self, d_unit: int, hidden: int = 0, n_out: int = 3):
        super().__init__()
        self.net = (nn.Linear(d_unit, n_out) if hidden == 0 else
                    nn.Sequential(nn.Linear(d_unit, hidden), nn.GELU(),
                                  nn.Linear(hidden, n_out)))

    def forward(self, x):
        return self.net(x)


r2 = {}
# R2-cells: post-pool placement (reads z_op cells, d_readout=128)
for hid in (0, 128, 256):
    m2 = ResidualFlowHead(cfg.readout.d_readout, hidden=hid)
    r2[f"cells_hidden{hid}"] = sum(p.numel() for p in m2.parameters())
# R2-tokens: pre-pool placement (reads patch tokens, d_model=384)
for hid in (0, 128, 256):
    m2 = ResidualFlowHead(enc.d_model, hidden=hid)
    r2[f"tokens_hidden{hid}"] = sum(p.numel() for p in m2.parameters())
out["R2_residual_flow_head"] = r2

m2 = ResidualFlowHead(enc.d_model, hidden=128)
with torch.no_grad():
    y2 = m2(torch.randn(2, th * tw, enc.d_model))
out["R2_smoke"] = {"out": list(y2.shape), "ok": list(y2.shape) == [2, th * tw, 3]}

# ------------------------------------------------- ladder / group bookkeeping --
out["ladder"] = {
    "MODULE_GROUPS": list(MODULE_GROUPS),
    "LADDER_UNTRAINED_GROUPS": sorted(LADDER_UNTRAINED_GROUPS),
    "STAGE_GROUPS": {k: list(v) for k, v in STAGE_GROUPS.items()},
    "S-J_is_MODULE_GROUPS_identity": STAGE_GROUPS["S-J"] is MODULE_GROUPS,
    "S-J_equals_module_groups_by_value": (
        tuple(STAGE_GROUPS["S-J"]) == tuple(MODULE_GROUPS)),
}

print(json.dumps(out, indent=2))
