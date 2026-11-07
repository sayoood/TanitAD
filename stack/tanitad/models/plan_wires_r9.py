"""refcv9 C8 -- the three PLANNER WIRES (REFCV9_DESIGN 2.7; PI 10-25: "the planner must use the map head"). Pure tensor code.

  W-MAP-1  ``critic_feature_banded``  the drivable critic's INPUT read from the BANDED channels (P(drv), drv_sdf) at the footprint
           corners instead of the 8-way softmax sum; a corner on an ``unobs`` cell (or outside the 100 m x +-30 m extent) reads
           "no evidence" exactly as today's out-of-range corner does: (1, 1, 0) -- and (+1, +1) on the two sdf features.
  W-MAP-2  ``channels_1m`` + ``MapChanProj``  drv / drv_sdf / sdw / lane / unobs, DETACHED, resampled to 1 m over the FULL 100 m x +-30 m
           (observed-weighted means; ``unobs`` = the unobserved fraction), then a ZERO-INIT 1x1 projection added to the planner BEV pool
           on the pool's OWN window (a 1x1 over concat([pool, map]) whose map-channel weights are zero == pool + proj(map)).
  W-BOX    ``BoxTokens`` + ``gated_mix``  top-32 box3d slots by presence -> tokens (slot embedding + MLP of x, y, l, w, yaw, v_rel, class),
           appended to the agent-token set of the planner AND the tactical decoder behind a ZERO-INIT gate. ``gated_mix`` makes the gate
           EXACT: out = attend(agents) + g * (attend(agents + box) - attend(agents)); g = 0 returns the agents-only output bit-for-bit
           (extra keys in one softmax would move the attention mass even with zero-valued tokens), and dL/dg is non-zero at step 0.

Every input is camera-derived (``out["perception"]["map_banded"]`` / ``["box_slots"]``); nothing here sees a tactical logit.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tanitad.models import map_banded as MB

DRV_FEATS_BANDED = 5            # (min P(drv), mean P(drv), share with evidence, min sdf / clip, mean sdf / clip)
NO_EVIDENCE = (1.0, 1.0, 0.0, 1.0, 1.0)
N_MAP_CHAN = 5                  # drv, drv_sdf, sdw, lane, unobs
BOX_K = 32
_I = {c: i for i, c in enumerate(MB.CHANNELS)}


def _evidence_mask(inside: Tensor, unobs: Tensor) -> Tensor:
    """A corner carries evidence iff it lies inside the band AND the lift observed its cell."""
    return inside & ~unobs


def critic_feature_banded(paths: Tensor, mb: dict, y_half_m: float | None = None) -> Tensor:
    """``paths`` [B, N, S, 2] (the candidate plan, rear-axle points) x the banded head's output ``mb`` ({"logits": [near, mid, far]
    [B, 10, rows, cols], "unobs": [..] [B, rows, cols] bool}) -> [B, N, 5]. DETACHED (no gradient into the map head)."""
    from tanitad.refs import refcv8_conditioning as r8c
    with torch.no_grad():
        b, n, s, _ = paths.shape
        pts = r8c.footprint_corners(paths.detach()).reshape(b, n * s * 4, 2)
        p_all = torch.ones(b, n * s * 4, device=pts.device)
        d_all = torch.ones_like(p_all)
        ev_all = torch.zeros_like(p_all, dtype=torch.bool)
        geoms = _geoms(mb)
        bidx = torch.arange(b, device=pts.device)[:, None]
        for (x0, x1, cell, rows, cols), z, un in zip(geoms, mb["logits"], mb["unobs"]):
            yh = cols * cell / 2.0
            x, y = pts[..., 0], pts[..., 1]
            inside = (x >= x0) & (x < x1) & (y > -yh) & (y < yh)
            ix = ((x - x0) / cell).floor().long().clamp(0, rows - 1)
            iy = ((y + yh) / cell).floor().long().clamp(0, cols - 1)
            zz = z.detach().permute(0, 2, 3, 1)[bidx, ix, iy].to(torch.float32)          # [B, P, 10]
            ev = _evidence_mask(inside, un.detach()[bidx, ix, iy])
            p_all = torch.where(ev, torch.sigmoid(zz[..., _I["drv"]]), p_all)
            d_all = torch.where(ev, zz[..., _I["drv_sdf"]].clamp(-MB.SDF_CLIP_M, MB.SDF_CLIP_M) / MB.SDF_CLIP_M, d_all)
            ev_all = ev_all | ev
        ev3, p3, d3 = (t.reshape(b, n, -1) for t in (ev_all, p_all, d_all))
        cnt = ev3.to(torch.float32).sum(-1)
        one = torch.ones_like(cnt)
        pmean = torch.where(cnt > 0, (p3 * ev3).sum(-1) / cnt.clamp_min(1.0), one)
        dmean = torch.where(cnt > 0, (d3 * ev3).sum(-1) / cnt.clamp_min(1.0), one)
        return torch.stack([p3.amin(-1), pmean, cnt / float(ev3.shape[-1]), d3.amin(-1), dmean], -1)


def _geoms(mb: dict) -> list:
    """(x0, x1, cell, rows, cols) per band, read OFF the logits' shapes (never a literal extent or a literal band start): the near band
    starts at 0 and every next band starts where the previous one ENDS, so a near band that ends at 20 m (E1-NEAR20) moves the mid band
    to 20-40 m with no constant to keep in step. At the default layout the tuples are the old ones (0/10/40 m)."""
    out, x0 = [], float(MB.BAND_START_M[0])
    for k, z in enumerate(mb["logits"]):
        cell = MB.BAND_CELL_M[k]
        rows, cols = int(z.shape[-2]), int(z.shape[-1])
        out.append((x0, x0 + rows * cell, cell, rows, cols))
        x0 = x0 + rows * cell
    return out


def channels_1m(mb: dict) -> Tensor:
    """The banded head's decision channels at 1 m over the FULL extent: [B, 5, X, Y] = drv, drv_sdf / clip, sdw, lane, unobs.
    DETACHED. Observed-weighted means (an unobserved sub-cell never contaminates a value); a 1 m cell with no observed sub-cell
    reads 0 on the four value channels and 1 on ``unobs``."""
    with torch.no_grad():
        rows_out = []
        for k, (z, un) in enumerate(zip(mb["logits"], mb["unobs"])):
            z, obs = z.detach().to(torch.float32), (~un.detach()).to(torch.float32)[:, None]
            f = int(round(1.0 / MB.BAND_CELL_M[k]))
            val = torch.stack([torch.sigmoid(z[:, _I["drv"]]),
                               z[:, _I["drv_sdf"]].clamp(-MB.SDF_CLIP_M, MB.SDF_CLIP_M) / MB.SDF_CLIP_M,
                               torch.sigmoid(z[:, _I["sdw"]]), torch.sigmoid(z[:, _I["lane"]])], 1)
            num, den = F.avg_pool2d(val * obs, f), F.avg_pool2d(obs, f)
            v = torch.where(den > 0, num / den.clamp_min(1e-6), torch.zeros_like(num))
            rows_out.append(torch.cat([v, 1.0 - den], 1))
        return torch.cat(rows_out, 2)


class MapChanProj(nn.Module):
    """ZERO-INIT 1x1 over the 5 map channels at 1 m, windowed onto the planner pool's grid and ADDED to its output."""

    def __init__(self, d_out: int, dst_grid):
        super().__init__()
        self.proj = nn.Conv2d(N_MAP_CHAN, int(d_out), 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.dst = (float(dst_grid.x_fwd_m), float(dst_grid.y_half_m), float(dst_grid.cell_m))
        k = 1.0 / self.dst[2]
        if abs(k - round(k)) > 1e-9 or round(k) < 1:
            raise ValueError(f"[plan-wires] the pool cell {self.dst[2]} m does not divide the 1 m channel grid")
        self.up = int(round(k))

    def forward(self, chan: Tensor) -> Tensor:
        """chan [B, 5, X, Y] (1 m, full extent) -> [B, d_out, X_dst / cell, Y_dst / cell]."""
        x_win, y_half, _ = self.dst
        y_src = chan.shape[-1] / 2.0
        r1, c0, c1 = int(round(x_win)), int(round(y_src - y_half)), int(round(y_src + y_half))
        if r1 > chan.shape[-2] or c0 < 0:
            raise ValueError(f"[plan-wires] the pool window {self.dst} lies outside the {tuple(chan.shape[-2:])} m channel field")
        y = self.proj(chan[:, :, :r1, c0:c1].to(self.proj.weight.dtype))
        if self.up > 1:
            y = y.repeat_interleave(self.up, 2).repeat_interleave(self.up, 3)
        return y


def box_features(slots: dict, k: int = BOX_K) -> tuple:
    """Top-``k`` slots by ``presence_logit`` (the plain presence; B3's quality-aware presence trains the SAME logit) -> (feat [B, k, F],
    presence [B, k], order [B, k]). DETACHED: the planner loss cannot bend the box head through the tokens (stamped)."""
    with torch.no_grad():
        pl = slots["presence_logit"].detach()
        kk = min(int(k), pl.shape[1])
        top = pl.topk(kk, dim=1).indices                                                   # [B, kk]
        g = lambda t: t.detach().gather(1, top[..., None].expand(-1, -1, t.shape[-1])) if t.dim() == 3 else t.detach().gather(1, top)
        box, yv, rates, cls_p = g(slots["box"]), g(slots["yaw_vec"]), g(slots["rates"]), g(torch.softmax(slots["cls_logits"], -1))
        pres = torch.sigmoid(g(pl))
        feat = torch.cat([box[..., :2] / torch.tensor([60.0, 16.0], device=box.device), box[..., 2:3] / 10.0, box[..., 3:4] / 4.0,
                          yv, rates[..., :2] / 10.0, cls_p, pres[..., None]], -1)
        return feat, pres, top


class BoxTokens(nn.Module):
    """slot-rank embedding + MLP of (x, y, l, w, sin, cos, v_rel_x, v_rel_y, class posterior, presence) -> [B, k, d_out], scaled by the
    presence (the AgentTokenEmbed convention). ``pad`` is all False (the 32 are structurally present)."""

    def __init__(self, d_out: int, n_cls: int, k: int = BOX_K):
        super().__init__()
        self.k, self.n_cls = int(k), int(n_cls)
        self.slot_emb = nn.Embedding(self.k, int(d_out))
        nn.init.normal_(self.slot_emb.weight, std=0.02)
        self.mlp = nn.Sequential(nn.Linear(9 + self.n_cls, int(d_out)), nn.GELU(), nn.Linear(int(d_out), int(d_out)))
        self.norm = nn.LayerNorm(int(d_out))

    def forward(self, slots: dict) -> tuple:
        feat, pres, _ = box_features(slots, self.k)
        tok = self.norm(self.mlp(feat) + self.slot_emb.weight[None, :feat.shape[1]]) * pres[..., None]
        return tok, torch.zeros(tok.shape[:2], dtype=torch.bool, device=tok.device)


def gated_mix(attend, tok: Tensor, pad: Tensor | None, n_box: int, gate: Tensor) -> Tensor:
    """``tok`` = [agents | box] along dim 1 (``n_box`` trailing box tokens). out = attend(agents) + g * (attend(all) - attend(agents));
    g == 0 is the agents-only output bit-for-bit (both branches are finite), and d out / d g != 0 at g = 0."""
    if n_box <= 0:
        return attend(tok, pad)
    a_only = attend(tok[:, :-n_box], None if pad is None else pad[:, :-n_box])
    full = attend(tok, pad)
    return a_only + gate * (full - a_only)


# ---- the per-forward side channel for W-BOX: how many trailing tokens of the agent set are box tokens (0 = none). Set by
# RefCModel.forward (reset at its start, so a stale value never outlives a forward); read by the two gated consumers.
_CTX = {"n_box": 0}


def set_n_box(n: int) -> None:
    _CTX["n_box"] = int(n)


def n_box() -> int:
    return int(_CTX["n_box"])
