"""refcv9 C1 -- the BANDED, CHANNELLED map head, its targets, its R7 losses and ``unobs`` (REFCV9_DESIGN 2.1 / 2.2).

Opt-in: ``refc_v3_train.py --r9-map-banded`` (with ``--map-hires on --map-hires-near-lift-m 10``). OFF = nothing here is built,
imported on the hot path or logged: the build, state_dict, forward and loss are the tip's.

THE REPRESENTATION (2.1) -- three bands over the SAME extent (100 m x +-30 m: 103,200 cells vs 600,000)::

    near  0-10 m  @ 0.10 m  100 x 600   the refcv8 10 cm decode (refine trunk + the near lift narrowed 20 -> 10 m)
    mid   10-40 m @ 0.25 m  120 x 240   the lift's native grid
    far   40-100 m @ 0.50 m 120 x 120   the lift's grid, 2 x 2 average-pooled

``CHANNELS`` per band: ``drv`` (sigmoid, codes 1 2 3 4 6) | ``drv_sdf`` (signed distance to the drv boundary, metres, clipped
+-2) | ``sdw`` (code 7) | ``lane`` (dilated heatmap, code 2) + ``lane_ox`` / ``lane_oy`` (vector, metres, from the cell centre to
the nearest GT lane position) | ``edge_h`` (dilated heatmap, code 5) | ``xwalk`` | ``arrow`` | ``hatch``.

``unobs`` = the lift's OWN per-frame validity (``~lift_valid``: geometric, vision-only, NOT learned), emitted WITH the map.

THE LOSS (2.2, R7): ``drv sdw xwalk arrow hatch`` sigmoid focal (gamma 2) + soft Dice; ``lane edge_h`` the same on the dilated
(~0.5 m wide) targets; ``drv_sdf`` / offsets L1 on cells within 2 m of a boundary / line. Each channel x band term is a mean over
the band's own supervised cells (``seen & ~unobs``); band weights 1 / 1 / 0.5; channels are SUMMED (each is an independent binary
task, so each keeps a CE-sized gradient -- a mean over channels would divide every channel's gradient by 10 and silently change
the map's share of the trunk against the C0 control); plus a max-entropy term at 1e-3 on ``unobs`` cells (PON eq. 2).

Everything below is pure tensor code: targets are built on the batch's device from the batch's existing 10 cm raster
(``batch["map_fine"]``, the 8-code GT), never from a second label store.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tanitad.data.semantic_map_gt_fine import FINE_CELL_M, FINE_CLASSES, NOT_SEEN_CODE

__all__ = ["FLAG", "DEST", "BAND_NAMES", "BAND_CELL_M", "BAND_WEIGHTS", "CHANNELS", "PROB_CHANNELS", "L1_CHANNELS",
           "DRV_CODES", "BandGeom", "band_geometry", "band_starts", "check_near_m", "NEAR_M_DEFAULT", "NEAR_M_ADMITTED",
           "BAND_START_M", "BandedMapHead", "unobs_masks", "build_banded_targets",
           "banded_map_loss", "signed_distance", "dilate_thin", "target_fractions", "assert_target_content",
           "telemetry_keys", "N_CHANNELS", "MAXENT_W", "FOCAL_GAMMA", "SDF_CLIP_M", "THIN_RADIUS_CELLS"]

FLAG = "--r9-map-banded"
DEST = "r9_map_banded"


def _code(name: str) -> int:
    return FINE_CLASSES.index(name)


C_DRIVABLE, C_LANE, C_XWALK = _code("drivable"), _code("lane / road line"), _code("crosswalk")
C_ARROW, C_EDGE, C_HATCH, C_SIDEWALK = _code("arrow / text"), _code("non-drivable edge"), _code("hatched area"), \
    _code("sidewalk / verge")
#: the drivable SURFACE: markings lie ON the road (the PI's MUST tier; refcv8_conditioning.DRIVABLE_CODES)
DRV_CODES = (C_DRIVABLE, C_LANE, C_XWALK, C_ARROW, C_HATCH)

BAND_NAMES = ("near", "mid", "far")
#: the DEFAULT layout (near 0-10 m): every band-layout consumer takes the near-band end from the CONFIG (``near_m``), this tuple is only
#: the default's value -- read it through :func:`band_starts`, never index it for a run's geometry (E1-NEAR20, PREREG_M1_NEAR20)
BAND_START_M = (0.0, 10.0, 40.0)
#: E1-NEAR20: the near band's end, 10 m (the default = M1 / M2) or 20 m (arm M1N20 = C0's near lift). ONE lever, a pre-registered set.
NEAR_M_DEFAULT = 10.0
NEAR_M_ADMITTED = (10.0, 20.0)
BAND_CELL_M = (0.10, 0.25, 0.50)
#: band weights near / mid / far (2.2): the far field, with the lowest image support, cannot dominate
BAND_WEIGHTS = (1.0, 1.0, 0.5)
CHANNELS = ("drv", "drv_sdf", "sdw", "lane", "lane_ox", "lane_oy", "edge_h", "xwalk", "arrow", "hatch")
N_CHANNELS = len(CHANNELS)
PROB_CHANNELS = ("drv", "sdw", "lane", "edge_h", "xwalk", "arrow", "hatch")
L1_CHANNELS = ("drv_sdf", "lane_ox", "lane_oy")
FOCAL_GAMMA = 2.0
MAXENT_W = 1e-3
SDF_CLIP_M = 2.0                  # |sdf| clip AND the L1 supervision radius for the SDF and the offsets
THIN_RADIUS_CELLS = 2             # Chebyshev radius at 10 cm: a 1-cell line -> (2 * 2 + 1) * 0.1 = 0.5 m wide
_DICE_EPS = 1.0
_INF = 1.0e4


class BandGeom(NamedTuple):
    name: str
    x0: float
    x1: float
    cell: float
    rows: int
    cols: int
    weight: float

    @property
    def n_cells(self) -> int:
        return self.rows * self.cols

    @property
    def fine_rows(self) -> tuple:
        return int(round(self.x0 / FINE_CELL_M)), int(round(self.x1 / FINE_CELL_M))


def check_near_m(near_m) -> float:
    """The near band's end, REFUSED unless it is one of the pre-registered ``NEAR_M_ADMITTED`` (10 = M1, 20 = M1N20)."""
    try:
        v = float(near_m)
    except (TypeError, ValueError):
        raise ValueError(f"[map-banded] near_m {near_m!r} is not a number") from None
    if not any(abs(v - a) < 1e-9 for a in NEAR_M_ADMITTED):
        raise ValueError(f"[map-banded] near_m {near_m!r} is not one of the admitted near-band ends {NEAR_M_ADMITTED} m")
    return v


def band_starts(near_m: float = NEAR_M_DEFAULT) -> tuple:
    """``(near.x0, mid.x0, far.x0)`` for a near band ending at ``near_m``: the default's ``BAND_START_M`` at 10 m, ``(0, 20, 40)`` at 20 m."""
    return (BAND_START_M[0], check_near_m(near_m), BAND_START_M[2])


def band_geometry(x_max_m: float, y_half_m: float, near_m: float = NEAR_M_DEFAULT) -> tuple:
    """The three bands over the declared extent, the NEAR band ending at ``near_m`` (default 10 m = M1; 20 m = M1N20: near 0-20 m
    @ 0.10 m = 200 x 600, mid 20-40 m @ 0.25 m = 80 x 240, far unchanged). REFUSES an extent the bands cannot tile (the far band
    must be a whole number of 0.5 m rows, the width a whole number of 0.5 m cells: the far band is the 0.25 m lift 2 x 2-pooled)."""
    x_max, yh = float(x_max_m), float(y_half_m)
    starts = band_starts(near_m)
    if not x_max > starts[2] + 1e-9:
        raise ValueError(f"[map-banded] x_max {x_max} m leaves no far band (needs > {starts[2]} m)")
    ends = (starts[1], starts[2], x_max)
    out = []
    for k in range(3):
        cell = BAND_CELL_M[k]
        rows, cols = (ends[k] - starts[k]) / cell, 2.0 * yh / cell
        if abs(rows - round(rows)) > 1e-6 or abs(cols - round(cols)) > 1e-6:
            raise ValueError(f"[map-banded] band {BAND_NAMES[k]} ({starts[k]}-{ends[k]} m, {cell} m cells) does not tile "
                             f"the extent {x_max} m x +-{yh} m")
        out.append(BandGeom(BAND_NAMES[k], starts[k], ends[k], cell, int(round(rows)), int(round(cols)),
                            float(BAND_WEIGHTS[k])))
    return tuple(out)


# =========================================================================== #
# the head                                                                    #
# =========================================================================== #
class _BandDecoder(nn.Module):
    """``[B, d_up, h, w] -> [B, 10, h, w]``: two 3x3 + GN + GELU and a 1x1 -- the refine trunk's recipe at the band's own grid."""

    def __init__(self, d_up: int, groups: int):
        super().__init__()
        self.conv1 = nn.Conv2d(d_up, d_up, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, d_up)
        self.conv2 = nn.Conv2d(d_up, d_up, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, d_up)
        self.head = nn.Conv2d(d_up, N_CHANNELS, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.gelu(self.norm1(self.conv1(x)))
        x = F.gelu(self.norm2(self.conv2(x)))
        return self.head(x)


class BandedMapHead(nn.Module):
    """The banded decoder. ⭐ It REUSES the refcv8 refine trunk for the NEAR band (``refine.inp`` -> bilinear x2.5 -> + near lift
    -> near refine -> ``refine.conv1/2``), so the warm start from refcv8 EMA-48,852 is meaningful there; it owns ``head_near``
    (the 10 channel 1x1) and the ``mid`` / ``far`` decoders (new, randomly initialised). ``refine.cls`` (the 8-way readout) is NOT
    built under the flag: nothing would train it. Built LAST by ``MapHiresBranch`` so every other parameter initialises as before."""

    def __init__(self, x_max_m: float, y_half_m: float, *, d_up: int, groups: int, near_m: float = NEAR_M_DEFAULT):
        super().__init__()
        self.near_m = check_near_m(near_m)                          # E1-NEAR20: the near band's end (10 = M1, 20 = M1N20)
        self.geom = band_geometry(x_max_m, y_half_m, self.near_m)
        self.d_up = int(d_up)
        self.head_near = nn.Conv2d(int(d_up), N_CHANNELS, 1)
        self.mid = _BandDecoder(int(d_up), int(groups))
        self.far = _BandDecoder(int(d_up), int(groups))

    def forward(self, refine, near, near_refine, feats: Tensor, fmap_s8: Tensor, grid: Tensor, valid: Tensor,
                near_block_zeros: bool = False) -> list:
        """-> ``[near, mid, far]`` logits, each ``[B, 10, rows, cols]`` (float, the band's own grid)."""
        g0, g1, g2 = self.geom
        r_mid0, r_mid1 = int(round(g1.x0 / 0.25)), int(round(g1.x1 / 0.25))
        if int(feats.shape[2]) != int(round(g2.x1 / 0.25)) or int(feats.shape[3]) != 2 * g2.cols:
            raise ValueError(f"[map-banded] the 0.25 m features are {tuple(feats.shape[2:])}, the bands tile "
                             f"{(int(round(g2.x1 / 0.25)), 2 * g2.cols)}")
        u = refine.inp(feats)                                       # 1x1 + GN over the WHOLE 0.25 m map, as refcv8
        # ---- near: the 10 cm decode of the rows 0-10 m
        x = F.interpolate(u[:, :, :int(round(g0.x1 / 0.25))], size=(g0.rows, g0.cols), mode="bilinear", align_corners=False)
        if near is not None:
            n = near(fmap_s8, grid, valid)
            if tuple(n.shape[2:]) != (g0.rows, g0.cols):
                raise ValueError(f"[map-banded] the near lift covers {tuple(n.shape[2:])}, the near band is "
                                 f"{(g0.rows, g0.cols)}: build it with --map-hires-near-lift-m {g0.x1:g}")
            x = x + n
            for blk in (near_refine or ()):
                x = blk(x, zero_input=bool(near_block_zeros))
        elif near_refine:
            raise ValueError("[map-banded] near refine blocks without a near lift")
        x = F.gelu(refine.norm1(refine.conv1(x)))
        x = F.gelu(refine.norm2(refine.conv2(x)))
        z0 = self.head_near(x)
        # ---- mid: the lift's native grid
        z1 = self.mid(u[:, :, r_mid0:r_mid1])
        # ---- far: 2 x 2-pooled
        z2 = self.far(F.avg_pool2d(u[:, :, r_mid1:], 2))
        for z, g in zip((z0, z1, z2), self.geom):
            if tuple(z.shape[2:]) != (g.rows, g.cols):
                raise RuntimeError(f"[map-banded] band {g.name} logits {tuple(z.shape)} != {(g.rows, g.cols)}")
        return [z0, z1, z2]


def unobs_masks(lift_valid: Tensor, geom: tuple) -> list:
    """``lift_valid`` ``[B, X4, Y4]`` bool (0.25 m, the lift's own validity) -> ``[near, mid, far]`` ``[B, rows, cols]`` bool,
    True = NOT OBSERVED. near: each 10 cm cell takes the 0.25 m cell its centre falls in (``nearest-exact``, the rule
    ``lift_valid_to_fine`` already applies to refcv8's CE); mid: the lift cell itself; far: a 0.5 m cell is observed only when
    ALL FOUR of its lift cells are (conservative, like the near lift's validity)."""
    if lift_valid.dim() != 3 or lift_valid.dtype != torch.bool:
        raise ValueError(f"lift_valid must be [B, X, Y] bool, got {tuple(lift_valid.shape)} {lift_valid.dtype}")
    g0, g1, g2 = geom
    r1, r2 = int(round(g1.x0 / 0.25)), int(round(g1.x1 / 0.25))
    v = lift_valid
    near = F.interpolate(v[:, :int(round(g0.x1 / 0.25))][:, None].float(), size=(g0.rows, g0.cols),
                         mode="nearest-exact")[:, 0] > 0.5
    mid = v[:, r1:r2]
    far = F.avg_pool2d(v[:, r2:][:, None].float(), 2)[:, 0] > 0.999
    out = [~near, ~mid, ~far]
    for m, g in zip(out, geom):
        if tuple(m.shape[1:]) != (g.rows, g.cols):
            raise ValueError(f"unobs band {g.name} {tuple(m.shape[1:])} != {(g.rows, g.cols)}")
    return out


# =========================================================================== #
# targets                                                                     #
# =========================================================================== #
def dilate_thin(mask: Tensor, radius: int = THIN_RADIUS_CELLS) -> Tensor:
    """``[B, H, W]`` bool -> Chebyshev dilation by ``radius`` 10 cm cells: a 1-cell line becomes ``2 * radius + 1`` cells =
    0.5 m wide (``THIN_RADIUS_CELLS`` 2)."""
    if radius <= 0:
        return mask
    x = mask[:, None].float()
    return F.max_pool2d(x, 2 * radius + 1, stride=1, padding=radius)[:, 0] > 0.5


def _shifted(x: Tensor, k: int, dim: int, fill) -> Tensor:
    """``out[i] = x[i + k]`` along ``dim``; ``fill`` outside."""
    n = x.shape[dim]
    out = torch.full_like(x, fill)
    if abs(k) >= n:
        return out
    if k >= 0:
        out.narrow(dim, 0, n - k).copy_(x.narrow(dim, k, n - k))
    else:
        out.narrow(dim, -k, n + k).copy_(x.narrow(dim, 0, n + k))
    return out


def _nearest_source(src: Tensor, radius: int, payload: Tensor | None = None):
    """Exact Euclidean nearest-source search, window-limited and SEPARABLE (two passes of ``2 * radius + 1`` shifts instead of
    one pass of ``pi * radius^2``): ``src`` ``[B, h, w]`` bool -> ``(d2 [B, h, w]`` squared distance in CELLS (>= 1e8 where no
    source lies within ``radius``), ``payload at the nearest source [B, P, h, w]`` or None). Any source within Euclidean
    ``radius`` has both lattice offsets within ``radius``, so the result is exact there. A source cell is at distance 0."""
    B, h, w = src.shape
    g = torch.full((B, h, w), _INF, device=src.device)
    pv = None if payload is None else torch.zeros_like(payload)
    for k in sorted(range(-radius, radius + 1), key=abs):                  # nearest first: ties keep the nearer row
        better = _shifted(src, k, 1, False) & (float(abs(k)) < g)
        g = torch.where(better, torch.full_like(g, float(abs(k))), g)
        if payload is not None:
            pv = torch.where(better[:, None], _shifted(payload, k, 2, 0.0), pv)
    d2 = torch.full((B, h, w), _INF * _INF, device=src.device)
    pay = None if payload is None else torch.zeros_like(payload)
    for m in sorted(range(-radius, radius + 1), key=abs):
        gs = _shifted(g, m, 2, _INF)
        cand = float(m * m) + gs * gs
        better = cand < d2
        d2 = torch.where(better, cand, d2)
        if payload is not None:
            pay = torch.where(better[:, None], _shifted(pv, m, 3, 0.0), pay)
    return d2, pay


def signed_distance(inside: Tensor, outside: Tensor, cell_m: float, clip_m: float = SDF_CLIP_M):
    """``inside`` / ``outside`` ``[B, h, w]`` bool (the two SEEN sides of the boundary; an unseen cell is on neither) ->
    ``(sdf [B, h, w] metres, clipped to +-clip_m; positive inside, valid [B, h, w] bool)``.

    ⭐ The boundary lies midway between the last inside and the first outside cell CENTRE, so a cell's distance is
    ``d(centre, nearest opposite centre) - cell / 2``: a straight axis-aligned kerb gives ``+-(k + 0.5) * cell`` -- LINEAR in the
    lateral distance with slope 1 (the E0e control). ``valid`` = a seen cell whose opposite side lies within ``clip_m``
    (``|sdf| < clip_m``): the L1 is supervised there and nowhere else."""
    R = int(math.ceil(clip_m / cell_m)) + 1
    d2_to_out, _ = _nearest_source(outside, R)
    d2_to_in, _ = _nearest_source(inside, R)
    cell = float(cell_m)
    d_in = d2_to_out.clamp_max(_INF * _INF).sqrt() * cell - 0.5 * cell
    d_out = d2_to_in.clamp_max(_INF * _INF).sqrt() * cell - 0.5 * cell
    sdf = torch.where(inside, d_in, torch.where(outside, -d_out, torch.zeros_like(d_in)))
    found = torch.where(inside, d2_to_out, d2_to_in) <= float(R * R)
    valid = (inside | outside) & found & (sdf.abs() < clip_m - 1e-6)
    return sdf.clamp(-clip_m, clip_m), valid


def _lane_vectors(lane_f: Tensor, g: BandGeom, y_half_m: float):
    """Per band cell: the vector (metres) from the cell centre to the nearest GT lane POSITION -- the centroid of the 10 cm lane
    cells inside the nearest cell that holds any (so the offset carries the SUB-CELL position). -> ``(off [B, 2, rows, cols]``
    (x ahead, y left), ``valid [B, rows, cols]`` = a lane position within ``SDF_CLIP_M`` of the centre)``."""
    B, hf, wf = lane_f.shape
    dev = lane_f.device
    ratio = g.cell / FINE_CELL_M
    fi, fj = torch.arange(hf, device=dev), torch.arange(wf, device=dev)
    bi = torch.floor((fi.float() + 0.5) / ratio).long().clamp(0, g.rows - 1)
    bj = torch.floor((fj.float() + 0.5) / ratio).long().clamp(0, g.cols - 1)
    flat = (bi[:, None] * g.cols + bj[None, :]).reshape(1, -1).expand(B, -1)
    xf = g.x0 + (fi.float() + 0.5) * FINE_CELL_M
    yf = -float(y_half_m) + (fj.float() + 0.5) * FINE_CELL_M
    m = lane_f.float()
    z = torch.zeros(B, g.rows * g.cols, device=dev)
    cnt = z.clone().scatter_add_(1, flat, m.reshape(B, -1))
    sx = z.clone().scatter_add_(1, flat, (m * xf[None, :, None]).reshape(B, -1))
    sy = z.clone().scatter_add_(1, flat, (m * yf[None, None, :]).reshape(B, -1))
    cnt, sx, sy = (t.reshape(B, g.rows, g.cols) for t in (cnt, sx, sy))
    src = cnt > 0
    pay = torch.stack([sx / cnt.clamp_min(1.0), sy / cnt.clamp_min(1.0)], dim=1)
    R = int(math.ceil(SDF_CLIP_M / g.cell)) + 1
    d2, nearest = _nearest_source(src, R, pay)
    cx = g.x0 + (torch.arange(g.rows, device=dev).float() + 0.5) * g.cell
    cy = -float(y_half_m) + (torch.arange(g.cols, device=dev).float() + 0.5) * g.cell
    centre = torch.stack([cx[:, None].expand(g.rows, g.cols), cy[None, :].expand(g.rows, g.cols)], dim=0)[None]
    off = nearest - centre
    valid = (d2 <= float(R * R)) & (off.pow(2).sum(1).sqrt() <= SDF_CLIP_M)
    return off, valid


def build_banded_targets(codes: Tensor, *, x_max_m: float, y_half_m: float, near_m: float = NEAR_M_DEFAULT) -> list:
    """The 10 cm GT raster ``codes`` ``[B, H, W]`` uint8 (``NOT_SEEN_CODE`` 255) -> one dict per band (``near, mid, far``):

    ``seen`` bool; ``drv sdw lane edge_h xwalk arrow hatch`` float {0, 1}; ``sdf`` metres + ``sdf_sup`` bool; ``off``
    ``[B, 2, rows, cols]`` metres + ``off_sup`` bool. Area classes take the code at the band cell's CENTRE (the same rule as
    the lift's validity); the THIN classes (lane, edge) are dilated to ~0.5 m at 10 cm and then max-pooled onto the band grid,
    so they stay >= 0.5 m wide at every band. ``drv`` is the CODE SET (1, 2, 3, 4, 6), never an argmax of the 8-way raster."""
    geom = band_geometry(x_max_m, y_half_m, near_m)
    H, W = int(round(float(x_max_m) / FINE_CELL_M)), int(round(2.0 * float(y_half_m) / FINE_CELL_M))
    if codes.dim() != 3 or tuple(codes.shape[1:]) != (H, W) or codes.dtype != torch.uint8:
        raise ValueError(f"[map-banded] codes must be [B, {H}, {W}] uint8, got {tuple(codes.shape)} {codes.dtype}")
    c = codes.long()
    lane_dil = dilate_thin(c == C_LANE)
    edge_dil = dilate_thin(c == C_EDGE)
    out = []
    for g in geom:
        r0, r1 = g.fine_rows
        ratio = g.cell / FINE_CELL_M
        ri = (torch.floor((torch.arange(g.rows, device=c.device).float() + 0.5) * ratio).long().clamp(max=r1 - r0 - 1) + r0)
        cj = torch.floor((torch.arange(g.cols, device=c.device).float() + 0.5) * ratio).long().clamp(max=W - 1)
        s = c[:, ri][:, :, cj]                                           # centre sample [B, rows, cols]
        seen = s != NOT_SEEN_CODE
        drv = _drv_mask(s)

        def thin(d):
            return F.adaptive_max_pool2d(d[:, r0:r1][:, None].float(), (g.rows, g.cols))[:, 0]
        sdf, sdf_ok = signed_distance(drv & seen, (~drv) & seen, g.cell)
        off, off_ok = _lane_vectors((c[:, r0:r1] == C_LANE), g, y_half_m)
        out.append({"seen": seen, "drv": drv.float(), "sdw": (s == C_SIDEWALK).float(), "lane": thin(lane_dil),
                    "edge_h": thin(edge_dil), "xwalk": (s == C_XWALK).float(), "arrow": (s == C_ARROW).float(),
                    "hatch": (s == C_HATCH).float(), "sdf": sdf, "sdf_sup": sdf_ok & seen, "off": off,
                    "off_sup": off_ok & seen})
    return out


def _drv_mask(s: Tensor) -> Tensor:
    """THE drivable-surface definition: membership in the CODE SET ``DRV_CODES`` (not ``argmax == drivable``)."""
    m = torch.zeros_like(s, dtype=torch.bool)
    for k in DRV_CODES:
        m |= s == k
    return m


def target_fractions(targets: list) -> dict:
    """``{band: {channel: positive fraction among the SEEN cells}}`` (+ ``sdf_sup`` / ``off_sup`` supervised fractions)."""
    out = {}
    for g, t in zip(BAND_NAMES, targets):
        seen = t["seen"]
        n = max(int(seen.sum()), 1)
        d = {ch: float((t[ch] > 0.5)[seen].sum()) / n for ch in PROB_CHANNELS}
        d["drv_sdf"] = float(t["sdf_sup"].sum()) / n
        d["lane_off"] = float(t["off_sup"].sum()) / n
        out[g] = d
    return out


def assert_target_content(fractions: dict, *, bands=BAND_NAMES, channels=None) -> None:
    """A content assertion on REAL data: every channel has non-zero positives in every band asked for. Raises with the table."""
    ch = tuple(channels or (PROB_CHANNELS + ("drv_sdf", "lane_off")))
    bad = [(b, c) for b in bands for c in ch if not fractions[b][c] > 0.0]
    if bad:
        raise AssertionError("[map-banded] channels with ZERO positives: %s\n%s" % (bad, format_fractions(fractions)))


def format_fractions(fractions: dict) -> str:
    chs = list(next(iter(fractions.values())).keys())
    rows = ["band   " + " ".join("%8s" % c for c in chs)]
    for b, d in fractions.items():
        rows.append("%-6s " % b + " ".join("%8.5f" % d[c] for c in chs))
    return "\n".join(rows)


# =========================================================================== #
# losses                                                                      #
# =========================================================================== #
def supervised_mask(seen: Tensor, unobs: Tensor) -> Tensor:
    """THE ``unobs`` rule: a cell is supervised iff the GT saw it AND the lift observed it."""
    return seen & ~unobs


def sigmoid_focal(z: Tensor, y: Tensor, gamma: float = FOCAL_GAMMA) -> Tensor:
    """Per-cell sigmoid focal loss, no alpha (the Dice term carries the class balance)."""
    ce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    p = torch.sigmoid(z)
    pt = p * y + (1.0 - p) * (1.0 - y)
    return (1.0 - pt).pow(gamma) * ce


def soft_dice(p: Tensor, y: Tensor, sup: Tensor) -> Tensor:
    """``1 - (2 sum(p y) + eps) / (sum(p) + sum(y) + eps)`` pooled over the batch's supervised cells."""
    s = sup.float()
    inter = (p * y * s).sum()
    return 1.0 - (2.0 * inter + _DICE_EPS) / ((p * s).sum() + (y * s).sum() + _DICE_EPS)


def _mean_over(x: Tensor, m: Tensor) -> Tensor:
    return (x * m.float()).sum() / m.float().sum().clamp_min(1.0)


def telemetry_keys(with_metrics: bool = True) -> tuple:
    """Every key :func:`banded_map_loss` emits (except ``loss``): ONE per channel x band, so G-LIVE / Watch see each term."""
    ks = ["map_banded_loss_%s_%s" % (ch, b) for ch in CHANNELS for b in BAND_NAMES]
    ks += ["map_banded_maxent_%s" % b for b in BAND_NAMES] + ["map_banded_n_%s" % b for b in BAND_NAMES]
    ks += ["map_banded_unobs_frac_%s" % b for b in BAND_NAMES]
    if with_metrics:
        ks += ["map_banded_%s_%s_%s" % (st, ch, b) for st in ("pos", "inter", "union") for ch in PROB_CHANNELS
               for b in BAND_NAMES]
    return tuple(ks)


def banded_map_loss(out: dict, targets: list, *, with_metrics: bool = True) -> dict:
    """``out = {"logits": [near, mid, far] [B, 10, h, w], "unobs": [..] [B, h, w] bool}`` + :func:`build_banded_targets` ->
    ``{"loss": sum_b w_b (sum_c L_bc + 1e-3 * maxent_b), <one telemetry key per channel x band>, ...}``. A band with nothing
    supervised contributes an exact 0 attached to the graph."""
    row, total = {}, None
    ch_i = {c: i for i, c in enumerate(CHANNELS)}
    prob_idx = [ch_i[c] for c in PROB_CHANNELS]
    for b, bn in enumerate(BAND_NAMES):
        z_all, un, t = out["logits"][b].float(), out["unobs"][b], targets[b]
        sup = supervised_mask(t["seen"], un)
        band = z_all.sum() * 0.0
        for ch in PROB_CHANNELS:
            z, y = z_all[:, ch_i[ch]], t[ch]
            lf = _mean_over(sigmoid_focal(z, y), sup)
            ld = soft_dice(torch.sigmoid(z), y, sup)
            lc = lf + ld
            band = band + lc
            row["map_banded_loss_%s_%s" % (ch, bn)] = lc.detach()
            if with_metrics:
                with torch.no_grad():
                    pr, s = torch.sigmoid(z) > 0.5, sup
                    yb = y > 0.5
                    row["map_banded_pos_%s_%s" % (ch, bn)] = float((yb & s).sum())
                    row["map_banded_inter_%s_%s" % (ch, bn)] = float((pr & yb & s).sum())
                    row["map_banded_union_%s_%s" % (ch, bn)] = float(((pr | yb) & s).sum())
        for ch, y, m in (("drv_sdf", t["sdf"], sup & t["sdf_sup"]), ("lane_ox", t["off"][:, 0], sup & t["off_sup"]),
                         ("lane_oy", t["off"][:, 1], sup & t["off_sup"])):
            lc = _mean_over((z_all[:, ch_i[ch]] - y).abs(), m)
            band = band + lc
            row["map_banded_loss_%s_%s" % (ch, bn)] = lc.detach()
        # PON eq. 2: push the sigmoid channels toward 0.5 (max entropy) where the camera saw nothing
        p = torch.sigmoid(z_all[:, prob_idx]).clamp(1e-6, 1.0 - 1e-6)
        neg_h = (p * p.log() + (1.0 - p) * (1.0 - p).log()) / math.log(2.0)          # in [-1, 0] per channel
        me = _mean_over(neg_h.sum(1), un)
        row["map_banded_maxent_%s" % bn] = me.detach()
        row["map_banded_n_%s" % bn] = float(sup.sum())
        row["map_banded_unobs_frac_%s" % bn] = float(un.float().mean())
        bl = float(BAND_WEIGHTS[b]) * (band + MAXENT_W * me)
        total = bl if total is None else total + bl
    row["loss"] = total
    return row

