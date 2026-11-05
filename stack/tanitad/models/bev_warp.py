"""refcv9 T1 -- the TEMPORAL BEV WARP (REFCV9_DESIGN 2 R8 "blind zone 0-4.3 m: 21.5 % of the near corridor is camera-invisible"; 4.3 arm T1;
6 D3, the PI's ruling: YES as an E1 arm, PAST poses only, never future; a BEV-history-SHUFFLED control must read worse).

What it does.  The 10 cm branch's shared lift turns the stride-8 map of ONE frame into a 0.25 m ego-frame BEV ``[B, C, X, Y]`` (x forward in
[0, x_max_m), y left in [-y_half_m, y_half_m)); cells the camera does not reach carry the lift's ``unobserved`` embedding.  T1 lifts the
frames at ``t - lag`` (default 0.5 s and 1.0 s) through the SAME lift, rigidly warps each into t0's ego grid by the PAST ego motion (SE(2) on the
ground plane, bilinear; a cell that leaves the grid, or that the past frame itself did not observe, is 0 with a validity mask) and fuses them into
the current BEV through a ZERO-INIT gated 1x1 conv::

    bev_fused = bev + gate * conv1x1( cat_p [ warped_p , mask_p ] ),      gate = 0 at init  =>  bev_fused == bev  (exact)

Conventions (pinned by tests/test_bev_warp.py, the ANALYTIC control): ego frame +x forward, +y LEFT, CCW-positive yaw (``lan.to_ego_frame``,
``refb_labels.ego_frame``); a pose row is ``[x, y, yaw, v]`` in the clip-local world frame.  The relative transform of a past frame is built from
the two poses ALONE (``relative_se2``): it reads no label, no future pose, no situation-classifier output.

Pure tensor code: nothing here imports the trainer.  The flag's contract with the trainer is ``tanitad/train/bev_warp_flag.py``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

#: the episode contract's step: 10 Hz (``tanitad.data.clip_clock.NOMINAL_DT_S``); a lag in seconds is a whole number of these rows.
DT_S = 0.1
#: the design's pair: t - 0.5 s and t - 1.0 s.
LAGS_S_DEFAULT = (0.5, 1.0)
#: per-lag input channels of the fuse: the warped BEV plus ONE validity channel.
MASK_CH = 1
#: ``--r9-bev-warp-fuse-init``: ``rezero`` = today's build (the 1x1's PyTorch default init, gate 0); ``identity`` = the COPY init (``BevWarpFuse.init_identity_``).
FUSE_INITS = ("rezero", "identity")


def lags_to_rows(lags_s, dt_s: float = DT_S) -> tuple:
    """Lags in seconds -> whole rows (``round(lag / dt)``). REFUSES a lag that is not a whole number of rows (within 1e-6 s), a lag < 1 row,
    and any order but strictly increasing (the fuse's input channels are in this order)."""
    rows = []
    for s in lags_s:
        s = float(s)
        r = int(round(s / dt_s))
        if r < 1 or abs(r * dt_s - s) > 1e-6:
            raise ValueError("[bev-warp] lag %g s is not a whole number (>= 1) of %g s rows" % (s, dt_s))
        rows.append(r)
    if not rows or any(b <= a for a, b in zip(rows, rows[1:])):
        raise ValueError("[bev-warp] lags %s must be non-empty and strictly increasing" % (list(lags_s),))
    return tuple(rows)


def _wrap_pi(a: torch.Tensor) -> torch.Tensor:
    return a - (2.0 * math.pi) * torch.floor((a + math.pi) / (2.0 * math.pi))


def relative_se2(pose_now: torch.Tensor, pose_past: torch.Tensor) -> torch.Tensor:
    """``T(t0 <- t-k)`` from TWO poses ``[..., >=3]`` (x, y, yaw; broadcastable leading dims) -> ``[..., 3] = (dx, dy, dyaw)`` float64:
    ``(dx, dy)`` = the PAST ego origin expressed in the NOW ego frame, ``dyaw = wrap(yaw_past - yaw_now)``.  Reads the two poses and nothing else."""
    pn, pp = pose_now.double(), pose_past.double()
    dxw = pp[..., 0] - pn[..., 0]
    dyw = pp[..., 1] - pn[..., 1]
    c, s = torch.cos(pn[..., 2]), torch.sin(pn[..., 2])
    dx = dxw * c + dyw * s                       # R(-yaw_now) (t_past - t_now)
    dy = -dxw * s + dyw * c
    return torch.stack([dx, dy, _wrap_pi(pp[..., 2] - pn[..., 2])], dim=-1)


def sample_grid(T: torch.Tensor, x_max_m: float, y_half_m: float, X: int, Y: int):
    """The INVERSE-warp sampling grid: for every cell centre ``c`` of the NOW grid, where in the PAST grid does it come from?
    ``q = R(-dyaw) (c - d)`` (``d, dyaw`` = ``T``).  ``T`` ``[N, 3]`` -> ``(grid [N, X, Y, 2] float32 in ``F.grid_sample`` order (width = the Y axis first),
    ``in_bounds [N, X, Y]`` bool: q lies inside the past grid's extent)``.  Computed in float64."""
    dev = T.device
    cx = ((torch.arange(X, device=dev, dtype=torch.float64) + 0.5) * (float(x_max_m) / X)).view(1, X, 1)
    cy = (-float(y_half_m) + (torch.arange(Y, device=dev, dtype=torch.float64) + 0.5) * (2.0 * float(y_half_m) / Y)).view(1, 1, Y)
    T = T.double()
    ux = cx - T[:, 0].view(-1, 1, 1)
    uy = cy - T[:, 1].view(-1, 1, 1)
    c, s = torch.cos(T[:, 2]).view(-1, 1, 1), torch.sin(T[:, 2]).view(-1, 1, 1)
    qx = c * ux + s * uy
    qy = -s * ux + c * uy
    gu = qy / float(y_half_m)                    # width axis (Y):  [-y_half, +y_half] -> [-1, 1]  (align_corners=False: cell edges)
    gv = 2.0 * qx / float(x_max_m) - 1.0         # height axis (X): [0, x_max]         -> [-1, 1]
    inb = (gu.abs() <= 1.0) & (gv.abs() <= 1.0)
    return torch.stack([gu, gv], dim=-1).float(), inb


def warp_bev(feat: torch.Tensor, src_valid, T: torch.Tensor, x_max_m: float, y_half_m: float):
    """Warp a PAST BEV ``feat [N, C, X, Y]`` into the NOW grid by ``T [N, 3]`` (``relative_se2``).  ``src_valid [N, X, Y]`` bool (the cells the
    past frame itself observed; ``None`` = all) -> ``(warped [N, C, X, Y], mask [N, 1, X, Y] float)``.  ``mask`` is 1 where the NOW cell holds a
    real past observation (inside the past grid AND observed there); everywhere else ``warped`` is EXACTLY 0.  A sample with ``T == 0`` EXACTLY
    takes the identity path (``feat`` itself, masked by ``src_valid``): bilinear sampling at pixel centres is the identity only up to float rounding."""
    N, C, X, Y = feat.shape
    grid, inb = sample_grid(T, x_max_m, y_half_m, X, Y)
    warped = F.grid_sample(feat, grid.to(feat.dtype), mode="bilinear", padding_mode="zeros", align_corners=False)
    m = inb
    if src_valid is not None:
        sv = F.grid_sample(src_valid.to(feat.dtype).unsqueeze(1), grid.to(feat.dtype), mode="bilinear", padding_mode="zeros", align_corners=False)
        m = inb & (sv[:, 0] > 0.5)
    zero = (T == 0).all(dim=-1).view(-1, 1, 1)
    if src_valid is not None:
        m = torch.where(zero, src_valid.bool(), m)
    else:
        m = torch.where(zero, torch.ones_like(m), m)
    mf = m.unsqueeze(1).to(feat.dtype)
    warped = torch.where(zero.view(-1, 1, 1, 1), feat, warped) * mf
    return warped, mf


class BevWarpFuse(nn.Module):
    """The ZERO-INIT gated 1x1 fuse of the warped past BEVs into the current BEV.

    ``forward(bev, warped, masks)``: ``bev [B, C, X, Y]``, ``warped`` / ``masks`` lists of ``P`` tensors (``[B, C, X, Y]`` / ``[B, 1, X, Y]``).
    ``out = bev + gate * fuse(cat_p [warped_p, mask_p])`` with a PER-CHANNEL ``gate`` initialised to exactly 0: the output is ``bev`` (bit for
    bit, +-0.0 aside) until the optimiser moves the gate.  At gate 0 the gate's gradient is non-zero (the fuse output is) and the fuse's is
    EXACTLY 0.0 -- the standard ReZero shape; the fuse's gradient wakes after the first gate step."""

    def __init__(self, d_bev: int, n_past: int, x_max_m: float, y_half_m: float, lag_rows: tuple):
        super().__init__()
        if int(n_past) != len(lag_rows):
            raise ValueError("[bev-warp] n_past %d != %d lags" % (n_past, len(lag_rows)))
        self.d_bev, self.n_past = int(d_bev), int(n_past)
        self.x_max_m, self.y_half_m = float(x_max_m), float(y_half_m)
        self.lag_rows = tuple(int(r) for r in lag_rows)
        #: NO BIAS: with no history (every mask 0, every warped 0) the fuse output is EXACTLY 0, so a clip-start window is unchanged for ANY gate
        self.fuse = nn.Conv2d(self.n_past * (self.d_bev + MASK_CH), self.d_bev, 1, bias=False)
        self.gate = nn.Parameter(torch.zeros(self.d_bev))
        #: HOW the fuse weight was built (``rezero`` = the 1x1's default init; ``identity`` = ``init_identity_``): a build-time fact G-DVB reads, NOT a state_dict entry --
        #: ``init_identity_`` is the ONLY writer of ``"identity"``, so the string cannot be set without the weights being written.
        self.init_mode = "rezero"
        #: refcv9 T1h ``--r9-bev-warp-fill-only``: when True the gated history term is applied ONLY on cells the CURRENT lift does not observe
        #: (``out = bev + gate * fuse(x) * (1 - valid_now)``); ``lift_hook`` passes ``valid_now``, ``mix`` REFUSES to run fill-only without it.  OFF: today's ``mix``, bit for bit.
        self.fill_only = False
        #: one-shot staged context (``stage`` -> the lift hook): (s8_past [B, P, C8, h, w], T [B, P, 3], has [B, P] bool)
        self._ctx = None
        #: refcv9 T1h (``--r9-bev-warp-supervise-hist``): when True the lift hook ALSO records ``hist_valid`` (the union of the lags' warp-valid masks, [B, X, Y] bool,
        #: GEOMETRY ONLY) for ``MapHiresBranch.forward`` to emit as ``map_hires_hist_valid`` (``take_hist_valid``).  OFF (the default): nothing is computed or kept.
        self.emit_hist = False
        self._hist_valid = None

    def init_identity_(self) -> None:
        """refcv9 T1h ``--r9-bev-warp-fuse-init identity`` -- the COPY init: ``W[c, p*(C+1)+c] = 1/P`` for every lag ``p`` and channel ``c``, every other entry 0
        (the mask columns included), so ``fuse(cat_p[warped_p, mask_p])`` is EXACTLY the mean over the P lags of the warped past BEV (channel order from ``mix``: the
        ``(warped_p, mask_p)`` pairs concatenated).  The gate stays at its exact 0: the step-0 forward is the baseline's, bit for bit, and the gate's step-0 gradient is
        ``sum_cells dL/dbev * mean_p(warped_p)`` instead of a random-W projection.  Called ONLY at fresh build (``bev_warp_flag.attach``); every checkpoint load that
        follows (resume, the eval loader's strict load) overwrites it with the trained weights."""
        c, p = self.d_bev, self.n_past
        with torch.no_grad():
            w = self.fuse.weight
            w.zero_()
            idx = torch.arange(c, device=w.device)
            for k in range(p):
                w[idx, k * (c + MASK_CH) + idx, 0, 0] = 1.0 / p
        self.init_mode = "identity"

    def mix(self, bev, warped, masks, valid_now=None):
        """``valid_now`` ``[B, X, Y]`` bool (the lift's own ``valid.any(dim=1)``): read ONLY when ``fill_only`` (then REQUIRED); the default path never touches it."""
        if len(warped) != self.n_past or len(masks) != self.n_past:
            raise ValueError("[bev-warp] fuse built for %d past frames, got %d / %d" % (self.n_past, len(warped), len(masks)))
        x = torch.cat([t for pair in zip(warped, masks) for t in pair], dim=1)
        if not self.fill_only:
            return bev + self.gate.view(1, -1, 1, 1).to(bev.dtype) * self.fuse(x)
        if valid_now is None:
            raise ValueError("[bev-warp] --r9-bev-warp-fill-only is built but no valid_now reached mix: the history would silently be added on EVERY cell")
        return bev + self.gate.view(1, -1, 1, 1).to(bev.dtype) * self.fuse(x) * (~valid_now.bool()).unsqueeze(1).to(bev.dtype)      # (1 - valid_now): history only where the camera sees nothing now

    # ---- the staged context + the forward hook on the shared lift ------------------------------------------------------------ #
    def stage(self, s8_past: torch.Tensor, T: torch.Tensor, has: torch.Tensor) -> None:
        if tuple(s8_past.shape[:2]) != (T.shape[0], self.n_past) or tuple(T.shape) != (s8_past.shape[0], self.n_past, 3) \
                or tuple(has.shape) != (s8_past.shape[0], self.n_past):
            raise ValueError("[bev-warp] staged history shapes disagree: s8 %s, T %s, has %s (P = %d)"
                             % (tuple(s8_past.shape), tuple(T.shape), tuple(has.shape), self.n_past))
        self._ctx = (s8_past, T, has.bool())

    def lift_hook(self, lift: nn.Module, inputs: tuple, out: torch.Tensor):
        """``register_forward_hook`` target on ``mh.lift`` (a BOUND method, so a deepcopy of the model -- the EMA -- carries its own).  Replaces the
        lift's output by the fused BEV.  REFUSES (never skips) a forward with no staged history: a T1-built model run without its history is a
        different model, and it would read as "the warp did nothing"."""
        ctx, self._ctx = self._ctx, None             # ONE-SHOT: a second lift call in the same forward would otherwise reuse a stale history
        if ctx is None:
            raise RuntimeError("[bev-warp] the model is built with the temporal BEV warp but no history was staged for this forward "
                               "(bev_warp_flag.stage(model, batch, device) before model(...)); refusing to run it history-less")
        s8_past, T, has = ctx
        _, grid, valid = inputs
        src_valid = valid.any(dim=1)                 # [B, X, Y]: the cells THIS camera reaches -- the same camera at t - k
        warped, masks = [], []
        with torch.no_grad():                        # BEVFormer / StreamMapNet convention: the history is features, not a second graph
            for p in range(self.n_past):
                bev_p = type(lift).forward(lift, s8_past[:, p].to(out.dtype), grid, valid)         # the SAME lift weights; bypasses this hook
                w, m = warp_bev(bev_p, src_valid, T[:, p], self.x_max_m, self.y_half_m)
                h = has[:, p].view(-1, 1, 1, 1).to(w.dtype)
                warped.append(w * h)
                masks.append(m * h)
        self._hist_valid = hist_valid_of(masks) if self.emit_hist else None            # T1h only; the mask list is the SAME one ``mix`` reads
        if self.fill_only:
            return self.mix(out, warped, masks, valid_now=src_valid)        # T1h fill-only: the SAME `valid.any(dim=1)` the hook already holds
        return self.mix(out, warped, masks)

    def take_hist_valid(self):
        """ONE-SHOT read of the last lift call's ``hist_valid`` (``None`` unless ``emit_hist``); ``MapHiresBranch.forward`` calls it right after ``self.lift``."""
        hv, self._hist_valid = self._hist_valid, None
        return hv


def hist_valid_of(masks) -> torch.Tensor:
    """refcv9 T1h: the cells at least one lag's warped past lift covers -- the union of the lag masks ``warp_bev`` returned (``[B, 1, X, Y]`` float, ALREADY multiplied
    by the lag's ``has``) -> ``[B, X, Y]`` bool.  GEOMETRY ONLY: every ingredient of a mask is ``T`` (poses), the camera's own ``valid`` and ``has``; the warped CONTENT
    never enters it, so the shuffled-history control (same poses, donor frames) has the SAME union."""
    return torch.stack([m[:, 0] > 0.5 for m in masks], dim=0).any(dim=0)


def derangement(n: int, seed: int = 0) -> list:
    """A FIXED derangement of ``range(n)`` (``perm[i] != i`` for every i), deterministic in ``seed``: a seeded permutation whose fixed points are
    rotated among themselves (a single fixed point is swapped with its neighbour).  ``n >= 2``."""
    if n < 2:
        raise ValueError("[bev-warp] a derangement needs >= 2 elements, got %d" % n)
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).tolist()
    fixed = [i for i in range(n) if perm[i] == i]
    if len(fixed) == 1:
        i = fixed[0]
        j = (i + 1) % n
        perm[i], perm[j] = perm[j], perm[i]
    elif len(fixed) > 1:
        vals = [perm[i] for i in fixed]
        for i, v in zip(fixed, vals[1:] + vals[:1]):
            perm[i] = v
    assert sorted(perm) == list(range(n)) and all(perm[i] != i for i in range(n)), "derangement construction failed"
    return perm
