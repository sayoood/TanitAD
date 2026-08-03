"""Spatial grid readout and frozen calibrated probes.

A7: spatial readout >> global pooling for position-faithful compact state.
A3 (calibrated-decode doctrine): probes that read *imagination* must be fitted
on the predictor's own imagined latents, not on real-frame encodings — the
systematic manifold shift of imagination is absorbed into the probe weights
(measured 0.97 vs 0.66 direction accuracy on identical predictions).

Probes are fitted offline (closed-form ridge) and frozen. They are the minimal
form of the H13 extraction heads and carry zero training burden.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _adaptive_avg_matrix(n_in: int, n_out: int) -> Tensor:
    """The averaging matrix ``[n_out, n_in]`` that ``AdaptiveAvgPool1d`` applies along one axis.

    PyTorch's adaptive pooling assigns output bin ``i`` the input range
    ``[floor(i * n_in / n_out), ceil((i + 1) * n_in / n_out))`` and takes its mean. With static
    sizes that is a constant linear map, so the whole op becomes a matmul — which ONNX exports and
    ``adaptive_avg_pool2d`` does not (see :class:`SpatialGridReadout`).

    ⚠️ Bins are deliberately UNEQUAL where the sizes do not tile (11 -> 4 gives 3/3/3/2), because
    that is exactly what the trained weights were fitted against. Equalising them would be a
    silent model change dressed up as an export fix.
    """
    m = torch.zeros(n_out, n_in)
    for i in range(n_out):
        start = (i * n_in) // n_out
        end = -((-(i + 1) * n_in) // n_out)          # ceil((i+1)*n_in/n_out)
        m[i, start:end] = 1.0 / float(end - start)
    return m


class SpatialGridReadout(nn.Module):
    """Token grid [B, N, D] -> compact state [B, Gh*Gw*d_r] preserving layout.

    ``token_grid`` (opt-in, default ``None``) states the token grid as
    ``(rows, cols)`` for a NON-SQUARE encoder input; ``None`` infers the square
    grid from ``n_tokens`` exactly as before. ``grid_w`` likewise defaults to
    ``grid`` (square readout).

    ⭐ The readout is the geometry FIREWALL of the stack: whatever the input
    geometry, the compact state stays ``grid * grid_w * d_readout``. A wide
    256x640 input therefore produces the SAME state_dim 2048 the flagship
    predictor, tactical/strategic policies and every grounding head expect —
    which is why widening the field is a data+encoder change and not a
    whole-model redesign.

    ⚠️ ``grid_w`` is the ONE knob that breaks that firewall: setting it spends
    readout cells asymmetrically and CHANGES ``state_dim`` (4x10x128 = 5120
    instead of 2048), which makes every downstream checkpoint unloadable. It is
    ``None`` by default and should stay that way unless someone is deliberately
    paying that price — this is the cost the FOV-audit stream flagged.
    """

    def __init__(self, n_tokens: int, d_model: int, grid: int = 4,
                 d_readout: int = 32,
                 token_grid: tuple[int, int] | None = None,
                 grid_w: int | None = None):
        super().__init__()
        if token_grid is None:
            hw = int(n_tokens ** 0.5)
            assert hw * hw == n_tokens, (
                f"readout got {n_tokens} tokens, which is not a square grid. "
                f"Pass token_grid=(rows, cols) for a non-square encoder.")
            th, tw = hw, hw
        else:
            th, tw = int(token_grid[0]), int(token_grid[1])
            assert th * tw == n_tokens, (
                f"token_grid {th}x{tw} does not match n_tokens {n_tokens}")
        gw = grid if grid_w is None else int(grid_w)
        self.hw, self.grid = th, grid          # .hw kept: square back-compat
        self.token_h, self.token_w, self.grid_w = th, tw, gw
        # Pooling route (converged with the FOV-audit stream 2026-07-27).
        # MEASURED (route_and_rig_2026-07-27.json): where the token grid tiles
        # evenly, `AvgPool2d((th//grid, tw//gw))` and `AdaptiveAvgPool2d((grid,
        # gw))` are the SAME operation — bit-identical at 16x16 (deployed) and
        # 16x40, and equal to float32 summation noise (<= 6e-8) at 24x60 / 24x24.
        # The exact kernel is used wherever it applies, so the DEPLOYED path
        # stays byte-for-byte what it was; adaptive is the fallback that makes a
        # non-tiling grid (e.g. 16x42) expressible instead of an assert.
        self.exact_pool = (th % grid == 0 and tw % gw == 0)
        self.pool = (nn.AvgPool2d((th // grid, tw // gw)) if self.exact_pool
                     else nn.AdaptiveAvgPool2d((grid, gw)))
        # ⛔ ONNX. `AdaptiveAvgPool2d` with an output size that is not a factor of the input is
        # UNEXPORTABLE — `SymbolicValueError: adaptive_avg_pool2d, output size that are not factor
        # of input size`, at every opset and both MHA-fastpath settings. MEASURED 2026-08-03 on
        # Thor: this is what blocks the ENCODER from exporting at the DEPLOYED 176x624 geometry
        # (11x39 tokens onto a 4x4 readout grid), and therefore blocks backlog item O2 — the
        # designated fallback for the single largest lever in the whole Thor result.
        #
        # The fix is not an approximation. Adaptive pooling with STATIC input and output sizes is a
        # FIXED LINEAR OPERATOR: output bin i averages input rows [floor(i*H/G), ceil((i+1)*H/G)).
        # Materialising that as two constant averaging matrices makes the same computation a pair
        # of matmuls, which every opset exports. The bins, and therefore the numbers, are identical
        # by construction — `tests/test_readout_onnx_pool.py` measures the residual against
        # `F.adaptive_avg_pool2d` rather than asserting it, including at the deployed 11x39 -> 4x4.
        #
        # ⚠️ NOT persistent: these are derived constants, so they must never enter the state_dict.
        # A persistent buffer here would make every existing checkpoint fail a STRICT load.
        if not self.exact_pool:
            self.register_buffer("pool_mh", _adaptive_avg_matrix(th, grid),
                                 persistent=False)
            self.register_buffer("pool_mw", _adaptive_avg_matrix(tw, gw),
                                 persistent=False)
        self.proj = nn.Linear(d_model, d_readout)
        self.out_dim = grid * gw * d_readout

    def forward(self, tokens: Tensor) -> Tensor:
        b, n, d = tokens.shape
        x = tokens.transpose(1, 2).reshape(b, d, self.token_h, self.token_w)
        if self.exact_pool:
            x = self.pool(x)                              # [B, D, Gh, Gw] — deployed path, untouched
        else:
            # [Gh,H] @ [B,D,H,W] -> [B,D,Gh,W] @ [W,Gw] -> [B,D,Gh,Gw]
            mh = self.pool_mh.to(x.dtype)
            mw = self.pool_mw.to(x.dtype)
            x = torch.matmul(torch.matmul(mh, x), mw.transpose(0, 1))
        x = x.flatten(2).transpose(1, 2)                  # [B, Gh*Gw, D]
        return self.proj(x).flatten(1)                    # [B, Gh*Gw*d_r]


class RidgeProbe:
    """Frozen linear readout fitted in closed form: W = (X'X + a I)^-1 X'Y.

    Fit either on real encodings (probe_real: reads current state) or on
    imagined latents (probe_imag: reads imagination — A3 calibration).
    Never trained in-loop; never receives gradients.
    """

    def __init__(self, alpha: float = 1e-3):
        self.alpha = alpha
        self.W: Tensor | None = None
        self.b: Tensor | None = None

    def fit(self, feats: Tensor, targets: Tensor) -> "RidgeProbe":
        """feats [n, f], targets [n, t]. Solves ridge with intercept."""
        with torch.no_grad():
            x = feats.double()
            y = targets.double()
            x_mean, y_mean = x.mean(0), y.mean(0)
            xc, yc = x - x_mean, y - y_mean
            f = xc.shape[1]
            gram = xc.T @ xc + self.alpha * torch.eye(f, dtype=x.dtype, device=x.device)
            self.W = torch.linalg.solve(gram, xc.T @ yc)          # [f, t]
            self.b = y_mean - x_mean @ self.W
        return self

    def predict(self, feats: Tensor) -> Tensor:
        assert self.W is not None, "probe not fitted"
        with torch.no_grad():
            return (feats.double() @ self.W + self.b).to(feats.dtype)

    def r2(self, feats: Tensor, targets: Tensor) -> float:
        pred = self.predict(feats)
        ss_res = (pred - targets).pow(2).sum()
        ss_tot = (targets - targets.mean(0)).pow(2).sum().clamp_min(1e-12)
        return float(1.0 - ss_res / ss_tot)
