"""H-JEPA -- a hierarchy of latent predictive world models, implemented FROM THE PAPER.

Primary source (banked as ``lib:2610.06805``): Zhang, Terver, Rabbat, LeCun, Balestriero,
*H-JEPA: End-to-End Learning of Hierarchical World Models* (arXiv 2610.06805, 5 Oct 2026).
Nothing here was copied from released code; every design choice cites a paper section, and every
place the paper is silent or the PDF text extraction was ambiguous is marked ``UNVERIFIED``.

DEFAULT-OFF BY CONSTRUCTION: this module is pure PyTorch, imports nothing from ``tanitad`` and is
imported by nothing in ``tanitad``. Wiring it into a trainer is a separate, gated step
(``Project Steering/PLAN_V8F_REFAV1_HJEPA_2026-10-17.md`` S5); see
``TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-hjepa-module/RESULT.md``.

What is in here (section numbers are the paper's)
-------------------------------------------------
* :class:`SIGReg`           -- Epps-Pulley statistic averaged over M random unit directions,
                               17-knot trapezoid quadrature on [0, 3] that exploits the even
                               integrand, grouping (T, N, D), n prefactor, and an ECF all-reduce
                               that makes the statistic exactly invariant to data-parallel
                               sharding (App. K, K.3).
* :class:`IDMHead`          -- inverse-dynamics head and the Eq. (8) loss (S4.3, App. L.7).
* :class:`LevelEncoder`     -- level >= 2 state encoder: a 2-layer MLP over the lower-level latent
                               (S2.1, Eq. 1; App. C.1 / C.2).
* :class:`MacroActionEncoder` -- pools the ``s`` lower-level action embeddings of one upper-level
                               transition into one macro-action (Eq. 1; App. C.2).
* :class:`CausalPredictor`  -- causal transformer with adaptive-LayerNorm action conditioning;
                               teacher-forced one-step loss (Eq. 2).
* :class:`HJEPAStack`       -- the levels, the end-to-end loss (Eq. 3 + Eq. 8), autoregressive
                               rollout, goal cost, candidate scoring (Eq. 5), the top-down
                               gradient planner (Eqs. 4-6, K_{l+1} = 1) and two diagnostics:
                               :meth:`HJEPAStack.action_sensitivity` and :func:`rho_within_episode`.

LEVEL-1 ENCODER IS EXTERNAL. ``z^(1)`` is whatever the chassis produces (a ViT CLS projection in
the paper; for TanitAD the pooled scene/BEV latent). The stack takes ``z1_seq`` as an INPUT, so
gradients from every upper-level objective flow into it when the caller keeps the graph
(paper E4: "upper-level losses back-propagate into lower encoders", no EMA, no stop-gradient).

CONVENTIONS
-----------
* A latent is a VECTOR per frame, ``(B, T, D)``. A chassis that produces tokens must flatten or
  pool them before this module (integration item, see RESULT.md).
* ``a_seq[:, t]`` is the action block for the transition ``t -> t+1``, so a clip with ``T1``
  level-1 states has ``T1 - 1`` actions.
* Level numbering is 1-based everywhere (``level=1`` is the finest).
* The "planning action space" of a level is: RAW actions for level 1, MACRO-ACTION embeddings for
  levels >= 2 (App. E: upper-level search is anchored to macro-actions seen in training).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "SIGReg", "IDMHead", "LevelEncoder", "MacroActionEncoder", "CausalPredictor",
    "LevelConfig", "HJEPAEncoding", "PlanResult", "HJEPAStack",
    "rho_within_episode", "donor_indices", "monotonicity", "param_grad_report",
    "dead_parameters",
]


# =============================================================================================
# SIGReg  (App. K, K.3)
# =============================================================================================

class _EcfAllReduceMean(torch.autograd.Function):
    """``y = all_reduce_sum(x) / W`` whose backward is the IDENTITY.

    Why identity and not ``1/W`` or an all-reduce: data-parallel training averages the per-rank
    parameter gradients (``(1/W) * sum_r g_r``). Rank r's surrogate gradient with respect to its
    OWN local ECF is ``dL/dECF_global * (1/W)`` if you differentiate ``y`` honestly, and the DDP
    average would then divide by W a second time. With an identity backward each rank carries
    ``dL/dECF_global * dECF_r/dx_r`` and the DDP average reproduces EXACTLY the single-device
    gradient of the global statistic. (Derivation in ``RESULT.md``; verified numerically in
    ``test_hjepa.py::test_sigreg_gloo_two_process`` against a 1-process reference.)
    """

    @staticmethod
    def forward(ctx, x: Tensor, group) -> Tensor:  # noqa: D401
        import torch.distributed as dist
        y = x.detach().clone().contiguous()
        dist.all_reduce(y, op=dist.ReduceOp.SUM, group=group)
        return y / dist.get_world_size(group)

    @staticmethod
    def backward(ctx, g: Tensor):
        return g, None


def _dist_reduce(cos_m: Tensor, sin_m: Tensor, group=None):
    """Default ``reduce_fn``: average the local ECFs over the data-parallel group.

    Returns ``(cos_global, sin_global, world_size)``. A no-op (``W = 1``) when no process group
    is initialised, so single-device code is unchanged.
    """
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()):
        return cos_m, sin_m, 1
    w = dist.get_world_size(group)
    if w == 1:
        return cos_m, sin_m, 1
    both = _EcfAllReduceMean.apply(torch.stack([cos_m, sin_m]), group)
    return both[0], both[1], w


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian regulariser (Epps-Pulley), Eq. (9) and App. K / K.3.

    ``SIGReg(Z) = (1/M) sum_m T_EP(h_m)`` with ``h_m = Z u_m`` for M random unit directions.

    ``T_EP(h) = n * int |phi_n(t) - phi_0(t)|^2 w(t) dt`` where ``phi_n`` is the empirical
    characteristic function of the n projected samples, ``phi_0(t) = exp(-t^2/2)`` the standard
    normal's and ``w(t) = exp(-t^2/2)`` the Gaussian weighting (Balestriero & LeCun, as cited by
    App. K; the unit window width is UNVERIFIED against the released code -- the paper's own
    floor of 1.04 to 1.37 over n = 32 to 2048 is consistent with it, see ``null_floor``).
    The integral runs over [-3, 3] and is evaluated on the half-domain [0, 3] with 17 trapezoid
    knots, doubling the weights because the integrand is even (App. K).

    Input is grouped ``(T, N, D)``: ONE normality test per timestep over the ``N`` sequences of the
    batch, averaged over ``T`` (Table 5 "SIGReg grouping (T, N, D)"). A 2-D input ``(N, D)`` is a
    single group. With this grouping time is in the GROUP axis, so SIGReg carries NO barrier
    against slow-feature collapse (Prop. 2): that is what the IDM term is for.

    ``n`` PREFACTOR (K.3): the statistic is scaled by the number of samples that entered the test.
    On target data it sits at a floor (``null_floor``, ~1.05) independent of n; on collapsed data
    it grows linearly in n, which keeps the anti-collapse gradient per sample constant. Dividing
    by n instead silently disables the loss (the repo's own historic bug, see ``sigreg.py``).

    Data parallelism (K.3): TEP is a statistic of the whole batch, not a sum of per-sample terms.
    ``forward`` averages the local real/imaginary ECFs across ranks BEFORE the quadrature and
    passes ``n_local * W`` as the prefactor, so the statistic is exactly invariant to how a fixed
    total batch is sharded (equal shards assumed; drop the last partial batch). The reduce is a
    callable ``reduce_fn(cos_mean, sin_mean) -> (cos, sin, world_size)`` so it can be simulated
    in-process (tests) without a process group.
    """

    def __init__(self, num_projections: int = 1024, knots: int = 17, t_max: float = 3.0):
        super().__init__()
        if knots < 3:
            raise ValueError("need at least 3 knots")
        self.num_projections = int(num_projections)
        t = torch.linspace(0.0, t_max, knots, dtype=torch.float64)
        dt = t_max / (knots - 1)
        # trapezoid on [0, t_max] = (dt/2, dt, ..., dt, dt/2); x2 for the even integrand
        # => (dt, 2dt, ..., 2dt, dt).
        w = torch.full((knots,), 2.0 * dt, dtype=torch.float64)
        w[0] = dt
        w[-1] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("phi0", window.clone(), persistent=False)  # N(0,1) ECF == window
        self.register_buffer("qweights", w * window, persistent=False)

    # -- pieces -----------------------------------------------------------------------------
    def null_floor(self) -> float:
        """E[T_EP] on exactly Gaussian data, for ANY n, D, M, T.

        For iid ``h ~ N(0,1)``: ``E|phi_n(t) - phi_0(t)|^2 = (1 - |phi_0(t)|^2) / n``, so after
        the ``n`` prefactor the expectation is ``sum_k w_k (1 - exp(-t_k^2))`` (~1.05).
        """
        return float((self.qweights * (1.0 - torch.exp(-self.t.square()))).sum())

    def sample_directions(self, dim: int, *, device=None, dtype=torch.float32,
                          generator: Optional[torch.Generator] = None) -> Tensor:
        u = torch.randn(dim, self.num_projections, device=device, dtype=dtype, generator=generator)
        return u / u.norm(dim=0, keepdim=True).clamp_min(1e-12)

    def local_ecf(self, x: Tensor, directions: Tensor):
        """Empirical characteristic function of the projections, averaged over the sample axis.

        ``x`` is ``(T, N, D)``; returns ``(cos_mean, sin_mean)`` each ``(T, M, K)`` (the real and
        imaginary parts of ``phi_n``). This is the quantity that crosses the wire in K.3.
        """
        proj = x @ directions                                   # (T, N, M)
        arg = proj.unsqueeze(-1) * self.t.to(proj.dtype)        # (T, N, M, K)
        return torch.cos(arg).mean(dim=1), torch.sin(arg).mean(dim=1)

    def statistic_from_ecf(self, cos_mean: Tensor, sin_mean: Tensor, n_total: int) -> Tensor:
        """Quadrature + prefactor. ``n_total`` is the sample count of the GLOBAL batch."""
        dt = cos_mean.dtype
        err = (cos_mean - self.phi0.to(dt)).square() + sin_mean.square()    # (T, M, K)
        stat = (err * self.qweights.to(dt)).sum(dim=-1) * float(n_total)     # (T, M)
        return stat.mean()

    def forward(self, x: Tensor, *, generator: Optional[torch.Generator] = None,
                directions: Optional[Tensor] = None,
                reduce_fn: Optional[Callable] = None, sync: bool = True) -> Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(f"SIGReg expects (T, N, D) or (N, D); got {tuple(x.shape)}")
        cd = torch.float64 if x.dtype == torch.float64 else torch.float32
        x = x.to(cd)
        auto_dist = False
        if directions is None:
            directions = self.sample_directions(x.shape[-1], device=x.device, dtype=cd,
                                                generator=generator)
            auto_dist = True
        else:
            directions = directions.to(device=x.device, dtype=cd)
        if reduce_fn is None and sync:
            reduce_fn = _dist_reduce
            if auto_dist:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                    # every rank must project onto the SAME directions
                    directions = directions.contiguous()
                    dist.broadcast(directions, src=0)
        cos_m, sin_m = self.local_ecf(x, directions)
        n_total = x.shape[1]
        if reduce_fn is not None:
            cos_m, sin_m, world = reduce_fn(cos_m, sin_m)
            n_total = n_total * int(world)
        return self.statistic_from_ecf(cos_m, sin_m, n_total)


# =============================================================================================
# Inverse dynamics  (Eq. 8, Prop. 3)
# =============================================================================================

class IDMHead(nn.Module):
    """MLP on ``[z_t, z_{t+1}]`` -> action (S4.3).

    ``L_idm = 1/(T-1) sum_t (1/d_a) || I(z_t, z_{t+1}) - sg(a_t) ||^2`` -- Eq. (8). The target is
    stop-gradient'ed (so at level >= 2 the macro-action encoder is NOT trained through this
    term), while the gradient reaches the encoder through the two states. With unit-variance
    actions that are not predictable from episode identity, a slow-feature-collapsed encoder
    has ``L_idm >= 1`` (Prop. 3) -- the O(1) barrier SIGReg lacks.
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 256, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2 * latent_dim, hidden), nn.GELU()]
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, action_dim))
        self.net = nn.Sequential(*layers)
        self.latent_dim = latent_dim
        self.action_dim = action_dim

    def forward(self, z_t: Tensor, z_next: Tensor) -> Tensor:
        return self.net(torch.cat([z_t, z_next], dim=-1))

    def loss(self, z: Tensor, a_target: Tensor) -> Tensor:
        """``z`` is ``(B, T, D)``, ``a_target`` is ``(B, T-1, d_a)`` (detached here)."""
        pred = self(z[:, :-1], z[:, 1:])
        return (pred - a_target.detach()).pow(2).mean()   # mean over d_a, t and batch


# =============================================================================================
# Upper-level encoders  (Eq. 1)
# =============================================================================================

def _windows(z: Tensor, w: int, s: int) -> Tensor:
    """``(N, T, D) -> (N, n, w*D)`` with ``n = (T - w)//s + 1``; window ``i`` = ``z[i*s : i*s+w]``."""
    n_, t_, d_ = z.shape
    if t_ < w:
        raise ValueError(f"need at least window={w} lower-level states, got {t_}")
    n = (t_ - w) // s + 1
    idx = torch.arange(n, device=z.device)[:, None] * s + torch.arange(w, device=z.device)[None, :]
    return z[:, idx].reshape(n_, n, w * d_)


class LevelEncoder(nn.Module):
    """Level >= 2 state encoder ``E^(l)``: a two-layer MLP over the lower-level latent window.

    App. C.1: "a two-layer MLP encoder with hidden width 384, applied to the lower-level latent".
    All experiments use ``w = 1`` (pointwise); the window is concatenated when ``w > 1``.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 384, window: int = 1):
        super().__init__()
        self.in_dim, self.out_dim, self.window = in_dim, out_dim, window
        self.net = nn.Sequential(nn.Linear(window * in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, out_dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MacroActionEncoder(nn.Module):
    """``A^(l)`` for l >= 2: pools ``s`` lower-level action embeddings into one macro-action.

    * ``mode="mlp"`` (DROID, App. C.2): concatenate the ``s`` embeddings -> MLP with one hidden
      layer (width 64) -> ``out_dim`` -> final LayerNorm.
    * ``mode="transformer"`` (single-scene, App. C.1: "a one-layer transformer pools the two
      intervening lower-level action embeddings"): project, add a learned position embedding,
      one pre-LN encoder layer, flatten, linear -> ``out_dim`` -> final LayerNorm. The final
      LayerNorm in this mode is an ASSUMPTION (the paper states it for the MLP variant only).
    """

    def __init__(self, embed_dim: int, stride: int, out_dim: int, hidden: int = 64,
                 mode: str = "mlp", heads: int = 1):
        super().__init__()
        if mode not in ("mlp", "transformer"):
            raise ValueError(f"mode {mode!r}")
        self.mode, self.stride, self.embed_dim, self.out_dim = mode, stride, embed_dim, out_dim
        if mode == "mlp":
            self.net = nn.Sequential(nn.Linear(stride * embed_dim, hidden), nn.GELU(),
                                     nn.Linear(hidden, out_dim))
        else:
            self.proj = nn.Linear(embed_dim, hidden)
            self.pos = nn.Parameter(torch.randn(stride, hidden) * 0.02)
            self.layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=2 * hidden,
                                                    dropout=0.0, batch_first=True, norm_first=True)
            self.out = nn.Linear(stride * hidden, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, e: Tensor) -> Tensor:
        """``e`` is ``(..., s, embed_dim)`` -> ``(..., out_dim)``."""
        lead = e.shape[:-2]
        if self.mode == "mlp":
            y = self.net(e.reshape(*lead, self.stride * self.embed_dim))
        else:
            h = self.proj(e.reshape(-1, self.stride, self.embed_dim)) + self.pos
            h = self.layer(h)
            y = self.out(h.reshape(-1, self.stride * h.shape[-1])).reshape(*lead, self.out_dim)
        return self.norm(y)


# =============================================================================================
# Causal predictor with AdaLN action conditioning  (Eq. 2)
# =============================================================================================

class _AdaLNBlock(nn.Module):
    """Pre-LN transformer block whose LayerNorms are modulated by the action embedding.

    ``x <- x + g1 * Attn(LN(x)(1 + c1) + s1)``, ``x <- x + g2 * MLP(LN(x)(1 + c2) + s2)`` with
    ``(s1, c1, g1, s2, c2, g2) = Linear(SiLU(action_emb))`` -- a per-token adaptive LayerNorm
    (shift, scale, gate). The attention is hand-written on ``scaled_dot_product_attention`` so
    the grad / no-grad code paths are numerically identical (the torch ``nn.MultiheadAttention``
    fast path is not).
    """

    def __init__(self, width: int, heads: int, ff: int, cond_dim: int, dropout: float,
                 adaln_init: str):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads, self.dropout = heads, dropout
        self.ln1 = nn.LayerNorm(width, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(width, elementwise_affine=False)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, ff), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ff, width), nn.Dropout(dropout))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * width))
        lin: nn.Linear = self.mod[1]  # type: ignore[assignment]
        with torch.no_grad():
            if adaln_init == "zero":                  # DiT adaLN-Zero: the block starts as identity
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
            elif adaln_init == "identity_gate":       # default: gates 1, shift/scale ~ 0
                nn.init.normal_(lin.weight, std=0.02)
                nn.init.zeros_(lin.bias)
                lin.bias[2 * width:3 * width] = 1.0   # g1
                lin.bias[5 * width:6 * width] = 1.0   # g2
            else:
                raise ValueError(f"adaln_init {adaln_init!r}")

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        s1, c1, g1, s2, c2, g2 = self.mod(cond).chunk(6, dim=-1)
        b, l, w = x.shape
        h = self.ln1(x) * (1 + c1) + s1
        q, k, v = self.qkv(h).reshape(b, l, 3, self.heads, w // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        x = x + g1 * self.proj(a.transpose(1, 2).reshape(b, l, w))
        h = self.ln2(x) * (1 + c2) + s2
        return x + g2 * self.mlp(h)


class CausalPredictor(nn.Module):
    """``F^(l)``: causal transformer predicting the next latent from past latents and actions.

    ``forward(z_ctx, a_ctx)`` takes ``(B, L, D)`` latents and ``(B, L, E)`` action embeddings and
    returns ``(B, L, D)``: position ``i`` is the prediction of ``z_{i+1}`` given ``z_{<= i}`` and
    ``a_{<= i}`` (teacher forcing, Eq. 2). The planner calls it with ``L = 1`` (context c = 1,
    App. E), which training covers: under the causal mask the first position attends only to
    itself.

    Defaults follow App. C.1 (width 192, 4 layers, 8 heads is the paper's level-1; the tiny
    defaults here are for tests -- set the real sizes in :class:`LevelConfig`).
    ``residual=False``: the network outputs the next latent directly (the paper does not
    describe a skip from ``z_t``); ``residual=True`` is offered only for experiments, because it
    makes the identity predictor -- the slow-feature collapse -- the zero-initialised solution.
    """

    def __init__(self, latent_dim: int, action_dim: int, width: int = 192, depth: int = 4,
                 heads: int = 6, ff: Optional[int] = None, context: int = 3, dropout: float = 0.0,
                 adaln_init: str = "identity_gate", residual: bool = False):
        super().__init__()
        self.latent_dim, self.action_dim, self.context = latent_dim, action_dim, context
        self.residual = residual
        self.in_proj = nn.Linear(latent_dim, width)
        self.pos = nn.Parameter(torch.randn(context, width) * 0.02)
        self.blocks = nn.ModuleList([
            _AdaLNBlock(width, heads, ff or 4 * width, action_dim, dropout, adaln_init)
            for _ in range(depth)])
        self.final_ln = nn.LayerNorm(width)
        self.out = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, latent_dim))

    def forward(self, z_ctx: Tensor, a_ctx: Tensor) -> Tensor:
        L = z_ctx.shape[1]
        if L > self.context:
            raise ValueError(f"context length {L} exceeds the predictor's {self.context}")
        x = self.in_proj(z_ctx) + self.pos[:L]
        for blk in self.blocks:
            x = blk(x, a_ctx)
        y = self.out(self.final_ln(x))
        return y + z_ctx if self.residual else y


# =============================================================================================
# The stack
# =============================================================================================

@dataclass
class LevelConfig:
    """One level of the hierarchy.

    ``latent_dim``   width D_l of this level's latent.
    ``action_dim``   the PLANNING action space of this level: the raw action dim at level 1, the
                     macro-action embedding dim at levels >= 2.
    ``stride``, ``window``   s_l, w_l from level l-1 to l (ignored at level 1). Paper: s = 2-3,
                     w = 1.
    ``act_embed_dim``   level 1 only: if set, ``A^(1)`` is an MLP raw -> this dim; else identity
                     (DROID: "Level 1 consumes raw actions").
    ``beta`` / ``gamma``   SIGReg and IDM weights (paper DROID: level 1 = 0.08 / 100, level 2 =
                     0.16 / 50; the IDM and SIGReg weights must be tuned JOINTLY at level 1, level
                     2 is insensitive -- Fig. 18).
    ``act_sigreg``   weight of SIGReg on the macro-action embeddings, levels >= 2 only (0.005 on
                     DROID, C.2: "to avoid slow-feature collapse").
    """
    latent_dim: int
    action_dim: int
    stride: int = 1
    window: int = 1
    act_embed_dim: Optional[int] = None
    enc_hidden: int = 384
    macro_hidden: int = 64
    macro_mode: str = "mlp"
    pred_width: int = 192
    pred_depth: int = 4
    pred_heads: int = 6
    pred_ff: Optional[int] = None
    pred_dropout: float = 0.0
    context: int = 3
    adaln_init: str = "identity_gate"
    idm_hidden: int = 256
    beta: float = 0.08
    gamma: float = 100.0
    act_sigreg: float = 0.005


@dataclass
class HJEPAEncoding:
    """Per-level tensors (index ``l - 1``): states ``(B, T_l, D_l)``; action embeddings that
    condition the level's predictor ``(B, T_l - 1, E_l)``; the IDM regression targets (level 1:
    the RAW actions; level >= 2: the macro-actions)."""
    z: list
    act: list
    idm_target: list


@dataclass
class PlanResult:
    actions: Tensor                 # (B, H_1, d_a) best level-1 plan
    level_actions: dict = field(default_factory=dict)   # l -> (B, H_l, A_l)
    level_rollouts: dict = field(default_factory=dict)  # l -> (B, H_l, D_l)
    costs: dict = field(default_factory=dict)           # l -> (B,) best cost
    init_costs: dict = field(default_factory=dict)      # l -> (B,) best cost among the INITIAL candidates
    history: dict = field(default_factory=dict)         # l -> [mean best cost per iteration]
    iters: dict = field(default_factory=dict)           # l -> iterations actually run


class HJEPAStack(nn.Module):
    """A stack of JEPA world models, one latent / stride / predictor per level (S2).

    ``levels[0]`` is level 1 (its state encoder lives OUTSIDE this module); ``levels[l-1]`` for
    ``l >= 2`` adds a :class:`LevelEncoder` and a :class:`MacroActionEncoder`. Every level has a
    :class:`CausalPredictor` and an :class:`IDMHead`; SIGReg is applied to every level's states
    and (l >= 2) to its macro-actions. ``predictor_overrides`` swaps in any module with the
    ``(z_ctx, a_ctx) -> z_next`` interface (identity / linear stubs in the tests).
    """

    def __init__(self, levels: Sequence[LevelConfig], *, sigreg_projections: int = 1024,
                 pred_reduction: str = "mean",
                 predictor_overrides: Optional[Mapping[int, nn.Module]] = None):
        super().__init__()
        if not levels:
            raise ValueError("need at least one level")
        if pred_reduction not in ("mean", "sum"):
            raise ValueError("pred_reduction must be 'mean' or 'sum'")
        self.cfg: list[LevelConfig] = list(levels)
        self.pred_reduction = pred_reduction
        L = len(self.cfg)
        for lv, c in enumerate(self.cfg[1:], start=2):
            if c.stride < 1 or c.window < 1:
                raise ValueError(f"level {lv}: stride/window must be >= 1")

        c1 = self.cfg[0]
        if c1.act_embed_dim is None:
            self.act_embed: nn.Module = nn.Identity()
            e1 = c1.action_dim
        else:
            self.act_embed = nn.Sequential(nn.Linear(c1.action_dim, c1.act_embed_dim), nn.GELU(),
                                           nn.Linear(c1.act_embed_dim, c1.act_embed_dim))
            e1 = c1.act_embed_dim
        self.embed_dims = [e1] + [c.action_dim for c in self.cfg[1:]]

        self.encoders = nn.ModuleList([
            LevelEncoder(self.cfg[i - 1].latent_dim, self.cfg[i].latent_dim,
                         self.cfg[i].enc_hidden, self.cfg[i].window) for i in range(1, L)])
        self.macro_encoders = nn.ModuleList([
            MacroActionEncoder(self.embed_dims[i - 1], self.cfg[i].stride, self.cfg[i].action_dim,
                               self.cfg[i].macro_hidden, self.cfg[i].macro_mode)
            for i in range(1, L)])
        preds = []
        for i, c in enumerate(self.cfg):
            lv = i + 1
            if predictor_overrides and lv in predictor_overrides:
                preds.append(predictor_overrides[lv])
            else:
                preds.append(CausalPredictor(c.latent_dim, self.embed_dims[i], c.pred_width,
                                             c.pred_depth, c.pred_heads, c.pred_ff, c.context,
                                             c.pred_dropout, c.adaln_init))
        self.predictors = nn.ModuleList(preds)
        self.idm_heads = nn.ModuleList([
            IDMHead(c.latent_dim, c.action_dim, c.idm_hidden) for c in self.cfg])
        self.sigreg = SIGReg(sigreg_projections)

        # Per-level planning-action prior (App. E): defaults are N(0, 1) with NO clipping.
        for lv, c in enumerate(self.cfg, start=1):
            self.register_buffer(f"prior_mean_{lv}", torch.zeros(c.action_dim))
            self.register_buffer(f"prior_std_{lv}", torch.ones(c.action_dim))
            self.register_buffer(f"prior_lo_{lv}", torch.full((c.action_dim,), -math.inf))
            self.register_buffer(f"prior_hi_{lv}", torch.full((c.action_dim,), math.inf))

    # ---- bookkeeping ----------------------------------------------------------------------
    @property
    def n_levels(self) -> int:
        return len(self.cfg)

    def built_levers(self) -> dict:
        """What the BUILT modules / the loss actually read -- the G-DVB reader (never the config
        that built them, SPEC_REFCV7 §2). Keyed by level."""
        out = {}
        for lv, c in enumerate(self.cfg, start=1):
            pred = self.predictors[lv - 1]
            row = dict(latent_dim=c.latent_dim, action_dim=c.action_dim,
                       context=getattr(pred, "context", None),
                       idm_head_built=self.idm_heads[lv - 1] is not None,
                       idm_live=c.gamma > 0.0, beta=c.beta, gamma=c.gamma,
                       sigreg_live=c.beta > 0.0)
            if lv >= 2:
                enc = self.encoders[lv - 2]
                mac = self.macro_encoders[lv - 2]
                row.update(stride=mac.stride, window=enc.window, act_sigreg=c.act_sigreg,
                           act_sigreg_live=c.act_sigreg > 0.0, macro_mode=mac.mode)
            out[lv] = row
        return out

    def parameter_counts(self) -> dict:
        n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        out = {}
        for lv in range(1, self.n_levels + 1):
            row = dict(predictor=n(self.predictors[lv - 1]), idm=n(self.idm_heads[lv - 1]))
            if lv >= 2:
                row.update(encoder=n(self.encoders[lv - 2]), macro=n(self.macro_encoders[lv - 2]))
            else:
                row.update(act_embed=n(self.act_embed))
            out[lv] = row
        out["total"] = n(self)
        return out

    def param_groups(self, lr: float, upper_lr_mult: float = 0.2, weight_decay: float = 1e-3) -> list:
        """Optimizer groups: level 1 at ``lr``, every upper level at ``lr * upper_lr_mult``.

        Paper E4: peak LR 5e-5 at level 1 and 1e-5 (a fifth) at every upper level, AdamW with
        weight decay 1e-3 (App. C.1). The level-1 ENCODER is external and must be added by the
        caller. Every parameter of the stack appears in exactly one group.
        """
        g1 = [p for m in (self.act_embed, self.predictors[0], self.idm_heads[0])
              for p in m.parameters()]
        groups = [dict(params=g1, lr=lr, weight_decay=weight_decay, name="level1")]
        for lv in range(2, self.n_levels + 1):
            ps = [p for m in (self.encoders[lv - 2], self.macro_encoders[lv - 2],
                              self.predictors[lv - 1], self.idm_heads[lv - 1])
                  for p in m.parameters()]
            groups.append(dict(params=ps, lr=lr * upper_lr_mult, weight_decay=weight_decay,
                               name=f"level{lv}"))
        groups = [g for g in groups if g["params"]]
        return groups

    # ---- structure: Eq. (1) -----------------------------------------------------------------
    def _apply_encoder(self, level: int, z: Tensor, *, strict: bool = False) -> Tensor:
        """Lift ``(..., T, D_{l-1})`` to ``(..., n, D_l)``: ``z^l_i = E^l(z^{l-1}_{i s : i s + w})``."""
        c = self.cfg[level - 1]
        lead = z.shape[:-2]
        t_, d_ = z.shape[-2:]
        if strict and (t_ - c.window) % c.stride != 0:
            raise ValueError(
                f"level {level}: a lower-level trajectory of {t_} states (incl. the initial one) "
                f"does not end on a window boundary (window={c.window}, stride={c.stride}); the "
                f"paper's horizon is H = K*s + w - 1, so the trailing steps would not affect the "
                f"cost")
        win = _windows(z.reshape(-1, t_, d_), c.window, c.stride)
        out = self.encoders[level - 2](win)
        return out.reshape(*lead, out.shape[-2], out.shape[-1])

    def lift(self, z_traj: Tensor, from_level: int, to_level: int, *, strict: bool = True) -> Tensor:
        """Eq. (5): push a level-``from_level`` trajectory (INCLUDING its initial state at index 0)
        through the upper encoders. Output index ``i`` of level ``l`` is the lift of lower-level
        index ``i * s_l`` (``w = 1``), so index 0 is the lifted initial state."""
        z = z_traj
        for lv in range(from_level + 1, to_level + 1):
            z = self._apply_encoder(lv, z, strict=strict)
        return z

    def initial_latents(self, z1_0: Tensor, upto: Optional[int] = None) -> list:
        """Encode the current state at every level (pointwise encoders only, ``w = 1``)."""
        upto = upto or self.n_levels
        zs = [z1_0]
        for lv in range(2, upto + 1):
            if self.cfg[lv - 1].window != 1:
                raise NotImplementedError(
                    "initial_latents needs a window of lower-level states when w > 1; "
                    "the paper plans with w = 1 at every level")
            zs.append(self.encoders[lv - 2](zs[-1]))
        return zs

    def encode(self, z1_seq: Tensor, a_seq: Tensor) -> HJEPAEncoding:
        """Per-level latents and action embeddings from a level-1 sequence (Eq. 1).

        ``z1_seq`` ``(B, T1, D1)``, ``a_seq`` ``(B, T1 - 1, d_a)``. Level ``l`` has
        ``T_l = (T_{l-1} - w) // s + 1`` states and ``T_l - 1`` macro-actions; macro-action ``t``
        pools the lower-level embeddings ``[t s + w - 1, (t + 1) s + w - 1)``. Trailing lower-level
        states that do not complete a window are dropped.
        """
        if z1_seq.dim() != 3 or a_seq.dim() != 3:
            raise ValueError("z1_seq (B,T,D) and a_seq (B,T-1,d_a) are required")
        if a_seq.shape[1] != z1_seq.shape[1] - 1:
            raise ValueError(f"a_seq has {a_seq.shape[1]} blocks for {z1_seq.shape[1]} states; "
                             f"expected T-1 = {z1_seq.shape[1] - 1}")
        if z1_seq.shape[-1] != self.cfg[0].latent_dim or a_seq.shape[-1] != self.cfg[0].action_dim:
            raise ValueError("z1_seq / a_seq widths do not match level 1's LevelConfig")
        zs, acts, tgts = [z1_seq], [self.act_embed(a_seq)], [a_seq]
        for lv in range(2, self.n_levels + 1):
            c = self.cfg[lv - 1]
            z_up = self._apply_encoder(lv, zs[-1])
            n = z_up.shape[1]
            if n < 2:
                raise ValueError(f"clip too short for level {lv}: only {n} state(s); need >= 2 "
                                 f"(window={c.window}, stride={c.stride})")
            e = acts[-1]
            lo = c.window - 1
            e_sub = e[:, lo:lo + (n - 1) * c.stride]
            e_grp = e_sub.reshape(e.shape[0], n - 1, c.stride, e.shape[-1])
            a_up = self.macro_encoders[lv - 2](e_grp)
            zs.append(z_up)
            acts.append(a_up)
            tgts.append(a_up)
        return HJEPAEncoding(zs, acts, tgts)

    # ---- the loss: Eq. (2), (3), (8) --------------------------------------------------------
    def _pred_loss(self, level: int, z: Tensor, a: Tensor, detach_target: bool) -> Tensor:
        """Teacher-forced one-step loss, Eq. (2), equal weight per predicted position.

        Sequences longer than the predictor's context are processed in consecutive chunks of
        ``c`` positions (the first position of each chunk attends only to itself, as at planning
        time). ``pred_reduction='mean'`` averages the squared error over the latent width (the
        magnitudes in Fig. 10 -- prediction loss 0.01 to 0.2 on SIGReg-whitened latents of width
        64 to 384 -- are only consistent with a per-dimension mean; Eq. (2) written literally is
        the sum. UNVERIFIED against the released code).
        """
        pred = self.predictors[level - 1]
        c = getattr(pred, "context", None) or (z.shape[1] - 1)
        total, count = z.new_zeros(()), 0
        t_ = z.shape[1]
        for s in range(0, t_ - 1, c):
            e = min(s + c, t_ - 1)
            out = pred(z[:, s:e], a[:, s:e])
            tgt = z[:, s + 1:e + 1]
            if detach_target:
                tgt = tgt.detach()
            sq = (out - tgt).pow(2)
            per = sq.mean(dim=-1) if self.pred_reduction == "mean" else sq.sum(dim=-1)
            total = total + per.sum()
            count += per.numel()
        return total / max(count, 1)

    def loss(self, z1_seq: Tensor, a_seq: Tensor, *, generator: Optional[torch.Generator] = None,
             detach_target: bool = False) -> dict:
        """End-to-end objective ``sum_l [ L_pred + beta_l SIGReg(Z^l) + gamma_l L_idm ]`` (Eq. 3
        and Eq. 8) plus ``act_sigreg_l SIGReg(A^l)`` for l >= 2.

        Returns a flat dict: ``total`` (differentiable, the sum of the weighted terms), the
        UNWEIGHTED terms ``pred/L{l}``, ``sigreg/L{l}``, ``idm/L{l}``, ``act_sigreg/L{l}``
        (l >= 2) and the weighted per-level sums ``level/L{l}``. ``detach_target=True`` computes
        the prediction targets without a graph -- paper K.1 measures that detachment changes
        nothing, and it saves the backward through the target frames.
        """
        enc = self.encode(z1_seq, a_seq)
        out: dict = {}
        total = z1_seq.new_zeros(()).to(torch.float32)
        for lv in range(1, self.n_levels + 1):
            c = self.cfg[lv - 1]
            z, a = enc.z[lv - 1], enc.act[lv - 1]
            pred = self._pred_loss(lv, z, a, detach_target)
            sg = self.sigreg(z.transpose(0, 1), generator=generator)
            idm = self.idm_heads[lv - 1].loss(z, enc.idm_target[lv - 1])
            level_loss = pred + c.beta * sg + c.gamma * idm
            out[f"pred/L{lv}"], out[f"sigreg/L{lv}"], out[f"idm/L{lv}"] = pred, sg, idm
            if lv >= 2:
                asg = self.sigreg(a.transpose(0, 1), generator=generator)
                out[f"act_sigreg/L{lv}"] = asg
                level_loss = level_loss + c.act_sigreg * asg
            out[f"level/L{lv}"] = level_loss
            total = total + level_loss
        out["total"] = total
        return out

    def forward(self, z1_seq: Tensor, a_seq: Tensor, **kw) -> dict:
        return self.loss(z1_seq, a_seq, **kw)

    # ---- rollout, costs, scoring -------------------------------------------------------------
    def rollout(self, level: int, z0: Tensor, actions: Tensor, *, context: int = 1) -> Tensor:
        """Autoregressive roll ``Roll^(l)`` (S2.2): ``(z_1, ..., z_H)`` from ``z0`` under ``actions``.

        ``actions`` is ``(..., H, A_l)`` in the level's PLANNING action space (raw at level 1,
        macro-action embeddings above). ``z0`` is ``(D_l,)`` (broadcast) or ``lead + (D_l,)``.
        Each step conditions the predictor on the last ``context`` states (planning uses
        ``context = 1``, App. E). Differentiable in both arguments.
        """
        pred = self.predictors[level - 1]
        cap = getattr(pred, "context", None)
        if cap is not None and context > cap:
            raise ValueError(f"context {context} exceeds level {level}'s predictor context {cap}")
        lead, (h_, a_dim) = actions.shape[:-2], actions.shape[-2:]
        a = actions.reshape(-1, h_, a_dim)
        n = a.shape[0]
        if z0.dim() == 1:
            z = z0.expand(n, -1)
        else:
            if tuple(z0.shape[:-1]) != tuple(lead):
                raise ValueError(f"z0 {tuple(z0.shape)} must be (D,) or match actions' leading dims {tuple(lead)}")
            z = z0.reshape(n, -1)
        e = self.act_embed(a) if level == 1 else a
        zs = [z]
        for h in range(h_):
            lo = max(0, h + 1 - context)
            zc = torch.stack(zs[lo:h + 1], dim=1)
            nxt = pred(zc, e[:, lo:h + 1])[:, -1]
            zs.append(nxt)
        out = torch.stack(zs[1:], dim=1)
        return out.reshape(*lead, h_, out.shape[-1])

    def goal_cost(self, z_pred: Tensor, goal: Tensor, level: int, last_k: Optional[int] = None) -> Tensor:
        """Squared Euclidean distance of the final predicted state(s) to ``goal`` (Eq. 4).

        App. E: "The latent-distance cost scores the last two predicted states at level 1 and the
        last state at upper levels"; the mean over those states is used (sum vs mean is
        UNVERIFIED). ``z_pred`` ``(..., H, D)``, ``goal`` ``(..., D)`` -> ``(...)``.
        """
        k = last_k or (2 if level == 1 else 1)
        k = min(k, z_pred.shape[-2])
        diff = z_pred[..., -k:, :] - goal.unsqueeze(-2)
        return diff.pow(2).sum(dim=-1).mean(dim=-1)

    def score_candidates(self, z1_t: Tensor, candidate_action_blocks: Tensor, goal_latent: Tensor,
                         level: int = 2, *, subgoals: Optional[Tensor] = None, lam: float = 0.0,
                         last_k: Optional[int] = None, return_details: bool = False):
        """Score S level-1 candidate plans by their distance to a goal in the level-``level`` latent.

        Rolls level 1 under each candidate, lifts the trajectory through the upper encoders
        (Eq. 5) and takes the squared distance of the final lifted state to ``goal_latent``
        (E2: the abstract-latent goal cost). ``candidate_action_blocks`` is ``(B, S, H1, d_a)``,
        ``z1_t`` ``(B, D1)``, ``goal_latent`` ``(B, D_level)``. ``H1`` must satisfy the paper's
        ``H = K s + w - 1`` alignment at every lifted level (``ValueError`` otherwise).
        Optional ``subgoals`` ``(B, K', D_level)`` + ``lam`` add Eq. (6)'s intermediate terms
        against the lifted states ``1..K'``. Returns the cost ``(B, S)`` (lower is better).
        """
        if candidate_action_blocks.dim() != 4:
            raise ValueError("candidate_action_blocks must be (B, S, H1, d_a)")
        b, s_, h_, _ = candidate_action_blocks.shape
        z0 = z1_t.unsqueeze(1).expand(b, s_, -1)
        roll = self.rollout(1, z0, candidate_action_blocks)                       # (B,S,H,D1)
        full = torch.cat([z0.unsqueeze(2), roll], dim=2)                          # (B,S,H+1,D1)
        lifted = self.lift(full, 1, level, strict=True)                           # (B,S,n,D_l)
        if lifted.shape[2] < 2:
            raise ValueError("horizon too short: no lifted state beyond the initial one")
        pred = lifted[:, :, 1:]
        cost = self.goal_cost(pred, goal_latent.unsqueeze(1), level, last_k)
        if subgoals is not None and lam != 0.0:
            kk = subgoals.shape[1]
            d = (pred[:, :, :kk] - subgoals.unsqueeze(1)).pow(2).sum(-1)
            cost = cost + lam * d.sum(-1)
        if return_details:
            return cost, dict(rollout=roll, lifted=lifted)
        return cost

    # ---- action prior (App. E) ---------------------------------------------------------------
    def set_action_prior(self, level: int, mean=None, std=None, lo=None, hi=None) -> None:
        for name, val in (("mean", mean), ("std", std), ("lo", lo), ("hi", hi)):
            if val is not None:
                buf = getattr(self, f"prior_{name}_{level}")
                buf.copy_(torch.as_tensor(val, dtype=buf.dtype, device=buf.device).expand_as(buf))

    def fit_action_prior(self, level: int, actions: Tensor, *, lo_q: float = 0.02,
                         hi_q: float = 0.98, max_keep: int = 4000) -> None:
        """Fit the per-dimension Gaussian + percentile box to the last ``max_keep`` training
        actions of ``level`` (App. E: the upper levels' search is anchored to the macro-actions
        seen in training and clipped to their 2nd-98th percentiles)."""
        a = actions.detach().reshape(-1, actions.shape[-1])[-max_keep:].float()
        self.set_action_prior(level, mean=a.mean(0), std=a.std(0).clamp_min(1e-6),
                              lo=torch.quantile(a, lo_q, dim=0), hi=torch.quantile(a, hi_q, dim=0))

    # ---- the top-down planner: Eqs. (4)-(6) ---------------------------------------------------
    def _optimize(self, cost_fn: Callable[[Tensor], Tensor], init: Tensor, *, lr: float,
                  n_iters: int, patience: int, rel_tol: float, weight_decay: float,
                  lo: Tensor, hi: Tensor, project: Optional[Callable[[Tensor], Tensor]]):
        """AdamW on the actions only (gradients taken w.r.t. ``a``; model parameters are never
        touched), at most ``n_iters`` steps, early stop when the batch-mean best cost stalls for
        ``patience`` steps, returning the best iterate of every candidate (App. E)."""
        a = init.detach().clone().requires_grad_(True)
        opt = torch.optim.AdamW([a], lr=lr, weight_decay=weight_decay)
        best_cost = torch.full(init.shape[:2], math.inf, device=init.device, dtype=init.dtype)
        best_a = init.detach().clone()
        init_cost, hist, stall, prev = None, [], 0, math.inf

        def _track(cost: Tensor, cur: Tensor):
            nonlocal best_cost, best_a
            better = cost.detach() < best_cost
            best_cost = torch.where(better, cost.detach(), best_cost)
            best_a = torch.where(better[..., None, None], cur.detach(), best_a)

        for it in range(n_iters):
            cost = cost_fn(a)
            if init_cost is None:
                init_cost = cost.detach().min(dim=1).values
            _track(cost, a)
            cur = best_cost.mean().item()
            hist.append(cur)
            if it > 0:
                stall = stall + 1 if (prev - cur) <= rel_tol * max(abs(prev), 1e-12) else 0
                if stall >= patience:
                    break
            prev = cur
            (g,) = torch.autograd.grad(cost.sum(), a)
            a.grad = g
            opt.step()
            with torch.no_grad():
                a.copy_(torch.max(torch.min(a, hi), lo))
                if project is not None:
                    a.copy_(project(a))
        with torch.no_grad():
            final = cost_fn(a)
            if init_cost is None:
                init_cost = final.min(dim=1).values
            _track(final, a)
        return best_a, best_cost, init_cost, hist, len(hist)

    def plan_topdown(self, z1_0: Tensor, goal: Tensor, *, top_horizon: int,
                     goal_level: Optional[int] = None, num_candidates=16, n_iters: int = 90,
                     lr=0.1, lam: float = 0.0, subgoal_K: Optional[Mapping[int, int]] = None,
                     patience: int = 10, rel_tol: float = 1e-4, weight_decay: float = 0.0,
                     bounds: Optional[Mapping[int, tuple]] = None,
                     project_fn: Optional[Callable[[int, Tensor], Tensor]] = None,
                     init_actions: Optional[Mapping[int, Tensor]] = None, init_noise: float = 0.0,
                     last_k: Optional[int] = None,
                     generator: Optional[torch.Generator] = None) -> PlanResult:
        """Gradient-based hierarchical planning through the learned levels (S2.2, Eqs. 4-6).

        The top planned level (``goal_level``, default the highest) minimises the squared distance
        of its final predicted state to ``goal`` (Eq. 4) over macro-actions. Each lower level
        ``l`` then minimises, over its own actions, the distance between its rollout LIFTED into
        level ``l+1`` (Eq. 5) and the first ``K_{l+1}`` states the level above predicted (the
        subgoals, Eq. 6; ``K = 1`` is closed-loop planning, the default); intermediate subgoals
        get weight ``lam``. The lower-level horizon is derived: ``H_l = K s_{l+1} + w_{l+1} - 1``.
        Level 1 returns primitive actions.

        ``goal`` lives in the latent space of ``goal_level`` (for TanitAD: a PREDICTED goal latent;
        its admissibility is the caller's, see PLAN §4.3). Solver (App. E): AdamW, ``n_iters``
        (paper: at most 90) with early stopping, candidates from the level's Gaussian action prior
        (candidate 0 at the mean), clipped after every step to ``bounds[level] = (lo, hi)`` (box /
        kappa bounds; per-channel, broadcast over time) or the fitted prior's percentile box, and
        an optional ``project_fn(level, actions) -> actions`` for non-box constraints (e.g. a
        friction-circle clip). ``init_actions[level]`` ``(B, S, H, A)`` or ``(B, H, A)`` seeds the
        search (use it to REFINE an existing plan: ``num_candidates=1``); further candidates are
        the seed plus ``init_noise`` Gaussian noise.

        ``z1_0`` ``(B, D1)`` or ``(D1,)``; ``goal`` ``(B, D_goal)`` or ``(D_goal,)``. Runs in eval
        mode (dropout off) and restores the previous mode.
        """
        top = goal_level or self.n_levels
        if z1_0.dim() == 1:
            z1_0 = z1_0.unsqueeze(0)
        if goal.dim() == 1:
            goal = goal.unsqueeze(0)
        b = z1_0.shape[0]
        was_training = self.training
        self.eval()
        try:
            with torch.enable_grad():
                z0 = [z.detach() for z in self.initial_latents(z1_0.detach(), top)]
                goal = goal.detach()
                res = PlanResult(actions=z1_0)
                sub_k = dict(subgoal_K or {})

                def _opt(level, horizon, cost_fn, n_cand):
                    c = self.cfg[level - 1]
                    dev, dt = z1_0.device, z1_0.dtype
                    mean = getattr(self, f"prior_mean_{level}").to(dev, dt)
                    std = getattr(self, f"prior_std_{level}").to(dev, dt)
                    lo = getattr(self, f"prior_lo_{level}").to(dev, dt)
                    hi = getattr(self, f"prior_hi_{level}").to(dev, dt)
                    if bounds and level in bounds:
                        lo = torch.as_tensor(bounds[level][0], device=dev, dtype=dt).expand(c.action_dim)
                        hi = torch.as_tensor(bounds[level][1], device=dev, dtype=dt).expand(c.action_dim)
                    if init_actions and level in init_actions:
                        seed = init_actions[level].to(dev, dt)
                        if seed.dim() == 3:
                            seed = seed.unsqueeze(1)
                        if seed.shape[2] != horizon:
                            raise ValueError(f"init_actions[{level}] horizon {seed.shape[2]} != {horizon}")
                        if seed.shape[1] < n_cand:
                            extra = seed[:, :1].expand(b, n_cand - seed.shape[1], horizon, -1)
                            noise = init_noise * torch.randn(extra.shape, device=dev, dtype=dt,
                                                             generator=generator)
                            seed = torch.cat([seed, extra + noise], dim=1)
                        init = seed[:, :n_cand]
                    else:
                        noise = torch.randn(b, n_cand, horizon, c.action_dim, device=dev, dtype=dt,
                                            generator=generator)
                        init = mean + std * noise
                        init[:, 0] = mean.expand(b, horizon, c.action_dim)
                    init = torch.max(torch.min(init, hi), lo)
                    proj = (lambda a, _l=level: project_fn(_l, a)) if project_fn else None
                    if proj is not None:                       # the SEED must be feasible too
                        init = proj(init)
                    lr_l = lr[level] if isinstance(lr, Mapping) else lr
                    return self._optimize(cost_fn, init, lr=float(lr_l), n_iters=n_iters,
                                          patience=patience, rel_tol=rel_tol,
                                          weight_decay=weight_decay, lo=lo, hi=hi, project=proj)

                for level in range(top, 0, -1):
                    n_cand = num_candidates[level] if isinstance(num_candidates, Mapping) else int(num_candidates)
                    d_lv = z0[level - 1]
                    if level == top:
                        horizon = int(top_horizon)

                        def cost_fn(a, _lv=level, _d=d_lv):
                            z0e = _d.unsqueeze(1).expand(b, a.shape[1], -1)
                            roll = self.rollout(_lv, z0e, a)
                            return self.goal_cost(roll, goal.unsqueeze(1), _lv, last_k)
                    else:
                        up = self.cfg[level]                       # config of level + 1
                        K = int(sub_k.get(level, 1))
                        if K < 1 or K > res.level_rollouts[level + 1].shape[1]:
                            raise ValueError(f"subgoal_K[{level}]={K} outside 1..H_{level + 1}")
                        horizon = K * up.stride + up.window - 1
                        sub = res.level_rollouts[level + 1][:, :K].detach()      # (B,K,D_up)

                        def cost_fn(a, _lv=level, _d=d_lv, _sub=sub, _K=K):
                            z0e = _d.unsqueeze(1).expand(b, a.shape[1], -1)
                            roll = self.rollout(_lv, z0e, a)
                            full = torch.cat([z0e.unsqueeze(2), roll], dim=2)
                            lifted = self._apply_encoder(_lv + 1, full, strict=True)[:, :, 1:]
                            dd = (lifted - _sub.unsqueeze(1)).pow(2).sum(-1)       # (B,S,K)
                            return dd[..., -1] + lam * dd[..., :-1].sum(-1)

                    best_a, best_cost, init_cost, hist, nit = _opt(level, horizon, cost_fn, n_cand)
                    pick = best_cost.argmin(dim=1)
                    ar = torch.arange(b, device=pick.device)
                    chosen = best_a[ar, pick]
                    with torch.no_grad():
                        roll = self.rollout(level, d_lv, chosen)
                    res.level_actions[level] = chosen.detach()
                    res.level_rollouts[level] = roll.detach()
                    res.costs[level] = best_cost[ar, pick].detach()
                    res.init_costs[level] = init_cost.detach()
                    res.history[level] = hist
                    res.iters[level] = nit
                res.actions = res.level_actions[1]
                return res
        finally:
            self.train(was_training)

    # ---- diagnostics -----------------------------------------------------------------------
    @torch.no_grad()
    def action_sensitivity(self, z_start: Tensor, actions: Tensor, *, level: int = 1,
                           ks: Sequence[int] = (1, 5, 10),
                           channels: Optional[Mapping[str, Sequence[int]]] = None,
                           perm: Optional[Tensor] = None, episode_ids: Optional[Tensor] = None,
                           seed: int = 0, eps: float = 1e-12, per_window: bool = False) -> dict:
        """Relative rollout change when (a channel of) the actions is swapped for another
        episode's -- ``PREREG_HJ_STAGE0.md`` H-HJ-COLLAPSE-1 / paper Tab. 12.

        ``sens_k(c) = mean over windows of || z^_k(a with channel c from the donor) - z^_k(a) ||_2
        / || z^_k(a) - z_start ||_2`` (``z^_k`` = the k-step rollout of level ``level``). The
        denominator is floored at ``eps`` so an identity predictor reads exactly 0.0 (0 / eps),
        never NaN. ``actions`` is ``(B, H, A_l)`` in the level's planning space, ``z_start``
        ``(B, D_l)``; each of the B windows is one swap unit. The donor is a fixed permutation:
        ``perm`` if given, else :func:`donor_indices` (``episode_ids`` -> a donor from a DIFFERENT
        episode; ``seed`` fixes it). ``channels`` maps a name to action-channel indices (default
        ``{"all": every channel}``); only those channels are swapped.

        Returns ``{"sens/<name>/k<k>": mean-of-ratios, "sens_rom/<name>/k<k>": ratio-of-means}``
        as Python floats. The mean-of-ratios is the PREREG definition; the ratio-of-means is
        reported next to it because the former is dominated by near-stationary windows.

        ``per_window=True`` (RA-1, 2026-10-17; default False = unchanged behaviour) ALSO returns
        ``"win/<name>/k<k>"`` = the per-window ratios as a numpy array ``(B,)``, so a caller can put
        an episode-cluster interval on them. Their mean is exactly ``sens/<name>/k<k>``.
        """
        was_training = self.training
        self.eval()
        try:
            b, h_, a_dim = actions.shape
            if max(ks) > h_:
                raise ValueError(f"k={max(ks)} exceeds the {h_} supplied action steps")
            if perm is None:
                perm = donor_indices(b, episode_ids=episode_ids, seed=seed).to(actions.device)
            channels = dict(channels) if channels else {"all": list(range(a_dim))}
            base = self.rollout(level, z_start, actions)
            disp = (base - z_start.unsqueeze(1)).norm(dim=-1)                    # (B,H)
            out: dict = {}
            for name, idx in channels.items():
                swapped = actions.clone()
                ix = torch.as_tensor(list(idx), device=actions.device, dtype=torch.long)
                swapped[..., ix] = actions[perm][..., ix]
                alt = self.rollout(level, z_start, swapped)
                delta = (alt - base).norm(dim=-1)                                # (B,H)
                for k in ks:
                    num, den = delta[:, k - 1], disp[:, k - 1]
                    out[f"sens/{name}/k{k}"] = float((num / den.clamp_min(eps)).mean())
                    out[f"sens_rom/{name}/k{k}"] = float(num.mean() / den.mean().clamp_min(eps))
                    if per_window:
                        out[f"win/{name}/k{k}"] = (num / den.clamp_min(eps)).cpu().numpy()
            return out
        finally:
            self.train(was_training)


# =============================================================================================
# Free functions
# =============================================================================================

def rho_within_episode(z: Tensor, episode_ids: Optional[Tensor] = None) -> float:
    """Within-episode variance share, paper Eq. (10): ``E_b[tr Var_t(z_t | b)] / tr Var_{b,t}(z_t)``.

    ``z`` is ``(E, T, D)`` (E episodes of T steps) or, with ``episode_ids`` ``(n,)``, a flat
    ``(n, D)`` (ragged episodes). 0 at exact slow-feature collapse (``z_t`` constant within every
    episode); 1 minus the between-episode share otherwise. Population variances (``unbiased=False``)
    throughout, so ``rho + between-share == 1`` exactly for equal-length episodes.
    """
    z = z.detach().double()
    if episode_ids is None:
        if z.dim() != 3:
            raise ValueError("z must be (E, T, D) when episode_ids is None")
        within = z.var(dim=1, unbiased=False).sum(dim=-1).mean()
        total = z.reshape(-1, z.shape[-1]).var(dim=0, unbiased=False).sum()
        return float(within / total)
    if z.dim() != 2:
        raise ValueError("z must be (n, D) when episode_ids is given")
    ids = episode_ids.reshape(-1)
    within_sum, n = 0.0, 0
    uniq = torch.unique(ids)
    for e in uniq:
        ze = z[ids == e]
        within_sum += float(ze.var(dim=0, unbiased=False).sum())
        n += 1
    total = float(z.var(dim=0, unbiased=False).sum())
    return (within_sum / n) / total


def donor_indices(n: int, *, episode_ids: Optional[Tensor] = None, seed: int = 0) -> Tensor:
    """A fixed donor assignment for the action-swap diagnostic.

    Without ``episode_ids``: a seeded random derangement (no window is its own donor; ``n >= 2``).
    With ``episode_ids``: a seeded derangement over EPISODES; window ``i`` of episode ``e`` takes
    the window of episode ``pi(e)`` with the same within-episode rank (modulo that episode's
    window count), so every donor comes from a different episode. Needs >= 2 episodes.
    """
    g = torch.Generator().manual_seed(seed)
    if episode_ids is None:
        if n < 2:
            raise ValueError("need at least 2 windows")
        p = torch.randperm(n, generator=g)
        for i in range(n):
            if p[i] == i:
                j = (i + 1) % n
                p[i], p[j] = p[j].clone(), p[i].clone()
        return p
    ids = episode_ids.detach().cpu().reshape(-1)
    uniq = torch.unique(ids)
    ne = len(uniq)
    if ne < 2:
        raise ValueError("need at least 2 episodes")
    pe = torch.randperm(ne, generator=g)
    for i in range(ne):
        if pe[i] == i:
            j = (i + 1) % ne
            pe[i], pe[j] = pe[j].clone(), pe[i].clone()
    members = [torch.nonzero(ids == u).reshape(-1) for u in uniq]
    out = torch.empty(len(ids), dtype=torch.long)
    for ei, mem in enumerate(members):
        donor_mem = members[int(pe[ei])]
        for rank, wi in enumerate(mem.tolist()):
            out[wi] = donor_mem[rank % len(donor_mem)]
    return out


def monotonicity(costs: Tensor) -> Tensor:
    """Goal-cost monotonicity along a trajectory: the NEGATIVE Spearman correlation between the
    cost and time (``costs`` is ``(..., T)``; 1.0 = perfectly decreasing). The statistic of paper
    Tab. 10 / ``PREREG_HJ_STAGE0.md`` BAR-HJ-COST. Ordinal ranks (ties broken by order)."""
    c = costs.detach().double()
    r = c.argsort(dim=-1).argsort(dim=-1).double()
    t = torch.arange(c.shape[-1], dtype=torch.float64, device=c.device)
    rc, tc = r - r.mean(-1, keepdim=True), t - t.mean()
    corr = (rc * tc).sum(-1) / (rc.norm(dim=-1) * tc.norm()).clamp_min(1e-12)
    return -corr


def param_grad_report(module: nn.Module) -> dict:
    """``{name: sum |grad|}`` for every parameter; a parameter with NO ``.grad`` tensor reads
    exactly 0.0 (so a deliberately unused control parameter is a true zero, not a missing key).
    Call after ``backward()``. The G-LIVE building block: ``dead_parameters`` lists the zeros."""
    return {n: (0.0 if p.grad is None else float(p.grad.abs().sum()))
            for n, p in module.named_parameters()}


def dead_parameters(module: nn.Module, report: Optional[dict] = None) -> list:
    """Names of trainable parameters whose gradient is exactly 0.0 (or absent)."""
    rep = report if report is not None else param_grad_report(module)
    return sorted(n for n, p in module.named_parameters() if p.requires_grad and rep[n] == 0.0)
