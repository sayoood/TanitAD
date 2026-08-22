"""Operative predictor: causal, action-conditioned, residual, multi-horizon.

Validated design (A4 bake-off): residual/delta prediction + change-weighted
latent loss beat plain MSE (0.97 vs 0.71) and flow sampling (0.44) for
action-conditioned short-horizon control. Multi-horizon heads (k in {1,2,4})
double as a training signal and an inference accelerator (MTP, H5).

Action conditioning is FiLM (scale/shift per layer) on continuous
(steer, accel) — cond_proj accepts arbitrary action_dim (H12 command
embeddings concatenate here later).
"""

from __future__ import annotations

#: Multiplier applied to the DEFAULT init of the residual delta heads.
#: Chosen so the initial delta is comparable to the latent's own movement rather
#: than ~1000x above it, while keeping the head non-zero so gradient still
#: reaches the predictor body (see OperativePredictor.__init__).
RESIDUAL_HEAD_INIT_SCALE = 1e-3

import torch
from torch import Tensor, nn

from tanitad.config import PredictorConfig
from tanitad.models._validate import validate_operative_inputs


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, d: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * d)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        return x * (1.0 + scale) + shift


class CausalBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.film = FiLM(cond_dim, d)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: Tensor, cond: Tensor, mask: Tensor) -> Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        x = x + self.mlp(self.film(self.norm2(x), cond))
        return x


class ModernCausalBlock(nn.Module):
    """The ViT-5-recipe counterpart of :class:`CausalBlock` (PI 2026-08-13).

    RMSNorm for both norms · QK-Norm inside attention · LayerScale (1e-5) on
    both residual branches · no bias on qkv/proj · SDPA attention · GeLU MLP
    kept (ViT-5 measured SwiGLU HURTING compact models against LayerScale).
    The FiLM conditioning seam is IDENTICAL to CausalBlock — same zero-init,
    same placement before the MLP — so every conditioning property that is
    tested (identity start, intent reachability, the g_tac seam) carries over
    unchanged. At 12 layers deep this is where ViT-5's ablations say QK-Norm
    and LayerScale stop being cosmetic; at the old depth 6 they were optional.
    """

    def __init__(self, d: int, n_heads: int, cond_dim: int,
                 ls_init: float = 1e-5):
        super().__init__()
        from tanitad.models.encoder import RMSNorm
        if d % n_heads:
            raise ValueError(f"d_model {d} not divisible by n_heads {n_heads}")
        self.n_heads, self.head_dim = n_heads, d // n_heads
        self.norm1 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.ls1 = nn.Parameter(ls_init * torch.ones(d))
        self.film = FiLM(cond_dim, d)
        self.norm2 = RMSNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))
        self.ls2 = nn.Parameter(ls_init * torch.ones(d))

    def forward(self, x: Tensor, cond: Tensor, mask: Tensor) -> Tensor:
        B, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        # mask convention matches nn.MultiheadAttention's float attn_mask
        am = mask if mask is None else mask.to(q.dtype)
        o = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=am)
        x = x + self.ls1 * self.proj(o.transpose(1, 2).reshape(B, N, D))
        return x + self.ls2 * self.mlp(self.film(self.norm2(x), cond))


class OperativePredictor(nn.Module):
    """Predicts future compact states from a causal window of (state, action).

    forward(states [B, W, D], actions [B, W, A], intent=None)
        -> {k: z_hat_{t+k} [B, D]}
    where t is the last window position. Residual: z_hat = z_t + delta_k.

    ``intent_dim`` (optional, D-030): when set, an ``intent`` token [B, intent_dim]
    from the tactical policy is projected and ADDED to the per-step action FiLM
    conditioning, so the tactical brain steers the operative dynamics (closing
    the hierarchy). ``intent=None`` reproduces the base behaviour exactly; base
    models build the predictor with ``intent_dim=None`` (no extra params), so a
    vanilla WorldModel checkpoint stays a strict subset of a 4b one.

    ``gated_intent`` (v2 lever 7): add a ReZero-style learnable scalar gate on the
    intent term (H26 — the ungated intent_proj norm ~31.4 COMPETED WITH act_emb
    ~28.3, diluting the action conditioning; engaging intent measured net-harmful
    to the operative). Init 0.1 so it starts action-dominant and grows only if
    training earns it. Default False => the term is ungated (implicit gate 1.0):
    no extra param, byte-identical state_dict to the pre-lever model.
    """

    def __init__(self, cfg: PredictorConfig, state_dim: int,
                 intent_dim: int | None = None, gated_intent: bool = False):
        super().__init__()
        self.cfg = cfg
        self.state_dim = state_dim      # plain int; not a buffer => state_dict unchanged
        d = cfg.d_model
        self.in_proj = nn.Linear(state_dim, d)
        self.act_emb = nn.Sequential(nn.Linear(cfg.action_dim, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.zeros(1, cfg.window, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            (ModernCausalBlock if cfg.modern else CausalBlock)(
                d, cfg.n_heads, cond_dim=d) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(d)
        # THE RESIDUAL DELTA HEADS ARE DOWN-SCALED, NOT ZEROED.
        #
        # `forward` computes `out[k] = z_t + delta`, with `delta` produced from
        # `self.norm(...)` -- a LayerNorm, so its output is O(1) PER DIM whatever
        # the latent's scale. v6's operative latent has mean|z| = 0.015581 and
        # moves 0.000892 per tick (MEASURED 2026-08-22, stride-1 latents at the
        # true dt=0.1s tick). A default-init head therefore starts emitting a
        # delta ~1000x LARGER than the movement it must predict, and the run
        # spends itself shrinking it. v6F-SW-30k's own o5_step1 was still
        # 535x WORSE than predicting NO CHANGE at step 20,000, and a scalar
        # rescale of the trained heads could not rescue it -- the error fell
        # monotonically to alpha=0, i.e. the learned delta carried no signal.
        #
        # WHY DOWN-SCALED AND NOT ZERO-INIT. Zeroing the OUTPUT head sets
        # dL/dh_last = W^T . dL/dout = 0, which stops gradient reaching the
        # ENTIRE predictor body -- blocks, in_proj, act_emb and intent_proj all
        # stall until the head leaves zero. `test_v6_staged.py::
        # test_planner_surface_is_total` caught exactly that: intent_proj went
        # invisible to the gradient probe. This is why FiLM (an INTERNAL
        # modulation, main path untouched) is zero-init a few lines above while
        # an output head must not be.
        #
        # SCOPE: initialisation only. state_dict shapes are unchanged, so every
        # existing checkpoint loads byte-identically.
        self.heads = nn.ModuleDict(
            {str(k): nn.Linear(d, state_dim) for k in cfg.horizons})
        if cfg.residual:
            for h in self.heads.values():
                h.weight.data.mul_(RESIDUAL_HEAD_INIT_SCALE)
                h.bias.data.mul_(RESIDUAL_HEAD_INIT_SCALE)
        self.out_proj = nn.Linear(state_dim, d)  # reserved: feed predictions back
        # Tactical-intent conditioning (D-030). Projected into the FiLM cond
        # space and added to the action embedding. Non-zero init so a live FiLM
        # makes the intent steer the output; FiLM's own zero-init keeps the
        # identity start (intent has no effect until the FiLM weights train).
        self.intent_proj = (nn.Linear(intent_dim, d)
                            if intent_dim is not None else None)
        # ReZero gate on the intent term (H26 fix): scalar, init 0.1, so the
        # operative starts action-dominant and the intent grows only if earned.
        # None (default-off) leaves the term ungated => byte-identical state_dict.
        self.intent_gate = (nn.Parameter(torch.tensor(0.1))
                            if gated_intent and intent_dim is not None else None)

    def forward(self, states: Tensor, actions: Tensor,
                intent: Tensor | None = None) -> dict[int, Tensor]:
        # `-O`-proof, named-axis contract check. Supersedes the bare `assert`,
        # which was stripped under `python -O` and let a short window run
        # SILENTLY (pos slice, causal mask and FiLM cond all re-align).
        validate_operative_inputs(states, actions, self.cfg.window,
                                  self.state_dim, self.cfg.action_dim)
        b, w, _ = states.shape
        x = self.in_proj(states) + self.pos[:, :w]
        cond = self.act_emb(actions)                        # [B, W, D]
        if intent is not None:
            if self.intent_proj is None:
                raise ValueError("predictor built without intent_dim cannot "
                                 "consume an intent token")
            term = self.intent_proj(intent).unsqueeze(1)          # broadcast W
            if self.intent_gate is not None:
                term = self.intent_gate * term
            cond = cond + term
        mask = torch.triu(torch.ones(w, w, device=states.device, dtype=torch.bool),
                          diagonal=1)
        for blk in self.blocks:
            x = blk(x, cond, mask)
        h_last = self.norm(x[:, -1])                        # [B, D]
        z_t = states[:, -1]
        out: dict[int, Tensor] = {}
        for k in self.cfg.horizons:
            delta = self.heads[str(k)](h_last)
            out[k] = z_t + delta if self.cfg.residual else delta
        return out


def change_weighted_mse(pred: Tensor, target: Tensor, prev: Tensor,
                        eps: float = 1e-6) -> Tensor:
    """MSE weighted by how much each latent dim actually changed (A4).

    Prevents static content from dominating the loss — the driving analog of
    ignoring parked background in favor of the consequence of the action.
    """
    w = (target - prev).abs()
    w = w / w.mean(dim=-1, keepdim=True).clamp_min(eps)
    return (w * (pred - target).pow(2)).mean()
