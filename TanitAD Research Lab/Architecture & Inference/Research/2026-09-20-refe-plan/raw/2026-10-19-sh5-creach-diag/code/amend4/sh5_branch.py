"""SH5: the scorer-branch DAC-proximity read.  A small PROPOSAL-ANCHORED cross-attention branch over a token's cached `visual_ctx` [P=7680, 256] (frozen; P2's trunk and rows are never touched).

Per proposal the 9 key-time footprint points (x, y, yaw) are embedded by an MLP (+ a learned time embedding), 2 layers of multi-head cross-attention (d 256, 4 heads, pre-LN, + FFN)
read the token's ctx, the 9 point states are POOLED by a learned time-indexed linear projection (concat over the 9 points -> LN -> Linear), and two heads sit on the pooled vector:
  (i) a regression of the 9 key-time |clearance| (metres, Huber; SIGN-CLEAN: |dac_clear_min| is excluded), (ii) a DAC logit (BCE on the DAC label).
Both heads are ZERO-INITIALISED, so an untrained branch emits z_S == 0.0 exactly and L+S == L (C-ID).  Inference is VISION-ONLY: the clearance is a TRAINING TARGET only.

Reachability trace of every lever (CLAUDE.md "built, tested and unreachable"):  CLI flag -> Learner spec -> `loss_terms(w_reg=...)` -> `total` -> `.backward()`; tested with exact-zero controls.
Pure torch; the same code runs on CPU (fp32) in the tests and on CUDA under bf16 autocast.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sh5_lib as L


@dataclass(frozen=True)
class SH5Cfg:
    d: int = L.D_MODEL
    heads: int = L.HEADS
    layers: int = L.LAYERS
    ffn: int = L.FFN
    d_ctx: int = 256
    n_pts: int = L.N_PTS
    n_reg: int = L.N_REG
    fourier_k: int = L.FOURIER_K

    def __post_init__(self):
        if self.d % self.heads:
            raise ValueError("d must be divisible by heads")


def point_features(pts: torch.Tensor, k: int) -> torch.Tensor:
    """pts [..., 3] (x, y, yaw) -> [..., 4 + 4k]: scaled x, y; cos, sin of the yaw; sin / cos of x and y at k octaves (wavelengths 128 m ... 128/2^(k-1) m)."""
    x, y, h = pts[..., 0], pts[..., 1], pts[..., 2]
    f = [x / 30.0, y / 10.0, torch.cos(h), torch.sin(h)]
    for j in range(k):
        w = (2.0 ** j) * math.pi / 64.0
        f += [torch.sin(w * x), torch.cos(w * x), torch.sin(w * y), torch.cos(w * y)]
    return torch.stack(f, dim=-1)


class CrossLayer(nn.Module):
    def __init__(self, c: SH5Cfg):
        super().__init__()
        self.c = c
        self.ln_q = nn.LayerNorm(c.d)
        self.q = nn.Linear(c.d, c.d)
        self.k = nn.Linear(c.d_ctx, c.d)
        self.v = nn.Linear(c.d_ctx, c.d)
        self.o = nn.Linear(c.d, c.d)
        self.ln_f = nn.LayerNorm(c.d)
        self.ffn1 = nn.Linear(c.d, c.ffn)
        self.ffn2 = nn.Linear(c.ffn, c.d)

    def kv(self, ctx_n: torch.Tensor):
        """ctx_n [T, P, d_ctx] (already layer-normed) -> K, V [T, H, P, dh]"""
        T, P, _ = ctx_n.shape
        H, dh = self.c.heads, self.c.d // self.c.heads
        return self.k(ctx_n).view(T, P, H, dh).transpose(1, 2), self.v(ctx_n).view(T, P, H, dh).transpose(1, 2)

    def forward(self, h: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """h [S, Q, d]; K, V [S, H, P, dh] (already gathered per set)"""
        S, Q, _ = h.shape
        H, dh = self.c.heads, self.c.d // self.c.heads
        q = self.q(self.ln_q(h)).view(S, Q, H, dh).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, K.to(q.dtype), V.to(q.dtype))
        h = h + self.o(a.transpose(1, 2).reshape(S, Q, H * dh))
        return h + self.ffn2(F.gelu(self.ffn1(self.ln_f(h))))


class SH5Branch(nn.Module):
    def __init__(self, c: SH5Cfg = SH5Cfg()):
        super().__init__()
        self.c = c
        self.ctx_ln = nn.LayerNorm(c.d_ctx)
        self.q_in = nn.Linear(4 + 4 * c.fourier_k, c.d)
        self.q_out = nn.Linear(c.d, c.d)
        self.t_emb = nn.Parameter(torch.randn(c.n_pts, c.d) * 0.02)
        self.layers = nn.ModuleList([CrossLayer(c) for _ in range(c.layers)])
        self.pool_ln = nn.LayerNorm(c.n_pts * c.d)
        self.pool = nn.Linear(c.n_pts * c.d, c.d)
        self.reg_head = nn.Linear(c.d, c.n_reg)
        self.dac_head = nn.Linear(c.d, 1)
        for h in (self.reg_head, self.dac_head):                  # ZERO-INIT: z_S == 0.0 exactly before any step (C-ID)
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    # ---- token-level work, done once per token per step -----------------------------------------------------------------------------------------------------------------------
    def kv(self, ctx: torch.Tensor):
        """ctx [T, P, d_ctx] (bf16 / fp32 as cached) -> list over layers of (K, V) [T, H, P, dh].  LayerNorm in fp32."""
        n = self.ctx_ln(ctx.float())
        return [lay.kv(n.to(ctx.dtype) if ctx.dtype != torch.float32 else n) for lay in self.layers]

    # ---- set-level work -------------------------------------------------------------------------------------------------------------------------------------------------------
    def forward(self, pts: torch.Tensor, kv: list, set_to_tok: torch.Tensor):
        """pts [S, Pp, 9, 3]; kv from `kv()`; set_to_tok [S] long (row of the token in the kv tensors) -> (dac_logit [S, Pp], reg [S, Pp, 9])"""
        S, Pp, n, _ = pts.shape
        h = self.q_out(F.gelu(self.q_in(point_features(pts, self.c.fourier_k)))) + self.t_emb          # [S, Pp, 9, d]
        h = h.reshape(S, Pp * n, self.c.d)
        for lay, (K, V) in zip(self.layers, kv):
            h = lay(h, K[set_to_tok], V[set_to_tok])
        h = h.reshape(S, Pp, n * self.c.d)
        p = F.gelu(self.pool(self.pool_ln(h)))
        return self.dac_head(p).squeeze(-1), self.reg_head(p)


# ---- losses (prereg s4) ------------------------------------------------------------------------------------------------------------------------------------------------------------------
def loss_terms(logit: torch.Tensor, reg: torch.Tensor, y: torch.Tensor, tgt: torch.Tensor, w_reg: float) -> dict:
    """BCE on the DAC label + w_reg * Huber(delta 1 m) on the 9 key-time |clearances|.  w_reg == 0 removes the term from the graph (S-noprox); the reg head then receives EXACTLY zero gradient."""
    bce = F.binary_cross_entropy_with_logits(logit.float(), y.float())
    out = {"bce": bce}
    if w_reg > 0.0:
        out["reg"] = F.huber_loss(reg.float(), tgt.float(), delta=L.HUBER_DELTA)
        out["total"] = bce + w_reg * out["reg"]
    else:
        out["total"] = bce
    return out


# ---- G-LIVE (the launch-gate-style check of prereg s8) -------------------------------------------------------------------------------------------------------------------------------
def param_groups(m: SH5Branch) -> dict:
    g = {"ctx_ln": list(m.ctx_ln.parameters()), "q_mlp": list(m.q_in.parameters()) + list(m.q_out.parameters()), "t_emb": [m.t_emb],
         "pool": list(m.pool_ln.parameters()) + list(m.pool.parameters()), "reg_head": list(m.reg_head.parameters()), "dac_head": list(m.dac_head.parameters())}
    for i, lay in enumerate(m.layers):
        g[f"layer{i}.ln_q"] = list(lay.ln_q.parameters())
        g[f"layer{i}.qkvo"] = [p for mod in (lay.q, lay.k, lay.v, lay.o) for p in mod.parameters()]
        g[f"layer{i}.ffn"] = list(lay.ln_f.parameters()) + list(lay.ffn1.parameters()) + list(lay.ffn2.parameters())
    return g


def grad_abs_sums(m: SH5Branch) -> dict:
    return {k: float(sum(p.grad.abs().sum().item() for p in ps if p.grad is not None)) for k, ps in param_groups(m).items()}


def glive(m: SH5Branch, make_batch, w_reg: float, steps: int = 3, lr: float = L.LR, ac=None) -> dict:
    """Run `steps` Adam steps on one fixed batch (the heads are zero at init, so the layers below receive an EXACT zero gradient on step 1 by construction: legitimate, and why the check is
    read after the heads have moved).  Returns {"finite": bool, "terms": {...}, "grad_abs_sum": {group: float}, "dead_groups": [...], "zero_control": float, "ok": bool}.
    The ZERO CONTROL: ctx_ln is frozen (requires_grad False) in a deep copy and must read EXACTLY 0.0 while the live copy's reads > 0 - this is what proves the probe can tell frozen from live.
    `ac` = the autocast context factory the TRAINER uses (Amendment 1): the cached ctx is bf16, and the first Linear on it only accepts bf16 input under autocast, so a probe that runs outside the trainer's autocast
    crashes on the real ctx (it passed every CPU test because the tests carried an fp32 ctx).  Default None = no autocast (fp32 tests)."""
    import contextlib
    import copy
    ac = ac if ac is not None else contextlib.nullcontext
    results = {}
    for tag, frozen in (("live", False), ("frozen_ctx_ln", True)):
        mm = copy.deepcopy(m)
        mm.train()
        if frozen:
            for p in mm.ctx_ln.parameters():
                p.requires_grad_(False)
        opt = torch.optim.Adam([p for p in mm.parameters() if p.requires_grad], lr=lr)
        finite = True
        terms = {}
        for s in range(steps):
            pts, set_to_tok, ctx, y, tgt = make_batch()
            with ac():
                kv = mm.kv(ctx)
                logit, reg = mm(pts, kv, set_to_tok)
            lt = loss_terms(logit, reg, y, tgt, w_reg)
            finite = finite and all(bool(torch.isfinite(v)) for v in lt.values())
            terms = {k: float(v) for k, v in lt.items()}
            opt.zero_grad(set_to_none=True)
            lt["total"].backward()
            if s < steps - 1:
                opt.step()
        results[tag] = {"finite": finite, "terms": terms, "grad": grad_abs_sums(mm)}
    live, fro = results["live"]["grad"], results["frozen_ctx_ln"]["grad"]
    dead = [k for k, v in live.items() if not v > 0.0 and not (k == "reg_head" and w_reg == 0.0)]
    exp_zero = ["reg_head"] if w_reg == 0.0 else []
    zero_control = fro["ctx_ln"]
    ok = (results["live"]["finite"] and results["frozen_ctx_ln"]["finite"] and not dead and zero_control == 0.0 and live["ctx_ln"] > 0.0
          and all(live[k] == 0.0 for k in exp_zero))
    return {"finite": results["live"]["finite"], "terms": results["live"]["terms"], "grad_abs_sum": live, "dead_groups": dead, "zero_control_frozen_ctx_ln": zero_control, "expected_exact_zero": exp_zero, "ok": bool(ok)}


# ---- planted controls (prereg s8): built from PRIVILEGED signed clearances, used ONLY by C-REACH / C-POWER; never an input of the real arms -----------------------------------------
def plant_block(pts: np.ndarray, signed_clear: np.ndarray, d_ctx: int = 256, n_slots: int = 7680) -> np.ndarray:
    """pts [S, 64, 9, 3] (one token's sets), signed_clear [S, 64, 9] (metres) -> the planted rows [n_entries, d_ctx] float32 to overwrite the LAST n_entries ctx slots of that token:
    channels 0..31 = a fixed Fourier code of (x, y) (8 octaves x sin/cos x 2 coords), channel 32 = signed clearance / 10, channel 33 = 1 (planted flag), the rest 0.
    One entry per (set, proposal, key time); identical (x, y) -> identical entry (the t = 0 point is shared by every proposal).  n_entries <= n_slots is asserted."""
    S, Pp, n, _ = pts.shape
    xy = pts[..., :2].reshape(-1, 2).astype(np.float64)
    sc = signed_clear.reshape(-1).astype(np.float64)
    if len(xy) > n_slots:
        raise ValueError(f"{len(xy)} plant entries exceed the {n_slots} ctx slots")
    code = []
    for j in range(8):
        w = (2.0 ** j) * math.pi / 64.0
        code += [np.sin(w * xy[:, 0]), np.cos(w * xy[:, 0]), np.sin(w * xy[:, 1]), np.cos(w * xy[:, 1])]
    blk = np.zeros((len(xy), d_ctx), np.float64)
    blk[:, :32] = np.stack(code, axis=1)
    blk[:, 32] = np.clip(sc, -L.CLEAR_CAP, L.CLEAR_CAP) / 10.0
    blk[:, 33] = 1.0
    return blk.astype(np.float32)
