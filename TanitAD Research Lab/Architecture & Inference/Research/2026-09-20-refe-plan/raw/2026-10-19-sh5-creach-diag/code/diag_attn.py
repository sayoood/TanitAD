"""SH5 C-REACH diagnosis, grid 3 (Research Lab, 2026-10-19): ATTENTION-SHARPNESS levers and the attention-logit diagnostic.

Finding (f) (`raw/probe2_ideal_reader.json`): an IDEAL softmax(beta x kernel) read of the plant reaches the bar only with SHARP attention (beta >= 16 on the normalised Fourier kernel,
a logit range of ~2 beta ~ 30 over the plant entries); a flat read (beta 4) caps at ALLQ 0.73-0.77.  This module
  (1) implements three one-lever changes to the SEALED `sh5_branch.CrossLayer` WITHOUT editing it (a subclass swapped in after construction; the sealed file is imported read-only):
      * "temp":   a learnable per-head temperature tau = exp(mult * s) multiplying the logits (s init log(tau0) / mult; `mult` is a fixed reparametrisation so Adam can move log tau by
                  ~mult x lr per step -- at lr 3e-4 a plain log-temperature could move by <= 0.37 in 1,236 steps, i.e. tau <= 1.45: no lever at all);
      * "cos":    QK-norm (cosine) attention, logits = tau * cos(q_h, k_h), learnable tau (same parametrisation), init tau0 = 16 = the ideal reader's beta;
      * "qkinit": the q and k projection WEIGHTS of every layer scaled by `scale` at init (the logit scale x scale^2); the forward is the sealed one.
      tau is clamped at TAU_MAX = 100.  With tau = 1 the "temp" layer is the sealed layer BIT FOR BIT (tested in `check_levers_cpu.py`).
  (2) measures what the trained attention ACTUALLY does on the plant (`attn_token_stats`): per layer and head, the plant attention mass, the mass on the query's OWN entry, the logit range
      over the plant entries and over all slots, top-1 position accuracy, and beta_eff = the least-squares slope of the logits on the ideal reader's kernel (per-query centred) -- i.e. the
      ideal reader's beta that the learned attention implements, on the same kernel definition as probe 2 (octaves = min(query octaves, plant octaves))."""
from __future__ import annotations

import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "C:/Users/Admin/par18/sh5/code")
import sh5_branch as B                 # noqa: E402

TAU_MAX = 100.0


class LeverLayer(B.CrossLayer):
    """The sealed CrossLayer with a learnable temperature ("temp") or cosine attention ("cos").  Swapped in by `apply_lever` (class swap + one new parameter `s` [H])."""

    def tau(self) -> torch.Tensor:
        return torch.exp(torch.clamp(self.s.float() * self.mult, max=math.log(TAU_MAX)))

    def kv(self, ctx_n: torch.Tensor):
        K, V = super().kv(ctx_n)
        if self.mode == "cos":
            K = F.normalize(K.float(), dim=-1).to(K.dtype)
        return K, V

    def qk(self, h: torch.Tensor, K: torch.Tensor):
        """-> (q, K) such that the sealed SDPA call F.scaled_dot_product_attention(q, K, V) computes this layer's attention (default scale 1/sqrt(dh))."""
        S, Q, _ = h.shape
        H, dh = self.c.heads, self.c.d // self.c.heads
        q = self.q(self.ln_q(h)).view(S, Q, H, dh).transpose(1, 2)
        tau = self.tau().view(1, H, 1, 1)
        if self.mode == "temp":
            q = (q.float() * tau).to(q.dtype)
        elif self.mode == "cos":
            q = (F.normalize(q.float(), dim=-1) * tau * math.sqrt(dh)).to(q.dtype)
        else:
            raise ValueError(self.mode)
        return q, K.to(q.dtype)

    def forward(self, h: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        S, Q, _ = h.shape
        H, dh = self.c.heads, self.c.d // self.c.heads
        q, Kq = self.qk(h, K)
        a = F.scaled_dot_product_attention(q, Kq, V.to(q.dtype))
        h = h + self.o(a.transpose(1, 2).reshape(S, Q, H * dh))
        return h + self.ffn2(F.gelu(self.ffn1(self.ln_f(h))))


def sealed_qk(lay, h, K):
    S, Q, _ = h.shape
    H, dh = lay.c.heads, lay.c.d // lay.c.heads
    q = lay.q(lay.ln_q(h)).view(S, Q, H, dh).transpose(1, 2)
    return q, K.to(q.dtype)


def layer_qk(lay, h, K):
    return lay.qk(h, K) if isinstance(lay, LeverLayer) else sealed_qk(lay, h, K)


def apply_lever(model: B.SH5Branch, lever) -> B.SH5Branch:
    """In place.  lever = None | {"mode": "temp"|"cos", "tau0": float, "mult": float} | {"mode": "qkinit", "scale": float}.  Rebuild the optimizer afterwards."""
    if not lever:
        return model
    mode = lever["mode"]
    for lay in model.layers:
        if mode == "qkinit":
            with torch.no_grad():
                lay.q.weight.mul_(float(lever["scale"]))
                lay.k.weight.mul_(float(lever["scale"]))
        elif mode in ("temp", "cos"):
            dev = lay.q.weight.device
            lay.__class__ = LeverLayer
            lay.mode, lay.mult = mode, float(lever["mult"])
            lay.s = nn.Parameter(torch.full((lay.c.heads,), math.log(float(lever["tau0"])) / lay.mult, device=dev))
        else:
            raise ValueError(f"unknown lever {lever}")
    return model


def taus(model) -> list:
    return [[round(float(x), 4) for x in lay.tau().detach().cpu()] if isinstance(lay, LeverLayer) else None for lay in model.layers]


# ---- the diagnostic ------------------------------------------------------------------------------------------------------------------------------------------------------------------
def kernel(xq: torch.Tensor, xe: torch.Tensor, n_oct: int) -> torch.Tensor:
    """probe 2's normalised Fourier kernel: mean over octaves j < n_oct of [cos(w_j dx) + cos(w_j dy)] / 2, w_j = 2^j pi / 64.  xq [Q,2], xe [E,2] -> [Q,E] in [-1, 1]."""
    dx = xq[:, None, 0] - xe[None, :, 0]
    dy = xq[:, None, 1] - xe[None, :, 1]
    k = torch.zeros_like(dx)
    for j in range(n_oct):
        w = (2.0 ** j) * math.pi / 64.0
        k += torch.cos(w * dx) + torch.cos(w * dy)
    return k / (2 * n_oct)


def beta_fit(lp: torch.Tensor, ker: torch.Tensor):
    """lp [..., Q, E] logits over the plant entries, ker [Q, E] (broadcast).  Per-query centring, pooled least squares -> (sum lc*kc, sum kc^2, sum lc^2) over Q, E (per leading index)."""
    lc = lp - lp.mean(-1, keepdim=True)
    kc = ker - ker.mean(-1, keepdim=True)
    kc = kc.expand_as(lc)
    return (lc * kc).sum((-2, -1)), (kc * kc).sum((-2, -1)), (lc * lc).sum((-2, -1))


@torch.no_grad()
def attn_token_stats(model: B.SH5Branch, pts: torch.Tensor, ctx: torch.Tensor, n_oct: int, ac) -> list:
    """ONE token: pts [S,64,9,3] (all proposals, eval order), ctx [1,P,Dc] WITH the plant written in its last S*576 slots (entry s*576 + p*9 + t = query (s, p, t)).
    -> per layer a dict of per-head SUMS (accumulate over tokens with `acc_add`)."""
    S, Pp, n, _ = pts.shape
    Qn = Pp * n
    P = ctx.shape[1]
    npl = S * Qn
    s2t = torch.zeros(S, dtype=torch.long, device=pts.device)
    xy = pts[..., :2].reshape(S * Qn, 2).float()
    with ac():
        kv = model.kv(ctx)
        h = model.q_out(F.gelu(model.q_in(B.point_features(pts, model.c.fourier_k)))) + model.t_emb
        h = h.reshape(S, Qn, model.c.d)
        out = []
        for lay, (K, V) in zip(model.layers, kv):
            Kg, Vg = K[s2t], V[s2t]
            q, Kq = layer_qk(lay, h, Kg)
            dh = q.shape[-1]
            lg = (q.float() @ Kq.float().transpose(-1, -2)) / math.sqrt(dh)                     # [S,H,Q,P] the exact pre-softmax logits of the SDPA call
            a = torch.softmax(lg, -1)
            lp = lg[..., P - npl:]                                                                 # [S,H,Q,npl]
            r = {"n_q": float(S * Qn)}
            r["plant_mass"] = a[..., P - npl:].sum(-1).sum((0, 2))                                 # [H] summed over queries
            own = torch.arange(Qn, device=pts.device)
            sm = torch.stack([a[s, :, own, P - npl + s * Qn + own] for s in range(S)])              # [S,H,Q]
            r["self_mass"] = sm.sum((0, 2))
            r["range_plant"] = (lp.max(-1).values - lp.min(-1).values).sum((0, 2))
            r["range_all"] = (lg.max(-1).values - lg.min(-1).values).sum((0, 2))
            num = den = ss = 0.0
            top1 = 0.0
            for s in range(S):
                ker = kernel(xy[s * Qn:(s + 1) * Qn], xy, n_oct)                                   # [Q, npl]
                a_, b_, c_ = beta_fit(lp[s], ker)                                                  # [H]
                num, den, ss = num + a_, den + b_, ss + c_
                am = lp[s].argmax(-1)                                                              # [H,Q] -> entry index
                d = (xy[am] - xy[s * Qn:(s + 1) * Qn][None]).norm(dim=-1)
                top1 = top1 + (d < 0.5).float().sum(-1)
            r["beta_num"], r["beta_den"], r["beta_ss"], r["top1_0p5m"] = num, den, ss, top1
            out.append({k: (v.detach().double().cpu() if torch.is_tensor(v) else v) for k, v in r.items()})
            h = lay(h, Kg, Vg)
    return out


def acc_add(acc, tok_stats):
    if acc is None:
        return [dict(d) for d in tok_stats]
    for A, d in zip(acc, tok_stats):
        for k, v in d.items():
            A[k] = A[k] + v
    return acc


def acc_report(acc, n_tok: int) -> list:
    rep = []
    for li, A in enumerate(acc):
        nq = A["n_q"]
        beta = (A["beta_num"] / A["beta_den"]).tolist()
        corr = (A["beta_num"] / torch.sqrt(A["beta_den"] * A["beta_ss"])).tolist()
        rep.append({"layer": li, "n_tokens": n_tok, "beta_eff": [round(x, 3) for x in beta], "beta_eff_max": round(max(beta), 3), "logit_kernel_corr": [round(x, 4) for x in corr],
                    "plant_mass": [round(float(x) / nq, 4) for x in A["plant_mass"]], "self_mass": [round(float(x) / nq, 5) for x in A["self_mass"]],
                    "range_plant_mean": [round(float(x) / nq, 3) for x in A["range_plant"]], "range_all_mean": [round(float(x) / nq, 3) for x in A["range_all"]],
                    "top1_within_0p5m": [round(float(x) / nq, 4) for x in A["top1_0p5m"]]})
    return rep
