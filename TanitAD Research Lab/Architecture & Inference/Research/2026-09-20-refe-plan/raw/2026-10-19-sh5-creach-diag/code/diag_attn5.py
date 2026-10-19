"""SH5 C-REACH diagnosis, grid 5 (Research Lab, 2026-10-20): `diag_attn` (UNCHANGED, sha256-pinned by PREREG_GRID3) + ONE new lever option, fp32 attention.

  lever = {"mode": "cos", "tau0": 16.0, "mult": 10.0, "fp32": true}: the grid-3 cosine layer, but the attention itself (q, K, V, the logits and the softmax) runs in FP32 with autocast
  DISABLED inside the SDPA call.  The q / k / v / o projections still run under the trainer's bf16 autocast, exactly as before; only the normalised q and K and the attention product are
  kept in fp32.  With "fp32" absent or false, `apply_lever` is `diag_attn.apply_lever` bit for bit (it delegates).
Everything else (the diagnostic, `taus`, `acc_add`, `acc_report`, `kernel`) is re-exported from `diag_attn` unchanged, so `diag_pre3` runs with this module swapped in for `DA`."""
from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "C:/Users/Admin/par18/sh5diag/code")
import diag_attn as DA                 # noqa: E402
from diag_attn import B, TAU_MAX, LeverLayer, kernel, beta_fit, attn_token_stats, acc_add, acc_report, taus, sealed_qk, layer_qk   # noqa: E402,F401


class Fp32LeverLayer(LeverLayer):
    """LeverLayer whose attention runs in fp32 (autocast off inside the SDPA call)."""

    def kv(self, ctx_n: torch.Tensor):
        K, V = B.CrossLayer.kv(self, ctx_n)
        if self.mode == "cos":
            K = F.normalize(K.float(), dim=-1)
        return K.float(), V.float()

    def qk(self, h: torch.Tensor, K: torch.Tensor):
        S, Q, _ = h.shape
        H, dh = self.c.heads, self.c.d // self.c.heads
        q = self.q(self.ln_q(h)).view(S, Q, H, dh).transpose(1, 2).float()
        tau = self.tau().view(1, H, 1, 1).float()
        if self.mode == "temp":
            q = q * tau
        elif self.mode == "cos":
            q = F.normalize(q, dim=-1) * tau * math.sqrt(dh)
        else:
            raise ValueError(self.mode)
        return q, K.float()

    def forward(self, h: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        S, Q, _ = h.shape
        H, dh = self.c.heads, self.c.d // self.c.heads
        q, Kq = self.qk(h, K)
        dev = "cuda" if q.is_cuda else "cpu"
        with torch.autocast(device_type=dev, enabled=False):
            a = F.scaled_dot_product_attention(q.float(), Kq.float(), V.float())
        h = h + self.o(a.transpose(1, 2).reshape(S, Q, H * dh))
        return h + self.ffn2(F.gelu(self.ffn1(self.ln_f(h))))


def apply_lever(model, lever):
    if not lever or not lever.get("fp32"):
        return DA.apply_lever(model, lever)
    if lever["mode"] not in ("temp", "cos"):
        raise ValueError(f"fp32 needs a temp/cos lever: {lever}")
    DA.apply_lever(model, {k: v for k, v in lever.items() if k != "fp32"})
    for lay in model.layers:
        lay.__class__ = Fp32LeverLayer
    return model
