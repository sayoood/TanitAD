"""HiCAP trainable part (paper sec:backbone, sec:train): motion adapter, fusion transformer, level readouts,
node towers, cost/refinement heads, optional gated audio read, and the candidate re-ranker. ~27 M parameters at defaults.

Frozen backbone tokens arrive CACHED: vlm [B, 3(depths), Nv, d_v] (+ last-token state [B, d_v]) and their age in seconds [B].
Inputs at inference: frames (via the adapter), ego (v, a_lon, yaw_rate) -- curvature is NOT an input (PI ruling 2026-10-25),
nav token + args and route checkpoint (stamped oracle inputs, each with a learned null row), optional audio tokens.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F

LEVELS = ("s1", "s2lat", "s2lon", "s3P", "s3V", "goal")          # level queries, in this order


@dataclass
class HiCAPConfig:
    d: int = 512
    layers: int = 6
    heads: int = 8
    d_v: int = 2048                    # frozen backbone hidden size (Edge reasoner 2048; 8B tier 4096)
    n_depths: int = 3
    n_vlm_tokens: int = 128
    n_motion_tokens: int = 16
    motion_hw: tuple = (64, 160)
    motion_frames: int = 3
    d_text: int = 2048                 # cached text-tower embedding of a token definition
    n_args: int = 4                    # numeric arguments per node (v_target, within_m, ...)
    n_strategic: int = 7               # declared; 5 usable
    n_lat: int = 3
    n_lon: int = 7
    n_nav_args: int = 4                # d_next, d_end, dyaw_next, valid
    n_goals: int = 22
    path_feat: int = 32
    prof_feat: int = 20
    d_audio: int = 768
    n_cost: int = 3                    # ttc, clearance, progress deficit
    n_slots: int = 8
    matryoshka: tuple = (128, 256, 512)
    drop: float = 0.0
    rerank_layers: int = 3
    rerank_experts: int = 3            # routed deterministically by lateral class
    logit_init: float = math.log(1 / 0.07)
    logit_max: float = math.log(100.0)


def fourier(x: torch.Tensor, n_freq: int = 8) -> torch.Tensor:
    """x [...,k] -> [..., k*2*n_freq] sin/cos features at octave frequencies."""
    f = (2.0 ** torch.arange(n_freq, device=x.device, dtype=x.dtype)) * math.pi
    a = x.unsqueeze(-1) * f
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


class MLP(nn.Module):
    def __init__(self, i, h, o, act=nn.GELU):
        super().__init__(); self.net = nn.Sequential(nn.Linear(i, h), act(), nn.Linear(h, o))
    def forward(self, x): return self.net(x)


class MotionAdapter(nn.Module):
    """3-frame stack (+ inter-frame differences) at 64x160 -> 16 motion tokens. ~2 M parameters. The only trained vision part."""
    def __init__(self, c: HiCAPConfig):
        super().__init__()
        cin = 3 * c.motion_frames + 3 * (c.motion_frames - 1)      # RGB frames + differences
        ch = (48, 96, 160, 224)
        L = []; i = cin
        for o in ch:
            L += [nn.Conv2d(i, o, 3, stride=2, padding=1), nn.GroupNorm(8, o), nn.GELU()]; i = o
        self.conv = nn.Sequential(*L)                               # 64x160 -> 4x10
        self.pool = nn.AdaptiveAvgPool2d((4, 4))                    # -> 16 tokens
        self.proj = nn.Linear(ch[-1], c.d)
        self.pos = nn.Parameter(torch.zeros(1, c.n_motion_tokens, c.d))
        self.n = c.n_motion_tokens

    def forward(self, frames: torch.Tensor) -> torch.Tensor:        # [B, F, 3, H, W]
        B, Fn = frames.shape[:2]
        diffs = frames[:, 1:] - frames[:, :-1]
        x = torch.cat([frames, diffs], 1).flatten(1, 2)
        x = self.pool(self.conv(x)).flatten(2).transpose(1, 2)     # [B,16,C]
        return self.proj(x) + self.pos


class VLMProjector(nn.Module):
    """Cached frozen tokens [B,D,N,d_v] -> [B,N,d]: per-depth LayerNorm+Linear, learned softmax mixture over depths,
    Fourier embedding of the token AGE (tokens are reused between refreshes), plus the last-token (CLM) state as one extra token."""
    def __init__(self, c: HiCAPConfig):
        super().__init__()
        self.ln = nn.ModuleList([nn.LayerNorm(c.d_v) for _ in range(c.n_depths)])
        self.lin = nn.ModuleList([nn.Linear(c.d_v, c.d) for _ in range(c.n_depths)])
        self.mix = nn.Parameter(torch.zeros(c.n_depths))
        self.pos = nn.Parameter(torch.zeros(1, c.n_vlm_tokens, c.d))
        self.age = MLP(2 * 8, c.d, c.d)
        self.last_ln = nn.LayerNorm(c.d_v); self.last = nn.Linear(c.d_v, c.d)

    def forward(self, vlm, last, age_s):
        h = torch.stack([self.lin[k](self.ln[k](vlm[:, k])) for k in range(vlm.shape[1])], 1)   # [B,D,N,d]
        w = torch.softmax(self.mix, 0).view(1, -1, 1, 1)
        x = (h * w).sum(1) + self.pos[:, : h.shape[2]]
        a = self.age(fourier(age_s.view(-1, 1) / 2.0, 8)).unsqueeze(1)                         # age in units of 2 s
        return torch.cat([self.last(self.last_ln(last)).unsqueeze(1), x + a], 1)               # [B,1+N,d]


class NullableToken(nn.Module):
    """MLP over a small numeric vector with a learned null row (dropout / absent oracle input)."""
    def __init__(self, k, d):
        super().__init__(); self.mlp = MLP(k * 16, d, d); self.null = nn.Parameter(torch.zeros(d)); self.k = k
    def forward(self, x, present: torch.Tensor | None = None):
        t = self.mlp(fourier(x, 8))
        if present is None: return t
        m = present.view(-1, 1).to(t.dtype)
        return m * t + (1 - m) * self.null


class Block(nn.Module):
    def __init__(self, d, heads, drop=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.att = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(d); self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x, pad=None):
        h = self.n1(x); x = x + self.att(h, h, h, key_padding_mask=pad, need_weights=False)[0]
        return x + self.ff(self.n2(x))


class GatedAudio(nn.Module):
    """q' = q + present * tanh(g) * CrossAttn(q, audio tokens), g = 0 at init, so audio-null is BIT-EXACTLY the original (paper sec:audio).
    audio=None skips the branch entirely; `present` [B] (mixed batches) zeroes the residual of samples without audio, because
    cross-attention over zero tokens is not a null."""
    def __init__(self, c: HiCAPConfig):
        super().__init__()
        self.proj = nn.Linear(c.d_audio, c.d); self.att = nn.MultiheadAttention(c.d, c.heads, batch_first=True)
        self.g = nn.Parameter(torch.zeros(len(LEVELS)))
    def forward(self, q, audio, present=None):                     # q [B,6,d], audio [B,k,d_audio], present [B] or None
        if audio is None: return q
        a = self.proj(audio)
        r = self.att(q, a, a, need_weights=False)[0]
        if present is not None: r = r * present.to(r.dtype).view(-1, 1, 1)
        return q + torch.tanh(self.g).view(1, -1, 1) * r


class ReadoutHead(nn.Module):
    """H_l: R^d -> R^1024 -> R^d, L2-normalised; Matryoshka-nested widths."""
    def __init__(self, d): super().__init__(); self.m = MLP(d, 2 * d, d)
    def forward(self, q, width: int | None = None):
        s = self.m(q)
        if width is not None: s = s[..., :width]
        return F.normalize(s, dim=-1)


class Fusion(nn.Module):
    def __init__(self, c: HiCAPConfig):
        super().__init__()
        self.c = c
        self.vlm = VLMProjector(c); self.motion = MotionAdapter(c)
        self.ego = NullableToken(3, c.d)                              # v, a_lon, yaw rate (curvature excluded)
        self.nav_tok = nn.Embedding(3, c.d); self.nav_args = NullableToken(c.n_nav_args, c.d)
        self.route = NullableToken(4, c.d)                            # x, y, psi, valid (v9 RC-A50, noised at train)
        self.queries = nn.Parameter(torch.randn(1, len(LEVELS), c.d) * 0.02)
        self.blocks = nn.ModuleList([Block(c.d, c.heads, c.drop) for _ in range(c.layers)])
        self.norm = nn.LayerNorm(c.d)
        self.audio = GatedAudio(c)

    def forward(self, vlm, last, age_s, frames, ego, nav, nav_args, route,
                ego_present=None, nav_present=None, route_present=None, audio=None, audio_present=None):
        B = vlm.shape[0]
        toks = [self.vlm(vlm, last, age_s), self.motion(frames),
                self.ego(ego, ego_present).unsqueeze(1),
                (self.nav_tok(nav) + self.nav_args(nav_args, nav_present)).unsqueeze(1),
                self.route(route, route_present).unsqueeze(1)]
        x = torch.cat(toks + [self.queries.expand(B, -1, -1)], 1)
        for b in self.blocks: x = b(x)
        q = self.norm(x[:, -len(LEVELS):])
        return self.audio(q, audio, audio_present), x[:, :-len(LEVELS)]   # level outputs [B,6,d], scene tokens


class NodeTower(nn.Module):
    """Level 1/2 node embeddings a_v = norm(G([t_v ; phi(args_v)])), t_v the cached text embedding of the definition."""
    def __init__(self, c: HiCAPConfig, d_out=None):
        super().__init__(); self.g = MLP(c.d_text + c.n_args * 16, 2 * c.d, d_out or c.d)
    def forward(self, text, args): return F.normalize(self.g(torch.cat([text, fourier(args, 8)], -1)), dim=-1)


class NumericTower(nn.Module):
    def __init__(self, k, c: HiCAPConfig, n_cls=0):
        super().__init__(); self.g = MLP(k, 2 * c.d, c.d); self.cls = nn.Embedding(max(n_cls, 1), c.d) if n_cls else None
    def forward(self, feat, cls=None):
        h = self.g(feat)
        if self.cls is not None and cls is not None: h = h + self.cls(cls)
        return F.normalize(h, dim=-1)


class Rerank(nn.Module):
    """S4: candidates (queries) attend to scene tokens in a light decoder; MoE FFN routed DETERMINISTICALLY by the lateral class;
    per-candidate sub-score heads (ttc, clearance, progress deficit, logit)."""
    def __init__(self, c: HiCAPConfig):
        super().__init__(); self.c = c
        self.cand = MLP(c.d * 3 + c.n_slots * 2, c.d, c.d)
        self.layers = nn.ModuleList()
        for _ in range(c.rerank_layers):
            self.layers.append(nn.ModuleDict(dict(
                n1=nn.LayerNorm(c.d), att=nn.MultiheadAttention(c.d, c.heads, batch_first=True), n2=nn.LayerNorm(c.d),
                moe=nn.ModuleList([nn.Sequential(nn.Linear(c.d, 2 * c.d), nn.GELU(), nn.Linear(2 * c.d, c.d)) for _ in range(c.rerank_experts)]))))
        self.out = nn.Linear(c.d, c.n_cost + 1)

    def forward(self, a_p, a_q, a_cell, traj_slots, scene, lat_cls):  # [B,K,d] x3, [B,K,8,2], [B,N,d], [B,K]
        B, K = a_p.shape[:2]
        h = self.cand(torch.cat([a_p, a_q, a_cell, traj_slots.flatten(2)], -1))
        for L in self.layers:
            h = h + L["att"](L["n1"](h), scene, scene, need_weights=False)[0]
            z = L["n2"](h)
            ex = torch.stack([m(z) for m in L["moe"]], -2)                          # [B,K,E,d]
            sel = lat_cls.clamp(max=self.c.rerank_experts - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.c.d)
            h = h + ex.gather(2, sel).squeeze(2)                                    # deterministic routing
        o = self.out(h)
        return o[..., :-1], o[..., -1]                                              # cost predictions [B,K,3], logit [B,K]


class HiCAP(nn.Module):
    def __init__(self, c: HiCAPConfig | None = None):
        super().__init__()
        c = c or HiCAPConfig(); self.c = c
        self.fusion = Fusion(c)
        self.heads = nn.ModuleDict({k: ReadoutHead(c.d) for k in LEVELS if k != "goal"})
        self.goal_head = MLP(c.d, c.d, c.n_goals + 3 * 3)               # 22 goal logits + goal point (x,y,t) at 2/4/6 s
        self.tower1 = NodeTower(c); self.tower_lat = NodeTower(c); self.tower_lon = NodeTower(c)
        self.tower_path = NumericTower(c.path_feat, c, c.n_lat); self.tower_prof = NumericTower(c.prof_feat, c, c.n_lon)
        self.theta = nn.ParameterDict({k: nn.Parameter(torch.tensor(c.logit_init)) for k in ("s1", "s2lat", "s2lon", "s3P", "s3V")})
        self.pair_bias = nn.Parameter(torch.zeros(c.n_lat, c.n_lon))     # learned scalar b(c_lat, c_lon); -inf mask applied separately
        self.pq_head = MLP(c.d, c.d, 1)                                 # pair head for (p,q): on s3P*s3V*a_p*a_q
        self.cost = nn.ModuleList([MLP(c.d, c.d, 1) for _ in range(c.n_cost)])   # Hydra-style, on s3 (.) a
        self.refine = MLP(2 * c.d, c.d, c.n_slots * 2)
        self.rerank = Rerank(c)

    def tau(self, k): return self.theta[k].clamp(max=self.c.logit_max).exp()

    def encode(self, batch, width=None):
        q, scene = self.fusion(batch["vlm"], batch["last"], batch["age_s"], batch["frames"], batch["ego"], batch["nav"],
                               batch["nav_args"], batch["route"], batch.get("ego_present"), batch.get("nav_present"),
                               batch.get("route_present"), batch.get("audio"), batch.get("audio_present"))
        s = {k: self.heads[k](q[:, i], width) for i, k in enumerate(LEVELS) if k != "goal"}
        g = self.goal_head(q[:, LEVELS.index("goal")])
        return s, g[:, : self.c.n_goals], g[:, self.c.n_goals:].view(-1, 3, 3), scene

    def n_params(self): return sum(p.numel() for p in self.parameters())

    def breakdown(self) -> dict:
        out = {n: sum(p.numel() for p in mod.parameters()) for n, mod in self.named_children()}
        out["_own"] = sum(p.numel() for p in self.parameters(recurse=False))
        return out
