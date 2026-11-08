"""STRAT-HEADS library: three strategic heads on the FROZEN ladder world model and the per-frame latent cache (Stream C of the band streams, 2026-11-08).

  head_ctx   VISION ONLY. Reads the cached mean adapter token (`zmean`, 256) of the current and past frames. Outputs country (25, class-balanced), illumination (4), weather (4), surface (3), visibility (3).
  head_ckpt  Time-indexed route checkpoints at 6, 9, ..., 30 s (x, y, heading, ego frame at NOW), decoded from the L3 states the frozen world model rolls (`HJEPAStack.rollout`, level 3, k <= 10 steps of 3.0 s).
             Training rolls under LOGGED macro-actions (mode `logged`) or a NEUTRAL zero action (mode `zero`, the only mode that exists at inference); `ck_act_mix` is the share of zero-action samples.
  head_gstr  Derived goals: 8 strategic goal tokens, by_time_s, within_m, and two context-derived constraints (speed margin, headway margin). Reads the L3 state at NOW, nav and the DETACHED head_ctx posteriors.

Gradient discipline: the world model is frozen (requires_grad False, eval, no_grad); each head's loss reaches only that head; head_ctx receives exactly 0.0 from head_gstr's loss (the detach is measured, and a
mutation that removes it goes red). Nothing here reads an ego quantity inside head_ctx or head_gstr; the source scans and the invariance probes below pin that.
Evidence classes: every number this module produces is MEASURED on the cache (tier T0: teacher-forced vision latents; checkpoints under logged actions are T0 diagnostics). STRATEGIC is reported PAUSED (PI item 25).
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import math
import os
import textwrap
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------------------------------- schema
ILLUM = ("dark", "lit_night", "twilight", "day")                 # glare folded into day (merge_full.py)
WEATHER = ("clear", "rain", "snow", "fog")
SURFACE = ("dry", "wet", "snow_ice")
VISIB = ("good", "reduced", "poor")
COUNTRIES = ("United States", "Germany", "France", "Italy", "Sweden", "Spain", "Portugal", "Greece", "Austria", "Finland", "Croatia", "Netherlands", "Denmark", "Slovenia",
             "Estonia", "Slovakia", "Belgium", "Czechia", "Lithuania", "Poland", "Romania", "Luxembourg", "Latvia", "Hungary", "Bulgaria")        # CENSUS_STRAT0.json, 25
CTX_ATTRS = (("illumination", ILLUM), ("weather", WEATHER), ("surface", SURFACE), ("visibility", VISIB))
CTX_DIMS = (len(COUNTRIES),) + tuple(len(c) for _, c in CTX_ATTRS)         # 25, 4, 4, 3, 3
STR_TOKENS = ("FOLLOW_ROUTE", "TURN_LEFT_FOLLOW_ROUTE", "TURN_RIGHT_FOLLOW_ROUTE", "STOP_AT_FOLLOW_ROUTE", "EXIT_LEFT_FOLLOW_ROUTE", "EXIT_RIGHT_FOLLOW_ROUTE",
              "LANE_CHANGE_L_FOLLOW_ROUTE", "LANE_CHANGE_R_FOLLOW_ROUTE")                    # vocab_v7.STRATEGIC_GOAL_TOKENS_V7 (a test pins equality)
HORIZONS_S = tuple(float(h) for h in range(6, 31, 3))                                          # 6, 9, ..., 30
N_HZ = len(HORIZONS_S)
DT = 0.2                                                                                      # cache grid step (s)
HIST_OFFSETS = (0, 5, 10, 25)                                                                 # frames back: now, 1 s, 2 s, 5 s
STEP_FUT = 150                                                                                # raw future actions read per sample (30 s)
CK_SCALE_XY_PER_S = 10.0                                                                      # m per s of horizon, normaliser of the L1 loss (not a model of the path)
CK_SCALE_H = 1.0                                                                              # rad
CON_SCALE = (0.4, 3.0)                                                                        # speed margin fraction, headway margin s (normalisers)
BT_SCALE = 30.0

# context-derived constraint RULE (HYPOTHESIS, a design item for the PI; SPEC_HWM section 3): speed margin = fraction of the road's allowed speed to give up, headway margin = extra time gap in s.
_SPEED = {"illumination": (0.10, 0.05, 0.03, 0.0), "weather": (0.0, 0.10, 0.20, 0.20), "surface": (0.0, 0.05, 0.15), "visibility": (0.0, 0.05, 0.15)}
_HEAD = {"illumination": (0.5, 0.3, 0.2, 0.0), "weather": (0.0, 0.5, 1.0, 1.0), "surface": (0.0, 0.5, 1.0), "visibility": (0.0, 0.3, 1.0)}


def constraint_rule(ctx4: np.ndarray) -> tuple:
    """ctx4 [..., 4] ints (-1 unknown) -> (target [..., 2] float32, mask [...] bool). Unknown attributes add nothing; the row is masked only when ALL four are unknown."""
    ctx4 = np.asarray(ctx4)
    out = np.zeros(ctx4.shape[:-1] + (2,), np.float32)
    for i, (name, _) in enumerate(CTX_ATTRS):
        k = ctx4[..., i]
        for tab, col in ((_SPEED[name], 0), (_HEAD[name], 1)):
            v = np.asarray(tab, np.float32)[np.clip(k, 0, len(tab) - 1)] * (k >= 0)
            out[..., col] += v
    out[..., 0] = np.minimum(out[..., 0], CON_SCALE[0])
    out[..., 1] = np.minimum(out[..., 1], CON_SCALE[1])
    return out, (ctx4 >= 0).any(-1)


# ---------------------------------------------------------------------------------------------------- source scans and probes
CTX_FORBIDDEN = frozenset({"act", "act_f", "act_f_n", "nav", "v0", "limit", "pose", "speed", "ego", "kin3", "p10f", "situation", "sitclf", "tactical", "gstr", "g_tok", "ck", "ckpt", "label", "labels",
                           "country_y", "ctx_y", "z3_now", "rolled"})
GSTR_FORBIDDEN = frozenset({"act", "act_f", "act_f_n", "v0", "limit", "pose", "speed", "ego", "kin3", "p10f", "situation", "sitclf", "situations", "situation_class", "tactical", "head_t", "tac", "label",
                            "labels", "g_tok", "country_y", "ctx_y", "zhist", "hist_ok", "ck_y", "ckpt"})
CKPT_FORBIDDEN = frozenset({"act", "act_f", "v0", "limit", "pose", "speed", "ego", "kin3", "p10f", "situation", "sitclf", "tactical", "nav", "label", "labels", "zhist"})
CTX_ALLOWED_KEYS = frozenset({"zhist", "hist_ok"})
GSTR_ALLOWED_KEYS = frozenset({"nav"})
NONVISION_KEYS = ("act_f", "act_f_n", "nav", "v0", "ck_y", "ck_m", "country", "ctx4", "g_tok", "g_bt", "g_wi", "g_m", "con_y", "con_m")


class KeyGuard(dict):
    """A dict that records which keys are read (the signature test for the vision-only rule)."""

    def __init__(self, d):
        super().__init__(d)
        self.read = set()

    def __getitem__(self, k):
        self.read.add(k)
        return super().__getitem__(k)

    def __contains__(self, k):
        self.read.add(k)
        return super().__contains__(k)

    def get(self, k, default=None):
        self.read.add(k)
        return super().get(k, default)


def source_problems(fn, forbidden) -> list:
    """Identifiers, attributes and string subscripts of ``fn``'s CODE (not its docstring) that name a forbidden input."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    f = tree.body[0]
    body = ast.Module(body=[s for s in f.body if not (isinstance(s, ast.Expr) and isinstance(getattr(s, "value", None), ast.Constant) and isinstance(s.value.value, str))], type_ignores=[])
    names = set(a.arg for a in f.args.args + f.args.kwonlyargs)
    for n in ast.walk(body):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
        elif isinstance(n, ast.Subscript) and isinstance(getattr(n, "slice", None), ast.Constant) and isinstance(n.slice.value, str):
            names.add(n.slice.value)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
            names.add(n.args[0].value)
    return sorted(names & set(forbidden))


def other_batch(batch: dict, keep=("zhist", "hist_ok")) -> dict:
    """Every key except ``keep`` replaced by a different value of the same shape (a bool flips, a float is redrawn, an int shifts)."""
    g = torch.Generator().manual_seed(77)
    out = {}
    for k, v in batch.items():
        if k in keep or not torch.is_tensor(v):
            out[k] = v
        elif v.is_floating_point():
            out[k] = torch.randn(v.shape, generator=g).to(v.dtype) * 3.0 + 1.0
        elif v.dtype == torch.bool:
            out[k] = ~v
        else:
            out[k] = v + 1
    return out


def assert_ctx_vision_only(head, batch: dict) -> dict:
    """head_ctx is a function of the VISION latents alone. Raises AssertionError unless (1) the head's own forward code names no non-vision input, (2) a key-guarded run reads only CTX_ALLOWED_KEYS,
    (3) the output is bit-identical when every other batch key changes, and (4) it differs when ``zhist`` changes."""
    src = source_problems(type(head).forward, CTX_FORBIDDEN)
    if src:
        raise AssertionError(f"head_ctx code names non-vision inputs: {src}")
    was = head.training
    head.eval()
    try:
        with torch.no_grad():
            g = KeyGuard(batch)
            o0 = head(g)
            extra = g.read - CTX_ALLOWED_KEYS
            if extra:
                raise AssertionError(f"head_ctx READS batch keys {sorted(extra)} (admissible: {sorted(CTX_ALLOWED_KEYS)})")
            o1 = head(other_batch(batch))
            for k in o0:
                if not torch.equal(o0[k], o1[k]):
                    raise AssertionError(f"head_ctx output {k} moved by {float((o0[k] - o1[k]).abs().max())} when only non-vision inputs changed")
            mv = dict(batch)
            mv["zhist"] = batch["zhist"] + torch.randn(batch["zhist"].shape, generator=torch.Generator().manual_seed(3)) * 0.5
            o2 = head(mv)
            if all(torch.equal(o0[k], o2[k]) for k in o0):
                raise AssertionError("head_ctx output did not move when the vision latents changed (the head is dead)")
    finally:
        head.train(was)
    return {"vision_only": True, "reads": sorted(g.read), "source_forbidden_hits": []}


def assert_gstr_goal_path(head, batch: dict, z3: torch.Tensor, ctx_p: torch.Tensor) -> dict:
    """The goal path reads {L3 state, nav, ctx posteriors} and nothing else: source scan (no situation-classifier or ego names), key-guard (only nav), bit-identity under a change of every other batch key."""
    src = source_problems(type(head).forward, GSTR_FORBIDDEN)
    if src:
        raise AssertionError(f"head_gstr code names forbidden inputs (situation classifier / ego / labels): {src}")
    was = head.training
    head.eval()
    try:
        with torch.no_grad():
            g = KeyGuard(batch)
            o0 = head(g, z3, ctx_p)
            extra = g.read - GSTR_ALLOWED_KEYS
            if extra:
                raise AssertionError(f"head_gstr READS batch keys {sorted(extra)} (admissible: {sorted(GSTR_ALLOWED_KEYS)})")
            o1 = head(other_batch(batch, keep=("nav",)), z3, ctx_p)
            for k in o0:
                if not torch.equal(o0[k], o1[k]):
                    raise AssertionError(f"head_gstr output {k} moved when only non-input batch keys changed")
    finally:
        head.train(was)
    return {"reads": sorted(g.read), "source_forbidden_hits": [], "inputs": ["z3_now (frozen WM L3 state)", "nav", "ctx posteriors (detached)"]}


def tensor_sha(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode() + str(tuple(t.shape)).encode() + str(t.dtype).encode() + t.numpy().tobytes())
    return h.hexdigest()


def file_sha(p, blk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(blk), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------- frozen world model
class FrozenWM:
    """The ladder world model's level encoders, macro-action encoders and level-3 predictor, frozen. Everything runs under no_grad in eval mode. Requires window-1 encoders (K = 1 history), so a
    per-frame state is the pointwise lift of that frame's z1; the tick frames reproduce ``HJEPAStack.encode`` bit for bit (checked in the selfcheck)."""

    def __init__(self, wm):
        self.wm = wm
        st = wm.stack
        self.stack = st
        if st.n_levels != 3:
            raise ValueError(f"FrozenWM needs 3 levels, the stack has {st.n_levels}")
        if wm.enc1 is not None or any(int(st.encoders[i].window) != 1 for i in range(2)):
            raise ValueError("FrozenWM supports the K = 1 world model only (window-1 encoders)")
        self.s2, self.s3 = int(st.cfg[1].stride), int(st.cfg[2].stride)
        self.step_s = DT * self.s2 * self.s3
        self.n_steps = int(round(HORIZONS_S[-1] / self.step_s))
        self.d3 = int(st.cfg[2].latent_dim)
        wm.eval()
        for p in wm.parameters():
            p.requires_grad_(False)

    def sha(self) -> str:
        return tensor_sha(self.wm.state_dict())

    @torch.no_grad()
    def z3_of(self, zmean: torch.Tensor) -> torch.Tensor:
        """zmean [B, D] (cache) -> the L3 state of that frame [B, D3]."""
        z1 = self.wm.z1_from_mean(zmean.float()[:, None])[:, 0]
        return self.stack.encoders[1](self.stack.encoders[0](z1))

    @torch.no_grad()
    def macro3(self, act_raw: torch.Tensor) -> torch.Tensor:
        """act_raw [B, n_steps * s2 * s3, 2] raw actions from NOW (action j = transition j -> j+1) -> L3 macro-actions [B, n_steps, A3] (the planning action space of level 3)."""
        st = self.stack
        a = self.wm.scale_actions(act_raw.float())
        e1 = st.act_embed(a)
        B = e1.shape[0]
        n2 = e1.shape[1] // self.s2
        a2 = st.macro_encoders[0](e1[:, :n2 * self.s2].reshape(B, n2, self.s2, e1.shape[-1]))
        n3 = n2 // self.s3
        return st.macro_encoders[1](a2[:, :n3 * self.s3].reshape(B, n3, self.s3, a2.shape[-1]))

    @torch.no_grad()
    def rolled(self, z3: torch.Tensor, act_raw: torch.Tensor) -> torch.Tensor:
        """L3 states after 1 .. n_steps steps, [B, n_steps, D3], under the given raw actions (context 1, as planning)."""
        return self.stack.rollout(3, z3, self.macro3(act_raw), context=1)


# ---------------------------------------------------------------------------------------------------- heads
class CtxHead(nn.Module):
    """Vision only. Input: ``batch['zhist']`` [B, H, D] (the cached mean adapter token at NOW and at past offsets) and ``batch['hist_ok']`` [B, H]."""

    def __init__(self, d_lat: int = 256, hidden: int = 256, dims=CTX_DIMS, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(2 * d_lat)
        self.trunk = nn.Sequential(nn.Linear(2 * d_lat, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU())
        self.out = nn.ModuleList([nn.Linear(hidden, n) for n in dims])
        self.names = ("country",) + tuple(n for n, _ in CTX_ATTRS)

    def forward(self, batch) -> dict:
        z = batch["zhist"].float()
        ok = batch["hist_ok"].float()
        pooled = (z * ok[..., None]).sum(1) / ok.sum(1, keepdim=True).clamp_min(1.0)
        h = self.trunk(self.norm(torch.cat([z[:, 0], pooled], -1)))
        return {n: o(h) for n, o in zip(self.names, self.out)}


class CkptHead(nn.Module):
    """Decodes the rolled L3 state at each checkpoint horizon (x, y, heading), in normalised units. Reads only its tensor arguments."""

    def __init__(self, d3: int = 128, hidden: int = 256, horizons=HORIZONS_S, step_s: float = 3.0, emb: int = 16):
        super().__init__()
        self.register_buffer("horizons_s", torch.tensor(horizons, dtype=torch.float32))
        self.register_buffer("k_index", torch.tensor([int(round(h / step_s)) - 1 for h in horizons], dtype=torch.long))      # rolled[:, k-1] is the state after k steps
        self.hz_emb = nn.Parameter(torch.randn(len(horizons), emb) * 0.1)
        self.norm_r, self.norm_n = nn.LayerNorm(d3), nn.LayerNorm(d3)
        self.mlp = nn.Sequential(nn.Linear(2 * d3 + emb, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))

    def forward(self, z3_now: torch.Tensor, rolled: torch.Tensor) -> torch.Tensor:
        r = self.norm_r(rolled[:, self.k_index])                               # [B, n_hz, D3]
        n = self.norm_n(z3_now)[:, None].expand(-1, r.shape[1], -1)
        e = self.hz_emb[None].expand(r.shape[0], -1, -1)
        return self.mlp(torch.cat([r, n, e], -1))                              # [B, n_hz, 3]


class GstrHead(nn.Module):
    """Derived strategic goals. Inputs: the L3 state at NOW, nav (``batch['nav']``) and the ctx posteriors, which arrive already detached."""

    def __init__(self, d3: int = 128, nav_dim: int = 8, hidden: int = 256, ctx_dim: int = sum(CTX_DIMS)):
        super().__init__()
        self.nav_dim = nav_dim
        self.norm = nn.LayerNorm(d3)
        self.trunk = nn.Sequential(nn.Linear(d3 + nav_dim + ctx_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.tok = nn.Linear(hidden, len(STR_TOKENS))
        self.reg = nn.Linear(hidden, 2)                                        # by_time_s / BT_SCALE, within_m / within scale
        self.con = nn.Linear(hidden, 2)                                        # speed margin / 0.4, headway margin / 3 s
        self.register_buffer("within_scale", torch.tensor(300.0))

    def forward(self, batch, z3_now: torch.Tensor, ctx_p: torch.Tensor) -> dict:
        h = self.trunk(torch.cat([self.norm(z3_now), batch["nav"].float(), ctx_p], -1))
        return {"tok": self.tok(h), "reg": self.reg(h), "con": self.con(h)}


class StratHeads(nn.Module):
    def __init__(self, d_lat=256, d3=128, nav_dim=8, ctx_hidden=256, ck_hidden=256, g_hidden=256, step_s=3.0, dropout=0.1):
        super().__init__()
        self.ctx = CtxHead(d_lat, ctx_hidden, dropout=dropout)
        self.ckpt = CkptHead(d3, ck_hidden, step_s=step_s)
        self.gstr = GstrHead(d3, nav_dim, g_hidden)

    def ctx_for_gstr(self, out: dict) -> torch.Tensor:
        """The edge head_ctx -> head_gstr: the softmax posteriors, DETACHED. The detach is what keeps gstr's loss off head_ctx (measured as an exact 0.0 in the selfcheck)."""
        return torch.cat([F.softmax(out[n], -1) for n in self.ctx.names], -1).detach()

    def forward_all(self, batch: dict, z3_now: torch.Tensor, rolled: torch.Tensor) -> dict:
        c = self.ctx(batch)
        return {"ctx": c, "ck": self.ckpt(z3_now, rolled), "g": self.gstr(batch, z3_now, self.ctx_for_gstr(c))}


# ---------------------------------------------------------------------------------------------------- losses
def wrap(d: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(d), torch.cos(d))


def ck_scale(horizons: torch.Tensor) -> torch.Tensor:
    return torch.stack([CK_SCALE_XY_PER_S * horizons, CK_SCALE_XY_PER_S * horizons, torch.full_like(horizons, CK_SCALE_H)], -1)         # [n_hz, 3]


def loss_ctx(out: dict, batch: dict, wts: dict) -> tuple:
    """CE per attribute with per-attribute masks (label -1 = unknown); country uses the class-balanced weight vector, the others their own."""
    terms = {}
    lab = {"country": batch["country"]}
    for i, (n, _) in enumerate(CTX_ATTRS):
        lab[n] = batch["ctx4"][:, i]
    for n, y in lab.items():
        m = y >= 0
        if m.any():
            terms[n] = F.cross_entropy(out[n][m], y[m], weight=wts.get(n))
        else:
            terms[n] = out[n].sum() * 0.0
    return sum(terms.values()) / len(terms), terms


def loss_ck(pred: torch.Tensor, batch: dict, horizons: torch.Tensor, mask: torch.Tensor) -> tuple:
    """Masked L1 per horizon in normalised units (x, y, wrapped heading). ``mask`` [B, n_hz]."""
    sc = ck_scale(horizons)[None]
    y = batch["ck_y"]
    d = (pred - y / sc)
    d = torch.stack([d[..., 0], d[..., 1], wrap(pred[..., 2] * CK_SCALE_H - y[..., 2]) / CK_SCALE_H], -1)
    per = d.abs().sum(-1) * mask.float()                                       # [B, n_hz]
    n = mask.float().sum(0).clamp_min(1.0)
    per_h = per.sum(0) / n
    return per.sum() / mask.float().sum().clamp_min(1.0), per_h


def loss_g(out: dict, batch: dict, wts: torch.Tensor, within_scale: float) -> tuple:
    m = batch["g_m"]
    terms = {}
    terms["tok"] = F.cross_entropy(out["tok"][m], batch["g_tok"][m], weight=wts) if m.any() else out["tok"].sum() * 0.0
    ma = m & torch.isfinite(batch["g_bt"]) & torch.isfinite(batch["g_wi"]) & (batch["g_tok"] != 0)
    if ma.any():
        tgt = torch.stack([batch["g_bt"][ma] / BT_SCALE, batch["g_wi"][ma] / within_scale], -1)
        terms["args"] = F.smooth_l1_loss(out["reg"][ma], tgt)
    else:
        terms["args"] = out["reg"].sum() * 0.0
    mc = batch["con_m"]
    if mc.any():
        tgt = batch["con_y"][mc] / torch.tensor(CON_SCALE, device=batch["con_y"].device)
        terms["con"] = F.smooth_l1_loss(out["con"][mc], tgt)
    else:
        terms["con"] = out["con"].sum() * 0.0
    return sum(terms.values()), terms


# ---------------------------------------------------------------------------------------------------- data
@dataclass
class Clip:
    cid: str
    z: np.ndarray                  # [T, D] f32
    ok: np.ndarray                 # [T] bool
    lab: dict = field(default_factory=dict)
    side: dict = field(default_factory=dict)


def load_zclip(path: str) -> tuple:
    with np.load(path, allow_pickle=False) as d:
        return d["zmean"].astype(np.float32), d["ok"].astype(bool), d["t_s"].astype(np.float32)


class Labels:
    """Per-clip label arrays on the 0.2 s cache grid. Context + goal columns come from the release frames (parquet, `merge_full.py` schema); checkpoints from stream A's `strategic_geom_<split>.npz`
    (keys `clip_ids`, `clip__off`, `clip__n`, `ck_xyh` [N, 9, 3], `ck_valid` [N, 9]; band_labels/build_strategic_geom.py) or, as a fallback, the release's `ckpt_*` columns."""

    def __init__(self, release_dir: str, geom_npz: str | None = None, clips=None):
        import pyarrow.parquet as pq
        import glob
        cols = ["clip_id", "frame_idx", "country", "weather", "surface", "visibility", "illumination", "illum_img", "g_str_token", "g_str_by_time_s", "g_str_within_m", "g_str_valid",
                "ckpt_x", "ckpt_y", "ckpt_heading", "ckpt_valid_mask"]
        self.rows = {}
        for f in sorted(glob.glob(os.path.join(release_dir, "frames", "*.parquet"))):
            have = set(pq.read_schema(f).names)
            tab = pq.read_table(f, columns=[c for c in cols if c in have])
            for c, df in tab.to_pandas().groupby("clip_id", sort=False):
                if clips is None or c in clips:
                    self.rows[c] = df.sort_values("frame_idx")
        self.geom = None
        if geom_npz:
            z = np.load(geom_npz, allow_pickle=False)
            ids = [str(x) for x in z["clip_ids"]]
            off, n = z["clip__off"], z["clip__n"]
            self.geom = {c: (int(off[i]), int(n[i])) for i, c in enumerate(ids)}
            self._xyh, self._valid = z["ck_xyh"], z["ck_valid"]
            if self._xyh.shape[1:] != (N_HZ, 3):
                raise ValueError(f"{geom_npz}: ck_xyh is {self._xyh.shape}, expected [N, {N_HZ}, 3]")

    def clip_ids(self) -> list:
        return sorted(self.rows)

    def get(self, cid: str, T: int) -> dict | None:
        df = self.rows.get(cid)
        if df is None:
            return None
        idx = df["frame_idx"].to_numpy().astype(int)
        keep = idx < T
        df, idx = df[keep], idx[keep]

        def col(name, dtype, fill):
            a = np.full(T, fill, dtype)
            if name in df:
                a[idx] = df[name].to_numpy()
            return a

        country = df["country"].iloc[0] if len(df) else None
        out = {"country": COUNTRIES.index(country) if country in COUNTRIES else -1}
        ctx = np.full((T, 4), -1, np.int64)
        for i, (name, classes) in enumerate(CTX_ATTRS):
            src = name if name in df else ("illum_img" if name == "illumination" and "illum_img" in df else None)
            if src is None:
                continue
            for j, v in zip(idx, df[src].to_numpy()):
                v = "day" if (name == "illumination" and v == "glare") else v
                ctx[j, i] = classes.index(v) if v in classes else -1
        out["ctx4"] = ctx
        tok = np.full(T, -1, np.int64)
        if "g_str_token" in df:
            for j, v in zip(idx, df["g_str_token"].to_numpy()):
                tok[j] = STR_TOKENS.index(v) if v in STR_TOKENS else -1
        out["g_tok"] = tok
        out["g_bt"] = col("g_str_by_time_s", np.float32, np.nan)
        out["g_wi"] = col("g_str_within_m", np.float32, np.nan)
        gv = col("g_str_valid", bool, False)
        out["g_m"] = gv & (tok >= 0)
        con, cm = constraint_rule(ctx)
        out["con_y"], out["con_m"] = con, cm
        ck = np.zeros((T, N_HZ, 3), np.float32)
        cv = np.zeros((T, N_HZ), bool)
        if self.geom is not None and cid in self.geom:
            o, n = self.geom[cid]
            m = min(n, T)
            ck[:m], cv[:m] = self._xyh[o:o + m], self._valid[o:o + m]
        elif "ckpt_x" in df:
            for j, x, y, h, v in zip(idx, df["ckpt_x"], df["ckpt_y"], df["ckpt_heading"], df["ckpt_valid_mask"]):
                k = len(x)
                ck[j, :k] = np.stack([x, y, h], -1)
                cv[j, :k] = v
        out["ck_y"], out["ck_m"] = ck, cv & np.isfinite(ck).all(-1)
        return out


def load_side(path: str) -> dict:
    """act [Te, 2] raw (a m/s^2, yaw rate rad/s) from NOW's clock on the 0.2 s grid (Te >= T), v [Te], nav [T, Dn]."""
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def load_clipset(zdir: str, labels: Labels, side_dir: str | None, clip_ids=None, limit: int = 0) -> list:
    import glob
    files = sorted(glob.glob(os.path.join(zdir, "*.npz")))
    out = []
    for f in files:
        cid = os.path.basename(f)[:-4]
        if clip_ids is not None and cid not in clip_ids:
            continue
        z, ok, _ = load_zclip(f)
        lab = labels.get(cid, len(z))
        if lab is None:
            continue
        side = {}
        if side_dir:
            sp = os.path.join(side_dir, cid + ".npz")
            if not os.path.exists(sp):
                continue
            side = load_side(sp)
        out.append(Clip(cid, z, ok, lab, side))
        if limit and len(out) >= limit:
            break
    return out


def country_weights(clips: list, power: float) -> torch.Tensor:
    n = np.zeros(len(COUNTRIES))
    for c in clips:
        if c.lab["country"] >= 0:
            n[c.lab["country"]] += 1
    w = np.where(n > 0, (n.sum() / np.maximum(n, 1)) ** power, 0.0)
    w = w / max(w[n > 0].mean(), 1e-9) if (n > 0).any() else w
    return torch.tensor(w, dtype=torch.float32)


def class_weights(clips: list, key: str, idx: int | None, n_cls: int, power: float) -> torch.Tensor:
    n = np.zeros(n_cls)
    for c in clips:
        a = c.lab[key] if idx is None else c.lab[key][:, idx]
        a = a[a >= 0]
        n += np.bincount(a, minlength=n_cls)[:n_cls]
    w = np.where(n > 0, (n.sum() / np.maximum(n, 1)) ** power, 0.0)
    w = w / max(w[n > 0].mean(), 1e-9) if (n > 0).any() else w
    return torch.tensor(w, dtype=torch.float32)


class Sampler:
    """Draws (clip, frame) samples and assembles batches. ``ck_act_mix`` = share of zero-action rollouts, ``hist_offsets`` the frames back the ctx head reads."""

    def __init__(self, clips: list, batch: int, seed: int, ck_act_mix: float = 0.5, hist_offsets=HIST_OFFSETS, nav_dim: int = 8):
        self.clips, self.batch, self.ck_act_mix, self.hist_offsets, self.nav_dim = clips, int(batch), float(ck_act_mix), tuple(hist_offsets), int(nav_dim)
        self.rng = np.random.default_rng(seed)
        self.pool = [(i, t) for i, c in enumerate(clips) for t in range(len(c.z)) if c.ok[t]]
        if not self.pool:
            raise ValueError("Sampler: no usable (clip, frame) pair")

    def draw(self, n: int | None = None) -> list:
        n = self.batch if n is None else n
        return [self.pool[j] for j in self.rng.integers(0, len(self.pool), n)]

    def assemble(self, items: list, zero_mask: np.ndarray | None = None) -> dict:
        B, H, D = len(items), len(self.hist_offsets), self.clips[items[0][0]].z.shape[1]
        zh, hok = np.zeros((B, H, D), np.float32), np.zeros((B, H), bool)
        nav, af, afn, v0 = np.zeros((B, self.nav_dim), np.float32), np.zeros((B, STEP_FUT, 2), np.float32), np.zeros(B, np.int64), np.zeros(B, np.float32)
        ck_y, ck_m = np.zeros((B, N_HZ, 3), np.float32), np.zeros((B, N_HZ), bool)
        country, ctx4 = np.zeros(B, np.int64), np.zeros((B, 4), np.int64)
        g_tok, g_bt, g_wi, g_m = np.zeros(B, np.int64), np.zeros(B, np.float32), np.zeros(B, np.float32), np.zeros(B, bool)
        con_y, con_m, ci = np.zeros((B, 2), np.float32), np.zeros(B, bool), np.zeros(B, np.int64)
        for b, (i, t) in enumerate(items):
            c = self.clips[i]
            for h, off in enumerate(self.hist_offsets):
                j = t - off
                if j >= 0 and c.ok[j]:
                    zh[b, h], hok[b, h] = c.z[j], True
            hok[b, 0] = True
            s = c.side
            if s:
                if "nav" in s:
                    nv = np.nan_to_num(s["nav"][min(t, len(s["nav"]) - 1)]).astype(np.float32)
                    nav[b, :min(len(nv), self.nav_dim)] = nv[:self.nav_dim]
                a = np.nan_to_num(s["act"][t:t + STEP_FUT])
                af[b, :len(a)] = a
                afn[b] = len(a)
                if "v" in s:
                    v0[b] = float(np.nan_to_num(s["v"][min(t, len(s["v"]) - 1)]))
            L = c.lab
            ck_y[b], ck_m[b] = L["ck_y"][t], L["ck_m"][t]
            country[b], ctx4[b] = L["country"], L["ctx4"][t]
            g_tok[b], g_bt[b], g_wi[b], g_m[b] = max(L["g_tok"][t], 0), L["g_bt"][t], L["g_wi"][t], L["g_m"][t]
            con_y[b], con_m[b], ci[b] = L["con_y"][t], L["con_m"][t], i
        zm = (self.rng.random(B) < self.ck_act_mix) if zero_mask is None else zero_mask
        T = torch.as_tensor
        return {"zhist": T(zh), "hist_ok": T(hok), "nav": T(nav), "act_f": T(af), "act_f_n": T(afn), "v0": T(v0), "ck_y": T(ck_y), "ck_m": T(ck_m), "country": T(country), "ctx4": T(ctx4),
                "g_tok": T(g_tok), "g_bt": T(g_bt), "g_wi": T(g_wi), "g_m": T(g_m), "con_y": T(con_y), "con_m": T(con_m), "zero_act": T(zm), "ci": T(ci),
                "t_idx": T(np.array([t for _, t in items], np.int64))}


def ck_mask(batch: dict, mode_zero: torch.Tensor, step_s: float) -> torch.Tensor:
    """[B, n_hz] bool: label valid, and (logged mode) the logged actions reach the horizon."""
    need = torch.tensor([int(round(h / DT)) for h in HORIZONS_S])                     # raw actions needed to reach the horizon
    have = batch["act_f_n"][:, None] >= need[None]
    return batch["ck_m"] & (mode_zero[:, None] | have)


def prepare_wm(fz: FrozenWM, batch: dict, zero_act: torch.Tensor) -> tuple:
    """z3 at NOW and the rolled L3 states for the whole batch; per row the actions are the logged ones or zeros."""
    z3 = fz.z3_of(batch["zhist"][:, 0])
    act = torch.where(zero_act[:, None, None], torch.zeros_like(batch["act_f"]), batch["act_f"])
    return z3, fz.rolled(z3, act)
