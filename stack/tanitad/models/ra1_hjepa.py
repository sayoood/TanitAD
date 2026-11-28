"""RA-1 (refav1-HJ mini): the arms R0 / A0 / A1 / A2 / A1x (+ exploratory A1h, + the mechanism control A1a) on POOLED FROZEN
DINOv3-L tokens.

Spec: ``Project Steering/PREREG_HJ_STAGE0.md`` (H-HJ-SENS-1 / H-HJ-K-1 / H-HJ-COST-1) -- no bar is changed here. Nothing in this
file re-implements the H-JEPA objective: SIGReg, the IDM head and loss, the level / macro-action encoders, the AdaLN causal
predictor, ``action_sensitivity``, ``monotonicity``, ``rho_within_episode`` and ``dead_parameters`` all come from
``tanitad.models.hjepa`` (45 tests). What is new here is ONLY the wiring around them:

* ``PooledAdapter``  -- the trainable level-1 ENCODER on the frozen pooled tokens ``(.., 40, 1024) -> (.., D1)``.
* ``Readout``        -- the 6 s trajectory readout over ``[level latents, 8 standardised P10 features]``, as a RESIDUAL on the
                        frozen P10 prior (``P10Prior``: ``kd_x(steer unit) + MLP``). IDENTICAL in every arm (matched capacity).
* ``RA1Model``       -- adapter + (optional) ``HJEPAStack`` + readout, and the one loss that joins them.

ARMS (all share adapter, readout, data, optimiser, seeds; they differ ONLY in what the arm name says)
    R0   adapter -> readout over ``[z1, P10]``. No world model: no predictor, no SIGReg, no IDM. The frozen-encoder ceiling.
    A0   2-level H-JEPA, SIGReg only (gamma = gamma_up = 0).
    A1   A0 + IDM at both levels (``beta``/``gamma`` are LEVEL 1's; upper levels use the paper's DROID level-2 values 0.16 / 50).
    A2   A1 + level 3 (s3 = 3 => 1.8 s).
    A1x  A1 with the WM losses computed on a per-window TIME-SHUFFLED frame order: a DELIBERATE REGRESSION that must lose to A1.
    A1h  EXPLORATORY (orchestrator addendum 2026-10-17; never decides H-HJ-SENS-1): A1 + HISTORY DROPOUT in every level's
         predictor (the copycat remedy, arXiv 2010.14876): with probability ``hist_drop`` per (window, position) the position
         is predicted from itself alone (context 1) instead of from its causal history.
    A1a  MECHANISM CONTROL (PREREG_RA1_ES Addendum D, 2026-10-18; does NOT reopen H-HJ-K-1): A1 with every window's ACTION
         SEQUENCE replaced by ANOTHER window's (a seeded derangement of the batch, ``window_derangement``) before the WM losses.
         The frames stay in true order (unlike A1x, which keeps each window's own actions and so cannot remove them); what is
         removed is the pairing of a window's frames with its own actions -- the TRUE action information.

THE ACTION VECTOR (3 channels, STANDARDISED with train statistics so Prop. 3's unit-variance premise holds):
    ``a`` (m/s^2, the acceleration of the step that OPENS at frame t), channel 1, ``v`` (m/s, the measured speed at frame t:
    refav1's speed conditioning channel, kept so the lateral-vs-speed sensitivity read is comparable with RA-0).
    ``a_seq[:, t]`` is the block for the transition ``t -> t+1`` (hjepa.py convention).
    CHANNEL 1 is what ``cfg.kin_mode`` says (PI ruling D-PI-YAWRATE-1, 2026-10-17: *"the model should get measured ego motion and
    its history, instead of steering angle which represents a sort of action, the model should get the measured yaw rate"*):
      ``kin_mode="yaw"`` (the compliant, DEFAULT-in-the-trainer mode): ``yaw_rate`` (rad/s) = the MEASURED yaw rate of the pose
      track, ONE grid frame back (``ego_channels_from_poses``' backward wrapped difference / 0.2 s) -- measured motion, not an
      action; the P10 features and the kd_x base are the yaw-derived ``kappa0`` ones and the base is integrated as a CURVATURE.
      ``kin_mode="steer"`` (legacy; the 2026-10-17 22:35 box-2 runs; NON-DECIDING since the ruling): ``steer`` (rad, a ROAD-WHEEL
      STEER ANGLE = arctan(2.9 kappa), NOT a curvature -- RETRACTION_LOG R46). Every path integration goes through
      ``taniteval.floors_kin`` with ``action_units="steer"``.

DESIGN CONSTANTS NOT FIXED BY THE PREREG (identical across arms; hand-back item): ``w_traj`` = 10 (trajectory loss in metres vs the
WM terms), level-1-only reading of the (beta, gamma) grid, upper-level weights 0.16 / 50 / act_sigreg 0.005, adapter 64-wide token
projection, T1 = 19 frames (3.6 s: exactly 3 level-2 and 1 level-3 transitions of s = 3).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from tanitad.models.hjepa import (HJEPAStack, LevelConfig, dead_parameters, monotonicity,
                                  param_grad_report)

ARMS = ("R0", "A0", "A1", "A2", "A1x", "A1h", "A1a")
#: the action vector's channel names per kin mode (PI ruling D-PI-YAWRATE-1); channel 1 is the lateral-control channel
ACT_CHANNELS_BY_MODE = {"steer": ("a", "steer", "v"), "yaw": ("a", "yaw_rate", "v")}
ACT_CHANNELS = ACT_CHANNELS_BY_MODE["steer"]            # legacy name (the steer representation)
ACT_DIM = len(ACT_CHANNELS)
KIN_MODES = tuple(ACT_CHANNELS_BY_MODE)


def ctl_channel(kin_mode: str) -> str:
    """The name of the lateral-control action channel (SENS' control read): ``steer`` | ``yaw_rate``."""
    return ACT_CHANNELS_BY_MODE[kin_mode][1]
HORIZON = 30                      # 30 x 0.2 s = 6 s
DT = 0.2
GRIDS = {"4x10": 40, "8x20": 160}  # grid -> tokens per frame


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    n_levels: int       # world-model levels built (R0: 0)
    wm: bool            # a HJEPAStack exists and its loss is added
    idm: bool           # IDM weights must be > 0
    shuffle_time: bool  # A1x
    hist_drop: bool     # A1h
    shuffle_act: bool = False   # A1a: action sequences permuted ACROSS windows before the WM losses


ARM_SPECS = {
    "R0": ArmSpec("R0", 0, False, False, False, False),
    "A0": ArmSpec("A0", 2, True, False, False, False),
    "A1": ArmSpec("A1", 2, True, True, False, False),
    "A2": ArmSpec("A2", 3, True, True, False, False),
    "A1x": ArmSpec("A1x", 2, True, True, True, False),
    "A1h": ArmSpec("A1h", 2, True, True, False, True),
    "A1a": ArmSpec("A1a", 2, True, True, False, False, shuffle_act=True),
}


@dataclass(frozen=True)
class RA1Config:
    """Everything that builds the model OR enters the loss. Saved in every checkpoint; the eval loader builds from it."""
    arm: str
    d_feat: int = 1024
    n_tok: int = 40
    grid: str = "4x10"
    d_tok: int = 64
    adapter_hidden: int = 512
    d1: int = 128
    d_up: int = 128
    macro_dim: int = 16
    enc_hidden: int = 384
    macro_hidden: int = 64
    pred_width: int = 192
    pred_depth: int = 4
    pred_heads: int = 6
    context: int = 3
    stride: int = 3
    t1: int = 19
    readout_hidden: int = 256
    sigreg_proj: int = 1024
    beta: float | None = None          # level 1 SIGReg weight (the grid's beta); MUST be named for every WM arm
    gamma: float | None = None         # level 1 IDM weight (the grid's gamma); MUST be named for A1/A2/A1x/A1h
    beta_up: float = 0.16
    gamma_up: float = 50.0
    act_sigreg: float = 0.005
    w_traj: float = 10.0
    hist_drop: float = 0.0
    kin_mode: str = "steer"            # steer (legacy; a config serialised before the ruling has no such key) | yaw (compliant)

    @property
    def spec(self) -> ArmSpec:
        return ARM_SPECS[self.arm]

    def n_up(self) -> int:
        return max(self.spec.n_levels - 1, 0)

    def level_weights(self) -> list:
        """``[(beta_l, gamma_l)]`` per WM level -- the EFFECTIVE weights (A0 has gamma exactly 0.0)."""
        if not self.spec.wm:
            return []
        g1 = float(self.gamma or 0.0) if self.spec.idm else 0.0
        gu = float(self.gamma_up) if self.spec.idm else 0.0
        return [(float(self.beta), g1)] + [(float(self.beta_up), gu)] * self.n_up()

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        if d["kin_mode"] == "steer":          # LEGACY-COMPATIBLE: a steer config serialises (and hashes) exactly as before the ruling,
            d.pop("kin_mode")                 # so a checkpoint written by the old code still resumes / re-loads under this one
        return d

    def sha12(self) -> str:
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True).encode()).hexdigest()[:12]


def validate_config(cfg: RA1Config) -> None:
    """Launch-gate spirit: a flag whose default disables its mechanism must be NAMED, and a flag that cannot reach the loss
    is refused. Raises ``ValueError``."""
    if cfg.arm not in ARMS:
        raise ValueError(f"arm {cfg.arm!r} not in {ARMS}")
    if cfg.kin_mode not in KIN_MODES:
        raise ValueError(f"kin_mode {cfg.kin_mode!r} not in {KIN_MODES}")
    if cfg.grid not in GRIDS or GRIDS[cfg.grid] != cfg.n_tok:
        raise ValueError(f"grid {cfg.grid!r} has {GRIDS.get(cfg.grid)} tokens/frame, config says n_tok={cfg.n_tok}")
    sp = cfg.spec
    if not sp.wm:
        stray = [k for k in ("beta", "gamma") if getattr(cfg, k) is not None] + (["hist_drop"] if cfg.hist_drop else [])
        if stray:
            raise ValueError(f"R0 has no world model: {stray} cannot reach any loss -- remove them from argv")
    else:
        if cfg.beta is None or not cfg.beta > 0.0:
            raise ValueError(f"{cfg.arm}: --beta must be named and > 0 (SIGReg is live in every WM arm)")
        if sp.idm and (cfg.gamma is None or not cfg.gamma > 0.0):
            raise ValueError(f"{cfg.arm}: --gamma must be named and > 0 -- an unnamed IDM weight silently makes this A0")
        if not sp.idm and cfg.gamma not in (None, 0.0):
            raise ValueError("A0 is SIGReg-only: --gamma must be absent or 0")
        if sp.idm and not cfg.gamma_up > 0.0:
            raise ValueError("--gamma-up must be > 0 for an IDM arm")
        if cfg.pred_width % cfg.pred_heads:
            raise ValueError("pred_width must be divisible by pred_heads")
        if cfg.context < 1:
            raise ValueError("context >= 1")
        t2 = (cfg.t1 - 1) // cfg.stride + 1
        if (cfg.t1 - 1) % cfg.stride or t2 < 2:
            raise ValueError(f"t1={cfg.t1} must satisfy (t1-1) % stride == 0 and give >= 2 level-2 states")
        if sp.n_levels >= 3 and ((t2 - 1) % cfg.stride or t2 < 1 + cfg.stride):
            raise ValueError(f"t1={cfg.t1}: level 2 has {t2} states, which does not end on a level-3 window boundary")
    if sp.hist_drop:
        if not 0.0 < cfg.hist_drop <= 1.0:
            raise ValueError("A1h needs --hist-drop in (0, 1] (named explicitly)")
    elif cfg.hist_drop:
        raise ValueError(f"--hist-drop is only meaningful for A1h, not {cfg.arm}")


# ============================================================================================ modules
class PooledAdapter(nn.Module):
    """The level-1 ENCODER: per-token projection (LayerNorm -> Linear), flatten (position-aware: 4x10 grid), MLP -> ``d_out``."""

    def __init__(self, d_feat: int, n_tok: int, d_tok: int, hidden: int, d_out: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_feat)
        self.tok = nn.Linear(d_feat, d_tok)
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(n_tok * d_tok, hidden), nn.GELU(), nn.Linear(hidden, d_out))

    def forward(self, x: Tensor) -> Tensor:                     # (..., n_tok, d_feat) float
        return self.mlp(self.tok(self.norm(x)).flatten(-2))


class Readout(nn.Module):
    """``[LN(level latents), P10 features(8, standardised)] -> 30 x 2`` residual on the P10 trajectory. The output layer starts
    NEAR zero (std 1e-3, not exactly 0: an exact zero would give the earlier layers a zero gradient at step 0, which the
    G-LIVE check could not tell from a dead layer)."""

    def __init__(self, d_lat: int, d_f: int, hidden: int, horizon: int = HORIZON):
        super().__init__()
        self.horizon = horizon
        self.norm = nn.LayerNorm(d_lat)
        self.net = nn.Sequential(nn.Linear(d_lat + d_f, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, horizon * 2))
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, lat: Tensor, f8n: Tensor) -> Tensor:
        y = self.net(torch.cat([self.norm(lat), f8n], dim=-1))
        return y.reshape(*y.shape[:-1], self.horizon, 2)


class HistoryDropPredictor(nn.Module):
    """A1h: wraps a ``CausalPredictor`` (same parameters, no new ones). In TRAINING, each (window, position) is, with probability
    ``p``, predicted from ITSELF ALONE (context 1: the causal history of past latents and past actions is removed); at eval the
    wrapper is the identity. Position 0 of any call already has no history. ``generator`` (CPU) makes the draw reproducible."""

    def __init__(self, inner: nn.Module, p: float):
        super().__init__()
        self.inner, self.p = inner, float(p)
        self.context = inner.context
        self.generator: torch.Generator | None = None
        self.n_dropped = 0           # positions replaced by their context-1 prediction (diagnostic)
        self.n_positions = 0

    def forward(self, z_ctx: Tensor, a_ctx: Tensor) -> Tensor:
        full = self.inner(z_ctx, a_ctx)
        b, l, d = z_ctx.shape
        if not self.training or self.p <= 0.0 or l == 1:
            return full
        solo = self.inner(z_ctx.reshape(b * l, 1, d), a_ctx.reshape(b * l, 1, a_ctx.shape[-1])).reshape(b, l, d)
        drop = (torch.rand(b, l, generator=self.generator) < self.p).to(z_ctx.device)
        self.n_dropped += int(drop.sum())
        self.n_positions += b * l
        return torch.where(drop[..., None], solo, full)


def time_shuffle_perm(b: int, t: int, generator: torch.Generator | None = None) -> Tensor:
    """``(b, t)`` per-window frame permutations; a row that came out as the identity is rolled by one, so EVERY window is
    genuinely shuffled (A1x is a regression arm: an identity row would silently be an A1 window)."""
    idx = torch.rand(b, t, generator=generator).argsort(dim=1)
    ident = (idx == torch.arange(t)[None]).all(dim=1)
    if ident.any():
        idx[ident] = torch.roll(idx[ident], 1, dims=1)
    return idx


def window_derangement(b: int, generator: torch.Generator | None = None) -> Tensor:
    """A1a: ``(b,)`` -- window ``i`` receives the action sequence of window ``perm[i]``, and NO window keeps its own (a
    derangement). Built as ONE random b-cycle (``perm[order[k]] = order[k+1]``), so the non-identity fraction is exactly 1.0:
    a fixed point would silently be an A1 window, the same rule ``time_shuffle_perm`` enforces for A1x."""
    if b < 2:
        raise ValueError(f"A1a permutes actions ACROSS windows: a batch of {b} has no other window to take them from")
    order = torch.randperm(b, generator=generator)
    perm = torch.empty(b, dtype=torch.long)
    perm[order] = torch.roll(order, -1)
    return perm


class RA1Model(nn.Module):
    def __init__(self, cfg: RA1Config, act_mean: Tensor, act_std: Tensor):
        super().__init__()
        validate_config(cfg)
        self.cfg = cfg
        sp = cfg.spec
        self.act_channels = ACT_CHANNELS_BY_MODE[cfg.kin_mode]
        self.shuffle_act = bool(sp.shuffle_act)      # A1a; forward_batch reads THIS built flag, and G-DVB reads it back
        self.adapter = PooledAdapter(cfg.d_feat, cfg.n_tok, cfg.d_tok, cfg.adapter_hidden, cfg.d1)
        self.register_buffer("act_mean", torch.as_tensor(act_mean, dtype=torch.float32).clone())
        self.register_buffer("act_std", torch.as_tensor(act_std, dtype=torch.float32).clone())
        self.stack: HJEPAStack | None = None
        if sp.wm:
            w = cfg.level_weights()
            levels = [LevelConfig(latent_dim=cfg.d1, action_dim=ACT_DIM, act_embed_dim=None, enc_hidden=cfg.enc_hidden,
                                  pred_width=cfg.pred_width, pred_depth=cfg.pred_depth, pred_heads=cfg.pred_heads,
                                  context=cfg.context, beta=w[0][0], gamma=w[0][1])]
            for l in range(2, sp.n_levels + 1):
                levels.append(LevelConfig(latent_dim=cfg.d_up, action_dim=cfg.macro_dim, stride=cfg.stride, window=1,
                                          enc_hidden=cfg.enc_hidden, macro_hidden=cfg.macro_hidden, macro_mode="mlp",
                                          pred_width=cfg.pred_width, pred_depth=cfg.pred_depth, pred_heads=cfg.pred_heads,
                                          context=cfg.context, beta=w[l - 1][0], gamma=w[l - 1][1],
                                          act_sigreg=cfg.act_sigreg))
            self.stack = HJEPAStack(levels, sigreg_projections=cfg.sigreg_proj)
            if sp.hist_drop:
                for i in range(len(self.stack.predictors)):
                    self.stack.predictors[i] = HistoryDropPredictor(self.stack.predictors[i], cfg.hist_drop)
        d_lat = cfg.d1 + cfg.n_up() * cfg.d_up
        self.readout = Readout(d_lat, 8, cfg.readout_hidden)

    # ---- latents -------------------------------------------------------------------------------------
    def levels_of(self, z1: Tensor) -> list:
        """``[z1, z2, ...]`` for EVERY frame (the level encoders are pointwise, w = 1)."""
        zs = [z1]
        if self.stack is not None:
            for enc in self.stack.encoders:
                zs.append(enc(zs[-1]))
        return zs

    def std_actions(self, a_raw: Tensor) -> Tensor:
        return (a_raw - self.act_mean) / self.act_std

    def predict_traj(self, z1: Tensor, f8n: Tensor, p10_traj: Tensor) -> Tensor:
        return p10_traj + self.readout(torch.cat(self.levels_of(z1), dim=-1), f8n)

    def set_generators(self, hist_gen: torch.Generator | None) -> None:
        if self.stack is not None:
            for p in self.stack.predictors:
                if isinstance(p, HistoryDropPredictor):
                    p.generator = hist_gen

    # ---- the loss ------------------------------------------------------------------------------------
    def forward_batch(self, batch: dict, *, generator: torch.Generator | None = None,
                      shuffle_generator: torch.Generator | None = None) -> dict:
        """``batch``: tokens ``(B,T1,n_tok,d_feat)`` fp8/float, act ``(B,T1,3)`` RAW, f8n ``(B,T1,8)``, p10_traj / gt
        ``(B,T1,30,2)``, valid ``(B,T1)`` bool. Returns the dict of terms; ``total`` is the differentiable objective."""
        z1 = self.adapter(batch["tokens"].float())                                   # (B,T1,D1)
        traj = self.predict_traj(z1, batch["f8n"], batch["p10_traj"])
        err = torch.sqrt((traj - batch["gt"]).pow(2).sum(-1) + 1e-12).mean(-1)       # (B,T1) ADE per frame, metres
        m = batch["valid"].to(err.dtype)
        traj_loss = (err * m).sum() / m.sum().clamp_min(1.0)
        out: dict = {"traj": traj_loss, "ade_m": traj_loss.detach()}
        total = self.cfg.w_traj * traj_loss
        if self.stack is not None:
            a_seq = self.std_actions(batch["act"][:, :-1])
            if self.shuffle_act:                                       # A1a: another window's actions, frames untouched
                aperm = window_derangement(a_seq.shape[0], shuffle_generator).to(a_seq.device)
                own = a_seq
                a_seq = a_seq[aperm]
                out["act_shuffle_nonidentity_frac"] = float((aperm != torch.arange(a_seq.shape[0], device=a_seq.device)).float().mean())
                out["act_shuffle_changed_frac"] = float((a_seq != own).flatten(1).any(1).float().mean())
            zin = z1
            if self.cfg.spec.shuffle_time:
                perm = time_shuffle_perm(z1.shape[0], z1.shape[1], shuffle_generator).to(z1.device)
                zin = torch.gather(z1, 1, perm[..., None].expand(-1, -1, z1.shape[-1]))
                out["shuffle_nonidentity_frac"] = float((perm != torch.arange(z1.shape[1], device=z1.device)[None]).float().mean())
            wm = self.stack.loss(zin, a_seq, generator=generator)
            for k, v in wm.items():
                if k != "total":
                    out[k] = v
            for lv, lc in enumerate(self.stack.cfg, start=1):          # the weights the BUILT stack applies, not the config's
                out[f"idm_term/L{lv}"] = lc.gamma * wm[f"idm/L{lv}"]       # EXACTLY 0.0 when gamma_l == 0.0
                out[f"sigreg_term/L{lv}"] = lc.beta * wm[f"sigreg/L{lv}"]
            out["wm_total"] = wm["total"]
            total = total + wm["total"]
        out["total"] = total
        return out


def build_model(cfg: RA1Config, act_mean, act_std) -> RA1Model:
    return RA1Model(cfg, torch.as_tensor(act_mean), torch.as_tensor(act_std))


def optimizer_groups(model: RA1Model, lr: float, upper_lr_mult: float, weight_decay: float) -> list:
    """Adapter + readout at ``lr``; level 1 of the stack at ``lr``; every upper level at ``lr * upper_lr_mult`` (paper E4).
    Every parameter appears in EXACTLY one group (asserted)."""
    groups = [dict(params=list(model.adapter.parameters()), lr=lr, weight_decay=weight_decay, name="adapter"),
              dict(params=list(model.readout.parameters()), lr=lr, weight_decay=weight_decay, name="readout")]
    if model.stack is not None:
        groups += model.stack.param_groups(lr, upper_lr_mult, weight_decay)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)), "a parameter is in two optimiser groups"
    assert len(ids) == sum(1 for _ in model.parameters()), "a parameter is in NO optimiser group"
    return groups


# ============================================================================================ launch gate (G-DVB / G-LIVE)
def declared_vs_built(cfg: RA1Config, model: RA1Model) -> list:
    """G-DVB: every declared lever read BACK from the built modules (never from the config that built them)."""
    rows = []

    def add(name, declared, built):
        ok = (abs(declared - built) < 1e-12) if isinstance(declared, float) and isinstance(built, (int, float)) else declared == built
        rows.append(dict(lever=name, declared=declared, built=built, ok=bool(ok)))

    add("act_channels", ACT_CHANNELS_BY_MODE[cfg.kin_mode], model.act_channels)
    add("act_dim", ACT_DIM, int(model.act_mean.numel()))
    add("d1", cfg.d1, model.adapter.mlp[-1].out_features)
    add("n_tok*d_tok", cfg.n_tok * cfg.d_tok, model.adapter.mlp[1].in_features)
    add("w_traj", float(cfg.w_traj), float(model.cfg.w_traj))
    add("n_wm_levels", cfg.spec.n_levels, 0 if model.stack is None else model.stack.n_levels)
    if model.stack is not None:
        bl = model.stack.built_levers()
        for lv, (b_l, g_l) in enumerate(cfg.level_weights(), start=1):
            add(f"beta_L{lv}", b_l, bl[lv]["beta"])
            add(f"gamma_L{lv}", g_l, bl[lv]["gamma"])
            add(f"idm_live_L{lv}", g_l > 0.0, bl[lv]["idm_live"])
            add(f"context_L{lv}", cfg.context, bl[lv]["context"])
            if lv >= 2:
                add(f"stride_L{lv}", cfg.stride, bl[lv]["stride"])
                add(f"act_sigreg_L{lv}", float(cfg.act_sigreg), bl[lv]["act_sigreg"])
        wrappers = [p for p in model.stack.predictors if isinstance(p, HistoryDropPredictor)]
        add("hist_drop_wrapped_levels", model.stack.n_levels if cfg.spec.hist_drop else 0, len(wrappers))
        if cfg.spec.hist_drop:
            add("hist_drop_p", float(cfg.hist_drop), float(wrappers[0].p))
    add("readout_in", cfg.d1 + cfg.n_up() * cfg.d_up, model.readout.net[0].in_features - 8)
    add("shuffle_act", bool(cfg.spec.shuffle_act), bool(getattr(model, "shuffle_act", False)))   # A1a (Addendum D)
    return rows


def expected_dead(cfg: RA1Config) -> set:
    """Parameters that are legitimately gradient-free by construction (prefixes). A0's IDM heads exist but gamma = 0."""
    return {"stack.idm_heads."} if cfg.arm == "A0" else set()


def live_report(model: RA1Model, batch: dict, *, seed: int = 0) -> dict:
    """G-LIVE on a REAL batch: every declared loss term present and finite; the IDM term's gradient into the ADAPTER is
    non-zero in every IDM arm and EXACTLY 0.0 in A0 (the zero control); no unexpected dead parameter."""
    cfg = model.cfg
    dev = next(model.parameters()).device
    g = torch.Generator(device=dev).manual_seed(seed)
    model.train()
    model.zero_grad(set_to_none=True)
    if cfg.spec.hist_drop:
        model.set_generators(torch.Generator().manual_seed(seed))
    out = model.forward_batch(batch, generator=g, shuffle_generator=torch.Generator().manual_seed(seed))
    expect = ["traj", "total"]
    if model.stack is not None:
        for lv in range(1, model.stack.n_levels + 1):
            expect += [f"pred/L{lv}", f"sigreg/L{lv}", f"idm/L{lv}", f"idm_term/L{lv}", f"sigreg_term/L{lv}"]
            if lv >= 2:
                expect.append(f"act_sigreg/L{lv}")
    missing = [k for k in expect if k not in out]
    nonfinite = [k for k in expect if k in out and not bool(torch.isfinite(torch.as_tensor(out[k])).all())]
    idm_grad_adapter = None
    if model.stack is not None:
        ad = [p for p in model.adapter.parameters()]
        tot = 0.0
        for lv in range(1, model.stack.n_levels + 1):
            term = out[f"idm_term/L{lv}"]
            if not term.requires_grad:
                continue
            gr = torch.autograd.grad(term, ad, retain_graph=True, allow_unused=True)
            tot += sum(float(x.abs().sum()) for x in gr if x is not None)
        idm_grad_adapter = tot
    out["total"].backward()
    rep = param_grad_report(model)
    dead = dead_parameters(model, rep)
    allowed = expected_dead(cfg)
    unexpected = [n for n in dead if not any(n.startswith(p) for p in allowed)]
    idm_ok = True
    if model.stack is not None:
        idm_val = [float(out[f"idm_term/L{lv}"].detach()) for lv in range(1, model.stack.n_levels + 1)]
        idm_ok = (idm_grad_adapter > 0.0 and all(v > 0.0 for v in idm_val)) if cfg.spec.idm else \
            (idm_grad_adapter == 0.0 and all(v == 0.0 for v in idm_val))
    shuffle_frac = float(out["shuffle_nonidentity_frac"]) if "shuffle_nonidentity_frac" in out else None
    shuffle_ok = (shuffle_frac is not None and shuffle_frac > 0.5) if cfg.spec.shuffle_time else (shuffle_frac is None)
    # A1a: the permutation is non-identity in EVERY window (a derangement) AND the actions the WM loss receives really are not the
    # window's own in more than half of the windows (a computed-but-unapplied permutation reads 0.0 here); every other arm: none
    act_frac = float(out["act_shuffle_nonidentity_frac"]) if "act_shuffle_nonidentity_frac" in out else None
    act_changed = float(out["act_shuffle_changed_frac"]) if "act_shuffle_changed_frac" in out else None
    act_ok = (act_frac == 1.0 and act_changed is not None and act_changed > 0.5) if cfg.spec.shuffle_act else \
        (act_frac is None and act_changed is None)
    ok = not missing and not nonfinite and not unexpected and idm_ok and shuffle_ok and act_ok
    return dict(ok=bool(ok), shuffle_nonidentity_frac=shuffle_frac, shuffle_ok=bool(shuffle_ok),
                act_shuffle_nonidentity_frac=act_frac, act_shuffle_changed_frac=act_changed, act_shuffle_ok=bool(act_ok),
                missing=missing, nonfinite=nonfinite, unexpected_dead=unexpected,
                expected_dead=sorted(n for n in dead if any(n.startswith(p) for p in allowed)),
                idm_term_grad_into_adapter=idm_grad_adapter, idm_ok=bool(idm_ok),
                terms={k: float(v.detach()) for k, v in out.items() if isinstance(v, Tensor) and v.numel() == 1})


def estimate_vram_gb(model: RA1Model, batch: dict, *, slack: float = 1.25) -> dict:
    """ESTIMATE (not a measurement): params x16 B (weights + grads + 2 Adam moments) + activation bytes counted by forward
    hooks on leaf modules (outputs x 4 B x 2 for the backward) + the fp32 token copy. On CUDA the trainer also logs
    ``max_memory_allocated`` -- THAT is the measurement."""
    acts = [0]
    hooks = []
    for m in model.modules():
        if not list(m.children()):
            hooks.append(m.register_forward_hook(lambda _m, _i, o: acts.__setitem__(0, acts[0] + (o.numel() if isinstance(o, Tensor) else 0))))
    model.train()
    with torch.no_grad():
        model.forward_batch(batch, generator=torch.Generator(device=next(model.parameters()).device).manual_seed(0),
                            shuffle_generator=torch.Generator().manual_seed(0))
    for h in hooks:
        h.remove()
    n_par = sum(p.numel() for p in model.parameters())
    tok = batch["tokens"]
    tok_bytes = tok.numel() * 4 * 3          # fp32 copy + LayerNorm output + token projection input grad
    est = (n_par * 16 + acts[0] * 4 * 2 + tok_bytes) * slack
    return dict(params=n_par, params_gb=n_par * 16 / 2 ** 30, activations_gb=acts[0] * 8 / 2 ** 30,
                token_buffers_gb=tok_bytes / 2 ** 30, slack=slack, estimate_gb=est / 2 ** 30, kind="ESTIMATE")


# ============================================================================================ eval diagnostics
@torch.no_grad()
def sensitivity_per_window(model: RA1Model, z_start: Tensor, fut_actions_raw: Tensor, eid: Tensor, *,
                           ks=(1, 5, 10), seed: int = 0, perm: Tensor | None = None) -> dict:
    """Level-1 action sensitivity (PREREG definition, delegated to ``HJEPAStack.action_sensitivity``) for c in
    {all, a, <lateral channel>, v} where the lateral channel is ``steer`` (legacy) or ``yaw_rate`` (the compliant mode; PI ruling
    D-PI-YAWRATE-1); ``perm = arange`` is the SELF-SWAP CONTROL and must read exactly 0.0. T0: recorded future actions are
    consumed by the rollout. Returns ``hjepa``'s dict (floats + per-window arrays under ``win/...``)."""
    assert model.stack is not None, "R0 has no world model"
    chans = {"all": [0, 1, 2], "a": [0], model.act_channels[1]: [1], "v": [2]}
    return model.stack.action_sensitivity(z_start, model.std_actions(fut_actions_raw), level=1, ks=ks, channels=chans,
                                          perm=perm, episode_ids=None if perm is not None else eid, seed=seed,
                                          per_window=True)


def cost_windows(zs_by_level: list, t_list, strides=(1, 3, 9), horizon: int = HORIZON) -> dict:
    """GOAL-COST MONOTONICITY (H-HJ-COST-1, T0: the +6 s state is the TRUE future frame's latent). For one clip's per-frame
    latents ``zs_by_level[l-1]`` ``(T, D_l)`` and window origins ``t``: the squared distance of the level-``l`` state at frames
    ``t, t+s, ..., < t+horizon`` (the GOAL frame t+30 itself is excluded: its distance is 0 by construction) to the
    level-``l`` state at ``t+30``; value = ``hjepa.monotonicity`` (= -Spearman(cost, time), 1.0 = perfectly decreasing).
    ``native`` = level l on its own clock (L1 stride 1, L2 stride 3, L3 stride 9); ``matched`` = level 1 on level 2's stride 3."""
    t = np.asarray(t_list, np.int64)
    out: dict = {}

    def mono(z: Tensor, stride: int) -> Tensor:
        idx = torch.as_tensor(t[:, None] + np.arange(0, horizon, stride)[None])
        goal = z[torch.as_tensor(t + horizon)]
        d = (z[idx] - goal[:, None]).pow(2).sum(-1)
        return monotonicity(d)

    out["L1_native"] = mono(zs_by_level[0], 1)
    out["L1_matched_s3"] = mono(zs_by_level[0], 3)
    if len(zs_by_level) >= 2:
        out["L2_native"] = mono(zs_by_level[1], 3)
    if len(zs_by_level) >= 3:
        out["L3_native"] = mono(zs_by_level[2], 9)
    return {k: v.numpy() for k, v in out.items()}
