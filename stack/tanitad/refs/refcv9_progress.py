"""refcv9 R9-PF -- the clean-split, regime-factored PROGRESS head and the in-forward re-timing layer.

Design memo: `REFCV9_INMODEL_DESIGN.md` (Architecture FlyWheel, 2026-10-26). What this moves into the model:

* SHIP-R8 A4 (H2cn): a progress head fitted ONLY on clips the trunk never saw beats the shipped re-timing stage
  (E3 -1.198 vs -0.868; false decel 0.137 -> 0.060 separated) -- MEASURED post-hoc on refcv8.
* SHIP-R8 A5 (FACTORED): the lon DECISION applied to the SPEED only, the PATH from the planner (missed decel -0.08
  separated) -- MEASURED post-hoc on refcv8.

The head (``R9ProgressHead``) reads ``z_tac.detach()`` plus nine IMU-measurable t0 scalars (H2cn's exact set,
``progress_scalar.feats``) and emits a 3-way longitudinal decision (decel / cruise / accel -- the MD/FD metric's own
GT definition) and ONE arc-length profile per decision. Inference takes the decision's profile and re-times the
SELECTED path to it (every point stays on the selected polyline or its end ray: lateral is untouched by construction),
BEFORE ``plan_vmax_clamp``.

⛔ The head never trains the trunk (every input is detached) and is fitted only in the calibration stage
(``--r9-calib-stage``) on an external clean split; ``--r9-progress-head`` without it is refused in training.
⛔ Curvature at t0 is NOT an input (PI 10-18); nothing here reads ``ego_state[:, 3]``.
⛔ The head's output feeds the re-timing only -- never the goal / nav input, selection, or the cap's limit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tanitad.train.config_hygiene import strict_fields

#: decision classes, in this order everywhere (``l2_common.CLS``)
REGIMES: tuple[str, ...] = ("decel", "cruise", "accel")
N_REGIMES = len(REGIMES)
#: the MD/FD metric's GT threshold on dv = v(5-6 s chord) - v0  (``l2_common``: +-1.5 m/s, dt 1.0 s)
DV_THRESHOLD_MS = 1.5
#: the scalar set (``progress_scalar.feats`` without the leading 1): H2cn's MEASURED inputs
SCALAR_NAMES: tuple[str, ...] = ("v0", "a0", "omega0", "v0_omega0", "v0_sq", "a0_v0", "ceiling", "ceiling_minus_v0",
                                 "ceiling_valid")
N_SCALARS = len(SCALAR_NAMES)
CEILING_CAP_MS = 40.0
#: LON6 group -> regime (``l2_common``: decel = BRAKE_TO, CREEP, HOLD; cruise = FOLLOW, CRUISE; accel = ACCELERATE)
LON6_TO_REGIME: tuple[int, ...] = (1, 1, 0, 0, 0, 2)
DECISIONS: tuple[str, ...] = ("regime", "trunk_lon")
R9_INIT_SEED_OFFSET = 9_091
KEY_PREFIX = "r9_prog"
#: the root-model freeze record G-DVB reads BY NAME (``declared_vs_built._R9_CALIB_ATTR``)
CALIB_MARKER_ATTR = "_tanitad_r9_calib"


@strict_fields
@dataclass
class R9Config:
    """``enable=False`` (the default) builds NOTHING: a refcv8 config is unchanged in everything it builds."""
    enable: bool = False
    decision: str = "regime"
    hidden: int = 128
    decel_tau: float | None = None
    w_prog: float = 1.0
    w_regime: float = 1.0
    calib_stage: bool = False
    seed: int = 20261026

    def to_dict(self) -> dict:
        return asdict(self)


def is_refcv9_key(key: str) -> bool:
    return any(part == KEY_PREFIX or part.startswith(KEY_PREFIX + ".") for part in key.split("."))


# ------------------------------------------------------------------------------------------------ geometry (torch)
def polyline_geometry(O: Tensor) -> dict:
    """Torch port of ``tanitad.retime_stage.polyline_geometry``: vertices V_0 = (0, 0), V_k = O[:, k-1]; cumulative arc
    length L [B, S+1]; unit directions u [B, S, 2] (0 on a zero-length segment); u_end = direction of the LAST
    positive-length segment, +x if the whole polyline has zero length."""
    if O.dim() != 3 or O.shape[-1] != 2:
        raise ValueError(f"[refcv9] plans must be [B, S, 2], got {tuple(O.shape)}")
    if not bool(torch.isfinite(O).all()):
        raise ValueError("[refcv9] non-finite plan position: the re-timing is undefined")
    b, s, _ = O.shape
    V = torch.cat([O.new_zeros(b, 1, 2), O], dim=1)
    seg = V[:, 1:] - V[:, :-1]
    ln = torch.hypot(seg[..., 0], seg[..., 1])
    L = torch.cat([O.new_zeros(b, 1), torch.cumsum(ln, dim=1)], dim=1)
    pos = ln > 0
    u = torch.where(pos[..., None], seg / torch.where(pos, ln, torch.ones_like(ln))[..., None], torch.zeros_like(seg))
    anyp = pos.any(dim=1)
    last = torch.where(anyp, s - 1 - torch.argmax(torch.flip(pos, dims=[1]).to(torch.int32), dim=1),
                       torch.zeros(b, dtype=torch.long, device=O.device))
    u_end = u[torch.arange(b, device=O.device), last].clone()
    u_end[~anyp] = torch.tensor([1.0, 0.0], dtype=O.dtype, device=O.device)
    return {"V": V, "L": L, "u": u, "u_end": u_end, "fallback": ~anyp}


def eval_polyline(geo: Mapping[str, Tensor], s: Tensor) -> Tensor:
    """C(s) at arc length ``s`` [B, m] (s >= 0): linear along the polyline, beyond L_S along the end ray from V_S
    (``retime_stage.eval_polyline``). Differentiable in ``s``; the geometry is whatever the caller passed."""
    V, L, u, u_end = geo["V"], geo["L"], geo["u"], geo["u_end"]
    n_seg = u.shape[1]
    idx = torch.zeros(s.shape, dtype=torch.long, device=s.device)
    for j in range(1, n_seg):
        idx = idx + (s >= L[:, j:j + 1]).to(torch.long)
    rows = torch.arange(s.shape[0], device=s.device)[:, None]
    P = V[rows, idx] + (s - L[rows, idx])[..., None] * u[rows, idx]
    beyond = s >= L[:, n_seg:n_seg + 1]
    Pe = V[:, n_seg:n_seg + 1] + (s - L[:, n_seg:n_seg + 1])[..., None] * u_end[:, None, :]
    return torch.where(beyond[..., None], Pe, P)


def arc_at_slots(O: Tensor) -> Tensor:
    """Cumulative chord arc length through (0, 0) and the S slot points [B, S] (``retime_stage.prior_arc``)."""
    return polyline_geometry(O)["L"][:, 1:]


def monotone_progress(s: Tensor) -> Tensor:
    """Clip at 0, then a running max over slots: a progress profile never runs backwards."""
    return torch.cummax(s.clamp_min(0.0), dim=1).values


def retime(path: Tensor, s_target: Tensor) -> Tensor:
    """Re-time the SELECTED path to ``s_target`` [B, S]. The path's vertices are DETACHED (no gradient reaches
    selection); every returned point lies on the path's polyline or its end ray."""
    geo = polyline_geometry(path.detach())
    return eval_polyline(geo, s_target.to(path.dtype))


# ------------------------------------------------------------------------------------------------ admissible inputs
def ego_t0(poses: Tensor, n_past: int, dt: float) -> tuple[Tensor, Tensor, Tensor]:
    """(v0, a0, omega0) [B] at the LAST OBSERVED frame -- H2cn's feature builder (``progress_scalar.feats``):
    v0 / omega0 from ``ego_channels_from_poses``, a0 from ``prior_controls("ha0_ext_pose")``. IMU-measurable;
    reads ``poses[:, :n_past]`` only."""
    from tanitad.models import kinematic_prior as KP
    from tanitad.models.ego_history import ego_channels_from_poses
    pw = poses[:, :int(n_past)].to(torch.float32)
    ch = ego_channels_from_poses(pw, int(n_past), dt=float(dt))
    a0, _k0 = KP.prior_controls("ha0_ext_pose", pw, int(n_past), None, dt=float(dt))
    if a0.dim() == 2:
        a0 = a0[:, 0]
    return ch[:, -1, 0], a0.reshape(-1).to(torch.float32), ch[:, -1, 2]


def admissible_scalars(v0: Tensor, a0: Tensor, om: Tensor, v_lim: Tensor | None, v_lim_valid: Tensor | None) -> Tensor:
    """[B, 9] in ``SCALAR_NAMES`` order. The ceiling is the FED speed-limit label (an admissible input); absent or
    invalid -> 40 m/s and valid 0, exactly as ``progress_scalar.feats``."""
    v0, a0, om = (t.reshape(-1).to(torch.float32) for t in (v0, a0, om))
    if v_lim is None:
        valid = torch.zeros_like(v0, dtype=torch.bool)
        vl = torch.full_like(v0, CEILING_CAP_MS)
    else:
        lim = v_lim.reshape(-1).to(torch.float32)
        valid = torch.isfinite(lim)
        if v_lim_valid is not None:
            valid = valid & (v_lim_valid.reshape(-1).to(torch.float32) > 0.5)
        vl = torch.where(valid, lim.clamp_max(CEILING_CAP_MS), torch.full_like(lim, CEILING_CAP_MS))
    return torch.stack([v0, a0, om, v0 * om, v0 ** 2, a0 * v0, vl, vl - v0, valid.to(torch.float32)], dim=1)


# ------------------------------------------------------------------------------------------------ the head
class R9ProgressHead(nn.Module):
    """[z_tac.detach() ; 9 scalars] -> per-feature standardisation (running statistics, set in the calibration stage)
    -> Linear(d + 9, hidden) -> GELU -> {regime logits [B, 3], profiles [B, 3, S]} (arc-length residual over the
    prior, metres). H2cn's body (one GELU hidden layer of 128), with a decision head and one profile per decision."""

    def __init__(self, d_lat: int, n_slots: int, hidden: int = 128):
        super().__init__()
        self.d_lat, self.n_slots = int(d_lat), int(n_slots)
        d_in = self.d_lat + N_SCALARS
        self.norm = nn.BatchNorm1d(d_in, affine=False, momentum=None)
        self.body = nn.Sequential(nn.Linear(d_in, int(hidden)), nn.GELU())
        self.regime = nn.Linear(int(hidden), N_REGIMES)
        self.profile = nn.Linear(int(hidden), N_REGIMES * self.n_slots)
        with torch.no_grad():                    # start at the PRIOR's progress (dS = 0) and a flat decision
            self.profile.weight.mul_(0.01)
            self.profile.bias.zero_()
            self.regime.bias.zero_()

    def forward(self, z: Tensor, x: Tensor) -> dict:
        if z.shape[-1] != self.d_lat or x.shape[-1] != N_SCALARS:
            raise ValueError(f"[refcv9] head expects z [B, {self.d_lat}] and x [B, {N_SCALARS}], got "
                             f"{tuple(z.shape)} / {tuple(x.shape)}")
        h = self.body(self.norm(torch.cat([x.to(z.dtype), z], dim=-1)))
        return {"regime_logits": self.regime(h),
                "profiles": self.profile(h).reshape(-1, N_REGIMES, self.n_slots)}


def count_params(head: nn.Module) -> int:
    return int(sum(p.numel() for p in head.parameters()))


def trunk_lon_regime(lon_logits_v7: Tensor) -> Tensor:
    """The trunk lon head's OWN argmax over the six LON6 v7 columns, mapped to a regime (the SELFSHARP decision)."""
    from tanitad.refs import refcv8_conditioning as r8c
    cols = torch.as_tensor(r8c.LON6_V7_IDS, device=lon_logits_v7.device)
    a6 = lon_logits_v7.detach().index_select(-1, cols).argmax(dim=-1)
    return torch.as_tensor(LON6_TO_REGIME, device=a6.device)[a6]


def decide(regime_logits: Tensor, decel_tau: float | None = None) -> Tensor:
    """The HARD decision (the in-model 'self-sharpened' lon): argmax, or decel iff p_decel >= tau (tau fixed on the
    calibration split, never on eval)."""
    if decel_tau is None:
        return regime_logits.argmax(dim=-1)
    p = regime_logits.softmax(dim=-1)
    rest = 1 + p[:, 1:].argmax(dim=-1)
    return torch.where(p[:, 0] >= float(decel_tau), torch.zeros_like(rest), rest)


def select_profile(profiles: Tensor, r: Tensor) -> Tensor:
    return profiles[torch.arange(profiles.shape[0], device=profiles.device), r.to(torch.long)]


def head_forward(head: R9ProgressHead, cfg: R9Config, z_tac: Tensor, x: Tensor, s_prior: Tensor,
                 lon_logits_v7: Tensor | None) -> dict:
    """Everything the forward emits. ``z_tac`` is DETACHED here -- the stop-gradient is part of the function."""
    ho = head(z_tac.detach(), x.detach())
    if cfg.decision == "regime":
        r_hat = decide(ho["regime_logits"].detach(), cfg.decel_tau)
    elif cfg.decision == "trunk_lon":
        if lon_logits_v7 is None:
            raise ValueError("[refcv9] --r9-decision trunk_lon but the forward exported no lon logits")
        r_hat = trunk_lon_regime(lon_logits_v7)
    else:
        raise ValueError(f"[refcv9] unknown decision {cfg.decision!r} (one of {DECISIONS})")
    d_s = select_profile(ho["profiles"], r_hat)
    s_t = monotone_progress(s_prior.detach() + d_s)
    return {"r9_regime_logits": ho["regime_logits"], "r9_profiles": ho["profiles"], "r9_decision": r_hat,
            "r9_s_prior": s_prior.detach(), "r9_s_target": s_t, "r9_x": x.detach()}


# ------------------------------------------------------------------------------------------------ targets + loss
def gt_regime(traj_tgt: Tensor, slot_valid: Tensor, v0: Tensor, horizons) -> tuple[Tensor, Tensor]:
    """GT decision: dv = |p(6.0 s) - p(5.0 s)| / 1.0 s - v0, decel <= -1.5, accel >= +1.5 (``l2_common``).
    ``horizons`` are the plan's tick indices; slots 50 and 60 must exist. Returns (r* [B], ok [B])."""
    h = [int(t) for t in horizons]
    if 50 not in h or 60 not in h:
        raise ValueError(f"[refcv9] the GT decision needs the 5.0 s and 6.0 s slots; horizons are {h}")
    i5, i6 = h.index(50), h.index(60)
    v_end = (traj_tgt[:, i6] - traj_tgt[:, i5]).norm(dim=-1) / 1.0
    dv = v_end - v0.reshape(-1).to(v_end.dtype)
    r = torch.where(dv <= -DV_THRESHOLD_MS, torch.zeros_like(dv, dtype=torch.long),
                    torch.where(dv >= DV_THRESHOLD_MS, torch.full_like(dv, 2, dtype=torch.long),
                                torch.ones_like(dv, dtype=torch.long)))
    ok = slot_valid[:, i5].to(torch.bool) & slot_valid[:, i6].to(torch.bool)
    return r, ok


def r9_losses(cfg: R9Config, out: Mapping[str, Any], traj_tgt: Tensor, slot_valid: Tensor, v0: Tensor, horizons,
              calib_mask: Tensor | None = None) -> tuple[Tensor, dict]:
    """L = w_prog * masked-mean |s_P + dS[r*] - s_H|  +  w_regime * CE(regime, r*), on CALIBRATION rows only.
    The profile is TEACHER-FORCED on the GT decision r*. Neither weight guards the term (a zero weight multiplies)."""
    prof, logits, s_p = out["r9_profiles"], out["r9_regime_logits"], out["r9_s_prior"]
    s_h = arc_at_slots(traj_tgt.detach().to(prof.dtype))
    r_star, ok = gt_regime(traj_tgt.detach(), slot_valid, v0, horizons)
    cal = (torch.ones_like(ok) if calib_mask is None else calib_mask.to(torch.bool).reshape(-1))
    sv = slot_valid.to(torch.bool) & cal[:, None]
    d_s = select_profile(prof, r_star)
    err = (s_p + d_s - s_h).abs()
    w = sv.to(err.dtype)
    l_prog = (err * w).sum() / w.sum().clamp_min(1.0)
    m = ok & cal
    if bool(m.any()):
        l_reg = F.cross_entropy(logits[m], r_star[m])
    else:
        l_reg = logits.sum() * 0.0
    total = float(cfg.w_prog) * l_prog + float(cfg.w_regime) * l_reg
    with torch.no_grad():
        acc = (logits[m].argmax(-1) == r_star[m]).float().mean() if bool(m.any()) else logits.new_zeros(())
    tele = {"r9_l_prog": l_prog.detach(), "r9_l_regime": l_reg.detach(), "r9_regime_acc": acc,
            "r9_calib_rows": float(cal.sum()), "r9_decel_rows": float(((r_star == 0) & m).sum())}
    return total, tele


# ------------------------------------------------------------------------------------------------ trainer helpers
def is_on(args) -> bool:
    return bool(getattr(args, "r9_progress_head", False))


def config_from_args(args) -> R9Config:
    return R9Config(enable=True, decision=str(args.r9_decision), hidden=int(args.r9_hidden),
                    decel_tau=(None if args.r9_decel_tau is None else float(args.r9_decel_tau)),
                    w_prog=float(args.w_r9_prog), w_regime=float(args.w_r9_regime),
                    calib_stage=bool(args.r9_calib_stage), seed=int(args.r9_seed))


def pin_args(args) -> None:
    """The refusals. ⛔ A built head with no clean rows in training is the dead-seam class (J3, the joint external
    stream, is NOT built), so --r9-progress-head trains ONLY under --r9-calib-stage."""
    on, cal = is_on(args), bool(getattr(args, "r9_calib_stage", False))
    if cal and not on:
        raise SystemExit("[refcv9] --r9-calib-stage without --r9-progress-head: there is nothing to calibrate")
    if not on:
        return
    if not cal:
        raise SystemExit("[refcv9] --r9-progress-head trains ONLY in the calibration stage (--r9-calib-stage, on a "
                         "split the trunk never saw): fitted on the trunk's own clips it learns the memorised relation "
                         "(L1 O2). The joint external-stream form (J3) is not built.")
    if str(getattr(args, "r9_decision", "regime")) not in DECISIONS:
        raise SystemExit(f"[refcv9] --r9-decision must be one of {DECISIONS}")
    tau = getattr(args, "r9_decel_tau", None)
    if tau is not None and not (0.0 < float(tau) < 1.0):
        raise SystemExit(f"[refcv9] --r9-decel-tau {tau}: must lie in (0, 1)")
    if float(getattr(args, "w_r9_prog", 0.0) or 0.0) <= 0.0:
        raise SystemExit("[refcv9] --r9-progress-head with --w-r9-prog 0: the profiles would never train (design: 1.0)")
    if str(getattr(args, "r9_decision", "regime")) == "regime" and float(getattr(args, "w_r9_regime", 0.0) or 0.0) <= 0.0:
        raise SystemExit("[refcv9] --r9-decision regime with --w-r9-regime 0: the decision would never train "
                         "(design: 1.0)")


def apply_calib_freeze(model) -> dict:
    """Stage 2: EVERY tensor outside the R9 head is frozen (requires_grad False). Returns the record."""
    n_tr, n_fr = 0, 0
    for name, p in model.named_parameters():
        keep = is_refcv9_key(name)
        p.requires_grad_(keep)
        n_tr += p.numel() if keep else 0
        n_fr += 0 if keep else p.numel()
    if n_tr == 0:
        raise SystemExit("[refcv9] --r9-calib-stage: the R9 head is not built (nothing would train)")
    rec = {"schema": "refcv9_calib_stage/1", "modules": [KEY_PREFIX], "n_trainable": int(n_tr), "n_frozen": int(n_fr),
           "forward_mode": "eval (inference path; the head's BatchNorm statistics in train mode)"}
    setattr(model, CALIB_MARKER_ATTR, dict(rec))      # the DECLARATION G-DVB reads: everything outside `modules` frozen
    return rec


def calib_stage_active(model) -> bool:
    c = getattr(getattr(model, "cfg", None), "refcv9", None)
    return bool(c is not None and c.enable and c.calib_stage and getattr(model, "r9_prog", None) is not None)


def frozen_grad_leak(model) -> float:
    """max |grad| over every NON-R9 parameter (None counts as 0) -- the stop-gradient's check."""
    m = 0.0
    for name, p in model.named_parameters():
        if not is_refcv9_key(name) and p.grad is not None:
            m = max(m, float(p.grad.detach().abs().max()))
    return m


def register_dvb_levers(register) -> None:
    from tanitad.train import declared_vs_built as dvb

    def _built(m, a):
        want = bool(dvb._a(a, "r9_progress_head", False))
        return dvb._eq("r9_progress_head", want, getattr(m, "r9_prog", None) is not None, "model.r9_prog is not None")

    def _field(dest, attr, conv, default):
        def chk(m, a):
            on = bool(dvb._a(a, "r9_progress_head", False))
            raw = dvb._a(a, dest, default)
            want = None if raw is None else conv(raw)
            if not on:
                return dvb._eq(dest, None if default is None else conv(default), want,
                               "argv (refcv9 off: must hold the default)")
            c = getattr(getattr(m, "cfg", None), "refcv9", None)
            got = None if c is None else getattr(c, attr, None)
            if isinstance(want, float):
                return dvb._near(dest, want, got, f"cfg.refcv9.{attr}")
            return dvb._eq(dest, want, got, f"cfg.refcv9.{attr}")
        return chk

    def _calib(m, a):
        want = bool(dvb._a(a, "r9_calib_stage", False))
        if not want:
            return dvb._eq("r9_calib_stage", False, want, "argv (calibration stage off)")
        n_open = sum(1 for n, p in m.named_parameters() if p.requires_grad and not is_refcv9_key(n))
        n_head = sum(1 for n, p in m.named_parameters() if p.requires_grad and is_refcv9_key(n))
        return (dvb._eq("r9_calib_stage", 0, n_open, "trainable tensors OUTSIDE the progress head")
                + dvb._eq("r9_calib_stage", True, n_head > 0, "the progress head is trainable"))

    register("r9_progress_head", "built", _built)
    register("r9_calib_stage", "built", _calib)
    for dest, attr, conv, dflt, kind in (("r9_decision", "decision", str, "regime", "built"),
                                         ("r9_hidden", "hidden", int, 128, "built"),
                                         ("r9_decel_tau", "decel_tau", float, None, "built"),
                                         ("r9_seed", "seed", int, 20261026, "built"),
                                         ("w_r9_prog", "w_prog", float, 0.0, "loss"),
                                         ("w_r9_regime", "w_regime", float, 0.0, "loss")):
        register(dest, kind, _field(dest, attr, conv, dflt))
