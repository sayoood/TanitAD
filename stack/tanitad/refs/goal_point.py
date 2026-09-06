"""E15 — a PREDICTED METRIC GOAL POINT, and the param-free geometry it enters through.

WHY THIS EXISTS, in one measured sentence
=========================================
refcv4b's categorical nav edge (E13) is a **PRESENCE-GATED BIAS**: inverting the token
left<->right on all 1,743 commanded windows costs ``+0.0022 m [-0.0006, +0.0052]`` ADE —
**not separated** — while REMOVING it costs ``+0.0961 m`` separated, i.e. **content 2.3 %,
presence 97.7 %** (`2026-09-06-refcv4b-navpred/RESULT.md`, `raw/flip_analysis.json`).
This module is the replacement, and the reason it is expected to behave differently is
**not optimism** — it is measured, in `2026-09-06-goal-point/raw/GP_LATERAL_ADAPTIVE.json`:

* **LATERAL axis** (longitudinal choice held at the model's own, only ``a_lat`` varied,
  leak-guarded, n = 4,257 / 140 eps): an **oracle 3-way categorical command's best
  possible deterministic decoding is "go straight" for ALL THREE route classes** (modal
  oracle lateral level 4 of 9 at 51.6 % / 88.6 % / 55.1 % support). It is
  **bit-identical to the no-information goal** and separated WORSE than the model's own
  pick — ``+0.1356 [+0.0880, +0.1946]`` m bank ADE, ``-0.7842`` turn accuracy. An
  oracle goal point recovers ``-0.0414 [-0.0508, -0.0328]`` m (**78.4 %** of the
  selection ceiling) and ``+0.1115 [+0.0713, +0.1548]`` turn accuracy, separated.
* ⚠️ **On that axis a POINT and a BEARING TIE EXACTLY** (-0.0414 / -0.0414; +0.1115 /
  +0.1115). That is an IDENTITY, not a coincidence: anchors compared at the same ARC
  all sit at ~the same range, so distance and cosine are monotonically related. An
  earlier draft of this module claimed the point beat the bearing by 38 %; that was an
  ARC-ORIGIN defect (the anchor arc was measured from slot 0, ~5 m down the road, while
  the goal arc was measured from the car) and it is retracted here.
* ⭐⭐ **LONGITUDINAL axis** — the one that owns 88.7 % of the oracle gap, with the goal
  at a fixed TIME of 4.0 s (n = 4,286 / 141 eps): a metric goal point recovers
  ``-0.0945 [-0.1164, -0.0757]`` m — **57.1 % of the longitudinal selection ceiling** —
  and ``-0.1184`` m/s of speed MAE, both separated. The SAME goal with its RANGE
  stripped (a bearing — which is exactly what the shipped S6 seam predicts, and exactly
  what a route command is) is separated **WORSE by +2.3632 m**. A range corruption
  (x0.5 / x2.0) is separated worse by +1.88 / +1.68 m.
  ⇒ **THE POINT'S ENTIRE MARGINAL VALUE OVER A BEARING IS RANGE, AND RANGE IS
  LONGITUDINAL.** That is why this module's registered goal is at a fixed TIME beyond
  the scored horizon, not at a fixed arc.
* ⛔ ``nav_cmd`` is ``("follow", "left", "right", "straight")`` — a purely LATERAL
  vocabulary. There is no categorical arm on the longitudinal axis to compare against,
  because a categorical route command cannot express one.

⛔ WHAT THE GOAL POINT IS COMPUTED FROM — the admissibility declaration
======================================================================
At **inference**: the strategic context token only (vision). At **training** the LABEL is
the ego's own future path, which is sanctioned — *"for ground truth data ... you can use
both ego and other label, for inference only vision"* (PI, 2026-08-03).

The admissibility check the ruling asks for, run explicitly:
**could this goal signal have been computed from the situation classifier's output?**
**NO.** The situation classifier is a separate model (the sitclf stream's ``head_img``,
labelled off `tanitad/data/situations.py`); it is not imported here, not in this graph,
not a batch field, and not a label source. There is no output to launder. The head also
does not read ``z_tac`` or the tactical ``lat``/``lon`` logits — the cascade runs
strategic -> tactical, never the reverse — so the goal path is disjoint from the
*tactical* classifier as well, which is stronger than the ruling requires.

⚠️ The trunk IS shared with ``route_head`` and the tactical head (all read the encoder).
Declared, and it is not a back door for the same reason ``RefCModel.goal_provenance``
gives: a shared ENCODER can only launder a signal that EXISTS in the graph, and the
classifier's does not. Attributability is bought by the ZERO-INIT projections, so the
exact ablation is "set the gate/projection to 0".

⛔ THE LEAK GUARD IS LOAD-BEARING, NOT A DETAIL
==============================================
A goal point INSIDE the scored 2 s horizon **is the answer**, not a route signal.
MEASURED in the same probe: an unguarded 20 m goal looked like the best design point;
with `tanitad.data.lan.horizon_lead_m` applied (``max(2 s GT arc, v0 * t_pred) +
min_lead_m``) the guarded mean arc is **28.23 m** and the unguarded short-arc advantage
is largely the leak. :func:`guard_arc_m` is therefore the ONLY sanctioned way to choose
the arc, and it is the programme's own guard, IMPORTED rather than re-derived.

⭐ MEASURED HEAD REQUIREMENT — a bar on the HEAD, checkable before any planner claim.
Injecting noise into the oracle goal point and re-reading the recovery:

  LATERAL sigma   0.5 m -> ADE  +44.8 % / turn +118.9 %   (both still separated)
                  1.0 m -> ADE  -80.7 % / turn  +71.7 %   (ADE now NEGATIVE)
                  2.0 m -> ADE -390.8 % / turn  +52.8 %
                  4.0 m -> ADE   -1025 % / turn  -54.7 %  (turn benefit gone)
  RANGE   sigma   0.5 m -> 54.0 %   1.0 m -> 47.2 %   2.0 m -> 17.2 % (all separated)
                  4.0 m -> -57.9 % (separated WORSE)

⇒ registered bar: **<= 1.0 m RMS lateral AND <= 2.0 m RMS range** on the 4 s goal point.
Below 0.5 m lateral the ADE contribution is positive too. Above 4 m on either axis the
lever is measured NEGATIVE, so the head's own error is a GATE, not a diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from tanitad.data.lan import (LanConfig, cumulative_arclength, horizon_lead_m,
                              resample_arclength)

#: Feature layout the model consumes — pinned by tests/test_goal_point.py.
#:   0: x / arc   along-track, normalised by the arc length (so it is ~1 by design)
#:   1: y / arc   signed lateral offset, normalised — THE channel that carries the turn
#:   2: valid     1.0 iff a real, leak-guarded goal exists for this window
GOAL_POINT_FEATS = 3
GOAL_POINT_FEAT_NAMES = ("x_norm", "y_norm", "valid")

#: ⛔⛔ THE TWO FORWARD KWARGS THAT MUST **NOT** BE PLUMBED FROM A BATCH.
#:
#: ``RefCV3Model.forward`` accepts ``gp_point`` / ``gp_valid``. They exist for the
#: PREREG §2 eval-time interventions (mirror ``y -> -y``, range ``x0.5``, straight,
#: shuffled, withheld) so the whole panel rolls in ONE process on ONE surface. They
#: are a **DIAGNOSTIC PORT, never a training or rollout input.**
#:
#: ⛔ THE RULING, so the RL / adapter side does not have to guess. A generic
#: "plumb every kwarg the forward accepts" adapter would feed these from the batch,
#: and there is exactly one thing a batch could supply them FROM: the ego's own
#: future path, i.e. the LABEL. That is a SUPPLIED route, which is optimistic by
#: construction on PhysicalAI, and it would turn the deployable arm's own
#: prediction into an oracle read at inference — the leak this whole edge is built
#: to avoid. **The deployable arm predicts its goal from vision inside the forward**
#: (``gp_head(ctx)``, active whenever ``goal_point_inject`` is on), so leaving both
#: at ``None`` is not a gap: it is the correct call, and the goal still reaches the
#: model.
#:
#: ⭐ EXPORTED so a signature-drift guard can EXCLUDE these two BY NAME AND BY
#: REASON instead of hardcoding a list that rots. A guard of the shape
#: ``set(FORWARD_KEYS) == params - {"self", "frames", "steps"}`` should read
#: ``- set(goal_point.DIAGNOSTIC_ONLY_FORWARD_KWARGS)`` as well.
#: Raised as ``D-RL-ADAPTER-GOALPOINT-DRIFT`` by the RL stream on 2026-09-06 and
#: ruled here, by the stream that owns the edge.
DIAGNOSTIC_ONLY_FORWARD_KWARGS = ("gp_point", "gp_valid")

_EPS = 1e-9


@dataclass(frozen=True)
class GoalPointConfig:
    """Shape + safety parameters. The defaults are the MEASURED design point.

    ``t_pred_s`` and ``min_lead_m`` are handed to the LAN guard unchanged; changing
    either changes what "admissible" means and must be re-registered.
    """

    #: "time" is the REGISTERED mode — the measured one. "arc" is kept because the
    #: lateral probe is defined on it and because it is the LAN-native parameterisation,
    #: but it is measured to be DEGENERATE against a bearing and must not be registered
    #: as "a goal point" without saying so.
    goal_mode: str = "time"
    t_goal_s: float = 4.0          # MUST exceed t_pred_s — asserted, not assumed
    range_norm_m: float = 40.0     # a FIXED scale: normalising by v0 would put the ego
    #                                speed into the conditioning and is not needed
    t_pred_s: float = 2.0          # the scored horizon the guard must clear
    min_lead_m: float = 5.0        # LanConfig's default margin, restated so it is visible
    arc_min_m: float = 10.0        # a standstill window still gets a finite arc
    arc_max_m: float = 80.0        # beyond this the probe measures ~zero recovery
    d_goal: int = 64               # conditioning width (== RefCV3Config.d_nav)
    lat_clip: float = 1.0          # |y/scale| clip, matching LanConfig.lat_clip
    xy_clip: float = 4.0           # |x/scale| clip (36 m/s * 4 s / 40 m = 3.6)

    def __post_init__(self) -> None:
        if self.goal_mode not in ("time", "arc"):
            raise ValueError(f"goal_mode must be 'time' or 'arc', got {self.goal_mode!r}")
        if not (0.0 < self.arc_min_m <= self.arc_max_m):
            raise ValueError(f"arc bounds must satisfy 0 < min <= max, got "
                             f"{self.arc_min_m} / {self.arc_max_m}")
        if self.t_pred_s <= 0 or self.min_lead_m < 0:
            raise ValueError("t_pred_s must be positive and min_lead_m non-negative")
        # ⛔ THE LEAK GUARD IN TIME MODE IS THIS ONE LINE. A goal at or inside the scored
        # horizon IS the answer; refusing it here means no arm can be built that leaks.
        if self.t_goal_s <= self.t_pred_s:
            raise ValueError(f"t_goal_s ({self.t_goal_s}) must be strictly beyond the "
                             f"scored horizon t_pred_s ({self.t_pred_s}) — a goal inside "
                             f"the horizon is the ANSWER, not a route signal")
        if self.d_goal <= 0 or self.lat_clip <= 0 or self.range_norm_m <= 0:
            raise ValueError("d_goal, lat_clip and range_norm_m must be positive")

    def lan_cfg(self) -> LanConfig:
        """The guard's own config, so the margin can never desync from the guard."""
        return LanConfig(min_lead_m=self.min_lead_m, lat_clip=self.lat_clip)


# ---------------------------------------------------------------------------
# the arc: where the goal is allowed to sit
# ---------------------------------------------------------------------------

def guard_arc_m(v0: float, gt_path_ego: np.ndarray | None = None,
                cfg: GoalPointConfig | None = None) -> float:
    """The SHORTEST ADMISSIBLE arc length for this window, in metres.

    ⛔ This is :func:`tanitad.data.lan.horizon_lead_m` — the programme's own leak guard —
    clamped to ``[arc_min_m, arc_max_m]``. It is IMPORTED, never re-derived: a guard
    re-implemented beside the thing it guards is the exact defect the navpred RESULT
    retracted two published numbers for (a control re-implemented beside its harness
    drifts into a regression against a correct sibling).

    ``gt_path_ego`` is the ground-truth 2 s path when it is available (label build);
    at inference only ``v0`` is, and the guard takes the MAX of the two so supplying
    less can only mask MORE.
    """
    cfg = cfg or GoalPointConfig()
    lead = horizon_lead_m(gt_path_ego=gt_path_ego, v0=float(v0),
                          t_pred_s=cfg.t_pred_s, cfg=cfg.lan_cfg())
    return float(min(max(lead, cfg.arc_min_m), cfg.arc_max_m))


# ---------------------------------------------------------------------------
# the label (TRAIN ONLY — ego future path; never read at inference)
# ---------------------------------------------------------------------------

def goal_point_label(future_xy_ego: np.ndarray, arc_m: float) -> tuple[np.ndarray, bool]:
    """Ego-frame future path -> ``(point[2], valid)`` at ``arc_m`` along it.

    ⛔ TRAIN ONLY. ``valid`` is False when the future path is SHORTER than ``arc_m`` —
    the label does not exist and must not be faked, because a faked goal is a constant
    the head would happily learn (the `os_navshuf`/`point_STRAIGHT` failure mode).
    Resampling is `lan.resample_arclength`, which refuses to extrapolate past the end.
    """
    p = np.asarray(future_xy_ego, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] < 2:
        return np.zeros(2, dtype=np.float32), False
    p = np.concatenate([np.zeros((1, 2)), p], axis=0)   # measure from the CAR
    if float(cumulative_arclength(p)[-1]) < float(arc_m):
        return np.zeros(2, dtype=np.float32), False
    pts, ok = resample_arclength(p, [float(arc_m)])
    if not bool(np.asarray(ok).reshape(-1)[0]):
        return np.zeros(2, dtype=np.float32), False
    return np.asarray(pts, dtype=np.float32).reshape(2), True


def goal_slot_index(t_goal_s: float, horizons_ticks=None, hz: float = 10.0) -> int:
    """The bank slot whose time equals ``t_goal_s``. Raises when there is none.

    ⛔ DERIVED from ``refc_v3.V3_HORIZONS``, never re-typed: a hardcoded tuple here
    would silently become a different experiment the moment the horizons move, and an
    APPROXIMATE slot would compare the goal at one time to the anchor at another.
    """
    if horizons_ticks is None:
        from tanitad.refs.refc_v3 import V3_HORIZONS as horizons_ticks
    secs = [round(float(h) / hz, 6) for h in horizons_ticks]
    key = round(float(t_goal_s), 6)
    if key not in secs:
        raise ValueError(f"t_goal_s {t_goal_s} is not a model horizon; available {secs}")
    return secs.index(key)


def goal_point_label_at_time(future_xy_ego, valid, t_goal_s: float,
                             dt: float = 0.1):
    """Ego-frame future path (10 Hz, sample k at ``(k + 1) * dt`` s) -> the goal point
    at ``t_goal_s`` and its validity. THE REGISTERED LABEL.

    ⛔ TRAIN ONLY. A window whose future ends before ``t_goal_s`` has NO label and it
    must not be faked: a faked goal is a constant, and a constant goal is MEASURED to be
    separated WORSE than the model's own selection (``point_TIME_CONSTANT`` +1.4978 m).
    """
    p = np.asarray(future_xy_ego, dtype=np.float64).reshape(-1, 2)
    v = np.asarray(valid).reshape(-1)
    k = int(round(float(t_goal_s) / float(dt))) - 1
    if k < 0 or k >= p.shape[0] or k >= v.shape[0] or not bool(v[k]):
        return np.zeros(2, dtype=np.float32), False
    return p[k].astype(np.float32), True


def encode_goal_point(point_ego, valid, scale_m,
                      cfg: GoalPointConfig | None = None) -> np.ndarray:
    """``(point, valid, scale)`` -> the flat ``[3]`` feature the model consumes.

    ``scale_m`` is ``cfg.range_norm_m`` in TIME mode — a FIXED constant, so the ego
    speed never enters the conditioning — and the guard arc in ARC mode.

    An invalid goal is EXACTLY zeros including ``valid = 0`` — so "no goal" is
    distinguishable from "goal straight ahead" (``x_norm ~ 1, y_norm = 0, valid = 1``).
    That is the X15 rule: withheld and genuinely-zero must differ in the FLAG, and that
    only works if the values really are zero when the flag is.
    """
    cfg = cfg or GoalPointConfig()
    out = np.zeros(GOAL_POINT_FEATS, dtype=np.float32)
    if not bool(valid):
        return out
    s = max(float(scale_m), _EPS)
    p = np.asarray(point_ego, dtype=np.float64).reshape(2)
    out[0] = np.clip(p[0] / s, -cfg.xy_clip, cfg.xy_clip)
    out[1] = np.clip(p[1] / s, -cfg.lat_clip, cfg.lat_clip)
    out[2] = 1.0
    return out


# ---------------------------------------------------------------------------
# the head (VISION ONLY at inference)
# ---------------------------------------------------------------------------

class GoalPointHead(nn.Module):
    """``ctx -> (x_norm, y_norm)`` — the predicted goal point, normalised by the arc.

    Deliberately a bare ``Linear``: the point of E15 is the FORM of the signal
    (metric, per-window, continuous), not head capacity, and a bigger head would make
    the comparison against E13's 4-row embedding a capacity comparison (C34).
    """

    def __init__(self, d_ctx: int):
        super().__init__()
        self.proj = nn.Linear(d_ctx, 2)

    def forward(self, ctx: Tensor) -> Tensor:
        return self.proj(ctx)


class GoalPointConditioning(nn.Module):
    """The E15 injection: ``[B, 3] -> d_goal -> (tactical, strategic)``, ZERO-INIT.

    Structurally the E13 nav block with the categorical table replaced by an MLP over a
    METRIC vector, and deliberately so — same sites, same additive form, same zero-init,
    so the ONE variable between the arms is the TYPE of the conditioning signal.

    ⚠️ Zero-init buys that the edge is bit-inert at step 0, NOT that a goal-point build
    equals a nav build at step 0: the two have different parameter shapes and every
    subsequent RNG draw shifts. The arms are compared by TRAINING them.
    """

    def __init__(self, d_goal: int, d_tac: int, d_ctx: int):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(GOAL_POINT_FEATS, d_goal), nn.GELU(),
                                 nn.Linear(d_goal, d_goal))
        self.to_tac = nn.Linear(d_goal, d_tac)
        self.to_str = nn.Linear(d_goal, d_ctx)
        for lin in (self.to_tac, self.to_str):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, feats: Tensor) -> tuple[Tensor, Tensor]:
        """``feats`` [B, 3] -> ``(to_tactical [B, d_tac], to_strategic [B, d_ctx])``.

        ⛔ Both outputs are multiplied by the validity bit, so a window with NO goal
        contributes EXACTLY zero rather than the encoder's bias — the same discipline
        the ego block uses (`refc_v3.py`, `es * keep_b`). Without this the "withheld"
        regime would silently become a learned constant, which is the presence-gated
        bias this whole module exists to remove.
        """
        if feats.dim() != 2 or feats.shape[1] != GOAL_POINT_FEATS:
            raise ValueError(f"goal features must be [B, {GOAL_POINT_FEATS}], got "
                             f"{tuple(feats.shape)}")
        keep = feats[:, 2:3]
        e = self.enc(feats)
        return self.to_tac(e) * keep, self.to_str(e) * keep


# ---------------------------------------------------------------------------
# the PARAM-FREE geometric compatibility — why the value cannot be ignored
# ---------------------------------------------------------------------------

def anchor_point_at_arc(bank: Tensor, arc_m: Tensor) -> Tensor:
    """``bank`` [B, N, S, 2] -> the point at arc length ``arc_m`` [B] along each anchor.

    ⛔⛔ ARC IS MEASURED FROM THE CAR, not from waypoint 0. The bank's first slot is
    already ~0.5 s ahead (V3_HORIZONS starts at 5 ticks), so measuring from it would put
    the anchor's arc origin ~5 m down the road while :func:`goal_point_label` measures
    the GOAL's arc from the ego origin — a systematic several-metre mismatch between the
    two quantities being compared, which is the derived-constant trap in geometry
    costume. The ego origin is prepended here for exactly the reason `lan.py` prepends
    it in ``horizon_lead_m``: *"the arc length is measured from the car, not from
    waypoint 0"*. Caught by ``test_anchor_point_at_arc_reads_a_KNOWN_value_*``.

    ⚠️ An anchor SHORTER than the arc is clamped to its own endpoint. That is a real
    approximation (30.6 % of anchor-windows in the probe) and it is stated rather than
    hidden: it makes distant goals LESS discriminative, which is the direction that
    understates the lever, never overstates it.
    """
    if bank.dim() != 4 or bank.shape[-1] != 2:
        raise ValueError(f"bank must be [B, N, S, 2], got {tuple(bank.shape)}")
    bank = torch.cat([bank.new_zeros(bank.shape[0], bank.shape[1], 1, 2), bank], dim=2)
    b, n, s, _ = bank.shape
    if s < 2:
        return bank[:, :, -1]
    seg = torch.linalg.vector_norm(bank[:, :, 1:] - bank[:, :, :-1], dim=-1)
    cum = torch.cat([bank.new_zeros(b, n, 1), seg.cumsum(-1)], dim=-1)     # [B, N, S]
    a = arc_m.reshape(b, 1, 1).to(bank.dtype)
    j = (cum < a).sum(-1).clamp(1, s - 1)                                  # [B, N]
    idx = j.unsqueeze(-1)
    c0 = cum.gather(-1, idx - 1).squeeze(-1)
    c1 = cum.gather(-1, idx).squeeze(-1)
    w = torch.where(c1 > c0, (a.reshape(b, 1) - c0) / (c1 - c0).clamp_min(_EPS),
                    torch.zeros_like(c0)).clamp(0.0, 1.0).unsqueeze(-1)
    g = idx.unsqueeze(-1).expand(b, n, 1, 2)
    p0 = bank.gather(2, g - 1).squeeze(2)
    p1 = bank.gather(2, g).squeeze(2)
    return p0 * (1.0 - w) + p1 * w


def anchor_goal_prior(goal_xy: Tensor, valid: Tensor, arc_m: Tensor,
                      bank: Tensor) -> Tensor:
    """[B, N] — the PARAM-FREE compatibility between the goal point and every anchor.

    ``score_i = -||anchor_i(arc) - goal|| / arc``, and EXACTLY ``0`` on a row whose goal
    is invalid.

    ⭐⭐ THIS IS THE VALUE-SENSITIVITY MECHANISM, and it is the whole architectural
    argument. E13's nav entered as ``Embedding(4) -> Linear -> add``: a 4-row table
    whose BETWEEN-row variance is free to shrink to zero while the MEAN survives as a
    useful bias — which is exactly what it did (content 2.3 %). A metric goal entering
    a param-free geometric score has no such degree of freedom: move the goal and the
    ranking moves, by construction. There is no "presence" here to gate on, because
    presence without content scores every anchor identically.
    ⛔ The learned gate that multiplies this term is still free to go to zero. That is
    why value-sensitivity is a pre-registered GATE with a committed minimum degradation
    under a corrupted goal, not an architectural promise.
    """
    if goal_xy.dim() != 2 or goal_xy.shape[1] != 2:
        raise ValueError(f"goal_xy must be [B, 2], got {tuple(goal_xy.shape)}")
    pts = anchor_point_at_arc(bank, arc_m)                                 # [B, N, 2]
    d = torch.linalg.vector_norm(pts - goal_xy.unsqueeze(1), dim=-1)       # [B, N]
    s = -d / arc_m.reshape(-1, 1).clamp_min(_EPS).to(d.dtype)
    return s * valid.reshape(-1, 1).to(d.dtype)


# ---------------------------------------------------------------------------
# supervision
# ---------------------------------------------------------------------------

def anchor_goal_prior_at_time(goal_xy: Tensor, valid: Tensor, bank: Tensor,
                              slot: int, scale_m: float) -> Tensor:
    """[B, N] — the PARAM-FREE compatibility at a fixed TIME slot. THE REGISTERED FORM.

    ``score_i = -||bank_i[slot] - goal|| / scale_m``, EXACTLY 0 on an invalid row.

    ⛔ ``goal_xy`` IS IN METRES, in the ego frame, like ``bank``. ``scale_m``
    divides the resulting DISTANCE and does not convert the input. The head
    emits a NORMALISED point, so a caller holding the head's output wants
    :func:`anchor_goal_prior_at_time_from_norm`, which states the conversion in
    its name — see that docstring for the measurement that made this line
    necessary.

    WHY TIME AND NOT ARC, measured (`raw/GP_LONG_AXIS_t4.json`,
    `raw/GP_LATERAL_ADAPTIVE.json`): at a fixed ARC every candidate sits at ~the same
    range, so ``-||a - p||`` and ``cos(a, p)`` are monotonically related and a point IS
    a bearing — they tie to four decimals on the lateral axis. At a fixed TIME the
    candidates differ in RANGE, and range is where 88.7 % of the oracle gap lives: the
    point recovers 57.1 % of the longitudinal selection ceiling while the same goal with
    its range stripped is separated WORSE by +2.3632 m. The metric half of "metric goal
    point" is the whole lever, and it exists only in time mode.
    """
    if goal_xy.dim() != 2 or goal_xy.shape[1] != 2:
        raise ValueError(f"goal_xy must be [B, 2], got {tuple(goal_xy.shape)}")
    if bank.dim() != 4 or not (-bank.shape[2] <= slot < bank.shape[2]):
        raise ValueError(f"slot {slot} out of range for bank {tuple(bank.shape)}")
    d = torch.linalg.vector_norm(bank[:, :, slot] - goal_xy.unsqueeze(1), dim=-1)
    return (-d / max(float(scale_m), _EPS)) * valid.reshape(-1, 1).to(d.dtype)


def anchor_goal_prior_at_time_from_norm(goal_norm: Tensor, valid: Tensor,
                                        bank: Tensor, slot: int,
                                        scale_m: float) -> Tensor:
    """[B, N] — :func:`anchor_goal_prior_at_time` fed the head's OWN output.

    ⛔⛔ THIS FUNCTION EXISTS BECAUSE THE TWO CONSUMERS OF "the goal point"
    TAKE DIFFERENT UNITS, AND THE MISMATCH READS EXACTLY LIKE AN ANSWER.

    * :class:`GoalPointHead` emits, and :func:`encode_goal_point` /
      :func:`goal_point_loss` consume, the **NORMALISED** point
      ``(x / scale_m, y / scale_m)`` — that is what the loss supervises and what
      :func:`lateral_rmse_m` multiplies back up to report metres.
    * :func:`anchor_goal_prior_at_time` compares the goal against ``bank`` and
      the bank is in **METRES**, so its ``goal_xy`` is in metres too; ``scale_m``
      there only makes the resulting SCORE scale-free.

    MEASURED 2026-09-06 while wiring GP-1: passing the head's normalised point
    straight into the prior (which is what PREREG §8's draft snippet did, and
    what ``test_the_hook_emits_the_point_for_the_selection_seam`` asserts the
    SHAPE of) puts the goal ~2.5 m from the car instead of ~24 m. The term is
    finite, the shapes match, the seam trains — and a MIRRORED goal then moved
    the ranked score by 7.7e-4 and flipped **0 of 4** picks on a rig whose
    anchors span ±66–82 m laterally, i.e. the pre-registered value-sensitivity
    gate would have failed for a UNITS reason and been read as "the geometric
    prior does not respond to the goal".

    Same family as `anchors.pt`'s ``controls[:, 1]`` read as curvature instead
    of lateral acceleration (396 g vs 0.31 g — both tables plausible, one from
    the shipped file): a true quantity quoted outside its scope. The fix is the
    same one that closed that case — put the conversion in ONE named place and
    say the units in the name.
    """
    return anchor_goal_prior_at_time(
        goal_norm * float(scale_m), valid, bank, slot, scale_m)


def goal_point_loss(pred: Tensor, tgt: Tensor, valid: Tensor) -> Tensor:
    """Smooth-L1 over the NORMALISED goal point, masked. Invalid rows contribute
    EXACTLY zero (the mask multiplies the summand, so the gradient at a masked row is
    structurally zero, not merely small) — the same construction as
    `refc_v3.masked_goal_loss`."""
    if pred.shape != tgt.shape or pred.shape[0] != valid.shape[0]:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(tgt.shape)} "
                         f"vs {tuple(valid.shape)}")
    m = valid.reshape(-1).to(pred.dtype)
    per = torch.nn.functional.smooth_l1_loss(pred, tgt, reduction="none").sum(-1)
    return (per * m).sum() / m.sum().clamp_min(1.0)


def lateral_rmse_m(pred: Tensor, tgt: Tensor, valid: Tensor, scale_m) -> Tensor:
    """The head's own pre-registered readout: RMS LATERAL error in METRES at the arc.

    Committed bar **<= 1.0 m** (and <= 2.0 m on range, :func:`range_rmse_m`). MEASURED:
    at 1.0 m injected lateral error the oracle goal point's recovery of the lateral
    bank-ADE ceiling flips from +78.4 % to **-80.7 %**; at 4.0 m the turn benefit is
    gone too. Reported every eval, in METRES, so a planner claim is never made on a head
    that has not earned it.
    """
    return _axis_rmse(pred, tgt, valid, scale_m, 1)


def range_rmse_m(pred: Tensor, tgt: Tensor, valid: Tensor, scale_m) -> Tensor:
    """RMS ALONG-TRACK error in METRES — the half a bearing cannot carry.

    Committed bar **<= 2.0 m**: recovery of the LONGITUDINAL selection ceiling is
    57.1 % (sigma 0) -> 47.2 % (1 m) -> 17.2 % (2 m, still separated) -> **-57.9 %
    (4 m, separated WORSE)**.
    """
    return _axis_rmse(pred, tgt, valid, scale_m, 0)


def _axis_rmse(pred: Tensor, tgt: Tensor, valid: Tensor, scale_m, axis: int) -> Tensor:
    m = valid.reshape(-1).to(pred.dtype)
    sc = (scale_m if torch.is_tensor(scale_m)
          else torch.as_tensor(float(scale_m), dtype=pred.dtype))
    d = (pred[:, axis] - tgt[:, axis]) * sc.reshape(-1).to(pred.dtype)
    return ((d ** 2 * m).sum() / m.sum().clamp_min(1.0)).sqrt()


def goal_point_provenance(cfg: GoalPointConfig | None = None) -> dict:
    """The machine-readable admissibility declaration, written into every run's
    ``config.json`` and asserted by ``tests/test_goal_point.py`` — so the claim lives
    in the RUN RECORD rather than in a docstring a reader has to trust.

    Same shape as ``RefCModel.goal_provenance`` on purpose: two goal declarations that
    drift apart is the label-space defect class again.
    """
    cfg = cfg or GoalPointConfig()
    return {
        "edge": "E15",
        "form": ("geometric METRIC point (x, y) in the ego frame at a fixed TIME "
                 "beyond the scored horizon (goal_mode=%r, t_goal_s=%.1f, "
                 "t_pred_s=%.1f)" % (cfg.goal_mode, cfg.t_goal_s, cfg.t_pred_s)),
        "inference_inputs": ["the strategic context token (vision)"],
        "supplied_or_predicted": "predicted",
        "contains_situation_classifier_output": False,
        "situation_classifier_in_graph": False,
        "reads_tactical_state_or_logits": False,
        "shared_trunk_with": ["route_head", "tactical head"],
        "shared_trunk_justification":
            "shared ENCODER, not a shared signal; a shared trunk can only launder a "
            "signal that EXISTS in the graph, and the situation classifier's does not. "
            "Attributability comes from the zero-init projections and gate, so the "
            "exact ablation is 'set them to 0'.",
        "label_source": ("the ego's own FUTURE path, arc-length resampled by "
                         "tanitad.data.lan.resample_arclength — TRAIN ONLY, never read "
                         "at inference (PI 2026-08-03: labels may use ego; inference "
                         "is vision-only)"),
        "leak_guard": ("tanitad.data.lan.horizon_lead_m, IMPORTED: "
                       "max(2 s GT arc length, v0 * %.1f) + %.1f m, clamped to "
                       "[%.1f, %.1f] m" % (cfg.t_pred_s, cfg.min_lead_m,
                                           cfg.arc_min_m, cfg.arc_max_m)),
        "admissibility_check":
            "could this goal have been computed from the situation classifier's "
            "output? NO — that model is not imported, not in this graph, not a batch "
            "field and not a label source here.",
        "value_sensitivity_mechanism":
            "a param-free geometric compatibility with the anchor bank "
            "(anchor_goal_prior_at_time): moving the goal moves the ranking by "
            "construction, "
            "so there is no presence-without-content regime to collapse into — the "
            "measured failure mode of the categorical E13 edge.",
        "head_requirement_lateral_m": 1.0,
        "head_requirement_range_m": 2.0,
        "head_requirement_basis":
            "MEASURED. LATERAL (raw/GP_LATERAL_ADAPTIVE.json): at 1.0 m injected "
            "lateral error the oracle goal point's recovery of the lateral-selection "
            "ceiling flips from +78.4 % to -80.7 %. RANGE (raw/GP_LONG_AXIS_t4.json): "
            "recovery of the longitudinal ceiling 57.1 % -> 17.2 % at 2 m -> -57.9 % "
            "at 4 m, separated WORSE.",
        "leak_guard_time_mode":
            "GoalPointConfig REFUSES t_goal_s <= t_pred_s at construction, so no arm "
            "that leaks the scored horizon can be built at all.",
    }
