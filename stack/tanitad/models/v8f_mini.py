"""v8F-mini -- the PI's three-layer v8F (S -> T -> O) on CACHED frozen DINOv3-L features (track M, 8 GB GPU).

Spec: ``Project Steering/PLAN_V8F_REFAV1_HJEPA_2026-10-17.md`` §13 (binding) and ``PREREG_V8F_MINI.md``.

* **S (strategic, 1 Hz)**   scene summary (unless ``navonly``) + nav + distance + ego dynamics + max speed
                            -> route CHECKPOINTS (x, y, heading) at 30 / 50 / 80 m of arc, ego frame.
* **T (tactical, 2 Hz)**    checkpoints (S, held) + inputs + scene (unless ``navonly``)
                            -> v9 goal logits, lat / lon action logits, constraint regressions.
* **O (operative, 10 Hz)**  T's output (held) + checkpoints (held) + inputs + scene
                            -> a 6 s trajectory (60 x 0.1 s) as a RESIDUAL on the P10 prior.

SAMPLE-AND-HOLD: each layer is evaluated at its OWN tick frame (NOW - age) and the lower layer reads the held
output. Ages are in 0.2 s cache steps (the cache grid): T ticked ``a_t`` in 0..2 steps ago (< 0.5 s), S ticked
``a_s - a_t`` in 0..4 steps before T (< 1.0 s). Held checkpoints are re-expressed in the reading layer's frame with
the MEASURED odometry between the two tick frames (``pose`` -- an ego-dynamics input, not a label).
SCHEDULED SAMPLING: during training, with probability ``p_gt`` (1.0 -> 0.2) a layer reads the GT upper-layer output
(v9 labels at the tick frame) instead of the predicted one; at inference never (``forward_infer`` reads no label).

ADMISSIBILITY (binding 2026-08-03): S, T and O are computed from vision + nav + ego dynamics + max speed ONLY.
``assert_no_situation_classifier_input`` pins it: the inference entry point reads exactly
:data:`INFERENCE_INPUT_KEYS` (recorded at call time) and the source mentions no situation classifier.

CONSISTENCY: differentiable hinge penalties for SPEED / STOP / TURN / LEAD that are zero iff the numpy rule of
``taniteval.v8f_consistency`` says "no violation" (agreement is tested); plus the HARD inference projection
(:func:`project_speed`) onto the R1 speed cap and the Kamm circle (mu 0.7).
"""
from __future__ import annotations

import ast
import inspect
import math
from dataclasses import dataclass, field, fields

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

DT = 0.2
K30, N60 = 30, 60
CKPT_ARCS = (30.0, 50.0, 80.0)
N_GOAL, N_LAT, N_LON, N_CONS = 22, 5, 7, 26
CODE_DIM = N_GOAL + N_LAT + N_LON + N_CONS
TURN_L_I, TURN_R_I, STOP_I = 1, 2, 6                      # GOAL22 indices
LON_FOLLOW = 3
CONS_NAMES = ("lat_theta_deg", "turn_t_start_s", "turn_t_end_s", "turn_d_start_m", "turn_len_m", "turn_dyaw_deg",
              "turn_r_arc_m", "turn_exit_x_m", "turn_exit_y_m", "turn_exit_psi_deg", "lc_t_cross_s", "lc_d_cross_m",
              "lon_v_target_ms", "lon_t_reach_s", "lon_d_reach_m", "lon_t_start_s", "stop_x_m", "stop_y_m",
              "lead_gap_m", "lead_tg_s", "lead_gap_min_m", "lead_tg_min_s", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")
CI = {n: i for i, n in enumerate(CONS_NAMES)}
#: PREREG_V8F_TAC Amendment 1: a PARTITION of T's code [goal 22 | lat 5 | lon 7 | cons 26] into six groups, so a per-group oracle can
#: replace one part at a time. Every code index belongs to exactly one group (pinned by test_v8f_tac.py).
_C0 = N_GOAL + N_LAT + N_LON
CODE_GROUPS = {
    "goal": tuple(range(0, N_GOAL)),
    "latdec": tuple(range(N_GOAL, N_GOAL + N_LAT)),
    "londec": tuple(range(N_GOAL + N_LAT, _C0)),
    "latcons": tuple(_C0 + CI[n] for n in ("lat_theta_deg", "turn_t_start_s", "turn_t_end_s", "turn_d_start_m", "turn_len_m",
                                            "turn_dyaw_deg", "turn_r_arc_m", "turn_exit_x_m", "turn_exit_y_m", "turn_exit_psi_deg",
                                            "lc_t_cross_s", "lc_d_cross_m")),
    "lonspeed": tuple(_C0 + CI[n] for n in ("lon_v_target_ms", "lon_t_reach_s", "lon_d_reach_m", "lon_t_start_s", "v_a_ms", "v_end_ms",
                                             "v_lo_ms", "v_hi_ms")),
    "stoplead": tuple(_C0 + CI[n] for n in ("stop_x_m", "stop_y_m", "lead_gap_m", "lead_tg_s", "lead_gap_min_m", "lead_tg_min_s")),
}
#: the two composites the Amendment 1 decision reads: the categorical DECISIONS vs the continuous CONSTRAINTS
CODE_COMPOSITES = {"decisions": ("goal", "latdec", "londec"), "constraints": ("latcons", "lonspeed", "stoplead")}
#: PREREG_V8F_KIND: a SECOND partition of the same 26 constraint cells, by what a cell MEASURES rather than by which decision it hangs off:
#: speeds, angles, times-to-event, time gaps, geometry (metres of path) and lead gaps. Each cell is in exactly one kind and the six kinds' union is
#: the ``constraints`` composite (pinned by test_v8f_kind.py). A kind's name must not collide with CODE_GROUPS / CODE_COMPOSITES. ``code_group_index``
#: takes a kind anywhere it takes a group: ``speed``, ``speed+angle``, ``all-speed`` (GT everywhere EXCEPT the speed cells).
CODE_KINDS = {
    "speed": tuple(_C0 + CI[n] for n in ("lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")),
    "angle": tuple(_C0 + CI[n] for n in ("lat_theta_deg", "turn_dyaw_deg", "turn_exit_psi_deg")),
    "tte": tuple(_C0 + CI[n] for n in ("turn_t_start_s", "turn_t_end_s", "lc_t_cross_s", "lon_t_reach_s", "lon_t_start_s")),
    "timegap": tuple(_C0 + CI[n] for n in ("lead_tg_s", "lead_tg_min_s")),
    "geom": tuple(_C0 + CI[n] for n in ("turn_d_start_m", "turn_len_m", "turn_r_arc_m", "turn_exit_x_m", "turn_exit_y_m", "lc_d_cross_m",
                                         "lon_d_reach_m", "stop_x_m", "stop_y_m")),
    "leadgap": tuple(_C0 + CI[n] for n in ("lead_gap_m", "lead_gap_min_m")),
}


def code_group_index(spec: str) -> list:
    """``"goal+londec"`` | ``"decisions"`` | ``"speed+angle"`` (a PREREG_V8F_KIND kind) | ``"all"`` | ``"all-<spec>"`` -> sorted code indices. Unknown names are refused.
    ``all-<spec>`` is the COMPLEMENT (PREREG_V8F_TAC Amendment 2's leave-one-out: GT everywhere except ``<spec>``)."""
    if spec == "all":
        return list(range(CODE_DIM))
    if spec.startswith("all-"):
        drop = set(code_group_index(spec[len("all-"):]))
        keep = [i for i in range(CODE_DIM) if i not in drop]
        if not keep:
            raise ValueError(f"code group {spec!r} selects nothing")
        return keep
    names, idx = [], set()
    for part in spec.split("+"):
        if part in CODE_COMPOSITES:
            names.extend(CODE_COMPOSITES[part])
        elif part in CODE_GROUPS:
            names.append(part)
        elif part in CODE_KINDS:                    # PREREG_V8F_KIND: a kind of constraint cell
            idx.update(CODE_KINDS[part])
        else:
            raise ValueError(f"unknown code group {part!r}: use {sorted(CODE_GROUPS) + sorted(CODE_COMPOSITES) + sorted(CODE_KINDS)} or 'all', joined by '+'")
    idx.update(i for n in names for i in CODE_GROUPS[n])
    return sorted(idx)


#: PREREG_V8F_MIXO (`--code-mix`): the granularity of O's teacher forcing. ``whole`` (the default, every earlier arm bit for bit) = ONE GT-or-own switch per sample over the whole
#: code; ``kind`` = an independent switch per sample AND per UNIT, at the same p_gt, so O also trains on a code that is partly right and partly T's own.
CODE_MIX_MODES = ("whole", "kind")
#: the units ``kind`` draws over: the decisions (goal + latdec + londec, one unit) and the six constraint kinds of PREREG_V8F_KIND. Every code index is in EXACTLY one unit (the
#: decisions' 34 indices and the 26 constraint cells; pinned by test_v8f_mixo.py).
CODE_MIX_UNITS = {"decisions": tuple(i for n in CODE_COMPOSITES["decisions"] for i in CODE_GROUPS[n]), **CODE_KINDS}
N_CODE_UNITS = len(CODE_MIX_UNITS)
if sorted(i for idx in CODE_MIX_UNITS.values() for i in idx) != list(range(CODE_DIM)):
    raise ImportError("CODE_MIX_UNITS is not a partition of the code: every index must be in exactly one unit")
#: code index -> its unit's column in the [B, N_CODE_UNITS] draw (built from the partition checked above)
CODE_UNIT_OF = tuple(next(u for u, idx in enumerate(CODE_MIX_UNITS.values()) if i in idx) for i in range(CODE_DIM))


#: PREREG_V8F_FLAT (`--o-interface`): what O reads of the two layers above it. ``full`` (the default, every earlier arm bit for bit, the RNG stream included) = S's checkpoints and T's
#: code. ``cut`` = a CONSTANT in place of BOTH (T's code = zeros, S's checkpoints for O = the nominal ``ck_nom`` with ok = 0), applied after every override so no eval can reach O through them.
#: S and T are still built, trained on their own losses and run; T still reads S. Only the wires into O are cut.
O_INTERFACE_MODES = ("full", "cut")


LIMITS_MS = (30 / 3.6, 50 / 3.6, 100 / 3.6, 120 / 3.6)
#: the scale the WM auxiliary divides action channel 1 by, per kin mode (PI ruling D-PI-YAWRATE-1): the recorded steer angle
#: (rad) in the legacy mode, the MEASURED yaw rate (rad/s) in the compliant one. 0.4 keeps the scaled channel's std at 0.255 vs the
#: steer channel's 0.267 (MEASURED on the first 200 sha12 train clips: std omega 0.1022 rad/s, std steer 0.0802 rad).
CH1_SCALE = {"steer": 0.3, "yaw": 0.4}
#: the scale the WM auxiliary divides action channel 0 (longitudinal acceleration, m/s^2) by.
WM_ACC_SCALE = 2.0
#: EXACTLY the batch keys the inference path may read. Everything else (``lab*``, ``gt``, ``s_h``, ``lead_gap_h`` ...)
#: is a label or an evaluator-side quantity and is never touched by ``forward_infer``.
INFERENCE_INPUT_KEYS = frozenset({"feats", "frame_ok", "p10f", "act", "nav", "pose", "limit", "limit_ok", "base",
                                  "a_t", "a_s", "pose_now", "pose_t", "pose_s"})


# ------------------------------------------------------------------------------------------------ config
@dataclass
class V8FMiniConfig:
    """Every field is a lever; ``built_levers`` reports what the BUILT model actually does so the trainer can
    refuse a launch whose argv declares something the model did not build."""
    d: int = 256
    heads: int = 4
    ff: int = 512
    d_feat: int = 1024
    n_tok: int = 40
    hist: int = 4
    hmax: int = 10
    n_q: int = 4
    depth_s: int = 1
    depth_t: int = 1
    depth_o: int = 2
    nb: int = 12                       # residual basis size per axis
    res_scale_m: float = 2.0
    navonly: bool = False              # S and T lose the scene input (PREREG M1-navonly)
    consistency: bool = True           # False = penalties at 0 (PREREG M1-nocons)
    wm: str = "off"                    # off | a (H-JEPA aux without IDM) | b (with IDM)
    wm_dim: int = 128
    wm_strides: tuple = (2, 2)
    w_ckpt: float = 1.0
    w_tac: float = 1.0
    w_cons: float = 0.5
    w_traj: float = 1.0
    w_pen: float = 0.5                 # x each of SPEED / STOP / TURN / LEAD vs the GT constraints
    w_pen_self: float = 0.25           # x each, vs the model's OWN predicted constraints
    w_wm: float = 0.1
    wm_beta: float = 0.08
    wm_gamma: float = 100.0
    p_gt_start: float = 1.0
    p_gt_end: float = 0.2
    max_age_t: int = 2
    max_age_s: int = 6
    detach_upper: bool = False         # stop O/T gradients into the upper layer's output
    kin_mode: str = "steer"            # steer | yaw: what the kinematic inputs (P10 features, base, WM action channel 1) are made of
    cons_absent: str = "mask"          # mask: an ABSENT constraint is out of T's loss (M1 / M2b as built); zero: T's target there is the 0
                                       # that _code_gt writes and O is teacher-forced on (PREREG_V8F_CZ)
    window_start: int = -1             # PREREG_V8F_HIST Amendment 1 (`--window-start`): the first TRAINING window's cache step t AND the held-age clamp's origin
                                       # (a DATA lever: the model never reads it, the checkpoint records it). -1 = absent = hist - 1 = every earlier arm, bit for bit
    code_mix: str = "whole"            # PREREG_V8F_MIXO (`--code-mix`): how O's teacher forcing draws GT-or-own over T's code. whole = one switch per sample (every earlier arm,
                                       # bit for bit, RNG stream included); kind = an independent switch per sample AND per unit (CODE_MIX_UNITS). Training only: the eval never teacher-forces
    o_interface: str = "full"          # PREREG_V8F_FLAT (`--o-interface`): full = O reads S's checkpoints and T's code (every earlier arm, bit for bit); cut = O reads a CONSTANT for both (the FLAT control:
                                       # T's code = zeros AFTER every override, S's checkpoints = the nominal ck_nom with ok = 0). S and T are still built, trained and run
    w_acc: float = 0.0                 # PREREG_V8F_OACC (`--w-acc`): the weight of the acceleration-matching term on O's plan (accel_term below). 0.0 = the term is NOT in the total = every earlier arm,
                                       # bit for bit (the term is still LOGGED under T["acc"]). > 0 = OACC's X: its value is fixed by the PREREG's rule, written into the argv, never tuned

    @property
    def window_start_effective(self) -> int:
        """The number the data side must have built: ``hist - 1`` when the flag is absent (-1), else the flag."""
        return self.hist - 1 if self.window_start == -1 else self.window_start

    def __post_init__(self):
        if self.kin_mode not in CH1_SCALE:
            raise ValueError(f"kin_mode must be one of {tuple(CH1_SCALE)}, got {self.kin_mode!r}")
        if self.wm not in ("off", "a", "b"):
            raise ValueError(f"wm must be off|a|b, got {self.wm!r}")
        if self.cons_absent not in ("mask", "zero"):
            raise ValueError(f"cons_absent must be mask|zero, got {self.cons_absent!r}")
        if self.code_mix not in CODE_MIX_MODES:
            raise ValueError(f"code_mix must be one of {CODE_MIX_MODES}, got {self.code_mix!r}")
        if self.o_interface not in O_INTERFACE_MODES:
            raise ValueError(f"o_interface must be one of {O_INTERFACE_MODES}, got {self.o_interface!r}")
        if not (math.isfinite(self.w_acc) and self.w_acc >= 0.0):
            raise ValueError(f"w_acc must be a finite number >= 0, got {self.w_acc!r}")
        if self.hmax < self.hist + self.max_age_s:
            raise ValueError("hmax must cover hist + max_age_s")
        if self.window_start != -1 and not 0 <= self.window_start <= self.hist - 1:
            raise ValueError(f"window_start must be -1 (absent = hist - 1) or in [0, hist - 1] = [0, {self.hist - 1}], got {self.window_start}")
        if self.wm != "off" and self.hmax < 2 * self.wm_strides[0] * self.wm_strides[1] + 4:
            raise ValueError("hmax too short for the WM levels")


# ------------------------------------------------------------------------------------------------ geometry
def wrap(a: Tensor) -> Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi


def transform_ckpt(ck: Tensor, src: Tensor, dst: Tensor) -> Tensor:
    """Re-express ``ck`` [B, M, 3] (x, y, psi) given in the ``src`` frame in the ``dst`` frame; poses [B, 3] = world
    (x, y, yaw) of each tick frame (measured odometry)."""
    cs, ss = torch.cos(src[:, 2])[:, None], torch.sin(src[:, 2])[:, None]
    wx = src[:, 0:1] + cs * ck[..., 0] - ss * ck[..., 1]
    wy = src[:, 1:2] + ss * ck[..., 0] + cs * ck[..., 1]
    dx, dy = wx - dst[:, 0:1], wy - dst[:, 1:2]
    cd, sd = torch.cos(dst[:, 2])[:, None], torch.sin(dst[:, 2])[:, None]
    return torch.stack([cd * dx + sd * dy, -sd * dx + cd * dy, wrap(ck[..., 2] + src[:, 2:3] - dst[:, 2:3])], -1)


# ------------------------------------------------------------------------------------------------ P10 prior
class P10Prior(nn.Module):
    """The frozen, fitted P10 floor: ``kd_x(correct-unit) + MLP(standardised 8 kinematic features)``.

    Built from ``taniteval.floors_kin.fit_p10`` output ``(net, (xm, xs, ym), best)``. ``forward(f8, base)`` takes the
    RAW (unstandardised) 8 features and the kd_x base ``[B, 30, 2]`` and returns the P10 prediction ``[B, 30, 2]``."""

    def __init__(self, xm, xs, ym, hidden: int = 256, k: int = K30):
        super().__init__()
        self.k = int(k)
        self.net = nn.Sequential(nn.Linear(8, hidden), nn.GELU(), nn.Linear(hidden, 2 * self.k))
        for nm, v in (("xm", xm), ("xs", xs), ("ym", ym)):
            self.register_buffer(nm, torch.as_tensor(np.asarray(v), dtype=torch.float32).clone())
        self.requires_grad_(False)

    @classmethod
    def from_fit(cls, fit):
        net, (xm, xs, ym), _best = fit
        m = cls(xm, xs, ym, hidden=net[0].out_features, k=net[2].out_features // 2)
        m.net.load_state_dict(net.state_dict())
        return m

    @classmethod
    def synthetic(cls, seed: int = 0):
        """Random NON-ZERO weights (so 'zero residual reproduces P10' is not satisfied trivially)."""
        g = torch.Generator().manual_seed(seed)
        m = cls(torch.zeros(8), torch.ones(8), torch.zeros(2 * K30))
        with torch.no_grad():
            for p in m.net.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
        return m

    def standardise(self, f8: Tensor) -> Tensor:
        return (f8 - self.xm) / self.xs

    def forward(self, f8: Tensor, base: Tensor) -> Tensor:
        y = self.net(self.standardise(f8)) + self.ym
        return base + y.reshape(-1, self.k, 2)


def save_p10(prior: P10Prior, path, meta: dict | None = None):
    torch.save({"state": prior.state_dict(), "hidden": prior.net[0].out_features, "k": prior.k, "meta": meta or {}}, path)


def p10_meta(path) -> dict:
    """The ``meta`` dict a P10 artifact was saved with (``units`` is ``steer`` | ``pose_kappa``; absent => ``steer``)."""
    return dict(torch.load(path, map_location="cpu", weights_only=True).get("meta") or {})


def check_p10_units(path, kin_mode: str) -> str:
    """REFUSE a P10 artifact whose declared feature unit is not the one ``kin_mode`` builds (a steer prior fed yaw-derived
    features -- or the reverse -- is a silently wrong model, not an error the first forward pass would raise). Returns the units."""
    want = {"steer": "steer", "yaw": "pose_kappa"}[kin_mode]
    got = p10_meta(path).get("units", "steer")
    if got != want:
        raise ValueError(f"P10 prior {path} declares units {got!r} but kin_mode {kin_mode!r} needs {want!r} "
                         f"(a steer-unit prior cannot be fed yaw-derived features, nor the reverse)")
    return got


def load_p10(path, kin_mode: str | None = None) -> P10Prior:
    """Load the frozen prior. With ``kin_mode`` named, the artifact's declared units must match it (``check_p10_units``)."""
    if kin_mode is not None:
        check_p10_units(path, kin_mode)
    o = torch.load(path, map_location="cpu", weights_only=True)
    m = P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * o["k"]), hidden=o["hidden"], k=o["k"])
    m.load_state_dict(o["state"])
    return m


def upsample_prior(p30: Tensor) -> Tensor:
    """P10 on the 0.2 s grid [B, 30, 2] -> the 0.1 s grid [B, 60, 2]: odd indices (tau = 0.2 j) ARE the knots,
    even indices are the midpoints (origin at tau = 0)."""
    B = p30.shape[0]
    knots = torch.cat([p30.new_zeros(B, 1, 2), p30], 1)               # tau = 0, 0.2, ..., 6.0
    mid = 0.5 * (knots[:, :-1] + knots[:, 1:])                          # tau = 0.1, 0.3, ...
    out = torch.stack([mid, knots[:, 1:]], 2).reshape(B, N60, 2)       # i even -> mid (tau 0.1(i+1)), i odd -> knot
    return out


def subsample_02(traj60: Tensor) -> Tensor:
    return traj60[:, 1::2]


def traj_terms(traj30: Tensor, gt: Tensor, gok: Tensor) -> tuple:
    """O's two position terms on the 0.2 s knots, ``(ade, fde)``: the mean / last-knot L2 distance to the GT, averaged over the GT-valid windows (``gok`` [B]). The loss reads this; so does the
    w_acc rule of PREREG_V8F_OACC (``v8f_mini_eval_oacc.py wacc``), so the trajectory loss it balances against IS the trainer's, never a re-implementation."""
    d = torch.sqrt(((traj30 - gt) ** 2).sum(-1) + 1e-9)
    return (d.mean(1) * gok).sum() / gok.sum().clamp_min(1.0), (d[:, -1] * gok).sum() / gok.sum().clamp_min(1.0)


def second_diff_02(xy: Tensor) -> Tensor:
    """PREREG_V8F_OACC: the second difference, on the 0.2 s plan grid, of a plan ``[B, 30, 2]`` (tau = 0.2 ... 6.0 s) INCLUDING the origin knot (tau = 0, the ego at NOW), exactly as
    ``upsample_prior`` prepends it: 31 knots -> 29 interior centres -> ``[B, 29, 2]``, in m per (0.2 s)^2. HSPEC's stage 2 uses the same 29 rows (``d2_matrix``)."""
    P = torch.cat([xy.new_zeros(xy.shape[0], 1, 2), xy], 1)
    return P[:, 2:] - 2.0 * P[:, 1:-1] + P[:, :-2]


def accel_term(traj30: Tensor, gt: Tensor, gok: Tensor) -> Tensor:
    """PREREG_V8F_OACC: ``1/|M| sum_{t in M} || D2 tau30(t) - D2 y30(t) ||^2`` -- O's plan against the human's, on the second difference. ``gok`` [B] is the GT-valid mask the ADE/FDE terms
    use (a window's GT is valid for ALL 30 steps or for none: ``gt_future``), so M = the valid windows x the 29 centres, and the mean is over those cells. A batch with no valid window gives 0.0."""
    d2 = second_diff_02(traj30) - second_diff_02(gt.to(traj30.dtype))
    sq = (d2 ** 2).sum(-1)                                   # [B, 29]: the squared L2 norm over the two axes
    g = gok.to(sq.dtype)
    return (sq.mean(1) * g).sum() / g.sum().clamp_min(1.0)


# ------------------------------------------------------------------------------------------------ hard projection
def project_speed(traj: Tensor, cap_ms: Tensor | None, mu: float = 0.7, dt: float = 0.1, g: float = 9.81) -> Tensor:
    """HARD inference projection of a path ``[B, N, 2]`` (dt s per step) onto speed <= cap (the R1 cap, from the
    max-speed INPUT) and the Kamm circle (|v^2 kappa| <= mu g). Geometry is preserved; only the timing changes:
    each interval's speed is capped, then the path is re-sampled at the new arc-length positions of the old path."""
    B, N, _ = traj.shape
    P = torch.cat([traj.new_zeros(B, 1, 2), traj], 1)
    seg = P[:, 1:] - P[:, :-1]
    L = torch.sqrt((seg ** 2).sum(-1) + 1e-12)
    v = L / dt
    ang = torch.atan2(seg[..., 1], seg[..., 0])
    dpsi = torch.zeros_like(ang)
    dpsi[:, 1:] = wrap(ang[:, 1:] - ang[:, :-1])
    kappa = dpsi.abs() / L.clamp_min(1e-3)
    kappa = torch.where(L > 1e-3, kappa, torch.zeros_like(kappa))
    v_kamm = torch.sqrt(mu * g / kappa.clamp_min(1e-6))
    vcap = v_kamm if cap_ms is None else torch.minimum(v_kamm, cap_ms.reshape(B, 1).expand(B, N))
    v_new = torch.minimum(v, vcap)
    S_old = torch.cat([L.new_zeros(B, 1), torch.cumsum(L, 1)], 1)
    S_new = torch.cumsum(v_new * dt, 1)
    idx = (torch.searchsorted(S_old.contiguous(), S_new.contiguous(), right=True) - 1).clamp(0, N - 1)
    L_i = torch.gather(L, 1, idx)
    frac = (S_new - torch.gather(S_old, 1, idx)) / L_i.clamp_min(1e-9)
    frac = torch.where(L_i > 1e-9, frac.clamp(0.0, 1.0), torch.zeros_like(frac))
    P0 = torch.gather(P, 1, idx[..., None].expand(-1, -1, 2))
    sg = torch.gather(seg, 1, idx[..., None].expand(-1, -1, 2))
    return P0 + frac[..., None] * sg


# ------------------------------------------------------------------------------------------------ penalties (torch)
RULE_DB_DEG, TURN_MIN_OVERLAP, SPEED_TOL, STOP_V, STOP_PASS, LEAD_TOL, LEAD_VF = 1.0, 1.0, 0.5, 1.0, 2.0, 0.2, 1.0
BAND_KNOT = 10


def plan_kin_t(xy: Tensor) -> dict:
    """Torch twin of ``v8f_consistency.plan_kinematics`` (xy [B, 30, 2] at 0.2 s)."""
    B, K, _ = xy.shape
    P = torch.cat([xy.new_zeros(B, 1, 2), xy], 1)
    seg = P[:, 1:] - P[:, :-1]
    L = torch.sqrt((seg ** 2).sum(-1) + 1e-12)
    s = torch.cat([L.new_zeros(B, 1), torch.cumsum(L, 1)], 1)
    cur = xy.new_zeros(B)
    psi = [cur]
    for j in range(K):
        mov = (seg[:, j] ** 2).sum(-1) > 1e-12                     # == |seg| > 1e-6, the numpy rule's test
        dx = torch.where(mov, seg[:, j, 0], torch.ones_like(cur))
        dy = torch.where(mov, seg[:, j, 1], torch.zeros_like(cur))
        ang = torch.atan2(dy, dx)
        ang = cur + wrap(ang - cur)
        cur = torch.where(mov, ang, cur)
        psi.append(cur)
    return {"s": s, "v": L / DT, "psi": torch.stack(psi, 1)}


def _interp_t(vals: Tensor, tau: Tensor) -> Tensor:
    x = (tau / DT).clamp(0.0, vals.shape[1] - 1.0)
    i0 = x.floor().clamp(max=vals.shape[1] - 2).long()
    f = x - i0.to(x.dtype)
    a, b = torch.gather(vals, 1, i0[:, None])[:, 0], torch.gather(vals, 1, (i0 + 1)[:, None])[:, 0]
    return a * (1 - f) + b * f


def consistency_penalties(xy: Tensor, c: dict) -> dict:
    """Hinge penalties (>= 0), one per rule, in the rule's own units, plus ``viol`` (bool) and ``active`` masks.
    ``c``: tensors with the schema of ``v8f_consistency.violations`` (NaN = unknown). hinge > 0 <=> violated."""
    kin = plan_kin_t(xy)
    s, v, psi = kin["s"], kin["v"], kin["psi"]
    B = xy.shape[0]
    z = xy.new_zeros(B)
    pen, viol, act = {}, {}, {}
    # SPEED
    cap = torch.fmin(c["limit_ms"], c["v_hi_ms"]) if hasattr(torch, "fmin") else torch.minimum(c["limit_ms"], c["v_hi_ms"])
    a = torch.isfinite(cap)
    vmax = v[:, BAND_KNOT:].amax(1)
    h = F.relu(vmax - torch.where(a, cap, z) - SPEED_TOL)
    pen["SPEED"], viol["SPEED"], act["SPEED"] = torch.where(a, h, z), a & (h > 0), a
    # STOP
    d, tstop = c["d_stop"], c["t_stop"]
    a = c["stop_active"] & torch.isfinite(d) & (tstop <= 6.0 + 1e-9)
    dd = torch.where(a, d, z)
    s_end = s[:, -1]
    reached = a & (s_end >= dd)
    j = (torch.searchsorted(s.detach().contiguous(), dd[:, None].contiguous(), right=True)[:, 0] - 1).clamp(0, v.shape[1] - 1)
    v_at = torch.gather(v, 1, j[:, None])[:, 0]
    h1, h2 = F.relu(v_at - STOP_V), F.relu(s_end - (dd + STOP_PASS))
    h = torch.where(reached, h1, z) + torch.where(a, h2, z)
    pen["STOP"], viol["STOP"], act["STOP"] = h, a & (h > 0), a
    # TURN
    side = c["turn_side"]
    t0 = torch.clamp(torch.nan_to_num(c["turn_t_start"], nan=float("inf")), min=0.0)
    t1 = torch.clamp(torch.nan_to_num(c["turn_t_end"], nan=-float("inf")), max=6.0)
    a = (side != 0) & ((t1 - t0) >= TURN_MIN_OVERLAP - 1e-9)
    t0c, t1c = torch.where(a, t0, z), torch.where(a, t1, z)
    dpsi = torch.rad2deg(_interp_t(psi, t1c) - _interp_t(psi, t0c))
    h = F.relu(RULE_DB_DEG - side.to(dpsi.dtype) * dpsi)
    pen["TURN"], viol["TURN"], act["TURN"] = torch.where(a, h, z), a & (h > 0), a
    # LEAD
    fin = torch.isfinite(c["lead_gap_h"])
    gap = torch.nan_to_num(c["lead_gap_h"], nan=0.0) + torch.nan_to_num(c["s_h"], nan=0.0) - s[:, 1:]
    tg = gap.clamp_min(0.0) / v.clamp_min(LEAD_VF)
    tgm = torch.where(fin, tg, torch.full_like(tg, float("inf")))[:, BAND_KNOT - 1:].amin(1)
    have = torch.isfinite(tgm)
    a = torch.isfinite(c["lead_tg_min"]) & have
    h = F.relu(torch.where(a, c["lead_tg_min"], z) - LEAD_TOL - torch.where(a, tgm, z))
    pen["LEAD"], viol["LEAD"], act["LEAD"] = torch.where(a, h, z), a & (h > 0), a
    return {"pen": pen, "viol": viol, "active": act}


PEN_SCALE = {"SPEED": 1.0, "STOP": 1.0, "TURN": 10.0, "LEAD": 0.5}
RULES = ("SPEED", "STOP", "TURN", "LEAD")


def limit_ms_from_onehot(limit: Tensor, ok: Tensor) -> Tensor:
    lim = (limit * limit.new_tensor(LIMITS_MS)).sum(-1)
    return torch.where(ok, lim, torch.full_like(lim, float("nan")))


def gt_constraints_t(lab: dict, limit_ms: Tensor, lead_gap_h: Tensor, s_h: Tensor) -> dict:
    """Torch twin of ``constraints_from_v9`` on a collated label pack (batch row = current frame)."""
    y = lab["goal_y"] * lab["goal_w"]
    tl, tr = y[:, TURN_L_I] > 0.5, y[:, TURN_R_I] > 0.5
    cons = lab["cons"]
    has = lab["has"]
    side = torch.where(tl, 1, torch.where(tr, -1, 0)) * has.long()
    nan = torch.full_like(limit_ms, float("nan"))
    lead = torch.where((lab["lead_valid"] == 1) & has, cons[:, CI["lead_tg_min_s"]], nan)
    return {"limit_ms": limit_ms, "v_hi_ms": torch.where(has, cons[:, CI["v_hi_ms"]], nan),
            "stop_active": (y[:, STOP_I] > 0.5) & has, "d_stop": cons[:, CI["lon_d_reach_m"]],
            "t_stop": cons[:, CI["lon_t_reach_s"]], "turn_side": side, "turn_t_start": cons[:, CI["turn_t_start_s"]],
            "turn_t_end": cons[:, CI["turn_t_end_s"]], "lead_tg_min": lead, "lead_gap_h": lead_gap_h, "s_h": s_h}


def pred_constraints_t(out: dict, limit_ms: Tensor, lead_gap_h: Tensor, s_h: Tensor) -> dict:
    """The model's OWN constraints (detached) -- the self-consistency target."""
    gp = out["goal_prob"].detach()
    cons = out["cons_pred"].detach()
    tl = (gp[:, TURN_L_I] > 0.5) & (gp[:, TURN_L_I] >= gp[:, TURN_R_I])
    tr = (gp[:, TURN_R_I] > 0.5) & ~tl
    follow = out["lon_logits"].detach().argmax(-1) == LON_FOLLOW
    nan = torch.full_like(limit_ms, float("nan"))
    return {"limit_ms": limit_ms, "v_hi_ms": cons[:, CI["v_hi_ms"]], "stop_active": gp[:, STOP_I] > 0.5,
            "d_stop": cons[:, CI["lon_d_reach_m"]], "t_stop": cons[:, CI["lon_t_reach_s"]],
            "turn_side": torch.where(tl, 1, torch.where(tr, -1, 0)), "turn_t_start": cons[:, CI["turn_t_start_s"]],
            "turn_t_end": cons[:, CI["turn_t_end_s"]],
            "lead_tg_min": torch.where(follow, cons[:, CI["lead_tg_min_s"]], nan),
            "lead_gap_h": lead_gap_h, "s_h": s_h}


# ------------------------------------------------------------------------------------------------ modules
class Reader(nn.Module):
    """Learned queries cross-attend [cond tokens ; scene tokens]. ``uses_scene=False`` BUILDS no scene path."""

    def __init__(self, d, n_q, heads, ff, depth, uses_scene: bool):
        super().__init__()
        self.uses_scene = bool(uses_scene)
        self.q = nn.Parameter(torch.randn(n_q, d) * 0.02)
        layer = nn.TransformerDecoderLayer(d, heads, ff, dropout=0.0, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, depth)
        self.norm = nn.LayerNorm(d)

    def forward(self, cond: Tensor, scene: Tensor | None, pad: Tensor | None) -> Tensor:
        B = cond.shape[0]
        if self.uses_scene and scene is not None:
            mem = torch.cat([cond, scene], 1)
            mp = torch.cat([torch.zeros(B, cond.shape[1], dtype=torch.bool, device=cond.device), pad], 1)
        else:
            mem, mp = cond, None
        q = self.q[None].expand(B, -1, -1) + cond.mean(1, keepdim=True)
        return self.norm(self.dec(q, mem, memory_key_padding_mask=mp))


def _mlp(d_in, d_hid, d_out, zero_last=False):
    m = nn.Sequential(nn.Linear(d_in, d_hid), nn.GELU(), nn.Linear(d_hid, d_out))
    if zero_last:
        nn.init.zeros_(m[2].weight)
        nn.init.zeros_(m[2].bias)
    return m


class WMAux(nn.Module):
    """M2: the H-JEPA world-model auxiliary (``tanitad.models.hjepa``), levels z1 / z2 / z3 on the O / T / S cycles
    (0.2 s / 0.4 s / 0.8 s on the cache grid -- the nearest multiples to 0.1 / 0.5 / 1.0 s). a = without IDM
    (gamma 0), b = with IDM (beta 0.08, gamma 100)."""

    def __init__(self, cfg: V8FMiniConfig):
        super().__init__()
        from tanitad.models.hjepa import HJEPAStack, LevelConfig   # S3's module; absent => this arm cannot be built
        gamma = cfg.wm_gamma if cfg.wm == "b" else 0.0
        mk = lambda **kw: LevelConfig(latent_dim=cfg.wm_dim, beta=cfg.wm_beta, gamma=gamma, pred_width=128,
                                      pred_depth=2, pred_heads=4, enc_hidden=128, macro_hidden=32, **kw)
        lv = [mk(action_dim=2, act_embed_dim=32),
              mk(action_dim=32, stride=cfg.wm_strides[0]), mk(action_dim=32, stride=cfg.wm_strides[1])]
        self.stack = HJEPAStack(lv, sigreg_projections=256)
        self.proj = nn.Sequential(nn.LayerNorm(cfg.d), nn.Linear(cfg.d, cfg.wm_dim))
        self.gamma = gamma
        self.ch1_scale = CH1_SCALE[cfg.kin_mode]

    def z1(self, adapt: Tensor) -> Tensor:
        """Level-1 latent of every cached frame: the mean-pooled adapter token, projected. ``adapt`` [B, F, n_tok, d] -> [B, F, D].
        ONE spelling: ``forward`` (training) and rung beta's planner (``tanitad.models.v8f_beta``) both call it."""
        return self.proj(adapt.mean(2))

    def scale_actions(self, act: Tensor) -> Tensor:
        """Raw action channels ``act`` [..., 2] = (a [m/s^2], ch1) -> the WM's action units. ch1 = the recorded steer (kin_mode steer) or
        the MEASURED yaw rate (kin_mode yaw, PI ruling D-PI-YAWRATE-1). ONE spelling for training and for the beta planner."""
        return torch.stack([act[..., 0] / WM_ACC_SCALE, act[..., 1] / self.ch1_scale], -1)

    def forward(self, adapt: Tensor, act: Tensor) -> dict:
        z1 = self.z1(adapt)                                                     # [B, F, D]
        a = self.scale_actions(act)[:, :-1]                                     # action of the transition j -> j+1
        return self.stack.loss(z1, a)


class V8FMini(nn.Module):
    def __init__(self, cfg: V8FMiniConfig, prior: P10Prior, cons_mu=None, cons_sd=None):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.prior = prior
        self.register_buffer("cons_mu", torch.zeros(N_CONS) if cons_mu is None else torch.as_tensor(cons_mu, dtype=torch.float32))
        self.register_buffer("cons_sd", torch.ones(N_CONS) if cons_sd is None else torch.as_tensor(cons_sd, dtype=torch.float32))
        self.register_buffer("ck_nom", torch.tensor([[a, 0.0, 0.0] for a in CKPT_ARCS]))
        self.register_buffer("ck_scale", torch.tensor([10.0, 10.0, 0.5]))
        # scene adapter (shared) -- the only place frozen features meet trainable weights
        self.adapter = nn.Sequential(nn.LayerNorm(cfg.d_feat), nn.Linear(cfg.d_feat, d))
        self.pos = nn.Parameter(torch.randn(cfg.n_tok, d) * 0.02)
        self.time = nn.Parameter(torch.randn(cfg.hist, d) * 0.02)
        # conditioning tokens
        self.e_ego, self.e_nav, self.e_lim = nn.Linear(8, d), nn.Linear(5, d), nn.Linear(5, d)
        self.e_age = nn.Linear(2, d)
        self.e_ck, self.e_code = nn.Linear(12, d), nn.Linear(CODE_DIM, d)
        sc = not cfg.navonly
        self.read_s = Reader(d, cfg.n_q, cfg.heads, cfg.ff, cfg.depth_s, uses_scene=sc)
        self.read_t = Reader(d, cfg.n_q, cfg.heads, cfg.ff, cfg.depth_t, uses_scene=sc)
        self.read_o = Reader(d, cfg.n_q, cfg.heads, cfg.ff, cfg.depth_o, uses_scene=True)
        fl = cfg.n_q * d
        self.head_s = _mlp(fl, 2 * d, 9)
        self.trunk_t = _mlp(fl, 2 * d, d)
        self.h_goal, self.h_lat, self.h_lon, self.h_cons = (nn.Linear(d, N_GOAL), nn.Linear(d, N_LAT),
                                                            nn.Linear(d, N_LON), nn.Linear(d, N_CONS))
        self.head_o = _mlp(fl, 2 * d, 2 * cfg.nb, zero_last=True)         # zero-init: the residual STARTS at P10
        i = torch.arange(N60, dtype=torch.float32)
        basis = torch.stack([torch.cos(k * math.pi * (i + 0.5) / N60) for k in range(cfg.nb)], 1)    # [60, nb]
        self.register_buffer("basis", basis * ((i + 1) / N60)[:, None])       # ramp: the residual vanishes at tau -> 0
        self.wm = WMAux(cfg) if cfg.wm != "off" else None
        self.p_gt = 1.0

    # ---- helpers -------------------------------------------------------------------------------------------
    def _ego(self, f8):
        return self.e_ego(self.prior.standardise(f8))

    @staticmethod
    def _at(x: Tensor, idx: Tensor) -> Tensor:                     # x [B, F, ...] -> x[b, idx[b]]
        return x[torch.arange(x.shape[0], device=x.device), idx]

    def _scene(self, adapt: Tensor, ok: Tensor, last: Tensor):
        """tokens [B, hist*n_tok, d] (+ pad mask) of the `hist` frames ending at index `last` [B]."""
        cfg = self.cfg
        pos = last[:, None] - (cfg.hist - 1 - torch.arange(cfg.hist, device=adapt.device))[None]
        g = torch.gather(adapt, 1, pos[:, :, None, None].expand(-1, -1, adapt.shape[2], adapt.shape[3]))
        g = g + self.time[None, :, None, :]
        pad = ~torch.gather(ok, 1, pos)
        return g.flatten(1, 2), pad[:, :, None].expand(-1, -1, adapt.shape[2]).flatten(1)

    def _code_pred(self, out: dict) -> Tensor:
        return torch.cat([out["goal_prob"], F.softmax(out["lat_logits"], -1), F.softmax(out["lon_logits"], -1),
                          out["cons_std"]], -1)

    def _code_gt(self, lab: dict) -> Tensor:
        y, w = lab["goal_y"], lab["goal_w"]
        g = y * w + 0.5 * (1 - w)
        def al(m, n):
            m = m.float()
            s = m.sum(-1, keepdim=True)
            return torch.where(s > 0, m / s.clamp_min(1.0), torch.full_like(m, 1.0 / n))
        c = torch.nan_to_num((lab["cons"] - self.cons_mu) / self.cons_sd, nan=0.0)
        return torch.cat([g, al(lab["lat_al"], N_LAT), al(lab["lon_al"], N_LON), c], -1)

    def _ck_cond(self, ck: Tensor, ok: Tensor) -> Tensor:
        x = torch.cat([ck[..., :2] / 50.0, ck[..., 2:3], ok[..., None].to(ck.dtype)], -1)
        return self.e_ck(x.flatten(1))

    # ---- the three layers -----------------------------------------------------------------------------------
    def run_s(self, b, adapt, last_s, age_rel=None):
        cfg = self.cfg
        i = last_s
        cond = torch.stack([self._ego(self._at(b["p10f"], i)), self.e_nav(self._at(b["nav"], i)),
                            self.e_lim(torch.cat([b["limit"], b["limit_ok"][:, None].float()], -1))], 1)
        sc, pad = (self._scene(adapt, b["frame_ok"], i) if not cfg.navonly else (None, None))
        h = self.read_s(cond, sc, pad).flatten(1)
        o = self.head_s(h).reshape(-1, 3, 3)
        ck = self.ck_nom[None] + o.float() * self.ck_scale
        return ck                                                          # [B, 3, 3] in S's tick-frame

    def run_t(self, b, adapt, last_t, ck_t, ck_ok):
        cfg = self.cfg
        i = last_t
        cond = torch.stack([self._ego(self._at(b["p10f"], i)), self.e_nav(self._at(b["nav"], i)),
                            self.e_lim(torch.cat([b["limit"], b["limit_ok"][:, None].float()], -1)),
                            self._ck_cond(ck_t, ck_ok)], 1)
        sc, pad = (self._scene(adapt, b["frame_ok"], i) if not cfg.navonly else (None, None))
        h = F.gelu(self.trunk_t(self.read_t(cond, sc, pad).flatten(1)))
        cs = self.h_cons(h).float()
        out = {"goal_logits": self.h_goal(h).float(), "lat_logits": self.h_lat(h).float(),
               "lon_logits": self.h_lon(h).float(), "cons_std": cs}
        out["goal_prob"] = torch.sigmoid(out["goal_logits"])
        out["cons_pred"] = cs * self.cons_sd + self.cons_mu
        return out

    def run_o(self, b, adapt, last_o, ck_o, ck_ok, code):
        cfg = self.cfg
        i = last_o
        ages = torch.stack([b["a_t"].float() * DT, b["a_s"].float() * DT], -1)
        cond = torch.stack([self._ego(self._at(b["p10f"], i)), self.e_nav(self._at(b["nav"], i)),
                            self.e_lim(torch.cat([b["limit"], b["limit_ok"][:, None].float()], -1)),
                            self._ck_cond(ck_o, ck_ok), self.e_code(code), self.e_age(ages)], 1)
        sc, pad = self._scene(adapt, b["frame_ok"], i)
        h = self.read_o(cond, sc, pad).flatten(1)
        with torch.autocast(device_type=h.device.type, enabled=False):          # geometry + the prior stay fp32
            coef = self.head_o(h.float()).reshape(-1, 2, cfg.nb) * cfg.res_scale_m      # [B, 2, nb]
            res = torch.einsum("tk,bck->btc", self.basis, coef)                # [B, 60, 2]
            p30 = self.prior(self._at(b["p10f"], i).float(), b["base"].float())
            return upsample_prior(p30) + res, p30

    # ---- forward ---------------------------------------------------------------------------------------------
    def _encode(self, b):
        adapt = self.adapter(b["feats"].float()) + self.pos[None, None]            # [B, F, n_tok, d]
        return adapt

    def _code_mix_mask(self, has: Tensor, p_gt: float, gen=None) -> Tensor:
        """PREREG_V8F_MIXO, ``code_mix = kind``: [B, CODE_DIM] bool, True where O reads the GT code's value instead of T's own. ONE Bernoulli(p_gt) draw per sample AND per
        unit of :data:`CODE_MIX_UNITS` (a [B, N_CODE_UNITS] draw, independent), broadcast to the unit's indices through :data:`CODE_UNIT_OF`; a window with no T label
        (``has`` False) reads T's own code in every unit. Every cell's marginal GT probability is p_gt (``whole``'s); only the joint changes."""
        draw = torch.rand(has.shape[0], N_CODE_UNITS, generator=gen, device=has.device) < p_gt
        unit = torch.as_tensor(CODE_UNIT_OF, dtype=torch.long, device=has.device)
        return (draw & has[:, None])[:, unit]

    def _forward(self, b, p_gt: float, use_labels: bool, gen=None, code_override: Tensor | None = None):
        cfg = self.cfg
        B = b["feats"].shape[0]
        dev = b["feats"].device
        Fn = b["feats"].shape[1]
        adapt = self._encode(b)
        last_now = torch.full((B,), Fn - 1, dtype=torch.long, device=dev)
        last_t, last_s = last_now - b["a_t"], last_now - b["a_s"]
        out = {}
        # ---- S at its tick
        ck_s = self.run_s(b, adapt, last_s)
        out["ckpt"] = ck_s
        ck_ok_pred = torch.ones(B, 3, device=dev)
        ck_src, ck_ok = ck_s, ck_ok_pred
        if use_labels and p_gt > 0:
            use = (torch.rand(B, generator=gen, device=dev) < p_gt)[:, None] & b["lab_s"]["rc_ok"] & b["lab_s"]["has"][:, None]
            ck_src = torch.where(use[..., None], b["lab_s"]["rc"], ck_s)
        if cfg.detach_upper:
            ck_src = ck_src.detach()
        ck_for_t = transform_ckpt(ck_src, b["pose_s"], b["pose_t"])
        # ---- T at its tick
        tout = self.run_t(b, adapt, last_t, ck_for_t, ck_ok)
        out.update(tout)
        code = self._code_pred(tout)
        if use_labels and p_gt > 0:
            if cfg.code_mix == "kind":                                  # PREREG_V8F_MIXO: an independent GT-or-own switch per sample AND per unit
                code = torch.where(self._code_mix_mask(b["lab_t"]["has"], p_gt, gen), self._code_gt(b["lab_t"]), code)
            else:                                                       # whole: the pre-MIXO lines, verbatim (the RNG stream is B's)
                use_t = (torch.rand(B, generator=gen, device=dev) < p_gt) & b["lab_t"]["has"]
                code = torch.where(use_t[:, None], self._code_gt(b["lab_t"]), code)
        if code_override is not None:                                   # forward_diag only (PREREG_V8F_TAC)
            if code_override.shape != code.shape:
                raise ValueError(f"code_override {tuple(code_override.shape)} != T's code {tuple(code.shape)}")
            code = code_override.to(device=code.device, dtype=code.dtype)
        if cfg.detach_upper:
            code = code.detach()
        out["code"] = code                                               # the code O actually read (o_interface full); under cut, the code OFFERED to the interface (T's, after every override): O reads zeros there
        ck_for_o = transform_ckpt(ck_src, b["pose_s"], b["pose_now"])
        if cfg.o_interface == "cut":                                     # PREREG_V8F_FLAT: O reads a CONSTANT for both of its hierarchical inputs; T has already read S (run_t above), so only the wires INTO O are cut
            ck_for_o = self.ck_nom[None].expand(B, -1, -1)               # S's checkpoints for O = the nominal arcs ...
            ck_ok = torch.zeros_like(ck_ok)                              # ... flagged absent (ok = 0)
            code = torch.zeros_like(code)                                # T's code = zeros, AFTER every override (the teacher forcing and forward_diag's code_override)
        traj60, p30 = self.run_o(b, adapt, last_now, ck_for_o, ck_ok, code)
        out["traj60"], out["traj30"], out["p10"] = traj60, subsample_02(traj60), p30
        out["adapt"] = adapt
        return out

    def forward_infer(self, batch: dict) -> dict:
        """INFERENCE: reads only :data:`INFERENCE_INPUT_KEYS` -- no label, no GT, no situation classifier."""
        g = _KeyGuard(batch)
        out = self._forward(g, 0.0, False)
        out["_keys_read"] = set(g.read)
        return out

    def forward_diag(self, batch: dict, code_override: Tensor) -> dict:
        """DIAGNOSTIC ONLY (PREREG_V8F_TAC): inference exactly as :meth:`forward_infer`, except that O reads ``code_override``
        [B, CODE_DIM] in place of T's held output: a DONOR window's predicted code (TAC-USED: does O use T at all?) or the GT code
        (TAC-HEADROOM: what a perfect tactical layer would buy; a T0 oracle, never a result). The batch is read through the same
        key guard, so the model still sees only :data:`INFERENCE_INPUT_KEYS`; any label in the override was read by the EVALUATOR.
        Never called by training or by :meth:`forward_infer`."""
        g = _KeyGuard(batch)
        out = self._forward(g, 0.0, False, code_override=code_override)
        out["_keys_read"] = set(g.read)
        return out

    def forward(self, batch: dict, p_gt: float | None = None, gen=None) -> dict:
        return self._forward(batch, self.p_gt if p_gt is None else p_gt, True, gen)

    # ---- losses ----------------------------------------------------------------------------------------------
    def loss(self, batch: dict, p_gt: float | None = None, gen=None) -> dict:
        cfg = self.cfg
        out = self.forward(batch, p_gt, gen)
        T = {}
        B = batch["feats"].shape[0]
        # S: L1 on position + heading, at S's tick frame
        ls, lt = batch["lab_s"], batch["lab_t"]
        ok = (ls["rc_ok"] & ls["has"][:, None]).float()
        e = (out["ckpt"] - ls["rc"]).abs() / self.ck_scale
        e = torch.cat([e[..., :2].sum(-1, keepdim=True), e[..., 2:3]], -1).sum(-1)
        T["ckpt"] = (e * ok).sum() / ok.sum().clamp_min(1.0)
        # T: masked BCE / partial-label CE / Huber at T's tick frame
        w = lt["goal_w"] * lt["has"][:, None].float()
        bce = F.binary_cross_entropy_with_logits(out["goal_logits"], lt["goal_y"], reduction="none")
        T["goal"] = (bce * w).sum() / w.sum().clamp_min(1.0)
        T["lat"] = self._partial_ce(out["lat_logits"], lt["lat_al"], lt["has"])
        T["lon"] = self._partial_ce(out["lon_logits"], lt["lon_al"], lt["has"])
        tgt = (lt["cons"] - self.cons_mu) / self.cons_sd
        if cfg.cons_absent == "zero":                # PREREG_V8F_CZ: an absent constraint's target is the 0 O reads under teacher forcing
            m = lt["has"][:, None].expand_as(tgt)
            tgt = torch.where(torch.isfinite(tgt), tgt, torch.zeros_like(tgt))
        else:
            m = torch.isfinite(tgt) & lt["has"][:, None]
        hub = F.huber_loss(out["cons_std"], torch.nan_to_num(tgt), reduction="none", delta=1.0)
        T["cons"] = (hub * m).sum() / m.sum().clamp_min(1.0)
        # O: ADE / FDE on the 0.2 s knots
        gt, gok = batch["gt"], batch["gt_ok"].float()
        T["ade"], T["fde"] = traj_terms(out["traj30"], gt, gok)      # the same two lines as before, now ONE spelling that PREREG_V8F_OACC's w_acc rule also calls
        T["acc"] = accel_term(out["traj30"], gt, gok)                # PREREG_V8F_OACC: always LOGGED (the G-LIVE probe and the reader read it); in the total only when w_acc > 0
        # consistency penalties
        lim = limit_ms_from_onehot(batch["limit"], batch["limit_ok"])
        if cfg.consistency:
            cg = gt_constraints_t(batch["lab"], lim, batch["lead_gap_h"], batch["s_h"])
            pg = consistency_penalties(out["traj30"], cg)
            cs = pred_constraints_t(out, lim, batch["lead_gap_h"], batch["s_h"])
            ps = consistency_penalties(out["traj30"], cs)
            for r in RULES:
                T[f"pen_{r}"] = (pg["pen"][r] / PEN_SCALE[r]).mean()
                T[f"pens_{r}"] = (ps["pen"][r] / PEN_SCALE[r]).mean()
            out["viol"] = pg["viol"]
        if self.wm is not None:
            wm = self.wm(out["adapt"], batch["act"])
            T["wm"] = wm["total"]
        total = (cfg.w_ckpt * T["ckpt"] + cfg.w_tac * (T["goal"] + 0.5 * (T["lat"] + T["lon"]) + cfg.w_cons * T["cons"])
                 + cfg.w_traj * (T["ade"] / 3.0 + 0.5 * T["fde"] / 6.0))
        total = total + cfg.w_acc * T["acc"] if cfg.w_acc > 0.0 else total          # PREREG_V8F_OACC: ONE line (a mutation deletes it); w_acc 0 adds nothing: B's total, bit for bit
        if cfg.consistency:
            total = total + cfg.w_pen * sum(T[f"pen_{r}"] for r in RULES) + cfg.w_pen_self * sum(T[f"pens_{r}"] for r in RULES)
        if self.wm is not None:
            total = total + cfg.w_wm * T["wm"]
        T["total"] = total
        out["terms"] = T
        return out

    @staticmethod
    def _partial_ce(logits: Tensor, allowed: Tensor, has: Tensor) -> Tensor:
        """-log sum_{c in allowed} p_c ; rows with an empty allowed set (IGNORE) or no label are masked."""
        valid = allowed.any(-1) & has
        lse = torch.logsumexp(logits, -1)
        lsa = torch.logsumexp(logits.masked_fill(~allowed, float("-inf")), -1)
        l = torch.where(valid, lse - lsa, torch.zeros_like(lse))
        return l.sum() / valid.float().sum().clamp_min(1.0)

    # ---- bookkeeping ------------------------------------------------------------------------------------------
    def trainable_groups(self) -> dict:
        g = {"adapter": [self.adapter, self.pos, self.time], "cond": [self.e_ego, self.e_nav, self.e_lim, self.e_age,
             self.e_ck, self.e_code], "S": [self.read_s, self.head_s], "T": [self.read_t, self.trunk_t, self.h_goal,
             self.h_lat, self.h_lon, self.h_cons], "O": [self.read_o, self.head_o]}
        if self.wm is not None:
            g["WM"] = [self.wm]
        return g

    def built_levers(self) -> dict:
        """What the BUILT model does (the G-DVB 'built' side)."""
        cfg = self.cfg
        gp = lambda mods: int(sum(p.numel() for m in mods for p in (m.parameters() if isinstance(m, nn.Module) else [m])
                                  if p.requires_grad))
        return {"navonly": (not self.read_s.uses_scene) and (not self.read_t.uses_scene) and self.read_o.uses_scene,
                "consistency": bool(cfg.consistency), "wm": "off" if self.wm is None else ("b" if self.wm.gamma > 0 else "a"),
                "cons_absent": cfg.cons_absent, "code_mix": cfg.code_mix,        # PREREG_V8F_MIXO: the teacher-forcing granularity the BUILT model's _forward reads
                "o_interface": cfg.o_interface,                                # PREREG_V8F_FLAT: what the BUILT model's _forward hands O (full: S's checkpoints + T's code; cut: constants)
                "w_acc": cfg.w_acc,                                            # PREREG_V8F_OACC: the weight the BUILT loss gives the acceleration term (0.0 = not in the total)
                "wm_idm_gamma": 0.0 if self.wm is None else float(self.wm.gamma),
                "kin_mode": self.cfg.kin_mode, "wm_ch1_scale": None if self.wm is None else float(self.wm.ch1_scale),
                "p_gt_start": cfg.p_gt_start, "p_gt_end": cfg.p_gt_end,      # PREREG_V8F_PGT: the scheduled-sampling schedule the train loop reads
                "hist": int(self.time.shape[0]), "hmax": cfg.hmax,           # PREREG_V8F_HIST (F-3): the BUILT history (the per-frame time table's rows) and batch depth
                "window_start": cfg.window_start_effective,                  # Amendment 1: the EFFECTIVE first-window step (hist - 1 when the flag is absent)
                "prior_frozen": not any(p.requires_grad for p in self.prior.parameters()),
                "residual_zero_init": bool(float(self.head_o[2].weight.detach().abs().sum()) == 0.0),
                "params": {k: gp(v) for k, v in self.trainable_groups().items()},
                "n_params_trainable": int(sum(p.numel() for p in self.parameters() if p.requires_grad)),
                "w": {"w_pen": cfg.w_pen if cfg.consistency else 0.0, "w_pen_self": cfg.w_pen_self if cfg.consistency else 0.0,
                      "w_wm": cfg.w_wm if self.wm is not None else 0.0,
                      "w_tac": cfg.w_tac, "w_cons": cfg.w_cons}}                       # PREREG_V8F_WC: T's constraint term is w_tac x w_cons


class _KeyGuard(dict):
    """A dict that records which keys the model reads (the signature test for the admissibility rule)."""

    def __init__(self, d):
        super().__init__(d)
        self.read = set()

    def __getitem__(self, k):
        self.read.add(k)
        return super().__getitem__(k)

    def __contains__(self, k):
        self.read.add(k)
        return super().__contains__(k)


def p_gt_schedule(step: int, total: int, p0: float = 1.0, p1: float = 0.2) -> float:
    """Scheduled sampling: linear p_GT from `p0` to `p1` over the run (PREREG: 1.0 -> 0.2)."""
    return float(p0 + (p1 - p0) * min(max(step / max(total, 1), 0.0), 1.0))


def cons_stats(join: dict) -> tuple:
    """Mean / std of the 26 constraint cells over labelled rows (finite entries only), for the standardised head."""
    import warnings
    c = join["cons"][join["has"]].astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                   # all-NaN columns -> mu 0, sd 1 below
        mu = np.nanmean(c, 0)
        sd = np.nanstd(c, 0)
    return np.where(np.isfinite(mu), mu, 0.0).astype(np.float32), np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0).astype(np.float32)


# ------------------------------------------------------------------------------------------------ admissibility
_FORBIDDEN_TOKENS = ("situation", "sitclf", "scenario_cls", "head_img")


def scan_source_for_forbidden(src: str) -> list:
    """Forbidden tokens among the identifiers / attributes / arguments / string subscripts of the CODE in ``src``
    (docstrings and comments are not code)."""
    names = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Name):
            names.add(n.id.lower())
        elif isinstance(n, ast.Attribute):
            names.add(n.attr.lower())
        elif isinstance(n, ast.arg):
            names.add(n.arg.lower())
        elif isinstance(n, ast.Subscript) and isinstance(getattr(n, "slice", None), ast.Constant) and isinstance(n.slice.value, str):
            names.add(n.slice.value.lower())
    return sorted(t for t in _FORBIDDEN_TOKENS if any(t in nm for nm in names if nm != "_forbidden_tokens"))


def assert_no_situation_classifier_input(model: V8FMini | None = None, batch: dict | None = None) -> dict:
    """Binding rule of 2026-08-03: the goal path and the situation path are information-disjoint at inference.

    Checks (raises AssertionError): (1) this module's CODE (not its docstrings/comments) names no situation
    classifier; (2) ``forward_infer`` takes exactly one data argument, ``batch``; (3) if a model + batch are given,
    the keys ``forward_infer`` actually READS are a subset of :data:`INFERENCE_INPUT_KEYS`, and none of them is a
    label key (``lab*``) or contains a forbidden token."""
    bad = scan_source_for_forbidden(inspect.getsource(inspect.getmodule(V8FMini)))
    assert not bad, f"situation-classifier tokens in code: {bad}"
    sig = list(inspect.signature(V8FMini.forward_infer).parameters)
    assert sig == ["self", "batch"], f"forward_infer signature {sig}"
    keys = set()
    if model is not None and batch is not None:
        was = model.training
        model.eval()
        with torch.no_grad():
            keys = model.forward_infer(batch)["_keys_read"]
        model.train(was)
        extra = keys - INFERENCE_INPUT_KEYS
        assert not extra, f"inference reads keys outside the admissible set: {sorted(extra)}"
        assert not any(k.startswith("lab") or any(t in k for t in _FORBIDDEN_TOKENS) for k in keys)
    return {"allowed": sorted(INFERENCE_INPUT_KEYS), "read": sorted(keys)}
