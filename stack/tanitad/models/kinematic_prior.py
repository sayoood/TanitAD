"""refcv7 NEW-1 -- THE CAUSAL KINEMATIC PRIOR ``P`` and THE ONE FUNCTION THAT
TURNS A RESIDUAL INTO THE PLAN.

SPEC: ``Project Steering/SPEC_REFCV7.md`` section 1, NEW-1 (registered 2026-09-26):
*"the plan is a residual on a causal kinematic prior ... The planner denoises the
residual Delta, with anchors defined in residual space; the plan is P + Delta ...
The exact prior used by the battery's echo (``ha0_ext``) is to be matched and
cited file:line."*

============================================================================
WHAT THE BATTERY'S ECHO ``ha0_ext`` IS -- read at source (tip 59f0d46)
============================================================================
The refcv6 standard battery produces ``ha0_ext`` through ``refcv3_arm.py``
(``FlyWheels/.../2026-09-23-refcv6-standard-tests/battery/code/refcv6_roll.py:48``
loads it by path; ``refcv6_panel.py:548-558`` repeats the SAME two calls for the
6 s surface):

* ``taniteval/tools/refcv3_arm.py:2081-2082``  ``v_ep = ep.poses[:, 3]``,
  ``kap_ep = ep.actions[:, 0]`` -- the RECORDED STEER channel
  (``stack/tanitad/data/physicalai.py:621,632``: ``steer = arctan(2.9 * curvature)``);
* ``refcv3_arm.py:2087,2101``  ``t0 = t + W - 1`` (the last OBSERVED frame),
  ``v0 = float(pose_last[3])``;
* ``refcv3_arm.py:2125-2128``  ``ext = ra.hold_ext_controls(None, v_ep, kap_ep,
  t0, dt=0.1, stride=1)`` then ``integrate_select(ext[None].expand(n_f, 2), v0,
  grid, action_units="steer")``;
* ``taniteval/tools/refav1_arm.py:445-446``  ``a0 = (v[i0] - v[i0-1]) / dt`` and
  the control pair ``[a0, kap[i0]]``;
* ``refcv3_arm.py:1677-1692`` -> ``refav1_arm.py:471-493`` ->
  ``stack/tanitad/refs/refa_v1_plan.py:344-373``: ``kappa = tan(steer)/2.9``
  (``stack/tanitad/models/kinematic.py:57-59``), then ``rollout_unicycle`` from
  ``(0, 0, 0, v0)`` at ``dt = 0.1`` (``kinematic.py:220-264``: forward Euler,
  ``yaw += v*kappa*dt``, ``v = max(v + a*dt, 0)`` LAST), index-selected at the
  grid instants.

=> ``ha0_ext`` = CONSTANT LONGITUDINAL ACCELERATION ``a0`` + CONSTANT CURVATURE
``kappa0``, forward Euler at 0.1 s, speed clamped at zero.

(( !! )) IT IS NOT A CONSTANT-VELOCITY / CONSTANT-YAW-RATE MODEL, which is how
the SPEC and the brief describe it. The two coincide only when ``a0 == 0``:
with ``a0 != 0`` the speed changes and the yaw rate ``v * kappa0`` changes with
it. This module implements BOTH readings under distinct, honest names and the
report says which one refcv7 must set -- a flag value whose name misdescribes
its maths is exactly the class of defect ``flagship4b-phase0-30k`` taught.

============================================================================
THE MODES (``--residual-prior``)
============================================================================
``off``           no prior. Nothing in this module is reached; the decoder is
                  the refcv6 decoder bit for bit.
``ha0_ext``       THE ECHO, EXACTLY: ``a0 = (v[t0] - v[t0-1]) / dt`` and
                  ``kappa0 = tan(steer[t0]) / 2.9``. Needs the observed window's
                  RECORDED ACTIONS, an input the refcv6 model does not consume.
``ha0_ext_pose``  the echo's CONSTRUCTION with a POSE-TRACK curvature: the same
                  ``a0`` (bit-identical -- it is the ego-history encoder's own
                  backward difference, ``ego_history.py:96``) and
                  ``kappa0 = omega0 / max(v0, POSE_KAPPA_V_FLOOR)``, clamped to
                  ``+-POSE_KAPPA_CAP``, with ``omega0`` the encoder's own backward
                  yaw difference (``ego_history.py:98``). Reads ONLY the pose
                  window refcv6 already requires at every call site.
``cv_yawrate``    the SPEC's literal words: ``a = 0``, ``kappa0`` as
                  ``ha0_ext_pose``. Constant speed ``v0`` and, because the speed
                  is constant, constant yaw rate ``v0 * kappa0``.
``p10``           (v8F rung alpha, arm alpha-1; ``p10_prior.py``) the FROZEN learned
                  kinematic floor P10: a 30-knot, 6 s path, converted to PER-TICK
                  controls ``(a_j, kappa_j)`` so the SAME ``compose_ticks`` /
                  ``roll_plan`` / ``absolute_controls`` serve it. Reads the pose
                  window AND the recorded steer at t0 (``ACTION_MODES``, like
                  ``ha0_ext``). A per-tick prior is a ``[B, T]`` tensor where every
                  other mode carries a ``[B]`` constant -- see "THE PER-TICK PRIOR".
``p10_pose``      (v8F rung alpha, arm alpha-1, the POSE-ONLY variant) the same frozen learned floor, refit with P10's
                  third feature ``k`` = the POSE-TRACK curvature ``kappa0`` of ``ha0_ext_pose`` (same floor / cap, same
                  ``omega0``) instead of the recorded steer, and its ``kd_x`` base built from ``ha0_ext_pose`` instead
                  of ``ha0_ext``. Reads ONLY the observed pose window (``POSE_MODES``): ``prior_controls`` REFUSES a
                  supplied ``actions`` exactly as it does for ``ha0_ext_pose``. Its ``a0`` is ``ha0_ext_pose``'s ONE-frame
                  backward difference. The only thing that differs from alpha-0's prior is what is LEARNED on top.

MEASURED 2026-09-26 (T1, the refcv6 battery's step-30k S2 surface: 4,754 windows / 139 eval
clips; paired episode-cluster bootstrap, n_boot 2000; package
``Research/2026-09-26-refcv7-residual-prior/raw/measure_prior_vs_echo.json``):
  ``ha0_ext``       == the BANKED battery ``ha0_ext`` on 4,754/4,754 windows, max |diff| 0.0 m;
  ``ha0_ext_pose``  ADE 0-2 s minus the echo +0.0055 m [+0.0036, +0.0076]; 0-6 s +0.0223;
  ``cv_yawrate``    ADE 0-2 s minus the echo +0.2293 m [+0.1971, +0.2642]; 0-6 s +0.5300.
The TRAIN side (train-a6, 139 train clips) replicates the ordering
(``raw/measure_anchor_train.json``).

ADMISSIBILITY (stated, not assumed; the report escalates the open half):
* ``v0`` at t0 -- PI ruling 2026-09-02 (admissible).
* past ego data (speed, acceleration, yaw rate per step) -- SPEC_REFCV6_V2
  section 10.3 citing the 2026-09-02 ruling; refcv6 already feeds it.
* "measured current speed, acceleration, yaw rate" -- PI 2026-09-03
  (``stack/tanitad/refs/refc_v3.py:144-147``, E11').
* ``kappa0`` read from the STEER channel at t0 (``ha0_ext``) -- the battery's own
  lever panel records it as *"unruled at inference"*
  (``battery/code/lever_panel.py:14-15``). The E11' block implemented "measured
  current yaw rate" exactly that way (``refc_v3.py:246-249``), which is a
  precedent, not a ruling. The two POSE modes do not need it.

============================================================================
THE COMPOSITION -- and why it is done in CURVATURE space
============================================================================
refcv6's sampler state is a per-slot CONTROL sequence ``(a_lon, a_lat|kappa)``
(``refc.py:2499``) and every path it emits is an integration of controls
(``refc.py:2430-2445``, ``refc_sampler.py:349-390``). The prior ``P`` is itself
a constant-control integration through the SAME integrator. So ``P + Delta``
is composed where both live -- in control space:

    a(t)     = a0     + delta_a(t)
    kappa(t) = kappa0 + alat_to_curvature(delta_alat(t), v0)   ("alat" units)
             = kappa0 + delta_kappa(t)                          ("kappa" units)

and the plan is ``rollout_unicycle`` of that from ``v0``. The residual's lateral
channel is converted with the vocabulary's OWN ``alat_to_curvature``
(``refc_sampler.py:312-324``, the one spelling) and the prior's curvature is
ADDED AFTER that conversion, never pushed through it: MEASURED on the train-a6
manifest (139 train clips, 27,504 frames) ``|kappa_steer| > 0.12`` -- the
vocabulary's cap -- on 1.77 % of frames (max 0.2766), so composing in ``a_lat``
would clip the echo itself on those windows and ``Delta = 0`` would no longer
BE the echo.

=> ``Delta == 0`` rolls to ``P`` EXACTLY (``a0 + 0.0 == a0``,
``kappa0 + 0.0 == kappa0``), through the identical integrator call the battery
makes. Pinned by ``stack/tests/test_residual_prior.py``.

=> ``roll_plan`` is THE ONE FUNCTION that turns a residual into a path. A
consumer that rolls a residual any other way receives ``Delta`` as if it were
the plan; the decoder therefore exports ABSOLUTE controls (``absolute_controls``)
and never lets the residual state leave it.

============================================================================
THE PER-TICK PRIOR (mode ``p10``) -- what changes, and what is pinned not to
============================================================================
Every mode above is ONE constant ``(a0, kappa0)`` pair per row. P10 is a whole
path, so its controls differ at every 0.1 s tick. The functions below therefore
accept ``a0`` / ``kappa0`` of shape ``[B]`` (a constant: the code path is the
unchanged one, bit for bit) OR ``[B, T]`` (per tick):

* ``compose_ticks`` / ``roll_plan``: the residual is held over its slot span by
  ``controls_to_ticks`` as before and the prior is ADDED PER TICK afterwards. For
  a constant prior this is the same float32 sum the slot-wise form computes
  (``(a0 + d)`` held == ``held(d) + a0``), pinned bit-for-bit by
  ``test_residual_prior.py``.
* ``absolute_controls`` / ``residual_controls`` need the prior AT SLOT level
  (``u0_hat`` is per slot): that is the MEAN of the per-tick prior over the slot's
  span (``slot_prior``; the ``horizons`` are then REQUIRED). The plan itself is
  always rolled with the per-tick prior, so ``Delta == 0`` still rolls EXACTLY to
  the prior path ``P10u``; the slot mean only fixes what ``u0_hat`` is.
* ``prior_path`` rolls the per-tick prior from ``v0`` -> ``P10u``.

``P10u`` is P10's path RE-ROLLED through ``rollout_unicycle`` from the MEASURED
``v0``: :func:`controls_from_path` is the CLOSED-LOOP inverse (it tracks the
rolled state, so the entry-speed mismatch the open-loop
``kinematic.unicycle_controls_from_path`` documents does not accumulate into a
constant speed offset). ``ADE(P10u, P10)`` is MEASURED and reported, not assumed
(``Research/2026-10-20-v8f-rung-alpha/RESULT.md``).
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor

from tanitad.channel_admissibility import ChannelExclusion
from tanitad.models.ego_history import ego_channels_from_poses
from tanitad.models.kinematic import (STEER_WHEELBASE_M, kappa_of_steer,
                                      rollout_unicycle)

__all__ = [
    "RESIDUAL_PRIOR_MODES", "RESIDUAL_PRIOR_OFF", "POSE_MODES",
    "ACTION_MODES", "DT_TICK", "POSE_KAPPA_V_FLOOR", "POSE_KAPPA_CAP",
    "V0_WINDOW_TOL", "OUT_KEYS", "ECHO_SOURCE", "check_mode",
    "needs_actions", "prior_controls", "withhold", "prior_path",
    "compose_ticks", "roll_plan", "absolute_controls", "residual_controls",
    "prior_from_out", "zero_residual_index", "plan_check", "prior_stamp",
    "ResidualPriorError", "FORWARD_EXCLUSIONS", "TICK_MODES", "P10_MODE", "P10_POSE_MODE", "P10_MODES", "pose_kin3", "P10_TICKS", "P10_KAPPA_CAP",
    "knots_to_ticks", "controls_from_path", "slot_prior", "is_tick_prior", "p10_tick_prior",
]

#: ``off`` first: the default, and the value under which refcv6 builds
#: bit-identically.
RESIDUAL_PRIOR_OFF = "off"
P10_MODE = "p10"
#: the POSE-ONLY P10 (v8F alpha-1 as launched): no recorded-steer channel at inference
P10_POSE_MODE = "p10_pose"
#: the modes that carry a frozen P10 module
P10_MODES: tuple[str, ...] = (P10_MODE, P10_POSE_MODE)
RESIDUAL_PRIOR_MODES: tuple[str, ...] = (
    RESIDUAL_PRIOR_OFF, "ha0_ext", "ha0_ext_pose", "cv_yawrate", P10_MODE, P10_POSE_MODE)
#: the modes that read ONLY the observed pose window
POSE_MODES: tuple[str, ...] = ("ha0_ext_pose", "cv_yawrate", P10_POSE_MODE)
#: the modes that also read the observed window's RECORDED ACTIONS
#: (``p10`` reads the recorded steer at t0 as its kinematic feature ``k``, like ``ha0_ext``)
ACTION_MODES: tuple[str, ...] = ("ha0_ext", P10_MODE)
#: the modes whose prior is a PER-TICK control sequence ``[B, T]`` (every other mode is a constant ``[B]``)
TICK_MODES: tuple[str, ...] = P10_MODES
#: P10 covers 30 knots x 0.2 s = 6 s = 60 ticks of DT_TICK; a decoder horizon beyond it is REFUSED
P10_TICKS: int = 60

#: 10 Hz -- the v2ep provider grid, the ego-history encoder's ``dt`` and the
#: battery's ``DT_FRAME`` (``refcv3_arm.py:242``). BINDING.
DT_TICK: float = 0.1

#: Below this speed the pose track's yaw difference is noise, not a turn.
#: MEASURED on the train-a6 manifest (139 train clips, 27,504 frames; probe
#: ``2026-09-26-refcv7-residual-prior/code/probe_kappa_sources.py``): the pose
#: yaw rate against ``v * kappa_steer`` correlates 0.9979 / 0.9965 / 0.9957 /
#: 0.9969 in the 2-4 / 4-8 / 8-15 / 15-40 m/s bins and only 0.4579 / 0.2188 /
#: 0.2607 in the 0-0.5 / 0.5-1 / 1-2 m/s bins. The floor attenuates the
#: curvature estimate below 2 m/s instead of dividing noise by a small speed.
POSE_KAPPA_V_FLOOR: float = 2.0
#: A safety bound on the pose-derived curvature, ABOVE every value the steer
#: channel carries on the same 139 clips (max |kappa_steer| 0.2766 1/m).
POSE_KAPPA_CAP: float = 0.3
#: The curvature bound of the P10 -> per-tick-controls CONVERSION. NOT ``POSE_KAPPA_CAP``: that bound exists to
#: keep a noisy POSE-derived curvature plausible, and applied to P10's knot path it is the dominant source of
#: conversion error. MEASURED 2026-10-20 (eval139 S2 windows, n = 4,754; ``Research/2026-10-20-v8f-rung-alpha/raw/
#: sens_p10u.json``): ``ADE6(P10u, P10)`` mean 0.0463 m at cap 0.30 (worst window 7.44 m, 1 % of windows carry 57 %
#: of the error mass, all sharp turns at 3-7 m/s) -> 0.0339 at 0.5 -> 0.0231 at 1.0 -> 0.0154 unbounded. 1.0 1/m
#: (a 1 m radius) is a numerical-safety bound no vehicle reaches, binding on < 0.1 % of ticks.
P10_KAPPA_CAP: float = 1.0

#: The pose window's last speed must equal the ``v0`` the plan is rolled from
#: (m/s). Both are the SAME measured sample on every call site read at source
#: (trainer ``pose_hist[:, -1] == pose_last``; the battery; the NavSim bridge's
#: window interpolates EXACTLY at t0), so any gap above float noise means the
#: window ends at a different frame -- a prior built at the wrong t0.
V0_WINDOW_TOL: float = 1e-3

#: The keys a residual forward EMITS (the decoder AND ``RefCModel.forward``'s
#: pass-through). Absent -- not ``None`` -- on an ``off`` build.
OUT_KEYS: tuple[str, ...] = ("residual_prior_ctrl", "residual_prior_v",
                             "residual_prior_path")

#: The battery call this module's ``ha0_ext`` reproduces, verbatim.
ECHO_SOURCE: str = (
    "taniteval/tools/refcv3_arm.py:2125-2128 -> refav1_arm.hold_ext_controls("
    "None, v_ep, kap_ep, t0, dt=0.1, stride=1) -> integrate_select(..., "
    "action_units='steer') -> refa_v1_plan.unicycle_paths -> "
    "kinematic.rollout_unicycle")


class ResidualPriorError(ValueError):
    """The prior cannot be built as declared -- refused, never defaulted."""


# --------------------------------------------------------------------------- #
# the RL channel contract -- declared HERE because this seam owns the channel  #
# --------------------------------------------------------------------------- #
#: (( !! )) NEW-1 added ``ego_actions`` (the observed window's RECORDED
#: ``(steer, accel)``) to ``RefCModel.forward`` and ``RefCV3Model.forward``. Both
#: RL adapters DERIVE their channel set from those signatures and REFUSE an
#: undeclared channel (``tanitad.rl.refcv3_adapter.forward_conditioning_channels``;
#: ``tests/test_rl_forward_keys_cover_signature.py`` for ``refc_adapter``), and the
#: Thor full-suite gate of 2026-09-27 went RED on it in six RL test files --
#: correctly. The channel is read ONLY by ``residual_prior='ha0_ext'``
#: (:data:`ACTION_MODES`: ``ha0_ext``, which SPEC_REFCV7 section 10 (A5) EXCLUDES
#: for refcv7, and -- since v8F alpha-1 -- ``p10``, whose kinematic feature ``k`` is
#: the same recorded steer), so it is declared here, by its owner, as
#: must-not-be-plumbed: TEMPORARY, with its route back. The exclusion mechanism is build-independent, so on an
#: ``ha0_ext`` build the requirement is enforced one layer down, by the two
#: refusals the reason cites (both pinned by ``test_residual_prior.py``).
FORWARD_EXCLUSIONS: tuple[ChannelExclusion, ...] = (
    ChannelExclusion(
        channel="ego_actions",
        owner=("tanitad.models.kinematic_prior (refcv7 NEW-1 residual prior; "
               "SPEC_REFCV7 section 10, A5)"),
        reason=(
            "An RL rollout would have to take this from the logged window's "
            "RECORDED actions at t0: `ep.actions[t0]` = (atan(L * curvature), "
            "ax), read off the dataset's egomotion columns "
            "(`tanitad/data/physicalai.py::signals_at`; L = 2.9 m in the parity "
            "corpus's `const2p9` regime). The steer half at t0 is "
            "UNRULED AT INFERENCE -- the refcv6 battery's lever panel runs the "
            "echo that reads it as DIAGNOSTIC ONLY for that reason -- and "
            "SPEC_REFCV7 section 10 (A5) excludes the mode that reads it, "
            "`ha0_ext`; v8F alpha-1's `p10` (PLAN_V8F section 6) reads the SAME "
            "recorded steer as its kinematic feature `k` and is the second "
            "ACTION mode, awaiting the same ruling (DESIGN.md D1). refcv7 sets "
            "`ha0_ext_pose`, whose prior reads ONLY the "
            "past pose track, which the adapter already plumbs as "
            "`ego_poses`/`ego_n_past`. NOT a label in the goal-point sense, so "
            "the exclusion is TEMPORARY. Withholding it cannot silently change "
            "a rollout: on every build except `ha0_ext` / `p10` the channel is "
            "never read and a SUPPLIED one is REFUSED "
            "(`refc.py::RefCModel.forward`, 'would be SILENTLY DROPPED'), and "
            "on an `ha0_ext` or `p10` build the prior REFUSES to be built "
            "without it (`prior_controls`, 'Refusing rather than substituting "
            "the pose curvature'). `p10_pose` (the arm alpha-1 launches) is a "
            "POSE mode and never reads this channel. No state exists in which a "
            "withheld `ego_actions` is quietly replaced by a default."),
        unblock=(
            "A PI ruling that the recorded steer at t0 is admissible at "
            "inference (SPEC_REFCV7 section 10: it 'would make ha0_ext available "
            "in a later arm'; the same ruling admits `p10`) AND an RL arm that "
            "sets `--residual-prior ha0_ext` or `p10`. Then delete this declaration; add `ego_actions` to "
            "`tanitad.rl.refc_adapter.FORWARD_KEYS` in signature order with a "
            "CHANNEL_REQUIREMENTS record REQUIRED only when the build's "
            "`decoder.residual_prior` is in `ACTION_MODES` (the predicate "
            "machinery reads truthy dotted paths, so that needs a boolean view "
            "of the mode); declare it the same way in "
            "`tanitad.rl.refcv3_adapter.refc_channel_requirements`; and mint "
            "the window's `actions[t:t+w]` into the RL batch."),
        evidence=(
            "PUBLISHED-CODE, read 2026-09-27 at tip 12953d2: "
            "`kinematic_prior.prior_controls` (the ha0_ext refusal), "
            "`refc.py::RefCModel.forward` (the refusal of a supplied, unread "
            "`ego_actions`), `tanitad/data/physicalai.py::signals_at` (what "
            "`actions` holds), `FlyWheels/TanitAD_EvalFlyWheel/incoming/"
            "2026-09-23-refcv6-standard-tests/battery/code/lever_panel.py:14-15` "
            "('k0 at t0 admissibility is unruled at inference'), "
            "`Project Steering/SPEC_REFCV7.md` section 10 (A5, a756e81). "
            "MEASURED: `tests/test_residual_prior.py::"
            "test_refusals_at_build_and_at_forward` pins both refusals."),
        permanent=False,
        refs=("TanitAD Research Lab/Architecture & Inference/Research/"
              "2026-09-26-refcv7-residual-prior/",
              "Project Steering/SPEC_REFCV7.md"),
    ),
)


def check_mode(mode: str) -> str:
    """Return ``mode`` or raise. There is no silent fallback to ``off``."""
    if mode not in RESIDUAL_PRIOR_MODES:
        raise ResidualPriorError(
            f"residual_prior {mode!r} is not one of {RESIDUAL_PRIOR_MODES}")
    return mode


def needs_actions(mode: str) -> bool:
    """True when ``mode`` reads the observed window's recorded actions."""
    return check_mode(mode) in ACTION_MODES


def _wrap_pi(a: Tensor) -> Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _pose_kappa0(omega0: Tensor, v0: Tensor) -> Tensor:
    """The pose-track curvature of every POSE mode: ``clamp(omega0 / max(v0, POSE_KAPPA_V_FLOOR), +-POSE_KAPPA_CAP)``."""
    return (omega0 / v0.clamp_min(POSE_KAPPA_V_FLOOR)).clamp(-POSE_KAPPA_CAP, POSE_KAPPA_CAP)


def pose_kin3(poses: Tensor, n_past: int | None = None, *, dt: float = DT_TICK) -> Tensor:
    """``[B, 3]`` float64 = ``(v0, a0, kappa0)`` at the window's LAST observed frame, from the POSE track ONLY.

    The derivation is :func:`tanitad.models.ego_history.ego_channels_from_poses` (the ego-history encoder's own, which
    ``ha0_ext_pose`` calls): ``a0`` is the ONE-frame backward speed difference, ``omega0`` the ONE-frame wrapped yaw
    difference, ``kappa0`` through :func:`_pose_kappa0`. It is the feature triple of ``p10_pose``. Computed in float64
    (the fit's precision): ``ha0_ext_pose``'s own float32 wrap carries ~1e-7 rad of rounding noise per yaw difference,
    which is irrelevant to a prior but would exceed the 1e-4 m known-value tolerance at 6 s on fast windows.
    Every read is at an index <= ``n_past - 1`` (the slice is taken first), so a future read is impossible."""
    if poses.dim() != 3 or poses.shape[-1] < 4:
        raise ResidualPriorError(f"poses must be [B, T, >=4] (x, y, yaw, v), got {tuple(poses.shape)}")
    n = int(poses.shape[1] if n_past is None else n_past)
    if n < 2:
        raise ResidualPriorError(f"n_past {n} < 2: the pose prior's acceleration and yaw rate are backward differences "
                                 f"at t0 and need t0-1 inside the window.")
    ch = ego_channels_from_poses(poses[:, :n].to(torch.float64), n, dt=float(dt))       # [B, n, 3]
    v0, a0, om = ch[:, -1, 0], ch[:, -1, 1], ch[:, -1, 2]
    return torch.stack([v0, a0, _pose_kappa0(om, v0)], dim=-1)


def prior_controls(mode: str, poses: Tensor, n_past: int | None = None,
                   actions: Tensor | None = None, *,
                   dt: float = DT_TICK,
                   wheelbase: float = STEER_WHEELBASE_M,
                   p10=None
                   ) -> tuple[Tensor, Tensor]:
    """``(a0 [B], kappa0 [B])`` -- the prior's controls in GEOMETRY units
    (m/s^2, 1/m), from the OBSERVED window only. Mode ``p10`` returns the PER-TICK
    pair ``(a [B, 60], kappa [B, 60])`` (``p10`` = the frozen
    :class:`tanitad.models.p10_prior.FrozenP10`, REQUIRED for that mode).

    ``poses`` ``[B, T, >=4]`` = ``(x, y, yaw, v)``; ``n_past`` is how many
    leading steps are PAST (``None`` = all of them), exactly the contract of
    :func:`tanitad.models.ego_history.ego_channels_from_poses`, which is called
    here so the prior and the ego-history encoder read ONE derivation.
    ``actions`` ``[B, T, 2]`` = ``(steer, accel)`` is required by ``ha0_ext``
    and REFUSED by every other mode (a supplied input that is silently ignored
    is the false-provenance class).

    (( !! )) Every read is at index ``n_past - 1`` (t0) or ``n_past - 2``:
    ``ego_channels_from_poses`` slices ``poses[:, :n_past]`` before it computes
    anything, and ``actions`` is sliced the same way here.
    """
    check_mode(mode)
    if mode == RESIDUAL_PRIOR_OFF:
        raise ResidualPriorError(
            "prior_controls called with residual_prior='off' -- there is no "
            "prior to build; the caller must not reach this function.")
    if p10 is not None and mode not in P10_MODES:
        raise ResidualPriorError(
            f"a frozen P10 module was supplied to residual_prior={mode!r}, which does not read it -- it would "
            f"be silently dropped and the run record could not say which prior ran.")
    if poses.dim() != 3 or poses.shape[-1] < 4:
        raise ResidualPriorError(
            f"poses must be [B, T, >=4] (x, y, yaw, v), got "
            f"{tuple(poses.shape)}")
    n = int(poses.shape[1] if n_past is None else n_past)
    if n < 2:
        raise ResidualPriorError(
            f"n_past {n} < 2: the prior's acceleration and yaw rate are "
            f"backward differences at t0 and need t0-1 inside the window.")
    ch = ego_channels_from_poses(poses, n, dt=float(dt))       # [B, n, 3]
    v0 = ch[:, -1, 0]
    a0 = ch[:, -1, 1]              # (v[t0] - v[t0-1]) / dt  (ego_history.py:96)
    if mode == P10_MODE:
        if actions is None:
            raise ResidualPriorError(
                "residual_prior='p10' reads the RECORDED STEER at t0 (P10's kinematic feature k = "
                "actions[t0, 0], the channel `ha0_ext` reads) and no `actions` reached the prior. Refusing "
                "rather than substituting a pose curvature: that is a DIFFERENT prior and would be stamped "
                "as P10.")
        if p10 is None:
            raise ResidualPriorError(
                "residual_prior='p10' but no frozen P10 module reached prior_controls -- a P10 mode with no "
                "P10 would silently be ANOTHER prior.")
        if getattr(p10, "variant", "steer") != "steer":
            raise ResidualPriorError(
                f"residual_prior='p10' reads the recorded steer and needs the STEER-variant P10; the module supplied is "
                f"the {getattr(p10, 'variant', None)!r} one (fit on another input channel = ANOTHER prior).")
        return p10_tick_prior(p10, poses, n, actions, dt=float(dt))[:2]
    if mode == P10_POSE_MODE:
        if actions is not None:
            raise ResidualPriorError(
                f"`actions` were supplied to residual_prior={mode!r}, which reads ONLY the pose track -- they would be "
                f"silently dropped and the run record could not say which curvature the prior used.")
        if p10 is None:
            raise ResidualPriorError(
                "residual_prior='p10_pose' but no frozen P10 module reached prior_controls -- a P10 mode with no "
                "P10 would silently be ANOTHER prior.")
        if getattr(p10, "variant", None) != "pose":
            raise ResidualPriorError(
                f"residual_prior='p10_pose' reads ONLY the pose track and needs the POSE-variant P10; the module supplied "
                f"is the {getattr(p10, 'variant', None)!r} one (fit on the recorded steer: it would be fed a curvature "
                f"where it expects a steer angle).")
        return p10_tick_prior(p10, poses, n, None, dt=float(dt))[:2]
    if mode == "cv_yawrate":
        a0 = torch.zeros_like(a0)
    if mode == "ha0_ext":
        if actions is None:
            raise ResidualPriorError(
                "residual_prior='ha0_ext' reads the RECORDED STEER at t0 "
                "(refcv3_arm.py:2082, refav1_arm.py:446) and no `actions` "
                "reached the prior. Refusing rather than substituting the "
                "pose curvature: that is a DIFFERENT prior "
                "('ha0_ext_pose') and would be stamped as the echo.")
        if actions.dim() != 3 or actions.shape[-1] != 2:
            raise ResidualPriorError(
                f"actions must be [B, T, 2] = (steer, accel), got "
                f"{tuple(actions.shape)}")
        if actions.shape[0] != poses.shape[0] or actions.shape[1] < n:
            raise ResidualPriorError(
                f"actions {tuple(actions.shape)} do not cover the pose "
                f"window's {n} past steps for {poses.shape[0]} rows")
        steer0 = actions[:, n - 1, 0]
        kappa0 = kappa_of_steer(steer0, wheelbase)
        return a0, kappa0
    if actions is not None:
        raise ResidualPriorError(
            f"`actions` were supplied to residual_prior={mode!r}, which reads "
            f"ONLY the pose track -- they would be silently dropped and the "
            f"run record could not say which curvature the prior used.")
    omega0 = ch[:, -1, 2]          # wrap(yaw[t0] - yaw[t0-1]) / dt (:98)
    kappa0 = _pose_kappa0(omega0, v0)
    return a0, kappa0


def withhold(a0: Tensor, kappa0: Tensor,
             keep: Tensor | None) -> tuple[Tensor, Tensor]:
    """Zero the prior on rows whose measured ego state was WITHHELD.

    (( !! )) ``keep`` is the decoder's ego-dropout mask (``refc.py:4195-4200``).
    A withheld row's anchor bank is rolled at the reference speed so the
    dropout regime stays speed-blind (``refc.py:2094-2155``); a prior built from
    the same row's measured dynamics would hand the withheld state straight
    back through the candidate GEOMETRY. On a withheld row the prior is
    therefore the no-information prior -- zero controls -- and the residual
    vocabulary there is exactly the absolute refcv6 vocabulary at the
    reference speed. At eval nothing is withheld and this is the identity.
    """
    if keep is None:
        return a0, kappa0
    k = keep.reshape(-1).to(torch.bool)
    if a0.dim() == 2:                       # per-tick prior [B, T]: one decision per ROW
        k = k[:, None]
    z = torch.zeros_like(a0)
    return torch.where(k, a0, z), torch.where(k, kappa0, torch.zeros_like(kappa0))


# --------------------------------------------------------------------------- #
# the PER-TICK prior (mode p10)                                                #
# --------------------------------------------------------------------------- #
def is_tick_prior(a0: Tensor) -> bool:
    """True for a per-tick prior ``[B, T]`` (mode ``p10``), False for the constant ``[B]`` of every other mode."""
    return a0.dim() == 2


def _need_ticks(a0: Tensor, kappa0: Tensor, t_max: int, b: int) -> None:
    if kappa0.shape != a0.shape:
        raise ResidualPriorError(f"per-tick prior a0 {tuple(a0.shape)} and kappa0 {tuple(kappa0.shape)} differ")
    if a0.shape[0] != b:
        raise ResidualPriorError(f"per-tick prior has {a0.shape[0]} rows, expected {b}")
    if a0.shape[1] < t_max:
        raise ResidualPriorError(
            f"per-tick prior covers {a0.shape[1]} ticks but the plan needs {t_max}: P10 reaches 6 s = "
            f"{P10_TICKS} ticks and a decoder horizon beyond it would silently read a prior that is not there")


def slot_prior(x: Tensor, horizons: Sequence[int]) -> Tensor:
    """``[B, T]`` per-tick values -> ``[B, S]``: the MEAN over each slot's span ``(h[s-1], h[s]]`` ticks.

    This is what ``u0_hat`` (per slot) is measured against when the prior is per-tick; the plan is always rolled
    with the per-tick prior itself. The mean is an exact ``sum / n`` (float32) per span."""
    h = [int(v) for v in horizons]
    if any(b <= a for a, b in zip(h, h[1:])) or h[0] <= 0:
        raise ResidualPriorError(f"horizons must be positive and increasing, got {h}")
    if x.shape[1] < h[-1]:
        raise ResidualPriorError(f"per-tick prior covers {x.shape[1]} ticks, the last slot ends at {h[-1]}")
    edges = [0] + h
    x32 = x.to(torch.float32)
    return torch.stack([x32[:, a:b].mean(dim=1) for a, b in zip(edges[:-1], edges[1:])], dim=1)


def _slot_of(x: Tensor, horizons: Sequence[int] | None) -> Tensor:
    if horizons is None:
        raise ResidualPriorError(
            "a per-tick prior was handed to absolute_controls / residual_controls without `horizons`: u0_hat is "
            "per SLOT and the prior's slot mean cannot be formed without the slot spans. Refusing rather than "
            "guessing them.")
    return slot_prior(x, horizons)


def knots_to_ticks(p30: Tensor) -> Tensor:
    """``[B, 30, 2]`` P10 knots (0.2 s, origin implicit) -> ``[B, 60, 2]`` positions at 0.1 s ticks 1..60.

    Even ticks ARE the knots; odd ticks are the 4-point interpolatory subdivision midpoints
    ``(-p[m-2] + 9 p[m-1] + 9 p[m] - p[m+1]) / 16`` (C1; exact for a cubic), with the origin as ``p[0]`` and the
    ends extended by quadratic extrapolation (so the scheme is exact for a quadratic everywhere). Chosen over the linear midpoints of
    ``v8f_mini.upsample_prior`` because a linear path has a heading kink at every knot -- a curvature spike every
    other tick that the ``kappa`` cap would clip in a sharp turn (MEASURED: DESIGN.md sec. 3)."""
    if p30.dim() != 3 or p30.shape[-1] != 2:
        raise ResidualPriorError(f"knots must be [B, K, 2], got {tuple(p30.shape)}")
    p = p30.to(torch.float64)
    b, k, _ = p.shape
    pts = torch.cat([p.new_zeros(b, 1, 2), p], dim=1)                       # p[0..K], p[0] = origin
    left = 3 * pts[:, 0:1] - 3 * pts[:, 1:2] + pts[:, 2:3]                  # p[-1]: quadratic extrapolation
    right = 3 * pts[:, -1:] - 3 * pts[:, -2:-1] + pts[:, -3:-2]             # p[K+1]
    ext = torch.cat([left, pts, right], dim=1)                              # p[-1..K+1]
    # midpoint m (between p[m-1] and p[m]), m = 1..K: indices into ext are shifted by +1
    mid = (-ext[:, 0:k] + 9 * ext[:, 1:k + 1] + 9 * ext[:, 2:k + 2] - ext[:, 3:k + 3]) / 16.0
    out = torch.stack([mid, p], dim=2).reshape(b, 2 * k, 2)               # tick 2m-1 = mid, tick 2m = knot
    return out.to(p30.dtype)


def controls_from_path(path: Tensor, v0: Tensor, *, dt: float = DT_TICK, v_floor: float = POSE_KAPPA_V_FLOOR,
                       kappa_cap: float = POSE_KAPPA_CAP, min_speed: float | None = None) -> Tensor:
    """The CLOSED-LOOP inverse of :func:`kinematic.rollout_unicycle`: ``path`` ``[B, K, 2]`` (ego frame, ticks
    1..K, origin implicit) and the measured ``v0`` ``[B]`` -> per-tick controls ``[B, K, 2]`` = (accel, kappa)
    such that ``rollout_unicycle`` from ``(0, 0, 0, v0)`` reproduces ``path[1:]`` exactly wherever neither bound
    below binds.

    Why not :func:`kinematic.unicycle_controls_from_path`: that one is OPEN loop and takes no entry speed -- rolled
    from the measured ``v0`` it leaves a CONSTANT speed offset ``v0 - |p1| / dt`` for the whole horizon (its own
    docstring says so: ``entry_speed_mismatch``). This one tracks the ROLLED state, so a mismatch costs one tick.

    The update order is ``rollout_unicycle``'s (position and heading on the speed at the START of the tick, ``v``
    updated LAST, clamped at 0): control ``i`` fixes the state that moves tick ``i + 2``, so it is solved from
    segment ``i + 1`` (0-based), and the LAST control -- which no waypoint depends on -- repeats the previous one.
    The first tick's displacement is ``(v0 * dt, 0)`` by construction and is NOT controllable.

    Conventions inherited, not invented: a tick slower than ``kinematic.MIN_DS_MPS`` has no tangent, so the heading
    is HELD; the curvature divides by ``max(v, v_floor)`` (``POSE_KAPPA_V_FLOOR``: below it the heading change is
    attenuated instead of dividing noise by a small speed) and is clamped to ``+-kappa_cap``
    (``POSE_KAPPA_CAP``). Both bounds leave a heading error that the NEXT tick re-targets (that is the closed
    loop); ``ADE(P10u, P10)`` measures what is left."""
    from tanitad.models.kinematic import MIN_DS_MPS
    ms = float(MIN_DS_MPS if min_speed is None else min_speed)
    if path.dim() != 3 or path.shape[-1] != 2 or path.shape[1] < 2:
        raise ResidualPriorError(f"path must be [B, K>=2, 2], got {tuple(path.shape)}")
    p = torch.cat([path.new_zeros(path.shape[0], 1, 2), path], dim=1).to(torch.float64)
    seg = p[:, 1:] - p[:, :-1]                                              # tick j+1's displacement
    ds = (seg.pow(2).sum(-1) + 1e-18).sqrt()
    v_star = ds / float(dt)                                                 # speed during tick j+1
    psi_star = torch.atan2(seg[..., 1], seg[..., 0])
    k_ticks = path.shape[1]
    v = v0.reshape(-1).to(torch.float64)
    psi = torch.zeros_like(v)
    ctrl = torch.zeros(path.shape[0], k_ticks, 2, dtype=torch.float64, device=path.device)
    for i in range(k_ticks - 1):
        tgt_v = v_star[:, i + 1]
        a_i = (tgt_v - v) / float(dt)
        moving = tgt_v > ms
        tgt_psi = torch.where(moving, psi_star[:, i + 1], psi)             # no tangent -> hold the heading
        dpsi = (tgt_psi - psi + math.pi) % (2 * math.pi) - math.pi
        kap = (dpsi / (v.clamp_min(float(v_floor)) * float(dt))).clamp(-float(kappa_cap), float(kappa_cap))
        ctrl[:, i, 0], ctrl[:, i, 1] = a_i, kap
        psi = psi + v * kap * float(dt)                                     # rollout_unicycle's order
        v = (v + a_i * float(dt)).clamp_min(0.0)
    ctrl[:, -1] = ctrl[:, -2]                                               # the last control is unobservable
    return ctrl.to(torch.float32)


def p10_tick_prior(p10, poses: Tensor, n_past: int, actions: "Tensor | None", *, dt: float = DT_TICK
                   ) -> tuple[Tensor, Tensor, Tensor]:
    """``(a [B, 60], kappa [B, 60], p30 [B, 30, 2])``: the frozen P10 read on the OBSERVED window -> its knots ->
    per-tick controls rolled from the window's measured ``v0``. Pure function of the window (no RNG, no state),
    detached: nothing in here can carry a gradient. The P10's ``variant`` picks the feature source: ``"steer"`` reads the
    recorded steer at t0 (``actions`` REQUIRED), ``"pose"`` reads the pose track only (``actions`` REFUSED)."""
    from tanitad.models.p10_prior import P10_K, kin3_from_window
    variant = getattr(p10, "variant", "steer")
    with torch.no_grad():
        if variant == "pose":
            if actions is not None:
                raise ResidualPriorError("a POSE-variant P10 reads ONLY the pose track; `actions` were supplied")
            kin3 = pose_kin3(poses, n_past, dt=float(dt))
        elif variant == "steer":
            if actions is None:
                raise ResidualPriorError("a STEER-variant P10 reads the recorded steer at t0 and no `actions` were supplied")
            kin3 = kin3_from_window(poses, actions, n_past, dt_frame=float(dt))
        else:
            raise ResidualPriorError(f"P10 variant {variant!r} is not one of ('steer', 'pose')")
        p30 = p10(kin3)                                                     # [B, 30, 2] float32
        if p30.shape[1] != P10_K:
            raise ResidualPriorError(f"P10 returned {p30.shape[1]} knots, expected {P10_K}")
        ticks = knots_to_ticks(p30.to(torch.float64)).to(torch.float32)
        ctrl = controls_from_path(ticks, kin3[:, 0].to(torch.float32), dt=float(dt),
                                  v_floor=POSE_KAPPA_V_FLOOR, kappa_cap=P10_KAPPA_CAP)
    return ctrl[..., 0], ctrl[..., 1], p30


def _slot_index(horizons: Sequence[int], device) -> Tensor:
    return torch.tensor([int(h) - 1 for h in horizons], device=device,
                        dtype=torch.long)


def prior_path(a0: Tensor, kappa0: Tensor, v0: Tensor,
               horizons: Sequence[int], *, tick: float = DT_TICK) -> Tensor:
    """``P`` ``[B, S, 2]``: the constant ``(a0, kappa0)`` rolled from ``v0``.

    The SAME calls the battery makes (``refa_v1_plan.py:369-373``): a zero
    state with ``v0`` in slot 3, ``rollout_unicycle`` at ``tick``, then an index
    select at ``horizons - 1``. Float32 throughout, like ``roll_bank``.
    """
    h = [int(x) for x in horizons]
    b = int(v0.reshape(-1).shape[0])
    t_max = max(h)
    if a0.dim() == 2:                       # per-tick prior [B, T]
        _need_ticks(a0, kappa0, t_max, b)
        ctrl = torch.stack([a0[:, :t_max], kappa0[:, :t_max]], dim=-1).to(torch.float32)
    else:
        a = a0.reshape(-1).to(torch.float32)
        k = kappa0.reshape(-1).to(torch.float32)
        ctrl = torch.stack([a, k], dim=-1)[:, None, :].expand(b, t_max, 2)
    state0 = torch.zeros(b, 4, device=ctrl.device, dtype=torch.float32)
    state0[:, 3] = v0.reshape(-1).to(torch.float32)
    path = rollout_unicycle(state0, ctrl, dt=float(tick))[..., :2]
    return path.index_select(1, _slot_index(h, path.device))


def _lat_to_curvature(lat: Tensor, v: Tensor, control_units: str,
                      alat_v_floor: float, kappa_cap: float) -> Tensor:
    """The residual's lateral channel -> curvature, through the vocabulary's
    OWN conversion (``refc_sampler.alat_to_curvature``, the one spelling)."""
    from tanitad.refs.refc_sampler import alat_to_curvature
    if control_units == "alat":
        return alat_to_curvature(lat, v, alat_v_floor, kappa_cap)
    if control_units == "kappa":
        return lat
    raise ResidualPriorError(f"control_units {control_units!r} not in "
                             f"('kappa', 'alat')")


def compose_ticks(delta: Tensor, a0: Tensor, kappa0: Tensor, v: Tensor,
                  horizons: Sequence[int], *, control_units: str,
                  alat_v_floor: float = 4.0,
                  kappa_cap: float = 0.12) -> Tensor:
    """``Delta`` ``[B, N, S, 2]`` (per-slot, vocabulary units) + the prior ->
    tick controls ``[B, N, max(horizons), 2]`` = ``(a, kappa)`` in GEOMETRY.

    Each slot's residual is held over its span by ``controls_to_ticks`` -- the
    step that makes a constant control reproduce ``roll_bank``'s bank
    (``refc_sampler.py:327-346``) -- and the prior is added AFTER the
    residual's curvature conversion (module docstring).
    """
    from tanitad.refs.refc_sampler import controls_to_ticks
    if delta.dim() != 4 or delta.shape[-1] != 2:
        raise ResidualPriorError(f"delta must be [B, N, S, 2], got "
                                 f"{tuple(delta.shape)}")
    b = delta.shape[0]
    d32 = delta.to(torch.float32)
    vv = v.reshape(-1).to(torch.float32)
    if vv.shape[0] != b:
        raise ResidualPriorError(f"v has {vv.shape[0]} rows, expected {b}")
    if a0.dim() == 2:
        # PER-TICK prior [B, T] (mode p10): the residual (a, kappa) is converted in slot units and HELD over its
        # span exactly as before, then the prior is added tick by tick. For a constant prior this is the same
        # float32 sum the branch below computes, pinned bit for bit by test_residual_prior.py.
        dk = _lat_to_curvature(d32[..., 1], vv[:, None, None], control_units,
                               alat_v_floor, kappa_cap)
        held = controls_to_ticks(torch.stack([d32[..., 0], dk], dim=-1),
                                 tuple(int(x) for x in horizons))          # [B, N, T, 2]
        t_max = held.shape[-2]
        _need_ticks(a0, kappa0, t_max, b)
        pr = torch.stack([a0[:, :t_max], kappa0[:, :t_max]], dim=-1).to(torch.float32)   # [B, T, 2]
        return held + pr[:, None]
    a = a0.reshape(b, 1, 1).to(torch.float32) + d32[..., 0]
    dk = _lat_to_curvature(d32[..., 1], vv[:, None, None], control_units,
                           alat_v_floor, kappa_cap)
    k = kappa0.reshape(b, 1, 1).to(torch.float32) + dk
    return controls_to_ticks(torch.stack([a, k], dim=-1),
                             tuple(int(x) for x in horizons))


def roll_plan(delta: Tensor, a0: Tensor, kappa0: Tensor, v: Tensor,
              horizons: Sequence[int], *, control_units: str,
              tick: float = DT_TICK, alat_v_floor: float = 4.0,
              kappa_cap: float = 0.12) -> Tensor:
    """(( THE ONE FUNCTION )) ``P + Delta`` -> the plan ``[B, N, S, 2]`` in metres.

    ``delta`` ``[B, N, S, 2]`` is the residual control state (vocabulary units),
    ``a0``/``kappa0`` ``[B]`` the prior's controls (GEOMETRY), ``v`` ``[B]`` the
    speed the row is rolled from -- which MUST be the speed the prior was
    rolled from; the decoder passes the same tensor to both.

    ``delta == 0`` returns :func:`prior_path` for every candidate, bit for bit
    on CPU (``a0 + 0.0 == a0``; ``alat_to_curvature(0) == 0``). Float32
    integration regardless of the input dtype (``roll_bank``'s rule), output in
    the input dtype.
    """
    h = tuple(int(x) for x in horizons)
    b, n, s = delta.shape[0], delta.shape[1], delta.shape[2]
    if s != len(h):
        raise ResidualPriorError(f"delta has {s} slots, horizons has {len(h)}")
    ticks = compose_ticks(delta, a0, kappa0, v, h,
                          control_units=control_units,
                          alat_v_floor=alat_v_floor, kappa_cap=kappa_cap)
    t_max = ticks.shape[-2]
    ticks = ticks.reshape(b * n, t_max, 2)
    state0 = torch.zeros(b * n, 4, device=ticks.device, dtype=torch.float32)
    state0[:, 3] = v.reshape(-1).to(torch.float32)[:, None].expand(
        b, n).reshape(-1)
    path = rollout_unicycle(state0, ticks, dt=float(tick))[..., :2]
    return path.index_select(1, _slot_index(h, path.device)).reshape(
        b, n, s, 2).to(delta.dtype)


def _alat_scale(v: Tensor, alat_v_floor: float) -> Tensor:
    """``max(v, floor)^2`` -- the SAME divisor ``alat_to_curvature`` uses."""
    return v.clamp_min(float(alat_v_floor)) ** 2


def _slot_view(x: Tensor, ref: Tensor) -> Tensor:
    """A slot-level prior ``[B, S]`` shaped to broadcast against ``ref [B, ..., S, 2]``'s control columns."""
    return x.reshape(x.shape[0], *([1] * (ref.dim() - 3)), x.shape[1])


def absolute_controls(delta: Tensor, a0: Tensor, kappa0: Tensor, v: Tensor,
                      *, control_units: str,
                      alat_v_floor: float = 4.0,
                      horizons: Sequence[int] | None = None) -> Tensor:
    """The residual state -> ABSOLUTE controls in the vocabulary's units.

    This is what the decoder EXPORTS as ``u0_hat`` / ``layer_u0_hat``, so the
    residual never leaves it:
        a_lon_abs = a0 + delta_a
        lat_abs   = kappa0 * max(v, floor)^2 + delta_alat      ("alat")
                  = kappa0 + delta_kappa                        ("kappa")
    With these, the trainer's EXISTING x0 target
    (``refc_v3_train.py:4021-4037``: the GT path's controls, curvature scaled by
    ``max(v0, floor)^2``) needs no change, and its L1 equals the residual-space
    L1 ``|delta - (u_gt - prior)|`` exactly in real arithmetic.
    (( !! )) Rolling these ABSOLUTE controls through the legacy
    ``refc_sampler.roll_controls`` reproduces the plan only where no clamp
    binds; re-roll through :func:`roll_plan` via :func:`residual_controls`.
    """
    b = delta.shape[0]
    shp = (b, *([1] * (delta.dim() - 2)))
    d32 = delta.to(torch.float32)          # float32 like every roll here
    if a0.dim() == 2:                      # per-tick prior: its SLOT MEAN is what u0_hat is relative to
        a = _slot_view(_slot_of(a0, horizons), delta).to(torch.float32) + d32[..., 0]
        k0 = _slot_view(_slot_of(kappa0, horizons), delta).to(torch.float32)
    else:
        a = a0.reshape(shp).to(torch.float32) + d32[..., 0]
        k0 = kappa0.reshape(shp).to(torch.float32)
    if control_units == "alat":
        sc = _alat_scale(v.reshape(-1).to(torch.float32),
                         alat_v_floor).reshape(shp)
        lat = k0 * sc + d32[..., 1]
    elif control_units == "kappa":
        lat = k0 + d32[..., 1]
    else:
        raise ResidualPriorError(f"control_units {control_units!r} not in "
                                 f"('kappa', 'alat')")
    return torch.stack([a, lat], dim=-1).to(delta.dtype)


def residual_controls(u_abs: Tensor, a0: Tensor, kappa0: Tensor, v: Tensor,
                      *, control_units: str,
                      alat_v_floor: float = 4.0,
                      horizons: Sequence[int] | None = None) -> Tensor:
    """The inverse of :func:`absolute_controls` (exact in real arithmetic;
    float32 rounding otherwise). A per-tick prior needs ``horizons`` (slot means)."""
    b = u_abs.shape[0]
    shp = (b, *([1] * (u_abs.dim() - 2)))
    u32 = u_abs.to(torch.float32)
    if a0.dim() == 2:
        a = u32[..., 0] - _slot_view(_slot_of(a0, horizons), u_abs).to(torch.float32)
        k0 = _slot_view(_slot_of(kappa0, horizons), u_abs).to(torch.float32)
    else:
        a = u32[..., 0] - a0.reshape(shp).to(torch.float32)
        k0 = kappa0.reshape(shp).to(torch.float32)
    if control_units == "alat":
        sc = _alat_scale(v.reshape(-1).to(torch.float32),
                         alat_v_floor).reshape(shp)
        lat = u32[..., 1] - k0 * sc
    elif control_units == "kappa":
        lat = u32[..., 1] - k0
    else:
        raise ResidualPriorError(f"control_units {control_units!r} not in "
                                 f"('kappa', 'alat')")
    return torch.stack([a, lat], dim=-1).to(u_abs.dtype)


def prior_from_out(out: dict) -> "tuple[Tensor, Tensor, Tensor] | None":
    """``(a0, kappa0, v)`` exactly as the forward composed it, or ``None`` on
    an ``off`` output. THE way a consumer re-rolls exported controls
    (``decoder._state_to_path(..., prior=prior_from_out(out))``). A PARTIAL
    set of keys is refused: it means a whitelist dropped one of them."""
    have = [k for k in OUT_KEYS if k in out]
    if not have:
        return None
    if len(have) != len(OUT_KEYS):
        raise ResidualPriorError(
            f"the forward emitted {have} but not all of {OUT_KEYS} -- a "
            f"pass-through dropped a residual-prior key (the A16 class)")
    ctrl = out["residual_prior_ctrl"]
    return ctrl[..., 0], ctrl[..., 1], out["residual_prior_v"]


def zero_residual_index(anchor_controls: Tensor) -> int:
    """The vocabulary index whose residual is exactly ``(0, 0)`` -- the anchor
    that rolls to ``P`` itself. Refuses a vocabulary without one (the refcv6
    grid carries it at index 67, ``straight_ahead_control_index``)."""
    z = (anchor_controls[:, :2] == 0).all(dim=-1).nonzero().flatten()
    if z.numel() != 1:
        raise ResidualPriorError(
            f"the vocabulary has {int(z.numel())} exact (0, 0) controls; the "
            f"residual family needs exactly one -- the prior itself")
    return int(z[0])


def plan_check(out: dict, anchor_controls: Tensor, *, decoder=None,
               atol: float = 1e-5, fan_atol: float = 1e-3) -> dict:
    """The consumer-side proof that what a forward emitted is ``P + Delta``.

    Reads the forward's OUTPUT (and, for (2), the decoder's declared units):
      (1) the ``(0, 0)`` residual anchor of ``anchor_bank`` equals
          ``residual_prior_path`` on every row -- the BANK is composed on P;
      (2) with ``decoder`` given and ``u0_hat`` emitted: the emitted fan equals
          ``roll_plan(residual_controls(u0_hat), P)`` -- the FAN is composed on
          the same P, and the exported controls are the plan's controls;
      (3) the prior path is non-zero on every row whose roll speed is > 0 (the
          launch gate's G-LIVE clause, SPEC_REFCV7 section 2);
      (4) every emitted path is finite.
    Returns the measured numbers; ``ok`` is their conjunction. A mutation that
    hands a consumer ``Delta`` instead of ``P + Delta`` fails (1) or (2).
    """
    rp = prior_from_out(out)
    if rp is None:
        raise ResidualPriorError("plan_check on an output with no residual "
                                 "prior (an off build has nothing to check)")
    i0 = zero_residual_index(anchor_controls)
    bank = out["anchor_bank"].float()
    n_anchor = anchor_controls.shape[0]
    if bank.shape[1] % n_anchor:
        raise ResidualPriorError(f"bank width {bank.shape[1]} is not a "
                                 f"multiple of {n_anchor} anchors")
    p = out["residual_prior_path"].float()
    d0 = float((bank[:, i0] - p).abs().max())
    d_fan = None
    if decoder is not None and "u0_hat" in out:
        kw = dict(control_units=decoder.anchor_control_units,
                  alat_v_floor=decoder.anchor_alat_v_floor)
        delta = residual_controls(out["u0_hat"].float(), rp[0], rp[1], rp[2],
                                  horizons=decoder.anchor_horizons, **kw)
        fan = roll_plan(delta, rp[0], rp[1], rp[2], decoder.anchor_horizons,
                        tick=decoder.anchor_dt,
                        kappa_cap=decoder.anchor_kappa_cap, **kw)
        d_fan = float((fan - out["anchor_traj"].float()).abs().max())
    v = rp[2].float()
    moving = v > 0
    mag = p.norm(dim=-1).amax(dim=-1)
    n_mov = int(moving.sum())
    n_nonzero = int((mag[moving] > 0).sum()) if n_mov else 0
    finite = bool(torch.isfinite(out["anchor_traj"].float()).all()
                  and torch.isfinite(out["traj"].float()).all())
    # informative, not gated: on a withheld row the controls are zero BY DESIGN
    # (`withhold`), so this counts the rows that carry a measured prior at all
    ctrl_nz = int(((rp[0] != 0) | (rp[1] != 0)).reshape(rp[0].shape[0], -1).any(dim=1).sum())
    rep = {"rows": int(p.shape[0]), "rows_v_gt_0": n_mov,
           "rows_v_gt_0_prior_nonzero": n_nonzero,
           "rows_prior_controls_nonzero": ctrl_nz,
           "zero_residual_anchor": i0,
           "max_abs_zero_anchor_minus_prior_m": d0,
           "max_abs_fan_minus_reroll_m": d_fan,
           "finite": finite}
    rep["ok"] = bool(d0 <= atol and n_nonzero == n_mov and finite
                     and (d_fan is None or d_fan <= fan_atol))
    return rep


def prior_stamp(mode: str, p10: dict | None = None) -> dict:
    """What ``config.json`` and the declared-vs-built check record. ``p10`` (mode ``p10`` only) = the frozen
    artifact's identity: ``{"path", "file_sha256", "content_sha256", "meta"}`` -- REQUIRED for that mode and
    REFUSED for every other (a stamp that names a P10 the mode does not read is false provenance)."""
    check_mode(mode)
    if (mode in P10_MODES) != (p10 is not None):
        raise ResidualPriorError(
            f"prior_stamp(mode={mode!r}, p10={'given' if p10 is not None else None}): the P10 identity is "
            f"recorded if and only if the mode is one of {P10_MODES}")
    if p10 is not None and p10.get("variant") is not None:
        want_variant = "pose" if mode == P10_POSE_MODE else "steer"
        if p10["variant"] != want_variant:
            raise ResidualPriorError(
                f"prior_stamp(mode={mode!r}): the stamped P10 is the {p10['variant']!r} variant, this mode reads the "
                f"{want_variant!r} one -- a stamp that names another prior is false provenance")
    defs = {
        RESIDUAL_PRIOR_OFF: "no prior (refcv6 decoder, bit-identical)",
        "ha0_ext": ("constant a0 = (v[t0]-v[t0-1])/dt and constant kappa0 = "
                    "tan(steer[t0])/2.9 -- the battery's echo, exactly"),
        "ha0_ext_pose": ("constant a0 = (v[t0]-v[t0-1])/dt and constant "
                         "kappa0 = clamp(omega0/max(v0, %.1f), +-%.2f), omega0 "
                         "= wrap(yaw[t0]-yaw[t0-1])/dt"
                         % (POSE_KAPPA_V_FLOOR, POSE_KAPPA_CAP)),
        "cv_yawrate": ("a = 0 and constant kappa0 = clamp(omega0/max(v0, "
                       "%.1f), +-%.2f) -- constant speed v0, constant yaw "
                       "rate" % (POSE_KAPPA_V_FLOOR, POSE_KAPPA_CAP)),
        P10_MODE: ("the FROZEN learned kinematic floor P10 (30 knots x 0.2 s, kd_x + MLP on (v0, a0 stride-2, "
                   "steer[t0])) converted to PER-TICK controls by the closed-loop inverse of "
                   "rollout_unicycle (knots -> 0.1 s by 4-point interpolatory subdivision; kappa floor %.1f m/s, "
                   "cap %.2f 1/m) -- Delta == 0 rolls to P10u, P10's path re-rolled from the measured v0"
                   % (POSE_KAPPA_V_FLOOR, P10_KAPPA_CAP)),
        P10_POSE_MODE: ("the FROZEN learned kinematic floor P10, POSE-ONLY variant (30 knots x 0.2 s, kd_x_pose + MLP on "
                        "(v0, a0 = ONE-frame backward difference, kappa0 = clamp(omega0/max(v0, %.1f), +-%.2f)) of the "
                        "past pose track -- ha0_ext_pose's own triple) converted to PER-TICK controls by the closed-loop "
                        "inverse of rollout_unicycle (knots -> 0.1 s by 4-point interpolatory subdivision; kappa cap "
                        "%.2f 1/m) -- Delta == 0 rolls to P10u, P10's path re-rolled from the measured v0"
                        % (POSE_KAPPA_V_FLOOR, POSE_KAPPA_CAP, P10_KAPPA_CAP)),
    }
    extra = {} if p10 is None else {"p10": dict(p10), "prior_shape": "per tick [B, 60]"}
    return {**extra, "residual_prior": mode, "definition": defs[mode],
            "reads": ({"off": [], "ha0_ext": ["pose window", "actions[t0, 0]"],
                       "ha0_ext_pose": ["pose window"],
                       "cv_yawrate": ["pose window"],
                       P10_MODE: ["pose window (v[t0], v[t0-2])", "actions[t0, 0]"],
                       P10_POSE_MODE: ["pose window (v[t0], v[t0-1], yaw[t0], yaw[t0-1])"]}[mode]),
            "integrator": "tanitad.models.kinematic.rollout_unicycle, dt 0.1, "
                          "float32, index-select at horizons - 1",
            "composition": "a = a0 + delta_a; kappa = kappa0 + "
                           "alat_to_curvature(delta_alat, v)",
            # the SAME anchor artifact means a different thing under each value --
            # recorded here because the artifact itself does not say so
            "vocabulary_space": ("absolute" if mode == RESIDUAL_PRIOR_OFF
                                 else "residual on P"),
            "withheld_rows": "prior zeroed where ego_keep is False",
            "echo_source": ECHO_SOURCE,
            "equals_battery_echo": mode == "ha0_ext"}
